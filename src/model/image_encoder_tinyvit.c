/*
 * src/model/image_encoder_tinyvit.c - TinyViT encoder implementation
 *
 * Implements initialization, weight loading, and graph construction
 * for the TinyViT backbone used in EfficientSAM3. The encoder processes
 * an input image through a conv patch embed, 4 layers of MBConv and/or
 * TinyViTBlock stages, and a projection head to produce dense features.
 *
 * Weight loading uses gh_load_mmap exclusively -- all weight data remains
 * in the mmap region with zero copies. Attention biases are expanded from
 * compact [n_heads, n_offsets] to full [n_heads, ws*ws, ws*ws] at load time
 * to avoid gather ops during inference.
 *
 * Key types:  sam3_tinyvit
 * Depends on: image_encoder_tinyvit.h, graph_helpers.h
 * Used by:    vl_combiner.c (via backbone dispatch)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "image_encoder_tinyvit.h"
#include "graph_helpers.h"
#include "util/log.h"
#include "util/profile.h"

#define TVIT_PREFIX "detector_model.vision_encoder.backbone."

/* ── Initialization ──────────────────────────────────────────────── */

enum sam3_error sam3_tinyvit_init(struct sam3_tinyvit *tvit,
				  const int *embed_dims,
				  const int *depths,
				  const int *num_heads,
				  const int *window_sizes,
				  int n_layers,
				  int img_size,
				  int embed_dim,
				  int mlp_ratio)
{
	memset(tvit, 0, sizeof(*tvit));

	if (n_layers > SAM3_TVIT_MAX_LAYERS) {
		sam3_log_error("tvit_init: n_layers=%d exceeds max=%d",
			       n_layers, SAM3_TVIT_MAX_LAYERS);
		return SAM3_EINVAL;
	}

	for (int i = 0; i < n_layers; i++) {
		if (depths[i] > SAM3_TVIT_MAX_DEPTH) {
			sam3_log_error("tvit_init: depths[%d]=%d exceeds "
				       "max=%d", i, depths[i],
				       SAM3_TVIT_MAX_DEPTH);
			return SAM3_EINVAL;
		}
	}

	tvit->n_layers = n_layers;
	tvit->img_size = img_size;
	tvit->embed_dim = embed_dim;
	tvit->mlp_ratio = mlp_ratio;

	for (int i = 0; i < n_layers; i++) {
		tvit->embed_dims[i] = embed_dims[i];
		tvit->depths[i] = depths[i];
		tvit->num_heads[i] = num_heads[i];
		tvit->window_sizes[i] = window_sizes[i];
	}

	/*
	 * Compute grid_size by walking the stride chain:
	 * 2 stride-2 convs in patch embed + 1 PatchMerging per layer
	 * (except the last layer which has no downsample).
	 */
	int h = img_size;
	/* Patch embed: two stride-2 3x3 convs with pad=1 */
	h = (h + 2 * 1 - 3) / 2 + 1;	/* first conv */
	h = (h + 2 * 1 - 3) / 2 + 1;	/* second conv */

	/* PatchMerging in layers 0..n_layers-2 */
	for (int i = 0; i < n_layers - 1; i++)
		h = (h + 2 * 1 - 3) / 2 + 1;

	tvit->grid_size = h;

	/* Configure layers */
	for (int i = 0; i < n_layers; i++) {
		struct sam3_tvit_layer *layer = &tvit->layers[i];
		layer->n_blocks = depths[i];
		layer->embed_dim = embed_dims[i];
		layer->n_heads = num_heads[i];
		layer->window_size = window_sizes[i];
		layer->has_downsample = (i < n_layers - 1) ? 1 : 0;

		for (int b = 0; b < depths[i]; b++)
			layer->blocks[b].is_conv_block = (i == 0) ? 1 : 0;
	}

	sam3_log_info("tvit_init: img=%d grid=%d dims=[%d,%d,%d,%d] "
		      "depths=[%d,%d,%d,%d] mlp_ratio=%d",
		      img_size, tvit->grid_size,
		      embed_dims[0], embed_dims[1],
		      embed_dims[2], embed_dims[3],
		      depths[0], depths[1], depths[2], depths[3],
		      mlp_ratio);

	return SAM3_OK;
}

/* ── Weight loading helpers ──────────────────────────────────────── */

/*
 * load_tvit_conv - Load Conv2d_BN weights (TinyViT naming: .c. and .bn.).
 *
 * @cw:       Target conv weights struct
 * @wf:       Weight file
 * @arena:    Arena for tensor struct allocation
 * @prefix:   Full name prefix (e.g. "...patch_embed.seq.0.")
 * @oc:       Output channels
 * @kh:       Kernel height
 * @kw:       Kernel width
 * @ic:       Input channels per group
 * @has_bn:   Load BatchNorm params
 */
static enum sam3_error load_tvit_conv(
	struct sam3_tvit_conv_weights *cw,
	const struct sam3_weight_file *wf,
	struct sam3_arena *arena,
	const char *prefix,
	int oc, int kh, int kw, int ic,
	int has_bn)
{
	char name[256];

	int w_dims[] = {oc, kh, kw, ic};
	snprintf(name, sizeof(name), "%sc.weight", prefix);
	cw->conv_w = gh_load_mmap(wf, name, arena,
				    SAM3_DTYPE_F32, 4, w_dims);
	if (!cw->conv_w)
		return SAM3_ENOMEM;

	if (has_bn) {
		int bn_dims[] = {oc};

		snprintf(name, sizeof(name), "%sbn.weight", prefix);
		cw->bn_w = gh_load_mmap(wf, name, arena,
					  SAM3_DTYPE_F32, 1, bn_dims);
		if (!cw->bn_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name), "%sbn.bias", prefix);
		cw->bn_b = gh_load_mmap(wf, name, arena,
					  SAM3_DTYPE_F32, 1, bn_dims);
		if (!cw->bn_b)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name), "%sbn.running_mean", prefix);
		cw->bn_mean = gh_load_mmap(wf, name, arena,
					     SAM3_DTYPE_F32, 1, bn_dims);
		if (!cw->bn_mean)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name), "%sbn.running_var", prefix);
		cw->bn_var = gh_load_mmap(wf, name, arena,
					    SAM3_DTYPE_F32, 1, bn_dims);
		if (!cw->bn_var)
			return SAM3_ENOMEM;
	}

	return SAM3_OK;
}

/*
 * load_mbconv - Load MBConv block weights (ConvLayer stage).
 */
static enum sam3_error load_mbconv(
	struct sam3_tvit_mbconv *mb,
	const struct sam3_weight_file *wf,
	struct sam3_arena *arena,
	const char *prefix,
	int in_ch, int out_ch, int expand_ratio)
{
	char sub[256];
	int mid_ch = in_ch * expand_ratio;
	enum sam3_error err;

	/* conv1: 1x1, in_ch -> mid_ch */
	snprintf(sub, sizeof(sub), "%sconv1.", prefix);
	err = load_tvit_conv(&mb->conv1, wf, arena, sub,
			       mid_ch, 1, 1, in_ch, 1);
	if (err != SAM3_OK)
		return err;

	/* conv2: 3x3 DW, mid_ch -> mid_ch */
	snprintf(sub, sizeof(sub), "%sconv2.", prefix);
	err = load_tvit_conv(&mb->conv2, wf, arena, sub,
			       mid_ch, 3, 3, 1, 1);
	if (err != SAM3_OK)
		return err;

	/* conv3: 1x1, mid_ch -> out_ch */
	snprintf(sub, sizeof(sub), "%sconv3.", prefix);
	err = load_tvit_conv(&mb->conv3, wf, arena, sub,
			       out_ch, 1, 1, mid_ch, 1);
	if (err != SAM3_OK)
		return err;

	return SAM3_OK;
}

/*
 * load_patch_merging - Load PatchMerging downsample weights.
 */
static enum sam3_error load_patch_merging(
	struct sam3_tvit_patch_merging *pm,
	const struct sam3_weight_file *wf,
	struct sam3_arena *arena,
	const char *prefix,
	int in_ch, int out_ch)
{
	char sub[256];
	enum sam3_error err;

	/* conv1: 1x1, in_ch -> out_ch */
	snprintf(sub, sizeof(sub), "%sconv1.", prefix);
	err = load_tvit_conv(&pm->conv1, wf, arena, sub,
			       out_ch, 1, 1, in_ch, 1);
	if (err != SAM3_OK)
		return err;

	/* conv2: 3x3 DW stride 2, out_ch -> out_ch */
	snprintf(sub, sizeof(sub), "%sconv2.", prefix);
	err = load_tvit_conv(&pm->conv2, wf, arena, sub,
			       out_ch, 3, 3, 1, 1);
	if (err != SAM3_OK)
		return err;

	/* conv3: 1x1, out_ch -> out_ch */
	snprintf(sub, sizeof(sub), "%sconv3.", prefix);
	err = load_tvit_conv(&pm->conv3, wf, arena, sub,
			       out_ch, 1, 1, out_ch, 1);
	if (err != SAM3_OK)
		return err;

	return SAM3_OK;
}

/*
 * expand_attention_bias - Expand compact bias to full per-position matrix.
 *
 * The compact bias has shape [n_heads, n_offsets] where n_offsets = ws*ws
 * (one per unique (abs_dy, abs_dx) pair). This function computes the
 * relative position index table and gathers to produce the full
 * [n_heads, ws*ws, ws*ws] bias tensor.
 *
 * Algorithm: for each pair (p1, p2) of positions in the ws*ws window,
 * compute offset = (abs(p1_y - p2_y), abs(p1_x - p2_x)), map to a
 * unique index, then gather the corresponding bias value.
 */
static struct sam3_tensor *expand_attention_bias(
	const struct sam3_tensor *compact,
	struct sam3_arena *arena,
	int n_heads, int ws)
{
	int N = ws * ws;

	/*
	 * Build offset-to-index mapping. Offsets are (dy, dx) where
	 * 0 <= dy < ws and 0 <= dx < ws. Use a flat lookup table
	 * indexed by dy * ws + dx.
	 */
	int offset_map[196];	/* max ws=14 -> 196 */
	int n_offsets = 0;

	memset(offset_map, -1, sizeof(offset_map));

	/* Enumerate all position pairs to discover offset ordering */
	for (int p1y = 0; p1y < ws; p1y++) {
		for (int p1x = 0; p1x < ws; p1x++) {
			for (int p2y = 0; p2y < ws; p2y++) {
				for (int p2x = 0; p2x < ws; p2x++) {
					int dy = abs(p1y - p2y);
					int dx = abs(p1x - p2x);
					int key = dy * ws + dx;
					if (offset_map[key] < 0)
						offset_map[key] = n_offsets++;
				}
			}
		}
	}

	/* Verify compact tensor has expected n_offsets */
	if (compact->dims[1] != n_offsets) {
		sam3_log_error("tvit_bias: expected %d offsets, got %d",
			       n_offsets, compact->dims[1]);
		return NULL;
	}

	/* Allocate expanded bias [n_heads, N, N] */
	int out_dims[] = {n_heads, N, N};
	struct sam3_tensor *expanded;
	expanded = gh_alloc_tensor(arena, SAM3_DTYPE_F32, 3, out_dims);
	if (!expanded)
		return NULL;

	/* Gather: for each head, for each (p1, p2), look up bias */
	const float *src = (const float *)compact->data;
	float *dst = (float *)expanded->data;

	for (int h = 0; h < n_heads; h++) {
		for (int p1 = 0; p1 < N; p1++) {
			int p1y = p1 / ws;
			int p1x = p1 % ws;
			for (int p2 = 0; p2 < N; p2++) {
				int p2y = p2 / ws;
				int p2x = p2 % ws;
				int dy = abs(p1y - p2y);
				int dx = abs(p1x - p2x);
				int idx = offset_map[dy * ws + dx];
				dst[h * N * N + p1 * N + p2] =
					src[h * n_offsets + idx];
			}
		}
	}

	return expanded;
}

/*
 * load_attention - Load attention weights for one TinyViTBlock.
 */
static enum sam3_error load_attention(
	struct sam3_tvit_attention *attn,
	const struct sam3_weight_file *wf,
	struct sam3_arena *arena,
	const char *prefix,
	int channels, int n_heads, int ws)
{
	char name[256];
	enum sam3_error err;
	int qkv_dim = channels * 3;
	(void)err;

	/* QKV linear: [3*C, C] weight + [3*C] bias */
	{
		int w_dims[] = {qkv_dim, channels};
		snprintf(name, sizeof(name), "%sattn.qkv.weight", prefix);
		attn->qkv_w = gh_load_mmap(wf, name, arena,
					      SAM3_DTYPE_F32, 2, w_dims);
		if (!attn->qkv_w)
			return SAM3_ENOMEM;

		int b_dims[] = {qkv_dim};
		snprintf(name, sizeof(name), "%sattn.qkv.bias", prefix);
		attn->qkv_b = gh_load_mmap(wf, name, arena,
					      SAM3_DTYPE_F32, 1, b_dims);
		if (!attn->qkv_b)
			return SAM3_ENOMEM;
	}

	/* Proj linear: [C, C] weight + [C] bias */
	{
		int w_dims[] = {channels, channels};
		snprintf(name, sizeof(name), "%sattn.proj.weight", prefix);
		attn->proj_w = gh_load_mmap(wf, name, arena,
					      SAM3_DTYPE_F32, 2, w_dims);
		if (!attn->proj_w)
			return SAM3_ENOMEM;

		int b_dims[] = {channels};
		snprintf(name, sizeof(name), "%sattn.proj.bias", prefix);
		attn->proj_b = gh_load_mmap(wf, name, arena,
					      SAM3_DTYPE_F32, 1, b_dims);
		if (!attn->proj_b)
			return SAM3_ENOMEM;
	}

	/* Post-attention LayerNorm */
	{
		int ln_dims[] = {channels};

		snprintf(name, sizeof(name), "%sattn.norm.weight", prefix);
		attn->norm_w = gh_load_mmap(wf, name, arena,
					      SAM3_DTYPE_F32, 1, ln_dims);
		if (!attn->norm_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name), "%sattn.norm.bias", prefix);
		attn->norm_b = gh_load_mmap(wf, name, arena,
					      SAM3_DTYPE_F32, 1, ln_dims);
		if (!attn->norm_b)
			return SAM3_ENOMEM;
	}

	/* Attention bias: load compact, expand to full */
	{
		/*
		 * Compact bias shape: [n_heads, n_offsets].
		 * n_offsets = ws * ws (one per unique abs-offset pair).
		 */
		int n_offsets = ws * ws;
		int ab_dims[] = {n_heads, n_offsets};
		snprintf(name, sizeof(name),
			 "%sattn.attention_biases", prefix);
		struct sam3_tensor *compact;
		compact = gh_load_mmap(wf, name, arena,
					 SAM3_DTYPE_F32, 2, ab_dims);
		if (!compact)
			return SAM3_ENOMEM;

		attn->attn_bias = expand_attention_bias(compact, arena,
							  n_heads, ws);
		if (!attn->attn_bias)
			return SAM3_ENOMEM;
	}

	return SAM3_OK;
}

/* ── Main weight loading ─────────────────────────────────────────── */

enum sam3_error sam3_tinyvit_load(struct sam3_tinyvit *tvit,
				  const struct sam3_weight_file *wf,
				  struct sam3_arena *arena)
{
	char prefix[256];
	enum sam3_error err;

	/* ── Patch embedding ────────────────────────────────────── */

	int half_ch = tvit->embed_dims[0] / 2;

	/* patch_embed.seq.0: Conv(3->half_ch, k=3, s=2, p=1) + BN */
	snprintf(prefix, sizeof(prefix),
		 TVIT_PREFIX "patch_embed.seq.0.");
	err = load_tvit_conv(&tvit->patch_embed_0, wf, arena, prefix,
			       half_ch, 3, 3, 3, 1);
	if (err != SAM3_OK)
		return err;

	/* patch_embed.seq.2: Conv(half_ch->embed_dims[0], k=3, s=2) + BN */
	snprintf(prefix, sizeof(prefix),
		 TVIT_PREFIX "patch_embed.seq.2.");
	err = load_tvit_conv(&tvit->patch_embed_1, wf, arena, prefix,
			       tvit->embed_dims[0], 3, 3, half_ch, 1);
	if (err != SAM3_OK)
		return err;

	/* ── Layers 0-3 ─────────────────────────────────────────── */

	for (int l = 0; l < tvit->n_layers; l++) {
		struct sam3_tvit_layer *layer = &tvit->layers[l];
		int ch = layer->embed_dim;

		for (int b = 0; b < layer->n_blocks; b++) {
			struct sam3_tvit_block *block = &layer->blocks[b];
			char bprefix[256];
			snprintf(bprefix, sizeof(bprefix),
				 TVIT_PREFIX "layers.%d.blocks.%d.",
				 l, b);

			if (block->is_conv_block) {
				/* MBConv block */
				err = load_mbconv(&block->mbconv,
						    wf, arena, bprefix,
						    ch, ch, tvit->mlp_ratio);
				if (err != SAM3_OK)
					return err;
			} else {
				/* TinyViTBlock: attn + local_conv + mlp */
				err = load_attention(&block->attn,
						       wf, arena, bprefix,
						       ch, layer->n_heads,
						       layer->window_size);
				if (err != SAM3_OK)
					return err;

				/* Local conv: 3x3 DW + BN */
				snprintf(prefix, sizeof(prefix),
					 "%slocal_conv.", bprefix);
				err = load_tvit_conv(
					&block->local_conv,
					wf, arena, prefix,
					ch, 3, 3, 1, 1);
				if (err != SAM3_OK)
					return err;

				/* MLP: norm + fc1 + fc2 */
				char name[256];
				int ln_dims[] = {ch};
				int mlp_dim = ch * tvit->mlp_ratio;
				int fc1_dims[] = {mlp_dim, ch};
				int fc1_b_dims[] = {mlp_dim};
				int fc2_dims[] = {ch, mlp_dim};
				int fc2_b_dims[] = {ch};

				snprintf(name, sizeof(name),
					 "%smlp.norm.weight", bprefix);
				block->mlp_norm_w = gh_load_mmap(
					wf, name, arena,
					SAM3_DTYPE_F32, 1, ln_dims);
				if (!block->mlp_norm_w)
					return SAM3_ENOMEM;

				snprintf(name, sizeof(name),
					 "%smlp.norm.bias", bprefix);
				block->mlp_norm_b = gh_load_mmap(
					wf, name, arena,
					SAM3_DTYPE_F32, 1, ln_dims);
				if (!block->mlp_norm_b)
					return SAM3_ENOMEM;

				snprintf(name, sizeof(name),
					 "%smlp.fc1.weight", bprefix);
				block->mlp_fc1_w = gh_load_mmap(
					wf, name, arena,
					SAM3_DTYPE_F32, 2, fc1_dims);
				if (!block->mlp_fc1_w)
					return SAM3_ENOMEM;

				snprintf(name, sizeof(name),
					 "%smlp.fc1.bias", bprefix);
				block->mlp_fc1_b = gh_load_mmap(
					wf, name, arena,
					SAM3_DTYPE_F32, 1, fc1_b_dims);
				if (!block->mlp_fc1_b)
					return SAM3_ENOMEM;

				snprintf(name, sizeof(name),
					 "%smlp.fc2.weight", bprefix);
				block->mlp_fc2_w = gh_load_mmap(
					wf, name, arena,
					SAM3_DTYPE_F32, 2, fc2_dims);
				if (!block->mlp_fc2_w)
					return SAM3_ENOMEM;

				snprintf(name, sizeof(name),
					 "%smlp.fc2.bias", bprefix);
				block->mlp_fc2_b = gh_load_mmap(
					wf, name, arena,
					SAM3_DTYPE_F32, 1, fc2_b_dims);
				if (!block->mlp_fc2_b)
					return SAM3_ENOMEM;
			}
		}

		/* PatchMerging downsample (if present) */
		if (layer->has_downsample) {
			int out_ch = tvit->embed_dims[l + 1];
			snprintf(prefix, sizeof(prefix),
				 TVIT_PREFIX "layers.%d.downsample.",
				 l);
			err = load_patch_merging(&layer->downsample,
						   wf, arena, prefix,
						   ch, out_ch);
			if (err != SAM3_OK)
				return err;
		}
	}

	/* ── Projection head ────────────────────────────────────── */

	int final_ch = tvit->embed_dims[tvit->n_layers - 1];
	int ed = tvit->embed_dim;

	/* proj_conv1: 1x1, final_ch -> embed_dim, no BN */
	{
		char name[256];
		int w_dims[] = {ed, 1, 1, final_ch};
		snprintf(name, sizeof(name),
			 TVIT_PREFIX "projection.conv1.weight");
		tvit->proj_conv1.conv_w = gh_load_mmap(wf, name, arena,
							 SAM3_DTYPE_F32,
							 4, w_dims);
		if (!tvit->proj_conv1.conv_w)
			return SAM3_ENOMEM;
	}

	/* proj_bn: BN(embed_dim) */
	{
		char name[256];
		int bn_dims[] = {ed};

		snprintf(name, sizeof(name),
			 TVIT_PREFIX "projection.bn.weight");
		tvit->proj_bn.bn_w = gh_load_mmap(wf, name, arena,
						     SAM3_DTYPE_F32,
						     1, bn_dims);
		if (!tvit->proj_bn.bn_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 TVIT_PREFIX "projection.bn.bias");
		tvit->proj_bn.bn_b = gh_load_mmap(wf, name, arena,
						     SAM3_DTYPE_F32,
						     1, bn_dims);
		if (!tvit->proj_bn.bn_b)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 TVIT_PREFIX "projection.bn.running_mean");
		tvit->proj_bn.bn_mean = gh_load_mmap(wf, name, arena,
						       SAM3_DTYPE_F32,
						       1, bn_dims);
		if (!tvit->proj_bn.bn_mean)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 TVIT_PREFIX "projection.bn.running_var");
		tvit->proj_bn.bn_var = gh_load_mmap(wf, name, arena,
						      SAM3_DTYPE_F32,
						      1, bn_dims);
		if (!tvit->proj_bn.bn_var)
			return SAM3_ENOMEM;
	}

	/* proj_conv2: 3x3, embed_dim -> embed_dim, with bias */
	{
		char name[256];
		int w_dims[] = {ed, 3, 3, ed};
		snprintf(name, sizeof(name),
			 TVIT_PREFIX "projection.conv2.weight");
		tvit->proj_conv2.conv_w = gh_load_mmap(wf, name, arena,
							 SAM3_DTYPE_F32,
							 4, w_dims);
		if (!tvit->proj_conv2.conv_w)
			return SAM3_ENOMEM;

		int b_dims[] = {ed};
		snprintf(name, sizeof(name),
			 TVIT_PREFIX "projection.conv2.bias");
		tvit->proj_conv2.conv_b = gh_load_mmap(wf, name, arena,
							 SAM3_DTYPE_F32,
							 1, b_dims);
		if (!tvit->proj_conv2.conv_b)
			return SAM3_ENOMEM;
	}

	sam3_log_info("tvit_load: loaded weights for %d layers, projection",
		      tvit->n_layers);

	return SAM3_OK;
}

/* ── Graph construction helpers ──────────────────────────────────── */

/*
 * tvit_conv_bn - Apply conv2d + optional BN (no activation).
 */
static struct sam3_tensor *tvit_conv_bn(
	struct sam3_graph *g, struct sam3_arena *a,
	struct sam3_tensor *x,
	const struct sam3_tvit_conv_weights *cw,
	int stride, int padding, int groups)
{
	x = gh_conv2d(g, a, x, cw->conv_w, cw->conv_b,
		      stride, padding, groups);
	if (!x)
		return NULL;

	if (cw->bn_w) {
		x = gh_batchnorm(g, a, x, cw->bn_w, cw->bn_b,
				   cw->bn_mean, cw->bn_var);
		if (!x)
			return NULL;
	}

	return x;
}

/*
 * tvit_mbconv_forward - Evaluate one MBConv block with GELU.
 *
 * conv1(1x1) -> GELU -> conv2(3x3 DW) -> GELU -> conv3(1x1) ->
 * drop_path -> residual add -> GELU.
 */
static struct sam3_tensor *tvit_mbconv_forward(
	struct sam3_graph *g, struct sam3_arena *a,
	struct sam3_tensor *x,
	const struct sam3_tvit_mbconv *mb,
	int mid_ch)
{
	struct sam3_tensor *shortcut = x;

	/* conv1: 1x1 expand */
	x = tvit_conv_bn(g, a, x, &mb->conv1, 1, 0, 1);
	if (!x)
		return NULL;
	x = gh_gelu(g, a, x);
	if (!x)
		return NULL;

	/* conv2: 3x3 depthwise */
	x = tvit_conv_bn(g, a, x, &mb->conv2, 1, 1, mid_ch);
	if (!x)
		return NULL;
	x = gh_gelu(g, a, x);
	if (!x)
		return NULL;

	/* conv3: 1x1 project */
	x = tvit_conv_bn(g, a, x, &mb->conv3, 1, 0, 1);
	if (!x)
		return NULL;

	/* Residual add + GELU */
	x = gh_add(g, a, x, shortcut);
	if (!x)
		return NULL;
	x = gh_gelu(g, a, x);
	return x;
}

/*
 * tvit_patch_merging - Apply PatchMerging downsample.
 *
 * Input: [1, H, W, C_in] NHWC
 * conv1(1x1) -> GELU -> conv2(3x3 DW s=2) -> conv3(1x1)
 * Output: [1, H/2, W/2, C_out]
 */
static struct sam3_tensor *tvit_patch_merging(
	struct sam3_graph *g, struct sam3_arena *a,
	struct sam3_tensor *x,
	const struct sam3_tvit_patch_merging *pm,
	int out_ch)
{
	/* conv1: 1x1 */
	x = tvit_conv_bn(g, a, x, &pm->conv1, 1, 0, 1);
	if (!x)
		return NULL;
	x = gh_gelu(g, a, x);
	if (!x)
		return NULL;

	/* conv2: 3x3 DW stride 2 */
	x = tvit_conv_bn(g, a, x, &pm->conv2, 2, 1, out_ch);
	if (!x)
		return NULL;
	x = gh_gelu(g, a, x);
	if (!x)
		return NULL;

	/* conv3: 1x1 */
	x = tvit_conv_bn(g, a, x, &pm->conv3, 1, 0, 1);
	return x;
}

/*
 * tvit_window_attention - Window multi-head attention with position bias.
 *
 * Input: [num_windows, ws*ws, C]
 * LayerNorm -> QKV -> split (interleaved per-head) -> scale ->
 * Q@K^T -> +bias -> softmax -> @V -> proj
 * Output: [num_windows, ws*ws, C]
 *
 * Uses 4D batched matmul: [nw, nh, ws*ws, hd] to compute all
 * windows and heads in parallel.
 */
static struct sam3_tensor *tvit_window_attention(
	struct sam3_graph *g, struct sam3_arena *a,
	struct sam3_tensor *x,
	const struct sam3_tvit_attention *attn,
	int channels, int n_heads, int ws)
{
	int nw = x->dims[0];		/* num_windows */
	int N = ws * ws;		/* tokens per window */
	int hd = channels / n_heads;	/* head dim */

	/* Pre-attention LayerNorm (matches Python self.norm(x)) */
	x = gh_layernorm(g, a, x, attn->norm_w, attn->norm_b);
	if (!x)
		return NULL;

	/* Flatten to [nw*N, C] for QKV linear */
	int flat_dims[] = {nw * N, channels};
	struct sam3_tensor *x_flat;
	x_flat = gh_reshape(g, a, x, 2, flat_dims);
	if (!x_flat)
		return NULL;

	/* QKV linear: [nw*N, 3*C] */
	struct sam3_tensor *qkv;
	qkv = gh_linear(g, a, x_flat, attn->qkv_w, attn->qkv_b);
	if (!qkv)
		return NULL;

	/*
	 * Interleaved per-head split (matches Python layout):
	 * reshape to [nw, N, n_heads, 3*hd], then slice on dim 3.
	 */
	int qkv_4d[] = {nw, N, n_heads, 3 * hd};
	qkv = gh_reshape(g, a, qkv, 4, qkv_4d);
	if (!qkv)
		return NULL;

	struct sam3_tensor *Q, *K, *V;
	Q = gh_slice(g, a, qkv, 3, 0, hd);
	if (!Q)
		return NULL;
	K = gh_slice(g, a, qkv, 3, hd, 2 * hd);
	if (!K)
		return NULL;
	V = gh_slice(g, a, qkv, 3, 2 * hd, 3 * hd);
	if (!V)
		return NULL;
	/* Q, K, V: [nw, N, n_heads, hd] */

	/* Permute to [nw, n_heads, N, hd] */
	int perm[] = {0, 2, 1, 3};
	Q = gh_permute(g, a, Q, perm);
	if (!Q)
		return NULL;
	K = gh_permute(g, a, K, perm);
	if (!K)
		return NULL;
	V = gh_permute(g, a, V, perm);
	if (!V)
		return NULL;

	/* Scale Q by 1/sqrt(head_dim) */
	{
		int s_dims[] = {1};
		struct sam3_tensor *scale;
		scale = gh_alloc_tensor(a, SAM3_DTYPE_F32, 1, s_dims);
		if (!scale)
			return NULL;
		*(float *)scale->data = 1.0f / sqrtf((float)hd);
		Q = gh_mul(g, a, Q, scale);
		if (!Q)
			return NULL;
	}

	/* scores = Q @ K^T: [nw, n_heads, N, N] */
	struct sam3_tensor *KT = gh_transpose(g, a, K);
	if (!KT)
		return NULL;
	struct sam3_tensor *scores = gh_matmul(g, a, Q, KT);
	if (!scores)
		return NULL;

	/* Add precomputed attention bias [n_heads, N, N] broadcast on nw */
	scores = gh_add(g, a, scores, attn->attn_bias);
	if (!scores)
		return NULL;

	/* Softmax along last dim */
	scores = gh_softmax(g, a, scores);
	if (!scores)
		return NULL;

	/* attn_out = scores @ V: [nw, n_heads, N, hd] */
	struct sam3_tensor *attn_out = gh_matmul(g, a, scores, V);
	if (!attn_out)
		return NULL;

	/* Permute back to [nw, N, n_heads, hd] then reshape to [nw, N, C] */
	int unperm[] = {0, 2, 1, 3};
	attn_out = gh_permute(g, a, attn_out, unperm);
	if (!attn_out)
		return NULL;

	int merge_dims[] = {nw, N, channels};
	attn_out = gh_reshape(g, a, attn_out, 3, merge_dims);
	if (!attn_out)
		return NULL;

	/* Proj linear: flatten to [nw*N, C] -> linear -> reshape back */
	int proj_flat[] = {nw * N, channels};
	attn_out = gh_reshape(g, a, attn_out, 2, proj_flat);
	if (!attn_out)
		return NULL;
	attn_out = gh_linear(g, a, attn_out, attn->proj_w, attn->proj_b);
	if (!attn_out)
		return NULL;

	int proj_3d[] = {nw, N, channels};
	attn_out = gh_reshape(g, a, attn_out, 3, proj_3d);
	return attn_out;
}

/*
 * tvit_block_forward - Evaluate one TinyViTBlock.
 *
 * Input: [1, H, W, C] NHWC
 * 1. Window attention (with padding if H%ws != 0)
 * 2. Residual add
 * 3. Local conv: 3x3 DW + BN
 * 4. MLP: LN -> fc1 -> GELU -> fc2 -> residual add
 * Output: [1, H, W, C]
 */
static struct sam3_tensor *tvit_block_forward(
	struct sam3_graph *g, struct sam3_arena *a,
	struct sam3_tensor *x,
	const struct sam3_tvit_block *block,
	int H, int W, int C, int n_heads, int ws)
{
	/* Save residual as [H*W, C] */
	int flat_dims[] = {H * W, C};
	struct sam3_tensor *residual;
	residual = gh_reshape(g, a, x, 2, flat_dims);
	if (!residual)
		return NULL;

	/* ── Window Attention ────────────────────────────────── */

	int pad_h = (ws - H % ws) % ws;
	int pad_w = (ws - W % ws) % ws;
	int Hp = H + pad_h;
	int Wp = W + pad_w;
	int padded = (pad_h > 0 || pad_w > 0);

	/* Pad if needed: concat zero tensors along H and W */
	if (padded) {
		if (pad_h > 0) {
			int z_dims[] = {1, pad_h, W, C};
			struct sam3_tensor *zeros;
			zeros = gh_alloc_tensor(a, SAM3_DTYPE_F32,
						  4, z_dims);
			if (!zeros)
				return NULL;
			memset(zeros->data, 0,
			       (size_t)pad_h * W * C * sizeof(float));

			struct sam3_tensor *cat[] = {x, zeros};
			x = gh_concat(g, a, cat, 2, 1);
			if (!x)
				return NULL;
		}
		if (pad_w > 0) {
			int z_dims[] = {1, Hp, pad_w, C};
			struct sam3_tensor *zeros;
			zeros = gh_alloc_tensor(a, SAM3_DTYPE_F32,
						  4, z_dims);
			if (!zeros)
				return NULL;
			memset(zeros->data, 0,
			       (size_t)Hp * pad_w * C * sizeof(float));

			struct sam3_tensor *cat[] = {x, zeros};
			x = gh_concat(g, a, cat, 2, 2);
			if (!x)
				return NULL;
		}
	}
	/* x: [1, Hp, Wp, C] */

	/* Reshape to [Hp*Wp, C] for window partition */
	int padded_flat[] = {Hp * Wp, C};
	x = gh_reshape(g, a, x, 2, padded_flat);
	if (!x)
		return NULL;

	/* Window partition: [num_windows, ws*ws, C] */
	x = gh_window_partition(g, a, x, ws, Hp);
	if (!x)
		return NULL;

	/* Window attention */
	x = tvit_window_attention(g, a, x, &block->attn,
				    C, n_heads, ws);
	if (!x)
		return NULL;

	/* Window unpartition: [Hp*Wp, C] */
	x = gh_window_unpartition(g, a, x, ws, Hp);
	if (!x)
		return NULL;

	/* Unpad if needed */
	if (padded) {
		/* Reshape to [1, Hp, Wp, C] */
		int unpad_4d[] = {1, Hp, Wp, C};
		x = gh_reshape(g, a, x, 4, unpad_4d);
		if (!x)
			return NULL;

		/* Slice to [1, H, W, C] */
		x = gh_slice(g, a, x, 1, 0, H);
		if (!x)
			return NULL;
		if (pad_w > 0) {
			x = gh_slice(g, a, x, 2, 0, W);
			if (!x)
				return NULL;
		}

		/* Flatten to [H*W, C] */
		x = gh_reshape(g, a, x, 2, flat_dims);
		if (!x)
			return NULL;
	}
	/* x: [H*W, C] */

	/* Residual add */
	x = gh_add(g, a, x, residual);
	if (!x)
		return NULL;

	/* ── Local Conv ──────────────────────────────────────── */

	/* Reshape to [1, H, W, C] for conv2d */
	{
		int nhwc[] = {1, H, W, C};
		x = gh_reshape(g, a, x, 4, nhwc);
		if (!x)
			return NULL;
	}

	/* 3x3 DW conv + BN */
	x = tvit_conv_bn(g, a, x, &block->local_conv, 1, 1, C);
	if (!x)
		return NULL;

	/* Flatten back to [H*W, C] */
	x = gh_reshape(g, a, x, 2, flat_dims);
	if (!x)
		return NULL;

	/* ── MLP ─────────────────────────────────────────────── */

	/* Save residual */
	residual = x;

	/* Pre-MLP LayerNorm */
	x = gh_layernorm(g, a, x, block->mlp_norm_w, block->mlp_norm_b);
	if (!x)
		return NULL;

	/* MLP: fc1 -> GELU -> fc2 */
	x = gh_mlp(g, a, x,
		   block->mlp_fc1_w, block->mlp_fc1_b,
		   block->mlp_fc2_w, block->mlp_fc2_b,
		   SAM3_OP_GELU);
	if (!x)
		return NULL;

	/* Residual add */
	x = gh_add(g, a, x, residual);
	if (!x)
		return NULL;

	/* Reshape back to [1, H, W, C] */
	{
		int nhwc[] = {1, H, W, C};
		x = gh_reshape(g, a, x, 4, nhwc);
	}

	return x;
}

/* ── Main graph construction ─────────────────────────────────────── */

struct sam3_tensor *sam3_tinyvit_build(struct sam3_tinyvit *tvit,
				       struct sam3_backend *be,
				       struct sam3_tensor *image,
				       struct sam3_arena *scratch,
				       struct sam3_arena *persist,
				       struct sam3_profiler *profiler)
{
	struct sam3_graph g;
	enum sam3_error err;
	int gs = tvit->grid_size;
	int total_nodes = 0;

	SAM3_PROF_BEGIN(profiler, "tvit_build");

	/*
	 * Compute spatial sizes at each stage for persist buffer sizing.
	 * Max activation size is after patch embed: [1, h0, h0, embed_dims[0]].
	 */
	int h = tvit->img_size;
	h = (h + 2 - 3) / 2 + 1;	/* first patch embed conv */
	h = (h + 2 - 3) / 2 + 1;	/* second patch embed conv */
	int h0 = h;			/* 252 for 1008 input */

	size_t max_bytes = (size_t)h0 * h0 * tvit->embed_dims[0] *
			   sam3_dtype_size(SAM3_DTYPE_F32);
	void *x_buf = sam3_arena_alloc(persist, max_bytes);
	if (!x_buf)
		return NULL;

	int cur_h = h0, cur_w = h0, cur_ch = tvit->embed_dims[0];

	/* ── Phase 0: Preprocess + Patch Embed ──────────────── */

	sam3_graph_init(&g);

	/* Image: CHW [3, img, img] -> NHWC [1, img, img, 3] */
	int nchw_dims[] = {1, 3, tvit->img_size, tvit->img_size};
	struct sam3_tensor *x;
	x = gh_reshape(&g, scratch, image, 4, nchw_dims);
	if (!x)
		return NULL;
	int chw_to_hwc[] = {0, 2, 3, 1};
	x = gh_permute(&g, scratch, x, chw_to_hwc);
	if (!x)
		return NULL;

	/* Conv2d_BN(3->48, k=3, s=2, p=1) + GELU */
	x = tvit_conv_bn(&g, scratch, x, &tvit->patch_embed_0, 2, 1, 1);
	if (!x)
		return NULL;
	x = gh_gelu(&g, scratch, x);
	if (!x)
		return NULL;

	/* Conv2d_BN(48->96, k=3, s=2, p=1) — no activation */
	x = tvit_conv_bn(&g, scratch, x, &tvit->patch_embed_1, 2, 1, 1);
	if (!x)
		return NULL;
	/* x: [1, h0, h0, embed_dims[0]] */

	err = be->ops->graph_eval(be, &g);
	if (err != SAM3_OK) {
		sam3_log_error("tvit_build: patch_embed eval failed (%d)",
			       err);
		return NULL;
	}

	memcpy(x_buf, x->data,
	       (size_t)cur_h * cur_w * cur_ch *
	       sam3_dtype_size(SAM3_DTYPE_F32));
	total_nodes += g.n_nodes;

	sam3_log_debug("tvit_build: patch_embed done [1,%d,%d,%d] "
		       "(%d nodes)", cur_h, cur_w, cur_ch, g.n_nodes);

	/* ── Phases 1-4: Layers (per-block evaluation) ──────── */

	for (int l = 0; l < tvit->n_layers; l++) {
		struct sam3_tvit_layer *layer = &tvit->layers[l];
		int ch = layer->embed_dim;

		/* Evaluate each block individually to bound scratch usage */
		for (int b = 0; b < layer->n_blocks; b++) {
			struct sam3_tvit_block *block = &layer->blocks[b];

			sam3_arena_reset(scratch);
			sam3_graph_init(&g);

			int x_dims[] = {1, cur_h, cur_w, cur_ch};
			x = gh_tensor_wrap(scratch, SAM3_DTYPE_F32,
					     4, x_dims, x_buf);
			if (!x)
				return NULL;

			if (block->is_conv_block) {
				int mid_ch = ch * tvit->mlp_ratio;
				x = tvit_mbconv_forward(
					&g, scratch, x,
					&block->mbconv, mid_ch);
			} else {
				x = tvit_block_forward(
					&g, scratch, x, block,
					cur_h, cur_w, ch,
					layer->n_heads,
					layer->window_size);
			}
			if (!x)
				return NULL;

			err = be->ops->graph_eval(be, &g);
			if (err != SAM3_OK) {
				sam3_log_error("tvit_build: layer %d "
					       "block %d eval failed "
					       "(%d)", l, b, err);
				return NULL;
			}

			memcpy(x_buf, x->data,
			       (size_t)cur_h * cur_w * cur_ch *
			       sam3_dtype_size(SAM3_DTYPE_F32));
			total_nodes += g.n_nodes;
		}

		/* PatchMerging downsample (separate eval) */
		if (layer->has_downsample) {
			int out_ch = tvit->embed_dims[l + 1];

			sam3_arena_reset(scratch);
			sam3_graph_init(&g);

			int x_dims[] = {1, cur_h, cur_w, cur_ch};
			x = gh_tensor_wrap(scratch, SAM3_DTYPE_F32,
					     4, x_dims, x_buf);
			if (!x)
				return NULL;

			x = tvit_patch_merging(
				&g, scratch, x,
				&layer->downsample, out_ch);
			if (!x)
				return NULL;

			err = be->ops->graph_eval(be, &g);
			if (err != SAM3_OK) {
				sam3_log_error("tvit_build: layer %d "
					       "downsample eval failed "
					       "(%d)", l, err);
				return NULL;
			}

			cur_h = (cur_h + 2 - 3) / 2 + 1;
			cur_w = (cur_w + 2 - 3) / 2 + 1;
			cur_ch = out_ch;

			memcpy(x_buf, x->data,
			       (size_t)cur_h * cur_w * cur_ch *
			       sam3_dtype_size(SAM3_DTYPE_F32));
			total_nodes += g.n_nodes;
		}

		sam3_log_debug("tvit_build: layer %d done [1,%d,%d,%d] "
			       "(scratch %zu/%zu)",
			       l, cur_h, cur_w, cur_ch,
			       scratch->offset, scratch->size);
	}
	/* x_buf: [1, gs, gs, final_ch] */

	/* ── Phase 5: Projection head ───────────────────────── */

	sam3_arena_reset(scratch);
	sam3_graph_init(&g);

	{
		int x_dims[] = {1, gs, gs, cur_ch};
		x = gh_tensor_wrap(scratch, SAM3_DTYPE_F32,
				     4, x_dims, x_buf);
		if (!x)
			return NULL;
	}

	/* Conv1x1(final_ch -> embed_dim), no bias, no BN */
	x = gh_conv2d(&g, scratch, x, tvit->proj_conv1.conv_w,
		      NULL, 1, 0, 1);
	if (!x)
		return NULL;

	/* BN(embed_dim) */
	x = gh_batchnorm(&g, scratch, x,
			   tvit->proj_bn.bn_w, tvit->proj_bn.bn_b,
			   tvit->proj_bn.bn_mean, tvit->proj_bn.bn_var);
	if (!x)
		return NULL;

	/* GELU */
	x = gh_gelu(&g, scratch, x);
	if (!x)
		return NULL;

	/* Conv3x3(embed_dim, embed_dim, pad=1) + bias */
	x = gh_conv2d(&g, scratch, x, tvit->proj_conv2.conv_w,
		      tvit->proj_conv2.conv_b, 1, 1, 1);
	if (!x)
		return NULL;

	/* Flatten to [n_patches, embed_dim] */
	int out_dims[] = {gs * gs, tvit->embed_dim};
	x = gh_reshape(&g, scratch, x, 2, out_dims);
	if (!x)
		return NULL;

	err = be->ops->graph_eval(be, &g);
	if (err != SAM3_OK) {
		sam3_log_error("tvit_build: projection eval failed (%d)",
			       err);
		return NULL;
	}
	total_nodes += g.n_nodes;

	/* Copy result to persist arena */
	size_t out_bytes = (size_t)gs * gs * tvit->embed_dim *
			   sam3_dtype_size(SAM3_DTYPE_F32);
	void *out_buf = sam3_arena_alloc(persist, out_bytes);
	if (!out_buf)
		return NULL;
	memcpy(out_buf, x->data, out_bytes);

	struct sam3_tensor *result;
	result = gh_tensor_wrap(persist, SAM3_DTYPE_F32,
				  2, out_dims, out_buf);

	SAM3_PROF_END(profiler, "tvit_build");

	sam3_log_info("tvit_build: output [%d, %d] (%d nodes evaluated)",
		      gs * gs, tvit->embed_dim, total_nodes);

	return result;
}
