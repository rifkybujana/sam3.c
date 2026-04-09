/*
 * src/model/segmentation.c - SAM3 UniversalSegmentationHead graph construction
 *
 * Builds the MaskFormer-style segmentation head compute graph. The FPN
 * pixel decoder fuses multi-scale backbone features from coarse to fine
 * using nearest-neighbor upsampling + skip add + 3×3 conv + GroupNorm(8)
 * + ReLU. The mask prediction uses a 3-layer MLP on query embeddings
 * and a dot product with instance-projected pixel features.
 *
 * Weight prefix: detector_model.mask_decoder.*
 *
 * Key types:  sam3_seg_head
 * Depends on: segmentation.h, graph_helpers.h, util/log.h
 * Used by:    sam3_image.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>

#include "segmentation.h"
#include "graph_helpers.h"
#include "util/log.h"

#define WP "detector_model.mask_decoder."

enum sam3_error sam3_seg_head_init(struct sam3_seg_head *head,
				   int d_model, int n_attn_heads)
{
	if (!head || d_model <= 0 || n_attn_heads <= 0)
		return SAM3_EINVAL;

	memset(head, 0, sizeof(*head));
	head->d_model = d_model;
	head->n_attn_heads = n_attn_heads;

	return SAM3_OK;
}

enum sam3_error sam3_seg_head_load(struct sam3_seg_head *head,
				   const struct sam3_weight_file *wf,
				   struct sam3_arena *arena)
{
	int d = head->d_model;
	char name[128];

	int conv_w_dims[] = {d, d, 3, 3};
	int proj_w_dims[] = {d, d, 1, 1};
	int lin_w_dims[] = {d, d};
	int d_dims[] = {d};

	/* FPN pixel decoder: 3 stages */
	for (int i = 0; i < SAM3_SEG_FPN_STAGES; i++) {
		snprintf(name, sizeof(name),
			 WP "pixel_decoder.conv_layers.%d.weight", i);
		head->fpn[i].conv_w = gh_load_mmap(wf, name, arena,
			SAM3_DTYPE_F32, 4, conv_w_dims);
		if (!head->fpn[i].conv_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 WP "pixel_decoder.conv_layers.%d.bias", i);
		head->fpn[i].conv_b = gh_load_mmap(wf, name, arena,
			SAM3_DTYPE_F32, 1, d_dims);
		if (!head->fpn[i].conv_b)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 WP "pixel_decoder.norms.%d.weight", i);
		head->fpn[i].gn_w = gh_load_mmap(wf, name, arena,
			SAM3_DTYPE_F32, 1, d_dims);
		if (!head->fpn[i].gn_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 WP "pixel_decoder.norms.%d.bias", i);
		head->fpn[i].gn_b = gh_load_mmap(wf, name, arena,
			SAM3_DTYPE_F32, 1, d_dims);
		if (!head->fpn[i].gn_b)
			return SAM3_ENOMEM;
	}

	/* Instance projection: 1×1 conv */
	head->inst_proj_w = gh_load_mmap(wf,
		WP "instance_projection.weight", arena,
		SAM3_DTYPE_F32, 4, proj_w_dims);
	if (!head->inst_proj_w)
		return SAM3_ENOMEM;

	head->inst_proj_b = gh_load_mmap(wf,
		WP "instance_projection.bias", arena,
		SAM3_DTYPE_F32, 1, d_dims);
	if (!head->inst_proj_b)
		return SAM3_ENOMEM;

	/* Mask embedder: 3-layer MLP */
	for (int i = 0; i < SAM3_SEG_MASK_MLP_LAYERS; i++) {
		snprintf(name, sizeof(name),
			 WP "mask_embedder.layers.%d.weight", i);
		head->mask_mlp[i].w = gh_load_mmap(wf, name, arena,
			SAM3_DTYPE_F32, 2, lin_w_dims);
		if (!head->mask_mlp[i].w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 WP "mask_embedder.layers.%d.bias", i);
		head->mask_mlp[i].b = gh_load_mmap(wf, name, arena,
			SAM3_DTYPE_F32, 1, d_dims);
		if (!head->mask_mlp[i].b)
			return SAM3_ENOMEM;
	}

	/* Prompt cross-attention: separate Q/K/V/O projections */
	head->pxattn_q_w = gh_load_mmap(wf,
		WP "prompt_cross_attn.q_proj.weight", arena,
		SAM3_DTYPE_F32, 2, lin_w_dims);
	head->pxattn_q_b = gh_load_mmap(wf,
		WP "prompt_cross_attn.q_proj.bias", arena,
		SAM3_DTYPE_F32, 1, d_dims);
	head->pxattn_k_w = gh_load_mmap(wf,
		WP "prompt_cross_attn.k_proj.weight", arena,
		SAM3_DTYPE_F32, 2, lin_w_dims);
	head->pxattn_k_b = gh_load_mmap(wf,
		WP "prompt_cross_attn.k_proj.bias", arena,
		SAM3_DTYPE_F32, 1, d_dims);
	head->pxattn_v_w = gh_load_mmap(wf,
		WP "prompt_cross_attn.v_proj.weight", arena,
		SAM3_DTYPE_F32, 2, lin_w_dims);
	head->pxattn_v_b = gh_load_mmap(wf,
		WP "prompt_cross_attn.v_proj.bias", arena,
		SAM3_DTYPE_F32, 1, d_dims);
	head->pxattn_o_w = gh_load_mmap(wf,
		WP "prompt_cross_attn.o_proj.weight", arena,
		SAM3_DTYPE_F32, 2, lin_w_dims);
	head->pxattn_o_b = gh_load_mmap(wf,
		WP "prompt_cross_attn.o_proj.bias", arena,
		SAM3_DTYPE_F32, 1, d_dims);

	/* Prompt cross-attention norm */
	head->pxattn_norm_w = gh_load_mmap(wf,
		WP "prompt_cross_attn_norm.weight", arena,
		SAM3_DTYPE_F32, 1, d_dims);
	head->pxattn_norm_b = gh_load_mmap(wf,
		WP "prompt_cross_attn_norm.bias", arena,
		SAM3_DTYPE_F32, 1, d_dims);

	if (!head->pxattn_q_w || !head->pxattn_norm_b)
		return SAM3_ENOMEM;

	return SAM3_OK;
}

/*
 * build_prompt_cross_attn - Apply prompt cross-attention to encoder states.
 *
 * x_out = x + cross_attn(layernorm(x), text, text)
 *
 * Uses separate Q/K/V/O projections with per-head SDPA.
 */
static struct sam3_tensor *build_prompt_cross_attn(
	struct sam3_seg_head *head,
	struct sam3_graph *g,
	struct sam3_tensor *x,
	struct sam3_tensor *text,
	struct sam3_arena *a)
{
	int d = head->d_model;
	int n_heads = head->n_attn_heads;
	int head_dim = d / n_heads;

	/* LayerNorm on encoder states */
	struct sam3_tensor *normed = gh_layernorm(g, a, x,
		head->pxattn_norm_w, head->pxattn_norm_b);
	if (!normed)
		return NULL;

	/* Q from encoder states, K/V from text features */
	struct sam3_tensor *q = gh_linear(g, a, normed,
		head->pxattn_q_w, head->pxattn_q_b);
	struct sam3_tensor *k = gh_linear(g, a, text,
		head->pxattn_k_w, head->pxattn_k_b);
	struct sam3_tensor *v = gh_linear(g, a, text,
		head->pxattn_v_w, head->pxattn_v_b);
	if (!q || !k || !v)
		return NULL;

	/* Per-head SDPA */
	struct sam3_tensor *head_outs[64];
	for (int h = 0; h < n_heads; h++) {
		int hstart = h * head_dim;
		int hend = hstart + head_dim;

		struct sam3_tensor *hq = gh_slice(g, a, q, 1, hstart, hend);
		struct sam3_tensor *hk = gh_slice(g, a, k, 1, hstart, hend);
		struct sam3_tensor *hv = gh_slice(g, a, v, 1, hstart, hend);
		if (!hq || !hk || !hv)
			return NULL;

		struct sam3_tensor *ho = gh_sdpa(g, a, hq, hk, hv,
						 NULL, head_dim);
		if (!ho)
			return NULL;
		head_outs[h] = ho;
	}

	/* Concatenate heads */
	struct sam3_tensor *merged;
	if (n_heads == 1) {
		merged = head_outs[0];
	} else {
		merged = gh_concat(g, a, head_outs, n_heads, 1);
		if (!merged)
			return NULL;
	}

	/* Output projection */
	struct sam3_tensor *attn_out = gh_linear(g, a, merged,
		head->pxattn_o_w, head->pxattn_o_b);
	if (!attn_out)
		return NULL;

	/* Residual: x + attn_out */
	return gh_add(g, a, x, attn_out);
}

/*
 * build_pixel_decoder - FPN pixel decoder.
 *
 * Takes 3 features (encoder_nchw at 72×72, feat_2x, feat_4x) and
 * iterates from coarse to fine: interpolate + skip add + 3×3 conv +
 * GroupNorm(8) + ReLU.
 *
 * Only 2 FPN stages are used (matching Python: conv_layers[2] and
 * norms[2] are never invoked at inference). The encoder output at
 * 72×72 replaces the backbone's 1× feature as the starting point.
 *
 * Returns high-resolution pixel features [1, d, H4, W4].
 */
static struct sam3_tensor *build_pixel_decoder(
	struct sam3_seg_head *head,
	struct sam3_graph *g,
	struct sam3_tensor *enc_nchw,
	struct sam3_tensor *feat_2x,
	struct sam3_tensor *feat_4x,
	struct sam3_arena *a)
{
	/*
	 * Python pixel decoder (maskformer_segmentation.py:215-231):
	 *   backbone_feats = [feat_4x(288²), feat_2x(144²), enc(72²)]
	 *   prev = backbone_feats[-1] = enc at 72×72
	 *   fpn_feats = [feat_4x, feat_2x]
	 *   iterate reversed(fpn_feats): feat_2x, feat_4x
	 *
	 * Stage 0: upsample 72→144, add feat_2x, conv[0], norm[0], relu
	 * Stage 1: upsample 144→288, add feat_4x, conv[1], norm[1], relu
	 * conv[2] and norm[2] are allocated but never used.
	 */
	struct sam3_tensor *skip_feats[] = {feat_2x, feat_4x};
	struct sam3_tensor *prev = enc_nchw;
	int n_stages = 2;

	for (int i = 0; i < n_stages; i++) {
		struct sam3_tensor *curr = skip_feats[i];

		/* Upsample prev to match curr spatial dims.
		 * Python: F.interpolate(prev, size=curr.shape[-2:])
		 * Skip when already matching (e.g. enc and feat_1x both 72×72).
		 */
		int ch = curr->dims[2], cw = curr->dims[3];
		int ph = prev->dims[2], pw = prev->dims[3];

		if (ph != ch || pw != cw) {
			int scale = ch / ph;
			sam3_log_debug("seg: FPN stage %d upsample %dx%d -> "
				       "%dx%d (scale %d)",
				       i, ph, pw, ch, cw, scale);
			prev = gh_upsample(g, a, prev, scale);
			if (!prev) {
				sam3_log_error("seg: FPN stage %d upsample "
					       "fail", i);
				return NULL;
			}
		} else {
			sam3_log_debug("seg: FPN stage %d skip upsample "
				       "(%dx%d == %dx%d)",
				       i, ph, pw, ch, cw);
		}

		/* Skip connection: curr + upsampled prev */
		prev = gh_add(g, a, curr, prev);
		if (!prev) {
			sam3_log_error("seg: FPN stage %d add fail", i);
			return NULL;
		}

		/* 3×3 conv with stride=1, padding=1 */
		prev = gh_conv2d(g, a, prev,
				  head->fpn[i].conv_w,
				  head->fpn[i].conv_b,
				  1, 1);
		if (!prev) {
			sam3_log_error("seg: FPN stage %d conv fail", i);
			return NULL;
		}

		/* GroupNorm(8) + ReLU */
		prev = gh_groupnorm(g, a, prev,
				     head->fpn[i].gn_w,
				     head->fpn[i].gn_b,
				     SAM3_SEG_GN_GROUPS);
		if (!prev) {
			sam3_log_error("seg: FPN stage %d groupnorm fail", i);
			return NULL;
		}

		prev = gh_relu(g, a, prev);
		if (!prev) {
			sam3_log_error("seg: FPN stage %d relu fail", i);
			return NULL;
		}
	}

	return prev;
}

/*
 * build_mask_embedder - 3-layer MLP on query embeddings.
 *
 * layers[0]: linear + ReLU
 * layers[1]: linear + ReLU
 * layers[2]: linear (no activation)
 *
 * Returns [n_queries, d_model].
 */
static struct sam3_tensor *build_mask_embedder(
	struct sam3_seg_head *head,
	struct sam3_graph *g,
	struct sam3_tensor *queries,
	struct sam3_arena *a)
{
	struct sam3_tensor *x = queries;

	for (int i = 0; i < SAM3_SEG_MASK_MLP_LAYERS; i++) {
		x = gh_linear(g, a, x,
			       head->mask_mlp[i].w,
			       head->mask_mlp[i].b);
		if (!x)
			return NULL;

		/* ReLU on all but last layer */
		if (i < SAM3_SEG_MASK_MLP_LAYERS - 1) {
			x = gh_relu(g, a, x);
			if (!x)
				return NULL;
		}
	}

	return x;
}

struct sam3_tensor *sam3_seg_head_build_cross_attn(
	struct sam3_seg_head *head,
	struct sam3_graph *g,
	struct sam3_tensor *encoder_states,
	struct sam3_tensor *text_features,
	struct sam3_arena *arena)
{
	return build_prompt_cross_attn(head, g, encoder_states,
				        text_features, arena);
}

struct sam3_tensor *sam3_seg_head_build_pixel_decoder(
	struct sam3_seg_head *head,
	struct sam3_graph *g,
	struct sam3_tensor *enc_nchw,
	struct sam3_tensor *feat_2x,
	struct sam3_tensor *feat_4x,
	struct sam3_arena *arena)
{
	return build_pixel_decoder(
		head, g, enc_nchw, feat_2x, feat_4x, arena);
}

struct sam3_tensor *sam3_seg_head_build_fpn(
	struct sam3_seg_head *head,
	struct sam3_graph *g,
	struct sam3_tensor *enc_nchw,
	struct sam3_tensor *feat_2x,
	struct sam3_tensor *feat_4x,
	struct sam3_arena *arena)
{
	struct sam3_tensor *pixel_embed = build_pixel_decoder(
		head, g, enc_nchw, feat_2x, feat_4x, arena);
	if (!pixel_embed)
		return NULL;

	head->_debug_pixel_embed = pixel_embed;

	struct sam3_tensor *inst = gh_conv2d(g, arena, pixel_embed,
					      head->inst_proj_w,
					      head->inst_proj_b,
					      1, 0);
	return inst;
}

struct sam3_tensor *sam3_seg_head_build_mask_embed(
	struct sam3_seg_head *head,
	struct sam3_graph *g,
	struct sam3_tensor *queries,
	struct sam3_arena *arena)
{
	return build_mask_embedder(head, g, queries, arena);
}

struct sam3_tensor *sam3_seg_head_build(
	struct sam3_seg_head *head,
	struct sam3_graph *g,
	struct sam3_tensor *queries,
	struct sam3_tensor *encoder_states,
	struct sam3_tensor *feat_2x,
	struct sam3_tensor *feat_4x,
	int enc_h, int enc_w,
	struct sam3_arena *arena)
{
	int d = head->d_model;

	if (!head || !g || !queries || !encoder_states || !arena)
		return NULL;
	if (!feat_2x || !feat_4x)
		return NULL;

	sam3_log_debug("seg_head: queries [%d, %d], enc [%d, %d], "
		       "spatial %dx%d",
		       queries->dims[0], queries->dims[1],
		       encoder_states->dims[0], encoder_states->dims[1],
		       enc_h, enc_w);

	struct sam3_tensor *enc = encoder_states;

	/*
	 * Step 2: Reshape encoder output to NCHW.
	 * [seq, d_model] → manual transpose+reshape → [1, d, enc_h, enc_w]
	 *
	 * Done manually (not as graph op) because the input tensor is
	 * from the persist arena (materialized in a previous graph eval).
	 * The MLX cache can hold stale references for cross-eval tensors.
	 */
	{
		int seq = enc->dims[0];
		int nchw_d[] = {1, d, enc_h, enc_w};
		struct sam3_tensor *nchw = gh_alloc_tensor(arena,
			enc->dtype, 4, nchw_d);
		if (!nchw)
			return NULL;
		const float *src = (const float *)enc->data;
		float *dst = (float *)nchw->data;
		/* Transpose [seq, d] → [d, seq] then treat as [1,d,h,w] */
		for (int s = 0; s < seq; s++)
			for (int c = 0; c < d; c++)
				dst[c * seq + s] = src[s * d + c];
		enc = nchw;
	}

	/* Stash for debug: encoder states before and after cross-attn */
	head->_debug_enc_nchw = enc;

	/*
	 * Step 3: FPN pixel decoder.
	 * Fuses encoder NCHW + backbone features from coarse to fine.
	 */
	struct sam3_tensor *pixel_embed = build_pixel_decoder(
		head, g, enc, feat_2x, feat_4x, arena);
	if (!pixel_embed) {
		sam3_log_error("seg: pixel decoder failed");
		return NULL;
	}

	int final_h = pixel_embed->dims[2];
	int final_w = pixel_embed->dims[3];
	int final_hw = final_h * final_w;

	sam3_log_debug("seg: pixel embed [%d, %d, %d, %d]",
		       pixel_embed->dims[0], pixel_embed->dims[1],
		       final_h, final_w);

	/* Stash for debug */
	head->_debug_pixel_embed = pixel_embed;

	/*
	 * Step 4: Instance projection (1×1 conv).
	 * pixel_embed [1, d, H, W] → instance_embeds [1, d, H, W]
	 */
	struct sam3_tensor *inst = gh_conv2d(g, arena, pixel_embed,
					      head->inst_proj_w,
					      head->inst_proj_b,
					      1, 0);
	if (!inst) {
		sam3_log_error("seg: instance projection failed");
		return NULL;
	}

	/* Stash for debug */
	head->_debug_inst = inst;

	/*
	 * Step 5: Mask embedder MLP on queries.
	 * queries [n_queries, d] → mask_embed [n_queries, d]
	 */
	struct sam3_tensor *mask_embed = build_mask_embedder(
		head, g, queries, arena);
	if (!mask_embed) {
		sam3_log_error("seg: mask embedder MLP failed");
		return NULL;
	}

	/* Stash for debug */
	head->_debug_mask_embed = mask_embed;

	/*
	 * Step 6: Dot product for mask logits.
	 * einsum("bqc,bchw->bqhw") with batch=1:
	 *   mask_embed [n_q, d] @ inst_flat [d, H*W] → [n_q, H*W]
	 *
	 * Reshape instance features: [1, d, H, W] → [d, H*W]
	 */
	int flat_dims[] = {d, final_hw};
	struct sam3_tensor *inst_flat = gh_reshape(g, arena, inst,
						    2, flat_dims);
	if (!inst_flat)
		return NULL;

	struct sam3_tensor *masks = gh_matmul(g, arena,
					       mask_embed, inst_flat);
	if (!masks)
		return NULL;

	/* Reshape to spatial: [n_queries, H*W] → [n_queries, H, W] */
	int n_queries = queries->dims[0];
	int mask_dims[] = {n_queries, final_h, final_w};
	masks = gh_reshape(g, arena, masks, 3, mask_dims);
	if (!masks)
		return NULL;

	sam3_log_debug("seg: masks [%d, %d, %d]",
		       n_queries, final_h, final_w);

	return masks;
}
