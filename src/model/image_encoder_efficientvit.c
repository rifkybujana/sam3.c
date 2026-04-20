/*
 * src/model/image_encoder_efficientvit.c - EfficientViT-B encoder implementation
 *
 * Implements initialization, weight loading, and (stub) graph construction
 * for the EfficientViT-B backbone used in EfficientSAM3. The encoder
 * processes an input image through a conv stem, 4 stages of MBConv and/or
 * EfficientViTBlock layers, and a projection head to produce dense features.
 *
 * Weight loading uses gh_load_mmap exclusively -- all weight data remains
 * in the mmap region with zero copies. Weight names follow the convention
 * "detector_model.vision_encoder.backbone.<submodule>.<param>".
 *
 * Key types:  sam3_efficientvit
 * Depends on: image_encoder_efficientvit.h, graph_helpers.h
 * Used by:    vl_combiner.c (via backbone dispatch)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>

#include "image_encoder_efficientvit.h"
#include "graph_helpers.h"
#include "util/log.h"
#include "util/profile.h"

#define EVIT_PREFIX "detector_model.vision_encoder.backbone."

/* --- Initialization ─ --- */

enum sam3_error sam3_efficientvit_init(struct sam3_efficientvit *evit,
				       const int *width_list,
				       const int *depth_list,
				       int attn_dim,
				       int expand_ratio,
				       int img_size,
				       int embed_dim)
{
	memset(evit, 0, sizeof(*evit));

	/* Validate bounds */
	for (int i = 0; i < 5; i++) {
		if (depth_list[i] >= SAM3_EVIT_MAX_DEPTH) {
			sam3_log_error("evit_init: depth_list[%d]=%d "
				       "exceeds MAX_DEPTH=%d",
				       i, depth_list[i],
				       SAM3_EVIT_MAX_DEPTH);
			return SAM3_EINVAL;
		}
	}

	/* Store configuration */
	for (int i = 0; i < 5; i++) {
		evit->width_list[i] = width_list[i];
		evit->depth_list[i] = depth_list[i];
	}
	evit->attn_dim = attn_dim;
	evit->expand_ratio = expand_ratio;
	evit->img_size = img_size;
	evit->grid_size = img_size / 32;
	evit->embed_dim = embed_dim;

	/* Input stem has depth_list[0] residual DSConv blocks */
	evit->n_stem_blocks = depth_list[0];

	/*
	 * Configure stages 1-4 (indices 0-3).
	 * Each stage has 1 downsample MBConv + depth_list[s+1] blocks.
	 * Stages 3-4 (indices 2-3) use EfficientViTBlocks with LiteMLA.
	 */
	for (int s = 0; s < 4; s++) {
		struct sam3_evit_stage *stage = &evit->stages[s];
		stage->width = width_list[s + 1];
		stage->n_blocks = depth_list[s + 1] + 1; /* +1 for downsample */
		stage->has_attention = (s >= 2) ? 1 : 0;

		/* Mark EfficientViTBlock flags (skip block 0 = downsample) */
		for (int b = 0; b < stage->n_blocks; b++) {
			stage->blocks[b].is_evit_block =
				(stage->has_attention && b > 0) ? 1 : 0;
		}
	}

	sam3_log_info("evit_init: img=%d grid=%d widths=[%d,%d,%d,%d,%d] "
		      "attn_dim=%d expand=%d",
		      img_size, evit->grid_size,
		      width_list[0], width_list[1], width_list[2],
		      width_list[3], width_list[4],
		      attn_dim, expand_ratio);

	return SAM3_OK;
}

/* --- Weight loading helpers  --- */

/*
 * load_conv_weights - Load weights for one ConvLayer.
 *
 * @cw:       Target conv weights struct
 * @wf:       Weight file
 * @arena:    Arena for tensor struct allocation
 * @prefix:   Full name prefix (e.g. "...input_stem.0.")
 * @oc:       Output channels
 * @kh:       Kernel height
 * @kw:       Kernel width
 * @ic:       Input channels per group (ic/groups for depthwise)
 * @has_bias: Load conv bias
 * @has_bn:   Load BatchNorm params (weight, bias, running_mean, running_var)
 *
 * Returns SAM3_OK on success, SAM3_ENOMEM if arena is full.
 */
static enum sam3_error load_conv_weights(
	struct sam3_evit_conv_weights *cw,
	const struct sam3_weight_file *wf,
	struct sam3_arena *arena,
	const char *prefix,
	int oc, int kh, int kw, int ic,
	int has_bias, int has_bn)
{
	char name[256];

	/* Conv weight [OC, KH, KW, IC/groups] */
	int w_dims[] = {oc, kh, kw, ic};
	snprintf(name, sizeof(name), "%sconv.weight", prefix);
	cw->conv_w = gh_load_mmap(wf, name, arena,
				    SAM3_DTYPE_F32, 4, w_dims);
	if (!cw->conv_w)
		return SAM3_ENOMEM;

	/* Optional conv bias [OC] */
	if (has_bias) {
		int b_dims[] = {oc};
		snprintf(name, sizeof(name), "%sconv.bias", prefix);
		cw->conv_b = gh_load_mmap(wf, name, arena,
					    SAM3_DTYPE_F32, 1, b_dims);
		if (!cw->conv_b)
			return SAM3_ENOMEM;
	}

	/* Optional BatchNorm params */
	if (has_bn) {
		int bn_dims[] = {oc};

		snprintf(name, sizeof(name), "%snorm.weight", prefix);
		cw->bn_w = gh_load_mmap(wf, name, arena,
					  SAM3_DTYPE_F32, 1, bn_dims);
		if (!cw->bn_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name), "%snorm.bias", prefix);
		cw->bn_b = gh_load_mmap(wf, name, arena,
					  SAM3_DTYPE_F32, 1, bn_dims);
		if (!cw->bn_b)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name), "%snorm.running_mean", prefix);
		cw->bn_mean = gh_load_mmap(wf, name, arena,
					     SAM3_DTYPE_F32, 1, bn_dims);
		if (!cw->bn_mean)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name), "%snorm.running_var", prefix);
		cw->bn_var = gh_load_mmap(wf, name, arena,
					    SAM3_DTYPE_F32, 1, bn_dims);
		if (!cw->bn_var)
			return SAM3_ENOMEM;
	}

	return SAM3_OK;
}

/*
 * load_mbconv_block - Load weights for one MBConv block.
 *
 * @block:      Target block struct
 * @wf:         Weight file
 * @arena:      Arena for tensor structs
 * @prefix:     Block name prefix (e.g. "...stages.0.blocks.0.")
 * @in_ch:      Input channels
 * @out_ch:     Output channels
 * @expand:     Expansion ratio
 * @fewer_norm: If 1, inverted_conv and depth_conv use bias instead of BN
 */
static enum sam3_error load_mbconv_block(
	struct sam3_evit_block *block,
	const struct sam3_weight_file *wf,
	struct sam3_arena *arena,
	const char *prefix,
	int in_ch, int out_ch, int expand,
	int fewer_norm)
{
	char sub[256];
	int mid_ch = in_ch * expand;
	enum sam3_error err;

	/* inverted_conv: 1x1, in_ch -> mid_ch */
	snprintf(sub, sizeof(sub), "%sinverted_conv.", prefix);
	err = load_conv_weights(&block->inverted_conv, wf, arena, sub,
				  mid_ch, 1, 1, in_ch,
				  fewer_norm ? 1 : 0,
				  fewer_norm ? 0 : 1);
	if (err != SAM3_OK)
		return err;

	/* depth_conv: 3x3 depthwise, mid_ch -> mid_ch (groups=mid_ch) */
	snprintf(sub, sizeof(sub), "%sdepth_conv.", prefix);
	err = load_conv_weights(&block->depth_conv, wf, arena, sub,
				  mid_ch, 3, 3, 1,
				  fewer_norm ? 1 : 0,
				  fewer_norm ? 0 : 1);
	if (err != SAM3_OK)
		return err;

	/* point_conv: 1x1, mid_ch -> out_ch (always has BN, no bias) */
	snprintf(sub, sizeof(sub), "%spoint_conv.", prefix);
	err = load_conv_weights(&block->point_conv, wf, arena, sub,
				  out_ch, 1, 1, mid_ch,
				  0, 1);
	if (err != SAM3_OK)
		return err;

	return SAM3_OK;
}

/*
 * load_litemla_weights - Load weights for one LiteMLA context module.
 *
 * @ctx:      Target LiteMLA weights struct
 * @wf:       Weight file
 * @arena:    Arena for tensor structs
 * @prefix:   Block prefix (e.g. "...stages.2.blocks.1.")
 * @channels: Input/output channel count
 * @attn_dim: Attention head dimension
 */
static enum sam3_error load_litemla_weights(
	struct sam3_evit_litemla_weights *ctx,
	const struct sam3_weight_file *wf,
	struct sam3_arena *arena,
	const char *prefix,
	int channels, int attn_dim)
{
	char sub[256];
	enum sam3_error err;

	int n_heads = channels / attn_dim;
	int qkv_ch = channels + 2 * n_heads * attn_dim;

	/* qkv: 1x1 conv, no BN, no bias */
	snprintf(sub, sizeof(sub), "%scontext.qkv.", prefix);
	err = load_conv_weights(&ctx->qkv, wf, arena, sub,
				  qkv_ch, 1, 1, channels,
				  0, 0);
	if (err != SAM3_OK)
		return err;

	/*
	 * aggreg: depthwise 5x5 + pointwise 1x1 grouped.
	 * Both are raw nn.Conv2d (not ConvLayer), so tensor names
	 * use ".weight" directly, not ".conv.weight".
	 * No bias, no BN.
	 * DW groups = qkv_ch, PW groups = 3*n_heads.
	 */
	{
		char name[256];

		/* aggreg_dw: depthwise 5x5, qkv_ch channels */
		int dw_dims[] = {qkv_ch, 5, 5, 1};
		snprintf(name, sizeof(name),
			 "%scontext.aggreg.0.0.weight", prefix);
		ctx->aggreg_dw.conv_w = gh_load_mmap(wf, name, arena,
						       SAM3_DTYPE_F32,
						       4, dw_dims);
		if (!ctx->aggreg_dw.conv_w)
			return SAM3_ENOMEM;

		/* aggreg_pw: grouped 1x1, qkv_ch channels */
		int pw_dims[] = {qkv_ch, 1, 1, attn_dim};
		snprintf(name, sizeof(name),
			 "%scontext.aggreg.0.1.weight", prefix);
		ctx->aggreg_pw.conv_w = gh_load_mmap(wf, name, arena,
						       SAM3_DTYPE_F32,
						       4, pw_dims);
		if (!ctx->aggreg_pw.conv_w)
			return SAM3_ENOMEM;
	}

	/*
	 * proj: 1x1 ConvLayer + BN, no bias.
	 * Input is concatenated identity + aggregated = 2*channels.
	 */
	int proj_in = channels * 2;
	snprintf(sub, sizeof(sub), "%scontext.proj.", prefix);
	err = load_conv_weights(&ctx->proj, wf, arena, sub,
				  channels, 1, 1, proj_in,
				  0, 1);
	if (err != SAM3_OK)
		return err;

	return SAM3_OK;
}

/* --- Main weight loading  --- */

enum sam3_error sam3_efficientvit_load(struct sam3_efficientvit *evit,
					const struct sam3_weight_file *wf,
					struct sam3_arena *arena)
{
	char prefix[256];
	enum sam3_error err;
	int w0 = evit->width_list[0];

	/* --- Input stem  --- */

	/* stem_conv: Conv2d(3 -> w0, kernel=3, stride=2, pad=1) + BN */
	snprintf(prefix, sizeof(prefix), EVIT_PREFIX "input_stem.0.");
	err = load_conv_weights(&evit->stem_conv, wf, arena, prefix,
				  w0, 3, 3, 3,
				  0, 1);
	if (err != SAM3_OK)
		return err;

	/* stem residual DSConv blocks */
	for (int i = 0; i < evit->n_stem_blocks; i++) {
		/*
		 * DSConv in input_stem uses indices starting at 1:
		 *   input_stem.{i+1}.main.depth_conv.*
		 *   input_stem.{i+1}.main.point_conv.*
		 */
		snprintf(prefix, sizeof(prefix),
			 EVIT_PREFIX "input_stem.%d.main.depth_conv.", i + 1);
		err = load_conv_weights(&evit->stem_blocks[i].depth_conv,
					  wf, arena, prefix,
					  w0, 3, 3, 1,
					  0, 1);
		if (err != SAM3_OK)
			return err;

		snprintf(prefix, sizeof(prefix),
			 EVIT_PREFIX "input_stem.%d.main.point_conv.", i + 1);
		err = load_conv_weights(&evit->stem_blocks[i].point_conv,
					  wf, arena, prefix,
					  w0, 1, 1, w0,
					  0, 1);
		if (err != SAM3_OK)
			return err;
	}

	/* --- Stages 1-4  --- */

	for (int s = 0; s < 4; s++) {
		struct sam3_evit_stage *stage = &evit->stages[s];
		int in_ch = evit->width_list[s];
		int out_ch = evit->width_list[s + 1];

		/*
		 * Block 0: downsample MBConv (stride 2).
		 * Stages 3-4 (s >= 2) use fewer_norm for downsample.
		 */
		int fewer_norm_ds = (s >= 2) ? 1 : 0;
		snprintf(prefix, sizeof(prefix),
			 EVIT_PREFIX "stages.%d.blocks.0.", s);
		err = load_mbconv_block(&stage->blocks[0], wf, arena,
					  prefix, in_ch, out_ch,
					  evit->expand_ratio,
					  fewer_norm_ds);
		if (err != SAM3_OK)
			return err;

		/* Remaining blocks */
		for (int b = 1; b < stage->n_blocks; b++) {
			snprintf(prefix, sizeof(prefix),
				 EVIT_PREFIX "stages.%d.blocks.%d.",
				 s, b);

			if (stage->has_attention) {
				/*
				 * EfficientViTBlock: context (LiteMLA) +
				 * local (MBConv fewer_norm).
				 */

				/* Load LiteMLA context module */
				err = load_litemla_weights(
					&stage->blocks[b].context,
					wf, arena, prefix,
					out_ch, evit->attn_dim);
				if (err != SAM3_OK)
					return err;

				/*
				 * Load local MBConv (fewer_norm).
				 * Weight prefix uses "local." subpath.
				 */
				char local_prefix[256];
				snprintf(local_prefix, sizeof(local_prefix),
					 "%slocal.", prefix);
				err = load_mbconv_block(
					&stage->blocks[b], wf, arena,
					local_prefix, out_ch, out_ch,
					evit->expand_ratio, 1);
				if (err != SAM3_OK)
					return err;
			} else {
				/* Normal MBConv block (no fewer_norm) */
				err = load_mbconv_block(
					&stage->blocks[b], wf, arena,
					prefix, out_ch, out_ch,
					evit->expand_ratio, 0);
				if (err != SAM3_OK)
					return err;
			}
		}
	}

	/* --- Projection head  --- */

	/*
	 * Projection: conv1(1x1, final_ch -> embed_dim) -> BN(embed_dim)
	 *             -> GELU -> conv2(3x3, embed_dim -> embed_dim + bias).
	 * BN uses naming: projection.bn.{weight,bias,...} (not .norm.).
	 */
	int final_ch = evit->width_list[4];
	int ed = evit->embed_dim;

	/* proj_conv1: 1x1, final_ch -> embed_dim, no BN, no bias */
	{
		char name[256];
		int w_dims[] = {ed, 1, 1, final_ch};
		snprintf(name, sizeof(name),
			 EVIT_PREFIX "projection.conv1.weight");
		evit->proj_conv1.conv_w = gh_load_mmap(wf, name, arena,
							 SAM3_DTYPE_F32,
							 4, w_dims);
		if (!evit->proj_conv1.conv_w)
			return SAM3_ENOMEM;
	}

	/* proj_bn: BN(embed_dim) */
	{
		char name[256];
		int bn_dims[] = {ed};

		snprintf(name, sizeof(name),
			 EVIT_PREFIX "projection.bn.weight");
		evit->proj_bn.bn_w = gh_load_mmap(wf, name, arena,
						     SAM3_DTYPE_F32,
						     1, bn_dims);
		if (!evit->proj_bn.bn_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 EVIT_PREFIX "projection.bn.bias");
		evit->proj_bn.bn_b = gh_load_mmap(wf, name, arena,
						     SAM3_DTYPE_F32,
						     1, bn_dims);
		if (!evit->proj_bn.bn_b)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 EVIT_PREFIX "projection.bn.running_mean");
		evit->proj_bn.bn_mean = gh_load_mmap(wf, name, arena,
						       SAM3_DTYPE_F32,
						       1, bn_dims);
		if (!evit->proj_bn.bn_mean)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 EVIT_PREFIX "projection.bn.running_var");
		evit->proj_bn.bn_var = gh_load_mmap(wf, name, arena,
						      SAM3_DTYPE_F32,
						      1, bn_dims);
		if (!evit->proj_bn.bn_var)
			return SAM3_ENOMEM;
	}

	/* proj_conv2: 3x3, embed_dim -> embed_dim, with bias, no BN */
	{
		char name[256];
		int w_dims[] = {ed, 3, 3, ed};
		snprintf(name, sizeof(name),
			 EVIT_PREFIX "projection.conv2.weight");
		evit->proj_conv2.conv_w = gh_load_mmap(wf, name, arena,
							 SAM3_DTYPE_F32,
							 4, w_dims);
		if (!evit->proj_conv2.conv_w)
			return SAM3_ENOMEM;

		int b_dims[] = {ed};
		snprintf(name, sizeof(name),
			 EVIT_PREFIX "projection.conv2.bias");
		evit->proj_conv2.conv_b = gh_load_mmap(wf, name, arena,
							 SAM3_DTYPE_F32,
							 1, b_dims);
		if (!evit->proj_conv2.conv_b)
			return SAM3_ENOMEM;
	}

	sam3_log_info("evit_load: loaded weights for %d stem blocks, "
		      "4 stages, projection",
		      evit->n_stem_blocks);

	return SAM3_OK;
}

/* --- Graph construction helpers  --- */

/*
 * evit_conv_bn - Apply conv2d + optional BN (no activation).
 *
 * Caller applies activation (HSwish, GELU) separately.
 */
static struct sam3_tensor *evit_conv_bn(
	struct sam3_graph *g, struct sam3_arena *a,
	struct sam3_tensor *x,
	const struct sam3_evit_conv_weights *cw,
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
 * evit_mbconv_forward - Evaluate one MBConv block.
 *
 * inverted_conv(1x1) -> depth_conv(3x3 DW) -> point_conv(1x1).
 * Activation (HSwish) after inverted and depth convs.
 * No activation after point conv.
 */
static struct sam3_tensor *evit_mbconv_forward(
	struct sam3_graph *g, struct sam3_arena *a,
	struct sam3_tensor *x,
	const struct sam3_evit_block *block,
	int stride, int mid_ch)
{
	/* inverted_conv: 1x1 */
	x = evit_conv_bn(g, a, x, &block->inverted_conv, 1, 0, 1);
	if (!x)
		return NULL;
	x = gh_hswish(g, a, x);
	if (!x)
		return NULL;

	/* depth_conv: 3x3 depthwise */
	x = evit_conv_bn(g, a, x, &block->depth_conv,
			   stride, 1, mid_ch);
	if (!x)
		return NULL;
	x = gh_hswish(g, a, x);
	if (!x)
		return NULL;

	/* point_conv: 1x1, no activation */
	x = evit_conv_bn(g, a, x, &block->point_conv, 1, 0, 1);
	return x;
}

/*
 * evit_litemla_head - Linear attention for a single head.
 *
 * @qkv_2d:  QKV data for this head [H*W, 3*dim]
 * @ones_1d: Pre-allocated ones vector [H*W]
 * @dim:     Attention head dimension
 *
 * Returns attention output [H*W, dim].
 */
static struct sam3_tensor *evit_litemla_head(
	struct sam3_graph *g, struct sam3_arena *a,
	struct sam3_tensor *qkv_2d,
	struct sam3_tensor *ones_1d,
	int dim)
{
	/* Split q, k, v: each [H*W, dim] */
	struct sam3_tensor *q, *k, *v;
	q = gh_slice(g, a, qkv_2d, 1, 0, dim);
	if (!q)
		return NULL;
	k = gh_slice(g, a, qkv_2d, 1, dim, 2 * dim);
	if (!k)
		return NULL;
	v = gh_slice(g, a, qkv_2d, 1, 2 * dim, 3 * dim);
	if (!v)
		return NULL;

	/* ReLU kernel */
	q = gh_relu(g, a, q);
	if (!q)
		return NULL;
	k = gh_relu(g, a, k);
	if (!k)
		return NULL;

	/*
	 * Ones-padding trick for normalization:
	 * v_pad = concat([v, ones], axis=1) → [H*W, dim+1]
	 */
	int hw = qkv_2d->dims[0];
	int ones_dims[] = {hw, 1};
	struct sam3_tensor *ones_col;
	ones_col = gh_reshape(g, a, ones_1d, 2, ones_dims);
	if (!ones_col)
		return NULL;

	struct sam3_tensor *cat_inputs[] = {v, ones_col};
	struct sam3_tensor *v_pad;
	v_pad = gh_concat(g, a, cat_inputs, 2, 1); /* [H*W, dim+1] */
	if (!v_pad)
		return NULL;

	/*
	 * Linear attention matmuls:
	 *   v_padT @ k  → [dim+1, dim]
	 *   (v_padT@k) @ qT → [dim+1, H*W]
	 */
	struct sam3_tensor *v_padT = gh_transpose(g, a, v_pad);
	if (!v_padT)
		return NULL;
	struct sam3_tensor *qT = gh_transpose(g, a, q);
	if (!qT)
		return NULL;

	struct sam3_tensor *vk = gh_matmul(g, a, v_padT, k);
	if (!vk)
		return NULL;

	struct sam3_tensor *out = gh_matmul(g, a, vk, qT);
	if (!out)
		return NULL;
	/* out: [dim+1, H*W] */

	/* Split numerator / denominator */
	struct sam3_tensor *numer, *denom;
	numer = gh_slice(g, a, out, 0, 0, dim);     /* [dim, H*W] */
	if (!numer)
		return NULL;
	denom = gh_slice(g, a, out, 0, dim, dim + 1); /* [1, H*W] */
	if (!denom)
		return NULL;

	/* Reshape denom to 1D for broadcasting: [H*W] */
	int denom_1d_dims[] = {hw};
	denom = gh_reshape(g, a, denom, 1, denom_1d_dims);
	if (!denom)
		return NULL;

	/* Guard div-by-zero: denom += eps before dividing */
	{
		int eps_dims[] = {1};
		struct sam3_tensor *eps;
		eps = gh_alloc_tensor(a, SAM3_DTYPE_F32, 1, eps_dims);
		if (!eps)
			return NULL;
		*(float *)eps->data = 1e-6f;
		denom = gh_add(g, a, denom, eps);
		if (!denom)
			return NULL;
	}

	/* Normalize: numer / denom with last-dim broadcast */
	struct sam3_tensor *result = gh_div(g, a, numer, denom);
	if (!result)
		return NULL;
	/* result: [dim, H*W] */

	/* Transpose back to [H*W, dim] */
	result = gh_transpose(g, a, result);
	return result;
}

/*
 * evit_litemla_forward - Full LiteMLA context module.
 *
 * @x:       Input [1, H, W, channels] NHWC
 * @ctx:     LiteMLA weights
 * @channels: Channel count
 * @attn_dim: Head dimension
 *
 * Returns LiteMLA output [1, H, W, channels] after proj + BN.
 * Does NOT add skip connection — caller handles that.
 */
static struct sam3_tensor *evit_litemla_forward(
	struct sam3_graph *g, struct sam3_arena *a,
	struct sam3_tensor *x,
	const struct sam3_evit_litemla_weights *ctx,
	int channels, int attn_dim)
{
	int H = x->dims[1];
	int W = x->dims[2];
	int hw = H * W;
	int n_heads = channels / attn_dim;
	int qkv_ch = 3 * channels;
	int n_scales = 2; /* identity + 1 aggregated scale */
	int total_heads = n_scales * n_heads;

	/* qkv: Conv1x1(channels, 3*channels) — no BN, no act */
	struct sam3_tensor *qkv;
	qkv = gh_conv2d(g, a, x, ctx->qkv.conv_w, NULL, 1, 0, 1);
	if (!qkv)
		return NULL;
	/* qkv: [1, H, W, qkv_ch] */

	/* aggreg: DWConv(5x5) + PWConv(1x1, grouped) */
	struct sam3_tensor *agg;
	agg = gh_conv2d(g, a, qkv, ctx->aggreg_dw.conv_w, NULL,
			  1, 2, qkv_ch); /* 5x5 DW, pad=2, groups=qkv_ch */
	if (!agg)
		return NULL;
	agg = gh_conv2d(g, a, agg, ctx->aggreg_pw.conv_w, NULL,
			  1, 0, 3 * n_heads); /* 1x1, groups=3*n_heads */
	if (!agg)
		return NULL;
	/* agg: [1, H, W, qkv_ch] */

	/* concat identity qkv + aggregated: [1, H, W, 2*qkv_ch] */
	struct sam3_tensor *cat_inputs[] = {qkv, agg};
	struct sam3_tensor *multi;
	multi = gh_concat(g, a, cat_inputs, 2, 3); /* concat on C axis */
	if (!multi)
		return NULL;

	/*
	 * Flatten spatial and reshape for per-head processing:
	 * [1, H, W, 2*qkv_ch] → [H*W, total_heads, 3*dim]
	 */
	int flat_dims[] = {hw, total_heads, 3 * attn_dim};
	multi = gh_reshape(g, a, multi, 3, flat_dims);
	if (!multi)
		return NULL;

	/*
	 * Create ones vector [H*W] for the padding trick.
	 * Only the struct is allocated from scratch; we fill data
	 * directly since this runs once per build.
	 */
	int ones_dims[] = {hw};
	struct sam3_tensor *ones_1d;
	ones_1d = gh_alloc_tensor(a, SAM3_DTYPE_F32, 1, ones_dims);
	if (!ones_1d)
		return NULL;
	{
		float *od = (float *)ones_1d->data;
		for (int i = 0; i < hw; i++)
			od[i] = 1.0f;
	}

	/*
	 * Per-head linear attention.
	 * Slice each head [H*W, 3*dim], compute attention, collect.
	 */
	if (total_heads > 128) {
		sam3_log_error("evit_litemla: total_heads=%d exceeds "
			       "stack limit", total_heads);
		return NULL;
	}

	struct sam3_tensor *head_outputs[128]; /* 2*n_heads, max ~24 */
	for (int h = 0; h < total_heads; h++) {
		/* Slice head h from dim 1: [H*W, 3*dim] */
		int h_dims[] = {hw, 3 * attn_dim};
		struct sam3_tensor *head_qkv;

		/* multi is [H*W, total_heads, 3*dim] — slice dim 1 */
		head_qkv = gh_slice(g, a, multi, 1, h, h + 1);
		if (!head_qkv)
			return NULL;
		/* [H*W, 1, 3*dim] → reshape to [H*W, 3*dim] */
		head_qkv = gh_reshape(g, a, head_qkv, 2, h_dims);
		if (!head_qkv)
			return NULL;

		head_outputs[h] = evit_litemla_head(g, a, head_qkv,
						      ones_1d, attn_dim);
		if (!head_outputs[h])
			return NULL;
		/* head_outputs[h]: [H*W, dim] */
	}

	/* Concat all heads: [H*W, total_heads * dim] = [H*W, 2*channels] */
	struct sam3_tensor *attn_out;
	attn_out = gh_concat(g, a, head_outputs, total_heads, 1);
	if (!attn_out)
		return NULL;

	/* Reshape back to spatial: [1, H, W, 2*channels] */
	int spatial_dims[] = {1, H, W, n_scales * channels};
	attn_out = gh_reshape(g, a, attn_out, 4, spatial_dims);
	if (!attn_out)
		return NULL;

	/* proj: Conv1x1(2*channels -> channels) + BN */
	attn_out = evit_conv_bn(g, a, attn_out, &ctx->proj, 1, 0, 1);
	return attn_out;
}

/* --- Main graph construction  --- */

/*
 * sam3_efficientvit_build - Build and evaluate the EfficientViT encoder.
 *
 * Evaluates per-stage to bound scratch arena usage. Each stage builds
 * a subgraph, evaluates it, copies the result to a persist buffer,
 * then resets scratch for the next stage. This mirrors the per-block
 * batching strategy used by the Hiera ViT encoder.
 *
 * Phases: preprocess+stem, stage 0-3, projection.
 */
struct sam3_tensor *sam3_efficientvit_build(struct sam3_efficientvit *evit,
					     struct sam3_backend *be,
					     struct sam3_tensor *image,
					     struct sam3_arena *scratch,
					     struct sam3_arena *persist,
					     struct sam3_profiler *profiler)
{
	struct sam3_graph g;
	enum sam3_error err;
	int gs = evit->grid_size;
	int w0 = evit->width_list[0];
	int total_nodes = 0;

	SAM3_PROF_BEGIN(profiler, "evit_build");

	/*
	 * Allocate persist buffer for intermediate activations.
	 * Max size is after stem: [1, img/2, img/2, width_list[0]].
	 * Reused across all stages; each stage output fits within.
	 */
	int cur_h = evit->img_size / 2;
	int cur_w = cur_h;
	int cur_ch = w0;
	size_t max_bytes = (size_t)cur_h * cur_w * cur_ch *
			   sam3_dtype_size(SAM3_DTYPE_F32);
	void *x_buf = sam3_arena_alloc(persist, max_bytes);
	if (!x_buf)
		return NULL;

	/* --- Phase 0: Preprocess + Stem --- */

	sam3_graph_init(&g);

	/*
	 * Image preprocessing: CHW [3, img, img] → NHWC [1, img, img, 3].
	 * Matches the Hiera ViT input convention.
	 */
	int nchw_dims[] = {1, 3, evit->img_size, evit->img_size};
	struct sam3_tensor *x;
	x = gh_reshape(&g, scratch, image, 4, nchw_dims);
	if (!x)
		return NULL;
	int chw_to_hwc[] = {0, 2, 3, 1};
	x = gh_permute(&g, scratch, x, chw_to_hwc);
	if (!x)
		return NULL;
	/* x: [1, img, img, 3] NHWC */

	/* Conv2d(3 -> w0, k=3, s=2, p=1) + BN + HSwish */
	x = evit_conv_bn(&g, scratch, x, &evit->stem_conv, 2, 1, 1);
	if (!x)
		return NULL;
	x = gh_hswish(&g, scratch, x);
	if (!x)
		return NULL;
	/* x: [1, img/2, img/2, w0] */

	/* Stem residual DSConv blocks */
	for (int i = 0; i < evit->n_stem_blocks; i++) {
		struct sam3_tensor *skip = x;

		x = evit_conv_bn(&g, scratch, x,
				   &evit->stem_blocks[i].depth_conv,
				   1, 1, w0);
		if (!x)
			return NULL;
		x = gh_hswish(&g, scratch, x);
		if (!x)
			return NULL;

		x = evit_conv_bn(&g, scratch, x,
				   &evit->stem_blocks[i].point_conv,
				   1, 0, 1);
		if (!x)
			return NULL;

		x = gh_add(&g, scratch, x, skip);
		if (!x)
			return NULL;
	}

	err = be->ops->graph_eval(be, &g);
	if (err != SAM3_OK) {
		sam3_log_error("evit_build: stem eval failed (%d)", err);
		return NULL;
	}

	memcpy(x_buf, x->data,
	       (size_t)cur_h * cur_w * cur_ch *
	       sam3_dtype_size(SAM3_DTYPE_F32));
	total_nodes += g.n_nodes;

	sam3_log_debug("evit_build: stem done [1,%d,%d,%d] (%d nodes, "
		       "scratch %zu/%zu)",
		       cur_h, cur_w, cur_ch, g.n_nodes,
		       scratch->offset, scratch->size);

	/* --- Phases 1-4: Stages  --- */

	for (int s = 0; s < 4; s++) {
		struct sam3_evit_stage *stage = &evit->stages[s];
		int in_ch = evit->width_list[s];
		int out_ch = evit->width_list[s + 1];
		int mid_ch = in_ch * evit->expand_ratio;

		sam3_arena_reset(scratch);
		sam3_graph_init(&g);

		/* Wrap persist buffer as input tensor */
		int x_dims[] = {1, cur_h, cur_w, cur_ch};
		x = gh_tensor_wrap(scratch, SAM3_DTYPE_F32,
				     4, x_dims, x_buf);
		if (!x)
			return NULL;

		/*
		 * Block 0: downsample MBConv (stride 2).
		 * No residual (shape change).
		 */
		x = evit_mbconv_forward(&g, scratch, x,
					  &stage->blocks[0], 2, mid_ch);
		if (!x)
			return NULL;

		/* Remaining blocks */
		mid_ch = out_ch * evit->expand_ratio;

		for (int b = 1; b < stage->n_blocks; b++) {
			struct sam3_tensor *skip = x;

			if (stage->blocks[b].is_evit_block) {
				struct sam3_tensor *ctx_out;
				ctx_out = evit_litemla_forward(
					&g, scratch, x,
					&stage->blocks[b].context,
					out_ch, evit->attn_dim);
				if (!ctx_out)
					return NULL;
				x = gh_add(&g, scratch, ctx_out, skip);
				if (!x)
					return NULL;

				skip = x;
				x = evit_mbconv_forward(
					&g, scratch, x,
					&stage->blocks[b],
					1, mid_ch);
				if (!x)
					return NULL;
				x = gh_add(&g, scratch, x, skip);
				if (!x)
					return NULL;
			} else {
				x = evit_mbconv_forward(
					&g, scratch, x,
					&stage->blocks[b],
					1, mid_ch);
				if (!x)
					return NULL;
				x = gh_add(&g, scratch, x, skip);
				if (!x)
					return NULL;
			}
		}

		err = be->ops->graph_eval(be, &g);
		if (err != SAM3_OK) {
			sam3_log_error("evit_build: stage %d eval "
				       "failed (%d)", s, err);
			return NULL;
		}

		/* Update shape: spatial halved, channels changed */
		cur_h /= 2;
		cur_w /= 2;
		cur_ch = out_ch;

		memcpy(x_buf, x->data,
		       (size_t)cur_h * cur_w * cur_ch *
		       sam3_dtype_size(SAM3_DTYPE_F32));
		total_nodes += g.n_nodes;

		sam3_log_debug("evit_build: stage %d done [1,%d,%d,%d] "
			       "(%d nodes, scratch %zu/%zu)",
			       s, cur_h, cur_w, cur_ch, g.n_nodes,
			       scratch->offset, scratch->size);
	}
	/* x_buf: [1, gs, gs, final_ch] */

	/* --- Phase 5: Projection head --- */

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
	x = gh_conv2d(&g, scratch, x, evit->proj_conv1.conv_w,
		      NULL, 1, 0, 1);
	if (!x)
		return NULL;

	/* BN(embed_dim) */
	x = gh_batchnorm(&g, scratch, x,
			   evit->proj_bn.bn_w, evit->proj_bn.bn_b,
			   evit->proj_bn.bn_mean, evit->proj_bn.bn_var);
	if (!x)
		return NULL;

	/* GELU activation */
	x = gh_gelu(&g, scratch, x);
	if (!x)
		return NULL;

	/* Conv3x3(embed_dim, embed_dim, pad=1) + bias */
	x = gh_conv2d(&g, scratch, x, evit->proj_conv2.conv_w,
		      evit->proj_conv2.conv_b, 1, 1, 1);
	if (!x)
		return NULL;
	/* x: [1, gs, gs, embed_dim] */

	/*
	 * Flatten to [n_patches, embed_dim] for compatibility with
	 * the FPN neck, which expects the same shape as Hiera output.
	 */
	int out_dims[] = {gs * gs, evit->embed_dim};
	x = gh_reshape(&g, scratch, x, 2, out_dims);
	if (!x)
		return NULL;

	err = be->ops->graph_eval(be, &g);
	if (err != SAM3_OK) {
		sam3_log_error("evit_build: projection eval failed (%d)",
			       err);
		return NULL;
	}
	total_nodes += g.n_nodes;

	/* Copy result to persist arena */
	size_t out_bytes = (size_t)gs * gs * evit->embed_dim *
			   sam3_dtype_size(SAM3_DTYPE_F32);
	void *out_buf = sam3_arena_alloc(persist, out_bytes);
	if (!out_buf)
		return NULL;
	memcpy(out_buf, x->data, out_bytes);

	struct sam3_tensor *result;
	result = gh_tensor_wrap(persist, SAM3_DTYPE_F32,
				  2, out_dims, out_buf);

	SAM3_PROF_END(profiler, "evit_build");

	sam3_log_info("evit_build: output [%d, %d] (%d nodes evaluated)",
		      gs * gs, evit->embed_dim, total_nodes);

	return result;
}
