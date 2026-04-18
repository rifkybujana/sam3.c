/*
 * src/model/memory_encoder.c - SimpleMaskEncoder graph construction
 *
 * Builds the compute graph for encoding mask predictions and pixel
 * features into compact memory tokens. The pipeline follows the
 * Python SimpleMaskEncoder: sigmoid -> mask downsampler (4-layer
 * strided conv cascade) -> pixel feature projection -> add -> fuser
 * (2 CXBlock layers) -> output projection. All operations use NHWC
 * layout with OHWI conv weights.
 *
 * Weight prefix: tracker_model.maskmem_backbone.*
 *
 * Key types:  sam3_memory_encoder
 * Depends on: memory_encoder.h, graph_helpers.h, util/log.h
 * Used by:    model/tracker.c (future)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>

#include "memory_encoder.h"
#include "graph_helpers.h"
#include "util/log.h"

#define WP "tracker_model.maskmem_backbone."

enum sam3_error sam3_memory_encoder_init(struct sam3_memory_encoder *enc,
					 int in_dim, int out_dim)
{
	if (!enc || in_dim <= 0 || out_dim <= 0)
		return SAM3_EINVAL;

	memset(enc, 0, sizeof(*enc));
	enc->in_dim = in_dim;
	enc->out_dim = out_dim;
	enc->interpol_h = 1152;
	enc->interpol_w = 1152;

	return SAM3_OK;
}

/*
 * Compute the channel count at each mask downsampler layer.
 * Layer i has channels: stride^(i+1) * out_ch_base, capped at in_dim.
 * With stride=2, out_ch_base=1: 4, 16, 64, 256.
 */
static int ds_layer_channels(int layer, int in_dim)
{
	/* Channel progression: 4, 16, 64, 256 (for in_dim=256) */
	int ch = 1;
	for (int i = 0; i <= layer; i++)
		ch *= 4;
	if (ch > in_dim)
		ch = in_dim;
	return ch;
}

enum sam3_error sam3_memory_encoder_load(struct sam3_memory_encoder *enc,
					 const struct sam3_weight_file *wf,
					 struct sam3_arena *arena)
{
	if (!enc || !arena) {
		sam3_log_error("mem_enc load: NULL argument");
		return SAM3_EINVAL;
	}

	int d = enc->in_dim;
	char name[160];

	/*
	 * Mask downsampler: 4 layers.
	 * Python encoder sequential indices:
	 *   0,1,2 = conv0, ln0, gelu
	 *   3,4,5 = conv1, ln1, gelu
	 *   6,7,8 = conv2, ln2, gelu
	 *   9,10,11 = conv3, ln3, gelu
	 *   12 = final 1x1 projection
	 */
	int prev_ch = 1; /* mask input has 1 channel */
	for (int i = 0; i < SAM3_MEMENC_DS_LAYERS; i++) {
		int out_ch = ds_layer_channels(i, d);
		int seq_conv = i * 3;     /* 0, 3, 6, 9 */
		int seq_ln = i * 3 + 1;   /* 1, 4, 7, 10 */

		/* Conv weight: OHWI [out_ch, 3, 3, in_ch] */
		int cw_dims[] = {out_ch, 3, 3, prev_ch};
		snprintf(name, sizeof(name),
			 WP "mask_downsampler.encoder.%d.weight",
			 seq_conv);
		enc->ds[i].conv_w = gh_load_mmap(wf, name, arena,
			SAM3_DTYPE_F32, 4, cw_dims);
		if (!enc->ds[i].conv_w)
			return SAM3_ENOMEM;

		int cb_dims[] = {out_ch};
		snprintf(name, sizeof(name),
			 WP "mask_downsampler.encoder.%d.bias",
			 seq_conv);
		enc->ds[i].conv_b = gh_load_mmap(wf, name, arena,
			SAM3_DTYPE_F32, 1, cb_dims);
		if (!enc->ds[i].conv_b)
			return SAM3_ENOMEM;

		/* LayerNorm weight/bias [out_ch] */
		snprintf(name, sizeof(name),
			 WP "mask_downsampler.encoder.%d.weight",
			 seq_ln);
		enc->ds[i].ln_w = gh_load_mmap(wf, name, arena,
			SAM3_DTYPE_F32, 1, cb_dims);
		if (!enc->ds[i].ln_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 WP "mask_downsampler.encoder.%d.bias",
			 seq_ln);
		enc->ds[i].ln_b = gh_load_mmap(wf, name, arena,
			SAM3_DTYPE_F32, 1, cb_dims);
		if (!enc->ds[i].ln_b)
			return SAM3_ENOMEM;

		prev_ch = out_ch;
	}

	/* Final 1x1 projection: OHWI [in_dim, 1, 1, last_ch] */
	{
		int last_ch = ds_layer_channels(SAM3_MEMENC_DS_LAYERS - 1, d);
		int pw_dims[] = {d, 1, 1, last_ch};
		int pb_dims[] = {d};

		enc->ds_proj_w = gh_load_mmap(wf,
			WP "mask_downsampler.encoder.12.weight",
			arena, SAM3_DTYPE_F32, 4, pw_dims);
		if (!enc->ds_proj_w)
			return SAM3_ENOMEM;

		enc->ds_proj_b = gh_load_mmap(wf,
			WP "mask_downsampler.encoder.12.bias",
			arena, SAM3_DTYPE_F32, 1, pb_dims);
		if (!enc->ds_proj_b)
			return SAM3_ENOMEM;
	}

	/* Pixel feature projection: 1x1 conv OHWI [d, 1, 1, d] */
	{
		int pw_dims[] = {d, 1, 1, d};
		int pb_dims[] = {d};

		enc->pix_proj_w = gh_load_mmap(wf,
			WP "pix_feat_proj.weight",
			arena, SAM3_DTYPE_F32, 4, pw_dims);
		if (!enc->pix_proj_w)
			return SAM3_ENOMEM;

		enc->pix_proj_b = gh_load_mmap(wf,
			WP "pix_feat_proj.bias",
			arena, SAM3_DTYPE_F32, 1, pb_dims);
		if (!enc->pix_proj_b)
			return SAM3_ENOMEM;
	}

	/* Fuser: 2 CXBlock layers */
	int mlp_dim = d * SAM3_MEMENC_FUSER_EXPAND; /* 1024 */
	for (int i = 0; i < SAM3_MEMENC_FUSER_LAYERS; i++) {
		struct sam3_cxblock *blk = &enc->fuser[i];

		/* Depthwise conv: OHWI [d, 7, 7, 1] (groups=d) */
		int dw_dims[] = {d, 7, 7, 1};
		int d_dims[] = {d};
		int mlp_dims[] = {mlp_dim};

		snprintf(name, sizeof(name),
			 WP "fuser.layers.%d.dwconv.weight", i);
		blk->dwconv_w = gh_load_mmap(wf, name, arena,
			SAM3_DTYPE_F32, 4, dw_dims);
		if (!blk->dwconv_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 WP "fuser.layers.%d.dwconv.bias", i);
		blk->dwconv_b = gh_load_mmap(wf, name, arena,
			SAM3_DTYPE_F32, 1, d_dims);
		if (!blk->dwconv_b)
			return SAM3_ENOMEM;

		/* LayerNorm */
		snprintf(name, sizeof(name),
			 WP "fuser.layers.%d.norm.weight", i);
		blk->norm_w = gh_load_mmap(wf, name, arena,
			SAM3_DTYPE_F32, 1, d_dims);
		if (!blk->norm_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 WP "fuser.layers.%d.norm.bias", i);
		blk->norm_b = gh_load_mmap(wf, name, arena,
			SAM3_DTYPE_F32, 1, d_dims);
		if (!blk->norm_b)
			return SAM3_ENOMEM;

		/* pwconv1: Linear(d, mlp_dim) as [mlp_dim, d] */
		int pw1_w_dims[] = {mlp_dim, d};
		snprintf(name, sizeof(name),
			 WP "fuser.layers.%d.pwconv1.weight", i);
		blk->pwconv1_w = gh_load_mmap(wf, name, arena,
			SAM3_DTYPE_F32, 2, pw1_w_dims);
		if (!blk->pwconv1_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 WP "fuser.layers.%d.pwconv1.bias", i);
		blk->pwconv1_b = gh_load_mmap(wf, name, arena,
			SAM3_DTYPE_F32, 1, mlp_dims);
		if (!blk->pwconv1_b)
			return SAM3_ENOMEM;

		/* pwconv2: Linear(mlp_dim, d) as [d, mlp_dim] */
		int pw2_w_dims[] = {d, mlp_dim};
		snprintf(name, sizeof(name),
			 WP "fuser.layers.%d.pwconv2.weight", i);
		blk->pwconv2_w = gh_load_mmap(wf, name, arena,
			SAM3_DTYPE_F32, 2, pw2_w_dims);
		if (!blk->pwconv2_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 WP "fuser.layers.%d.pwconv2.bias", i);
		blk->pwconv2_b = gh_load_mmap(wf, name, arena,
			SAM3_DTYPE_F32, 1, d_dims);
		if (!blk->pwconv2_b)
			return SAM3_ENOMEM;

		/* Layer scale gamma [d] */
		snprintf(name, sizeof(name),
			 WP "fuser.layers.%d.gamma", i);
		blk->gamma = gh_load_mmap(wf, name, arena,
			SAM3_DTYPE_F32, 1, d_dims);
		if (!blk->gamma)
			return SAM3_ENOMEM;
	}

	/* Output projection: 1x1 conv OHWI [out_dim, 1, 1, d] */
	{
		int ow_dims[] = {enc->out_dim, 1, 1, d};
		int ob_dims[] = {enc->out_dim};

		enc->out_proj_w = gh_load_mmap(wf,
			WP "out_proj.weight",
			arena, SAM3_DTYPE_F32, 4, ow_dims);
		if (!enc->out_proj_w)
			return SAM3_ENOMEM;

		enc->out_proj_b = gh_load_mmap(wf,
			WP "out_proj.bias",
			arena, SAM3_DTYPE_F32, 1, ob_dims);
		if (!enc->out_proj_b)
			return SAM3_ENOMEM;
	}

	/*
	 * Precompute position encoding.
	 * Spatial resolution after downsampler: interpol/16 = 72x72
	 * num_pos_feats = out_dim (64), output is [H, W, 128].
	 */
	int pos_h = enc->interpol_h / 16;
	int pos_w = enc->interpol_w / 16;
	enum sam3_error err = sam3_pos_encoding_precompute(
		&enc->pos_enc, pos_h, pos_w, enc->out_dim, arena);
	if (err != SAM3_OK) {
		sam3_log_error("mem_enc: pos encoding precompute failed");
		return err;
	}

	sam3_log_info("memory encoder loaded (in=%d out=%d pos=%dx%d)",
		      d, enc->out_dim, pos_h, pos_w);
	return SAM3_OK;
}

/*
 * build_mask_downsampler - Build 4-layer strided conv cascade.
 *
 * Input: [1, Hm, Wm, 1] mask (post-sigmoid)
 * Output: [1, Hm/16, Wm/16, in_dim]
 *
 * Each layer: Conv2d(k=3,s=2,p=1) -> LayerNorm -> GELU
 * Final: Conv2d(k=1,s=1,p=0) projection to embed_dim.
 */
static struct sam3_tensor *build_mask_downsampler(
	struct sam3_memory_encoder *enc,
	struct sam3_graph *g,
	struct sam3_tensor *mask,
	struct sam3_arena *a)
{
	struct sam3_tensor *x = mask;

	for (int i = 0; i < SAM3_MEMENC_DS_LAYERS; i++) {
		x = gh_conv2d(g, a, x,
			      enc->ds[i].conv_w,
			      enc->ds[i].conv_b,
			      2, 1, 1);
		if (!x) {
			sam3_log_error("mem_enc: ds conv %d failed", i);
			return NULL;
		}

		x = gh_layernorm(g, a, x,
				 enc->ds[i].ln_w,
				 enc->ds[i].ln_b);
		if (!x) {
			sam3_log_error("mem_enc: ds layernorm %d failed", i);
			return NULL;
		}

		x = gh_gelu(g, a, x);
		if (!x) {
			sam3_log_error("mem_enc: ds gelu %d failed", i);
			return NULL;
		}
	}

	/* Final 1x1 projection to embed_dim */
	x = gh_conv2d(g, a, x, enc->ds_proj_w, enc->ds_proj_b,
		      1, 0, 1);
	if (!x) {
		sam3_log_error("mem_enc: ds final proj failed");
		return NULL;
	}

	return x;
}

/*
 * build_cxblock - Build one CXBlock (ConvNeXt-style block).
 *
 * Input: [1, H, W, dim] NHWC
 * Pipeline:
 *   1. Depthwise conv 7x7 (groups=dim, padding=3)
 *   2. LayerNorm on channel dim
 *   3. Linear: dim -> dim*4 (pwconv1) -> GELU -> dim*4 -> dim (pwconv2)
 *   4. Multiply by gamma (layer_scale)
 *   5. Residual add
 * Output: [1, H, W, dim] NHWC
 */
static struct sam3_tensor *build_cxblock(
	struct sam3_cxblock *blk,
	struct sam3_graph *g,
	struct sam3_tensor *input,
	int dim,
	struct sam3_arena *a)
{
	struct sam3_tensor *residual = input;

	/* 1. Depthwise conv 7x7, groups=dim */
	struct sam3_tensor *x = gh_conv2d(g, a, input,
		blk->dwconv_w, blk->dwconv_b,
		1, 3, dim);
	if (!x) {
		sam3_log_error("mem_enc: cxblock dwconv failed");
		return NULL;
	}

	/* 2. LayerNorm (on last dim, NHWC) */
	x = gh_layernorm(g, a, x, blk->norm_w, blk->norm_b);
	if (!x) {
		sam3_log_error("mem_enc: cxblock layernorm failed");
		return NULL;
	}

	/*
	 * 3. Pointwise MLP: reshape to [N*H*W, dim], apply linear,
	 *    then reshape back.
	 *
	 * Input is [1, H, W, dim]. Flatten spatial to [H*W, dim]
	 * for the linear layers, then reshape back.
	 */
	int h = x->dims[1];
	int w = x->dims[2];
	int hw = h * w;

	int flat_dims[] = {hw, dim};
	struct sam3_tensor *flat = gh_reshape(g, a, x, 2, flat_dims);
	if (!flat) {
		sam3_log_error("mem_enc: cxblock flat reshape failed (hw=%d dim=%d)",
			       hw, dim);
		return NULL;
	}

	/* pwconv1: dim -> dim*4 */
	flat = gh_linear(g, a, flat, blk->pwconv1_w, blk->pwconv1_b);
	if (!flat) {
		sam3_log_error("mem_enc: cxblock pwconv1 failed");
		return NULL;
	}

	flat = gh_gelu(g, a, flat);
	if (!flat) {
		sam3_log_error("mem_enc: cxblock gelu failed");
		return NULL;
	}

	/* pwconv2: dim*4 -> dim */
	flat = gh_linear(g, a, flat, blk->pwconv2_w, blk->pwconv2_b);
	if (!flat) {
		sam3_log_error("mem_enc: cxblock pwconv2 failed");
		return NULL;
	}

	/* Multiply by gamma (layer_scale, broadcast over H*W) */
	flat = gh_mul(g, a, flat, blk->gamma);
	if (!flat) {
		sam3_log_error("mem_enc: cxblock gamma mul failed");
		return NULL;
	}

	/* Reshape back to NHWC [1, H, W, dim] */
	int nhwc_dims[] = {1, h, w, dim};
	x = gh_reshape(g, a, flat, 4, nhwc_dims);
	if (!x) {
		sam3_log_error("mem_enc: cxblock final reshape failed");
		return NULL;
	}

	/* 5. Residual add */
	x = gh_add(g, a, residual, x);
	if (!x) {
		sam3_log_error("mem_enc: cxblock residual add failed");
		return NULL;
	}

	return x;
}

enum sam3_error sam3_memory_encoder_build(struct sam3_memory_encoder *enc,
					  struct sam3_graph *g,
					  struct sam3_tensor *pix_feat,
					  struct sam3_tensor *masks,
					  struct sam3_arena *arena,
					  struct sam3_tensor **out_feat,
					  struct sam3_tensor **out_pos)
{
	if (!enc || !g || !pix_feat || !masks || !arena ||
	    !out_feat || !out_pos) {
		sam3_log_error("mem_enc build: NULL argument");
		return SAM3_EINVAL;
	}

	int d = enc->in_dim;

	sam3_log_debug("mem_enc: pix_feat [%d,%d,%d,%d] mask [%d,%d,%d,%d]",
		       pix_feat->dims[0], pix_feat->dims[1],
		       pix_feat->dims[2], pix_feat->dims[3],
		       masks->dims[0], masks->dims[1],
		       masks->dims[2], masks->dims[3]);

	/*
	 * Python SimpleMaskEncoder is invoked from the tracker with
	 * skip_mask_sigmoid=True. The sigmoid / (mask > 0) branch and
	 * the sigmoid_scale_for_mem_enc / sigmoid_bias_for_mem_enc
	 * affine are applied by the caller (see
	 * preprocess_mask_for_mem_enc in sam3_video.c), so we skip the
	 * internal sigmoid here and feed `masks` straight to the
	 * downsampler.
	 *
	 * 1. Mask downsampler.
	 * Mask is expected at interpol_size already. The 4 conv layers
	 * with stride=2 each produce total stride=16.
	 */
	struct sam3_tensor *mask_down = build_mask_downsampler(
		enc, g, masks, arena);
	if (!mask_down) {
		sam3_log_error("mem_enc: mask downsampler failed");
		return SAM3_ENOMEM;
	}

	sam3_log_debug("mem_enc: mask_down [%d,%d,%d,%d]",
		       mask_down->dims[0], mask_down->dims[1],
		       mask_down->dims[2], mask_down->dims[3]);

	/* 3. Pixel feature projection: 1x1 conv */
	struct sam3_tensor *pix_proj = gh_conv2d(g, arena, pix_feat,
		enc->pix_proj_w, enc->pix_proj_b,
		1, 0, 1);
	if (!pix_proj) {
		sam3_log_error("mem_enc: pix proj failed");
		return SAM3_ENOMEM;
	}

	/* 4. Fuse: projected_pix + downsampled_mask */
	struct sam3_tensor *fused = gh_add(g, arena, pix_proj, mask_down);
	if (!fused) {
		sam3_log_error("mem_enc: fuse add failed");
		return SAM3_ENOMEM;
	}

	/* 5. Refine: 2 CXBlock layers */
	for (int i = 0; i < SAM3_MEMENC_FUSER_LAYERS; i++) {
		fused = build_cxblock(&enc->fuser[i], g, fused, d, arena);
		if (!fused) {
			sam3_log_error("mem_enc: cxblock %d failed", i);
			return SAM3_ENOMEM;
		}
	}

	/* 6. Output projection: 1x1 conv in_dim -> out_dim */
	struct sam3_tensor *feat = gh_conv2d(g, arena, fused,
		enc->out_proj_w, enc->out_proj_b,
		1, 0, 1);
	if (!feat) {
		sam3_log_error("mem_enc: out proj failed");
		return SAM3_ENOMEM;
	}

	sam3_log_debug("mem_enc: output feat [%d,%d,%d,%d]",
		       feat->dims[0], feat->dims[1],
		       feat->dims[2], feat->dims[3]);

	/* 7. Position encoding */
	struct sam3_tensor *pos = sam3_pos_encoding_get(&enc->pos_enc);
	if (!pos) {
		sam3_log_error("mem_enc: pos encoding not precomputed");
		return SAM3_EINVAL;
	}

	*out_feat = feat;
	*out_pos = pos;
	return SAM3_OK;
}
