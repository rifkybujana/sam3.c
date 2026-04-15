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

#define EVIT_PREFIX "detector_model.vision_encoder.backbone."

/* ── Initialization ──────────────────────────────────────────────── */

enum sam3_error sam3_efficientvit_init(struct sam3_efficientvit *evit,
				       const int *width_list,
				       const int *depth_list,
				       int attn_dim,
				       int expand_ratio,
				       int img_size)
{
	memset(evit, 0, sizeof(*evit));

	/* Validate bounds */
	for (int i = 0; i < 5; i++) {
		if (depth_list[i] > SAM3_EVIT_MAX_DEPTH) {
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

/* ── Weight loading helpers ──────────────────────────────────────── */

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
	 * Both have no BN and no bias.
	 * DW groups = total_kv_channels, PW groups = n_heads.
	 */
	int kv_ch = 2 * n_heads * attn_dim;

	snprintf(sub, sizeof(sub), "%scontext.aggreg.0.0.", prefix);
	err = load_conv_weights(&ctx->aggreg_dw, wf, arena, sub,
				  kv_ch, 5, 5, 1,
				  0, 0);
	if (err != SAM3_OK)
		return err;

	snprintf(sub, sizeof(sub), "%scontext.aggreg.0.1.", prefix);
	err = load_conv_weights(&ctx->aggreg_pw, wf, arena, sub,
				  kv_ch, 1, 1, attn_dim,
				  0, 0);
	if (err != SAM3_OK)
		return err;

	/* proj: 1x1 conv + BN, no bias */
	snprintf(sub, sizeof(sub), "%scontext.proj.", prefix);
	err = load_conv_weights(&ctx->proj, wf, arena, sub,
				  channels, 1, 1, channels,
				  0, 1);
	if (err != SAM3_OK)
		return err;

	return SAM3_OK;
}

/* ── Main weight loading ─────────────────────────────────────────── */

enum sam3_error sam3_efficientvit_load(struct sam3_efficientvit *evit,
					const struct sam3_weight_file *wf,
					struct sam3_arena *arena)
{
	char prefix[256];
	enum sam3_error err;
	int w0 = evit->width_list[0];

	/* ── Input stem ──────────────────────────────────────────── */

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

	/* ── Stages 1-4 ──────────────────────────────────────────── */

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

	/* ── Projection head ─────────────────────────────────────── */

	/*
	 * Projection: conv1(1x1) -> BN -> conv2(3x3, pad=1).
	 * The BN here uses a different naming convention:
	 *   projection.bn.weight (not projection.bn.norm.weight)
	 */
	int final_ch = evit->width_list[4];

	/* proj_conv1: 1x1, no BN, no bias */
	{
		char name[256];
		int w_dims[] = {final_ch, 1, 1, final_ch};
		snprintf(name, sizeof(name),
			 EVIT_PREFIX "projection.conv1.weight");
		evit->proj_conv1.conv_w = gh_load_mmap(wf, name, arena,
							 SAM3_DTYPE_F32,
							 4, w_dims);
		if (!evit->proj_conv1.conv_w)
			return SAM3_ENOMEM;
	}

	/* proj_bn: BN with projection.bn.{weight,bias,running_mean,running_var} */
	{
		char name[256];
		int bn_dims[] = {final_ch};

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

	/* proj_conv2: 3x3, no BN, no bias */
	{
		char name[256];
		int w_dims[] = {final_ch, 3, 3, final_ch};
		snprintf(name, sizeof(name),
			 EVIT_PREFIX "projection.conv2.weight");
		evit->proj_conv2.conv_w = gh_load_mmap(wf, name, arena,
							 SAM3_DTYPE_F32,
							 4, w_dims);
		if (!evit->proj_conv2.conv_w)
			return SAM3_ENOMEM;
	}

	sam3_log_info("evit_load: loaded weights for %d stem blocks, "
		      "4 stages, projection",
		      evit->n_stem_blocks);

	return SAM3_OK;
}

/* ── Graph construction (stub) ───────────────────────────────────── */

struct sam3_tensor *sam3_efficientvit_build(struct sam3_efficientvit *evit,
					     struct sam3_backend *be,
					     struct sam3_tensor *image,
					     struct sam3_arena *scratch,
					     struct sam3_arena *persist,
					     struct sam3_profiler *profiler)
{
	(void)be;
	(void)image;
	(void)scratch;
	(void)persist;
	(void)profiler;

	sam3_log_error("evit_build: not yet implemented "
		       "(grid=%d, final_ch=%d)",
		       evit->grid_size, evit->width_list[4]);
	return NULL;
}
