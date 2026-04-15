/*
 * src/model/image_encoder_efficientvit.h - EfficientViT-B image encoder
 *
 * Defines the EfficientViT-B series backbone for EfficientSAM3. The
 * architecture uses MBConv blocks for early stages and LiteMLA linear
 * attention blocks for later stages, with HSwish activation and
 * BatchNorm throughout. Parameterized by width_list and depth_list
 * to support B0/B1/B2/B3 variants.
 *
 * Key types:  sam3_efficientvit, sam3_evit_conv_weights, sam3_evit_block
 * Depends on: core/tensor.h, core/graph.h, core/alloc.h, core/weight.h
 * Used by:    vl_combiner.c (via backbone dispatch)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_IMAGE_ENCODER_EFFICIENTVIT_H
#define SAM3_MODEL_IMAGE_ENCODER_EFFICIENTVIT_H

#include "core/tensor.h"
#include "core/graph.h"
#include "core/alloc.h"
#include "core/weight.h"
#include "backend/backend.h"

struct sam3_profiler;

#define SAM3_EVIT_MAX_STAGES 5
#define SAM3_EVIT_MAX_DEPTH  10	/* B3 has up to 9 blocks per stage */

/*
 * sam3_evit_conv_weights - Weights for one ConvLayer (conv + optional BN).
 *
 * For standard ConvLayer: conv_w + bn_w/bn_b/bn_mean/bn_var.
 * For fewer_norm variant: conv_w + conv_b, BN pointers NULL.
 * For no-norm variant (LiteMLA qkv): conv_w only.
 */
struct sam3_evit_conv_weights {
	struct sam3_tensor *conv_w;	/* [OC, KH, KW, IC/groups] OHWI */
	struct sam3_tensor *conv_b;	/* [OC] or NULL */
	struct sam3_tensor *bn_w;	/* [OC] gamma, or NULL */
	struct sam3_tensor *bn_b;	/* [OC] beta, or NULL */
	struct sam3_tensor *bn_mean;	/* [OC] running mean, or NULL */
	struct sam3_tensor *bn_var;	/* [OC] running var, or NULL */
};

/*
 * sam3_evit_litemla_weights - Weights for one LiteMLA context module.
 *
 * LiteMLA: qkv(1x1, no BN) -> aggreg_dw(5x5 DW) -> aggreg_pw(1x1 grouped)
 *          -> proj(1x1 + BN).
 */
struct sam3_evit_litemla_weights {
	struct sam3_evit_conv_weights qkv;		/* 1x1, no BN, no bias */
	struct sam3_evit_conv_weights aggreg_dw;	/* 5x5 depthwise */
	struct sam3_evit_conv_weights aggreg_pw;	/* 1x1 grouped */
	struct sam3_evit_conv_weights proj;		/* 1x1 + BN */
};

/*
 * sam3_evit_block - One block in an EfficientViT stage.
 *
 * For MBConv blocks (is_evit_block=0):
 *   inverted_conv (1x1) -> depth_conv (3x3 DW) -> point_conv (1x1)
 *
 * For EfficientViTBlock (is_evit_block=1):
 *   context: LiteMLA + residual skip
 *   local:   MBConv (fewer_norm) + residual skip
 */
struct sam3_evit_block {
	int is_evit_block;	/* 0 = MBConv, 1 = EfficientViTBlock */

	/* MBConv weights (used by all blocks; for EfficientViTBlock
	 * this is the local module's MBConv) */
	struct sam3_evit_conv_weights inverted_conv;	/* 1x1 expand */
	struct sam3_evit_conv_weights depth_conv;	/* 3x3 depthwise */
	struct sam3_evit_conv_weights point_conv;	/* 1x1 project */

	/* LiteMLA context module (only for is_evit_block=1) */
	struct sam3_evit_litemla_weights context;
};

/*
 * sam3_evit_stage - One stage of the EfficientViT backbone.
 *
 * Each stage has a downsample MBConv (stride 2) at index 0,
 * followed by n_blocks-1 regular blocks. Stages 3-4 use
 * EfficientViTBlocks with LiteMLA attention.
 */
struct sam3_evit_stage {
	int n_blocks;		/* total blocks including downsample */
	int width;		/* output channel width */
	int has_attention;	/* 1 for stages with EfficientViTBlocks */
	struct sam3_evit_block blocks[SAM3_EVIT_MAX_DEPTH];
};

/*
 * sam3_efficientvit - Complete EfficientViT-B backbone.
 *
 * The architecture consists of:
 * 1. Input stem: Conv2d(3->w0, s=2) + BN + HSwish, residual DSConv blocks
 * 2. Stages 1-4: MBConv downsample + MBConv/EfficientViTBlock blocks
 * 3. Projection: Conv1x1 -> BN -> Conv3x3
 */
struct sam3_efficientvit {
	/* Configuration */
	int width_list[5];	/* channel widths per stage [24,48,96,192,384] */
	int depth_list[5];	/* block counts per stage [1,3,4,4,6] */
	int attn_dim;		/* LiteMLA head dimension (32) */
	int expand_ratio;	/* MBConv expansion ratio (4) */
	int img_size;		/* input image size (512 or 1024) */
	int grid_size;		/* output spatial size = img_size / 32 */

	/* Input stem: Conv2d(3->w0, s=2) + BN + HSwish */
	struct sam3_evit_conv_weights stem_conv;

	/* Input stem residual DSConv blocks (depth_list[0] blocks) */
	int n_stem_blocks;
	struct {
		struct sam3_evit_conv_weights depth_conv;  /* 3x3 DW + BN + HSwish */
		struct sam3_evit_conv_weights point_conv;  /* 1x1 + BN */
	} stem_blocks[SAM3_EVIT_MAX_DEPTH];

	/* Stages 1-4 (indices 0-3, corresponding to width_list[1..4]) */
	struct sam3_evit_stage stages[4];

	/* Projection head: Conv1x1 -> BN -> Conv3x3 */
	struct sam3_evit_conv_weights proj_conv1;	/* 1x1 */
	struct sam3_evit_conv_weights proj_bn;		/* BN only (no conv) */
	struct sam3_evit_conv_weights proj_conv2;	/* 3x3 */
};

/*
 * sam3_efficientvit_init - Initialize EfficientViT with configuration.
 *
 * @evit:         EfficientViT struct (caller-allocated, zeroed)
 * @width_list:   Channel widths for 5 stages [24, 48, 96, 192, 384]
 * @depth_list:   Block counts for 5 stages [1, 3, 4, 4, 6]
 * @attn_dim:     LiteMLA head dimension (32)
 * @expand_ratio: MBConv expansion ratio (4)
 * @img_size:     Input image size (512 or 1024)
 *
 * Sets dimensions, computes grid_size (img_size/32), configures which
 * stages use attention blocks. Returns SAM3_OK on success.
 */
enum sam3_error sam3_efficientvit_init(struct sam3_efficientvit *evit,
				       const int *width_list,
				       const int *depth_list,
				       int attn_dim,
				       int expand_ratio,
				       int img_size);

/*
 * sam3_efficientvit_load - Load EfficientViT weights from weight file.
 *
 * @evit:  Initialized EfficientViT struct
 * @wf:    Open weight file (may be NULL for zero-init fallback)
 * @arena: Arena for weight tensor struct allocation (data stays in mmap)
 *
 * Loads all weights using gh_load_mmap (no copies). Weight names
 * follow the prefix "detector_model.vision_encoder.backbone.".
 *
 * Returns SAM3_OK on success, SAM3_ENOMEM if the arena is full.
 */
enum sam3_error sam3_efficientvit_load(struct sam3_efficientvit *evit,
					const struct sam3_weight_file *wf,
					struct sam3_arena *arena);

/*
 * sam3_efficientvit_build - Evaluate EfficientViT backbone.
 *
 * @evit:    Initialized and loaded EfficientViT
 * @be:      Backend for graph evaluation
 * @image:   Input image tensor [1, img_size, img_size, 3] (F32, NHWC)
 * @scratch: Arena for intermediate tensors
 * @persist: Arena for persistent output buffer
 * @profiler: Profiler for sub-stage timing (may be NULL)
 *
 * Returns output features [1, grid_size, grid_size, C] allocated from
 * persist arena, or NULL on error.
 */
struct sam3_tensor *sam3_efficientvit_build(struct sam3_efficientvit *evit,
					     struct sam3_backend *be,
					     struct sam3_tensor *image,
					     struct sam3_arena *scratch,
					     struct sam3_arena *persist,
					     struct sam3_profiler *profiler);

#endif /* SAM3_MODEL_IMAGE_ENCODER_EFFICIENTVIT_H */
