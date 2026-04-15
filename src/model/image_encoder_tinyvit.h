/*
 * src/model/image_encoder_tinyvit.h - TinyViT image encoder
 *
 * Defines the TinyViT backbone for EfficientSAM3. The architecture uses
 * MBConv blocks in the first stage and TinyViTBlocks (window attention +
 * local conv + MLP) in later stages. At 1008x1008 input with grid_size=32,
 * it produces 128x128 masks (2x over EfficientViT-B1's 64x64).
 *
 * Key types:  sam3_tinyvit, sam3_tvit_block, sam3_tvit_layer
 * Depends on: core/tensor.h, core/graph.h, core/alloc.h, core/weight.h
 * Used by:    vl_combiner.c (via backbone dispatch)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_IMAGE_ENCODER_TINYVIT_H
#define SAM3_MODEL_IMAGE_ENCODER_TINYVIT_H

#include "core/tensor.h"
#include "core/graph.h"
#include "core/alloc.h"
#include "core/weight.h"
#include "backend/backend.h"

struct sam3_profiler;

#define SAM3_TVIT_MAX_LAYERS 4
#define SAM3_TVIT_MAX_DEPTH  8	/* TinyViT-L has max 6 blocks per layer */

/*
 * sam3_tvit_conv_weights - Weights for one Conv2d_BN (conv + BN).
 *
 * Same layout as sam3_evit_conv_weights: conv_w + optional bias +
 * optional BN (weight, bias, running_mean, running_var).
 */
struct sam3_tvit_conv_weights {
	struct sam3_tensor *conv_w;	/* [OC, KH, KW, IC/groups] OHWI */
	struct sam3_tensor *conv_b;	/* [OC] or NULL */
	struct sam3_tensor *bn_w;	/* [OC] gamma, or NULL */
	struct sam3_tensor *bn_b;	/* [OC] beta, or NULL */
	struct sam3_tensor *bn_mean;	/* [OC] running mean, or NULL */
	struct sam3_tensor *bn_var;	/* [OC] running var, or NULL */
};

/*
 * sam3_tvit_mbconv - Weights for one MBConv block (ConvLayer stage).
 *
 * Inverted bottleneck: conv1 (1x1 expand) + conv2 (3x3 DW) +
 * conv3 (1x1 project). Each has Conv2d + BN.
 */
struct sam3_tvit_mbconv {
	struct sam3_tvit_conv_weights conv1;	/* 1x1 expand */
	struct sam3_tvit_conv_weights conv2;	/* 3x3 depthwise */
	struct sam3_tvit_conv_weights conv3;	/* 1x1 project */
};

/*
 * sam3_tvit_patch_merging - Weights for PatchMerging downsample.
 *
 * Three convolutions: conv1 (1x1) + conv2 (3x3 DW stride 2) +
 * conv3 (1x1). Each has Conv2d + BN.
 */
struct sam3_tvit_patch_merging {
	struct sam3_tvit_conv_weights conv1;	/* 1x1 */
	struct sam3_tvit_conv_weights conv2;	/* 3x3 DW stride 2 */
	struct sam3_tvit_conv_weights conv3;	/* 1x1 */
};

/*
 * sam3_tvit_attention - Window attention weights for TinyViTBlock.
 *
 * QKV and proj linears with bias, post-attention LayerNorm, and
 * precomputed attention bias [n_heads, ws*ws, ws*ws].
 */
struct sam3_tvit_attention {
	struct sam3_tensor *qkv_w;	/* [3*C, C] */
	struct sam3_tensor *qkv_b;	/* [3*C] */
	struct sam3_tensor *proj_w;	/* [C, C] */
	struct sam3_tensor *proj_b;	/* [C] */
	struct sam3_tensor *norm_w;	/* [C] LN gamma */
	struct sam3_tensor *norm_b;	/* [C] LN beta */
	struct sam3_tensor *attn_bias;	/* [n_heads, ws*ws, ws*ws] expanded */
};

/*
 * sam3_tvit_block - One TinyViTBlock.
 *
 * For ConvLayer (layer 0): only mbconv is used.
 * For BasicLayer (layers 1-3): attention + local_conv + MLP.
 */
struct sam3_tvit_block {
	int is_conv_block;	/* 1 = MBConv only, 0 = TinyViTBlock */

	/* MBConv block (ConvLayer only) */
	struct sam3_tvit_mbconv mbconv;

	/* TinyViTBlock components (BasicLayer only) */
	struct sam3_tvit_attention attn;
	struct sam3_tvit_conv_weights local_conv;	/* 3x3 DW + BN */

	/* MLP: LN -> fc1 -> GELU -> fc2 */
	struct sam3_tensor *mlp_norm_w;		/* [C] pre-MLP LN gamma */
	struct sam3_tensor *mlp_norm_b;		/* [C] pre-MLP LN beta */
	struct sam3_tensor *mlp_fc1_w;		/* [mlp_dim, C] */
	struct sam3_tensor *mlp_fc1_b;		/* [mlp_dim] */
	struct sam3_tensor *mlp_fc2_w;		/* [C, mlp_dim] */
	struct sam3_tensor *mlp_fc2_b;		/* [C] */
};

/*
 * sam3_tvit_layer - One stage of the TinyViT backbone.
 *
 * Each layer has n_blocks blocks and an optional PatchMerging downsample.
 */
struct sam3_tvit_layer {
	int n_blocks;
	int embed_dim;		/* channel width for this layer */
	int n_heads;		/* attention heads (BasicLayer only) */
	int window_size;	/* window size (BasicLayer only) */
	int has_downsample;	/* 1 if PatchMerging follows */
	struct sam3_tvit_block blocks[SAM3_TVIT_MAX_DEPTH];
	struct sam3_tvit_patch_merging downsample;
};

/*
 * sam3_tinyvit - Complete TinyViT backbone.
 *
 * Architecture:
 * 1. Patch embed: two Conv2d_BN(stride 2) -> 252x252 @ embed_dims[0]
 * 2. Layer 0 (ConvLayer): MBConv blocks + PatchMerging
 * 3. Layers 1-3 (BasicLayer): TinyViTBlocks + optional PatchMerging
 * 4. Projection: Conv1x1 + BN + GELU + Conv3x3
 */
struct sam3_tinyvit {
	/* Configuration */
	int embed_dims[SAM3_TVIT_MAX_LAYERS];	/* [96,192,384,576] */
	int depths[SAM3_TVIT_MAX_LAYERS];	/* [2,2,6,2] */
	int num_heads[SAM3_TVIT_MAX_LAYERS];	/* [3,6,12,18] */
	int window_sizes[SAM3_TVIT_MAX_LAYERS];	/* [7,7,14,7] */
	int n_layers;				/* 4 */
	int img_size;				/* 1008 */
	int grid_size;				/* 32 */
	int embed_dim;				/* 1024 (projection output) */
	int mlp_ratio;				/* 4 */

	/* Patch embedding: two Conv2d_BN(stride 2) */
	struct sam3_tvit_conv_weights patch_embed_0;	/* Conv(3->48, k=3, s=2) */
	struct sam3_tvit_conv_weights patch_embed_1;	/* Conv(48->96, k=3, s=2) */

	/* Layers 0-3 */
	struct sam3_tvit_layer layers[SAM3_TVIT_MAX_LAYERS];

	/* Projection head: Conv1x1 + BN + GELU + Conv3x3 */
	struct sam3_tvit_conv_weights proj_conv1;	/* 1x1 */
	struct sam3_tvit_conv_weights proj_bn;		/* BN only */
	struct sam3_tvit_conv_weights proj_conv2;	/* 3x3 + bias */
};

/*
 * sam3_tinyvit_init - Initialize TinyViT with configuration.
 *
 * @tvit:         TinyViT struct (caller-allocated, zeroed)
 * @embed_dims:   Channel widths per layer [96, 192, 384, 576]
 * @depths:       Block counts per layer [2, 2, 6, 2]
 * @num_heads:    Attention heads per layer [3, 6, 12, 18]
 * @window_sizes: Window sizes per layer [7, 7, 14, 7]
 * @n_layers:     Number of layers (4)
 * @img_size:     Input image size (1008)
 * @embed_dim:    Projection output dimension (1024)
 * @mlp_ratio:    MLP expansion ratio (4)
 *
 * Computes grid_size by walking the stride chain.
 * Returns SAM3_OK on success.
 */
enum sam3_error sam3_tinyvit_init(struct sam3_tinyvit *tvit,
				  const int *embed_dims,
				  const int *depths,
				  const int *num_heads,
				  const int *window_sizes,
				  int n_layers,
				  int img_size,
				  int embed_dim,
				  int mlp_ratio);

/*
 * sam3_tinyvit_load - Load TinyViT weights from weight file.
 *
 * @tvit:  Initialized TinyViT struct
 * @wf:    Open weight file (may be NULL for zero-init fallback)
 * @arena: Arena for weight tensor struct allocation (data stays in mmap)
 *
 * Loads all weights using gh_load_mmap. Weight names follow the prefix
 * "detector_model.vision_encoder.backbone.".
 *
 * Returns SAM3_OK on success, SAM3_ENOMEM if the arena is full.
 */
enum sam3_error sam3_tinyvit_load(struct sam3_tinyvit *tvit,
				  const struct sam3_weight_file *wf,
				  struct sam3_arena *arena);

/*
 * sam3_tinyvit_build - Evaluate TinyViT backbone.
 *
 * @tvit:    Initialized and loaded TinyViT
 * @be:      Backend for graph evaluation
 * @image:   Input image tensor [1, img_size, img_size, 3] (F32, NHWC)
 * @scratch: Arena for intermediate tensors
 * @persist: Arena for persistent output buffer
 * @profiler: Profiler for sub-stage timing (may be NULL)
 *
 * Returns output features [grid_size^2, embed_dim] allocated from
 * persist arena, or NULL on error.
 */
struct sam3_tensor *sam3_tinyvit_build(struct sam3_tinyvit *tvit,
				       struct sam3_backend *be,
				       struct sam3_tensor *image,
				       struct sam3_arena *scratch,
				       struct sam3_arena *persist,
				       struct sam3_profiler *profiler);

#endif /* SAM3_MODEL_IMAGE_ENCODER_TINYVIT_H */
