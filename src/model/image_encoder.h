/*
 * src/model/image_encoder.h - SAM3 ViT image encoder backbone
 *
 * Defines the Vision Transformer (ViT) that processes the input image
 * into patch-level feature embeddings. The ViT uses patch embedding
 * (implemented as conv2d), 32 transformer blocks with multi-head
 * self-attention and GELU MLP, and RoPE for position encoding.
 * Global attention is used at layers 7, 15, 23, 31; the remaining
 * layers also use global attention in this initial implementation
 * (windowed attention is a future optimization).
 *
 * Key types:  sam3_vit
 * Depends on: core/tensor.h, core/graph.h, core/alloc.h, core/weight.h
 * Used by:    sam3.c (via sam3_set_image), tests/test_vit.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_IMAGE_ENCODER_H
#define SAM3_MODEL_IMAGE_ENCODER_H

#include "core/tensor.h"
#include "core/graph.h"
#include "core/alloc.h"
#include "core/weight.h"

#define SAM3_VIT_MAX_LAYERS      32
#define SAM3_VIT_N_GLOBAL_BLOCKS 4

struct sam3_vit {
	int img_size;		/* 1008 */
	int patch_size;		/* 14 */
	int embed_dim;		/* 1024 */
	int depth;		/* 32 */
	int n_heads;		/* 16 */
	int window_size;	/* 24 */
	int mlp_dim;		/* 4736 = embed_dim * 4.625 */
	int grid_size;		/* 72 = img_size / patch_size */
	int n_patches;		/* 5184 = grid_size^2 */

	/* Patch embedding (implemented as conv2d) */
	struct sam3_tensor *patch_embed_w; /* [embed_dim, 3, patch_size, patch_size] */
	struct sam3_tensor *patch_embed_b; /* [embed_dim] */

	/* RoPE precomputed frequencies */
	struct sam3_tensor *rope_cos; /* [n_patches, head_dim/2] */
	struct sam3_tensor *rope_sin; /* [n_patches, head_dim/2] */

	/* Per-layer weights */
	struct {
		struct sam3_tensor *ln1_w, *ln1_b;		/* [embed_dim] */
		struct sam3_tensor *qkv_w, *qkv_b;		/* [3*embed_dim, embed_dim] / [3*embed_dim] */
		struct sam3_tensor *proj_w, *proj_b;		/* [embed_dim, embed_dim] / [embed_dim] */
		struct sam3_tensor *ln2_w, *ln2_b;		/* [embed_dim] */
		struct sam3_tensor *mlp_fc1_w, *mlp_fc1_b;	/* [mlp_dim, embed_dim] / [mlp_dim] */
		struct sam3_tensor *mlp_fc2_w, *mlp_fc2_b;	/* [embed_dim, mlp_dim] / [embed_dim] */
		int is_global;	/* 1 for blocks 7, 15, 23, 31 */
	} layers[SAM3_VIT_MAX_LAYERS];
};

/*
 * sam3_vit_init - Initialize ViT with configuration.
 *
 * @vit:         ViT struct (caller-allocated, zeroed)
 * @img_size:    Input image size (1008)
 * @patch_size:  Patch size (14)
 * @embed_dim:   Embedding dimension (1024)
 * @depth:       Number of transformer layers (32)
 * @n_heads:     Number of attention heads (16)
 * @window_size: Window size for windowed attention (24)
 * @mlp_dim:     MLP hidden dimension (4736)
 * @arena:       Arena for RoPE frequency precomputation
 *
 * Sets dimensions, marks global attention blocks, precomputes RoPE.
 * Returns SAM3_OK on success, SAM3_ENOMEM if arena is full.
 */
enum sam3_error sam3_vit_init(struct sam3_vit *vit,
			       int img_size, int patch_size,
			       int embed_dim, int depth, int n_heads,
			       int window_size, int mlp_dim,
			       struct sam3_arena *arena);

/*
 * sam3_vit_load - Load ViT weights from weight file.
 *
 * @vit:   Initialized ViT struct
 * @wf:    Open weight file (may be NULL for zero-init fallback)
 * @arena: Arena for weight tensor allocation
 *
 * Looks up weight tensors by name and populates the struct. When a
 * weight is not found (or wf is NULL), a zero-initialized tensor of
 * the correct shape is allocated as a fallback.
 *
 * Returns SAM3_OK on success, SAM3_ENOMEM if the arena is full.
 */
enum sam3_error sam3_vit_load(struct sam3_vit *vit,
			       const struct sam3_weight_file *wf,
			       struct sam3_arena *arena);

/*
 * sam3_vit_build - Build ViT compute graph.
 *
 * @vit:   Initialized and loaded ViT
 * @g:     Graph to add nodes to
 * @image: Input image tensor [3, img_size, img_size] (F32, normalized)
 * @arena: Arena for intermediate tensors
 *
 * Returns output features [n_patches, embed_dim] or NULL on error.
 */
struct sam3_tensor *sam3_vit_build(struct sam3_vit *vit,
				    struct sam3_graph *g,
				    struct sam3_tensor *image,
				    struct sam3_arena *arena);

#endif /* SAM3_MODEL_IMAGE_ENCODER_H */
