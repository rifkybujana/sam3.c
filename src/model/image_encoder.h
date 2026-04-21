/*
 * src/model/image_encoder.h - SAM3 ViT image encoder backbone
 *
 * Defines the Vision Transformer (ViT) that processes the input image
 * into patch-level feature embeddings. The ViT uses patch embedding
 * (implemented as conv2d), 32 transformer blocks with multi-head
 * self-attention and GELU MLP, and RoPE for position encoding.
 * Global attention is used at layers 7, 15, 23, 31; the remaining
 * layers use mask-free windowed attention via window partitioning
 * with a small local RoPE table.
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
#include "backend/backend.h"

struct sam3_profiler;

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

	/*
	 * Patch embedding (implemented as conv2d). The .sam3 converter
	 * writes this weight in OHWI layout
	 * [embed_dim, patch_size, patch_size, 3], matching the NHWC
	 * conv dispatch path.
	 */
	struct sam3_tensor *patch_embed_w; /* [embed_dim, patch_size, patch_size, 3] */
	struct sam3_tensor *patch_embed_b; /* [embed_dim] */

	/* Absolute positional embedding (tiled from pretrain res) */
	struct sam3_tensor *pos_embed; /* [n_patches, embed_dim] after tiling */

	/* Pre-block layer norm (ln_pre) */
	struct sam3_tensor *ln_pre_w; /* [embed_dim] */
	struct sam3_tensor *ln_pre_b; /* [embed_dim] */

	/* Global RoPE precomputed frequencies [n_patches, head_dim/2] */
	struct sam3_tensor *rope_glo_cos; /* scaled global coords */
	struct sam3_tensor *rope_glo_sin; /* scaled global coords */

	/*
	 * Window-local RoPE: [ws*ws, head_dim/2] tiny tables used by
	 * mask-free windowed attention. Each window of patches gets the
	 * same local positions 0..ws-1, so a single small table suffices.
	 */
	struct sam3_tensor *rope_win_local_cos;
	struct sam3_tensor *rope_win_local_sin;

	/* Lazy precompute state */
	struct sam3_arena *model_arena;	/* stored during init for lazy precompute */
	int precomputed;		/* 1 after lazy precompute completes */
	struct sam3_tensor *raw_pos_embed; /* mmap'd [1, 577, embed_dim], tiled on first build */

	/*
	 * Note: Python ViT has ln_post=Identity (no params), so there is
	 * no final layer norm. The only ViT-level norm is ln_pre above.
	 */

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
 * @arena:       Model arena (stored for lazy precompute of RoPE tables)
 *
 * Sets dimensions and marks global attention blocks. RoPE tables
 * and position embed tiling are deferred to first sam3_vit_build()
 * call. Returns SAM3_OK on success.
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
 * sam3_vit_precompute - Eagerly compute RoPE tables and tiled pos_embed.
 *
 * @vit: Initialized and loaded ViT.
 *
 * Forces the lazy precomputation that normally runs on the first
 * sam3_vit_build() call. Call this during load so the precomputed
 * data lives in model_arena before set_image starts running; per-image
 * outputs go into per-slot cache arenas, so model_arena is stable after
 * load and the precomputed pointers remain valid across inferences.
 *
 * No-op if already precomputed. Returns SAM3_OK on success.
 */
enum sam3_error sam3_vit_precompute(struct sam3_vit *vit);

/*
 * sam3_vit_build - Evaluate ViT one transformer block at a time.
 *
 * Evaluates the ViT per-block, resetting the scratch arena between
 * blocks. Only the block output ([n_patches, embed_dim]) persists
 * between blocks in a buffer allocated from the persist arena.
 * This reduces peak memory from ~55 GB to ~2.5 GB.
 *
 * @vit:      Initialized and loaded ViT
 * @be:       Backend for per-block graph evaluation
 * @image:    Input image tensor [3, img_size, img_size] (F32, normalized)
 * @scratch:  Arena for per-block intermediate tensors (reset between blocks)
 * @persist:  Arena for the output buffer that survives across blocks
 * @profiler: Profiler for sub-stage timing (may be NULL)
 *
 * Returns output features [n_patches, embed_dim] allocated from
 * persist arena, or NULL on error.
 */
struct sam3_tensor *sam3_vit_build(struct sam3_vit *vit,
				    struct sam3_backend *be,
				    struct sam3_tensor *image,
				    struct sam3_arena *scratch,
				    struct sam3_arena *persist,
				    struct sam3_profiler *profiler);

#endif /* SAM3_MODEL_IMAGE_ENCODER_H */
