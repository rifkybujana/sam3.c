/*
 * src/model/memory_encoder.h - SimpleMaskEncoder for video tracking memory
 *
 * Encodes mask predictions and pixel features into compact memory tokens
 * for the memory bank. The pipeline applies sigmoid to mask logits, runs
 * a 4-layer strided-conv downsampler, projects pixel features, fuses
 * them with the downsampled mask, refines through 2 CXBlock layers,
 * and projects to the final memory dimension (64). Also provides 2D
 * sinusoidal position encoding for the spatial memory features.
 *
 * Weight prefix: tracker_model.maskmem_backbone.*
 *
 * Key types:  sam3_memory_encoder, sam3_cxblock
 * Depends on: core/tensor.h, core/graph.h, core/alloc.h, core/weight.h,
 *             model/position_encoding.h
 * Used by:    model/tracker.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_MEMORY_ENCODER_H
#define SAM3_MODEL_MEMORY_ENCODER_H

#include "core/tensor.h"
#include "core/graph.h"
#include "core/alloc.h"
#include "core/weight.h"
#include "position_encoding.h"

#define SAM3_MEMENC_DS_LAYERS    4   /* mask downsampler conv layers */
#define SAM3_MEMENC_FUSER_LAYERS 2   /* CXBlock layers in fuser */
#define SAM3_MEMENC_FUSER_EXPAND 4   /* MLP expansion ratio in CXBlock */

/* CXBlock: depthwise-separable convolution + MLP with layer scale */
struct sam3_cxblock {
	struct sam3_tensor *dwconv_w;  /* [dim, 7, 7, 1] depthwise OHWI */
	struct sam3_tensor *dwconv_b;  /* [dim] */
	struct sam3_tensor *norm_w;    /* [dim] */
	struct sam3_tensor *norm_b;    /* [dim] */
	struct sam3_tensor *pwconv1_w; /* [dim*4, dim] */
	struct sam3_tensor *pwconv1_b; /* [dim*4] */
	struct sam3_tensor *pwconv2_w; /* [dim, dim*4] */
	struct sam3_tensor *pwconv2_b; /* [dim] */
	struct sam3_tensor *gamma;     /* [dim] layer_scale */
};

struct sam3_memory_encoder {
	int in_dim;       /* 256: backbone feature dimension */
	int out_dim;      /* 64: output memory dimension */
	int interpol_h;   /* 1152: mask interpolation height */
	int interpol_w;   /* 1152: mask interpolation width */

	/* Mask downsampler: 4 conv layers with stride 2 (total stride 16) */
	struct {
		struct sam3_tensor *conv_w; /* OHWI */
		struct sam3_tensor *conv_b;
		struct sam3_tensor *ln_w;   /* LayerNorm weight */
		struct sam3_tensor *ln_b;   /* LayerNorm bias */
	} ds[SAM3_MEMENC_DS_LAYERS];

	/* Mask downsampler final 1x1 projection to embed_dim */
	struct sam3_tensor *ds_proj_w; /* [in_dim, 1, 1, ch_last] OHWI */
	struct sam3_tensor *ds_proj_b; /* [in_dim] */

	/* Pixel feature projection: 1x1 conv */
	struct sam3_tensor *pix_proj_w; /* [in_dim, 1, 1, in_dim] OHWI */
	struct sam3_tensor *pix_proj_b; /* [in_dim] */

	/* Fuser: 2 CXBlock layers */
	struct sam3_cxblock fuser[SAM3_MEMENC_FUSER_LAYERS];

	/* Output projection: 1x1 conv in_dim -> out_dim */
	struct sam3_tensor *out_proj_w; /* [out_dim, 1, 1, in_dim] OHWI */
	struct sam3_tensor *out_proj_b; /* [out_dim] */

	/* 2D sinusoidal position encoding */
	struct sam3_pos_encoding pos_enc;
};

/*
 * sam3_memory_encoder_init - Initialize memory encoder with dimensions.
 *
 * @enc:     Encoder struct (caller-allocated, zeroed and configured)
 * @in_dim:  Input/internal feature dimension (256)
 * @out_dim: Output memory dimension (64)
 *
 * Returns SAM3_OK on success, SAM3_EINVAL for bad arguments.
 */
enum sam3_error sam3_memory_encoder_init(struct sam3_memory_encoder *enc,
					 int in_dim, int out_dim);

/*
 * sam3_memory_encoder_load - Load weights from a weight file.
 *
 * @enc:   Initialized encoder
 * @wf:    Weight file (NULL for zero-init testing)
 * @arena: Arena for tensor allocation
 *
 * Weight prefix: tracker_model.maskmem_backbone.*
 * Falls back to zero-initialized tensors when wf is NULL.
 * Also precomputes the position encoding for the expected spatial
 * resolution (interpol_h/16 x interpol_w/16 = 72x72).
 *
 * Returns SAM3_OK on success, SAM3_ENOMEM if arena is full.
 */
enum sam3_error sam3_memory_encoder_load(struct sam3_memory_encoder *enc,
					 const struct sam3_weight_file *wf,
					 struct sam3_arena *arena);

/*
 * sam3_memory_encoder_build - Build memory encoder compute graph.
 *
 * @enc:      Loaded encoder
 * @g:        Graph to add nodes to
 * @pix_feat: Pixel features [1, H, W, in_dim] (NHWC)
 * @masks:    Mask input [1, Hm, Wm, 1] (NHWC). Already preprocessed by
 *            the caller: either (mask > 0).float() for cond frames with
 *            user prompts or sigmoid(mask) for propagation frames, then
 *            scaled by sigmoid_scale_for_mem_enc and offset by
 *            sigmoid_bias_for_mem_enc. The encoder applies no further
 *            sigmoid (matching Python skip_mask_sigmoid=True semantics).
 * @arena:    Arena for intermediate tensors
 * @out_feat: Output: encoded memory features [1, H, W, out_dim]
 * @out_pos:  Output: position encoding [H, W, out_dim*2]
 *
 * The mask is expected to already be at the downsampler input
 * resolution. Bilinear interpolation to interpol_size should be
 * handled by the caller before invoking this function.
 *
 * Pipeline:
 *  1. Mask downsampler: 4x (conv3x3/s2 + LayerNorm + GELU) + 1x1 proj
 *  2. Pixel feature projection: 1x1 conv
 *  3. Fuse: projected_pix + downsampled_mask
 *  4. Refine: 2x CXBlock (dwconv7x7 + LN + MLP + layer_scale + residual)
 *  5. Output projection: 1x1 conv (in_dim -> out_dim)
 *  6. Position encoding: precomputed 2D sinusoidal
 *
 * Returns SAM3_OK on success, SAM3_EINVAL for bad arguments,
 * SAM3_ENOMEM if graph or arena allocation fails.
 */
enum sam3_error sam3_memory_encoder_build(struct sam3_memory_encoder *enc,
					  struct sam3_graph *g,
					  struct sam3_tensor *pix_feat,
					  struct sam3_tensor *masks,
					  struct sam3_arena *arena,
					  struct sam3_tensor **out_feat,
					  struct sam3_tensor **out_pos);

#endif /* SAM3_MODEL_MEMORY_ENCODER_H */
