/*
 * src/model/segmentation.h - SAM3 segmentation head (pixel decoder + mask prediction)
 *
 * Defines the segmentation head that converts decoder output tokens and
 * image features into final segmentation masks. The pixel decoder applies
 * three stages of nearest-neighbor 2x upsampling with 1x1 conv, layernorm,
 * and ReLU to produce high-resolution pixel features. Mask logits are then
 * computed via dot product between query embeddings and upsampled features,
 * followed by sigmoid activation.
 *
 * Key types:  sam3_seg_head, sam3_pixel_decoder
 * Depends on: core/tensor.h, core/graph.h, core/alloc.h, core/weight.h
 * Used by:    sam3.c (top-level inference pipeline)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_SEGMENTATION_H
#define SAM3_MODEL_SEGMENTATION_H

#include "core/tensor.h"
#include "core/graph.h"
#include "core/alloc.h"
#include "core/weight.h"

#define SAM3_SEG_UPSAMPLE_STAGES 3

struct sam3_pixel_decoder {
	int hidden_dim;  /* 256 */

	/* 3 upsample stages: each has conv + layernorm */
	struct {
		struct sam3_tensor *conv_w, *conv_b;
		struct sam3_tensor *ln_w, *ln_b;
	} stages[SAM3_SEG_UPSAMPLE_STAGES];
};

struct sam3_seg_head {
	struct sam3_pixel_decoder pixel_dec;
	int d_model;  /* 256 */
};

/*
 * sam3_seg_head_init - Initialize segmentation head with configuration.
 *
 * @head:    Seg head struct (caller-allocated, zeroed)
 * @d_model: Model dimension (256)
 *
 * Returns SAM3_OK on success, SAM3_EINVAL if d_model <= 0.
 */
enum sam3_error sam3_seg_head_init(struct sam3_seg_head *head, int d_model);

/*
 * sam3_seg_head_load - Load segmentation head weights from weight file.
 *
 * @head:  Initialized seg head struct
 * @wf:    Open weight file (may be NULL for zero-init fallback)
 * @arena: Arena for weight tensor allocation
 *
 * Looks up weight tensors by name (seg_head.pixel_dec.stages.N.conv.*,
 * seg_head.pixel_dec.stages.N.ln.*) and populates the struct. When a
 * weight is not found (or wf is NULL), a zero-initialized tensor of
 * the correct shape is allocated as a fallback.
 *
 * Returns SAM3_OK on success, SAM3_ENOMEM if the arena is full.
 */
enum sam3_error sam3_seg_head_load(struct sam3_seg_head *head,
				   const struct sam3_weight_file *wf,
				   struct sam3_arena *arena);

/*
 * sam3_seg_head_build - Build segmentation head compute graph.
 *
 * @head:           Initialized and loaded seg head
 * @g:              Graph to add nodes to
 * @query_embed:    Decoder output [n_queries, d_model]
 * @pixel_features: Image features [n_pixels, d_model] (reshaped to 4D)
 * @grid_h:         Spatial height for reshape
 * @grid_w:         Spatial width for reshape
 * @arena:          Arena for intermediate tensors
 *
 * Pipeline:
 *  1. Reshape pixel_features to [1, d_model, grid_h, grid_w]
 *  2. 3 stages: upsample 2x -> 1x1 conv -> layernorm -> ReLU
 *  3. Flatten to [H*W, d_model]
 *  4. Dot product: query_embed @ pixel_features^T -> [n_queries, H*W]
 *  5. Sigmoid -> final mask probabilities
 *
 * Returns mask logits [n_queries, grid_h*8 * grid_w*8], or NULL on error.
 */
struct sam3_tensor *sam3_seg_head_build(
	struct sam3_seg_head *head,
	struct sam3_graph *g,
	struct sam3_tensor *query_embed,
	struct sam3_tensor *pixel_features,
	int grid_h, int grid_w,
	struct sam3_arena *arena);

#endif /* SAM3_MODEL_SEGMENTATION_H */
