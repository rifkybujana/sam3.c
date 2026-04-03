/*
 * src/model/prompt_encoder.h - SAM3 geometry encoder
 *
 * Encodes geometric prompts (points, boxes) into dense embeddings via
 * cross-attention to image features. Prepends a learnable CLS token,
 * runs 3 cross-attention layers (prompt Q attends to image KV), then
 * applies a post-projection. Output is [N+1, d_model].
 *
 * Key types:  sam3_geometry_encoder
 * Depends on: core/tensor.h, core/graph.h, core/alloc.h, core/weight.h
 * Used by:    sam3.c (top-level segmentation pipeline)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_PROMPT_ENCODER_H
#define SAM3_MODEL_PROMPT_ENCODER_H

#include "core/tensor.h"
#include "core/graph.h"
#include "core/alloc.h"
#include "core/weight.h"

#define SAM3_GEOM_ENC_MAX_LAYERS 3

struct sam3_geometry_encoder {
	int d_model;   /* 256 */
	int n_layers;  /* 3 */

	/* Point/box projection */
	struct sam3_tensor *point_proj_w, *point_proj_b;  /* [d_model, 2] */
	struct sam3_tensor *box_proj_w, *box_proj_b;      /* [d_model, 4] */
	struct sam3_tensor *cls_token;                    /* [1, d_model] */

	struct {
		struct sam3_tensor *ca_q_w, *ca_q_b;
		struct sam3_tensor *ca_kv_w, *ca_kv_b;
		struct sam3_tensor *ca_out_w, *ca_out_b;
		struct sam3_tensor *ca_ln_w, *ca_ln_b;
	} layers[SAM3_GEOM_ENC_MAX_LAYERS];

	struct sam3_tensor *post_proj_w, *post_proj_b;  /* post-encode */
};

/*
 * sam3_geometry_encoder_init - Set config fields for geometry encoder.
 *
 * @enc:      Encoder struct (caller-allocated)
 * @d_model:  Model dimension (256)
 * @n_layers: Number of cross-attention layers (3)
 *
 * Returns SAM3_OK on success, SAM3_EINVAL if n_layers exceeds max.
 */
enum sam3_error sam3_geometry_encoder_init(
	struct sam3_geometry_encoder *enc,
	int d_model, int n_layers);

/*
 * sam3_geometry_encoder_load - Load/allocate encoder weights.
 *
 * @enc:   Initialized encoder struct
 * @wf:    Open weight file (may be NULL for zero-init fallback)
 * @arena: Arena for weight tensor allocation
 *
 * Returns SAM3_OK on success, SAM3_ENOMEM if the arena is full.
 */
enum sam3_error sam3_geometry_encoder_load(
	struct sam3_geometry_encoder *enc,
	const struct sam3_weight_file *wf,
	struct sam3_arena *arena);

/*
 * sam3_geometry_encoder_build - Build geometry encoder graph.
 *
 * Prepends CLS token to prompt tokens, runs cross-attention layers
 * where prompt tokens (Q) attend to image features (KV), then
 * applies a post-projection.
 *
 * @enc:            Initialized and loaded encoder
 * @g:              Graph to add nodes to
 * @prompt_tokens:  [N, d_model] pre-projected prompt embeddings
 * @image_features: [M, d_model] image feature tokens
 * @arena:          Arena for intermediate tensors
 *
 * Returns output tensor [N+1, d_model], or NULL on error.
 */
struct sam3_tensor *sam3_geometry_encoder_build(
	struct sam3_geometry_encoder *enc,
	struct sam3_graph *g,
	struct sam3_tensor *prompt_tokens,
	struct sam3_tensor *image_features,
	struct sam3_arena *arena);

#endif /* SAM3_MODEL_PROMPT_ENCODER_H */
