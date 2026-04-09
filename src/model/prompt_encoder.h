/*
 * src/model/prompt_encoder.h - SAM3 geometry encoder
 *
 * Encodes geometric prompts (points, boxes) into dense embeddings via
 * 3-layer transformer encoder with self-attention, cross-attention to
 * image features, and FFN. Applies pre-encoder projection + LayerNorm,
 * runs 3 encoder layers, then post-encoder LayerNorm. Output is [N+1, d_model].
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

	/* Pool projection: grid_sample from img features + Linear(d, d) */
	struct sam3_tensor *pool_proj_w, *pool_proj_b;  /* [d_model, d_model] */

	/* Pos enc projection: sinusoidal pos encoding + Linear(d, d) */
	struct sam3_tensor *posenc_proj_w, *posenc_proj_b;  /* [d_model, d_model] */

	/* Image pre-norm for pool projection (LayerNorm) */
	struct sam3_tensor *img_pre_norm_w, *img_pre_norm_b;  /* [d_model] */

	/* Label embedding: type_embed added to projected prompts */
	struct sam3_tensor *label_embed;  /* [n_labels, d_model] */
	int n_labels;                    /* 2 (pos/neg point labels) */

	int n_heads;   /* 8 */

	struct {
		/* Self-attention (norm1 → self_attn → residual) */
		struct sam3_tensor *norm1_w, *norm1_b;	  /* [d_model] */
		struct sam3_tensor *sa_qkv_w, *sa_qkv_b; /* [3*d, d], [3*d] */
		struct sam3_tensor *sa_out_w, *sa_out_b;  /* [d, d], [d] */

		/* Cross-attention (norm2/ca_ln → cross_attn → residual) */
		struct sam3_tensor *ca_q_w, *ca_q_b;
		struct sam3_tensor *ca_kv_w, *ca_kv_b;
		struct sam3_tensor *ca_out_w, *ca_out_b;
		struct sam3_tensor *ca_ln_w, *ca_ln_b;

		/* FFN (norm3 → linear1 → relu → linear2 → residual) */
		struct sam3_tensor *norm3_w, *norm3_b;	    /* [d_model] */
		struct sam3_tensor *ffn_fc1_w, *ffn_fc1_b;  /* [2048, d], [2048] */
		struct sam3_tensor *ffn_fc2_w, *ffn_fc2_b;  /* [d, 2048], [d] */
	} layers[SAM3_GEOM_ENC_MAX_LAYERS];

	/*
	 * Pre-encoder projection: final_proj (Linear) + norm (LayerNorm).
	 * Applied BEFORE encoder layers in Python, despite the C weight
	 * name "post_proj" (renamed from Python's "final_proj").
	 */
	struct sam3_tensor *post_proj_w, *post_proj_b;  /* Linear [d, d] */
	struct sam3_tensor *norm_w, *norm_b;            /* LayerNorm [d] */

	/* Post-encoder LayerNorm (encode_norm in Python) */
	struct sam3_tensor *encode_norm_w, *encode_norm_b;  /* [d_model] */
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
 * Prepends CLS token, applies pre-encoder projection + LayerNorm,
 * runs 3-layer transformer encoder (self-attn + cross-attn + FFN),
 * then post-encoder LayerNorm. Position encoding is added to
 * cross-attention keys only (not values).
 *
 * @enc:            Initialized and loaded encoder
 * @g:              Graph to add nodes to
 * @prompt_tokens:  [N, d_model] pre-projected prompt embeddings
 * @image_features: [M, d_model] image feature tokens
 * @image_pos:      [M, d_model] image position encoding (added to keys)
 * @arena:          Arena for intermediate tensors
 *
 * Returns output tensor [N+1, d_model], or NULL on error.
 */
struct sam3_tensor *sam3_geometry_encoder_build(
	struct sam3_geometry_encoder *enc,
	struct sam3_graph *g,
	struct sam3_tensor *prompt_tokens,
	struct sam3_tensor *image_features,
	struct sam3_tensor *image_pos,
	struct sam3_arena *arena);

#endif /* SAM3_MODEL_PROMPT_ENCODER_H */
