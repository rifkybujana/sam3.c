/*
 * src/model/encoder.h - Transformer encoder fusion (6-layer)
 *
 * Defines the DETR encoder (detector_model.detr_encoder) that fuses image
 * features with text features through alternating self-attention and
 * cross-attention layers. Each of the 6 layers applies: (1) pre-norm self-attention on image
 * features with residual, (2) pre-norm cross-attention where image
 * features attend to text features with residual, (3) pre-norm FFN
 * with residual.
 *
 * Key types:  sam3_encoder_fusion
 * Depends on: core/tensor.h, core/graph.h, core/alloc.h, core/weight.h
 * Used by:    decoder.c, sam3.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_ENCODER_H
#define SAM3_MODEL_ENCODER_H

#include "core/tensor.h"
#include "core/graph.h"
#include "core/alloc.h"
#include "core/weight.h"

#define SAM3_ENC_FUSION_MAX_LAYERS 6

struct sam3_encoder_fusion {
	int d_model;	/* 256 */
	int n_heads;	/* 8 */
	int n_layers;	/* 6 */
	int d_ffn;	/* 2048 */

	struct {
		/* Self-attention on image features */
		struct sam3_tensor *sa_qkv_w, *sa_qkv_b;
		struct sam3_tensor *sa_out_w, *sa_out_b;
		struct sam3_tensor *sa_ln_w,  *sa_ln_b;

		/* Cross-attention: image queries, text keys/values */
		struct sam3_tensor *ca_q_w,   *ca_q_b;
		struct sam3_tensor *ca_kv_w,  *ca_kv_b;
		struct sam3_tensor *ca_out_w, *ca_out_b;
		struct sam3_tensor *ca_ln_w,  *ca_ln_b;

		/* FFN */
		struct sam3_tensor *ffn_fc1_w, *ffn_fc1_b;
		struct sam3_tensor *ffn_fc2_w, *ffn_fc2_b;
		struct sam3_tensor *ffn_ln_w,  *ffn_ln_b;
	} layers[SAM3_ENC_FUSION_MAX_LAYERS];

	/* Final layer norm applied after all layers */
	struct sam3_tensor *final_ln_w, *final_ln_b;
};

/*
 * sam3_encoder_fusion_init - Initialize encoder fusion with config.
 *
 * @enc:      Encoder struct (caller-allocated)
 * @d_model:  Model dimension (256)
 * @n_heads:  Number of attention heads (8)
 * @n_layers: Number of transformer layers (6)
 * @d_ffn:    FFN hidden dimension (2048)
 *
 * Returns SAM3_OK on success, SAM3_EINVAL if n_layers exceeds max.
 */
enum sam3_error sam3_encoder_fusion_init(struct sam3_encoder_fusion *enc,
					 int d_model, int n_heads,
					 int n_layers, int d_ffn);

/*
 * sam3_encoder_fusion_load - Load encoder weights from weight file.
 *
 * @enc:   Initialized encoder struct
 * @wf:    Open weight file (may be NULL for zero-init fallback)
 * @arena: Arena for weight tensor allocation
 *
 * Returns SAM3_OK on success, SAM3_ENOMEM if the arena is full.
 */
enum sam3_error sam3_encoder_fusion_load(struct sam3_encoder_fusion *enc,
					 const struct sam3_weight_file *wf,
					 struct sam3_arena *arena);

/*
 * sam3_encoder_fusion_build - Build encoder fusion graph.
 *
 * Fuses image features with text features through self-attention
 * and cross-attention. Each layer:
 *   1. Self-attention on image features (pre-norm + residual)
 *   2. Cross-attention: image attends to text (pre-norm + residual)
 *   3. FFN with relu activation (pre-norm + residual)
 *
 * @enc:            Initialized and loaded encoder
 * @g:              Graph to add nodes to
 * @image_features: [n_pixels, d_model]
 * @text_features:  [seq_len, d_model]
 * @arena:          Arena for intermediate tensors
 *
 * Returns fused features [n_pixels, d_model], or NULL on error.
 */
struct sam3_tensor *sam3_encoder_fusion_build(
	struct sam3_encoder_fusion *enc,
	struct sam3_graph *g,
	struct sam3_tensor *image_features,
	struct sam3_tensor *text_features,
	struct sam3_arena *arena);

/*
 * sam3_encoder_fusion_build_layer - Build a single encoder layer.
 *
 * Used for per-layer evaluation to avoid MLX shared-buffer issues.
 */
struct sam3_tensor *sam3_encoder_fusion_build_layer(
	struct sam3_encoder_fusion *enc,
	int layer_idx,
	struct sam3_graph *g,
	struct sam3_tensor *x,
	struct sam3_tensor *text_features,
	struct sam3_arena *arena);

/*
 * sam3_encoder_fusion_build_final - Apply final layer norm.
 */
struct sam3_tensor *sam3_encoder_fusion_build_final(
	struct sam3_encoder_fusion *enc,
	struct sam3_graph *g,
	struct sam3_tensor *x,
	struct sam3_arena *arena);

#endif /* SAM3_MODEL_ENCODER_H */
