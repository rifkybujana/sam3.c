/*
 * src/model/decoder.h - Transformer decoder (6-layer, 200 queries)
 *
 * Defines the transformer decoder that takes encoder output features and
 * text features, processes them through learned query embeddings with
 * self-attention, encoder cross-attention, text cross-attention, FFN,
 * and iterative box refinement. Each layer refines bounding box
 * predictions via a 3-layer MLP applied to the query embeddings.
 *
 * Key types:  sam3_decoder
 * Depends on: core/tensor.h, core/graph.h, core/alloc.h, core/weight.h
 * Used by:    sam3.c (top-level API)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_DECODER_H
#define SAM3_MODEL_DECODER_H

#include "core/tensor.h"
#include "core/graph.h"
#include "core/alloc.h"
#include "core/weight.h"

#define SAM3_DEC_MAX_LAYERS  6
#define SAM3_DEC_NUM_QUERIES 200

struct sam3_decoder {
	int d_model;	/* 256 */
	int n_heads;	/* 8 */
	int n_layers;	/* 6 */
	int d_ffn;	/* 2048 */
	int n_queries;	/* 200 */

	struct sam3_tensor *query_embed; /* [n_queries, d_model] */

	struct {
		/* Self-attention */
		struct sam3_tensor *sa_qkv_w, *sa_qkv_b;
		struct sam3_tensor *sa_out_w, *sa_out_b;
		struct sam3_tensor *sa_ln_w, *sa_ln_b;

		/* Cross-attention to encoder output */
		struct sam3_tensor *ca_q_w, *ca_q_b;
		struct sam3_tensor *ca_kv_w, *ca_kv_b;
		struct sam3_tensor *ca_out_w, *ca_out_b;
		struct sam3_tensor *ca_ln_w, *ca_ln_b;

		/* Text cross-attention */
		struct sam3_tensor *tca_q_w, *tca_q_b;
		struct sam3_tensor *tca_kv_w, *tca_kv_b;
		struct sam3_tensor *tca_out_w, *tca_out_b;
		struct sam3_tensor *tca_ln_w, *tca_ln_b;

		/* FFN */
		struct sam3_tensor *ffn_fc1_w, *ffn_fc1_b;
		struct sam3_tensor *ffn_fc2_w, *ffn_fc2_b;
		struct sam3_tensor *ffn_ln_w, *ffn_ln_b;

		/* Box refinement: 3-layer MLP */
		struct sam3_tensor *box_fc1_w, *box_fc1_b;
		struct sam3_tensor *box_fc2_w, *box_fc2_b;
		struct sam3_tensor *box_fc3_w, *box_fc3_b;
	} layers[SAM3_DEC_MAX_LAYERS];
};

/*
 * sam3_decoder_init - Initialize decoder with configuration.
 *
 * @dec:       Decoder struct (caller-allocated)
 * @d_model:   Model dimension (256)
 * @n_heads:   Number of attention heads (8)
 * @n_layers:  Number of transformer layers (6)
 * @d_ffn:     FFN hidden dimension (2048)
 * @n_queries: Number of learned object queries (200)
 *
 * Returns SAM3_OK on success, SAM3_EINVAL if n_layers exceeds max.
 */
enum sam3_error sam3_decoder_init(struct sam3_decoder *dec,
				  int d_model, int n_heads,
				  int n_layers, int d_ffn, int n_queries);

/*
 * sam3_decoder_load - Load decoder weights from weight file.
 *
 * @dec:   Initialized decoder struct
 * @wf:    Open weight file (may be NULL for zero-init fallback)
 * @arena: Arena for weight tensor allocation
 *
 * Returns SAM3_OK on success, SAM3_ENOMEM if the arena is full.
 */
enum sam3_error sam3_decoder_load(struct sam3_decoder *dec,
				  const struct sam3_weight_file *wf,
				  struct sam3_arena *arena);

/*
 * sam3_decoder_build - Build decoder graph.
 *
 * Runs N transformer decoder layers with self-attention on query
 * embeddings, cross-attention to encoder features, text cross-attention,
 * FFN, and iterative box refinement.
 *
 * @dec:           Initialized and loaded decoder
 * @g:             Graph to add nodes to
 * @enc_features:  Encoder output [n_pixels, d_model]
 * @text_features: Text embeddings [seq_len, d_model]
 * @box_out:       If non-NULL, receives predicted boxes [n_queries, 4]
 * @arena:         Arena for intermediate tensors
 *
 * Returns query embeddings [n_queries, d_model], or NULL on error.
 */
struct sam3_tensor *sam3_decoder_build(
	struct sam3_decoder *dec,
	struct sam3_graph *g,
	struct sam3_tensor *enc_features,
	struct sam3_tensor *text_features,
	struct sam3_tensor **box_out,
	struct sam3_arena *arena);

#endif /* SAM3_MODEL_DECODER_H */
