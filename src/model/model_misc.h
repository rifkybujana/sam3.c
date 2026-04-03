/*
 * src/model/model_misc.h - Shared utility types and dot-product scoring
 *
 * Provides the dot-product scorer used for computing text-query
 * confidence scores. The scorer applies a dot product between query
 * embeddings and text features, then runs a 2-layer MLP on the
 * query features to produce per-query confidence scores.
 *
 * Key types:  sam3_dot_scorer
 * Depends on: core/graph.h, core/alloc.h, core/tensor.h, core/weight.h
 * Used by:    decoder.c, sam3.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_MISC_H
#define SAM3_MODEL_MISC_H

#include "core/graph.h"
#include "core/alloc.h"
#include "core/tensor.h"
#include "core/weight.h"

/* Dot-product scoring: text-query confidence via 2-layer MLP */
struct sam3_dot_scorer {
	struct sam3_tensor *fc1_w, *fc1_b; /* [hidden_dim, input_dim] / [hidden_dim] */
	struct sam3_tensor *fc2_w, *fc2_b; /* [1, hidden_dim] / [1] */
	int input_dim;  /* 256 */
	int hidden_dim; /* 2048 */
};

/*
 * sam3_dot_scorer_load - Load dot scorer weights from weight file.
 *
 * @ds:    Dot scorer struct (caller-allocated, input_dim/hidden_dim set)
 * @wf:    Open weight file (may be NULL for zero-init fallback)
 * @arena: Arena for weight tensor allocation
 *
 * Returns SAM3_OK on success, SAM3_ENOMEM if the arena is full.
 */
enum sam3_error sam3_dot_scorer_load(struct sam3_dot_scorer *ds,
				     const struct sam3_weight_file *wf,
				     struct sam3_arena *arena);

/*
 * sam3_dot_scorer_build - Build dot scorer compute graph.
 *
 * Score query embeddings against text features. Computes a dot product
 * between queries and text, then runs a 2-layer MLP (relu activation)
 * on the query features for confidence.
 *
 * @ds:            Loaded dot scorer
 * @g:             Graph to add nodes to
 * @queries:       [n_queries, d_model]
 * @text_features: [seq_len, d_model]
 * @arena:         Arena for intermediate tensors
 *
 * Returns confidence scores [n_queries, 1], or NULL on error.
 */
struct sam3_tensor *sam3_dot_scorer_build(struct sam3_dot_scorer *ds,
					  struct sam3_graph *g,
					  struct sam3_tensor *queries,
					  struct sam3_tensor *text_features,
					  struct sam3_arena *arena);

#endif /* SAM3_MODEL_MISC_H */
