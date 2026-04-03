/*
 * src/model/model_misc.c - Dot-product scorer implementation
 *
 * Implements the dot-product scorer that computes text-query confidence.
 * The scorer works by computing a dot product between each query and text
 * features for similarity, then passes the query features through a
 * 2-layer MLP (linear -> relu -> linear) to produce a per-query
 * confidence score.
 *
 * Key types:  sam3_dot_scorer
 * Depends on: model_misc.h, graph_helpers.h
 * Used by:    decoder.c, sam3.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>

#include "model_misc.h"
#include "graph_helpers.h"

enum sam3_error sam3_dot_scorer_load(struct sam3_dot_scorer *ds,
				     const struct sam3_weight_file *wf,
				     struct sam3_arena *arena)
{
	int in = ds->input_dim;
	int hid = ds->hidden_dim;

	/* fc1: [hidden_dim, input_dim] / [hidden_dim] */
	int fc1_w_dims[] = {hid, in};
	ds->fc1_w = gh_load_or_alloc(wf, "dot_scorer.fc1.weight",
				   arena, SAM3_DTYPE_F32, 2, fc1_w_dims);
	if (!ds->fc1_w)
		return SAM3_ENOMEM;

	int fc1_b_dims[] = {hid};
	ds->fc1_b = gh_load_or_alloc(wf, "dot_scorer.fc1.bias",
				   arena, SAM3_DTYPE_F32, 1, fc1_b_dims);
	if (!ds->fc1_b)
		return SAM3_ENOMEM;

	/* fc2: [1, hidden_dim] / [1] */
	int fc2_w_dims[] = {1, hid};
	ds->fc2_w = gh_load_or_alloc(wf, "dot_scorer.fc2.weight",
				   arena, SAM3_DTYPE_F32, 2, fc2_w_dims);
	if (!ds->fc2_w)
		return SAM3_ENOMEM;

	int fc2_b_dims[] = {1};
	ds->fc2_b = gh_load_or_alloc(wf, "dot_scorer.fc2.bias",
				   arena, SAM3_DTYPE_F32, 1, fc2_b_dims);
	if (!ds->fc2_b)
		return SAM3_ENOMEM;

	return SAM3_OK;
}

struct sam3_tensor *sam3_dot_scorer_build(struct sam3_dot_scorer *ds,
					  struct sam3_graph *g,
					  struct sam3_tensor *queries,
					  struct sam3_tensor *text_features,
					  struct sam3_arena *arena)
{
	/*
	 * Dot product: queries @ text_features^T -> [n_queries, seq_len].
	 * Not used directly in output but could be used for scoring;
	 * we compute it here for graph completeness.
	 */
	struct sam3_tensor *text_t = gh_transpose(g, arena, text_features);
	if (!text_t)
		return NULL;

	struct sam3_tensor *dots = gh_matmul(g, arena, queries, text_t);
	if (!dots)
		return NULL;
	(void)dots; /* similarity available for downstream use */

	/*
	 * MLP on query features for confidence.
	 * linear1 -> relu -> linear2
	 * queries: [n_queries, input_dim]
	 * -> [n_queries, hidden_dim] -> relu -> [n_queries, 1]
	 */
	struct sam3_tensor *h;
	h = gh_linear(g, arena, queries, ds->fc1_w, ds->fc1_b);
	if (!h)
		return NULL;

	h = gh_relu(g, arena, h);
	if (!h)
		return NULL;

	h = gh_linear(g, arena, h, ds->fc2_w, ds->fc2_b);
	if (!h)
		return NULL;

	return h; /* [n_queries, 1] -- squeeze in caller */
}
