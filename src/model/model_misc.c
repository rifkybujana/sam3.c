/*
 * src/model/model_misc.c - Objectness scorer implementation
 *
 * Implements the objectness scorer matching SAM3's pred_obj_score_head.
 * The scorer computes a dot product between each query and text features
 * for similarity, then passes query features through a 3-layer MLP
 * (proj_in -> relu -> hidden -> relu -> proj_out) to produce a
 * per-query confidence score.
 *
 * Key types:  sam3_dot_scorer
 * Depends on: model_misc.h, graph_helpers.h
 * Used by:    sam3_image.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>

#include "model_misc.h"
#include "graph_helpers.h"

#define SCORE_PREFIX "tracker_model.mask_decoder.pred_obj_score_head."

enum sam3_error sam3_dot_scorer_load(struct sam3_dot_scorer *ds,
				     const struct sam3_weight_file *wf,
				     struct sam3_arena *arena)
{
	int in = ds->input_dim;
	int hid = ds->hidden_dim;

	/* proj_in: [hidden, input] / [hidden] */
	int pi_w_dims[] = {hid, in};
	ds->proj_in_w = gh_load_or_alloc(wf,
		SCORE_PREFIX "proj_in.weight",
		arena, SAM3_DTYPE_F32, 2, pi_w_dims);
	if (!ds->proj_in_w)
		return SAM3_ENOMEM;

	int pi_b_dims[] = {hid};
	ds->proj_in_b = gh_load_or_alloc(wf,
		SCORE_PREFIX "proj_in.bias",
		arena, SAM3_DTYPE_F32, 1, pi_b_dims);
	if (!ds->proj_in_b)
		return SAM3_ENOMEM;

	/* hidden (layers.0): [hidden, hidden] / [hidden] */
	int h_w_dims[] = {hid, hid};
	ds->hidden_w = gh_load_or_alloc(wf,
		SCORE_PREFIX "layers.0.weight",
		arena, SAM3_DTYPE_F32, 2, h_w_dims);
	if (!ds->hidden_w)
		return SAM3_ENOMEM;

	int h_b_dims[] = {hid};
	ds->hidden_b = gh_load_or_alloc(wf,
		SCORE_PREFIX "layers.0.bias",
		arena, SAM3_DTYPE_F32, 1, h_b_dims);
	if (!ds->hidden_b)
		return SAM3_ENOMEM;

	/* proj_out: [1, hidden] / [1] */
	int po_w_dims[] = {1, hid};
	ds->proj_out_w = gh_load_or_alloc(wf,
		SCORE_PREFIX "proj_out.weight",
		arena, SAM3_DTYPE_F32, 2, po_w_dims);
	if (!ds->proj_out_w)
		return SAM3_ENOMEM;

	int po_b_dims[] = {1};
	ds->proj_out_b = gh_load_or_alloc(wf,
		SCORE_PREFIX "proj_out.bias",
		arena, SAM3_DTYPE_F32, 1, po_b_dims);
	if (!ds->proj_out_b)
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
	 * 3-layer MLP on query features for objectness confidence.
	 * proj_in -> relu -> hidden -> relu -> proj_out
	 * queries: [n_queries, 256] -> [n_queries, 256] -> [n_queries, 1]
	 */
	struct sam3_tensor *h;
	h = gh_linear(g, arena, queries, ds->proj_in_w, ds->proj_in_b);
	if (!h)
		return NULL;

	h = gh_relu(g, arena, h);
	if (!h)
		return NULL;

	h = gh_linear(g, arena, h, ds->hidden_w, ds->hidden_b);
	if (!h)
		return NULL;

	h = gh_relu(g, arena, h);
	if (!h)
		return NULL;

	h = gh_linear(g, arena, h, ds->proj_out_w, ds->proj_out_b);
	if (!h)
		return NULL;

	return h; /* [n_queries, 1] -- caller applies sigmoid */
}
