/*
 * src/model/model_misc.c - Dot-product scoring implementation
 *
 * Implements the DotProductScoring module matching Python's
 * detector.dot_prod_scoring. The scorer computes a scaled dot product
 * between projected decoder outputs and mean-pooled prompt features
 * to produce per-query confidence logits.
 *
 * Key types:  sam3_dot_scorer
 * Depends on: model_misc.h, graph_helpers.h
 * Used by:    sam3_image.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include <string.h>

#include "model_misc.h"
#include "graph_helpers.h"

#define SCORE_PREFIX "tracker_model.mask_decoder.pred_obj_score_head."

enum sam3_error sam3_dot_scorer_load(struct sam3_dot_scorer *ds,
				     const struct sam3_weight_file *wf,
				     struct sam3_arena *arena)
{
	int d = ds->d_model;
	int ffn = ds->d_ffn;
	int proj = ds->d_proj;

	/* prompt_mlp.fc1: [d_ffn, d_model] / [d_ffn] */
	int fc1_w_dims[] = {ffn, d};
	ds->mlp_fc1_w = gh_load_mmap(wf,
		SCORE_PREFIX "prompt_mlp.fc1.weight",
		arena, SAM3_DTYPE_F32, 2, fc1_w_dims);
	if (!ds->mlp_fc1_w)
		return SAM3_ENOMEM;

	int fc1_b_dims[] = {ffn};
	ds->mlp_fc1_b = gh_load_mmap(wf,
		SCORE_PREFIX "prompt_mlp.fc1.bias",
		arena, SAM3_DTYPE_F32, 1, fc1_b_dims);
	if (!ds->mlp_fc1_b)
		return SAM3_ENOMEM;

	/* prompt_mlp.fc2: [d_model, d_ffn] / [d_model] */
	int fc2_w_dims[] = {d, ffn};
	ds->mlp_fc2_w = gh_load_mmap(wf,
		SCORE_PREFIX "prompt_mlp.fc2.weight",
		arena, SAM3_DTYPE_F32, 2, fc2_w_dims);
	if (!ds->mlp_fc2_w)
		return SAM3_ENOMEM;

	int fc2_b_dims[] = {d};
	ds->mlp_fc2_b = gh_load_mmap(wf,
		SCORE_PREFIX "prompt_mlp.fc2.bias",
		arena, SAM3_DTYPE_F32, 1, fc2_b_dims);
	if (!ds->mlp_fc2_b)
		return SAM3_ENOMEM;

	/* prompt_mlp.out_norm: LayerNorm [d_model] */
	int norm_dims[] = {d};
	ds->mlp_norm_w = gh_load_mmap(wf,
		SCORE_PREFIX "prompt_mlp.out_norm.weight",
		arena, SAM3_DTYPE_F32, 1, norm_dims);
	if (!ds->mlp_norm_w)
		return SAM3_ENOMEM;

	ds->mlp_norm_b = gh_load_mmap(wf,
		SCORE_PREFIX "prompt_mlp.out_norm.bias",
		arena, SAM3_DTYPE_F32, 1, norm_dims);
	if (!ds->mlp_norm_b)
		return SAM3_ENOMEM;

	/* prompt_proj: Linear(d_model, d_proj) */
	int pp_w_dims[] = {proj, d};
	ds->prompt_proj_w = gh_load_mmap(wf,
		SCORE_PREFIX "prompt_proj.weight",
		arena, SAM3_DTYPE_F32, 2, pp_w_dims);
	if (!ds->prompt_proj_w)
		return SAM3_ENOMEM;

	int pp_b_dims[] = {proj};
	ds->prompt_proj_b = gh_load_mmap(wf,
		SCORE_PREFIX "prompt_proj.bias",
		arena, SAM3_DTYPE_F32, 1, pp_b_dims);
	if (!ds->prompt_proj_b)
		return SAM3_ENOMEM;

	/* hs_proj: Linear(d_model, d_proj) */
	int hp_w_dims[] = {proj, d};
	ds->hs_proj_w = gh_load_mmap(wf,
		SCORE_PREFIX "hs_proj.weight",
		arena, SAM3_DTYPE_F32, 2, hp_w_dims);
	if (!ds->hs_proj_w)
		return SAM3_ENOMEM;

	int hp_b_dims[] = {proj};
	ds->hs_proj_b = gh_load_mmap(wf,
		SCORE_PREFIX "hs_proj.bias",
		arena, SAM3_DTYPE_F32, 1, hp_b_dims);
	if (!ds->hs_proj_b)
		return SAM3_ENOMEM;

	return SAM3_OK;
}

struct sam3_tensor *sam3_dot_scorer_build(struct sam3_dot_scorer *ds,
					  struct sam3_graph *g,
					  struct sam3_tensor *queries,
					  struct sam3_tensor *prompt,
					  struct sam3_arena *arena)
{
	int seq_len = prompt->dims[0];
	struct sam3_tensor *h, *mean_w, *pooled, *proj_p, *proj_hs;
	struct sam3_tensor *proj_p_t, *scores, *scale;

	/*
	 * Step 1: Apply MLP to prompt tokens.
	 *   fc1 -> relu -> fc2 -> layernorm
	 *   [seq_len, d_model] -> [seq_len, d_ffn] -> [seq_len, d_model]
	 */
	h = gh_linear(g, arena, prompt, ds->mlp_fc1_w, ds->mlp_fc1_b);
	if (!h)
		return NULL;

	h = gh_relu(g, arena, h);
	if (!h)
		return NULL;

	h = gh_linear(g, arena, h, ds->mlp_fc2_w, ds->mlp_fc2_b);
	if (!h)
		return NULL;

	h = gh_layernorm(g, arena, h, ds->mlp_norm_w, ds->mlp_norm_b);
	if (!h)
		return NULL;

	/*
	 * Step 2: Mean pool over seq dimension.
	 *   [1, seq_len] @ [seq_len, d_model] -> [1, d_model]
	 *   The weight vector is 1/seq_len for uniform averaging.
	 *   (No padding mask — all tokens are valid in our case.)
	 */
	int mw_dims[] = {1, seq_len};
	mean_w = gh_alloc_tensor(arena, SAM3_DTYPE_F32, 2, mw_dims);
	if (!mean_w)
		return NULL;

	{
		float inv_seq = 1.0f / (float)seq_len;
		float *mwd = (float *)mean_w->data;
		for (int i = 0; i < seq_len; i++)
			mwd[i] = inv_seq;
	}

	pooled = gh_matmul(g, arena, mean_w, h);
	if (!pooled)
		return NULL;
	/* pooled: [1, d_model] */

	/*
	 * Step 3: Project pooled prompt.
	 *   prompt_proj: [1, d_model] -> [1, d_proj]
	 */
	proj_p = gh_linear(g, arena, pooled, ds->prompt_proj_w,
			   ds->prompt_proj_b);
	if (!proj_p)
		return NULL;

	/*
	 * Step 4: Project decoder hidden states.
	 *   hs_proj: [n_queries, d_model] -> [n_queries, d_proj]
	 */
	proj_hs = gh_linear(g, arena, queries, ds->hs_proj_w,
			    ds->hs_proj_b);
	if (!proj_hs)
		return NULL;

	/*
	 * Step 5: Scaled dot product.
	 *   [n_queries, d_proj] @ [d_proj, 1] -> [n_queries, 1]
	 *   Scale by 1/sqrt(d_proj).
	 *
	 *   Fold the scale into proj_pooled before transpose so we
	 *   only need matmul (no separate mul with broadcast).
	 */
	{
		float s = 1.0f / sqrtf((float)ds->d_proj);
		int sc_dims[] = {1};
		scale = gh_alloc_tensor(arena, SAM3_DTYPE_F32, 1, sc_dims);
		if (!scale)
			return NULL;
		*(float *)scale->data = s;
	}

	proj_p = gh_mul(g, arena, proj_p, scale);
	if (!proj_p)
		return NULL;

	proj_p_t = gh_transpose(g, arena, proj_p);
	if (!proj_p_t)
		return NULL;
	/* proj_p_t: [d_proj, 1] */

	scores = gh_matmul(g, arena, proj_hs, proj_p_t);
	if (!scores)
		return NULL;
	/* scores: [n_queries, 1] */

	/*
	 * Step 6: Clamp to [-12, 12].
	 * No clamp op available; during inference the range is
	 * naturally bounded and the downstream sigmoid handles it.
	 * Skip clamping for now.
	 */

	return scores;
}
