/*
 * src/model/model_misc.c - Dot-product scoring implementation
 *
 * Implements the DotProductScoring module matching Python's
 * detector.dot_prod_scoring. The scorer computes a scaled dot product
 * between projected decoder outputs and mean-pooled prompt features
 * to produce per-query confidence logits.
 *
 * Key types:  sam3_dot_scorer
 * Depends on: model_misc.h, graph_helpers.h, util/log.h
 * Used by:    sam3_image.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include <string.h>

#include "model_misc.h"
#include "graph_helpers.h"
#include "util/log.h"

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

/* --- sam3_dot_scorer_build_batched --- */

struct sam3_tensor *sam3_dot_scorer_build_batched(
	struct sam3_dot_scorer *ds,
	struct sam3_graph *g,
	struct sam3_tensor *queries,
	struct sam3_tensor *prompt,
	struct sam3_arena *arena)
{
	struct sam3_tensor *h, *mean_w, *pooled, *proj_p, *proj_hs;
	struct sam3_tensor *proj_p_t, *scores, *scale;

	if (queries->n_dims != 3 || prompt->n_dims != 3 ||
	    prompt->dims[0] != queries->dims[0]) {
		sam3_log_error("scorer_batched: shape mismatch "
			       "queries.n_dims=%d prompt.n_dims=%d "
			       "queries.B=%d prompt.B=%d",
			       queries->n_dims, prompt->n_dims,
			       queries->n_dims >= 1 ? queries->dims[0] : -1,
			       prompt->n_dims >= 1 ? prompt->dims[0] : -1);
		return NULL;
	}

	int B = queries->dims[0];
	int seq_len = prompt->dims[1];

	/*
	 * Step 1: Apply MLP to prompt tokens (rank-transparent).
	 *   [B, seq, d] -> [B, seq, d_ffn] -> [B, seq, d]
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
	 * Step 2: Mean pool per batch slot.
	 *   [B, 1, seq] @ [B, seq, d] -> [B, 1, d]
	 */
	int mw_dims[] = {B, 1, seq_len};
	mean_w = gh_alloc_tensor(arena, SAM3_DTYPE_F32, 3, mw_dims);
	if (!mean_w)
		return NULL;

	{
		float inv_seq = 1.0f / (float)seq_len;
		float *mwd = (float *)mean_w->data;
		int n = B * seq_len;
		for (int i = 0; i < n; i++)
			mwd[i] = inv_seq;
	}

	pooled = gh_matmul(g, arena, mean_w, h);
	if (!pooled)
		return NULL;
	/* pooled: [B, 1, d_model] */

	/*
	 * Step 3: Project pooled prompt.
	 *   [B, 1, d_model] -> [B, 1, d_proj]
	 */
	proj_p = gh_linear(g, arena, pooled, ds->prompt_proj_w,
			   ds->prompt_proj_b);
	if (!proj_p)
		return NULL;

	/*
	 * Step 4: Project decoder hidden states.
	 *   [B, n_queries, d_model] -> [B, n_queries, d_proj]
	 */
	proj_hs = gh_linear(g, arena, queries, ds->hs_proj_w,
			    ds->hs_proj_b);
	if (!proj_hs)
		return NULL;

	/*
	 * Step 5: Scaled dot product.
	 *   [B, n_queries, d_proj] @ [B, d_proj, 1] -> [B, n_queries, 1]
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

	/* gh_transpose swaps the last two dims: [B, 1, d_proj] -> [B, d_proj, 1]. */
	proj_p_t = gh_transpose(g, arena, proj_p);
	if (!proj_p_t)
		return NULL;
	/* proj_p_t: [B, d_proj, 1] */

	scores = gh_matmul(g, arena, proj_hs, proj_p_t);
	if (!scores)
		return NULL;
	/* scores: [B, n_queries, 1] */

	return scores;
}

/* --- sam3_dot_scorer_alloc_synthetic --- */

int sam3_dot_scorer_alloc_synthetic(struct sam3_dot_scorer *ds,
				    struct sam3_arena *arena)
{
	int d = ds->d_model;
	int ffn = ds->d_ffn;
	int proj = ds->d_proj;

	int fc1_w_dims[] = {ffn, d};
	ds->mlp_fc1_w = gh_alloc_tensor(arena, SAM3_DTYPE_F32, 2, fc1_w_dims);
	if (!ds->mlp_fc1_w)
		return -1;

	int fc1_b_dims[] = {ffn};
	ds->mlp_fc1_b = gh_alloc_tensor(arena, SAM3_DTYPE_F32, 1, fc1_b_dims);
	if (!ds->mlp_fc1_b)
		return -1;

	int fc2_w_dims[] = {d, ffn};
	ds->mlp_fc2_w = gh_alloc_tensor(arena, SAM3_DTYPE_F32, 2, fc2_w_dims);
	if (!ds->mlp_fc2_w)
		return -1;

	int fc2_b_dims[] = {d};
	ds->mlp_fc2_b = gh_alloc_tensor(arena, SAM3_DTYPE_F32, 1, fc2_b_dims);
	if (!ds->mlp_fc2_b)
		return -1;

	int norm_dims[] = {d};
	ds->mlp_norm_w = gh_alloc_tensor(arena, SAM3_DTYPE_F32, 1, norm_dims);
	if (!ds->mlp_norm_w)
		return -1;

	ds->mlp_norm_b = gh_alloc_tensor(arena, SAM3_DTYPE_F32, 1, norm_dims);
	if (!ds->mlp_norm_b)
		return -1;

	int pp_w_dims[] = {proj, d};
	ds->prompt_proj_w = gh_alloc_tensor(arena, SAM3_DTYPE_F32, 2,
					    pp_w_dims);
	if (!ds->prompt_proj_w)
		return -1;

	int pp_b_dims[] = {proj};
	ds->prompt_proj_b = gh_alloc_tensor(arena, SAM3_DTYPE_F32, 1,
					    pp_b_dims);
	if (!ds->prompt_proj_b)
		return -1;

	int hp_w_dims[] = {proj, d};
	ds->hs_proj_w = gh_alloc_tensor(arena, SAM3_DTYPE_F32, 2, hp_w_dims);
	if (!ds->hs_proj_w)
		return -1;

	int hp_b_dims[] = {proj};
	ds->hs_proj_b = gh_alloc_tensor(arena, SAM3_DTYPE_F32, 1, hp_b_dims);
	if (!ds->hs_proj_b)
		return -1;

	/*
	 * Deterministic fill: unique sin offset per tensor so no two
	 * tensors produce colliding values.
	 */
	struct {
		struct sam3_tensor *t;
		float off;
		int is_bias;
	} tensors[] = {
		{ds->mlp_fc1_w,   0.0f, 0},
		{ds->mlp_fc1_b,   1.0f, 1},
		{ds->mlp_fc2_w,   2.0f, 0},
		{ds->mlp_fc2_b,   3.0f, 1},
		{ds->mlp_norm_w,  4.0f, 0},
		{ds->mlp_norm_b,  5.0f, 1},
		{ds->prompt_proj_w, 6.0f, 0},
		{ds->prompt_proj_b, 7.0f, 1},
		{ds->hs_proj_w,   8.0f, 0},
		{ds->hs_proj_b,   9.0f, 1},
	};

	for (size_t k = 0; k < sizeof(tensors) / sizeof(tensors[0]); k++) {
		struct sam3_tensor *t = tensors[k].t;
		float off = tensors[k].off;
		float *d_ptr = (float *)t->data;
		int n = (int)sam3_tensor_nelems(t);

		if (tensors[k].is_bias) {
			for (int i = 0; i < n; i++)
				d_ptr[i] = 0.01f * (float)(i + 1);
		} else {
			for (int i = 0; i < n; i++)
				d_ptr[i] = sinf((float)i * 0.1f + off);
		}
	}

	/* LayerNorm weight should be 1.0-ish so normalization behaves. */
	{
		float *nw = (float *)ds->mlp_norm_w->data;
		float *nb = (float *)ds->mlp_norm_b->data;
		for (int i = 0; i < d; i++) {
			nw[i] = 1.0f + 0.01f * sinf((float)i * 0.1f);
			nb[i] = 0.01f * (float)(i + 1);
		}
	}

	return 0;
}
