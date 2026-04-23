/*
 * src/model/model_misc.h - Dot-product scoring for SAM3 DETR
 *
 * Implements the DotProductScoring module that produces per-query
 * confidence logits. The scorer applies a 2-layer MLP + LayerNorm
 * to the prompt tokens, mean-pools them, projects both the pooled
 * prompt and decoder hidden states, then computes a scaled dot
 * product to produce [n_queries, 1] scores.
 *
 * Key types:  sam3_dot_scorer
 * Depends on: core/graph.h, core/alloc.h, core/tensor.h, core/weight.h
 * Used by:    sam3_image.c
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

/*
 * Dot-product scorer matching Python's DotProductScoring module.
 *
 * Architecture:
 *   1. prompt_mlp(prompt): fc1 -> relu -> fc2 -> layernorm
 *   2. pooled = mean_pool(transformed_prompt)
 *   3. proj_pooled = prompt_proj(pooled)
 *   4. proj_hs = hs_proj(queries)
 *   5. scores = (proj_hs @ proj_pooled^T) / sqrt(d_proj)
 *   6. clamp(scores, -12, 12)
 */
struct sam3_dot_scorer {
	/* prompt_mlp: 2-layer MLP on prompt tokens */
	struct sam3_tensor *mlp_fc1_w, *mlp_fc1_b;   /* [d_ffn, d_model] / [d_ffn] */
	struct sam3_tensor *mlp_fc2_w, *mlp_fc2_b;   /* [d_model, d_ffn] / [d_model] */
	struct sam3_tensor *mlp_norm_w, *mlp_norm_b;  /* [d_model] / [d_model] */

	/* projection heads */
	struct sam3_tensor *prompt_proj_w, *prompt_proj_b; /* [d_proj, d_model] / [d_proj] */
	struct sam3_tensor *hs_proj_w, *hs_proj_b;        /* [d_proj, d_model] / [d_proj] */

	int d_model;  /* 256 */
	int d_proj;   /* 256 */
	int d_ffn;    /* 2048 */
};

/*
 * sam3_dot_scorer_load - Load dot scorer weights from weight file.
 *
 * @ds:    Dot scorer struct (caller-allocated, d_model/d_proj/d_ffn set)
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
 * Produces per-query confidence logits via scaled dot-product scoring
 * between decoder hidden states and mean-pooled prompt features.
 *
 * @ds:      Loaded dot scorer
 * @g:       Graph to add nodes to
 * @queries: [n_queries, d_model] -- decoder output (last layer)
 * @prompt:  [seq_len, d_model] -- concatenated text + geometry features
 * @arena:   Arena for intermediate tensors
 *
 * Returns confidence logits [n_queries, 1], or NULL on error.
 * Caller should apply sigmoid for probabilities.
 */
struct sam3_tensor *sam3_dot_scorer_build(struct sam3_dot_scorer *ds,
					  struct sam3_graph *g,
					  struct sam3_tensor *queries,
					  struct sam3_tensor *prompt,
					  struct sam3_arena *arena);

/*
 * sam3_dot_scorer_build_batched - Batched dot-product scoring graph.
 *
 * Batched variant of sam3_dot_scorer_build. Every tensor gains a
 * leading batch dim B. Each batch slot runs the same 6-step pipeline
 * (prompt MLP -> mean pool -> dual projection -> scaled dot product)
 * independently; outputs for different slots are concatenated along
 * the batch dim.
 *
 * @ds:      Loaded dot scorer
 * @g:       Graph to add nodes to
 * @queries: [B, n_queries, d_model]
 * @prompt:  [B, seq_len, d_model]
 * @arena:   Arena for intermediate tensors
 *
 * Returns confidence logits [B, n_queries, 1], or NULL on error.
 * The caller applies sigmoid for probabilities.
 */
struct sam3_tensor *sam3_dot_scorer_build_batched(
	struct sam3_dot_scorer *ds,
	struct sam3_graph *g,
	struct sam3_tensor *queries,
	struct sam3_tensor *prompt,
	struct sam3_arena *arena);

/*
 * sam3_dot_scorer_alloc_synthetic - Allocate test weights on an arena.
 *
 * Fills a pre-initialized (d_model / d_proj / d_ffn set) dot scorer
 * with deterministic F32 values on @arena so unit tests can exercise
 * the graph without a real model file. Returns 0 on success, -1 on
 * allocation failure. Not intended for production use.
 */
int sam3_dot_scorer_alloc_synthetic(struct sam3_dot_scorer *ds,
				    struct sam3_arena *arena);

#endif /* SAM3_MODEL_MISC_H */
