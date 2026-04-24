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

	/* DAB-DETR conditional queries */
	struct sam3_tensor *reference_points; /* [n_queries, 4] */
	struct sam3_tensor *rph_fc1_w, *rph_fc1_b; /* ref_point_head layer 1 */
	struct sam3_tensor *rph_fc2_w, *rph_fc2_b; /* ref_point_head layer 2 */

	/* Box-relative positional bias MLPs */
	struct sam3_tensor *rpb_x_fc1_w, *rpb_x_fc1_b; /* Linear(2, 256) */
	struct sam3_tensor *rpb_x_fc2_w, *rpb_x_fc2_b; /* Linear(256, 8) */
	struct sam3_tensor *rpb_y_fc1_w, *rpb_y_fc1_b; /* Linear(2, 256) */
	struct sam3_tensor *rpb_y_fc2_w, *rpb_y_fc2_b; /* Linear(256, 8) */

	/* Output layer norm applied after all layers */
	struct sam3_tensor *output_ln_w, *output_ln_b;

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

/*
 * sam3_decoder_build_layer - Build a single decoder layer.
 *
 * Builds self-attention, vision cross-attention, text cross-attention,
 * FFN, and box refinement for layer @layer_idx. Caller must evaluate
 * the graph and persist results between layers to avoid MLX buffer reuse.
 *
 * @dec:           Initialized and loaded decoder
 * @layer_idx:     Layer index (0 to n_layers-1)
 * @g:             Graph to add nodes to
 * @q:             Query embeddings [n_queries, d_model]
 * @query_pos:     DAB-DETR position embedding [n_queries, d_model]
 * @enc_features:  Encoder output [n_pixels, d_model]
 * @enc_pos:       Encoder position embeddings [n_pixels, d_model] (may be NULL)
 * @text_features: Text embeddings [seq_len, d_model]
 * @boxes:         Box accumulator [n_queries, 4] (updated in place)
 * @arena:         Arena for intermediate tensors
 *
 * Returns updated queries [n_queries, d_model], or NULL on error.
 * @boxes is updated to include this layer's refinement.
 */
struct sam3_tensor *sam3_decoder_build_layer(
	struct sam3_decoder *dec,
	int layer_idx,
	struct sam3_graph *g,
	struct sam3_tensor *q,
	struct sam3_tensor *query_pos,
	struct sam3_tensor *enc_features,
	struct sam3_tensor *enc_pos,
	struct sam3_tensor *text_features,
	struct sam3_tensor **boxes,
	struct sam3_arena *arena);

/*
 * sam3_decoder_compute_query_pos - Compute DAB-DETR conditional query_pos.
 *
 * Computes sine position embedding from reference boxes on CPU, then
 * builds ref_point_head MLP graph nodes. Caller must evaluate graph.
 *
 * @dec:       Initialized and loaded decoder
 * @g:         Graph to add ref_point_head nodes to
 * @arena:     Arena for intermediate tensors
 * @ref_boxes: Reference box coordinates after sigmoid [n_queries, 4]
 *
 * Returns query_pos [n_queries, d_model], or NULL on error.
 */
struct sam3_tensor *sam3_decoder_compute_query_pos(
	struct sam3_decoder *dec,
	struct sam3_graph *g,
	struct sam3_arena *arena,
	const float *ref_boxes);

/*
 * sam3_decoder_compute_query_pos_batched - Batched query_pos.
 *
 * Same as sam3_decoder_compute_query_pos but accepts [B, nq, 4]
 * ref_boxes flat and returns [B, nq, d_model]. Sine embedding runs
 * CPU-side for all slots into a [B, nq, 2*d] tensor, then the
 * ref_point_head MLP (gh_linear + ReLU + gh_linear) runs on the
 * batched tensor (batch-transparent).
 *
 * @dec:       Initialized and loaded decoder
 * @g:         Graph to add ref_point_head nodes to
 * @arena:     Arena for intermediate tensors
 * @ref_boxes: [B * nq * 4] flat, sigmoid-ed cxcywh
 * @B:         Batch size (>= 1)
 *
 * Returns [B, nq, d_model] query_pos, or NULL.
 */
struct sam3_tensor *sam3_decoder_compute_query_pos_batched(
	struct sam3_decoder *dec,
	struct sam3_graph *g,
	struct sam3_arena *arena,
	const float *ref_boxes,
	int B);

/*
 * sam3_decoder_build_final - Apply output layer norm after all layers.
 *
 * @dec:   Initialized and loaded decoder
 * @g:     Graph to add nodes to
 * @q:     Query embeddings [n_queries, d_model]
 * @arena: Arena for intermediate tensors
 *
 * Returns normalized queries [n_queries, d_model], or NULL on error.
 */
struct sam3_tensor *sam3_decoder_build_final(
	struct sam3_decoder *dec,
	struct sam3_graph *g,
	struct sam3_tensor *q,
	struct sam3_arena *arena);

/*
 * Decoder substep builders for debugging. Each builds a single
 * substep of a decoder layer as a graph. Caller evaluates between
 * substeps and persists q for the next substep.
 */
struct sam3_tensor *sam3_decoder_build_sa(
	struct sam3_decoder *dec, int layer_idx,
	struct sam3_graph *g, struct sam3_tensor *q,
	struct sam3_tensor *query_pos, struct sam3_arena *arena);

/*
 * sam3_decoder_build_sa_batched - Batched decoder self-attention.
 *
 * Batched variant of sam3_decoder_build_sa. Accepts [B, nq, d] queries
 * and query_pos, returns [B, nq, d]. Per-head SDPA uses the 4D reshape
 * pattern derisked in commit f775e74.
 *
 * @dec:        Decoder
 * @layer_idx:  Which layer's weights to use
 * @g:          Graph
 * @q:          [B, nq, d_model]
 * @query_pos:  [B, nq, d_model]
 * @arena:      Arena
 *
 * Returns [B, nq, d_model], or NULL.
 */
struct sam3_tensor *sam3_decoder_build_sa_batched(
	struct sam3_decoder *dec, int layer_idx,
	struct sam3_graph *g, struct sam3_tensor *q,
	struct sam3_tensor *query_pos, struct sam3_arena *arena);

struct sam3_tensor *sam3_decoder_build_tca(
	struct sam3_decoder *dec, int layer_idx,
	struct sam3_graph *g, struct sam3_tensor *q,
	struct sam3_tensor *query_pos,
	struct sam3_tensor *text_features, struct sam3_arena *arena);

/*
 * sam3_decoder_build_tca_batched - Batched text cross-attention substep.
 *
 * Mirrors sam3_decoder_build_tca with batch dim leading. If
 * @text_features is NULL the call is a no-op (returns q unchanged).
 *
 * @q:             [B, n_q, d_model]
 * @query_pos:     [B, n_q, d_model]
 * @text_features: [B, n_text, d_model] or NULL
 * Returns [B, n_q, d_model].
 */
struct sam3_tensor *sam3_decoder_build_tca_batched(
	struct sam3_decoder *dec, int layer_idx,
	struct sam3_graph *g, struct sam3_tensor *q,
	struct sam3_tensor *query_pos,
	struct sam3_tensor *text_features, struct sam3_arena *arena);

struct sam3_tensor *sam3_decoder_build_ca(
	struct sam3_decoder *dec, int layer_idx,
	struct sam3_graph *g, struct sam3_tensor *q,
	struct sam3_tensor *query_pos,
	struct sam3_tensor *enc_features, struct sam3_tensor *enc_pos,
	struct sam3_tensor *rpb_mask, struct sam3_arena *arena);

/*
 * sam3_decoder_build_ca_batched - Batched vision cross-attention substep.
 *
 * Mirrors sam3_decoder_build_ca. When @enc_pos is non-NULL uses the
 * with-pos variant (K = enc+pos, V = enc, RPB mask applied); else
 * plain cross-attention. @rpb_mask when non-NULL must be shaped
 * [B, n_heads, n_q, n_kv] and is passed as SDPA's additive mask.
 *
 * @q:             [B, n_q, d_model]
 * @query_pos:     [B, n_q, d_model]
 * @enc_features:  [B, n_kv, d_model]
 * @enc_pos:       [n_kv, d_model] or [B, n_kv, d_model] or NULL
 * @rpb_mask:      [B, n_heads, n_q, n_kv] or NULL
 * Returns [B, n_q, d_model].
 */
struct sam3_tensor *sam3_decoder_build_ca_batched(
	struct sam3_decoder *dec, int layer_idx,
	struct sam3_graph *g, struct sam3_tensor *q,
	struct sam3_tensor *query_pos,
	struct sam3_tensor *enc_features, struct sam3_tensor *enc_pos,
	struct sam3_tensor *rpb_mask, struct sam3_arena *arena);

/*
 * sam3_decoder_compute_rpb - Compute box-relative positional bias mask.
 *
 * Produces an additive attention mask [n_heads, nq, H*W] on the CPU.
 * Called before each decoder layer's vision cross-attention.
 *
 * @dec:       Decoder with loaded RPB MLP weights
 * @ref_boxes: Reference boxes after sigmoid [nq, 4] (cxcywh format)
 * @H:         Feature map height (72)
 * @W:         Feature map width (72)
 * @out:       Output buffer, must hold n_heads * nq * H * W floats
 */
void sam3_decoder_compute_rpb(const struct sam3_decoder *dec,
			       const float *ref_boxes,
			       int H, int W, float *out);

/*
 * sam3_decoder_compute_rpb_batched - Batched RPB mask computation.
 *
 * Batched wrapper around sam3_decoder_compute_rpb. Iterates the
 * per-slot computation over B batch slots. Output is laid out so the
 * batch dim is leading: [B, n_heads, nq, H*W].
 *
 * @dec:       Decoder with loaded RPB MLP weights
 * @ref_boxes: [B, nq, 4] flat in cxcywh order (post-sigmoid)
 * @B:         Batch size (>= 1)
 * @H, @W:     Feature map spatial dims
 * @out:       Output buffer, must hold B * n_heads * nq * H * W floats
 */
void sam3_decoder_compute_rpb_batched(const struct sam3_decoder *dec,
				       const float *ref_boxes,
				       int B, int H, int W,
				       float *out);

struct sam3_tensor *sam3_decoder_build_ffn(
	struct sam3_decoder *dec, int layer_idx,
	struct sam3_graph *g, struct sam3_tensor *q,
	struct sam3_arena *arena);

/*
 * sam3_decoder_build_ffn_batched - Batched decoder FFN substep.
 *
 * gh_mlp + gh_add + gh_layernorm; all batch-transparent. Input/output
 * is [B, n_q, d_model].
 */
struct sam3_tensor *sam3_decoder_build_ffn_batched(
	struct sam3_decoder *dec, int layer_idx,
	struct sam3_graph *g, struct sam3_tensor *q,
	struct sam3_arena *arena);

/*
 * sam3_decoder_build_final_batched - Batched output layernorm.
 *
 * Trivial rank-3 wrapper around gh_layernorm using output_ln weights.
 * Input/output is [B, n_q, d_model].
 */
struct sam3_tensor *sam3_decoder_build_final_batched(
	struct sam3_decoder *dec,
	struct sam3_graph *g,
	struct sam3_tensor *q,
	struct sam3_arena *arena);

/*
 * sam3_decoder_build_layer_batched - Compose one batched decoder layer.
 *
 * Runs SA -> TCA -> CA -> FFN using the _batched substep builders.
 * Same param semantics as sam3_decoder_build_layer but every tensor
 * carries a leading batch dim B. Takes @rpb_mask directly instead of
 * the 2D version's boxes pointer-to-pointer; the caller (Task 14's
 * decoder loop) is responsible for running cpu_box_refine_batched +
 * sam3_decoder_compute_query_pos_batched + sam3_decoder_compute_rpb_batched
 * between layers — this builder does not update box state.
 *
 * @dec:           Decoder
 * @layer_idx:     Which layer's weights to use
 * @g:             Graph
 * @q:             [B, n_q, d_model]
 * @query_pos:     [B, n_q, d_model]
 * @enc_features:  [B, n_kv, d_model]
 * @enc_pos:       [n_kv, d_model] or [B, n_kv, d_model] or NULL
 * @text_features: [B, n_text, d_model] or NULL
 * @rpb_mask:      [B, n_heads, n_q, n_kv] or NULL
 * @arena:         Arena
 *
 * Returns updated queries [B, n_queries, d_model], or NULL.
 */
struct sam3_tensor *sam3_decoder_build_layer_batched(
	struct sam3_decoder *dec,
	int layer_idx,
	struct sam3_graph *g,
	struct sam3_tensor *q,
	struct sam3_tensor *query_pos,
	struct sam3_tensor *enc_features,
	struct sam3_tensor *enc_pos,
	struct sam3_tensor *text_features,
	struct sam3_tensor *rpb_mask,
	struct sam3_arena *arena);

#endif /* SAM3_MODEL_DECODER_H */
