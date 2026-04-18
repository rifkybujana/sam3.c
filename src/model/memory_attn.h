/*
 * src/model/memory_attn.h - 4-layer RoPE cross-attention transformer
 *
 * Implements the memory attention module (TransformerEncoderCrossAttention)
 * for SAM3 video tracking. Each of the 4 layers applies pre-norm
 * self-attention with 2D RoPE, pre-norm cross-attention from current
 * features to memory features (dim=64), and a pre-norm FFN with ReLU.
 * A final LayerNorm is applied after all layers.
 *
 * Weight prefix: tracker_model.transformer.encoder.*
 *
 * Key types:  sam3_memory_attn, sam3_memattn_layer
 * Depends on: core/tensor.h, core/graph.h, core/alloc.h, core/weight.h
 * Used by:    model/tracker.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_MEMORY_ATTN_H
#define SAM3_MODEL_MEMORY_ATTN_H

#include "core/tensor.h"
#include "core/graph.h"
#include "core/alloc.h"
#include "core/weight.h"

#define SAM3_MEMATTN_N_LAYERS 4
#define SAM3_MEMATTN_D_MODEL  256
#define SAM3_MEMATTN_FFN_DIM  2048

struct sam3_memattn_layer {
	/* Self-attention weights (separate Q/K/V) */
	struct sam3_tensor *sa_q_w, *sa_q_b;   /* [256, 256] / [256] */
	struct sam3_tensor *sa_k_w, *sa_k_b;   /* [256, 256] / [256] */
	struct sam3_tensor *sa_v_w, *sa_v_b;   /* [256, 256] / [256] */
	struct sam3_tensor *sa_out_w, *sa_out_b;

	/* Cross-attention weights */
	struct sam3_tensor *ca_q_w, *ca_q_b;   /* [256, 256] / [256] */
	struct sam3_tensor *ca_k_w, *ca_k_b;   /* [256, 64]  / [256] */
	struct sam3_tensor *ca_v_w, *ca_v_b;   /* [256, 64]  / [256] */
	struct sam3_tensor *ca_out_w, *ca_out_b;

	/* LayerNorm weights (pre-norm) */
	struct sam3_tensor *norm1_w, *norm1_b; /* before self-attn */
	struct sam3_tensor *norm2_w, *norm2_b; /* before cross-attn */
	struct sam3_tensor *norm3_w, *norm3_b; /* before FFN */

	/* FFN weights */
	struct sam3_tensor *ffn_fc1_w, *ffn_fc1_b; /* [2048, 256] / [2048] */
	struct sam3_tensor *ffn_fc2_w, *ffn_fc2_b; /* [256, 2048] / [256] */
};

struct sam3_memory_attn {
	int d_model;  /* 256 */
	int mem_dim;  /* 64: memory feature dimension (kv_in_dim) */
	int n_layers; /* 4 */
	int n_heads;  /* 1 */
	int feat_h;   /* 72: spatial grid height for RoPE */
	int feat_w;   /* 72: spatial grid width for RoPE */

	struct sam3_memattn_layer layers[SAM3_MEMATTN_N_LAYERS];

	/* Final LayerNorm after all layers */
	struct sam3_tensor *final_norm_w, *final_norm_b;

	/* RoPE: precomputed cos/sin tables for self-attention */
	struct sam3_tensor *rope_cos; /* [seq, head_dim/2] e.g. [5184, 128] */
	struct sam3_tensor *rope_sin;
};

/*
 * sam3_memory_attn_init - Initialize memory attention with dimensions.
 *
 * @attn:     Attention struct (caller-allocated, zeroed and configured)
 * @d_model:  Internal feature dimension (256)
 * @mem_dim:  Memory feature dimension / kv_in_dim (64)
 * @n_layers: Number of transformer layers (4)
 * @n_heads:  Number of attention heads (1)
 * @feat_h:   Spatial grid height for RoPE (72)
 * @feat_w:   Spatial grid width for RoPE (72)
 *
 * Returns SAM3_OK on success, SAM3_EINVAL for bad arguments.
 */
enum sam3_error sam3_memory_attn_init(struct sam3_memory_attn *attn,
				      int d_model, int mem_dim,
				      int n_layers, int n_heads,
				      int feat_h, int feat_w);

/*
 * sam3_memory_attn_load - Load weights and precompute RoPE tables.
 *
 * @attn:  Initialized attention module
 * @wf:    Weight file (NULL for zero-init testing)
 * @arena: Arena for tensor allocation
 *
 * Weight prefix: tracker_model.transformer.encoder.*
 * Falls back to zero-initialized tensors when wf is NULL.
 * Precomputes 2D axial RoPE cos/sin tables for the spatial grid.
 *
 * Returns SAM3_OK on success, SAM3_ENOMEM if arena is full.
 */
enum sam3_error sam3_memory_attn_load(struct sam3_memory_attn *attn,
				      const struct sam3_weight_file *wf,
				      struct sam3_arena *arena);

/*
 * sam3_memory_attn_build_full - Build the 4-layer transformer graph.
 *
 * @attn:    Loaded attention module
 * @g:       Graph to add nodes to
 * @current: Current frame features [seq, d_model] (seq = feat_h * feat_w)
 * @memory:  Memory features [n_mem, mem_dim]
 * @mem_pos: Memory position encoding [n_mem, mem_dim], or NULL.
 *           When non-NULL, added to memory before cross-attention key
 *           projection (matches pos_enc_at_cross_attn_keys=True).
 * @arena:   Arena for intermediate tensors
 * @output:  Output: transformed features [seq, d_model]
 *
 * NOTE: The Python reference applies pos_enc_at_input (output += 0.1 *
 * src_pos) before the first layer. The caller is responsible for adding
 * the scaled position encoding to current features before calling this.
 *
 * Pipeline per layer:
 *   1. x_norm = LayerNorm(norm1, x); x = x + self_attn(x_norm)
 *   2. x_norm = LayerNorm(norm2, x); x = x + cross_attn(x_norm, memory)
 *   3. x_norm = LayerNorm(norm3, x); x = x + ffn(x_norm)
 * After all layers: x = LayerNorm(final_norm, x)
 *
 * Returns SAM3_OK on success, SAM3_EINVAL for bad arguments,
 * SAM3_ENOMEM if graph or arena allocation fails.
 */
enum sam3_error sam3_memory_attn_build_full(struct sam3_memory_attn *attn,
					    struct sam3_graph *g,
					    struct sam3_tensor *current,
					    struct sam3_tensor *memory,
					    struct sam3_tensor *mem_pos,
					    struct sam3_arena *arena,
					    struct sam3_tensor **output);

#endif /* SAM3_MODEL_MEMORY_ATTN_H */
