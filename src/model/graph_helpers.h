/*
 * src/model/graph_helpers.h - Helper functions for building compute graphs
 *
 * Provides convenience wrappers around sam3_graph_add_op() that handle
 * output tensor allocation from the arena. Used by all model modules
 * to build their compute subgraphs with minimal boilerplate.
 *
 * Key types:  (none -- uses sam3_tensor, sam3_graph, sam3_arena)
 * Depends on: core/graph.h, core/alloc.h, core/tensor.h, core/weight.h
 * Used by:    model/ files (vitdet.c, text_encoder.c, decoder.c, etc.)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_GRAPH_HELPERS_H
#define SAM3_MODEL_GRAPH_HELPERS_H

#include "core/graph.h"
#include "core/alloc.h"
#include "core/tensor.h"
#include "core/weight.h"

/*
 * gh_alloc_tensor - Allocate a tensor with given shape from the arena.
 *
 * @arena:  Arena to allocate from
 * @dtype:  Data type for the tensor
 * @n_dims: Number of dimensions (1-4)
 * @dims:   Array of dimension sizes
 *
 * Allocates both the sam3_tensor struct and its data buffer from the
 * arena. Computes strides and nbytes automatically.
 * Returns NULL if the arena is full.
 */
struct sam3_tensor *gh_alloc_tensor(struct sam3_arena *arena,
				    enum sam3_dtype dtype,
				    int n_dims, const int *dims);

/*
 * gh_load_or_alloc - Load a weight tensor by name, or allocate zeroed.
 *
 * When wf is NULL or the tensor is not found, allocates a
 * zero-initialized tensor from the arena.
 */
struct sam3_tensor *gh_load_or_alloc(const struct sam3_weight_file *wf,
				      const char *name,
				      struct sam3_arena *arena,
				      enum sam3_dtype dtype,
				      int n_dims, const int *dims);

/* --- Unary activation ops (output shape = input shape) --- */

struct sam3_tensor *gh_gelu(struct sam3_graph *g, struct sam3_arena *a,
			    struct sam3_tensor *input);

struct sam3_tensor *gh_relu(struct sam3_graph *g, struct sam3_arena *a,
			    struct sam3_tensor *input);

struct sam3_tensor *gh_sigmoid(struct sam3_graph *g, struct sam3_arena *a,
			       struct sam3_tensor *input);

struct sam3_tensor *gh_silu(struct sam3_graph *g, struct sam3_arena *a,
			    struct sam3_tensor *input);

struct sam3_tensor *gh_softmax(struct sam3_graph *g, struct sam3_arena *a,
			       struct sam3_tensor *input);

/* --- Binary element-wise ops (output shape = first input's shape) --- */

struct sam3_tensor *gh_add(struct sam3_graph *g, struct sam3_arena *a,
			   struct sam3_tensor *x, struct sam3_tensor *b);

struct sam3_tensor *gh_mul(struct sam3_graph *g, struct sam3_arena *a,
			   struct sam3_tensor *x, struct sam3_tensor *b);

/*
 * gh_matmul - Matrix multiply: output [..., M, N] from
 *             [..., M, K] @ [..., K, N].
 */
struct sam3_tensor *gh_matmul(struct sam3_graph *g, struct sam3_arena *a,
			      struct sam3_tensor *x, struct sam3_tensor *w);

/*
 * gh_linear - Fully-connected layer: out = input @ weight^T + bias.
 *
 * @bias may be NULL for no bias.
 */
struct sam3_tensor *gh_linear(struct sam3_graph *g, struct sam3_arena *a,
			      struct sam3_tensor *input,
			      struct sam3_tensor *weight,
			      struct sam3_tensor *bias);

/*
 * gh_layernorm - Layer normalization with optional affine parameters.
 *
 * @gamma: Scale parameter (same size as last dim), or NULL.
 * @beta:  Shift parameter (same size as last dim), or NULL.
 */
struct sam3_tensor *gh_layernorm(struct sam3_graph *g, struct sam3_arena *a,
				struct sam3_tensor *input,
				struct sam3_tensor *gamma,
				struct sam3_tensor *beta);

/*
 * gh_reshape - Change tensor dimensions without copying data.
 */
struct sam3_tensor *gh_reshape(struct sam3_graph *g, struct sam3_arena *a,
			       struct sam3_tensor *input,
			       int n_dims, const int *dims);

/*
 * gh_transpose - Swap last two dimensions of a tensor.
 */
struct sam3_tensor *gh_transpose(struct sam3_graph *g, struct sam3_arena *a,
				 struct sam3_tensor *input);

/*
 * gh_concat - Concatenate tensors along a given axis.
 *
 * Output shape matches tensors[0] except on the concat axis, which
 * is the sum of all input axis dimensions.
 */
struct sam3_tensor *gh_concat(struct sam3_graph *g, struct sam3_arena *a,
			      struct sam3_tensor **tensors, int n, int axis);

/*
 * gh_slice - Extract a sub-tensor along a given axis.
 *
 * @axis:  Dimension to slice along
 * @start: Start index (inclusive)
 * @end:   End index (exclusive)
 */
struct sam3_tensor *gh_slice(struct sam3_graph *g, struct sam3_arena *a,
			     struct sam3_tensor *input,
			     int axis, int start, int end);

/*
 * gh_embed - Embedding table lookup.
 *
 * @table:   Embedding table [vocab_size, embed_dim]
 * @indices: Integer tensor of indices
 *
 * Output shape: [indices_nelems, embed_dim].
 */
struct sam3_tensor *gh_embed(struct sam3_graph *g, struct sam3_arena *a,
			     struct sam3_tensor *table,
			     struct sam3_tensor *indices);

/*
 * gh_upsample - Nearest-neighbor upsampling for 4D tensors.
 *
 * Input must be [N, C, H, W]. Output is [N, C, H*scale, W*scale].
 */
struct sam3_tensor *gh_upsample(struct sam3_graph *g, struct sam3_arena *a,
				struct sam3_tensor *input, int scale);

/*
 * gh_rope - Apply rotary position embedding.
 *
 * @input: [batch, seq, n_heads, head_dim] tensor
 * @cos_f: Precomputed cosine [seq, head_dim/2]
 * @sin_f: Precomputed sine [seq, head_dim/2]
 *
 * Returns rotated tensor, same shape as input.
 */
struct sam3_tensor *gh_rope(struct sam3_graph *g, struct sam3_arena *a,
			     struct sam3_tensor *input,
			     struct sam3_tensor *cos_f,
			     struct sam3_tensor *sin_f);

/*
 * gh_multihead_attention_rope - Multi-head attention with optional RoPE
 *                               and causal mask.
 *
 * @q:         Query  [batch, seq_q, d_model]
 * @k:         Key    (unused; QKV is packed via qkv_w)
 * @v:         Value  (unused; QKV is packed via qkv_w)
 * @qkv_w:    Packed QKV weight [3*d_model, d_model]
 * @qkv_b:    Packed QKV bias [3*d_model] or NULL
 * @out_w:    Output projection weight [d_model, d_model]
 * @out_b:    Output projection bias [d_model] or NULL
 * @n_heads:  Number of attention heads
 * @rope_cos: Precomputed RoPE cosine [seq, head_dim/2] or NULL
 * @rope_sin: Precomputed RoPE sine [seq, head_dim/2] or NULL
 * @attn_mask: Additive attention mask [bs, bs] or NULL
 *
 * Extends gh_multihead_attention with optional rotary position
 * embeddings applied after QKV projection and optional additive
 * mask applied to attention scores before softmax.
 *
 * Returns output tensor [batch*seq_q, d_model].
 */
struct sam3_tensor *gh_multihead_attention_rope(
	struct sam3_graph *g, struct sam3_arena *a,
	struct sam3_tensor *q,
	struct sam3_tensor *k,
	struct sam3_tensor *v,
	struct sam3_tensor *qkv_w,
	struct sam3_tensor *qkv_b,
	struct sam3_tensor *out_w,
	struct sam3_tensor *out_b,
	int n_heads,
	struct sam3_tensor *rope_cos,
	struct sam3_tensor *rope_sin,
	struct sam3_tensor *attn_mask);

/*
 * gh_multihead_attention - Full multi-head attention from primitives.
 *
 * @q:       Query  [batch, seq_q, d_model]
 * @k:       Key    (unused; QKV is packed via qkv_w)
 * @v:       Value  (unused; QKV is packed via qkv_w)
 * @qkv_w:  Packed QKV weight [3*d_model, d_model]
 * @qkv_b:  Packed QKV bias [3*d_model] or NULL
 * @out_w:  Output projection weight [d_model, d_model]
 * @out_b:  Output projection bias [d_model] or NULL
 * @n_heads: Number of attention heads
 *
 * Flattens to 2D internally for compatibility with the CPU backend's
 * 2D-only transpose and matmul kernels.  Each head is computed
 * independently via slice + 2D ops, then concatenated.
 *
 * Returns output tensor [batch*seq_q, d_model].
 */
struct sam3_tensor *gh_multihead_attention(
	struct sam3_graph *g, struct sam3_arena *a,
	struct sam3_tensor *q,
	struct sam3_tensor *k,
	struct sam3_tensor *v,
	struct sam3_tensor *qkv_w,
	struct sam3_tensor *qkv_b,
	struct sam3_tensor *out_w,
	struct sam3_tensor *out_b,
	int n_heads);

/*
 * gh_cross_attention - Multi-head cross-attention with separate Q and KV.
 *
 * Q is projected from q_src, K and V are projected from kv_src.
 * Uses packed KV weight [2*d_model, d_model] which is sliced into
 * separate K and V projections.
 *
 * @g:       Graph to add nodes to
 * @arena:   Arena for intermediate tensors
 * @q_src:   Query source [n_q, d_model]
 * @kv_src:  Key/value source [n_kv, d_model]
 * @q_w:     Query projection weight [d_model, d_model]
 * @q_b:     Query projection bias [d_model]
 * @kv_w:    Packed KV projection weight [2*d_model, d_model]
 * @kv_b:    Packed KV projection bias [2*d_model]
 * @out_w:   Output projection weight [d_model, d_model]
 * @out_b:   Output projection bias [d_model]
 * @n_heads: Number of attention heads
 *
 * Returns output tensor [n_q, d_model], or NULL on error.
 */
struct sam3_tensor *gh_cross_attention(
	struct sam3_graph *g, struct sam3_arena *arena,
	struct sam3_tensor *q_src,
	struct sam3_tensor *kv_src,
	struct sam3_tensor *q_w, struct sam3_tensor *q_b,
	struct sam3_tensor *kv_w, struct sam3_tensor *kv_b,
	struct sam3_tensor *out_w, struct sam3_tensor *out_b,
	int n_heads);

/*
 * gh_mlp - Two-layer MLP: linear -> activation -> linear.
 *
 * @activation: Op enum for the activation function (e.g. SAM3_OP_GELU).
 */
struct sam3_tensor *gh_mlp(struct sam3_graph *g, struct sam3_arena *a,
			   struct sam3_tensor *input,
			   struct sam3_tensor *fc1_w, struct sam3_tensor *fc1_b,
			   struct sam3_tensor *fc2_w, struct sam3_tensor *fc2_b,
			   enum sam3_op activation);

/*
 * gh_conv2d - 2D convolution with optional bias.
 *
 * @input:   [N, C_in, H, W]
 * @weight:  [C_out, C_in, KH, KW]
 * @bias:    [C_out] or NULL
 * @stride:  Convolution stride
 * @padding: Zero-padding on each side
 *
 * Bias is added via reshape+transpose+add+transpose+reshape.
 * Returns [N, C_out, OH, OW].
 */
struct sam3_tensor *gh_conv2d(struct sam3_graph *g, struct sam3_arena *a,
			      struct sam3_tensor *input,
			      struct sam3_tensor *weight,
			      struct sam3_tensor *bias,
			      int stride, int padding);

/*
 * gh_conv_transpose2d - 2D transposed convolution with optional bias.
 *
 * @input:   [N, C_in, H, W]
 * @weight:  [C_in, C_out, KH, KW] (PyTorch ConvTranspose2d layout)
 * @bias:    [C_out] or NULL
 * @stride:  Transposed convolution stride
 * @padding: Zero-padding on each side
 *
 * Returns [N, C_out, OH, OW] where OH = (H-1)*stride - 2*pad + KH.
 */
struct sam3_tensor *gh_conv_transpose2d(struct sam3_graph *g,
					struct sam3_arena *a,
					struct sam3_tensor *input,
					struct sam3_tensor *weight,
					struct sam3_tensor *bias,
					int stride, int padding);

/*
 * gh_maxpool2d - 2D max pooling.
 *
 * @input:       [N, C, H, W]
 * @kernel_size: Pooling window size
 * @stride:      Pooling stride
 *
 * Returns [N, C, OH, OW] where OH = (H - kernel) / stride + 1.
 */
struct sam3_tensor *gh_maxpool2d(struct sam3_graph *g, struct sam3_arena *a,
				struct sam3_tensor *input,
				int kernel_size, int stride);

#endif /* SAM3_MODEL_GRAPH_HELPERS_H */
