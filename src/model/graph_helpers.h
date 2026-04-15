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
 * gh_tensor_wrap - Create a tensor struct pointing to an external buffer.
 *
 * @arena:  Arena to allocate the tensor struct from
 * @dtype:  Data type for the tensor
 * @n_dims: Number of dimensions (1-4)
 * @dims:   Array of dimension sizes
 * @data:   External data buffer (not allocated from arena)
 *
 * Allocates only the sam3_tensor struct from the arena and sets its
 * data pointer to the provided buffer. No data allocation is performed.
 * Used for per-block ViT evaluation where the persistent buffer lives
 * in a separate arena from the scratch intermediates.
 * Returns NULL if the arena is full.
 */
struct sam3_tensor *gh_tensor_wrap(struct sam3_arena *arena,
				   enum sam3_dtype dtype,
				   int n_dims, const int *dims,
				   void *data);

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

/*
 * gh_load_mmap - Load a weight tensor by name, pointing data at the mmap region.
 *
 * Allocates only the sam3_tensor struct from the arena and sets its
 * data pointer directly into the weight file's mmap region. No memcpy.
 * Falls back to gh_alloc_tensor (zeroed) if wf is NULL or tensor not found.
 */
struct sam3_tensor *gh_load_mmap(const struct sam3_weight_file *wf,
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

struct sam3_tensor *gh_hswish(struct sam3_graph *g, struct sam3_arena *a,
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
 * gh_permute - General N-D axis permutation.
 *
 * @axes: Permutation of [0..n_dims-1]. For input of shape
 *        [d0, d1, ..., d_{n-1}], output has shape
 *        [d_axes[0], d_axes[1], ..., d_axes[n-1]].
 *
 * Example: axes=[0,2,1,3] on [1,seq,heads,hd] -> [1,heads,seq,hd].
 */
struct sam3_tensor *gh_permute(struct sam3_graph *g, struct sam3_arena *a,
			       struct sam3_tensor *input,
			       const int *axes);

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
 * gh_upsample - Nearest-neighbor upsampling for NHWC 4D tensors.
 *
 * @input: [N, H, W, C]
 * @scale: Integer upsample factor (same for H and W)
 *
 * Returns tensor [N, H*scale, W*scale, C].
 */
struct sam3_tensor *gh_upsample(struct sam3_graph *g, struct sam3_arena *a,
				struct sam3_tensor *input, int scale);

/*
 * gh_rope - Apply rotary position embedding.
 *
 * @input:  [batch, seq, n_heads, head_dim] tensor
 * @cos_f:  Precomputed cosine [seq, head_dim/2]
 * @sin_f:  Precomputed sine [seq, head_dim/2]
 * @grid_w: Grid width for 2D axial RoPE (0 = legacy scalar path)
 * @scale:  Position scale factor (e.g. window_size/grid_size for global)
 *
 * When grid_w > 0, the Metal backend uses mlx_fast_rope_dynamic with
 * per-axis offsets derived from the grid geometry instead of the
 * precomputed cos/sin tables. CPU backend always uses the tables.
 *
 * Returns rotated tensor, same shape as input.
 */
struct sam3_tensor *gh_rope(struct sam3_graph *g, struct sam3_arena *a,
			     struct sam3_tensor *input,
			     struct sam3_tensor *cos_f,
			     struct sam3_tensor *sin_f,
			     int grid_w, float scale);

/*
 * gh_sdpa - Fused scaled dot-product attention.
 *
 * Computes: softmax(Q @ K^T / sqrt(head_dim) + mask) @ V
 * as a single graph node. The backend implements this without
 * materializing the full attention score matrix.
 *
 * Accepts either single-head 2D or batched multi-head 4D inputs:
 *   2D: Q[seq_q, hd], K[seq_k, hd], V[seq_k, hd] -> [seq_q, hd]
 *   4D: Q[B, H, seq_q, hd], K[B, H, seq_k, hd], V[B, H, seq_k, hd]
 *       -> [B, H, seq_q, hd]
 *
 * @g:        Graph to add the SDPA node to
 * @a:        Arena for output tensor allocation
 * @Q:        Query tensor (2D or 4D)
 * @K:        Key tensor (same ndims as Q)
 * @V:        Value tensor (same ndims as Q)
 * @mask:     Additive attention mask [seq_q, seq_k], or NULL
 * @head_dim: Head dimension (used to compute scale = 1/sqrt(head_dim))
 *
 * Returns output tensor matching Q's shape, or NULL on error.
 */
struct sam3_tensor *gh_sdpa(struct sam3_graph *g, struct sam3_arena *a,
			    struct sam3_tensor *Q, struct sam3_tensor *K,
			    struct sam3_tensor *V, struct sam3_tensor *mask,
			    int head_dim);

/*
 * gh_window_partition - Reorder a flat patch grid into contiguous windows.
 *
 * Input is [n_patches, e] in natural row-major order
 * (idx = py * grid_size + px). Output is [n_windows, ws*ws, e] where
 * n_windows = (grid_size / ws)^2 and the inner ws*ws block holds the
 * patches of one window in row-major local order.
 *
 * Requires that grid_size is a multiple of ws (no padding).
 *
 * Implementation: 4-D reshape -> permute -> reshape, with cx folded
 * into the embedding dim to fit within SAM3_MAX_DIMS=4. The final
 * reshape materialises (one mlx_contiguous copy).
 */
struct sam3_tensor *gh_window_partition(struct sam3_graph *g,
					struct sam3_arena *a,
					struct sam3_tensor *x,
					int ws, int grid_size);

/*
 * gh_window_unpartition - Inverse of gh_window_partition.
 *
 * Takes [n_windows, ws*ws, e] and returns [n_patches, e] in natural
 * row-major order.
 */
struct sam3_tensor *gh_window_unpartition(struct sam3_graph *g,
					  struct sam3_arena *a,
					  struct sam3_tensor *x,
					  int ws, int grid_size);

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
 * @grid_w:    Grid width for 2D axial RoPE (0 = no fast path)
 * @rope_scale: Position scale for axial RoPE (ignored when grid_w=0)
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
	struct sam3_tensor *attn_mask,
	int grid_w, float rope_scale);

/*
 * gh_multihead_attention_rope_sep - MHA with separate Q/K/V weights and RoPE.
 *
 * Like gh_multihead_attention_rope but takes separate Q, K, V weight
 * and bias tensors instead of a packed [3*d_model, d_model] QKV tensor.
 * Avoids the fused QKV allocation during weight loading.
 *
 * @x:        Input [batch, seq, d_model]
 * @q_w:      Query weight [d_model, d_model]
 * @q_b:      Query bias [d_model] or NULL
 * @k_w:      Key weight [d_model, d_model]
 * @k_b:      Key bias [d_model] or NULL
 * @v_w:      Value weight [d_model, d_model]
 * @v_b:      Value bias [d_model] or NULL
 * @out_w:    Output projection weight [d_model, d_model]
 * @out_b:    Output projection bias [d_model] or NULL
 * @n_heads:  Number of attention heads
 * @rope_cos: Precomputed RoPE cosine [seq, head_dim/2] or NULL
 * @rope_sin: Precomputed RoPE sine [seq, head_dim/2] or NULL
 * @attn_mask: Additive attention mask [seq, seq] or NULL
 *
 * Returns output tensor [batch*seq, d_model].
 */
struct sam3_tensor *gh_multihead_attention_rope_sep(
	struct sam3_graph *g, struct sam3_arena *a,
	struct sam3_tensor *x,
	struct sam3_tensor *q_w, struct sam3_tensor *q_b,
	struct sam3_tensor *k_w, struct sam3_tensor *k_b,
	struct sam3_tensor *v_w, struct sam3_tensor *v_b,
	struct sam3_tensor *out_w,
	struct sam3_tensor *out_b,
	int n_heads,
	struct sam3_tensor *rope_cos,
	struct sam3_tensor *rope_sin,
	struct sam3_tensor *attn_mask,
	int grid_w, float rope_scale);

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
 * gh_multihead_attention_sep - MHA with separate Q/K/V weights (no RoPE).
 *
 * Convenience wrapper around gh_multihead_attention_rope_sep with
 * NULL RoPE and mask. Used by text encoder and encoder fusion.
 */
struct sam3_tensor *gh_multihead_attention_sep(
	struct sam3_graph *g, struct sam3_arena *a,
	struct sam3_tensor *x,
	struct sam3_tensor *q_w, struct sam3_tensor *q_b,
	struct sam3_tensor *k_w, struct sam3_tensor *k_b,
	struct sam3_tensor *v_w, struct sam3_tensor *v_b,
	struct sam3_tensor *out_w, struct sam3_tensor *out_b,
	int n_heads,
	struct sam3_tensor *attn_mask);

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
 * gh_cross_attention_sep - Cross-attention with separate K and V weights.
 *
 * Like gh_cross_attention but takes separate K and V weight/bias tensors
 * instead of a packed [2*d_model, d_model] KV tensor.
 */
struct sam3_tensor *gh_cross_attention_sep(
	struct sam3_graph *g, struct sam3_arena *arena,
	struct sam3_tensor *q_src,
	struct sam3_tensor *kv_src,
	struct sam3_tensor *q_w, struct sam3_tensor *q_b,
	struct sam3_tensor *k_w, struct sam3_tensor *k_b,
	struct sam3_tensor *v_w, struct sam3_tensor *v_b,
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
 * gh_groupnorm - Group normalization on NHWC tensor.
 *
 * @input:      [N, H, W, C] tensor
 * @gamma:      Per-channel scale [C], or NULL
 * @beta:       Per-channel bias [C], or NULL
 * @num_groups: Number of groups (C must be divisible)
 *
 * Returns a normalized tensor with the same shape as input.
 */
struct sam3_tensor *gh_groupnorm(struct sam3_graph *g, struct sam3_arena *a,
				 struct sam3_tensor *input,
				 struct sam3_tensor *gamma,
				 struct sam3_tensor *beta,
				 int num_groups);

/*
 * gh_batchnorm - Batch normalization (eval) with running statistics.
 *
 * @input:        NHWC tensor or any tensor with C as last dim
 * @gamma:        Per-channel scale [C], or NULL
 * @beta:         Per-channel bias [C], or NULL
 * @running_mean: Per-channel running mean [C]
 * @running_var:  Per-channel running variance [C]
 *
 * Returns a normalized tensor with the same shape as input.
 */
struct sam3_tensor *gh_batchnorm(struct sam3_graph *g, struct sam3_arena *a,
				 struct sam3_tensor *input,
				 struct sam3_tensor *gamma,
				 struct sam3_tensor *beta,
				 struct sam3_tensor *running_mean,
				 struct sam3_tensor *running_var);

/*
 * gh_conv2d - 2D convolution with NHWC input and OHWI weight.
 *
 * @input:   [N, H, W, C_in]
 * @weight:  [C_out, KH, KW, C_in/groups]
 * @bias:    [C_out] (optional, may be NULL)
 * @stride:  same stride for H and W
 * @padding: same padding for H and W
 * @groups:  number of groups (1 = standard conv, C_in = depthwise)
 *
 * Returns tensor [N, OH, OW, C_out].
 */
struct sam3_tensor *gh_conv2d(struct sam3_graph *g, struct sam3_arena *a,
			      struct sam3_tensor *input,
			      struct sam3_tensor *weight,
			      struct sam3_tensor *bias,
			      int stride, int padding, int groups);

/*
 * gh_conv_transpose2d - 2D transposed conv with NHWC input and OHWI
 * weight.
 *
 * @input:   [N, H, W, C_in]
 * @weight:  [C_out, KH, KW, C_in]
 * @bias:    [C_out] (optional, may be NULL)
 * @stride:  same stride for H and W
 * @padding: same padding for H and W
 *
 * Returns tensor [N, OH, OW, C_out] where
 * OH = (H - 1) * stride - 2 * padding + KH.
 */
struct sam3_tensor *gh_conv_transpose2d(struct sam3_graph *g,
					struct sam3_arena *a,
					struct sam3_tensor *input,
					struct sam3_tensor *weight,
					struct sam3_tensor *bias,
					int stride, int padding);

/*
 * gh_maxpool2d - Non-overlapping 2D max pooling on NHWC input.
 *
 * @input:       [N, H, W, C]
 * @kernel_size: Pooling window size (same for H and W)
 * @stride:      Pooling stride (same for H and W)
 *
 * Reshapes to [N, H/k, k, W/k, k, C] and reduces over the two k-axes
 * directly on the NHWC layout. Returns [N, H/k, W/k, C].
 */
struct sam3_tensor *gh_maxpool2d(struct sam3_graph *g, struct sam3_arena *a,
				 struct sam3_tensor *input,
				 int kernel_size, int stride);

#endif /* SAM3_MODEL_GRAPH_HELPERS_H */
