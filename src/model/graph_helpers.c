/*
 * src/model/graph_helpers.c - Helper functions for building compute graphs
 *
 * Implements convenience wrappers that allocate output tensors from the
 * arena and add nodes to the compute graph in a single call. These
 * helpers eliminate repetitive boilerplate in model construction code,
 * including composite patterns like linear layers, multi-head attention,
 * and MLPs.
 *
 * Key types:  (none -- uses sam3_tensor, sam3_graph, sam3_arena)
 * Depends on: graph_helpers.h, core/graph.h, core/alloc.h, core/tensor.h, core/weight.h
 * Used by:    model/ files (vitdet.c, text_encoder.c, decoder.c, etc.)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <string.h>
#include "util/log.h"

#include "graph_helpers.h"

/* ── Tensor allocation ───────────────────────────────────────────── */

struct sam3_tensor *gh_alloc_tensor(struct sam3_arena *arena,
				    enum sam3_dtype dtype,
				    int n_dims, const int *dims)
{
	struct sam3_tensor *t = (struct sam3_tensor *)
		sam3_arena_alloc(arena, sizeof(struct sam3_tensor));
	if (!t)
		return NULL;

	memset(t, 0, sizeof(*t));
	t->dtype = dtype;
	t->n_dims = n_dims;
	for (int i = 0; i < n_dims; i++)
		t->dims[i] = dims[i];

	sam3_tensor_compute_strides(t);

	int nelems = sam3_tensor_nelems(t);
	t->nbytes = (size_t)nelems * sam3_dtype_size(dtype);

	t->data = sam3_arena_alloc_raw(arena, t->nbytes);
	if (!t->data)
		return NULL;

	return t;
}

struct sam3_tensor *gh_tensor_wrap(struct sam3_arena *arena,
				   enum sam3_dtype dtype,
				   int n_dims, const int *dims,
				   void *data)
{
	struct sam3_tensor *t = (struct sam3_tensor *)
		sam3_arena_alloc(arena, sizeof(struct sam3_tensor));
	if (!t)
		return NULL;

	memset(t, 0, sizeof(*t));
	t->dtype = dtype;
	t->n_dims = n_dims;
	for (int i = 0; i < n_dims; i++)
		t->dims[i] = dims[i];

	sam3_tensor_compute_strides(t);

	t->nbytes = (size_t)sam3_tensor_nelems(t) * sam3_dtype_size(dtype);
	t->data = data;
	t->ephemeral = 1;

	return t;
}

struct sam3_tensor *gh_load_or_alloc(const struct sam3_weight_file *wf,
				      const char *name,
				      struct sam3_arena *arena,
				      enum sam3_dtype dtype,
				      int n_dims, const int *dims)
{
	if (wf) {
		const struct sam3_weight_tensor_desc *desc;
		desc = sam3_weight_find(wf, name);
		if (desc) {
			struct sam3_tensor *t;
			void *arena_buf;

			t = gh_alloc_tensor(arena, dtype, n_dims, dims);
			if (!t)
				return NULL;

			/*
			 * Save the arena data pointer, then let
			 * sam3_weight_to_tensor redirect t->data to
			 * the mmap region. Copy from mmap into the
			 * arena buffer so the tensor survives
			 * sam3_weight_close (which unmaps the file).
			 */
			arena_buf = t->data;
			sam3_weight_to_tensor(wf, desc, t);
			memcpy(arena_buf, t->data, t->nbytes);
			t->data = arena_buf;
			return t;
		}
	}
	sam3_log_warn("weight not found: %s (using zeros)", name);
	return gh_alloc_tensor(arena, dtype, n_dims, dims);
}

struct sam3_tensor *gh_load_mmap(const struct sam3_weight_file *wf,
				  const char *name,
				  struct sam3_arena *arena,
				  enum sam3_dtype dtype,
				  int n_dims, const int *dims)
{
	if (wf) {
		const struct sam3_weight_tensor_desc *desc;
		desc = sam3_weight_find(wf, name);
		if (desc) {
			struct sam3_tensor *t = (struct sam3_tensor *)
				sam3_arena_alloc(arena,
						 sizeof(struct sam3_tensor));
			if (!t)
				return NULL;

			memset(t, 0, sizeof(*t));
			sam3_weight_to_tensor(wf, desc, t);
			return t;
		}
	}
	sam3_log_warn("weight not found: %s (using zeros)", name);
	return gh_alloc_tensor(arena, dtype, n_dims, dims);
}

/* ── Unary activation ops ────────────────────────────────────────── */

static struct sam3_tensor *gh_unary(struct sam3_graph *g,
				    struct sam3_arena *a,
				    enum sam3_op op,
				    struct sam3_tensor *input)
{
	struct sam3_tensor *out = gh_alloc_tensor(a, input->dtype,
						  input->n_dims, input->dims);
	if (!out)
		return NULL;

	return sam3_graph_add_op(g, op,
				 (struct sam3_tensor *[]){input}, 1, out);
}

struct sam3_tensor *gh_gelu(struct sam3_graph *g, struct sam3_arena *a,
			    struct sam3_tensor *input)
{
	return gh_unary(g, a, SAM3_OP_GELU, input);
}

struct sam3_tensor *gh_relu(struct sam3_graph *g, struct sam3_arena *a,
			    struct sam3_tensor *input)
{
	return gh_unary(g, a, SAM3_OP_RELU, input);
}

struct sam3_tensor *gh_sigmoid(struct sam3_graph *g, struct sam3_arena *a,
			       struct sam3_tensor *input)
{
	return gh_unary(g, a, SAM3_OP_SIGMOID, input);
}

struct sam3_tensor *gh_silu(struct sam3_graph *g, struct sam3_arena *a,
			    struct sam3_tensor *input)
{
	return gh_unary(g, a, SAM3_OP_SILU, input);
}

struct sam3_tensor *gh_softmax(struct sam3_graph *g, struct sam3_arena *a,
			       struct sam3_tensor *input)
{
	return gh_unary(g, a, SAM3_OP_SOFTMAX, input);
}

/* ── Binary element-wise ops ─────────────────────────────────────── */

struct sam3_tensor *gh_add(struct sam3_graph *g, struct sam3_arena *a,
			   struct sam3_tensor *x, struct sam3_tensor *b)
{
	struct sam3_tensor *out = gh_alloc_tensor(a, x->dtype,
						  x->n_dims, x->dims);
	if (!out)
		return NULL;

	return sam3_graph_add_op(g, SAM3_OP_ADD,
				 (struct sam3_tensor *[]){x, b}, 2, out);
}

struct sam3_tensor *gh_mul(struct sam3_graph *g, struct sam3_arena *a,
			   struct sam3_tensor *x, struct sam3_tensor *b)
{
	struct sam3_tensor *out = gh_alloc_tensor(a, x->dtype,
						  x->n_dims, x->dims);
	if (!out)
		return NULL;

	return sam3_graph_add_op(g, SAM3_OP_MUL,
				 (struct sam3_tensor *[]){x, b}, 2, out);
}

/* ── Matrix multiply ─────────────────────────────────────────────── */

struct sam3_tensor *gh_matmul(struct sam3_graph *g, struct sam3_arena *a,
			      struct sam3_tensor *x, struct sam3_tensor *w)
{
	/* Output dims: all dims from x except last, last = w->dims[last] */
	int out_dims[SAM3_MAX_DIMS];
	int nd = x->n_dims;

	for (int i = 0; i < nd - 1; i++)
		out_dims[i] = x->dims[i];
	out_dims[nd - 1] = w->dims[w->n_dims - 1];

	struct sam3_tensor *out = gh_alloc_tensor(a, x->dtype, nd, out_dims);
	if (!out)
		return NULL;

	return sam3_graph_add_op(g, SAM3_OP_MATMUL,
				 (struct sam3_tensor *[]){x, w}, 2, out);
}

/* ── Linear layer ────────────────────────────────────────────────── */

struct sam3_tensor *gh_linear(struct sam3_graph *g, struct sam3_arena *a,
			      struct sam3_tensor *input,
			      struct sam3_tensor *weight,
			      struct sam3_tensor *bias)
{
	/* Transpose weight: swap last two dims */
	struct sam3_tensor *wt = gh_transpose(g, a, weight);
	if (!wt)
		return NULL;

	/* Matmul: input @ weight^T */
	struct sam3_tensor *out = gh_matmul(g, a, input, wt);
	if (!out)
		return NULL;

	/* Add bias if present */
	if (bias) {
		out = gh_add(g, a, out, bias);
		if (!out)
			return NULL;
	}

	return out;
}

/* ── Layer normalization ─────────────────────────────────────────── */

struct sam3_tensor *gh_layernorm(struct sam3_graph *g, struct sam3_arena *a,
				struct sam3_tensor *input,
				struct sam3_tensor *gamma,
				struct sam3_tensor *beta)
{
	struct sam3_tensor *out = gh_alloc_tensor(a, input->dtype,
						  input->n_dims, input->dims);
	if (!out)
		return NULL;

	if (gamma && beta) {
		return sam3_graph_add_op(g, SAM3_OP_LAYERNORM,
			(struct sam3_tensor *[]){input, gamma, beta},
			3, out);
	} else if (gamma) {
		return sam3_graph_add_op(g, SAM3_OP_LAYERNORM,
			(struct sam3_tensor *[]){input, gamma},
			2, out);
	} else {
		return sam3_graph_add_op(g, SAM3_OP_LAYERNORM,
			(struct sam3_tensor *[]){input},
			1, out);
	}
}

/* ── Reshape ─────────────────────────────────────────────────────── */

struct sam3_tensor *gh_reshape(struct sam3_graph *g, struct sam3_arena *a,
			       struct sam3_tensor *input,
			       int n_dims, const int *dims)
{
	struct sam3_tensor *out = gh_alloc_tensor(a, input->dtype,
						   n_dims, dims);
	if (!out)
		return NULL;

	return sam3_graph_add_op(g, SAM3_OP_RESHAPE,
				 (struct sam3_tensor *[]){input}, 1, out);
}

/* ── Transpose ───────────────────────────────────────────────────── */

struct sam3_tensor *gh_transpose(struct sam3_graph *g, struct sam3_arena *a,
				 struct sam3_tensor *input)
{
	int out_dims[SAM3_MAX_DIMS];
	int nd = input->n_dims;

	for (int i = 0; i < nd; i++)
		out_dims[i] = input->dims[i];

	/* Swap last two dimensions */
	out_dims[nd - 2] = input->dims[nd - 1];
	out_dims[nd - 1] = input->dims[nd - 2];

	struct sam3_tensor *out = gh_alloc_tensor(a, input->dtype,
						  nd, out_dims);
	if (!out)
		return NULL;

	struct sam3_tensor *result = sam3_graph_add_op(
		g, SAM3_OP_TRANSPOSE,
		(struct sam3_tensor *[]){input}, 1, out);
	if (!result)
		return NULL;

	/* Set explicit axes: swap last two dims. */
	struct sam3_node *node = &g->nodes[g->n_nodes - 1];
	for (int i = 0; i < nd; i++)
		node->params[i] = i;
	node->params[nd - 2] = nd - 1;
	node->params[nd - 1] = nd - 2;
	return result;
}

struct sam3_tensor *gh_permute(struct sam3_graph *g, struct sam3_arena *a,
			       struct sam3_tensor *input,
			       const int *axes)
{
	int out_dims[SAM3_MAX_DIMS];
	int nd = input->n_dims;

	if (nd > 4) {
		sam3_log_error("gh_permute: n_dims > 4 not supported");
		return NULL;
	}

	for (int i = 0; i < nd; i++)
		out_dims[i] = input->dims[axes[i]];

	struct sam3_tensor *out = gh_alloc_tensor(a, input->dtype,
						  nd, out_dims);
	if (!out)
		return NULL;

	struct sam3_tensor *result = sam3_graph_add_op(
		g, SAM3_OP_TRANSPOSE,
		(struct sam3_tensor *[]){input}, 1, out);
	if (!result)
		return NULL;

	struct sam3_node *node = &g->nodes[g->n_nodes - 1];
	for (int i = 0; i < nd; i++)
		node->params[i] = axes[i];
	return result;
}

/* ── Window partition / unpartition ──────────────────────────────── */

struct sam3_tensor *gh_window_partition(struct sam3_graph *g,
					struct sam3_arena *a,
					struct sam3_tensor *x,
					int ws, int grid_size)
{
	int e = x->dims[1];
	int nw = grid_size / ws;	/* windows per row/col */

	/*
	 * View [np, e] as [wy, cy, wx, cx*e] (4-D, contiguous view).
	 * Splitting np into (wy, cy, wx) requires that each is
	 * contiguous in memory: yes, because np is laid out as
	 * py * grid_size + px and we are splitting py = wy*ws + cy
	 * and px = wx*ws + cx with cx folded into the inner cx*e.
	 */
	int v1[] = {nw, ws, nw, ws * e};
	struct sam3_tensor *t1 = gh_reshape(g, a, x, 4, v1);
	if (!t1)
		return NULL;

	/* Permute (wy, cy, wx, cx*e) -> (wy, wx, cy, cx*e) */
	int perm[] = {0, 2, 1, 3};
	struct sam3_tensor *t2 = gh_permute(g, a, t1, perm);
	if (!t2)
		return NULL;

	/* Flatten (wy, wx) -> n_win, (cy, cx*e) -> (ws*ws, e). */
	int n_win = nw * nw;
	int win_pos = ws * ws;
	int v3[] = {n_win, win_pos, e};
	struct sam3_tensor *out = gh_reshape(g, a, t2, 3, v3);
	if (!out)
		return NULL;

	return out;
}

struct sam3_tensor *gh_window_unpartition(struct sam3_graph *g,
					  struct sam3_arena *a,
					  struct sam3_tensor *x,
					  int ws, int grid_size)
{
	int e = x->dims[2];
	int nw = grid_size / ws;

	/* View [n_win, ws*ws, e] as [wy, wx, cy, cx*e]. */
	int v1[] = {nw, nw, ws, ws * e};
	struct sam3_tensor *t1 = gh_reshape(g, a, x, 4, v1);
	if (!t1)
		return NULL;

	/* Permute (wy, wx, cy, cx*e) -> (wy, cy, wx, cx*e) */
	int perm[] = {0, 2, 1, 3};
	struct sam3_tensor *t2 = gh_permute(g, a, t1, perm);
	if (!t2)
		return NULL;

	/* Flatten back to [np, e]. */
	int np = grid_size * grid_size;
	int v3[] = {np, e};
	struct sam3_tensor *out = gh_reshape(g, a, t2, 2, v3);
	if (!out)
		return NULL;

	return out;
}

/* ── Concat ──────────────────────────────────────────────────────── */

/*
 * gh_concat_small - Concatenate up to SAM3_NODE_MAX_INPUTS tensors.
 */
static struct sam3_tensor *gh_concat_small(struct sam3_graph *g,
					    struct sam3_arena *a,
					    struct sam3_tensor **tensors,
					    int n, int axis)
{
	int out_dims[SAM3_MAX_DIMS];
	int nd = tensors[0]->n_dims;

	for (int i = 0; i < nd; i++)
		out_dims[i] = tensors[0]->dims[i];

	int total = 0;
	for (int i = 0; i < n; i++)
		total += tensors[i]->dims[axis];
	out_dims[axis] = total;

	struct sam3_tensor *out = gh_alloc_tensor(a, tensors[0]->dtype,
						  nd, out_dims);
	if (!out)
		return NULL;

	struct sam3_tensor *result = sam3_graph_add_op(g, SAM3_OP_CONCAT,
						       tensors, n, out);
	if (!result)
		return NULL;

	g->nodes[g->n_nodes - 1].params[0] = axis;
	return result;
}

struct sam3_tensor *gh_concat(struct sam3_graph *g, struct sam3_arena *a,
			      struct sam3_tensor **tensors, int n, int axis)
{
	/*
	 * If n fits in one node, emit directly. Otherwise, chain
	 * multiple concat ops in chunks of SAM3_NODE_MAX_INPUTS.
	 */
	if (n <= SAM3_NODE_MAX_INPUTS)
		return gh_concat_small(g, a, tensors, n, axis);

	struct sam3_tensor *chunks[SAM3_NODE_MAX_INPUTS];
	int n_chunks = 0;
	int i = 0;

	while (i < n) {
		int chunk_size = n - i;
		if (chunk_size > SAM3_NODE_MAX_INPUTS)
			chunk_size = SAM3_NODE_MAX_INPUTS;

		struct sam3_tensor *chunk;
		if (chunk_size == 1) {
			chunk = tensors[i];
		} else {
			chunk = gh_concat_small(g, a, &tensors[i],
						 chunk_size, axis);
			if (!chunk)
				return NULL;
		}

		chunks[n_chunks++] = chunk;
		i += chunk_size;
	}

	if (n_chunks == 1)
		return chunks[0];

	return gh_concat_small(g, a, chunks, n_chunks, axis);
}

/* ── Slice ───────────────────────────────────────────────────────── */

struct sam3_tensor *gh_slice(struct sam3_graph *g, struct sam3_arena *a,
			     struct sam3_tensor *input,
			     int axis, int start, int end)
{
	int out_dims[SAM3_MAX_DIMS];
	int nd = input->n_dims;

	for (int i = 0; i < nd; i++)
		out_dims[i] = input->dims[i];
	out_dims[axis] = end - start;

	struct sam3_tensor *out = gh_alloc_tensor(a, input->dtype,
						  nd, out_dims);
	if (!out)
		return NULL;

	struct sam3_tensor *result = sam3_graph_add_op(g, SAM3_OP_SLICE,
		(struct sam3_tensor *[]){input}, 1, out);
	if (!result)
		return NULL;

	/* Set slice params: axis, start, end */
	g->nodes[g->n_nodes - 1].params[0] = axis;
	g->nodes[g->n_nodes - 1].params[1] = start;
	g->nodes[g->n_nodes - 1].params[2] = end;
	return result;
}

/* ── Embedding ───────────────────────────────────────────────────── */

struct sam3_tensor *gh_embed(struct sam3_graph *g, struct sam3_arena *a,
			     struct sam3_tensor *table,
			     struct sam3_tensor *indices)
{
	int n_idx = sam3_tensor_nelems(indices);
	int embed_dim = table->dims[1];
	int out_dims[] = {n_idx, embed_dim};

	struct sam3_tensor *out = gh_alloc_tensor(a, table->dtype,
						  2, out_dims);
	if (!out)
		return NULL;

	return sam3_graph_add_op(g, SAM3_OP_EMBED,
		(struct sam3_tensor *[]){table, indices}, 2, out);
}

/* ── Upsample ────────────────────────────────────────────────────── */

struct sam3_tensor *gh_upsample(struct sam3_graph *g, struct sam3_arena *a,
				struct sam3_tensor *input, int scale)
{
	/* Input: [N, C, H, W] -> Output: [N, C, H*scale, W*scale] */
	int out_dims[4] = {
		input->dims[0],
		input->dims[1],
		input->dims[2] * scale,
		input->dims[3] * scale,
	};

	struct sam3_tensor *out = gh_alloc_tensor(a, input->dtype,
						  4, out_dims);
	if (!out)
		return NULL;

	struct sam3_tensor *result = sam3_graph_add_op(g, SAM3_OP_UPSAMPLE,
		(struct sam3_tensor *[]){input}, 1, out);
	if (!result)
		return NULL;

	g->nodes[g->n_nodes - 1].params[0] = scale;
	return result;
}

struct sam3_tensor *gh_upsample_nhwc(struct sam3_graph *g,
				     struct sam3_arena *a,
				     struct sam3_tensor *input, int scale)
{
	/*
	 * Input: [N, H, W, C] -> Output: [N, H*scale, W*scale, C]
	 *
	 * Slot usage for SAM3_OP_UPSAMPLE:
	 *   params[0] = scale (shared with NCHW path)
	 *   params[1] = NHWC flag (0 = legacy NCHW, 1 = NHWC)
	 * Slots 2 and 3 remain unused.
	 */
	int out_dims[4] = {
		input->dims[0],
		input->dims[1] * scale,
		input->dims[2] * scale,
		input->dims[3],
	};

	struct sam3_tensor *out = gh_alloc_tensor(a, input->dtype,
						  4, out_dims);
	if (!out)
		return NULL;

	struct sam3_tensor *result = sam3_graph_add_op(g, SAM3_OP_UPSAMPLE,
		(struct sam3_tensor *[]){input}, 1, out);
	if (!result)
		return NULL;

	struct sam3_node *node = &g->nodes[g->n_nodes - 1];
	node->params[0] = scale;
	node->params[1] = 1;  /* NHWC flag */
	return result;
}

/* ── Rotary position embedding ───────────────────────────────────── */

struct sam3_tensor *gh_rope(struct sam3_graph *g, struct sam3_arena *a,
			     struct sam3_tensor *input,
			     struct sam3_tensor *cos_f,
			     struct sam3_tensor *sin_f)
{
	struct sam3_tensor *out = gh_alloc_tensor(a, input->dtype,
						  input->n_dims, input->dims);
	if (!out)
		return NULL;

	struct sam3_tensor *result = sam3_graph_add_op(g, SAM3_OP_ROPE,
		(struct sam3_tensor *[]){input, cos_f, sin_f}, 3, out);
	if (!result)
		return NULL;

	/* params[0] = head_dim (last dim of 4D input) */
	g->nodes[g->n_nodes - 1].params[0] = input->dims[3];
	return result;
}

/* ── Fused scaled dot-product attention ─────────────────────────── */

struct sam3_tensor *gh_sdpa(struct sam3_graph *g, struct sam3_arena *a,
			    struct sam3_tensor *Q, struct sam3_tensor *K,
			    struct sam3_tensor *V, struct sam3_tensor *mask,
			    int head_dim)
{
	struct sam3_tensor *inputs[4];
	int n_inputs = 3;
	struct sam3_tensor *output;

	inputs[0] = Q;
	inputs[1] = K;
	inputs[2] = V;
	if (mask) {
		inputs[3] = mask;
		n_inputs = 4;
	}

	/* Output has same shape as Q: [seq, hd] or [batch, n_heads, seq, hd] */
	output = gh_alloc_tensor(a, Q->dtype, Q->n_dims, Q->dims);
	if (!output)
		return NULL;

	output = sam3_graph_add_op(g, SAM3_OP_SDPA, inputs,
				   n_inputs, output);
	if (!output)
		return NULL;

	g->nodes[g->n_nodes - 1].params[0] = head_dim;
	return output;
}

/* ── Multi-head attention ────────────────────────────────────────── */

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
	struct sam3_tensor *attn_mask)
{
	int batch = q->dims[0];
	int seq_q = q->dims[1];
	int d_model = q->dims[2];
	int head_dim = d_model / n_heads;
	int bs = batch * seq_q;

	(void)k;
	(void)v;

	/* Flatten to 2D and project QKV */
	int flat_dims[] = {bs, d_model};
	struct sam3_tensor *qf = gh_reshape(g, a, q, 2, flat_dims);
	if (!qf) {
		sam3_log_error("mha: reshape fail (arena %zu/%zu)",
			       a->offset, a->size);
		return NULL;
	}

	struct sam3_tensor *qkv = gh_linear(g, a, qf, qkv_w, qkv_b);
	if (!qkv) {
		sam3_log_error("mha: qkv linear fail (arena %zu/%zu)",
			       a->offset, a->size);
		return NULL;
	}

	/* Slice into Q, K, V */
	struct sam3_tensor *sq = gh_slice(g, a, qkv, 1, 0, d_model);
	struct sam3_tensor *sk = gh_slice(g, a, qkv, 1,
					  d_model, 2 * d_model);
	struct sam3_tensor *sv = gh_slice(g, a, qkv, 1,
					  2 * d_model, 3 * d_model);
	if (!sq || !sk || !sv) {
		sam3_log_error("mha: QKV slice fail (arena %zu/%zu)",
			       a->offset, a->size);
		return NULL;
	}

	/* Apply RoPE to Q and K if requested */
	if (rope_cos && rope_sin) {
		int rope_4d[] = {batch, seq_q, n_heads, head_dim};

		struct sam3_tensor *sq_4d;
		sq_4d = gh_reshape(g, a, sq, 4, rope_4d);
		if (!sq_4d)
			return NULL;
		sq_4d = gh_rope(g, a, sq_4d, rope_cos, rope_sin);
		if (!sq_4d)
			return NULL;
		sq = gh_reshape(g, a, sq_4d, 2, flat_dims);
		if (!sq)
			return NULL;

		struct sam3_tensor *sk_4d;
		sk_4d = gh_reshape(g, a, sk, 4, rope_4d);
		if (!sk_4d)
			return NULL;
		sk_4d = gh_rope(g, a, sk_4d, rope_cos, rope_sin);
		if (!sk_4d)
			return NULL;
		sk = gh_reshape(g, a, sk_4d, 2, flat_dims);
		if (!sk) {
			sam3_log_error("mha: rope fail (arena %zu/%zu)",
				       a->offset, a->size);
			return NULL;
		}
	}

	sam3_log_debug("mha: pre-SDPA arena %zu/%zu, n_heads=%d",
		       a->offset, a->size, n_heads);

	/*
	 * Batched multi-head SDPA.
	 *
	 * Input sq/sk/sv are [bs, d_model] where d_model = n_heads*head_dim
	 * in row-major layout: for each seq row, the channels are laid out
	 * as [h0_d0..h0_d{hd-1}, h1_d0..h1_d{hd-1}, ..., h{H-1}_d{hd-1}].
	 *
	 * MLX SDPA expects [B, H, S, D]. A naive 4D reshape is WRONG because
	 * it would place seq rows as the outer axis within each head. We
	 * need to physically transpose the seq/head axes:
	 *
	 *   [bs, d_model]
	 *     -> [seq_q, n_heads, head_dim]   (pure reshape, split last dim)
	 *     -> permute(1,0,2) -> [n_heads, seq_q, head_dim]
	 *     -> [1, n_heads, seq_q, head_dim] (pure reshape, unsqueeze)
	 *
	 * This assumes batch==1, which matches all current callers.
	 */
	int qkv_3d[]     = {seq_q, n_heads, head_dim};
	int qkv_4d[]     = {batch, n_heads, seq_q, head_dim};
	int perm_sh[]    = {1, 0, 2};

	struct sam3_tensor *sq_3d = gh_reshape(g, a, sq, 3, qkv_3d);
	struct sam3_tensor *sk_3d = gh_reshape(g, a, sk, 3, qkv_3d);
	struct sam3_tensor *sv_3d = gh_reshape(g, a, sv, 3, qkv_3d);
	if (!sq_3d || !sk_3d || !sv_3d) {
		sam3_log_error("mha: 3D reshape fail (arena %zu/%zu)",
			       a->offset, a->size);
		return NULL;
	}

	struct sam3_tensor *sq_hs = gh_permute(g, a, sq_3d, perm_sh);
	struct sam3_tensor *sk_hs = gh_permute(g, a, sk_3d, perm_sh);
	struct sam3_tensor *sv_hs = gh_permute(g, a, sv_3d, perm_sh);
	if (!sq_hs || !sk_hs || !sv_hs) {
		sam3_log_error("mha: permute fail (arena %zu/%zu)",
			       a->offset, a->size);
		return NULL;
	}

	struct sam3_tensor *sq_4d = gh_reshape(g, a, sq_hs, 4, qkv_4d);
	struct sam3_tensor *sk_4d = gh_reshape(g, a, sk_hs, 4, qkv_4d);
	struct sam3_tensor *sv_4d = gh_reshape(g, a, sv_hs, 4, qkv_4d);
	if (!sq_4d || !sk_4d || !sv_4d) {
		sam3_log_error("mha: 4D reshape fail (arena %zu/%zu)",
			       a->offset, a->size);
		return NULL;
	}

	struct sam3_tensor *attn_4d = gh_sdpa(g, a, sq_4d, sk_4d, sv_4d,
					       attn_mask, head_dim);
	if (!attn_4d) {
		sam3_log_error("mha: batched SDPA fail "
			       "(arena %zu/%zu, nodes %d/%d)",
			       a->offset, a->size,
			       g->n_nodes, SAM3_GRAPH_MAX_NODES);
		return NULL;
	}

	/*
	 * Reshape back: [1, n_heads, seq_q, head_dim]
	 *   -> [n_heads, seq_q, head_dim]        (drop batch)
	 *   -> permute(1,0,2) -> [seq_q, n_heads, head_dim]
	 *   -> [bs, d_model]                     (flatten last dim)
	 */
	int attn_3d_hs[] = {n_heads, seq_q, head_dim};
	struct sam3_tensor *attn_hs = gh_reshape(g, a, attn_4d,
						  3, attn_3d_hs);
	if (!attn_hs) {
		sam3_log_error("mha: post-SDPA 3D reshape fail "
			       "(arena %zu/%zu)",
			       a->offset, a->size);
		return NULL;
	}

	struct sam3_tensor *attn_sh = gh_permute(g, a, attn_hs, perm_sh);
	if (!attn_sh) {
		sam3_log_error("mha: post-SDPA permute fail "
			       "(arena %zu/%zu)",
			       a->offset, a->size);
		return NULL;
	}

	struct sam3_tensor *merged = gh_reshape(g, a, attn_sh, 2, flat_dims);
	if (!merged) {
		sam3_log_error("mha: merge reshape fail (arena %zu/%zu)",
			       a->offset, a->size);
		return NULL;
	}

	sam3_log_debug("mha: post-SDPA arena %zu/%zu", a->offset, a->size);

	struct sam3_tensor *out = gh_linear(g, a, merged, out_w, out_b);
	if (!out) {
		sam3_log_error("mha: out_proj fail (arena %zu/%zu)",
			       a->offset, a->size);
	}
	return out;
}

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
	struct sam3_tensor *attn_mask)
{
	int batch = x->dims[0];
	int seq = x->dims[1];
	int d_model = x->dims[2];
	int head_dim = d_model / n_heads;
	int bs = batch * seq;

	/* Flatten to 2D for linear projections */
	int flat_dims[] = {bs, d_model};
	struct sam3_tensor *xf = gh_reshape(g, a, x, 2, flat_dims);
	if (!xf) {
		sam3_log_error("mha_sep: reshape fail (arena %zu/%zu)",
			       a->offset, a->size);
		return NULL;
	}

	/* Separate Q, K, V projections */
	struct sam3_tensor *sq = gh_linear(g, a, xf, q_w, q_b);
	struct sam3_tensor *sk = gh_linear(g, a, xf, k_w, k_b);
	struct sam3_tensor *sv = gh_linear(g, a, xf, v_w, v_b);
	if (!sq || !sk || !sv) {
		sam3_log_error("mha_sep: QKV linear fail (arena %zu/%zu)",
			       a->offset, a->size);
		return NULL;
	}

	/* Apply RoPE to Q and K if requested */
	if (rope_cos && rope_sin) {
		int rope_4d[] = {batch, seq, n_heads, head_dim};

		struct sam3_tensor *sq_4d;
		sq_4d = gh_reshape(g, a, sq, 4, rope_4d);
		if (!sq_4d)
			return NULL;
		sq_4d = gh_rope(g, a, sq_4d, rope_cos, rope_sin);
		if (!sq_4d)
			return NULL;
		sq = gh_reshape(g, a, sq_4d, 2, flat_dims);
		if (!sq)
			return NULL;

		struct sam3_tensor *sk_4d;
		sk_4d = gh_reshape(g, a, sk, 4, rope_4d);
		if (!sk_4d)
			return NULL;
		sk_4d = gh_rope(g, a, sk_4d, rope_cos, rope_sin);
		if (!sk_4d)
			return NULL;
		sk = gh_reshape(g, a, sk_4d, 2, flat_dims);
		if (!sk) {
			sam3_log_error("mha_sep: rope fail "
				       "(arena %zu/%zu)",
				       a->offset, a->size);
			return NULL;
		}
	}

	sam3_log_debug("mha_sep: pre-SDPA arena %zu/%zu, n_heads=%d",
		       a->offset, a->size, n_heads);

	/*
	 * Batched multi-head SDPA. See gh_multihead_attention_rope for the
	 * full explanation of why a direct 4D reshape is incorrect: the
	 * seq/head axes must be physically transposed, not just relabeled.
	 */
	int qkv_3d[]     = {seq, n_heads, head_dim};
	int qkv_4d[]     = {batch, n_heads, seq, head_dim};
	int perm_sh[]    = {1, 0, 2};

	struct sam3_tensor *sq_3d = gh_reshape(g, a, sq, 3, qkv_3d);
	struct sam3_tensor *sk_3d = gh_reshape(g, a, sk, 3, qkv_3d);
	struct sam3_tensor *sv_3d = gh_reshape(g, a, sv, 3, qkv_3d);
	if (!sq_3d || !sk_3d || !sv_3d) {
		sam3_log_error("mha_sep: 3D reshape fail "
			       "(arena %zu/%zu)",
			       a->offset, a->size);
		return NULL;
	}

	struct sam3_tensor *sq_hs = gh_permute(g, a, sq_3d, perm_sh);
	struct sam3_tensor *sk_hs = gh_permute(g, a, sk_3d, perm_sh);
	struct sam3_tensor *sv_hs = gh_permute(g, a, sv_3d, perm_sh);
	if (!sq_hs || !sk_hs || !sv_hs) {
		sam3_log_error("mha_sep: permute fail (arena %zu/%zu)",
			       a->offset, a->size);
		return NULL;
	}

	struct sam3_tensor *sq_4d = gh_reshape(g, a, sq_hs, 4, qkv_4d);
	struct sam3_tensor *sk_4d = gh_reshape(g, a, sk_hs, 4, qkv_4d);
	struct sam3_tensor *sv_4d = gh_reshape(g, a, sv_hs, 4, qkv_4d);
	if (!sq_4d || !sk_4d || !sv_4d) {
		sam3_log_error("mha_sep: 4D reshape fail "
			       "(arena %zu/%zu)",
			       a->offset, a->size);
		return NULL;
	}

	struct sam3_tensor *attn_4d = gh_sdpa(g, a, sq_4d, sk_4d, sv_4d,
					       attn_mask, head_dim);
	if (!attn_4d) {
		sam3_log_error("mha_sep: batched SDPA fail "
			       "(arena %zu/%zu, nodes %d/%d)",
			       a->offset, a->size,
			       g->n_nodes, SAM3_GRAPH_MAX_NODES);
		return NULL;
	}

	int attn_3d_hs[] = {n_heads, seq, head_dim};
	struct sam3_tensor *attn_hs = gh_reshape(g, a, attn_4d,
						  3, attn_3d_hs);
	if (!attn_hs) {
		sam3_log_error("mha_sep: post-SDPA 3D reshape fail "
			       "(arena %zu/%zu)",
			       a->offset, a->size);
		return NULL;
	}

	struct sam3_tensor *attn_sh = gh_permute(g, a, attn_hs, perm_sh);
	if (!attn_sh) {
		sam3_log_error("mha_sep: post-SDPA permute fail "
			       "(arena %zu/%zu)",
			       a->offset, a->size);
		return NULL;
	}

	struct sam3_tensor *merged = gh_reshape(g, a, attn_sh, 2, flat_dims);
	if (!merged) {
		sam3_log_error("mha_sep: merge reshape fail "
			       "(arena %zu/%zu)",
			       a->offset, a->size);
		return NULL;
	}

	sam3_log_debug("mha_sep: post-SDPA arena %zu/%zu",
		       a->offset, a->size);

	struct sam3_tensor *out = gh_linear(g, a, merged, out_w, out_b);
	if (!out) {
		sam3_log_error("mha_sep: out_proj fail (arena %zu/%zu)",
			       a->offset, a->size);
	}
	return out;
}

struct sam3_tensor *gh_multihead_attention(
	struct sam3_graph *g, struct sam3_arena *a,
	struct sam3_tensor *q,
	struct sam3_tensor *k,
	struct sam3_tensor *v,
	struct sam3_tensor *qkv_w,
	struct sam3_tensor *qkv_b,
	struct sam3_tensor *out_w,
	struct sam3_tensor *out_b,
	int n_heads)
{
	return gh_multihead_attention_rope(g, a, q, k, v,
		qkv_w, qkv_b, out_w, out_b,
		n_heads, NULL, NULL, NULL);
}

/* ── Cross-attention ─────────────────────────────────────────────── */

struct sam3_tensor *gh_cross_attention(
	struct sam3_graph *g, struct sam3_arena *arena,
	struct sam3_tensor *q_src,
	struct sam3_tensor *kv_src,
	struct sam3_tensor *q_w, struct sam3_tensor *q_b,
	struct sam3_tensor *kv_w, struct sam3_tensor *kv_b,
	struct sam3_tensor *out_w, struct sam3_tensor *out_b,
	int n_heads)
{
	int d_model = q_src->dims[1];
	int head_dim = d_model / n_heads;

	/* Project Q: [n_q, d_model] */
	struct sam3_tensor *q = gh_linear(g, arena, q_src, q_w, q_b);
	if (!q)
		return NULL;

	/* Project KV: [n_kv, 2*d_model] */
	struct sam3_tensor *kv = gh_linear(g, arena, kv_src, kv_w, kv_b);
	if (!kv)
		return NULL;

	/* Slice K and V from packed KV */
	struct sam3_tensor *k = gh_slice(g, arena, kv, 1, 0, d_model);
	struct sam3_tensor *v = gh_slice(g, arena, kv, 1,
					  d_model, 2 * d_model);
	if (!k || !v)
		return NULL;

	/*
	 * Batched multi-head cross-attention. See gh_multihead_attention_rope
	 * for the full explanation: a direct reshape from [n, d_model] to
	 * [1, n_heads, n, head_dim] is WRONG -- the seq/head axes must be
	 * physically transposed, not just relabeled.
	 *
	 * Pattern (Q):
	 *   [n_q, d_model]
	 *     -> [n_q, n_heads, head_dim]   (pure reshape, split last dim)
	 *     -> permute(1,0,2) -> [n_heads, n_q, head_dim]
	 *     -> [1, n_heads, n_q, head_dim] (pure reshape, unsqueeze)
	 *
	 * K/V follow the same pattern with n_kv.
	 */
	int n_q = q->dims[0];
	int n_kv = k->dims[0];
	int q_3d[]  = {n_q,  n_heads, head_dim};
	int kv_3d[] = {n_kv, n_heads, head_dim};
	int q_4d[]  = {1, n_heads, n_q,  head_dim};
	int kv_4d[] = {1, n_heads, n_kv, head_dim};
	int perm_sh[] = {1, 0, 2};

	struct sam3_tensor *q_3 = gh_reshape(g, arena, q, 3, q_3d);
	struct sam3_tensor *k_3 = gh_reshape(g, arena, k, 3, kv_3d);
	struct sam3_tensor *v_3 = gh_reshape(g, arena, v, 3, kv_3d);
	if (!q_3 || !k_3 || !v_3)
		return NULL;

	struct sam3_tensor *q_hs = gh_permute(g, arena, q_3, perm_sh);
	struct sam3_tensor *k_hs = gh_permute(g, arena, k_3, perm_sh);
	struct sam3_tensor *v_hs = gh_permute(g, arena, v_3, perm_sh);
	if (!q_hs || !k_hs || !v_hs)
		return NULL;

	struct sam3_tensor *q4 = gh_reshape(g, arena, q_hs, 4, q_4d);
	struct sam3_tensor *k4 = gh_reshape(g, arena, k_hs, 4, kv_4d);
	struct sam3_tensor *v4 = gh_reshape(g, arena, v_hs, 4, kv_4d);
	if (!q4 || !k4 || !v4)
		return NULL;

	struct sam3_tensor *attn_4d = gh_sdpa(g, arena, q4, k4, v4,
					       NULL, head_dim);
	if (!attn_4d)
		return NULL;

	/*
	 * Reshape back:
	 *   [1, n_heads, n_q, head_dim]
	 *     -> [n_heads, n_q, head_dim]   (drop batch)
	 *     -> permute(1,0,2) -> [n_q, n_heads, head_dim]
	 *     -> [n_q, d_model]             (flatten last dim)
	 */
	int attn_3d_hs[] = {n_heads, n_q, head_dim};
	struct sam3_tensor *attn_hs = gh_reshape(g, arena, attn_4d,
						  3, attn_3d_hs);
	if (!attn_hs)
		return NULL;

	struct sam3_tensor *attn_sh = gh_permute(g, arena, attn_hs, perm_sh);
	if (!attn_sh)
		return NULL;

	int merged_dims[] = {n_q, d_model};
	struct sam3_tensor *merged = gh_reshape(g, arena, attn_sh,
						 2, merged_dims);
	if (!merged)
		return NULL;

	/* Output projection */
	return gh_linear(g, arena, merged, out_w, out_b);
}

struct sam3_tensor *gh_multihead_attention_sep(
	struct sam3_graph *g, struct sam3_arena *a,
	struct sam3_tensor *x,
	struct sam3_tensor *q_w, struct sam3_tensor *q_b,
	struct sam3_tensor *k_w, struct sam3_tensor *k_b,
	struct sam3_tensor *v_w, struct sam3_tensor *v_b,
	struct sam3_tensor *out_w, struct sam3_tensor *out_b,
	int n_heads,
	struct sam3_tensor *attn_mask)
{
	return gh_multihead_attention_rope_sep(g, a, x,
		q_w, q_b, k_w, k_b, v_w, v_b,
		out_w, out_b, n_heads,
		NULL, NULL, attn_mask);
}

struct sam3_tensor *gh_cross_attention_sep(
	struct sam3_graph *g, struct sam3_arena *arena,
	struct sam3_tensor *q_src,
	struct sam3_tensor *kv_src,
	struct sam3_tensor *q_w, struct sam3_tensor *q_b,
	struct sam3_tensor *k_w, struct sam3_tensor *k_b,
	struct sam3_tensor *v_w, struct sam3_tensor *v_b,
	struct sam3_tensor *out_w, struct sam3_tensor *out_b,
	int n_heads)
{
	int d_model = q_src->dims[1];
	int head_dim = d_model / n_heads;

	/* Project Q from q_src, K and V separately from kv_src */
	struct sam3_tensor *q = gh_linear(g, arena, q_src, q_w, q_b);
	struct sam3_tensor *k = gh_linear(g, arena, kv_src, k_w, k_b);
	struct sam3_tensor *v = gh_linear(g, arena, kv_src, v_w, v_b);
	if (!q || !k || !v)
		return NULL;

	/*
	 * Batched multi-head cross-attention. See gh_cross_attention for the
	 * full explanation: physical transpose of seq/head axes is required.
	 */
	int n_q = q->dims[0];
	int n_kv = k->dims[0];
	int q_3d[]  = {n_q,  n_heads, head_dim};
	int kv_3d[] = {n_kv, n_heads, head_dim};
	int q_4d[]  = {1, n_heads, n_q,  head_dim};
	int kv_4d[] = {1, n_heads, n_kv, head_dim};
	int perm_sh[] = {1, 0, 2};

	struct sam3_tensor *q_3 = gh_reshape(g, arena, q, 3, q_3d);
	struct sam3_tensor *k_3 = gh_reshape(g, arena, k, 3, kv_3d);
	struct sam3_tensor *v_3 = gh_reshape(g, arena, v, 3, kv_3d);
	if (!q_3 || !k_3 || !v_3)
		return NULL;

	struct sam3_tensor *q_hs = gh_permute(g, arena, q_3, perm_sh);
	struct sam3_tensor *k_hs = gh_permute(g, arena, k_3, perm_sh);
	struct sam3_tensor *v_hs = gh_permute(g, arena, v_3, perm_sh);
	if (!q_hs || !k_hs || !v_hs)
		return NULL;

	struct sam3_tensor *q4 = gh_reshape(g, arena, q_hs, 4, q_4d);
	struct sam3_tensor *k4 = gh_reshape(g, arena, k_hs, 4, kv_4d);
	struct sam3_tensor *v4 = gh_reshape(g, arena, v_hs, 4, kv_4d);
	if (!q4 || !k4 || !v4)
		return NULL;

	struct sam3_tensor *attn_4d = gh_sdpa(g, arena, q4, k4, v4,
					       NULL, head_dim);
	if (!attn_4d)
		return NULL;

	int attn_3d_hs[] = {n_heads, n_q, head_dim};
	struct sam3_tensor *attn_hs = gh_reshape(g, arena, attn_4d,
						  3, attn_3d_hs);
	if (!attn_hs)
		return NULL;

	struct sam3_tensor *attn_sh = gh_permute(g, arena, attn_hs, perm_sh);
	if (!attn_sh)
		return NULL;

	int merged_dims[] = {n_q, d_model};
	struct sam3_tensor *merged = gh_reshape(g, arena, attn_sh,
						 2, merged_dims);
	if (!merged)
		return NULL;

	/* Output projection */
	return gh_linear(g, arena, merged, out_w, out_b);
}

/* ── MLP ─────────────────────────────────────────────────────────── */

struct sam3_tensor *gh_mlp(struct sam3_graph *g, struct sam3_arena *a,
			   struct sam3_tensor *input,
			   struct sam3_tensor *fc1_w, struct sam3_tensor *fc1_b,
			   struct sam3_tensor *fc2_w, struct sam3_tensor *fc2_b,
			   enum sam3_op activation)
{
	/* First linear layer */
	struct sam3_tensor *hidden = gh_linear(g, a, input, fc1_w, fc1_b);
	if (!hidden)
		return NULL;

	/* Activation */
	struct sam3_tensor *activated = gh_unary(g, a, activation, hidden);
	if (!activated)
		return NULL;

	/* Second linear layer */
	return gh_linear(g, a, activated, fc2_w, fc2_b);
}

/* ── Group normalization ─────────────────────────────────────────── */

struct sam3_tensor *gh_groupnorm(struct sam3_graph *g, struct sam3_arena *a,
				 struct sam3_tensor *input,
				 struct sam3_tensor *gamma,
				 struct sam3_tensor *beta,
				 int num_groups)
{
	struct sam3_tensor *out = gh_alloc_tensor(a, input->dtype,
						  input->n_dims, input->dims);
	if (!out)
		return NULL;

	struct sam3_tensor *inputs[3];
	int n_inputs = 1;
	inputs[0] = input;
	if (gamma) {
		inputs[1] = gamma;
		n_inputs = 2;
	}
	if (beta) {
		inputs[2] = beta;
		n_inputs = 3;
	}

	out = sam3_graph_add_op(g, SAM3_OP_GROUPNORM,
				 inputs, n_inputs, out);
	if (!out)
		return NULL;

	g->nodes[g->n_nodes - 1].params[0] = num_groups;
	return out;
}

struct sam3_tensor *gh_groupnorm_nhwc(struct sam3_graph *g,
				      struct sam3_arena *a,
				      struct sam3_tensor *input,
				      struct sam3_tensor *gamma,
				      struct sam3_tensor *beta,
				      int num_groups)
{
	/*
	 * NHWC group norm. Output has the same NHWC shape as input.
	 *
	 * Slot usage for SAM3_OP_GROUPNORM:
	 *   params[0] = num_groups (shared with NCHW path)
	 *   params[1] = NHWC flag (0 = legacy NCHW, 1 = NHWC)
	 * Slots 2 and 3 remain unused.
	 */
	struct sam3_tensor *out = gh_alloc_tensor(a, input->dtype,
						  input->n_dims, input->dims);
	if (!out)
		return NULL;

	struct sam3_tensor *inputs[3];
	int n_inputs = 1;
	inputs[0] = input;
	if (gamma) {
		inputs[1] = gamma;
		n_inputs = 2;
	}
	if (beta) {
		inputs[2] = beta;
		n_inputs = 3;
	}

	out = sam3_graph_add_op(g, SAM3_OP_GROUPNORM,
				 inputs, n_inputs, out);
	if (!out)
		return NULL;

	struct sam3_node *node = &g->nodes[g->n_nodes - 1];
	node->params[0] = num_groups;
	node->params[1] = 1;  /* NHWC flag */
	return out;
}

/* ── Convolution helpers ──────────────────────────────────────────── */

/*
 * conv_add_bias - Add bias [C] to a 4D NCHW tensor via fused op.
 *
 * Uses SAM3_OP_BIAS_ADD which broadcasts bias along the channel
 * dimension. One op, one output allocation (same size as input).
 */
static struct sam3_tensor *conv_add_bias(struct sam3_graph *g,
					 struct sam3_arena *a,
					 struct sam3_tensor *x,
					 struct sam3_tensor *bias)
{
	struct sam3_tensor *out = gh_alloc_tensor(a, x->dtype,
						  x->n_dims, x->dims);
	if (!out)
		return NULL;

	struct sam3_tensor *inputs[] = {x, bias};
	return sam3_graph_add_op(g, SAM3_OP_BIAS_ADD, inputs, 2, out);
}

struct sam3_tensor *gh_conv2d(struct sam3_graph *g, struct sam3_arena *a,
			      struct sam3_tensor *input,
			      struct sam3_tensor *weight,
			      struct sam3_tensor *bias,
			      int stride, int padding)
{
	int N = input->dims[0];
	int H = input->dims[2];
	int W = input->dims[3];
	int OC = weight->dims[0];
	int KH = weight->dims[2];
	int KW = weight->dims[3];

	int OH = (H + 2 * padding - KH) / stride + 1;
	int OW = (W + 2 * padding - KW) / stride + 1;

	int out_dims[] = {N, OC, OH, OW};
	struct sam3_tensor *out = gh_alloc_tensor(a, input->dtype,
						  4, out_dims);
	if (!out)
		return NULL;

	struct sam3_tensor *inputs[] = {input, weight};
	out = sam3_graph_add_op(g, SAM3_OP_CONV2D, inputs, 2, out);
	if (!out)
		return NULL;

	struct sam3_node *node = &g->nodes[g->n_nodes - 1];
	node->params[0] = stride;
	node->params[1] = padding;

	if (bias)
		out = conv_add_bias(g, a, out, bias);

	return out;
}

struct sam3_tensor *gh_conv_transpose2d(struct sam3_graph *g,
					struct sam3_arena *a,
					struct sam3_tensor *input,
					struct sam3_tensor *weight,
					struct sam3_tensor *bias,
					int stride, int padding)
{
	int N = input->dims[0];
	int H = input->dims[2];
	int W = input->dims[3];
	int C_out = weight->dims[1];
	int KH = weight->dims[2];
	int KW = weight->dims[3];

	int OH = (H - 1) * stride - 2 * padding + KH;
	int OW = (W - 1) * stride - 2 * padding + KW;

	int out_dims[] = {N, C_out, OH, OW};
	struct sam3_tensor *out = gh_alloc_tensor(a, input->dtype,
						  4, out_dims);
	if (!out)
		return NULL;

	struct sam3_tensor *inputs[] = {input, weight};
	out = sam3_graph_add_op(g, SAM3_OP_CONV_TRANSPOSE2D,
				 inputs, 2, out);
	if (!out)
		return NULL;

	struct sam3_node *node = &g->nodes[g->n_nodes - 1];
	node->params[0] = stride;
	node->params[1] = padding;

	if (bias)
		out = conv_add_bias(g, a, out, bias);

	return out;
}

struct sam3_tensor *gh_conv2d_nhwc(struct sam3_graph *g,
				   struct sam3_arena *a,
				   struct sam3_tensor *input,
				   struct sam3_tensor *weight,
				   struct sam3_tensor *bias,
				   int stride, int padding)
{
	int N = input->dims[0];
	int H = input->dims[1];
	int W = input->dims[2];
	int OC = weight->dims[0];
	int KH = weight->dims[1];
	int KW = weight->dims[2];

	int OH = (H + 2 * padding - KH) / stride + 1;
	int OW = (W + 2 * padding - KW) / stride + 1;

	int out_dims[] = {N, OH, OW, OC};
	struct sam3_tensor *out = gh_alloc_tensor(a, input->dtype,
						  4, out_dims);
	if (!out)
		return NULL;

	struct sam3_tensor *inputs[] = {input, weight};
	out = sam3_graph_add_op(g, SAM3_OP_CONV2D, inputs, 2, out);
	if (!out)
		return NULL;

	struct sam3_node *node = &g->nodes[g->n_nodes - 1];
	node->params[0] = stride;
	node->params[1] = padding;
	node->params[2] = 1;  /* NHWC flag */

	if (bias)
		out = conv_add_bias(g, a, out, bias);

	return out;
}

struct sam3_tensor *gh_conv_transpose2d_nhwc(struct sam3_graph *g,
					     struct sam3_arena *a,
					     struct sam3_tensor *input,
					     struct sam3_tensor *weight,
					     struct sam3_tensor *bias,
					     int stride, int padding)
{
	int N = input->dims[0];
	int H = input->dims[1];
	int W = input->dims[2];
	int OC = weight->dims[0];
	int KH = weight->dims[1];
	int KW = weight->dims[2];

	int OH = (H - 1) * stride - 2 * padding + KH;
	int OW = (W - 1) * stride - 2 * padding + KW;

	int out_dims[] = {N, OH, OW, OC};
	struct sam3_tensor *out = gh_alloc_tensor(a, input->dtype,
						  4, out_dims);
	if (!out)
		return NULL;

	struct sam3_tensor *inputs[] = {input, weight};
	out = sam3_graph_add_op(g, SAM3_OP_CONV_TRANSPOSE2D,
				 inputs, 2, out);
	if (!out)
		return NULL;

	struct sam3_node *node = &g->nodes[g->n_nodes - 1];
	node->params[0] = stride;
	node->params[1] = padding;
	node->params[2] = 1;  /* NHWC flag */

	if (bias)
		out = conv_add_bias(g, a, out, bias);

	return out;
}

struct sam3_tensor *gh_maxpool2d(struct sam3_graph *g, struct sam3_arena *a,
				struct sam3_tensor *input,
				int kernel_size, int stride)
{
	int N = input->dims[0];
	int C = input->dims[1];
	int H = input->dims[2];
	int W = input->dims[3];

	int OH = (H - kernel_size) / stride + 1;
	int OW = (W - kernel_size) / stride + 1;

	int out_dims[] = {N, C, OH, OW};
	struct sam3_tensor *out = gh_alloc_tensor(a, input->dtype,
						  4, out_dims);
	if (!out)
		return NULL;

	struct sam3_tensor *inputs[] = {input};
	out = sam3_graph_add_op(g, SAM3_OP_MAXPOOL2D, inputs, 1, out);
	if (!out)
		return NULL;

	struct sam3_node *node = &g->nodes[g->n_nodes - 1];
	node->params[0] = kernel_size;
	node->params[1] = stride;

	return out;
}

struct sam3_tensor *gh_maxpool2d_nhwc(struct sam3_graph *g,
				      struct sam3_arena *a,
				      struct sam3_tensor *input,
				      int kernel_size, int stride)
{
	/*
	 * NHWC non-overlapping max pooling.
	 *
	 * Slot usage for SAM3_OP_MAXPOOL2D:
	 *   params[0] = kernel_size (shared with NCHW path)
	 *   params[1] = stride (shared with NCHW path)
	 *   params[2] = NHWC flag (0 = legacy NCHW, 1 = NHWC)
	 * Slot 3 remains unused. params[2] matches the slot used by
	 * gh_conv2d_nhwc so readers learn a single convention for the
	 * NHWC flag on spatial ops.
	 */
	int N = input->dims[0];
	int H = input->dims[1];
	int W = input->dims[2];
	int C = input->dims[3];

	int OH = (H - kernel_size) / stride + 1;
	int OW = (W - kernel_size) / stride + 1;

	int out_dims[] = {N, OH, OW, C};
	struct sam3_tensor *out = gh_alloc_tensor(a, input->dtype,
						  4, out_dims);
	if (!out)
		return NULL;

	struct sam3_tensor *inputs[] = {input};
	out = sam3_graph_add_op(g, SAM3_OP_MAXPOOL2D, inputs, 1, out);
	if (!out)
		return NULL;

	struct sam3_node *node = &g->nodes[g->n_nodes - 1];
	node->params[0] = kernel_size;
	node->params[1] = stride;
	node->params[2] = 1;  /* NHWC flag */

	return out;
}
