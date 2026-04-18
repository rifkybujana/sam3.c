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
 * Depends on: graph_helpers.h, memory_bank.h, core/graph.h, core/alloc.h,
 *             core/tensor.h, core/weight.h
 * Used by:    model/ files (vitdet.c, text_encoder.c, decoder.c,
 *             tracker.c, etc.)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "util/log.h"

#include "graph_helpers.h"
#include "memory_bank.h"

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

	if (arena->skip_data) {
		t->data = NULL;
		return t;
	}

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

struct sam3_tensor *gh_hswish(struct sam3_graph *g, struct sam3_arena *a,
			      struct sam3_tensor *input)
{
	return gh_unary(g, a, SAM3_OP_HSWISH, input);
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

struct sam3_tensor *gh_div(struct sam3_graph *g, struct sam3_arena *a,
			   struct sam3_tensor *x, struct sam3_tensor *y)
{
	struct sam3_tensor *out = gh_alloc_tensor(a, x->dtype,
						  x->n_dims, x->dims);
	if (!out)
		return NULL;

	return sam3_graph_add_op(g, SAM3_OP_DIV,
				 (struct sam3_tensor *[]){x, y}, 2, out);
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
	/* Validate element count: input and output must match. */
	int in_elems = sam3_tensor_nelems(input);
	int out_elems = 1;
	for (int i = 0; i < n_dims; i++)
		out_elems *= dims[i];
	if (in_elems != out_elems) {
		sam3_log_error("gh_reshape: element count mismatch "
			       "(input %d != output %d)", in_elems, out_elems);
		return NULL;
	}

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
	/* Input: [N, H, W, C] -> Output: [N, H*scale, W*scale, C] */
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

	g->nodes[g->n_nodes - 1].params[0] = scale;
	return result;
}

/* ── Rotary position embedding ───────────────────────────────────── */

struct sam3_tensor *gh_rope(struct sam3_graph *g, struct sam3_arena *a,
			     struct sam3_tensor *input,
			     struct sam3_tensor *cos_f,
			     struct sam3_tensor *sin_f,
			     int grid_w, float scale)
{
	struct sam3_tensor *out = gh_alloc_tensor(a, input->dtype,
						  input->n_dims, input->dims);
	if (!out)
		return NULL;

	struct sam3_tensor *result = sam3_graph_add_op(g, SAM3_OP_ROPE,
		(struct sam3_tensor *[]){input, cos_f, sin_f}, 3, out);
	if (!result)
		return NULL;

	struct sam3_node *node = &g->nodes[g->n_nodes - 1];
	node->params[0] = input->dims[3];	/* head_dim */
	node->params[1] = grid_w;		/* 0 = legacy path */
	memcpy(&node->params[2], &scale, sizeof(float));
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
	struct sam3_tensor *attn_mask,
	int grid_w, float rope_scale)
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
		sq_4d = gh_rope(g, a, sq_4d, rope_cos, rope_sin,
				grid_w, rope_scale);
		if (!sq_4d)
			return NULL;
		sq = gh_reshape(g, a, sq_4d, 2, flat_dims);
		if (!sq)
			return NULL;

		struct sam3_tensor *sk_4d;
		sk_4d = gh_reshape(g, a, sk, 4, rope_4d);
		if (!sk_4d)
			return NULL;
		sk_4d = gh_rope(g, a, sk_4d, rope_cos, rope_sin,
				grid_w, rope_scale);
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
	 * Input sq/sk/sv are [bs, d_model] = [batch*seq_q, d_model].
	 * MLX SDPA wants [B, H, S, D]. We achieve this with one 4-D
	 * reshape + one 4-D permute that works for any batch:
	 *
	 *   [bs, d_model]
	 *     -> [batch, seq_q, n_heads, head_dim]   (split bs and d_model)
	 *     -> permute(0,2,1,3) -> [batch, n_heads, seq_q, head_dim]
	 */
	int qkv_pre[]  = {batch, seq_q, n_heads, head_dim};
	int qkv_perm[] = {0, 2, 1, 3};

	struct sam3_tensor *sq_pre = gh_reshape(g, a, sq, 4, qkv_pre);
	struct sam3_tensor *sk_pre = gh_reshape(g, a, sk, 4, qkv_pre);
	struct sam3_tensor *sv_pre = gh_reshape(g, a, sv, 4, qkv_pre);
	if (!sq_pre || !sk_pre || !sv_pre) {
		sam3_log_error("mha: pre-SDPA reshape fail "
			       "(arena %zu/%zu)",
			       a->offset, a->size);
		return NULL;
	}

	struct sam3_tensor *sq_4d = gh_permute(g, a, sq_pre, qkv_perm);
	struct sam3_tensor *sk_4d = gh_permute(g, a, sk_pre, qkv_perm);
	struct sam3_tensor *sv_4d = gh_permute(g, a, sv_pre, qkv_perm);
	if (!sq_4d || !sk_4d || !sv_4d) {
		sam3_log_error("mha: pre-SDPA permute fail "
			       "(arena %zu/%zu)",
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
	 * Reshape back: [batch, n_heads, seq_q, head_dim]
	 *   -> permute(0,2,1,3) -> [batch, seq_q, n_heads, head_dim]
	 *   -> reshape -> [bs, d_model]
	 */
	int attn_perm_back[] = {0, 2, 1, 3};
	struct sam3_tensor *attn_back =
		gh_permute(g, a, attn_4d, attn_perm_back);
	if (!attn_back) {
		sam3_log_error("mha: post-SDPA permute fail "
			       "(arena %zu/%zu)",
			       a->offset, a->size);
		return NULL;
	}

	struct sam3_tensor *merged = gh_reshape(g, a, attn_back, 2, flat_dims);
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
	struct sam3_tensor *attn_mask,
	int grid_w, float rope_scale)
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
		sq_4d = gh_rope(g, a, sq_4d, rope_cos, rope_sin,
				grid_w, rope_scale);
		if (!sq_4d)
			return NULL;
		sq = gh_reshape(g, a, sq_4d, 2, flat_dims);
		if (!sq)
			return NULL;

		struct sam3_tensor *sk_4d;
		sk_4d = gh_reshape(g, a, sk, 4, rope_4d);
		if (!sk_4d)
			return NULL;
		sk_4d = gh_rope(g, a, sk_4d, rope_cos, rope_sin,
				grid_w, rope_scale);
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
		n_heads, NULL, NULL, NULL, 0, 0.0f);
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
		NULL, NULL, attn_mask, 0, 0.0f);
}

struct sam3_tensor *gh_cross_attention_sep(
	struct sam3_graph *g, struct sam3_arena *arena,
	struct sam3_tensor *q_src,
	struct sam3_tensor *kv_src,
	struct sam3_tensor *k_src,
	struct sam3_tensor *q_w, struct sam3_tensor *q_b,
	struct sam3_tensor *k_w, struct sam3_tensor *k_b,
	struct sam3_tensor *v_w, struct sam3_tensor *v_b,
	struct sam3_tensor *out_w, struct sam3_tensor *out_b,
	int n_heads)
{
	int d_model = q_src->dims[1];
	int head_dim = d_model / n_heads;

	/* K projects from k_src when provided, otherwise from kv_src */
	struct sam3_tensor *q = gh_linear(g, arena, q_src, q_w, q_b);
	struct sam3_tensor *k = gh_linear(g, arena,
		k_src ? k_src : kv_src, k_w, k_b);
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

/* ── Batch normalization ─────────────────────────────────────────── */

struct sam3_tensor *gh_batchnorm(struct sam3_graph *g, struct sam3_arena *a,
				 struct sam3_tensor *input,
				 struct sam3_tensor *gamma,
				 struct sam3_tensor *beta,
				 struct sam3_tensor *running_mean,
				 struct sam3_tensor *running_var)
{
	struct sam3_tensor *out = gh_alloc_tensor(a, input->dtype,
						  input->n_dims, input->dims);
	if (!out)
		return NULL;

	struct sam3_tensor *inputs[5];
	int n_inputs = 1;
	inputs[0] = input;
	if (gamma) { inputs[1] = gamma; n_inputs = 2; }
	if (beta)  { inputs[2] = beta;  n_inputs = 3; }
	if (running_mean) { inputs[3] = running_mean; n_inputs = 4; }
	if (running_var)  { inputs[4] = running_var;  n_inputs = 5; }

	return sam3_graph_add_op(g, SAM3_OP_BATCHNORM,
				 inputs, n_inputs, out);
}

/* ── Convolution helpers ──────────────────────────────────────────── */

/*
 * conv_add_bias - Add bias [C] to an NHWC 4D tensor via fused op.
 *
 * Uses SAM3_OP_BIAS_ADD which broadcasts bias along the innermost
 * channel dimension of an [N, H, W, C] tensor.
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
			      int stride, int padding, int groups)
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
	node->params[2] = groups;

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

	if (bias)
		out = conv_add_bias(g, a, out, bias);

	return out;
}

struct sam3_tensor *gh_maxpool2d(struct sam3_graph *g, struct sam3_arena *a,
				struct sam3_tensor *input,
				int kernel_size, int stride)
{
	/* Non-overlapping NHWC max pool: [N,H,W,C] -> [N,OH,OW,C]. */
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

	return out;
}

/* ── Tracker memory helpers ──────────────────────────────────────── */

struct sam3_tensor *gh_concat_mem(struct sam3_graph *g,
				  struct sam3_arena *a,
				  const struct sam3_memory_bank_view *view)
{
	if (!view || !view->bank)
		return NULL;

	const struct sam3_memory_bank *bank = view->bank;
	int total = view->n_cond + bank->n_non_cond;
	if (total <= 0)
		return NULL;

	/* Allocate a temporary pointer array from the arena. */
	struct sam3_tensor **tensors = (struct sam3_tensor **)
		sam3_arena_alloc(a,
				 (size_t)total * sizeof(struct sam3_tensor *));
	if (!tensors) {
		sam3_log_error("gh_concat_mem: arena alloc failed");
		return NULL;
	}

	int n = 0;

	for (int k = 0; k < view->n_cond; k++) {
		int idx = view->cond_idx[k];
		struct sam3_tensor *sp = bank->cond[idx].spatial_features;
		if (sp)
			tensors[n++] = sp;
	}
	for (int i = 0; i < bank->n_non_cond; i++) {
		struct sam3_tensor *sp = bank->non_cond[i].spatial_features;
		if (sp)
			tensors[n++] = sp;
	}

	if (n == 0)
		return NULL;

	if (n == 1)
		return tensors[0];

	return gh_concat(g, a, tensors, n, 0);
}

struct sam3_tensor *gh_tpos_enc_mem(struct sam3_graph *g,
				    struct sam3_arena *a,
				    const struct sam3_memory_bank_view *view,
				    struct sam3_tensor *tpos_table,
				    int current_frame_idx)
{
	(void)g;

	if (!view || !view->bank || !tpos_table)
		return NULL;

	const struct sam3_memory_bank *bank = view->bank;

	/* tpos_table is expected to be [num_maskmem, 1, 1, mem_dim]. */
	if (tpos_table->n_dims != 4) {
		sam3_log_error("gh_tpos_enc_mem: bad tpos_table dims %d",
			       tpos_table->n_dims);
		return NULL;
	}
	int num_maskmem = tpos_table->dims[0];
	int mem_dim     = tpos_table->dims[3];

	/* Count usable entries and compute total spatial tokens. */
	int total_spatial = 0;
	int n_entries     = view->n_cond + bank->n_non_cond;
	for (int k = 0; k < view->n_cond; k++) {
		int idx = view->cond_idx[k];
		struct sam3_tensor *sp = bank->cond[idx].spatial_features;
		if (sp)
			total_spatial += sp->dims[0];
	}
	for (int i = 0; i < bank->n_non_cond; i++) {
		struct sam3_tensor *sp = bank->non_cond[i].spatial_features;
		if (sp)
			total_spatial += sp->dims[0];
	}
	if (total_spatial == 0 || n_entries == 0)
		return NULL;

	int out_dims[] = {total_spatial, mem_dim};
	struct sam3_tensor *out = gh_alloc_tensor(a, SAM3_DTYPE_F32,
						  2, out_dims);
	if (!out) {
		sam3_log_error("gh_tpos_enc_mem: alloc failed");
		return NULL;
	}

	float       *dst = (float *)out->data;
	const float *tbl = (const float *)tpos_table->data;
	int          row = 0;

	(void)current_frame_idx;

	/*
	 * Slot selection matches Python sam3_tracker_base.py:614-676.
	 *
	 *   t = t_pos if not is_selected_cond_frame else 0
	 *   maskmem_enc = maskmem_enc + maskmem_tpos_enc[num_maskmem - t - 1]
	 *
	 * Cond frames (t = 0) always bind to the last slot,
	 *   slot = num_maskmem - 1.
	 *
	 * Non-cond frames occupy t_pos = 1..num_maskmem-1 in the sliding
	 * memory window (t_pos = num_maskmem-1 is the most recent, t_pos = 1
	 * is the oldest), so
	 *   slot = num_maskmem - 1 - t_pos.
	 * The bank stores non_cond[0] = oldest and non_cond[n-1] = newest,
	 * so the k-th most-recent entry (k = 1..n_non_cond, counting from
	 * the newest) sits at index i = n_non_cond - k and gets slot k - 1.
	 * For n_non_cond < num_maskmem - 1 Python simply drops the missing
	 * slots; our bank keeps whatever entries it has and binds each to
	 * its own k-based slot.
	 */
	for (int k = 0; k < view->n_cond; k++) {
		int idx = view->cond_idx[k];
		struct sam3_tensor *sp = bank->cond[idx].spatial_features;
		if (!sp)
			continue;

		int hw   = sp->dims[0];
		int slot = num_maskmem - 1;
		const float *tbl_slot = tbl + (size_t)slot * mem_dim;

		for (int j = 0; j < hw; j++) {
			memcpy(dst + (size_t)row * mem_dim, tbl_slot,
			       (size_t)mem_dim * sizeof(float));
			row++;
		}
	}

	for (int i = 0; i < bank->n_non_cond; i++) {
		struct sam3_tensor *sp = bank->non_cond[i].spatial_features;
		if (!sp)
			continue;

		int hw   = sp->dims[0];
		int slot = bank->n_non_cond - 1 - i;
		if (slot < 0)
			slot = 0;
		if (slot > num_maskmem - 2)
			slot = num_maskmem - 2;

		const float *tbl_slot = tbl + (size_t)slot * mem_dim;

		for (int j = 0; j < hw; j++) {
			memcpy(dst + (size_t)row * mem_dim, tbl_slot,
			       (size_t)mem_dim * sizeof(float));
			row++;
		}
	}

	return out;
}

struct sam3_tensor *gh_concat_obj_ptrs(struct sam3_graph *g,
				       struct sam3_arena *a,
				       const struct sam3_memory_bank_view *view,
				       int max_obj_ptrs)
{
	if (!view || !view->bank)
		return NULL;

	const struct sam3_memory_bank *bank = view->bank;
	int total_entries = view->n_cond + bank->n_non_cond;
	if (total_entries <= 0)
		return NULL;

	/*
	 * Upstream splits each [n_obj, hidden_dim=256] obj_pointer row
	 * into hidden_dim/mem_dim = 4 chunks of mem_dim=64 each and
	 * stacks them along axis 0 as additional memory tokens.
	 */
	const int mem_dim    = 64;
	const int hidden_dim = 256;
	const int split      = hidden_dim / mem_dim; /* 4 */

	struct sam3_tensor **tensors = (struct sam3_tensor **)
		sam3_arena_alloc(a, (size_t)total_entries *
				 sizeof(struct sam3_tensor *));
	if (!tensors) {
		sam3_log_error("gh_concat_obj_ptrs: arena alloc failed");
		return NULL;
	}

	int n_tensors    = 0;
	int total_n_obj  = 0;

	/*
	 * Select up to max_obj_ptrs most recent entries. Within each pass
	 * we walk newest-first so the cap drops oldest entries when
	 * saturated. Pass 0 visits selected cond entries via view->cond_idx
	 * (already sorted oldest-first); pass 1 visits bank->non_cond[]
	 * directly (oldest at index 0, newest at n_non_cond-1). Collected
	 * chunks are appended newest-first here; after the passes we
	 * reverse tensors[] so the concat output stays oldest-first.
	 */
	for (int pass = 0; pass < 2; pass++) {
		int count = (pass == 0) ? view->n_cond : bank->n_non_cond;

		for (int k = count - 1; k >= 0; k--) {
			int idx = (pass == 0) ? view->cond_idx[k] : k;
			const struct sam3_memory_entry *arr =
				(pass == 0) ? bank->cond : bank->non_cond;
			struct sam3_tensor *op = arr[idx].obj_pointer;
			if (!op)
				continue;

			int n_obj = op->dims[0];

			/* Cap so that total pre-reshape rows <= max. */
			if (max_obj_ptrs > 0 &&
			    total_n_obj + n_obj > max_obj_ptrs) {
				n_obj = max_obj_ptrs - total_n_obj;
				if (n_obj <= 0)
					break;
			}

			int reshaped_dims[] = {n_obj * split, mem_dim};
			struct sam3_tensor *chunk = gh_reshape(
				g, a, op, 2, reshaped_dims);
			if (!chunk) {
				sam3_log_error("gh_concat_obj_ptrs: reshape "
					       "failed");
				return NULL;
			}

			tensors[n_tensors++] = chunk;
			total_n_obj += n_obj;

			if (max_obj_ptrs > 0 && total_n_obj >= max_obj_ptrs)
				break;
		}
		if (max_obj_ptrs > 0 && total_n_obj >= max_obj_ptrs)
			break;
	}

	if (n_tensors == 0)
		return NULL;

	if (n_tensors == 1)
		return tensors[0];

	/* Reverse so concat rows come out oldest-first. */
	for (int lo = 0, hi = n_tensors - 1; lo < hi; lo++, hi--) {
		struct sam3_tensor *tmp = tensors[lo];
		tensors[lo] = tensors[hi];
		tensors[hi] = tmp;
	}

	return gh_concat(g, a, tensors, n_tensors, 0);
}

/*
 * gh_obj_ptrs_tpos_sine - see header for contract.
 *
 * Iterates the memory bank with the same selection rule as
 * gh_concat_obj_ptrs and writes one [hidden_dim] sine PE per
 * contributing chunk (4 chunks per 256-D pointer) into a single
 * arena tensor. The temporal distance is abs(current - entry) /
 * (max_obj_ptrs - 1), matching Python's _get_tpos_enc.
 */
struct sam3_tensor *gh_obj_ptrs_tpos_sine(
	struct sam3_graph *g,
	struct sam3_arena *a,
	const struct sam3_memory_bank_view *view,
	int max_obj_ptrs,
	int current_frame_idx,
	int hidden_dim)
{
	(void)g;

	if (!view || !view->bank || !a ||
	    hidden_dim < 2 || (hidden_dim & 1) != 0)
		return NULL;

	const struct sam3_memory_bank *bank = view->bank;

	const int mem_dim = 64;
	const int split   = hidden_dim / mem_dim;
	const int pe_dim  = hidden_dim / 2; /* half sine, half cosine */
	const float temperature = 10000.0f;
	const float t_diff_max =
		(max_obj_ptrs > 1) ? (float)(max_obj_ptrs - 1) : 1.0f;

	/*
	 * Collect contributing frame indices in the same order as
	 * gh_concat_obj_ptrs: cond pass then non_cond pass, within each
	 * pass entries visited newest-first (reverse iteration), then
	 * the full list is reversed at the end.
	 */
	int total_candidates = view->n_cond + bank->n_non_cond;
	if (total_candidates <= 0)
		return NULL;
	int *frame_idxs = (int *)sam3_arena_alloc(
		a, (size_t)total_candidates * sizeof(int));
	if (!frame_idxs)
		return NULL;

	int n_entries = 0;
	int total_n_obj = 0;

	for (int pass = 0; pass < 2; pass++) {
		int count = (pass == 0) ? view->n_cond : bank->n_non_cond;

		for (int k = count - 1; k >= 0; k--) {
			int idx = (pass == 0) ? view->cond_idx[k] : k;
			const struct sam3_memory_entry *arr =
				(pass == 0) ? bank->cond : bank->non_cond;
			struct sam3_tensor *op = arr[idx].obj_pointer;
			if (!op)
				continue;

			int n_obj = op->dims[0];
			if (max_obj_ptrs > 0 &&
			    total_n_obj + n_obj > max_obj_ptrs) {
				n_obj = max_obj_ptrs - total_n_obj;
				if (n_obj <= 0)
					break;
			}

			frame_idxs[n_entries++] = arr[idx].frame_idx;
			total_n_obj += n_obj;

			if (max_obj_ptrs > 0 && total_n_obj >= max_obj_ptrs)
				break;
		}
		if (max_obj_ptrs > 0 && total_n_obj >= max_obj_ptrs)
			break;
	}

	if (n_entries == 0)
		return NULL;

	/* Reverse to match gh_concat_obj_ptrs output ordering. */
	for (int lo = 0, hi = n_entries - 1; lo < hi; lo++, hi--) {
		int tmp = frame_idxs[lo];
		frame_idxs[lo] = frame_idxs[hi];
		frame_idxs[hi] = tmp;
	}

	int total_rows = n_entries * split;
	int out_dims[] = {total_rows, hidden_dim};
	struct sam3_tensor *out = gh_alloc_tensor(
		a, SAM3_DTYPE_F32, 2, out_dims);
	if (!out)
		return NULL;

	float *dst = (float *)out->data;

	for (int k = 0; k < n_entries; k++) {
		int dist = current_frame_idx - frame_idxs[k];
		if (dist < 0)
			dist = -dist;
		float norm_pos = (float)dist / t_diff_max;

		/*
		 * get_1d_sine_pe: first half sin, second half cos.
		 *   dim_t[j] = temperature ** (2*(j//2) / pe_dim)
		 *   pe[j]           = sin(norm_pos / dim_t[j])
		 *   pe[pe_dim + j]  = cos(norm_pos / dim_t[j])
		 */
		for (int r = 0; r < split; r++) {
			float *row = dst + ((size_t)k * split + r) * hidden_dim;
			for (int j = 0; j < pe_dim; j++) {
				float exponent =
					(float)(2 * (j / 2)) / (float)pe_dim;
				float dim_t = powf(temperature, exponent);
				float v = norm_pos / dim_t;
				row[j]          = sinf(v);
				row[pe_dim + j] = cosf(v);
			}
		}
	}

	return out;
}

struct sam3_tensor *gh_concat_rows(struct sam3_graph *g,
				   struct sam3_arena *a,
				   struct sam3_tensor *x,
				   struct sam3_tensor *y)
{
	if (!x)
		return y;
	if (!y)
		return x;

	struct sam3_tensor *pair[] = {x, y};
	return gh_concat(g, a, pair, 2, 0);
}

struct sam3_tensor *gh_zeros_like(struct sam3_graph *g,
				  struct sam3_arena *a,
				  struct sam3_tensor *src)
{
	(void)g;

	if (!src)
		return NULL;

	struct sam3_tensor *out = gh_alloc_tensor(a, src->dtype,
						  src->n_dims, src->dims);
	if (!out)
		return NULL;

	if (out->data && out->nbytes)
		memset(out->data, 0, out->nbytes);

	return out;
}
