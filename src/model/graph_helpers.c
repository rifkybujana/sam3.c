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
#include <math.h>

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

	t->data = sam3_arena_alloc(arena, t->nbytes);
	if (!t->data)
		return NULL;

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
			t = gh_alloc_tensor(arena, dtype, n_dims, dims);
			if (t)
				sam3_weight_to_tensor(wf, desc, t);
			return t;
		}
	}
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

	return sam3_graph_add_op(g, SAM3_OP_TRANSPOSE,
				 (struct sam3_tensor *[]){input}, 1, out);
}

/* ── Concat ──────────────────────────────────────────────────────── */

struct sam3_tensor *gh_concat(struct sam3_graph *g, struct sam3_arena *a,
			      struct sam3_tensor **tensors, int n, int axis)
{
	int out_dims[SAM3_MAX_DIMS];
	int nd = tensors[0]->n_dims;

	for (int i = 0; i < nd; i++)
		out_dims[i] = tensors[0]->dims[i];

	/* Sum along the concat axis */
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

	/* Set axis param on the node we just added */
	g->nodes[g->n_nodes - 1].params[0] = axis;
	return result;
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
	/*
	 * q is [batch, seq, d_model].
	 * qkv_w is [3*d_model, d_model].
	 *
	 * All intermediate computation uses 2D tensors so that the
	 * existing CPU transpose and matmul kernels (which are 2D-only)
	 * work correctly.  The batch dimension is flattened into seq
	 * for the QKV projection, and each head is processed via
	 * slice + reshape to stay in 2D.  When batch*heads > 1, a
	 * backend with batched matmul support would be needed for
	 * correct per-head attention; this implementation produces
	 * correct results for single-batch single-head or when the
	 * backend is extended.
	 */
	int batch = q->dims[0];
	int seq_q = q->dims[1];
	int d_model = q->dims[2];
	int head_dim = d_model / n_heads;
	int bs = batch * seq_q;

	(void)k;
	(void)v;

	/*
	 * Step 1: Flatten input to 2D and project QKV.
	 * [batch, seq, d_model] -> [batch*seq, d_model]
	 */
	int flat_dims[] = {bs, d_model};
	struct sam3_tensor *qf = gh_reshape(g, a, q, 2, flat_dims);
	if (!qf)
		return NULL;

	struct sam3_tensor *qkv = gh_linear(g, a, qf, qkv_w, qkv_b);
	if (!qkv)
		return NULL;
	/* qkv is [batch*seq, 3*d_model] */

	/* Step 2: Slice into Q, K, V along last axis (dim 1) */
	struct sam3_tensor *sq = gh_slice(g, a, qkv, 1, 0, d_model);
	struct sam3_tensor *sk = gh_slice(g, a, qkv, 1,
					  d_model, 2 * d_model);
	struct sam3_tensor *sv = gh_slice(g, a, qkv, 1,
					  2 * d_model, 3 * d_model);
	if (!sq || !sk || !sv)
		return NULL;
	/* Each is [batch*seq, d_model] = [bs, d_model] */

	/*
	 * Step 2b: Apply RoPE to Q and K if requested.
	 *
	 * Reshape from [bs, d_model] to [batch, seq_q, n_heads, head_dim],
	 * apply gh_rope, then reshape back to [bs, d_model].
	 */
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
		if (!sk)
			return NULL;
	}

	/*
	 * Step 3-5: Attention per head.
	 *
	 * For n_heads=1, this is simply:
	 *   scores = Q @ K^T  -> [bs, bs]
	 *   scale, softmax, attn_out = scores @ V
	 *
	 * For n_heads>1 with 2D-only ops, we slice each head's
	 * columns, compute attention, and concatenate.
	 */
	struct sam3_tensor *head_outs[SAM3_MAX_DIMS * 16]; /* up to 64 heads */
	for (int h = 0; h < n_heads; h++) {
		int hstart = h * head_dim;
		int hend = hstart + head_dim;

		/* Slice head columns: [bs, head_dim] */
		struct sam3_tensor *hq = gh_slice(g, a, sq, 1,
						   hstart, hend);
		struct sam3_tensor *hk = gh_slice(g, a, sk, 1,
						   hstart, hend);
		struct sam3_tensor *hv = gh_slice(g, a, sv, 1,
						   hstart, hend);
		if (!hq || !hk || !hv)
			return NULL;

		/* K^T: [head_dim, bs] */
		struct sam3_tensor *hkt = gh_transpose(g, a, hk);
		if (!hkt)
			return NULL;

		/* scores = Q_h @ K_h^T -> [bs, bs] */
		struct sam3_tensor *scores = gh_matmul(g, a, hq, hkt);
		if (!scores)
			return NULL;

		/*
		 * Scale by 1/sqrt(head_dim).
		 * Broadcast requires b to be [N] where N is a's last
		 * dim.  scores is [bs, bs], so we use a [bs] tensor.
		 */
		int scale_dims[] = {bs};
		struct sam3_tensor *scale_t = gh_alloc_tensor(
			a, q->dtype, 1, scale_dims);
		if (!scale_t)
			return NULL;
		float inv_sqrt = 1.0f / sqrtf((float)head_dim);
		float *sd = (float *)scale_t->data;
		for (int i = 0; i < bs; i++)
			sd[i] = inv_sqrt;

		struct sam3_tensor *scaled = gh_mul(g, a, scores, scale_t);
		if (!scaled)
			return NULL;

		/* Apply causal mask before softmax if provided */
		if (attn_mask) {
			scaled = gh_add(g, a, scaled, attn_mask);
			if (!scaled)
				return NULL;
		}

		/* softmax */
		struct sam3_tensor *attn = gh_softmax(g, a, scaled);
		if (!attn)
			return NULL;

		/* attn_out = attn @ V_h -> [bs, head_dim] */
		struct sam3_tensor *ho = gh_matmul(g, a, attn, hv);
		if (!ho)
			return NULL;

		head_outs[h] = ho;
	}

	/*
	 * Step 9: Concatenate heads along last axis.
	 * Each head_out is [bs, head_dim], concat -> [bs, d_model].
	 */
	struct sam3_tensor *merged;
	if (n_heads == 1) {
		merged = head_outs[0];
	} else {
		merged = gh_concat(g, a, head_outs, n_heads, 1);
		if (!merged)
			return NULL;
	}

	/* Step 10: Output projection */
	return gh_linear(g, a, merged, out_w, out_b);
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
	int n_kv = kv_src->dims[0];
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
	 * Per-head attention using 2D ops.
	 * Each head: slice columns, compute scaled dot-product attention.
	 */
	struct sam3_tensor *head_outs[64]; /* up to 64 heads */
	for (int h = 0; h < n_heads; h++) {
		int hstart = h * head_dim;
		int hend = hstart + head_dim;

		/* Slice head columns: [n_q/n_kv, head_dim] */
		struct sam3_tensor *hq, *hk, *hv;
		hq = gh_slice(g, arena, q, 1, hstart, hend);
		hk = gh_slice(g, arena, k, 1, hstart, hend);
		hv = gh_slice(g, arena, v, 1, hstart, hend);
		if (!hq || !hk || !hv)
			return NULL;

		/* K^T: [head_dim, n_kv] */
		struct sam3_tensor *hkt = gh_transpose(g, arena, hk);
		if (!hkt)
			return NULL;

		/* scores = Q_h @ K_h^T -> [n_q, n_kv] */
		struct sam3_tensor *scores = gh_matmul(g, arena, hq, hkt);
		if (!scores)
			return NULL;

		/* Scale by 1/sqrt(head_dim) */
		int scale_dims[] = {n_kv};
		struct sam3_tensor *scale_t;
		scale_t = gh_alloc_tensor(arena, q_src->dtype,
					   1, scale_dims);
		if (!scale_t)
			return NULL;

		float inv_sqrt = 1.0f / sqrtf((float)head_dim);
		float *sd = (float *)scale_t->data;
		for (int i = 0; i < n_kv; i++)
			sd[i] = inv_sqrt;

		struct sam3_tensor *scaled;
		scaled = gh_mul(g, arena, scores, scale_t);
		if (!scaled)
			return NULL;

		/* softmax */
		struct sam3_tensor *attn = gh_softmax(g, arena, scaled);
		if (!attn)
			return NULL;

		/* attn_out = attn @ V_h -> [n_q, head_dim] */
		struct sam3_tensor *ho = gh_matmul(g, arena, attn, hv);
		if (!ho)
			return NULL;

		head_outs[h] = ho;
	}

	/* Concatenate heads: [n_q, d_model] */
	struct sam3_tensor *merged;
	if (n_heads == 1) {
		merged = head_outs[0];
	} else {
		merged = gh_concat(g, arena, head_outs, n_heads, 1);
		if (!merged)
			return NULL;
	}

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

/* ── Convolution helpers ──────────────────────────────────────────── */

/*
 * conv_add_bias - Add bias [C_out] to a 4D NCHW tensor.
 *
 * Reshapes to 2D, transposes so channels are last dim for broadcast,
 * adds bias, transposes and reshapes back.
 */
static struct sam3_tensor *conv_add_bias(struct sam3_graph *g,
					 struct sam3_arena *a,
					 struct sam3_tensor *x,
					 struct sam3_tensor *bias)
{
	int n = x->dims[0];
	int c = x->dims[1];
	int h = x->dims[2];
	int w = x->dims[3];
	int n_spatial = h * w;

	/* [N, C, H, W] -> [C, N*H*W] */
	int flat2d[] = {c, n * n_spatial};
	struct sam3_tensor *flat = gh_reshape(g, a, x, 2, flat2d);
	if (!flat)
		return NULL;

	/* Transpose -> [N*H*W, C] for bias broadcast */
	flat = gh_transpose(g, a, flat);
	if (!flat)
		return NULL;

	/* Add bias [C] */
	flat = gh_add(g, a, flat, bias);
	if (!flat)
		return NULL;

	/* Transpose back -> [C, N*H*W] */
	flat = gh_transpose(g, a, flat);
	if (!flat)
		return NULL;

	/* Reshape back to NCHW */
	int nchw[] = {n, c, h, w};
	return gh_reshape(g, a, flat, 4, nchw);
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
