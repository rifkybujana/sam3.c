/*
 * src/model/encoder.c - Transformer encoder fusion implementation
 *
 * Implements the 6-layer transformer encoder that fuses image features
 * with text features. Each layer applies self-attention on image features,
 * cross-attention where image tokens attend to text tokens, and a FFN.
 * Cross-attention uses separate Q and KV projections (Q from image,
 * KV from text) with multi-head scaled dot-product attention.
 *
 * Key types:  sam3_encoder_fusion
 * Depends on: encoder.h, graph_helpers.h
 * Used by:    sam3.c (top-level image model)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "encoder.h"
#include "graph_helpers.h"

/*
 * load_or_alloc - Load a weight tensor by name, or allocate zeroed.
 *
 * Mirrors the pattern used in image_encoder.c. When wf is NULL or the
 * tensor is not found, allocates a zero-initialized tensor from the
 * arena.
 */
static struct sam3_tensor *load_or_alloc(const struct sam3_weight_file *wf,
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

/*
 * cross_attention - Multi-head cross-attention with separate Q and KV.
 *
 * Q is projected from q_src, K and V are projected from kv_src.
 * Uses packed KV weight [2*d_model, d_model] which is sliced into
 * separate K and V projections.
 *
 * @g:      Graph to add nodes to
 * @arena:  Arena for intermediate tensors
 * @q_src:  Query source [n_q, d_model]
 * @kv_src: Key/value source [n_kv, d_model]
 * @q_w:    Query projection weight [d_model, d_model]
 * @q_b:    Query projection bias [d_model]
 * @kv_w:   Packed KV projection weight [2*d_model, d_model]
 * @kv_b:   Packed KV projection bias [2*d_model]
 * @out_w:  Output projection weight [d_model, d_model]
 * @out_b:  Output projection bias [d_model]
 * @n_heads: Number of attention heads
 *
 * Returns output tensor [n_q, d_model], or NULL on error.
 */
static struct sam3_tensor *cross_attention(
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

enum sam3_error sam3_encoder_fusion_init(struct sam3_encoder_fusion *enc,
					 int d_model, int n_heads,
					 int n_layers, int d_ffn)
{
	if (n_layers < 1 || n_layers > SAM3_ENC_FUSION_MAX_LAYERS)
		return SAM3_EINVAL;

	memset(enc, 0, sizeof(*enc));
	enc->d_model = d_model;
	enc->n_heads = n_heads;
	enc->n_layers = n_layers;
	enc->d_ffn = d_ffn;

	return SAM3_OK;
}

enum sam3_error sam3_encoder_fusion_load(struct sam3_encoder_fusion *enc,
					 const struct sam3_weight_file *wf,
					 struct sam3_arena *arena)
{
	int d = enc->d_model;
	int d3 = d * 3;
	int d2 = d * 2;
	int ff = enc->d_ffn;
	char name[128];

	int d_dims[] = {d};
	int sa_qkv_w_dims[] = {d3, d};
	int sa_qkv_b_dims[] = {d3};
	int proj_w_dims[] = {d, d};
	int ca_kv_w_dims[] = {d2, d};
	int ca_kv_b_dims[] = {d2};
	int fc1_w_dims[] = {ff, d};
	int fc1_b_dims[] = {ff};
	int fc2_w_dims[] = {d, ff};

	for (int i = 0; i < enc->n_layers; i++) {
		/* Self-attention QKV */
		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.sa.qkv.weight", i);
		enc->layers[i].sa_qkv_w = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, sa_qkv_w_dims);
		if (!enc->layers[i].sa_qkv_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.sa.qkv.bias", i);
		enc->layers[i].sa_qkv_b = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, sa_qkv_b_dims);
		if (!enc->layers[i].sa_qkv_b)
			return SAM3_ENOMEM;

		/* Self-attention output */
		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.sa.out.weight", i);
		enc->layers[i].sa_out_w = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!enc->layers[i].sa_out_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.sa.out.bias", i);
		enc->layers[i].sa_out_b = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].sa_out_b)
			return SAM3_ENOMEM;

		/* Self-attention layer norm */
		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.sa.ln.weight", i);
		enc->layers[i].sa_ln_w = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].sa_ln_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.sa.ln.bias", i);
		enc->layers[i].sa_ln_b = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].sa_ln_b)
			return SAM3_ENOMEM;

		/* Cross-attention Q */
		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.ca.q.weight", i);
		enc->layers[i].ca_q_w = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!enc->layers[i].ca_q_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.ca.q.bias", i);
		enc->layers[i].ca_q_b = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ca_q_b)
			return SAM3_ENOMEM;

		/* Cross-attention KV (packed) */
		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.ca.kv.weight", i);
		enc->layers[i].ca_kv_w = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, ca_kv_w_dims);
		if (!enc->layers[i].ca_kv_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.ca.kv.bias", i);
		enc->layers[i].ca_kv_b = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, ca_kv_b_dims);
		if (!enc->layers[i].ca_kv_b)
			return SAM3_ENOMEM;

		/* Cross-attention output */
		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.ca.out.weight", i);
		enc->layers[i].ca_out_w = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!enc->layers[i].ca_out_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.ca.out.bias", i);
		enc->layers[i].ca_out_b = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ca_out_b)
			return SAM3_ENOMEM;

		/* Cross-attention layer norm */
		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.ca.ln.weight", i);
		enc->layers[i].ca_ln_w = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ca_ln_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.ca.ln.bias", i);
		enc->layers[i].ca_ln_b = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ca_ln_b)
			return SAM3_ENOMEM;

		/* FFN fc1 */
		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.ffn.fc1.weight", i);
		enc->layers[i].ffn_fc1_w = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, fc1_w_dims);
		if (!enc->layers[i].ffn_fc1_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.ffn.fc1.bias", i);
		enc->layers[i].ffn_fc1_b = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, fc1_b_dims);
		if (!enc->layers[i].ffn_fc1_b)
			return SAM3_ENOMEM;

		/* FFN fc2 */
		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.ffn.fc2.weight", i);
		enc->layers[i].ffn_fc2_w = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, fc2_w_dims);
		if (!enc->layers[i].ffn_fc2_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.ffn.fc2.bias", i);
		enc->layers[i].ffn_fc2_b = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ffn_fc2_b)
			return SAM3_ENOMEM;

		/* FFN layer norm */
		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.ffn.ln.weight", i);
		enc->layers[i].ffn_ln_w = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ffn_ln_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.ffn.ln.bias", i);
		enc->layers[i].ffn_ln_b = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ffn_ln_b)
			return SAM3_ENOMEM;
	}

	return SAM3_OK;
}

struct sam3_tensor *sam3_encoder_fusion_build(
	struct sam3_encoder_fusion *enc,
	struct sam3_graph *g,
	struct sam3_tensor *image_features,
	struct sam3_tensor *text_features,
	struct sam3_arena *arena)
{
	int n_pixels = image_features->dims[0];
	struct sam3_tensor *x = image_features;

	for (int i = 0; i < enc->n_layers; i++) {
		/*
		 * Step 1: Self-attention on image features.
		 * Pre-norm, multihead attention, residual.
		 */
		struct sam3_tensor *x_norm;
		x_norm = gh_layernorm(g, arena, x,
				       enc->layers[i].sa_ln_w,
				       enc->layers[i].sa_ln_b);
		if (!x_norm)
			return NULL;

		/* Reshape to 3D for gh_multihead_attention */
		int attn_dims[] = {1, n_pixels, enc->d_model};
		struct sam3_tensor *x3d;
		x3d = gh_reshape(g, arena, x_norm, 3, attn_dims);
		if (!x3d)
			return NULL;

		struct sam3_tensor *sa_out;
		sa_out = gh_multihead_attention(
			g, arena,
			x3d, x3d, x3d,
			enc->layers[i].sa_qkv_w,
			enc->layers[i].sa_qkv_b,
			enc->layers[i].sa_out_w,
			enc->layers[i].sa_out_b,
			enc->n_heads);
		if (!sa_out)
			return NULL;
		/* sa_out is [n_pixels, d_model] */

		x = gh_add(g, arena, x, sa_out);
		if (!x)
			return NULL;

		/*
		 * Step 2: Cross-attention (image attends to text).
		 * Pre-norm on image features, cross-attention, residual.
		 */
		x_norm = gh_layernorm(g, arena, x,
				       enc->layers[i].ca_ln_w,
				       enc->layers[i].ca_ln_b);
		if (!x_norm)
			return NULL;

		struct sam3_tensor *ca_out;
		ca_out = cross_attention(
			g, arena,
			x_norm, text_features,
			enc->layers[i].ca_q_w,
			enc->layers[i].ca_q_b,
			enc->layers[i].ca_kv_w,
			enc->layers[i].ca_kv_b,
			enc->layers[i].ca_out_w,
			enc->layers[i].ca_out_b,
			enc->n_heads);
		if (!ca_out)
			return NULL;

		x = gh_add(g, arena, x, ca_out);
		if (!x)
			return NULL;

		/*
		 * Step 3: FFN with residual.
		 * Pre-norm, MLP (relu activation), residual.
		 */
		x_norm = gh_layernorm(g, arena, x,
				       enc->layers[i].ffn_ln_w,
				       enc->layers[i].ffn_ln_b);
		if (!x_norm)
			return NULL;

		struct sam3_tensor *ff;
		ff = gh_mlp(g, arena, x_norm,
			     enc->layers[i].ffn_fc1_w,
			     enc->layers[i].ffn_fc1_b,
			     enc->layers[i].ffn_fc2_w,
			     enc->layers[i].ffn_fc2_b,
			     SAM3_OP_RELU);
		if (!ff)
			return NULL;

		x = gh_add(g, arena, x, ff);
		if (!x)
			return NULL;
	}

	return x; /* [n_pixels, d_model] */
}
