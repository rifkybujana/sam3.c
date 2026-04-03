/*
 * src/model/prompt_encoder.c - Geometry encoder graph construction
 *
 * Builds the compute graph for the geometry encoder which encodes
 * point/box prompts into dense embeddings via cross-attention to image
 * features. Uses the same cross-attention pattern as the transformer
 * encoder fusion: separate Q projection from prompt tokens, packed KV
 * projection from image features, multi-head scaled dot-product.
 *
 * Key types:  sam3_geometry_encoder
 * Depends on: prompt_encoder.h, graph_helpers.h
 * Used by:    sam3.c (top-level segmentation pipeline)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "prompt_encoder.h"
#include "graph_helpers.h"

/*
 * load_or_alloc - Load a weight tensor by name, or allocate zeroed.
 *
 * When wf is NULL or the tensor is not found, allocates a
 * zero-initialized tensor from the arena.
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
		struct sam3_tensor *scores;
		scores = gh_matmul(g, arena, hq, hkt);
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
		struct sam3_tensor *attn;
		attn = gh_softmax(g, arena, scaled);
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

enum sam3_error sam3_geometry_encoder_init(
	struct sam3_geometry_encoder *enc,
	int d_model, int n_layers)
{
	if (n_layers < 1 || n_layers > SAM3_GEOM_ENC_MAX_LAYERS)
		return SAM3_EINVAL;

	memset(enc, 0, sizeof(*enc));
	enc->d_model = d_model;
	enc->n_layers = n_layers;

	return SAM3_OK;
}

enum sam3_error sam3_geometry_encoder_load(
	struct sam3_geometry_encoder *enc,
	const struct sam3_weight_file *wf,
	struct sam3_arena *arena)
{
	int d = enc->d_model;
	int d2 = d * 2;
	char name[128];

	int d_dims[] = {d};
	int point_w_dims[] = {d, 2};
	int point_b_dims[] = {d};
	int box_w_dims[] = {d, 4};
	int box_b_dims[] = {d};
	int cls_dims[] = {1, d};
	int proj_w_dims[] = {d, d};
	int ca_kv_w_dims[] = {d2, d};
	int ca_kv_b_dims[] = {d2};

	/* Point projection */
	enc->point_proj_w = load_or_alloc(wf,
		"geom_enc.point_proj.weight", arena,
		SAM3_DTYPE_F32, 2, point_w_dims);
	if (!enc->point_proj_w)
		return SAM3_ENOMEM;

	enc->point_proj_b = load_or_alloc(wf,
		"geom_enc.point_proj.bias", arena,
		SAM3_DTYPE_F32, 1, point_b_dims);
	if (!enc->point_proj_b)
		return SAM3_ENOMEM;

	/* Box projection */
	enc->box_proj_w = load_or_alloc(wf,
		"geom_enc.box_proj.weight", arena,
		SAM3_DTYPE_F32, 2, box_w_dims);
	if (!enc->box_proj_w)
		return SAM3_ENOMEM;

	enc->box_proj_b = load_or_alloc(wf,
		"geom_enc.box_proj.bias", arena,
		SAM3_DTYPE_F32, 1, box_b_dims);
	if (!enc->box_proj_b)
		return SAM3_ENOMEM;

	/* CLS token */
	enc->cls_token = load_or_alloc(wf,
		"geom_enc.cls_token", arena,
		SAM3_DTYPE_F32, 2, cls_dims);
	if (!enc->cls_token)
		return SAM3_ENOMEM;

	/* Cross-attention layers */
	for (int i = 0; i < enc->n_layers; i++) {
		/* Q projection */
		snprintf(name, sizeof(name),
			 "geom_enc.layers.%d.ca_q.weight", i);
		enc->layers[i].ca_q_w = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!enc->layers[i].ca_q_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "geom_enc.layers.%d.ca_q.bias", i);
		enc->layers[i].ca_q_b = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ca_q_b)
			return SAM3_ENOMEM;

		/* KV projection (packed) */
		snprintf(name, sizeof(name),
			 "geom_enc.layers.%d.ca_kv.weight", i);
		enc->layers[i].ca_kv_w = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, ca_kv_w_dims);
		if (!enc->layers[i].ca_kv_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "geom_enc.layers.%d.ca_kv.bias", i);
		enc->layers[i].ca_kv_b = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, ca_kv_b_dims);
		if (!enc->layers[i].ca_kv_b)
			return SAM3_ENOMEM;

		/* Output projection */
		snprintf(name, sizeof(name),
			 "geom_enc.layers.%d.ca_out.weight", i);
		enc->layers[i].ca_out_w = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!enc->layers[i].ca_out_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "geom_enc.layers.%d.ca_out.bias", i);
		enc->layers[i].ca_out_b = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ca_out_b)
			return SAM3_ENOMEM;

		/* Layer norm */
		snprintf(name, sizeof(name),
			 "geom_enc.layers.%d.ca_ln.weight", i);
		enc->layers[i].ca_ln_w = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ca_ln_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "geom_enc.layers.%d.ca_ln.bias", i);
		enc->layers[i].ca_ln_b = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ca_ln_b)
			return SAM3_ENOMEM;
	}

	/* Post-projection */
	enc->post_proj_w = load_or_alloc(wf,
		"geom_enc.post_proj.weight", arena,
		SAM3_DTYPE_F32, 2, proj_w_dims);
	if (!enc->post_proj_w)
		return SAM3_ENOMEM;

	enc->post_proj_b = load_or_alloc(wf,
		"geom_enc.post_proj.bias", arena,
		SAM3_DTYPE_F32, 1, d_dims);
	if (!enc->post_proj_b)
		return SAM3_ENOMEM;

	return SAM3_OK;
}

struct sam3_tensor *sam3_geometry_encoder_build(
	struct sam3_geometry_encoder *enc,
	struct sam3_graph *g,
	struct sam3_tensor *prompt_tokens,
	struct sam3_tensor *image_features,
	struct sam3_arena *arena)
{
	int n_heads = enc->d_model / 32;  /* head_dim=32 -> 8 heads */

	/*
	 * Step 1: Prepend CLS token.
	 * prompt_tokens is [N, d_model], cls_token is [1, d_model].
	 * Result is [N+1, d_model].
	 */
	struct sam3_tensor *parts[] = {enc->cls_token, prompt_tokens};
	struct sam3_tensor *x;
	x = gh_concat(g, arena, parts, 2, 0);
	if (!x)
		return NULL;

	/*
	 * Step 2: Cross-attention layers.
	 * Each layer: LayerNorm on x, cross-attend to image, residual.
	 */
	for (int i = 0; i < enc->n_layers; i++) {
		struct sam3_tensor *x_norm;
		x_norm = gh_layernorm(g, arena, x,
				       enc->layers[i].ca_ln_w,
				       enc->layers[i].ca_ln_b);
		if (!x_norm)
			return NULL;

		struct sam3_tensor *ca_out;
		ca_out = cross_attention(
			g, arena,
			x_norm, image_features,
			enc->layers[i].ca_q_w,
			enc->layers[i].ca_q_b,
			enc->layers[i].ca_kv_w,
			enc->layers[i].ca_kv_b,
			enc->layers[i].ca_out_w,
			enc->layers[i].ca_out_b,
			n_heads);
		if (!ca_out)
			return NULL;

		x = gh_add(g, arena, x, ca_out);
		if (!x)
			return NULL;
	}

	/* Step 3: Post-projection (linear) */
	x = gh_linear(g, arena, x, enc->post_proj_w, enc->post_proj_b);

	return x; /* [N+1, d_model] */
}
