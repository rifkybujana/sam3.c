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

#include "prompt_encoder.h"
#include "graph_helpers.h"

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
	enc->point_proj_w = gh_load_or_alloc(wf,
		"geom_enc.point_proj.weight", arena,
		SAM3_DTYPE_F32, 2, point_w_dims);
	if (!enc->point_proj_w)
		return SAM3_ENOMEM;

	enc->point_proj_b = gh_load_or_alloc(wf,
		"geom_enc.point_proj.bias", arena,
		SAM3_DTYPE_F32, 1, point_b_dims);
	if (!enc->point_proj_b)
		return SAM3_ENOMEM;

	/* Box projection */
	enc->box_proj_w = gh_load_or_alloc(wf,
		"geom_enc.box_proj.weight", arena,
		SAM3_DTYPE_F32, 2, box_w_dims);
	if (!enc->box_proj_w)
		return SAM3_ENOMEM;

	enc->box_proj_b = gh_load_or_alloc(wf,
		"geom_enc.box_proj.bias", arena,
		SAM3_DTYPE_F32, 1, box_b_dims);
	if (!enc->box_proj_b)
		return SAM3_ENOMEM;

	/* CLS token */
	enc->cls_token = gh_load_or_alloc(wf,
		"geom_enc.cls_token", arena,
		SAM3_DTYPE_F32, 2, cls_dims);
	if (!enc->cls_token)
		return SAM3_ENOMEM;

	/* Cross-attention layers */
	for (int i = 0; i < enc->n_layers; i++) {
		/* Q projection */
		snprintf(name, sizeof(name),
			 "geom_enc.layers.%d.ca_q.weight", i);
		enc->layers[i].ca_q_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!enc->layers[i].ca_q_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "geom_enc.layers.%d.ca_q.bias", i);
		enc->layers[i].ca_q_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ca_q_b)
			return SAM3_ENOMEM;

		/* KV projection (packed) */
		snprintf(name, sizeof(name),
			 "geom_enc.layers.%d.ca_kv.weight", i);
		enc->layers[i].ca_kv_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, ca_kv_w_dims);
		if (!enc->layers[i].ca_kv_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "geom_enc.layers.%d.ca_kv.bias", i);
		enc->layers[i].ca_kv_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, ca_kv_b_dims);
		if (!enc->layers[i].ca_kv_b)
			return SAM3_ENOMEM;

		/* Output projection */
		snprintf(name, sizeof(name),
			 "geom_enc.layers.%d.ca_out.weight", i);
		enc->layers[i].ca_out_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!enc->layers[i].ca_out_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "geom_enc.layers.%d.ca_out.bias", i);
		enc->layers[i].ca_out_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ca_out_b)
			return SAM3_ENOMEM;

		/* Layer norm */
		snprintf(name, sizeof(name),
			 "geom_enc.layers.%d.ca_ln.weight", i);
		enc->layers[i].ca_ln_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ca_ln_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "geom_enc.layers.%d.ca_ln.bias", i);
		enc->layers[i].ca_ln_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ca_ln_b)
			return SAM3_ENOMEM;
	}

	/* Post-projection */
	enc->post_proj_w = gh_load_or_alloc(wf,
		"geom_enc.post_proj.weight", arena,
		SAM3_DTYPE_F32, 2, proj_w_dims);
	if (!enc->post_proj_w)
		return SAM3_ENOMEM;

	enc->post_proj_b = gh_load_or_alloc(wf,
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
		ca_out = gh_cross_attention(
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
