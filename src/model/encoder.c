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

#include "encoder.h"
#include "graph_helpers.h"

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
		enc->layers[i].sa_qkv_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, sa_qkv_w_dims);
		if (!enc->layers[i].sa_qkv_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.sa.qkv.bias", i);
		enc->layers[i].sa_qkv_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, sa_qkv_b_dims);
		if (!enc->layers[i].sa_qkv_b)
			return SAM3_ENOMEM;

		/* Self-attention output */
		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.sa.out.weight", i);
		enc->layers[i].sa_out_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!enc->layers[i].sa_out_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.sa.out.bias", i);
		enc->layers[i].sa_out_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].sa_out_b)
			return SAM3_ENOMEM;

		/* Self-attention layer norm */
		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.sa.ln.weight", i);
		enc->layers[i].sa_ln_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].sa_ln_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.sa.ln.bias", i);
		enc->layers[i].sa_ln_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].sa_ln_b)
			return SAM3_ENOMEM;

		/* Cross-attention Q */
		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.ca.q.weight", i);
		enc->layers[i].ca_q_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!enc->layers[i].ca_q_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.ca.q.bias", i);
		enc->layers[i].ca_q_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ca_q_b)
			return SAM3_ENOMEM;

		/* Cross-attention KV (packed) */
		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.ca.kv.weight", i);
		enc->layers[i].ca_kv_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, ca_kv_w_dims);
		if (!enc->layers[i].ca_kv_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.ca.kv.bias", i);
		enc->layers[i].ca_kv_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, ca_kv_b_dims);
		if (!enc->layers[i].ca_kv_b)
			return SAM3_ENOMEM;

		/* Cross-attention output */
		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.ca.out.weight", i);
		enc->layers[i].ca_out_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!enc->layers[i].ca_out_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.ca.out.bias", i);
		enc->layers[i].ca_out_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ca_out_b)
			return SAM3_ENOMEM;

		/* Cross-attention layer norm */
		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.ca.ln.weight", i);
		enc->layers[i].ca_ln_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ca_ln_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.ca.ln.bias", i);
		enc->layers[i].ca_ln_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ca_ln_b)
			return SAM3_ENOMEM;

		/* FFN fc1 */
		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.ffn.fc1.weight", i);
		enc->layers[i].ffn_fc1_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, fc1_w_dims);
		if (!enc->layers[i].ffn_fc1_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.ffn.fc1.bias", i);
		enc->layers[i].ffn_fc1_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, fc1_b_dims);
		if (!enc->layers[i].ffn_fc1_b)
			return SAM3_ENOMEM;

		/* FFN fc2 */
		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.ffn.fc2.weight", i);
		enc->layers[i].ffn_fc2_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, fc2_w_dims);
		if (!enc->layers[i].ffn_fc2_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.ffn.fc2.bias", i);
		enc->layers[i].ffn_fc2_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ffn_fc2_b)
			return SAM3_ENOMEM;

		/* FFN layer norm */
		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.ffn.ln.weight", i);
		enc->layers[i].ffn_ln_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ffn_ln_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "enc_fusion.layer.%d.ffn.ln.bias", i);
		enc->layers[i].ffn_ln_b = gh_load_or_alloc(
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
		ca_out = gh_cross_attention(
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
