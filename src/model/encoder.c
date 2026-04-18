/*
 * src/model/encoder.c - Transformer encoder fusion implementation
 *
 * Implements the DETR encoder (detector_model.detr_encoder) that fuses
 * image features with text features. Each of the 6 layers applies
 * self-attention on image features, cross-attention where image tokens
 * attend to text tokens, and a FFN.
 *
 * Weight loading maps separate Q/K/V projections directly from the
 * weight file via mmap. Graph building uses gh_multihead_attention_sep
 * and gh_cross_attention_sep which take separate weight tensors.
 *
 * Key types:  sam3_encoder_fusion
 * Depends on: encoder.h, graph_helpers.h
 * Used by:    sam3_image.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>

#include "encoder.h"
#include "graph_helpers.h"
#include "util/log.h"

#define ENC_PREFIX "detector_model.detr_encoder."

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
	int ff = enc->d_ffn;
	char name[128];

	int d_dims[] = {d};
	int proj_w_dims[] = {d, d};
	int fc1_w_dims[] = {ff, d};
	int fc1_b_dims[] = {ff};
	int fc2_w_dims[] = {d, ff};

	for (int i = 0; i < enc->n_layers; i++) {
		/* Self-attention Q/K/V projections: [d, d] / [d] */
		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.self_attn.q_proj.weight", i);
		enc->layers[i].sa_q_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!enc->layers[i].sa_q_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.self_attn.q_proj.bias", i);
		enc->layers[i].sa_q_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].sa_q_b)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.self_attn.k_proj.weight", i);
		enc->layers[i].sa_k_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!enc->layers[i].sa_k_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.self_attn.k_proj.bias", i);
		enc->layers[i].sa_k_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].sa_k_b)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.self_attn.v_proj.weight", i);
		enc->layers[i].sa_v_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!enc->layers[i].sa_v_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.self_attn.v_proj.bias", i);
		enc->layers[i].sa_v_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].sa_v_b)
			return SAM3_ENOMEM;

		/* Self-attention output projection */
		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.self_attn.o_proj.weight", i);
		enc->layers[i].sa_out_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!enc->layers[i].sa_out_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.self_attn.o_proj.bias", i);
		enc->layers[i].sa_out_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].sa_out_b)
			return SAM3_ENOMEM;

		/* Self-attention layer norm (layer_norm1) */
		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.layer_norm1.weight", i);
		enc->layers[i].sa_ln_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].sa_ln_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.layer_norm1.bias", i);
		enc->layers[i].sa_ln_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].sa_ln_b)
			return SAM3_ENOMEM;

		/* Cross-attention Q projection */
		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.cross_attn.q_proj.weight", i);
		enc->layers[i].ca_q_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!enc->layers[i].ca_q_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.cross_attn.q_proj.bias", i);
		enc->layers[i].ca_q_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ca_q_b)
			return SAM3_ENOMEM;

		/* Cross-attention K/V projections: [d, d] / [d] */
		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.cross_attn.k_proj.weight", i);
		enc->layers[i].ca_k_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!enc->layers[i].ca_k_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.cross_attn.k_proj.bias", i);
		enc->layers[i].ca_k_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ca_k_b)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.cross_attn.v_proj.weight", i);
		enc->layers[i].ca_v_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!enc->layers[i].ca_v_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.cross_attn.v_proj.bias", i);
		enc->layers[i].ca_v_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ca_v_b)
			return SAM3_ENOMEM;

		/* Cross-attention output projection */
		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.cross_attn.o_proj.weight", i);
		enc->layers[i].ca_out_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!enc->layers[i].ca_out_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.cross_attn.o_proj.bias", i);
		enc->layers[i].ca_out_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ca_out_b)
			return SAM3_ENOMEM;

		/* Cross-attention layer norm (layer_norm2) */
		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.layer_norm2.weight", i);
		enc->layers[i].ca_ln_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ca_ln_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.layer_norm2.bias", i);
		enc->layers[i].ca_ln_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ca_ln_b)
			return SAM3_ENOMEM;

		/* FFN fc1 */
		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.mlp.fc1.weight", i);
		enc->layers[i].ffn_fc1_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			2, fc1_w_dims);
		if (!enc->layers[i].ffn_fc1_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.mlp.fc1.bias", i);
		enc->layers[i].ffn_fc1_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, fc1_b_dims);
		if (!enc->layers[i].ffn_fc1_b)
			return SAM3_ENOMEM;

		/* FFN fc2 */
		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.mlp.fc2.weight", i);
		enc->layers[i].ffn_fc2_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			2, fc2_w_dims);
		if (!enc->layers[i].ffn_fc2_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.mlp.fc2.bias", i);
		enc->layers[i].ffn_fc2_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ffn_fc2_b)
			return SAM3_ENOMEM;

		/* FFN layer norm (layer_norm3) */
		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.layer_norm3.weight", i);
		enc->layers[i].ffn_ln_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ffn_ln_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.layer_norm3.bias", i);
		enc->layers[i].ffn_ln_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ffn_ln_b)
			return SAM3_ENOMEM;
	}

	/*
	 * Final layer norm: the DETR encoder has no top-level layer norm.
	 * Allocate identity norm (weight=1, bias=0) so the build code
	 * can unconditionally apply it as a no-op.
	 */
	enc->final_ln_w = gh_alloc_tensor(arena, SAM3_DTYPE_F32, 1, d_dims);
	if (!enc->final_ln_w)
		return SAM3_ENOMEM;
	{
		float *w = (float *)enc->final_ln_w->data;
		for (int j = 0; j < d; j++)
			w[j] = 1.0f;
	}

	enc->final_ln_b = gh_alloc_tensor(arena, SAM3_DTYPE_F32, 1, d_dims);
	if (!enc->final_ln_b)
		return SAM3_ENOMEM;
	/* bias is already zero from arena alloc */

	return SAM3_OK;
}

/*
 * encoder_self_attention_with_pos - Self-attention adding pos to Q/K only.
 *
 * Python: Q = K = layernorm(x) + pos, V = layernorm(x)
 *
 * Q and K are projected from (x_norm + pos) while V is projected from
 * x_norm alone (no positional encoding in V).
 */
static struct sam3_tensor *encoder_self_attention_with_pos(
	struct sam3_graph *g, struct sam3_arena *arena,
	struct sam3_tensor *x_norm,
	struct sam3_tensor *enc_pos,
	struct sam3_tensor *q_w, struct sam3_tensor *q_b,
	struct sam3_tensor *k_w, struct sam3_tensor *k_b,
	struct sam3_tensor *v_w, struct sam3_tensor *v_b,
	struct sam3_tensor *out_w, struct sam3_tensor *out_b,
	int d_model, int n_heads)
{
	int head_dim = d_model / n_heads;

	/* x_pos = x_norm + pos: source for Q and K */
	struct sam3_tensor *x_pos = gh_add(g, arena, x_norm, enc_pos);
	if (!x_pos)
		return NULL;

	/* Q and K from x_pos, V from x_norm (no pos) */
	struct sam3_tensor *q = gh_linear(g, arena, x_pos, q_w, q_b);
	struct sam3_tensor *k = gh_linear(g, arena, x_pos, k_w, k_b);
	struct sam3_tensor *v = gh_linear(g, arena, x_norm, v_w, v_b);
	if (!q || !k || !v)
		return NULL;

	/* Per-head SDPA */
	struct sam3_tensor *head_outs[64];
	for (int h = 0; h < n_heads; h++) {
		int hstart = h * head_dim;
		int hend = hstart + head_dim;

		struct sam3_tensor *hq, *hk, *hv;
		hq = gh_slice(g, arena, q, 1, hstart, hend);
		hk = gh_slice(g, arena, k, 1, hstart, hend);
		hv = gh_slice(g, arena, v, 1, hstart, hend);
		if (!hq || !hk || !hv)
			return NULL;

		struct sam3_tensor *ho;
		ho = gh_sdpa(g, arena, hq, hk, hv, NULL, head_dim);
		if (!ho)
			return NULL;

		head_outs[h] = ho;
	}

	/* Concatenate heads: [n_pixels, d_model] */
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

struct sam3_tensor *sam3_encoder_fusion_build_layer(
	struct sam3_encoder_fusion *enc,
	int layer_idx,
	struct sam3_graph *g,
	struct sam3_tensor *x,
	struct sam3_tensor *enc_pos,
	struct sam3_tensor *text_features,
	struct sam3_arena *arena)
{
	int n_pixels = x->dims[0];
	int i = layer_idx;

	/*
	 * Step 1: Self-attention on image features.
	 * Pre-norm, then Q=K=x+pos, V=x (if pos provided), residual.
	 */
	struct sam3_tensor *x_norm;
	x_norm = gh_layernorm(g, arena, x,
			       enc->layers[i].sa_ln_w,
			       enc->layers[i].sa_ln_b);
	if (!x_norm)
		return NULL;

	struct sam3_tensor *sa_out;
	if (enc_pos) {
		sa_out = encoder_self_attention_with_pos(
			g, arena, x_norm, enc_pos,
			enc->layers[i].sa_q_w,
			enc->layers[i].sa_q_b,
			enc->layers[i].sa_k_w,
			enc->layers[i].sa_k_b,
			enc->layers[i].sa_v_w,
			enc->layers[i].sa_v_b,
			enc->layers[i].sa_out_w,
			enc->layers[i].sa_out_b,
			enc->d_model, enc->n_heads);
	} else {
		/* Reshape to 3D for gh_multihead_attention_sep */
		int attn_dims[] = {1, n_pixels, enc->d_model};
		struct sam3_tensor *x3d;
		x3d = gh_reshape(g, arena, x_norm, 3, attn_dims);
		if (!x3d)
			return NULL;

		sa_out = gh_multihead_attention_sep(
			g, arena, x3d,
			enc->layers[i].sa_q_w,
			enc->layers[i].sa_q_b,
			enc->layers[i].sa_k_w,
			enc->layers[i].sa_k_b,
			enc->layers[i].sa_v_w,
			enc->layers[i].sa_v_b,
			enc->layers[i].sa_out_w,
			enc->layers[i].sa_out_b,
			enc->n_heads, NULL);
	}
	if (!sa_out)
		return NULL;

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
	ca_out = gh_cross_attention_sep(
		g, arena,
		x_norm, text_features, NULL,
		enc->layers[i].ca_q_w,
		enc->layers[i].ca_q_b,
		enc->layers[i].ca_k_w,
		enc->layers[i].ca_k_b,
		enc->layers[i].ca_v_w,
		enc->layers[i].ca_v_b,
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

	return x;
}

struct sam3_tensor *sam3_encoder_fusion_build_final(
	struct sam3_encoder_fusion *enc,
	struct sam3_graph *g,
	struct sam3_tensor *x,
	struct sam3_arena *arena)
{
	/*
	 * The DETR encoder has no top-level layer norm in the
	 * checkpoint — return the input unchanged. A LayerNorm
	 * with w=1/b=0 is NOT identity (it still normalizes).
	 */
	(void)enc;
	(void)g;
	(void)arena;
	return x;
}

struct sam3_tensor *sam3_encoder_fusion_build(
	struct sam3_encoder_fusion *enc,
	struct sam3_graph *g,
	struct sam3_tensor *image_features,
	struct sam3_tensor *text_features,
	struct sam3_arena *arena)
{
	struct sam3_tensor *x = image_features;

	for (int i = 0; i < enc->n_layers; i++) {
		x = sam3_encoder_fusion_build_layer(enc, i, g, x,
						     NULL, /* enc_pos */
						     text_features, arena);
		if (!x)
			return NULL;
	}

	return sam3_encoder_fusion_build_final(enc, g, x, arena);
}
