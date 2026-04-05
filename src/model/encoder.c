/*
 * src/model/encoder.c - Transformer encoder fusion implementation
 *
 * Implements the DETR encoder (detector_model.detr_encoder) that fuses
 * image features with text features. Each of the 6 layers applies
 * self-attention on image features, cross-attention where image tokens
 * attend to text tokens, and a FFN.
 *
 * Weight loading fuses separate Q/K/V projections from the weight file
 * into the packed QKV format expected by gh_multihead_attention and
 * gh_cross_attention. Cross-attention K/V projections take 256-dim
 * text features as input.
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

/*
 * fuse_3 - Load 3 separate [d, d] weights and fuse into [3d, d].
 *
 * Used for packing Q/K/V projections into a single QKV tensor.
 */
static struct sam3_tensor *fuse_3(const struct sam3_weight_file *wf,
				   const char *name_a,
				   const char *name_b,
				   const char *name_c,
				   struct sam3_arena *arena,
				   int d, int n_dims, const int *part_dims)
{
	struct sam3_tensor *a, *b, *c, *out;
	int fused_dims[2];

	a = gh_load_or_alloc(wf, name_a, arena, SAM3_DTYPE_F32,
			      n_dims, part_dims);
	b = gh_load_or_alloc(wf, name_b, arena, SAM3_DTYPE_F32,
			      n_dims, part_dims);
	c = gh_load_or_alloc(wf, name_c, arena, SAM3_DTYPE_F32,
			      n_dims, part_dims);
	if (!a || !b || !c)
		return NULL;

	if (n_dims == 2) {
		fused_dims[0] = 3 * d;
		fused_dims[1] = part_dims[1];
	} else {
		fused_dims[0] = 3 * d;
	}

	out = gh_alloc_tensor(arena, SAM3_DTYPE_F32, n_dims, fused_dims);
	if (!out)
		return NULL;

	memcpy(out->data, a->data, a->nbytes);
	memcpy((char *)out->data + a->nbytes, b->data, b->nbytes);
	memcpy((char *)out->data + a->nbytes + b->nbytes,
	       c->data, c->nbytes);

	return out;
}

/*
 * fuse_2 - Load 2 separate [d, d] weights and fuse into [2d, d].
 *
 * Used for packing K/V projections into a single KV tensor.
 */
static struct sam3_tensor *fuse_2(const struct sam3_weight_file *wf,
				   const char *name_a,
				   const char *name_b,
				   struct sam3_arena *arena,
				   int d, int n_dims, const int *part_dims)
{
	struct sam3_tensor *a, *b, *out;
	int fused_dims[2];

	a = gh_load_or_alloc(wf, name_a, arena, SAM3_DTYPE_F32,
			      n_dims, part_dims);
	b = gh_load_or_alloc(wf, name_b, arena, SAM3_DTYPE_F32,
			      n_dims, part_dims);
	if (!a || !b)
		return NULL;

	if (n_dims == 2) {
		fused_dims[0] = 2 * d;
		fused_dims[1] = part_dims[1];
	} else {
		fused_dims[0] = 2 * d;
	}

	out = gh_alloc_tensor(arena, SAM3_DTYPE_F32, n_dims, fused_dims);
	if (!out)
		return NULL;

	memcpy(out->data, a->data, a->nbytes);
	memcpy((char *)out->data + a->nbytes, b->data, b->nbytes);

	return out;
}

enum sam3_error sam3_encoder_fusion_load(struct sam3_encoder_fusion *enc,
					 const struct sam3_weight_file *wf,
					 struct sam3_arena *arena)
{
	int d = enc->d_model;
	int ff = enc->d_ffn;
	char q_name[128], k_name[128], v_name[128], name[128];

	int d_dims[] = {d};
	int proj_w_dims[] = {d, d};
	int fc1_w_dims[] = {ff, d};
	int fc1_b_dims[] = {ff};
	int fc2_w_dims[] = {d, ff};

	for (int i = 0; i < enc->n_layers; i++) {
		/*
		 * Self-attention: fuse separate Q/K/V into packed QKV.
		 * File has: self_attn.q_proj, self_attn.k_proj, self_attn.v_proj
		 * All [d, d] / [d].
		 */
		snprintf(q_name, sizeof(q_name),
			 ENC_PREFIX "layers.%d.self_attn.q_proj.weight", i);
		snprintf(k_name, sizeof(k_name),
			 ENC_PREFIX "layers.%d.self_attn.k_proj.weight", i);
		snprintf(v_name, sizeof(v_name),
			 ENC_PREFIX "layers.%d.self_attn.v_proj.weight", i);
		enc->layers[i].sa_qkv_w = fuse_3(wf, q_name, k_name, v_name,
						   arena, d, 2, proj_w_dims);
		if (!enc->layers[i].sa_qkv_w)
			return SAM3_ENOMEM;

		snprintf(q_name, sizeof(q_name),
			 ENC_PREFIX "layers.%d.self_attn.q_proj.bias", i);
		snprintf(k_name, sizeof(k_name),
			 ENC_PREFIX "layers.%d.self_attn.k_proj.bias", i);
		snprintf(v_name, sizeof(v_name),
			 ENC_PREFIX "layers.%d.self_attn.v_proj.bias", i);
		enc->layers[i].sa_qkv_b = fuse_3(wf, q_name, k_name, v_name,
						   arena, d, 1, d_dims);
		if (!enc->layers[i].sa_qkv_b)
			return SAM3_ENOMEM;

		/* Self-attention output projection */
		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.self_attn.o_proj.weight", i);
		enc->layers[i].sa_out_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!enc->layers[i].sa_out_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.self_attn.o_proj.bias", i);
		enc->layers[i].sa_out_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].sa_out_b)
			return SAM3_ENOMEM;

		/* Self-attention layer norm (layer_norm1) */
		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.layer_norm1.weight", i);
		enc->layers[i].sa_ln_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].sa_ln_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.layer_norm1.bias", i);
		enc->layers[i].sa_ln_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].sa_ln_b)
			return SAM3_ENOMEM;

		/* Cross-attention Q projection */
		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.cross_attn.q_proj.weight", i);
		enc->layers[i].ca_q_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!enc->layers[i].ca_q_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.cross_attn.q_proj.bias", i);
		enc->layers[i].ca_q_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ca_q_b)
			return SAM3_ENOMEM;

		/* Cross-attention KV: fuse K + V into packed [2d, d] */
		snprintf(k_name, sizeof(k_name),
			 ENC_PREFIX "layers.%d.cross_attn.k_proj.weight", i);
		snprintf(v_name, sizeof(v_name),
			 ENC_PREFIX "layers.%d.cross_attn.v_proj.weight", i);
		enc->layers[i].ca_kv_w = fuse_2(wf, k_name, v_name,
						  arena, d, 2, proj_w_dims);
		if (!enc->layers[i].ca_kv_w)
			return SAM3_ENOMEM;

		snprintf(k_name, sizeof(k_name),
			 ENC_PREFIX "layers.%d.cross_attn.k_proj.bias", i);
		snprintf(v_name, sizeof(v_name),
			 ENC_PREFIX "layers.%d.cross_attn.v_proj.bias", i);
		enc->layers[i].ca_kv_b = fuse_2(wf, k_name, v_name,
						  arena, d, 1, d_dims);
		if (!enc->layers[i].ca_kv_b)
			return SAM3_ENOMEM;

		/* Cross-attention output projection */
		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.cross_attn.o_proj.weight", i);
		enc->layers[i].ca_out_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!enc->layers[i].ca_out_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.cross_attn.o_proj.bias", i);
		enc->layers[i].ca_out_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ca_out_b)
			return SAM3_ENOMEM;

		/* Cross-attention layer norm (layer_norm2) */
		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.layer_norm2.weight", i);
		enc->layers[i].ca_ln_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ca_ln_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.layer_norm2.bias", i);
		enc->layers[i].ca_ln_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ca_ln_b)
			return SAM3_ENOMEM;

		/* FFN fc1 */
		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.mlp.fc1.weight", i);
		enc->layers[i].ffn_fc1_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, fc1_w_dims);
		if (!enc->layers[i].ffn_fc1_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.mlp.fc1.bias", i);
		enc->layers[i].ffn_fc1_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, fc1_b_dims);
		if (!enc->layers[i].ffn_fc1_b)
			return SAM3_ENOMEM;

		/* FFN fc2 */
		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.mlp.fc2.weight", i);
		enc->layers[i].ffn_fc2_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, fc2_w_dims);
		if (!enc->layers[i].ffn_fc2_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.mlp.fc2.bias", i);
		enc->layers[i].ffn_fc2_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ffn_fc2_b)
			return SAM3_ENOMEM;

		/* FFN layer norm (layer_norm3) */
		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.layer_norm3.weight", i);
		enc->layers[i].ffn_ln_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ffn_ln_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 ENC_PREFIX "layers.%d.layer_norm3.bias", i);
		enc->layers[i].ffn_ln_b = gh_load_or_alloc(
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

struct sam3_tensor *sam3_encoder_fusion_build_layer(
	struct sam3_encoder_fusion *enc,
	int layer_idx,
	struct sam3_graph *g,
	struct sam3_tensor *x,
	struct sam3_tensor *text_features,
	struct sam3_arena *arena)
{
	int n_pixels = x->dims[0];
	int i = layer_idx;

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

	return x;
}

struct sam3_tensor *sam3_encoder_fusion_build_final(
	struct sam3_encoder_fusion *enc,
	struct sam3_graph *g,
	struct sam3_tensor *x,
	struct sam3_arena *arena)
{
	return gh_layernorm(g, arena, x,
			     enc->final_ln_w, enc->final_ln_b);
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
						     text_features, arena);
		if (!x)
			return NULL;
	}

	return sam3_encoder_fusion_build_final(enc, g, x, arena);
}
