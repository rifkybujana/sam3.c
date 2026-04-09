/*
 * src/model/prompt_encoder.c - Geometry encoder graph construction
 *
 * Builds the compute graph for the geometry encoder which encodes
 * point/box prompts into dense embeddings via a 3-layer transformer
 * encoder. Each layer applies: self-attention, cross-attention to
 * image features (with position encoding on keys), and FFN (ReLU).
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
	enc->n_heads = 8;

	return SAM3_OK;
}

enum sam3_error sam3_geometry_encoder_load(
	struct sam3_geometry_encoder *enc,
	const struct sam3_weight_file *wf,
	struct sam3_arena *arena)
{
	int d = enc->d_model;
	int d2 = d * 2;
	int d3 = d * 3;
	int d_ffn = 2048;
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
	int sa_qkv_w_dims[] = {d3, d};
	int sa_qkv_b_dims[] = {d3};
	int ffn_fc1_w_dims[] = {d_ffn, d};
	int ffn_fc1_b_dims[] = {d_ffn};
	int ffn_fc2_w_dims[] = {d, d_ffn};

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

	/* Pool projection: Linear(d_model, d_model) for grid-sampled features */
	enc->pool_proj_w = gh_load_or_alloc(wf,
		"geom_enc.points_pool_project.weight", arena,
		SAM3_DTYPE_F32, 2, proj_w_dims);
	if (!enc->pool_proj_w)
		return SAM3_ENOMEM;

	enc->pool_proj_b = gh_load_or_alloc(wf,
		"geom_enc.points_pool_project.bias", arena,
		SAM3_DTYPE_F32, 1, d_dims);
	if (!enc->pool_proj_b)
		return SAM3_ENOMEM;

	/* Pos enc projection: Linear(d_model, d_model) for sinusoidal encoding */
	enc->posenc_proj_w = gh_load_or_alloc(wf,
		"geom_enc.points_pos_enc_project.weight", arena,
		SAM3_DTYPE_F32, 2, proj_w_dims);
	if (!enc->posenc_proj_w)
		return SAM3_ENOMEM;

	enc->posenc_proj_b = gh_load_or_alloc(wf,
		"geom_enc.points_pos_enc_project.bias", arena,
		SAM3_DTYPE_F32, 1, d_dims);
	if (!enc->posenc_proj_b)
		return SAM3_ENOMEM;

	/* Image pre-norm: LayerNorm(d_model) for pool projection */
	enc->img_pre_norm_w = gh_load_or_alloc(wf,
		"geom_enc.img_pre_norm.weight", arena,
		SAM3_DTYPE_F32, 1, d_dims);
	if (!enc->img_pre_norm_w)
		return SAM3_ENOMEM;

	enc->img_pre_norm_b = gh_load_or_alloc(wf,
		"geom_enc.img_pre_norm.bias", arena,
		SAM3_DTYPE_F32, 1, d_dims);
	if (!enc->img_pre_norm_b)
		return SAM3_ENOMEM;

	/* Label embedding: Embedding(2, d_model) for pos/neg point types */
	{
		int label_dims[] = {2, d};
		enc->n_labels = 2;
		enc->label_embed = gh_load_or_alloc(wf,
			"geom_enc.label_embed.weight", arena,
			SAM3_DTYPE_F32, 2, label_dims);
		if (!enc->label_embed)
			return SAM3_ENOMEM;
	}

	/* Per-layer weights */
	for (int i = 0; i < enc->n_layers; i++) {
		/* Self-attention pre-norm (norm1) */
		snprintf(name, sizeof(name),
			 "geom_enc.layers.%d.norm1.weight", i);
		enc->layers[i].norm1_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32, 1, d_dims);
		if (!enc->layers[i].norm1_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "geom_enc.layers.%d.norm1.bias", i);
		enc->layers[i].norm1_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32, 1, d_dims);
		if (!enc->layers[i].norm1_b)
			return SAM3_ENOMEM;

		/* Self-attention fused QKV [3d, d] */
		snprintf(name, sizeof(name),
			 "geom_enc.layers.%d.self_attn.in_proj_weight", i);
		enc->layers[i].sa_qkv_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, sa_qkv_w_dims);
		if (!enc->layers[i].sa_qkv_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "geom_enc.layers.%d.self_attn.in_proj_bias", i);
		enc->layers[i].sa_qkv_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, sa_qkv_b_dims);
		if (!enc->layers[i].sa_qkv_b)
			return SAM3_ENOMEM;

		/* Self-attention output projection */
		snprintf(name, sizeof(name),
			 "geom_enc.layers.%d.self_attn.out_proj.weight", i);
		enc->layers[i].sa_out_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!enc->layers[i].sa_out_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "geom_enc.layers.%d.self_attn.out_proj.bias", i);
		enc->layers[i].sa_out_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].sa_out_b)
			return SAM3_ENOMEM;

		/* Cross-attention pre-norm (norm2 = ca_ln) */
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

		/* Cross-attention Q projection */
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

		/* Cross-attention KV projection (packed) */
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

		/* Cross-attention output projection */
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

		/* FFN pre-norm (norm3) */
		snprintf(name, sizeof(name),
			 "geom_enc.layers.%d.norm3.weight", i);
		enc->layers[i].norm3_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].norm3_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "geom_enc.layers.%d.norm3.bias", i);
		enc->layers[i].norm3_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].norm3_b)
			return SAM3_ENOMEM;

		/* FFN linear1 [2048, d] */
		snprintf(name, sizeof(name),
			 "geom_enc.layers.%d.linear1.weight", i);
		enc->layers[i].ffn_fc1_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, ffn_fc1_w_dims);
		if (!enc->layers[i].ffn_fc1_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "geom_enc.layers.%d.linear1.bias", i);
		enc->layers[i].ffn_fc1_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, ffn_fc1_b_dims);
		if (!enc->layers[i].ffn_fc1_b)
			return SAM3_ENOMEM;

		/* FFN linear2 [d, 2048] */
		snprintf(name, sizeof(name),
			 "geom_enc.layers.%d.linear2.weight", i);
		enc->layers[i].ffn_fc2_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, ffn_fc2_w_dims);
		if (!enc->layers[i].ffn_fc2_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "geom_enc.layers.%d.linear2.bias", i);
		enc->layers[i].ffn_fc2_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!enc->layers[i].ffn_fc2_b)
			return SAM3_ENOMEM;
	}

	/* Pre-encoder projection (final_proj in Python, applied BEFORE layers) */
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

	/* Pre-encoder LayerNorm (norm in Python, paired with final_proj) */
	enc->norm_w = gh_load_or_alloc(wf,
		"geom_enc.norm.weight", arena,
		SAM3_DTYPE_F32, 1, d_dims);
	if (!enc->norm_w)
		return SAM3_ENOMEM;

	enc->norm_b = gh_load_or_alloc(wf,
		"geom_enc.norm.bias", arena,
		SAM3_DTYPE_F32, 1, d_dims);
	if (!enc->norm_b)
		return SAM3_ENOMEM;

	/* Post-encoder LayerNorm (encode_norm in Python) */
	enc->encode_norm_w = gh_load_or_alloc(wf,
		"geom_enc.encode_norm.weight", arena,
		SAM3_DTYPE_F32, 1, d_dims);
	if (!enc->encode_norm_w)
		return SAM3_ENOMEM;

	enc->encode_norm_b = gh_load_or_alloc(wf,
		"geom_enc.encode_norm.bias", arena,
		SAM3_DTYPE_F32, 1, d_dims);
	if (!enc->encode_norm_b)
		return SAM3_ENOMEM;

	return SAM3_OK;
}

struct sam3_tensor *sam3_geometry_encoder_build(
	struct sam3_geometry_encoder *enc,
	struct sam3_graph *g,
	struct sam3_tensor *prompt_tokens,
	struct sam3_tensor *image_features,
	struct sam3_tensor *image_pos,
	struct sam3_arena *arena)
{
	int d = enc->d_model;
	int n_heads = enc->n_heads;

	/*
	 * Step 1: Append CLS token after prompt tokens.
	 * Python: concat_padded_sequences(final_embeds, cls)
	 * prompt_tokens is [N, d_model], cls_token is [1, d_model].
	 * Result is [N+1, d_model] = [prompt_0, ..., prompt_N-1, CLS].
	 */
	struct sam3_tensor *parts[] = {prompt_tokens, enc->cls_token};
	struct sam3_tensor *x;
	x = gh_concat(g, arena, parts, 2, 0);
	if (!x)
		return NULL;

	/*
	 * Step 2: Pre-encoder projection + LayerNorm.
	 * Python: final_embeds = self.norm(self.final_proj(final_embeds))
	 * Applied BEFORE encoder layers (not after!).
	 */
	x = gh_linear(g, arena, x, enc->post_proj_w, enc->post_proj_b);
	if (!x)
		return NULL;
	x = gh_layernorm(g, arena, x, enc->norm_w, enc->norm_b);
	if (!x)
		return NULL;

	/*
	 * Step 3: Transformer encoder layers.
	 * Each layer: norm1 → self_attn → residual →
	 *             ca_ln → cross_attn(K=img+pos, V=img) → residual →
	 *             norm3 → ffn → residual
	 */
	for (int i = 0; i < enc->n_layers; i++) {
		/* Self-attention: norm1 → MHA → residual */
		struct sam3_tensor *x_norm;
		x_norm = gh_layernorm(g, arena, x,
				       enc->layers[i].norm1_w,
				       enc->layers[i].norm1_b);
		if (!x_norm)
			return NULL;

		/* Reshape to 3D for multihead attention [1, N+1, d] */
		int attn_dims[] = {1, x->dims[0], d};
		struct sam3_tensor *x3d;
		x3d = gh_reshape(g, arena, x_norm, 3, attn_dims);
		if (!x3d)
			return NULL;

		struct sam3_tensor *sa_out;
		sa_out = gh_multihead_attention(g, arena,
			x3d, x3d, x3d,
			enc->layers[i].sa_qkv_w,
			enc->layers[i].sa_qkv_b,
			enc->layers[i].sa_out_w,
			enc->layers[i].sa_out_b,
			n_heads);
		if (!sa_out)
			return NULL;

		x = gh_add(g, arena, x, sa_out);
		if (!x)
			return NULL;

		/*
		 * Cross-attention: ca_ln → Q from x_norm, K from
		 * img+pos, V from img (pos_enc_at_cross_attn_keys=True).
		 * Split packed kv_w [2d,d] into k_w [d,d] + v_w [d,d].
		 */
		x_norm = gh_layernorm(g, arena, x,
				       enc->layers[i].ca_ln_w,
				       enc->layers[i].ca_ln_b);
		if (!x_norm)
			return NULL;

		struct sam3_tensor *q;
		q = gh_linear(g, arena, x_norm,
			       enc->layers[i].ca_q_w,
			       enc->layers[i].ca_q_b);
		if (!q)
			return NULL;

		/* Split packed KV weight/bias for separate K/V sources */
		struct sam3_tensor *k_w, *v_w, *k_b, *v_b;
		k_w = gh_slice(g, arena, enc->layers[i].ca_kv_w,
				0, 0, d);
		v_w = gh_slice(g, arena, enc->layers[i].ca_kv_w,
				0, d, 2 * d);
		k_b = gh_slice(g, arena, enc->layers[i].ca_kv_b,
				0, 0, d);
		v_b = gh_slice(g, arena, enc->layers[i].ca_kv_b,
				0, d, 2 * d);
		if (!k_w || !v_w || !k_b || !v_b)
			return NULL;

		/* K from img+pos, V from img (no pos) */
		struct sam3_tensor *key_src = image_features;
		if (image_pos)
			key_src = gh_add(g, arena, image_features,
					  image_pos);

		struct sam3_tensor *k, *v;
		k = gh_linear(g, arena, key_src, k_w, k_b);
		v = gh_linear(g, arena, image_features, v_w, v_b);
		if (!k || !v)
			return NULL;

		/* Per-head SDPA */
		int head_dim = d / n_heads;
		struct sam3_tensor *head_outs[64];
		for (int h = 0; h < n_heads; h++) {
			int hs = h * head_dim;
			int he = hs + head_dim;
			struct sam3_tensor *hq, *hk, *hv, *ho;
			hq = gh_slice(g, arena, q, 1, hs, he);
			hk = gh_slice(g, arena, k, 1, hs, he);
			hv = gh_slice(g, arena, v, 1, hs, he);
			if (!hq || !hk || !hv)
				return NULL;
			ho = gh_sdpa(g, arena, hq, hk, hv,
				      NULL, head_dim);
			if (!ho)
				return NULL;
			head_outs[h] = ho;
		}

		struct sam3_tensor *merged;
		if (n_heads == 1) {
			merged = head_outs[0];
		} else {
			merged = gh_concat(g, arena, head_outs,
					    n_heads, 1);
			if (!merged)
				return NULL;
		}

		struct sam3_tensor *ca_out;
		ca_out = gh_linear(g, arena, merged,
				    enc->layers[i].ca_out_w,
				    enc->layers[i].ca_out_b);
		if (!ca_out)
			return NULL;

		x = gh_add(g, arena, x, ca_out);
		if (!x)
			return NULL;

		/* FFN: norm3 → linear1 → relu → linear2 → residual */
		x_norm = gh_layernorm(g, arena, x,
				       enc->layers[i].norm3_w,
				       enc->layers[i].norm3_b);
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

	/* Step 4: Post-encoder LayerNorm (encode_norm in Python) */
	x = gh_layernorm(g, arena, x,
			  enc->encode_norm_w, enc->encode_norm_b);

	return x; /* [N+1, d_model] */
}
