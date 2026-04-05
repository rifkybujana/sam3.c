/*
 * src/model/decoder.c - Transformer decoder implementation
 *
 * Implements the 6-layer transformer decoder with 200 learned object
 * queries. Each layer applies: (1) self-attention on queries, (2)
 * cross-attention to encoder features, (3) text cross-attention,
 * (4) FFN, and (5) iterative box refinement via a 3-layer MLP.
 * Box coordinates are accumulated as sigmoid(sum of deltas).
 *
 * Cross-attention uses separate Q and KV projections, implemented
 * via the same per-head slice-matmul-softmax pattern used in the
 * encoder fusion module.
 *
 * Weight loading fuses separate Q/K/V projections from the weight
 * file into the packed QKV format expected by gh_multihead_attention
 * and gh_cross_attention.
 *
 * Key types:  sam3_decoder
 * Depends on: decoder.h, graph_helpers.h, util/log.h
 * Used by:    sam3_image.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>

#include "decoder.h"
#include "graph_helpers.h"

#define DEC_PREFIX "detector_model.detr_decoder."

/*
 * box_refine_mlp - 3-layer MLP for box refinement.
 *
 * Produces box delta [n_queries, 4] from query embeddings.
 * Architecture: fc1 -> relu -> fc2 -> relu -> fc3.
 */
static struct sam3_tensor *box_refine_mlp(
	struct sam3_graph *g, struct sam3_arena *arena,
	struct sam3_tensor *queries,
	struct sam3_tensor *fc1_w, struct sam3_tensor *fc1_b,
	struct sam3_tensor *fc2_w, struct sam3_tensor *fc2_b,
	struct sam3_tensor *fc3_w, struct sam3_tensor *fc3_b)
{
	struct sam3_tensor *h;

	h = gh_linear(g, arena, queries, fc1_w, fc1_b);
	if (!h)
		return NULL;

	h = gh_relu(g, arena, h);
	if (!h)
		return NULL;

	h = gh_linear(g, arena, h, fc2_w, fc2_b);
	if (!h)
		return NULL;

	h = gh_relu(g, arena, h);
	if (!h)
		return NULL;

	h = gh_linear(g, arena, h, fc3_w, fc3_b);
	if (!h)
		return NULL;

	return h; /* [n_queries, 4] */
}

enum sam3_error sam3_decoder_init(struct sam3_decoder *dec,
				  int d_model, int n_heads,
				  int n_layers, int d_ffn, int n_queries)
{
	if (n_layers < 1 || n_layers > SAM3_DEC_MAX_LAYERS)
		return SAM3_EINVAL;

	memset(dec, 0, sizeof(*dec));
	dec->d_model = d_model;
	dec->n_heads = n_heads;
	dec->n_layers = n_layers;
	dec->d_ffn = d_ffn;
	dec->n_queries = n_queries;

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

enum sam3_error sam3_decoder_load(struct sam3_decoder *dec,
				  const struct sam3_weight_file *wf,
				  struct sam3_arena *arena)
{
	int d = dec->d_model;
	int ff = dec->d_ffn;
	int nq = dec->n_queries;
	char q_name[128], k_name[128], v_name[128], name[128];

	/* Learned query embeddings: [n_queries, d_model] */
	int qe_dims[] = {nq, d};
	dec->query_embed = gh_load_or_alloc(wf,
		DEC_PREFIX "query_embed.weight",
		arena, SAM3_DTYPE_F32, 2, qe_dims);
	if (!dec->query_embed)
		return SAM3_ENOMEM;

	/* Reusable dimension arrays */
	int d_dims[] = {d};
	int proj_w_dims[] = {d, d};
	int fc1_w_dims[] = {ff, d};
	int fc1_b_dims[] = {ff};
	int fc2_w_dims[] = {d, ff};

	/* Box head dimensions (shared, not per-layer) */
	int box_fc1_w_dims[] = {d, d};
	int box_fc2_w_dims[] = {d, d};
	int box_fc3_w_dims[] = {4, d};
	int box_fc3_b_dims[] = {4};

	/*
	 * Load shared box head once. The weight file stores a single
	 * box_head shared across all decoder layers.
	 */
	struct sam3_tensor *box_fc1_w, *box_fc1_b;
	struct sam3_tensor *box_fc2_w, *box_fc2_b;
	struct sam3_tensor *box_fc3_w, *box_fc3_b;

	box_fc1_w = gh_load_or_alloc(wf,
		DEC_PREFIX "box_head.layer1.weight",
		arena, SAM3_DTYPE_F32, 2, box_fc1_w_dims);
	if (!box_fc1_w)
		return SAM3_ENOMEM;
	box_fc1_b = gh_load_or_alloc(wf,
		DEC_PREFIX "box_head.layer1.bias",
		arena, SAM3_DTYPE_F32, 1, d_dims);
	if (!box_fc1_b)
		return SAM3_ENOMEM;

	box_fc2_w = gh_load_or_alloc(wf,
		DEC_PREFIX "box_head.layer2.weight",
		arena, SAM3_DTYPE_F32, 2, box_fc2_w_dims);
	if (!box_fc2_w)
		return SAM3_ENOMEM;
	box_fc2_b = gh_load_or_alloc(wf,
		DEC_PREFIX "box_head.layer2.bias",
		arena, SAM3_DTYPE_F32, 1, d_dims);
	if (!box_fc2_b)
		return SAM3_ENOMEM;

	box_fc3_w = gh_load_or_alloc(wf,
		DEC_PREFIX "box_head.layer3.weight",
		arena, SAM3_DTYPE_F32, 2, box_fc3_w_dims);
	if (!box_fc3_w)
		return SAM3_ENOMEM;
	box_fc3_b = gh_load_or_alloc(wf,
		DEC_PREFIX "box_head.layer3.bias",
		arena, SAM3_DTYPE_F32, 1, box_fc3_b_dims);
	if (!box_fc3_b)
		return SAM3_ENOMEM;

	for (int i = 0; i < dec->n_layers; i++) {
		/*
		 * Self-attention: fuse q_proj + k_proj + v_proj
		 * into packed QKV [3*d, d] / [3*d].
		 */
		snprintf(q_name, sizeof(q_name),
			 DEC_PREFIX "layers.%d.self_attn.q_proj.weight", i);
		snprintf(k_name, sizeof(k_name),
			 DEC_PREFIX "layers.%d.self_attn.k_proj.weight", i);
		snprintf(v_name, sizeof(v_name),
			 DEC_PREFIX "layers.%d.self_attn.v_proj.weight", i);
		dec->layers[i].sa_qkv_w = fuse_3(wf, q_name, k_name,
						   v_name, arena, d,
						   2, proj_w_dims);
		if (!dec->layers[i].sa_qkv_w)
			return SAM3_ENOMEM;

		snprintf(q_name, sizeof(q_name),
			 DEC_PREFIX "layers.%d.self_attn.q_proj.bias", i);
		snprintf(k_name, sizeof(k_name),
			 DEC_PREFIX "layers.%d.self_attn.k_proj.bias", i);
		snprintf(v_name, sizeof(v_name),
			 DEC_PREFIX "layers.%d.self_attn.v_proj.bias", i);
		dec->layers[i].sa_qkv_b = fuse_3(wf, q_name, k_name,
						   v_name, arena, d,
						   1, d_dims);
		if (!dec->layers[i].sa_qkv_b)
			return SAM3_ENOMEM;

		/* Self-attention output projection */
		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.self_attn.o_proj.weight", i);
		dec->layers[i].sa_out_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!dec->layers[i].sa_out_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.self_attn.o_proj.bias", i);
		dec->layers[i].sa_out_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].sa_out_b)
			return SAM3_ENOMEM;

		/* Self-attention layer norm */
		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.self_attn_layer_norm.weight", i);
		dec->layers[i].sa_ln_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].sa_ln_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.self_attn_layer_norm.bias", i);
		dec->layers[i].sa_ln_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].sa_ln_b)
			return SAM3_ENOMEM;

		/* Vision cross-attention: Q separate, K+V fused */
		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.vision_cross_attn.q_proj.weight", i);
		dec->layers[i].ca_q_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!dec->layers[i].ca_q_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.vision_cross_attn.q_proj.bias", i);
		dec->layers[i].ca_q_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].ca_q_b)
			return SAM3_ENOMEM;

		snprintf(k_name, sizeof(k_name),
			 DEC_PREFIX "layers.%d.vision_cross_attn.k_proj.weight", i);
		snprintf(v_name, sizeof(v_name),
			 DEC_PREFIX "layers.%d.vision_cross_attn.v_proj.weight", i);
		dec->layers[i].ca_kv_w = fuse_2(wf, k_name, v_name,
						  arena, d,
						  2, proj_w_dims);
		if (!dec->layers[i].ca_kv_w)
			return SAM3_ENOMEM;

		snprintf(k_name, sizeof(k_name),
			 DEC_PREFIX "layers.%d.vision_cross_attn.k_proj.bias", i);
		snprintf(v_name, sizeof(v_name),
			 DEC_PREFIX "layers.%d.vision_cross_attn.v_proj.bias", i);
		dec->layers[i].ca_kv_b = fuse_2(wf, k_name, v_name,
						  arena, d,
						  1, d_dims);
		if (!dec->layers[i].ca_kv_b)
			return SAM3_ENOMEM;

		/* Vision cross-attention output */
		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.vision_cross_attn.o_proj.weight", i);
		dec->layers[i].ca_out_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!dec->layers[i].ca_out_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.vision_cross_attn.o_proj.bias", i);
		dec->layers[i].ca_out_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].ca_out_b)
			return SAM3_ENOMEM;

		/* Vision cross-attention layer norm */
		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.vision_cross_attn_layer_norm.weight", i);
		dec->layers[i].ca_ln_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].ca_ln_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.vision_cross_attn_layer_norm.bias", i);
		dec->layers[i].ca_ln_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].ca_ln_b)
			return SAM3_ENOMEM;

		/* Text cross-attention: Q separate, K+V fused */
		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.text_cross_attn.q_proj.weight", i);
		dec->layers[i].tca_q_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!dec->layers[i].tca_q_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.text_cross_attn.q_proj.bias", i);
		dec->layers[i].tca_q_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].tca_q_b)
			return SAM3_ENOMEM;

		snprintf(k_name, sizeof(k_name),
			 DEC_PREFIX "layers.%d.text_cross_attn.k_proj.weight", i);
		snprintf(v_name, sizeof(v_name),
			 DEC_PREFIX "layers.%d.text_cross_attn.v_proj.weight", i);
		dec->layers[i].tca_kv_w = fuse_2(wf, k_name, v_name,
						   arena, d,
						   2, proj_w_dims);
		if (!dec->layers[i].tca_kv_w)
			return SAM3_ENOMEM;

		snprintf(k_name, sizeof(k_name),
			 DEC_PREFIX "layers.%d.text_cross_attn.k_proj.bias", i);
		snprintf(v_name, sizeof(v_name),
			 DEC_PREFIX "layers.%d.text_cross_attn.v_proj.bias", i);
		dec->layers[i].tca_kv_b = fuse_2(wf, k_name, v_name,
						   arena, d,
						   1, d_dims);
		if (!dec->layers[i].tca_kv_b)
			return SAM3_ENOMEM;

		/* Text cross-attention output */
		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.text_cross_attn.o_proj.weight", i);
		dec->layers[i].tca_out_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!dec->layers[i].tca_out_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.text_cross_attn.o_proj.bias", i);
		dec->layers[i].tca_out_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].tca_out_b)
			return SAM3_ENOMEM;

		/* Text cross-attention layer norm */
		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.text_cross_attn_layer_norm.weight", i);
		dec->layers[i].tca_ln_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].tca_ln_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.text_cross_attn_layer_norm.bias", i);
		dec->layers[i].tca_ln_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].tca_ln_b)
			return SAM3_ENOMEM;

		/* FFN fc1 */
		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.mlp.fc1.weight", i);
		dec->layers[i].ffn_fc1_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, fc1_w_dims);
		if (!dec->layers[i].ffn_fc1_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.mlp.fc1.bias", i);
		dec->layers[i].ffn_fc1_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, fc1_b_dims);
		if (!dec->layers[i].ffn_fc1_b)
			return SAM3_ENOMEM;

		/* FFN fc2 */
		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.mlp.fc2.weight", i);
		dec->layers[i].ffn_fc2_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, fc2_w_dims);
		if (!dec->layers[i].ffn_fc2_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.mlp.fc2.bias", i);
		dec->layers[i].ffn_fc2_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].ffn_fc2_b)
			return SAM3_ENOMEM;

		/* FFN layer norm */
		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.mlp_layer_norm.weight", i);
		dec->layers[i].ffn_ln_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].ffn_ln_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.mlp_layer_norm.bias", i);
		dec->layers[i].ffn_ln_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].ffn_ln_b)
			return SAM3_ENOMEM;

		/* Box refinement: shared head, same pointers per layer */
		dec->layers[i].box_fc1_w = box_fc1_w;
		dec->layers[i].box_fc1_b = box_fc1_b;
		dec->layers[i].box_fc2_w = box_fc2_w;
		dec->layers[i].box_fc2_b = box_fc2_b;
		dec->layers[i].box_fc3_w = box_fc3_w;
		dec->layers[i].box_fc3_b = box_fc3_b;
	}

	/* Output layer norm applied after all decoder layers */
	dec->output_ln_w = gh_load_or_alloc(wf,
		DEC_PREFIX "output_layer_norm.weight",
		arena, SAM3_DTYPE_F32, 1, d_dims);
	if (!dec->output_ln_w)
		return SAM3_ENOMEM;

	dec->output_ln_b = gh_load_or_alloc(wf,
		DEC_PREFIX "output_layer_norm.bias",
		arena, SAM3_DTYPE_F32, 1, d_dims);
	if (!dec->output_ln_b)
		return SAM3_ENOMEM;

	return SAM3_OK;
}

struct sam3_tensor *sam3_decoder_build_layer(
	struct sam3_decoder *dec,
	int layer_idx,
	struct sam3_graph *g,
	struct sam3_tensor *q,
	struct sam3_tensor *enc_features,
	struct sam3_tensor *text_features,
	struct sam3_tensor **boxes,
	struct sam3_arena *arena)
{
	int nq = dec->n_queries;
	int d = dec->d_model;
	int i = layer_idx;

	/*
	 * Step (a): Self-attention on queries.
	 * Pre-norm, multihead attention, residual.
	 */
	struct sam3_tensor *q_norm;
	q_norm = gh_layernorm(g, arena, q,
			       dec->layers[i].sa_ln_w,
			       dec->layers[i].sa_ln_b);
	if (!q_norm)
		return NULL;

	int attn_dims[] = {1, nq, d};
	struct sam3_tensor *q3d;
	q3d = gh_reshape(g, arena, q_norm, 3, attn_dims);
	if (!q3d)
		return NULL;

	struct sam3_tensor *sa_out;
	sa_out = gh_multihead_attention(
		g, arena,
		q3d, q3d, q3d,
		dec->layers[i].sa_qkv_w,
		dec->layers[i].sa_qkv_b,
		dec->layers[i].sa_out_w,
		dec->layers[i].sa_out_b,
		dec->n_heads);
	if (!sa_out)
		return NULL;

	q = gh_add(g, arena, q, sa_out);
	if (!q)
		return NULL;

	/*
	 * Step (b): Cross-attention to encoder features.
	 * Pre-norm on queries, cross-attend to encoder, residual.
	 */
	q_norm = gh_layernorm(g, arena, q,
			       dec->layers[i].ca_ln_w,
			       dec->layers[i].ca_ln_b);
	if (!q_norm)
		return NULL;

	struct sam3_tensor *ca_out;
	ca_out = gh_cross_attention(
		g, arena,
		q_norm, enc_features,
		dec->layers[i].ca_q_w,
		dec->layers[i].ca_q_b,
		dec->layers[i].ca_kv_w,
		dec->layers[i].ca_kv_b,
		dec->layers[i].ca_out_w,
		dec->layers[i].ca_out_b,
		dec->n_heads);
	if (!ca_out)
		return NULL;

	q = gh_add(g, arena, q, ca_out);
	if (!q)
		return NULL;

	/*
	 * Step (c): Text cross-attention.
	 * Pre-norm on queries, cross-attend to text, residual.
	 */
	q_norm = gh_layernorm(g, arena, q,
			       dec->layers[i].tca_ln_w,
			       dec->layers[i].tca_ln_b);
	if (!q_norm)
		return NULL;

	struct sam3_tensor *tca_out;
	tca_out = gh_cross_attention(
		g, arena,
		q_norm, text_features,
		dec->layers[i].tca_q_w,
		dec->layers[i].tca_q_b,
		dec->layers[i].tca_kv_w,
		dec->layers[i].tca_kv_b,
		dec->layers[i].tca_out_w,
		dec->layers[i].tca_out_b,
		dec->n_heads);
	if (!tca_out)
		return NULL;

	q = gh_add(g, arena, q, tca_out);
	if (!q)
		return NULL;

	/*
	 * Step (d): FFN with residual.
	 * Pre-norm, MLP (relu activation), residual.
	 */
	q_norm = gh_layernorm(g, arena, q,
			       dec->layers[i].ffn_ln_w,
			       dec->layers[i].ffn_ln_b);
	if (!q_norm)
		return NULL;

	struct sam3_tensor *ff;
	ff = gh_mlp(g, arena, q_norm,
		     dec->layers[i].ffn_fc1_w,
		     dec->layers[i].ffn_fc1_b,
		     dec->layers[i].ffn_fc2_w,
		     dec->layers[i].ffn_fc2_b,
		     SAM3_OP_RELU);
	if (!ff)
		return NULL;

	q = gh_add(g, arena, q, ff);
	if (!q)
		return NULL;

	/*
	 * Step (e): Box refinement (optional).
	 * When boxes is NULL, skip box refinement — this keeps q as
	 * the final graph node so the Metal backend copies it back.
	 * delta = box_mlp(q) -> [n_queries, 4]
	 * boxes = sigmoid(boxes + delta)
	 */
	if (boxes && *boxes) {
		struct sam3_tensor *delta;
		delta = box_refine_mlp(
			g, arena, q,
			dec->layers[i].box_fc1_w,
			dec->layers[i].box_fc1_b,
			dec->layers[i].box_fc2_w,
			dec->layers[i].box_fc2_b,
			dec->layers[i].box_fc3_w,
			dec->layers[i].box_fc3_b);
		if (!delta)
			return NULL;

		*boxes = gh_add(g, arena, *boxes, delta);
		if (!*boxes)
			return NULL;

		*boxes = gh_sigmoid(g, arena, *boxes);
		if (!*boxes)
			return NULL;
	}

	return q; /* [n_queries, d_model] */
}

struct sam3_tensor *sam3_decoder_build_final(
	struct sam3_decoder *dec,
	struct sam3_graph *g,
	struct sam3_tensor *q,
	struct sam3_arena *arena)
{
	q = gh_layernorm(g, arena, q,
			  dec->output_ln_w, dec->output_ln_b);
	return q;
}

struct sam3_tensor *sam3_decoder_build(
	struct sam3_decoder *dec,
	struct sam3_graph *g,
	struct sam3_tensor *enc_features,
	struct sam3_tensor *text_features,
	struct sam3_tensor **box_out,
	struct sam3_arena *arena)
{
	int nq = dec->n_queries;
	struct sam3_tensor *q = dec->query_embed;

	int box_dims[] = {nq, 4};
	struct sam3_tensor *boxes;
	boxes = gh_alloc_tensor(arena, SAM3_DTYPE_F32, 2, box_dims);
	if (!boxes)
		return NULL;
	memset(boxes->data, 0, boxes->nbytes);

	for (int i = 0; i < dec->n_layers; i++) {
		q = sam3_decoder_build_layer(dec, i, g, q,
					      enc_features,
					      text_features,
					      &boxes, arena);
		if (!q)
			return NULL;
	}

	if (box_out)
		*box_out = boxes;

	q = sam3_decoder_build_final(dec, g, q, arena);
	return q;
}
