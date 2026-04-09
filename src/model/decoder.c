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

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "decoder.h"
#include "graph_helpers.h"
#include "util/log.h"

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

	a = gh_load_mmap(wf, name_a, arena, SAM3_DTYPE_F32,
			      n_dims, part_dims);
	b = gh_load_mmap(wf, name_b, arena, SAM3_DTYPE_F32,
			      n_dims, part_dims);
	c = gh_load_mmap(wf, name_c, arena, SAM3_DTYPE_F32,
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

	a = gh_load_mmap(wf, name_a, arena, SAM3_DTYPE_F32,
			      n_dims, part_dims);
	b = gh_load_mmap(wf, name_b, arena, SAM3_DTYPE_F32,
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
	dec->query_embed = gh_load_mmap(wf,
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

	box_fc1_w = gh_load_mmap(wf,
		DEC_PREFIX "box_head.layer1.weight",
		arena, SAM3_DTYPE_F32, 2, box_fc1_w_dims);
	if (!box_fc1_w)
		return SAM3_ENOMEM;
	box_fc1_b = gh_load_mmap(wf,
		DEC_PREFIX "box_head.layer1.bias",
		arena, SAM3_DTYPE_F32, 1, d_dims);
	if (!box_fc1_b)
		return SAM3_ENOMEM;

	box_fc2_w = gh_load_mmap(wf,
		DEC_PREFIX "box_head.layer2.weight",
		arena, SAM3_DTYPE_F32, 2, box_fc2_w_dims);
	if (!box_fc2_w)
		return SAM3_ENOMEM;
	box_fc2_b = gh_load_mmap(wf,
		DEC_PREFIX "box_head.layer2.bias",
		arena, SAM3_DTYPE_F32, 1, d_dims);
	if (!box_fc2_b)
		return SAM3_ENOMEM;

	box_fc3_w = gh_load_mmap(wf,
		DEC_PREFIX "box_head.layer3.weight",
		arena, SAM3_DTYPE_F32, 2, box_fc3_w_dims);
	if (!box_fc3_w)
		return SAM3_ENOMEM;
	box_fc3_b = gh_load_mmap(wf,
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
		dec->layers[i].sa_out_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!dec->layers[i].sa_out_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.self_attn.o_proj.bias", i);
		dec->layers[i].sa_out_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].sa_out_b)
			return SAM3_ENOMEM;

		/* Self-attention layer norm */
		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.self_attn_layer_norm.weight", i);
		dec->layers[i].sa_ln_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].sa_ln_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.self_attn_layer_norm.bias", i);
		dec->layers[i].sa_ln_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].sa_ln_b)
			return SAM3_ENOMEM;

		/* Vision cross-attention: Q separate, K+V fused */
		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.vision_cross_attn.q_proj.weight", i);
		dec->layers[i].ca_q_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!dec->layers[i].ca_q_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.vision_cross_attn.q_proj.bias", i);
		dec->layers[i].ca_q_b = gh_load_mmap(
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
		dec->layers[i].ca_out_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!dec->layers[i].ca_out_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.vision_cross_attn.o_proj.bias", i);
		dec->layers[i].ca_out_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].ca_out_b)
			return SAM3_ENOMEM;

		/* Vision cross-attention layer norm */
		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.vision_cross_attn_layer_norm.weight", i);
		dec->layers[i].ca_ln_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].ca_ln_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.vision_cross_attn_layer_norm.bias", i);
		dec->layers[i].ca_ln_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].ca_ln_b)
			return SAM3_ENOMEM;

		/* Text cross-attention: Q separate, K+V fused */
		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.text_cross_attn.q_proj.weight", i);
		dec->layers[i].tca_q_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!dec->layers[i].tca_q_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.text_cross_attn.q_proj.bias", i);
		dec->layers[i].tca_q_b = gh_load_mmap(
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
		dec->layers[i].tca_out_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!dec->layers[i].tca_out_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.text_cross_attn.o_proj.bias", i);
		dec->layers[i].tca_out_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].tca_out_b)
			return SAM3_ENOMEM;

		/* Text cross-attention layer norm */
		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.text_cross_attn_layer_norm.weight", i);
		dec->layers[i].tca_ln_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].tca_ln_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.text_cross_attn_layer_norm.bias", i);
		dec->layers[i].tca_ln_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].tca_ln_b)
			return SAM3_ENOMEM;

		/* FFN fc1 */
		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.mlp.fc1.weight", i);
		dec->layers[i].ffn_fc1_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			2, fc1_w_dims);
		if (!dec->layers[i].ffn_fc1_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.mlp.fc1.bias", i);
		dec->layers[i].ffn_fc1_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, fc1_b_dims);
		if (!dec->layers[i].ffn_fc1_b)
			return SAM3_ENOMEM;

		/* FFN fc2 */
		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.mlp.fc2.weight", i);
		dec->layers[i].ffn_fc2_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			2, fc2_w_dims);
		if (!dec->layers[i].ffn_fc2_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.mlp.fc2.bias", i);
		dec->layers[i].ffn_fc2_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].ffn_fc2_b)
			return SAM3_ENOMEM;

		/* FFN layer norm */
		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.mlp_layer_norm.weight", i);
		dec->layers[i].ffn_ln_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].ffn_ln_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 DEC_PREFIX "layers.%d.mlp_layer_norm.bias", i);
		dec->layers[i].ffn_ln_b = gh_load_mmap(
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
	dec->output_ln_w = gh_load_mmap(wf,
		DEC_PREFIX "output_layer_norm.weight",
		arena, SAM3_DTYPE_F32, 1, d_dims);
	if (!dec->output_ln_w)
		return SAM3_ENOMEM;

	dec->output_ln_b = gh_load_mmap(wf,
		DEC_PREFIX "output_layer_norm.bias",
		arena, SAM3_DTYPE_F32, 1, d_dims);
	if (!dec->output_ln_b)
		return SAM3_ENOMEM;

	/*
	 * DAB-DETR conditional queries: learned reference boxes and
	 * MLP that converts sine position embeddings to query_pos.
	 *
	 * reference_points.weight: [n_queries, 4] learned box params
	 * ref_point_head: MLP(512, 256, 256, 2) = linear→relu→linear
	 */
	int rp_dims[] = {nq, 4};
	dec->reference_points = gh_load_mmap(wf,
		DEC_PREFIX "reference_points.weight",
		arena, SAM3_DTYPE_F32, 2, rp_dims);
	if (!dec->reference_points)
		return SAM3_ENOMEM;

	/* ref_point_head layer1: Linear(2*d_model, d_model) */
	int rph_fc1_w_dims[] = {d, 2 * d};
	dec->rph_fc1_w = gh_load_mmap(wf,
		DEC_PREFIX "ref_point_head.layer1.weight",
		arena, SAM3_DTYPE_F32, 2, rph_fc1_w_dims);
	if (!dec->rph_fc1_w)
		return SAM3_ENOMEM;

	dec->rph_fc1_b = gh_load_mmap(wf,
		DEC_PREFIX "ref_point_head.layer1.bias",
		arena, SAM3_DTYPE_F32, 1, d_dims);
	if (!dec->rph_fc1_b)
		return SAM3_ENOMEM;

	/* ref_point_head layer2: Linear(d_model, d_model) */
	dec->rph_fc2_w = gh_load_mmap(wf,
		DEC_PREFIX "ref_point_head.layer2.weight",
		arena, SAM3_DTYPE_F32, 2, proj_w_dims);
	if (!dec->rph_fc2_w)
		return SAM3_ENOMEM;

	dec->rph_fc2_b = gh_load_mmap(wf,
		DEC_PREFIX "ref_point_head.layer2.bias",
		arena, SAM3_DTYPE_F32, 1, d_dims);
	if (!dec->rph_fc2_b)
		return SAM3_ENOMEM;

	return SAM3_OK;
}

/*
 * gen_sine_position_embed - Generate sinusoidal position embeddings.
 *
 * Matches Python gen_sineembed_for_position() from model_misc.py.
 * For 4D input (x, y, w, h), output is [nq, 4 * num_feats] = [nq, 2*d_model].
 *
 * @arena:  Arena for output tensor allocation
 * @coords: Reference box coords after sigmoid [nq, 4] (y, x, w, h)
 * @nq:     Number of queries
 * @d_model: Model dimension (256), must be even
 *
 * Returns sine position embedding [nq, 2*d_model], or NULL on error.
 */
static struct sam3_tensor *gen_sine_position_embed(
	struct sam3_arena *arena,
	const float *coords, int nq, int d_model)
{
	int num_feats = d_model / 2; /* 128 */
	int out_dim = d_model * 2;   /* 512 */
	int out_dims[] = {nq, out_dim};

	struct sam3_tensor *out = gh_alloc_tensor(arena, SAM3_DTYPE_F32,
						  2, out_dims);
	if (!out)
		return NULL;

	float *dst = (float *)out->data;
	float scale = 2.0f * (float)M_PI;

	/*
	 * Precompute dim_t[j] = 10000^(2*floor(j/2) / num_feats).
	 * Python: dim_t = 10000 ** (2 * (dim_t // 2) / num_feats)
	 */
	float dim_t[256]; /* num_feats <= 256 */
	for (int j = 0; j < num_feats; j++) {
		float exp = 2.0f * (float)(j / 2) / (float)num_feats;
		dim_t[j] = powf(10000.0f, exp);
	}

	/*
	 * For each query, compute sine embeddings for 4 coordinates.
	 * Python order: pos_y, pos_x, pos_w, pos_h (note y before x).
	 * coords layout: [x, y, w, h] per query (standard box format).
	 *
	 * Python gen_sineembed uses: x_embed=pos[:,0], y_embed=pos[:,1]
	 * and output = cat(pos_y, pos_x, pos_w, pos_h)
	 * So coords[q*4 + 0] = x, coords[q*4 + 1] = y
	 * Output order: y, x, w, h → indices 1, 0, 2, 3
	 */
	int coord_order[] = {1, 0, 2, 3}; /* y, x, w, h */

	for (int q = 0; q < nq; q++) {
		float *row = dst + q * out_dim;
		for (int c = 0; c < 4; c++) {
			int ci = coord_order[c];
			float val = coords[q * 4 + ci] * scale;
			float *block = row + c * num_feats;

			/*
			 * Python interleaves sin/cos:
			 * pos_x = stack(sin(even), cos(odd)).flatten()
			 * even indices: sin, odd indices: cos
			 */
			for (int j = 0; j < num_feats; j += 2) {
				float v = val / dim_t[j];
				block[j] = sinf(v);
				block[j + 1] = cosf(v);
			}
		}
	}

	return out;
}

/*
 * sam3_decoder_compute_query_pos - Compute DAB-DETR query_pos.
 *
 * Computes query_pos from reference_points by:
 * 1. sigmoid(reference_points) → reference_boxes
 * 2. gen_sine_position_embed(reference_boxes) → sine_embed [nq, 2*d]
 * 3. ref_point_head(sine_embed) → query_pos [nq, d]
 *
 * The ref_point_head MLP is built as graph ops and must be evaluated.
 */
struct sam3_tensor *sam3_decoder_compute_query_pos(
	struct sam3_decoder *dec,
	struct sam3_graph *g,
	struct sam3_arena *arena,
	const float *ref_boxes)
{
	int nq = dec->n_queries;
	int d = dec->d_model;

	/* CPU: sine position embedding from reference boxes */
	struct sam3_tensor *sine_embed;
	sine_embed = gen_sine_position_embed(arena, ref_boxes, nq, d);
	if (!sine_embed) {
		sam3_log_error("decoder: sine embed alloc failed");
		return NULL;
	}

	/* Graph: ref_point_head MLP(512→256→256) with ReLU */
	struct sam3_tensor *qpos;
	qpos = gh_mlp(g, arena, sine_embed,
		       dec->rph_fc1_w, dec->rph_fc1_b,
		       dec->rph_fc2_w, dec->rph_fc2_b,
		       SAM3_OP_RELU);
	if (!qpos)
		sam3_log_error("decoder: ref_point_head mlp failed");

	return qpos;
}

/*
 * decoder_self_attention_with_pos - Self-attention with position embedding.
 *
 * Python: q = k = tgt + query_pos (for Q and K projections)
 *         v = tgt (raw, no position)
 *         out = self_attn(q, k, v)
 *
 * Since Q/K use tgt+pos but V uses raw tgt, we cannot use the packed
 * QKV approach. Instead, slice the QKV weights and project separately.
 */
static struct sam3_tensor *decoder_self_attention_with_pos(
	struct sam3_graph *g, struct sam3_arena *arena,
	struct sam3_tensor *tgt,
	struct sam3_tensor *query_pos,
	struct sam3_tensor *qkv_w, struct sam3_tensor *qkv_b,
	struct sam3_tensor *out_w, struct sam3_tensor *out_b,
	int d_model, int n_heads)
{
	int head_dim = d_model / n_heads;

	/* tgt + query_pos for Q and K projections */
	struct sam3_tensor *tgt_pos = gh_add(g, arena, tgt, query_pos);
	if (!tgt_pos)
		return NULL;

	/* Slice QKV weights: [3*d, d] → Q_w [d,d], K_w [d,d], V_w [d,d] */
	struct sam3_tensor *q_w, *k_w, *v_w;
	q_w = gh_slice(g, arena, qkv_w, 0, 0, d_model);
	k_w = gh_slice(g, arena, qkv_w, 0, d_model, 2 * d_model);
	v_w = gh_slice(g, arena, qkv_w, 0, 2 * d_model, 3 * d_model);
	if (!q_w || !k_w || !v_w)
		return NULL;

	/* Slice QKV bias: [3*d] → Q_b [d], K_b [d], V_b [d] */
	struct sam3_tensor *q_b, *k_b, *v_b;
	q_b = gh_slice(g, arena, qkv_b, 0, 0, d_model);
	k_b = gh_slice(g, arena, qkv_b, 0, d_model, 2 * d_model);
	v_b = gh_slice(g, arena, qkv_b, 0, 2 * d_model, 3 * d_model);
	if (!q_b || !k_b || !v_b)
		return NULL;

	/* Project: Q and K from tgt+pos, V from raw tgt */
	struct sam3_tensor *sq = gh_linear(g, arena, tgt_pos, q_w, q_b);
	struct sam3_tensor *sk = gh_linear(g, arena, tgt_pos, k_w, k_b);
	struct sam3_tensor *sv = gh_linear(g, arena, tgt, v_w, v_b);
	if (!sq || !sk || !sv)
		return NULL;

	/* Per-head SDPA */
	struct sam3_tensor *head_outs[64];
	for (int h = 0; h < n_heads; h++) {
		int hs = h * head_dim;
		int he = hs + head_dim;

		struct sam3_tensor *hq, *hk, *hv, *ho;
		hq = gh_slice(g, arena, sq, 1, hs, he);
		hk = gh_slice(g, arena, sk, 1, hs, he);
		hv = gh_slice(g, arena, sv, 1, hs, he);
		if (!hq || !hk || !hv)
			return NULL;

		ho = gh_sdpa(g, arena, hq, hk, hv, NULL, head_dim);
		if (!ho)
			return NULL;
		head_outs[h] = ho;
	}

	/* Concatenate heads */
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

/*
 * decoder_cross_attention_with_pos - Cross-attention adding pos to key only.
 *
 * Python: Q = q_src, K = kv_src + pos, V = kv_src (raw)
 * Since the packed KV weight projects both K and V from the same source,
 * we slice the KV weights apart and project K from (kv_src + pos) and
 * V from raw kv_src, matching PyTorch's MultiheadAttention(query, key, value).
 */
static struct sam3_tensor *decoder_cross_attention_with_pos(
	struct sam3_graph *g, struct sam3_arena *arena,
	struct sam3_tensor *q_src,
	struct sam3_tensor *kv_src,
	struct sam3_tensor *kv_pos,
	struct sam3_tensor *q_w, struct sam3_tensor *q_b,
	struct sam3_tensor *kv_w, struct sam3_tensor *kv_b,
	struct sam3_tensor *out_w, struct sam3_tensor *out_b,
	int d_model, int n_heads)
{
	int head_dim = d_model / n_heads;

	/* Project Q normally */
	struct sam3_tensor *sq = gh_linear(g, arena, q_src, q_w, q_b);
	if (!sq)
		return NULL;

	/* kv_src + pos for K projection */
	struct sam3_tensor *kv_with_pos = gh_add(g, arena, kv_src, kv_pos);
	if (!kv_with_pos)
		return NULL;

	/* Slice packed KV weights: [2*d, d] → K_w [d,d], V_w [d,d] */
	struct sam3_tensor *k_w, *v_w;
	k_w = gh_slice(g, arena, kv_w, 0, 0, d_model);
	v_w = gh_slice(g, arena, kv_w, 0, d_model, 2 * d_model);
	if (!k_w || !v_w)
		return NULL;

	/* Slice packed KV bias: [2*d] → K_b [d], V_b [d] */
	struct sam3_tensor *k_b, *v_b;
	k_b = gh_slice(g, arena, kv_b, 0, 0, d_model);
	v_b = gh_slice(g, arena, kv_b, 0, d_model, 2 * d_model);
	if (!k_b || !v_b)
		return NULL;

	/* K from (kv_src + pos), V from raw kv_src */
	struct sam3_tensor *sk = gh_linear(g, arena, kv_with_pos, k_w, k_b);
	struct sam3_tensor *sv = gh_linear(g, arena, kv_src, v_w, v_b);
	if (!sk || !sv)
		return NULL;

	/* Per-head SDPA */
	struct sam3_tensor *head_outs[64];
	for (int h = 0; h < n_heads; h++) {
		int hs = h * head_dim;
		int he = hs + head_dim;

		struct sam3_tensor *hq, *hk, *hv, *ho;
		hq = gh_slice(g, arena, sq, 1, hs, he);
		hk = gh_slice(g, arena, sk, 1, hs, he);
		hv = gh_slice(g, arena, sv, 1, hs, he);
		if (!hq || !hk || !hv)
			return NULL;

		ho = gh_sdpa(g, arena, hq, hk, hv, NULL, head_dim);
		if (!ho)
			return NULL;
		head_outs[h] = ho;
	}

	/* Concatenate heads */
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

struct sam3_tensor *sam3_decoder_build_layer(
	struct sam3_decoder *dec,
	int layer_idx,
	struct sam3_graph *g,
	struct sam3_tensor *q,
	struct sam3_tensor *query_pos,
	struct sam3_tensor *enc_features,
	struct sam3_tensor *enc_pos,
	struct sam3_tensor *text_features,
	struct sam3_tensor **boxes,
	struct sam3_arena *arena)
{
	int d = dec->d_model;
	int i = layer_idx;

	/*
	 * Step (a): Self-attention with position embedding.
	 * Python: q = k = tgt + query_pos, v = tgt (raw)
	 * Post-norm: attention, residual add, then layernorm.
	 */
	struct sam3_tensor *sa_out;
	sa_out = decoder_self_attention_with_pos(
		g, arena, q, query_pos,
		dec->layers[i].sa_qkv_w,
		dec->layers[i].sa_qkv_b,
		dec->layers[i].sa_out_w,
		dec->layers[i].sa_out_b,
		d, dec->n_heads);
	if (!sa_out)
		return NULL;

	q = gh_add(g, arena, q, sa_out);
	if (!q)
		return NULL;
	q = gh_layernorm(g, arena, q,
			  dec->layers[i].sa_ln_w,
			  dec->layers[i].sa_ln_b);
	if (!q)
		return NULL;

	/*
	 * Step (b): Text cross-attention (Python order: SA->TCA->CA).
	 * Python: Q = tgt + query_pos, K = V = memory_text
	 * Post-norm: cross-attend to text, residual add, layernorm.
	 * Skip if text_features is NULL (point-only prompts).
	 */
	struct sam3_tensor *q_with_pos;

	if (text_features) {
		q_with_pos = gh_add(g, arena, q, query_pos);
		if (!q_with_pos)
			return NULL;

		struct sam3_tensor *tca_out;
		tca_out = gh_cross_attention(
			g, arena,
			q_with_pos, text_features,
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
		q = gh_layernorm(g, arena, q,
				  dec->layers[i].tca_ln_w,
				  dec->layers[i].tca_ln_b);
		if (!q)
			return NULL;
	}

	/*
	 * Step (c): Cross-attention to encoder (image) features.
	 * Python: Q = tgt + query_pos, K = memory + memory_pos, V = memory
	 * Position is added to key but NOT value. Post-norm.
	 */
	q_with_pos = gh_add(g, arena, q, query_pos);
	if (!q_with_pos)
		return NULL;

	struct sam3_tensor *ca_out;
	if (enc_pos) {
		ca_out = decoder_cross_attention_with_pos(
			g, arena,
			q_with_pos, enc_features, enc_pos,
			dec->layers[i].ca_q_w,
			dec->layers[i].ca_q_b,
			dec->layers[i].ca_kv_w,
			dec->layers[i].ca_kv_b,
			dec->layers[i].ca_out_w,
			dec->layers[i].ca_out_b,
			d, dec->n_heads);
	} else {
		ca_out = gh_cross_attention(
			g, arena,
			q_with_pos, enc_features,
			dec->layers[i].ca_q_w,
			dec->layers[i].ca_q_b,
			dec->layers[i].ca_kv_w,
			dec->layers[i].ca_kv_b,
			dec->layers[i].ca_out_w,
			dec->layers[i].ca_out_b,
			dec->n_heads);
	}
	if (!ca_out)
		return NULL;

	q = gh_add(g, arena, q, ca_out);
	if (!q)
		return NULL;
	q = gh_layernorm(g, arena, q,
			  dec->layers[i].ca_ln_w,
			  dec->layers[i].ca_ln_b);
	if (!q)
		return NULL;

	/*
	 * Step (d): FFN with residual.
	 * Post-norm: MLP, residual add, layernorm.
	 */
	struct sam3_tensor *ff;
	ff = gh_mlp(g, arena, q,
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
	q = gh_layernorm(g, arena, q,
			  dec->layers[i].ffn_ln_w,
			  dec->layers[i].ffn_ln_b);
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

/*
 * Substep builders: each builds one substep of a decoder layer.
 * These exist for debugging — they allow the caller to evaluate
 * the graph between substeps and inspect intermediates.
 */

struct sam3_tensor *sam3_decoder_build_sa(
	struct sam3_decoder *dec, int layer_idx,
	struct sam3_graph *g, struct sam3_tensor *q,
	struct sam3_tensor *query_pos, struct sam3_arena *arena)
{
	int d = dec->d_model;
	int i = layer_idx;

	struct sam3_tensor *sa_out;
	sa_out = decoder_self_attention_with_pos(
		g, arena, q, query_pos,
		dec->layers[i].sa_qkv_w,
		dec->layers[i].sa_qkv_b,
		dec->layers[i].sa_out_w,
		dec->layers[i].sa_out_b,
		d, dec->n_heads);
	if (!sa_out)
		return NULL;

	q = gh_add(g, arena, q, sa_out);
	if (!q)
		return NULL;
	q = gh_layernorm(g, arena, q,
			  dec->layers[i].sa_ln_w,
			  dec->layers[i].sa_ln_b);
	return q;
}

struct sam3_tensor *sam3_decoder_build_tca(
	struct sam3_decoder *dec, int layer_idx,
	struct sam3_graph *g, struct sam3_tensor *q,
	struct sam3_tensor *query_pos,
	struct sam3_tensor *text_features, struct sam3_arena *arena)
{
	int i = layer_idx;

	if (!text_features)
		return q;

	struct sam3_tensor *q_with_pos;
	q_with_pos = gh_add(g, arena, q, query_pos);
	if (!q_with_pos)
		return NULL;

	struct sam3_tensor *tca_out;
	tca_out = gh_cross_attention(
		g, arena,
		q_with_pos, text_features,
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
	q = gh_layernorm(g, arena, q,
			  dec->layers[i].tca_ln_w,
			  dec->layers[i].tca_ln_b);
	return q;
}

struct sam3_tensor *sam3_decoder_build_ca(
	struct sam3_decoder *dec, int layer_idx,
	struct sam3_graph *g, struct sam3_tensor *q,
	struct sam3_tensor *query_pos,
	struct sam3_tensor *enc_features, struct sam3_tensor *enc_pos,
	struct sam3_arena *arena)
{
	int d = dec->d_model;
	int i = layer_idx;

	struct sam3_tensor *q_with_pos;
	q_with_pos = gh_add(g, arena, q, query_pos);
	if (!q_with_pos)
		return NULL;

	struct sam3_tensor *ca_out;
	if (enc_pos) {
		ca_out = decoder_cross_attention_with_pos(
			g, arena,
			q_with_pos, enc_features, enc_pos,
			dec->layers[i].ca_q_w,
			dec->layers[i].ca_q_b,
			dec->layers[i].ca_kv_w,
			dec->layers[i].ca_kv_b,
			dec->layers[i].ca_out_w,
			dec->layers[i].ca_out_b,
			d, dec->n_heads);
	} else {
		ca_out = gh_cross_attention(
			g, arena,
			q_with_pos, enc_features,
			dec->layers[i].ca_q_w,
			dec->layers[i].ca_q_b,
			dec->layers[i].ca_kv_w,
			dec->layers[i].ca_kv_b,
			dec->layers[i].ca_out_w,
			dec->layers[i].ca_out_b,
			dec->n_heads);
	}
	if (!ca_out)
		return NULL;

	q = gh_add(g, arena, q, ca_out);
	if (!q)
		return NULL;
	q = gh_layernorm(g, arena, q,
			  dec->layers[i].ca_ln_w,
			  dec->layers[i].ca_ln_b);
	return q;
}

struct sam3_tensor *sam3_decoder_build_ffn(
	struct sam3_decoder *dec, int layer_idx,
	struct sam3_graph *g, struct sam3_tensor *q,
	struct sam3_arena *arena)
{
	int i = layer_idx;

	struct sam3_tensor *ff;
	ff = gh_mlp(g, arena, q,
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
	q = gh_layernorm(g, arena, q,
			  dec->layers[i].ffn_ln_w,
			  dec->layers[i].ffn_ln_b);
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

	/* Compute initial reference boxes and query_pos */
	int rp_nelems = nq * 4;
	float *ref_boxes = (float *)sam3_arena_alloc(arena,
		(size_t)rp_nelems * sizeof(float));
	if (!ref_boxes)
		return NULL;
	const float *rp = (const float *)dec->reference_points->data;
	for (int j = 0; j < rp_nelems; j++)
		ref_boxes[j] = 1.0f / (1.0f + expf(-rp[j])); /* sigmoid */

	struct sam3_tensor *query_pos;
	query_pos = sam3_decoder_compute_query_pos(dec, g, arena,
						    ref_boxes);
	if (!query_pos)
		return NULL;

	int box_dims[] = {nq, 4};
	struct sam3_tensor *boxes;
	boxes = gh_alloc_tensor(arena, SAM3_DTYPE_F32, 2, box_dims);
	if (!boxes)
		return NULL;
	memset(boxes->data, 0, boxes->nbytes);

	for (int i = 0; i < dec->n_layers; i++) {
		q = sam3_decoder_build_layer(dec, i, g, q,
					      query_pos,
					      enc_features,
					      NULL, /* enc_pos */
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
