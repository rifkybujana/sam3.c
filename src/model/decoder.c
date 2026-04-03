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
 * Key types:  sam3_decoder
 * Depends on: decoder.h, graph_helpers.h
 * Used by:    sam3.c (top-level API)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>

#include "decoder.h"
#include "graph_helpers.h"

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

enum sam3_error sam3_decoder_load(struct sam3_decoder *dec,
				  const struct sam3_weight_file *wf,
				  struct sam3_arena *arena)
{
	int d = dec->d_model;
	int d3 = d * 3;
	int d2 = d * 2;
	int ff = dec->d_ffn;
	int nq = dec->n_queries;
	char name[128];

	/* Learned query embeddings: [n_queries, d_model] */
	int qe_dims[] = {nq, d};
	dec->query_embed = gh_load_or_alloc(wf, "decoder.query_embed",
					  arena, SAM3_DTYPE_F32,
					  2, qe_dims);
	if (!dec->query_embed)
		return SAM3_ENOMEM;

	/* Dimension arrays for weight loading */
	int d_dims[] = {d};
	int sa_qkv_w_dims[] = {d3, d};
	int sa_qkv_b_dims[] = {d3};
	int proj_w_dims[] = {d, d};
	int ca_kv_w_dims[] = {d2, d};
	int ca_kv_b_dims[] = {d2};
	int fc1_w_dims[] = {ff, d};
	int fc1_b_dims[] = {ff};
	int fc2_w_dims[] = {d, ff};

	/* Box refinement MLP dimensions */
	int box_fc1_w_dims[] = {d, d};
	int box_fc2_w_dims[] = {d, d};
	int box_fc3_w_dims[] = {4, d};
	int box_fc3_b_dims[] = {4};

	for (int i = 0; i < dec->n_layers; i++) {
		/* Self-attention QKV */
		snprintf(name, sizeof(name),
			 "decoder.layer.%d.sa.qkv.weight", i);
		dec->layers[i].sa_qkv_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, sa_qkv_w_dims);
		if (!dec->layers[i].sa_qkv_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "decoder.layer.%d.sa.qkv.bias", i);
		dec->layers[i].sa_qkv_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, sa_qkv_b_dims);
		if (!dec->layers[i].sa_qkv_b)
			return SAM3_ENOMEM;

		/* Self-attention output */
		snprintf(name, sizeof(name),
			 "decoder.layer.%d.sa.out.weight", i);
		dec->layers[i].sa_out_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!dec->layers[i].sa_out_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "decoder.layer.%d.sa.out.bias", i);
		dec->layers[i].sa_out_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].sa_out_b)
			return SAM3_ENOMEM;

		/* Self-attention layer norm */
		snprintf(name, sizeof(name),
			 "decoder.layer.%d.sa.ln.weight", i);
		dec->layers[i].sa_ln_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].sa_ln_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "decoder.layer.%d.sa.ln.bias", i);
		dec->layers[i].sa_ln_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].sa_ln_b)
			return SAM3_ENOMEM;

		/* Encoder cross-attention Q */
		snprintf(name, sizeof(name),
			 "decoder.layer.%d.ca.q.weight", i);
		dec->layers[i].ca_q_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!dec->layers[i].ca_q_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "decoder.layer.%d.ca.q.bias", i);
		dec->layers[i].ca_q_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].ca_q_b)
			return SAM3_ENOMEM;

		/* Encoder cross-attention KV */
		snprintf(name, sizeof(name),
			 "decoder.layer.%d.ca.kv.weight", i);
		dec->layers[i].ca_kv_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, ca_kv_w_dims);
		if (!dec->layers[i].ca_kv_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "decoder.layer.%d.ca.kv.bias", i);
		dec->layers[i].ca_kv_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, ca_kv_b_dims);
		if (!dec->layers[i].ca_kv_b)
			return SAM3_ENOMEM;

		/* Encoder cross-attention output */
		snprintf(name, sizeof(name),
			 "decoder.layer.%d.ca.out.weight", i);
		dec->layers[i].ca_out_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!dec->layers[i].ca_out_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "decoder.layer.%d.ca.out.bias", i);
		dec->layers[i].ca_out_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].ca_out_b)
			return SAM3_ENOMEM;

		/* Encoder cross-attention layer norm */
		snprintf(name, sizeof(name),
			 "decoder.layer.%d.ca.ln.weight", i);
		dec->layers[i].ca_ln_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].ca_ln_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "decoder.layer.%d.ca.ln.bias", i);
		dec->layers[i].ca_ln_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].ca_ln_b)
			return SAM3_ENOMEM;

		/* Text cross-attention Q */
		snprintf(name, sizeof(name),
			 "decoder.layer.%d.tca.q.weight", i);
		dec->layers[i].tca_q_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!dec->layers[i].tca_q_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "decoder.layer.%d.tca.q.bias", i);
		dec->layers[i].tca_q_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].tca_q_b)
			return SAM3_ENOMEM;

		/* Text cross-attention KV */
		snprintf(name, sizeof(name),
			 "decoder.layer.%d.tca.kv.weight", i);
		dec->layers[i].tca_kv_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, ca_kv_w_dims);
		if (!dec->layers[i].tca_kv_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "decoder.layer.%d.tca.kv.bias", i);
		dec->layers[i].tca_kv_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, ca_kv_b_dims);
		if (!dec->layers[i].tca_kv_b)
			return SAM3_ENOMEM;

		/* Text cross-attention output */
		snprintf(name, sizeof(name),
			 "decoder.layer.%d.tca.out.weight", i);
		dec->layers[i].tca_out_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, proj_w_dims);
		if (!dec->layers[i].tca_out_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "decoder.layer.%d.tca.out.bias", i);
		dec->layers[i].tca_out_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].tca_out_b)
			return SAM3_ENOMEM;

		/* Text cross-attention layer norm */
		snprintf(name, sizeof(name),
			 "decoder.layer.%d.tca.ln.weight", i);
		dec->layers[i].tca_ln_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].tca_ln_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "decoder.layer.%d.tca.ln.bias", i);
		dec->layers[i].tca_ln_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].tca_ln_b)
			return SAM3_ENOMEM;

		/* FFN fc1 */
		snprintf(name, sizeof(name),
			 "decoder.layer.%d.ffn.fc1.weight", i);
		dec->layers[i].ffn_fc1_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, fc1_w_dims);
		if (!dec->layers[i].ffn_fc1_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "decoder.layer.%d.ffn.fc1.bias", i);
		dec->layers[i].ffn_fc1_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, fc1_b_dims);
		if (!dec->layers[i].ffn_fc1_b)
			return SAM3_ENOMEM;

		/* FFN fc2 */
		snprintf(name, sizeof(name),
			 "decoder.layer.%d.ffn.fc2.weight", i);
		dec->layers[i].ffn_fc2_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, fc2_w_dims);
		if (!dec->layers[i].ffn_fc2_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "decoder.layer.%d.ffn.fc2.bias", i);
		dec->layers[i].ffn_fc2_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].ffn_fc2_b)
			return SAM3_ENOMEM;

		/* FFN layer norm */
		snprintf(name, sizeof(name),
			 "decoder.layer.%d.ffn.ln.weight", i);
		dec->layers[i].ffn_ln_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].ffn_ln_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "decoder.layer.%d.ffn.ln.bias", i);
		dec->layers[i].ffn_ln_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].ffn_ln_b)
			return SAM3_ENOMEM;

		/* Box refinement fc1: [d_model, d_model] */
		snprintf(name, sizeof(name),
			 "decoder.layer.%d.box.fc1.weight", i);
		dec->layers[i].box_fc1_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, box_fc1_w_dims);
		if (!dec->layers[i].box_fc1_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "decoder.layer.%d.box.fc1.bias", i);
		dec->layers[i].box_fc1_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].box_fc1_b)
			return SAM3_ENOMEM;

		/* Box refinement fc2: [d_model, d_model] */
		snprintf(name, sizeof(name),
			 "decoder.layer.%d.box.fc2.weight", i);
		dec->layers[i].box_fc2_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, box_fc2_w_dims);
		if (!dec->layers[i].box_fc2_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "decoder.layer.%d.box.fc2.bias", i);
		dec->layers[i].box_fc2_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, d_dims);
		if (!dec->layers[i].box_fc2_b)
			return SAM3_ENOMEM;

		/* Box refinement fc3: [4, d_model] / [4] */
		snprintf(name, sizeof(name),
			 "decoder.layer.%d.box.fc3.weight", i);
		dec->layers[i].box_fc3_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			2, box_fc3_w_dims);
		if (!dec->layers[i].box_fc3_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "decoder.layer.%d.box.fc3.bias", i);
		dec->layers[i].box_fc3_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32,
			1, box_fc3_b_dims);
		if (!dec->layers[i].box_fc3_b)
			return SAM3_ENOMEM;
	}

	return SAM3_OK;
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
	int d = dec->d_model;

	/*
	 * Start with learned query embeddings: [n_queries, d_model].
	 * These are the initial object queries.
	 */
	struct sam3_tensor *q = dec->query_embed;

	/*
	 * Initialize box accumulator as zeros [n_queries, 4].
	 * Each layer adds a delta and applies sigmoid.
	 */
	int box_dims[] = {nq, 4};
	struct sam3_tensor *boxes;
	boxes = gh_alloc_tensor(arena, SAM3_DTYPE_F32, 2, box_dims);
	if (!boxes)
		return NULL;
	memset(boxes->data, 0, boxes->nbytes);

	for (int i = 0; i < dec->n_layers; i++) {
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
		 * Step (e): Box refinement.
		 * delta = box_mlp(q) -> [n_queries, 4]
		 * boxes = sigmoid(boxes + delta)
		 */
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

		boxes = gh_add(g, arena, boxes, delta);
		if (!boxes)
			return NULL;

		boxes = gh_sigmoid(g, arena, boxes);
		if (!boxes)
			return NULL;
	}

	if (box_out)
		*box_out = boxes;

	return q; /* [n_queries, d_model] */
}
