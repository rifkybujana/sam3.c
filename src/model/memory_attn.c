/*
 * src/model/memory_attn.c - 4-layer RoPE cross-attention graph construction
 *
 * Builds the compute graph for the memory attention transformer used in
 * SAM3 video tracking. The module implements TransformerEncoderCrossAttention
 * with 4 layers of pre-norm self-attention (RoPE), pre-norm cross-attention
 * (queries from current features, keys/values from memory), and pre-norm
 * FFN (ReLU). A final LayerNorm follows the last layer.
 *
 * Weight prefix: tracker_model.transformer.encoder.*
 *
 * Key types:  sam3_memory_attn, sam3_memattn_layer
 * Depends on: memory_attn.h, graph_helpers.h, util/log.h
 * Used by:    model/tracker.c (future)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "memory_attn.h"
#include "graph_helpers.h"
#include "util/log.h"

#define WP "tracker_model.transformer.encoder."

enum sam3_error sam3_memory_attn_init(struct sam3_memory_attn *attn,
				      int d_model, int mem_dim,
				      int n_layers, int n_heads,
				      int feat_h, int feat_w)
{
	if (!attn || d_model <= 0 || mem_dim <= 0 ||
	    n_layers <= 0 || n_layers > SAM3_MEMATTN_N_LAYERS ||
	    n_heads <= 0 ||
	    feat_h <= 0 || feat_w <= 0) {
		sam3_log_error("mem_attn init: invalid argument");
		return SAM3_EINVAL;
	}

	memset(attn, 0, sizeof(*attn));
	attn->d_model = d_model;
	attn->mem_dim = mem_dim;
	attn->n_layers = n_layers;
	attn->n_heads = n_heads;
	attn->feat_h = feat_h;
	attn->feat_w = feat_w;

	return SAM3_OK;
}

/*
 * load_layer_weights - Load weights for a single transformer layer.
 *
 * Weight names follow the Python naming convention:
 *   layers.{i}.self_attn.{q,k,v,out}_proj.{weight,bias}
 *   layers.{i}.cross_attn_image.{q,k,v,out}_proj.{weight,bias}
 *   layers.{i}.linear1.{weight,bias}
 *   layers.{i}.linear2.{weight,bias}
 *   layers.{i}.norm{1,2,3}.{weight,bias}
 */
static enum sam3_error load_layer_weights(struct sam3_memattn_layer *layer,
					  int idx,
					  int d_model, int mem_dim,
					  int ffn_dim,
					  const struct sam3_weight_file *wf,
					  struct sam3_arena *arena)
{
	char name[160];
	int d_dims[] = {d_model};
	int dd_dims[] = {d_model, d_model};
	int ffn_d_dims[] = {ffn_dim};

	/* Self-attention Q/K/V/Out: all [d_model, d_model] */
	snprintf(name, sizeof(name),
		 WP "layers.%d.self_attn.q_proj.weight", idx);
	layer->sa_q_w = gh_load_mmap(wf, name, arena,
		SAM3_DTYPE_F32, 2, dd_dims);
	if (!layer->sa_q_w) return SAM3_ENOMEM;

	snprintf(name, sizeof(name),
		 WP "layers.%d.self_attn.q_proj.bias", idx);
	layer->sa_q_b = gh_load_mmap(wf, name, arena,
		SAM3_DTYPE_F32, 1, d_dims);
	if (!layer->sa_q_b) return SAM3_ENOMEM;

	snprintf(name, sizeof(name),
		 WP "layers.%d.self_attn.k_proj.weight", idx);
	layer->sa_k_w = gh_load_mmap(wf, name, arena,
		SAM3_DTYPE_F32, 2, dd_dims);
	if (!layer->sa_k_w) return SAM3_ENOMEM;

	snprintf(name, sizeof(name),
		 WP "layers.%d.self_attn.k_proj.bias", idx);
	layer->sa_k_b = gh_load_mmap(wf, name, arena,
		SAM3_DTYPE_F32, 1, d_dims);
	if (!layer->sa_k_b) return SAM3_ENOMEM;

	snprintf(name, sizeof(name),
		 WP "layers.%d.self_attn.v_proj.weight", idx);
	layer->sa_v_w = gh_load_mmap(wf, name, arena,
		SAM3_DTYPE_F32, 2, dd_dims);
	if (!layer->sa_v_w) return SAM3_ENOMEM;

	snprintf(name, sizeof(name),
		 WP "layers.%d.self_attn.v_proj.bias", idx);
	layer->sa_v_b = gh_load_mmap(wf, name, arena,
		SAM3_DTYPE_F32, 1, d_dims);
	if (!layer->sa_v_b) return SAM3_ENOMEM;

	snprintf(name, sizeof(name),
		 WP "layers.%d.self_attn.out_proj.weight", idx);
	layer->sa_out_w = gh_load_mmap(wf, name, arena,
		SAM3_DTYPE_F32, 2, dd_dims);
	if (!layer->sa_out_w) return SAM3_ENOMEM;

	snprintf(name, sizeof(name),
		 WP "layers.%d.self_attn.out_proj.bias", idx);
	layer->sa_out_b = gh_load_mmap(wf, name, arena,
		SAM3_DTYPE_F32, 1, d_dims);
	if (!layer->sa_out_b) return SAM3_ENOMEM;

	/* Cross-attention: Q is [d_model, d_model], K/V are [d_model, mem_dim] */
	snprintf(name, sizeof(name),
		 WP "layers.%d.cross_attn_image.q_proj.weight", idx);
	layer->ca_q_w = gh_load_mmap(wf, name, arena,
		SAM3_DTYPE_F32, 2, dd_dims);
	if (!layer->ca_q_w) return SAM3_ENOMEM;

	snprintf(name, sizeof(name),
		 WP "layers.%d.cross_attn_image.q_proj.bias", idx);
	layer->ca_q_b = gh_load_mmap(wf, name, arena,
		SAM3_DTYPE_F32, 1, d_dims);
	if (!layer->ca_q_b) return SAM3_ENOMEM;

	{
		int kv_dims[] = {d_model, mem_dim};

		snprintf(name, sizeof(name),
			 WP "layers.%d.cross_attn_image.k_proj.weight",
			 idx);
		layer->ca_k_w = gh_load_mmap(wf, name, arena,
			SAM3_DTYPE_F32, 2, kv_dims);
		if (!layer->ca_k_w) return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 WP "layers.%d.cross_attn_image.k_proj.bias",
			 idx);
		layer->ca_k_b = gh_load_mmap(wf, name, arena,
			SAM3_DTYPE_F32, 1, d_dims);
		if (!layer->ca_k_b) return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 WP "layers.%d.cross_attn_image.v_proj.weight",
			 idx);
		layer->ca_v_w = gh_load_mmap(wf, name, arena,
			SAM3_DTYPE_F32, 2, kv_dims);
		if (!layer->ca_v_w) return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 WP "layers.%d.cross_attn_image.v_proj.bias",
			 idx);
		layer->ca_v_b = gh_load_mmap(wf, name, arena,
			SAM3_DTYPE_F32, 1, d_dims);
		if (!layer->ca_v_b) return SAM3_ENOMEM;
	}

	snprintf(name, sizeof(name),
		 WP "layers.%d.cross_attn_image.out_proj.weight", idx);
	layer->ca_out_w = gh_load_mmap(wf, name, arena,
		SAM3_DTYPE_F32, 2, dd_dims);
	if (!layer->ca_out_w) return SAM3_ENOMEM;

	snprintf(name, sizeof(name),
		 WP "layers.%d.cross_attn_image.out_proj.bias", idx);
	layer->ca_out_b = gh_load_mmap(wf, name, arena,
		SAM3_DTYPE_F32, 1, d_dims);
	if (!layer->ca_out_b) return SAM3_ENOMEM;

	/* LayerNorm: norm1, norm2, norm3 */
	snprintf(name, sizeof(name),
		 WP "layers.%d.norm1.weight", idx);
	layer->norm1_w = gh_load_mmap(wf, name, arena,
		SAM3_DTYPE_F32, 1, d_dims);
	if (!layer->norm1_w) return SAM3_ENOMEM;

	snprintf(name, sizeof(name),
		 WP "layers.%d.norm1.bias", idx);
	layer->norm1_b = gh_load_mmap(wf, name, arena,
		SAM3_DTYPE_F32, 1, d_dims);
	if (!layer->norm1_b) return SAM3_ENOMEM;

	snprintf(name, sizeof(name),
		 WP "layers.%d.norm2.weight", idx);
	layer->norm2_w = gh_load_mmap(wf, name, arena,
		SAM3_DTYPE_F32, 1, d_dims);
	if (!layer->norm2_w) return SAM3_ENOMEM;

	snprintf(name, sizeof(name),
		 WP "layers.%d.norm2.bias", idx);
	layer->norm2_b = gh_load_mmap(wf, name, arena,
		SAM3_DTYPE_F32, 1, d_dims);
	if (!layer->norm2_b) return SAM3_ENOMEM;

	snprintf(name, sizeof(name),
		 WP "layers.%d.norm3.weight", idx);
	layer->norm3_w = gh_load_mmap(wf, name, arena,
		SAM3_DTYPE_F32, 1, d_dims);
	if (!layer->norm3_w) return SAM3_ENOMEM;

	snprintf(name, sizeof(name),
		 WP "layers.%d.norm3.bias", idx);
	layer->norm3_b = gh_load_mmap(wf, name, arena,
		SAM3_DTYPE_F32, 1, d_dims);
	if (!layer->norm3_b) return SAM3_ENOMEM;

	/* FFN: linear1 [ffn_dim, d_model], linear2 [d_model, ffn_dim] */
	{
		int fc1_dims[] = {ffn_dim, d_model};
		int fc2_dims[] = {d_model, ffn_dim};

		snprintf(name, sizeof(name),
			 WP "layers.%d.linear1.weight", idx);
		layer->ffn_fc1_w = gh_load_mmap(wf, name, arena,
			SAM3_DTYPE_F32, 2, fc1_dims);
		if (!layer->ffn_fc1_w) return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 WP "layers.%d.linear1.bias", idx);
		layer->ffn_fc1_b = gh_load_mmap(wf, name, arena,
			SAM3_DTYPE_F32, 1, ffn_d_dims);
		if (!layer->ffn_fc1_b) return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 WP "layers.%d.linear2.weight", idx);
		layer->ffn_fc2_w = gh_load_mmap(wf, name, arena,
			SAM3_DTYPE_F32, 2, fc2_dims);
		if (!layer->ffn_fc2_w) return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 WP "layers.%d.linear2.bias", idx);
		layer->ffn_fc2_b = gh_load_mmap(wf, name, arena,
			SAM3_DTYPE_F32, 1, d_dims);
		if (!layer->ffn_fc2_b) return SAM3_ENOMEM;
	}

	return SAM3_OK;
}

/*
 * precompute_rope_2d - Fill 2D axial RoPE cos/sin tables.
 *
 * For a grid of (feat_h x feat_w) positions with head_dim/2 frequency
 * bins, the first half uses the y-coordinate and the second half uses
 * the x-coordinate. theta=10000.
 *
 * Output shape: [feat_h * feat_w, head_dim / 2]
 */
static void precompute_rope_2d(float *cos_data, float *sin_data,
				int feat_h, int feat_w,
				int head_dim, float theta)
{
	int half_dim = head_dim / 4; /* freqs per axis */
	int rope_cols = head_dim / 2;

	for (int y = 0; y < feat_h; y++) {
		for (int x = 0; x < feat_w; x++) {
			int pos_idx = y * feat_w + x;
			float *cos_row = cos_data + pos_idx * rope_cols;
			float *sin_row = sin_data + pos_idx * rope_cols;

			/* First half: y-axis frequencies */
			for (int j = 0; j < half_dim; j++) {
				float freq = 1.0f / powf(theta,
					(float)(2 * j) /
					(float)(half_dim * 2));
				cos_row[j] = cosf((float)y * freq);
				sin_row[j] = sinf((float)y * freq);
			}
			/* Second half: x-axis frequencies */
			for (int j = 0; j < half_dim; j++) {
				float freq = 1.0f / powf(theta,
					(float)(2 * j) /
					(float)(half_dim * 2));
				cos_row[half_dim + j] =
					cosf((float)x * freq);
				sin_row[half_dim + j] =
					sinf((float)x * freq);
			}
		}
	}
}

enum sam3_error sam3_memory_attn_load(struct sam3_memory_attn *attn,
				      const struct sam3_weight_file *wf,
				      struct sam3_arena *arena)
{
	if (!attn || !arena) {
		sam3_log_error("mem_attn load: NULL argument");
		return SAM3_EINVAL;
	}

	int d = attn->d_model;
	int md = attn->mem_dim;
	int ffn = SAM3_MEMATTN_FFN_DIM;
	enum sam3_error err;

	/* Load per-layer weights */
	for (int i = 0; i < attn->n_layers; i++) {
		err = load_layer_weights(&attn->layers[i], i, d, md, ffn,
					 wf, arena);
		if (err != SAM3_OK) {
			sam3_log_error("mem_attn: layer %d load failed", i);
			return err;
		}
	}

	/* Final LayerNorm */
	{
		int d_dims[] = {d};

		attn->final_norm_w = gh_load_mmap(wf,
			WP "norm.weight", arena,
			SAM3_DTYPE_F32, 1, d_dims);
		if (!attn->final_norm_w) return SAM3_ENOMEM;

		attn->final_norm_b = gh_load_mmap(wf,
			WP "norm.bias", arena,
			SAM3_DTYPE_F32, 1, d_dims);
		if (!attn->final_norm_b) return SAM3_ENOMEM;
	}

	/* Precompute 2D axial RoPE cos/sin tables */
	{
		int seq = attn->feat_h * attn->feat_w;
		int head_dim = d / attn->n_heads;
		int rope_cols = head_dim / 2;
		int rope_dims[] = {seq, rope_cols};

		attn->rope_cos = gh_alloc_tensor(arena,
			SAM3_DTYPE_F32, 2, rope_dims);
		if (!attn->rope_cos) {
			sam3_log_error("mem_attn: rope_cos alloc failed");
			return SAM3_ENOMEM;
		}

		attn->rope_sin = gh_alloc_tensor(arena,
			SAM3_DTYPE_F32, 2, rope_dims);
		if (!attn->rope_sin) {
			sam3_log_error("mem_attn: rope_sin alloc failed");
			return SAM3_ENOMEM;
		}

		precompute_rope_2d(
			(float *)attn->rope_cos->data,
			(float *)attn->rope_sin->data,
			attn->feat_h, attn->feat_w,
			head_dim, 10000.0f);
	}

	sam3_log_info("memory attention loaded (%d layers, d=%d, mem=%d, "
		      "grid=%dx%d)",
		      attn->n_layers, d, md,
		      attn->feat_h, attn->feat_w);
	return SAM3_OK;
}

enum sam3_error sam3_memory_attn_build_full(struct sam3_memory_attn *attn,
					    struct sam3_graph *g,
					    struct sam3_tensor *current,
					    struct sam3_tensor *memory,
					    struct sam3_tensor *mem_pos,
					    struct sam3_arena *arena,
					    struct sam3_tensor **output)
{
	if (!attn || !g || !current || !memory || !arena || !output) {
		sam3_log_error("mem_attn build: NULL argument");
		return SAM3_EINVAL;
	}

	int seq = current->dims[0];
	int d = attn->d_model;

	sam3_log_debug("mem_attn: current [%d,%d] memory [%d,%d]",
		       current->dims[0], current->dims[1],
		       memory->dims[0], memory->dims[1]);

	/*
	 * Position encoding applies only to cross-attention keys,
	 * not values.  Python: pos_enc_at_cross_attn_keys=True means
	 * K = W_k * (memory + pos), V = W_v * memory.
	 */
	struct sam3_tensor *mem_keyed = NULL;
	if (mem_pos) {
		mem_keyed = gh_add(g, arena, memory, mem_pos);
		if (!mem_keyed) {
			sam3_log_error("mem_attn: memory + pos failed");
			return SAM3_ENOMEM;
		}
	}

	struct sam3_tensor *x = current; /* [seq, d_model] */

	for (int i = 0; i < attn->n_layers; i++) {
		struct sam3_memattn_layer *layer = &attn->layers[i];

		/*
		 * 1. Pre-norm self-attention with RoPE.
		 *    LayerNorm -> self-attn -> residual add.
		 */
		struct sam3_tensor *x_norm = gh_layernorm(g, arena, x,
			layer->norm1_w, layer->norm1_b);
		if (!x_norm) {
			sam3_log_error("mem_attn: layer %d norm1 failed", i);
			return SAM3_ENOMEM;
		}

		/*
		 * gh_multihead_attention_rope_sep expects [batch, seq, d].
		 * Reshape from [seq, d] to [1, seq, d].
		 */
		int sa_in_dims[] = {1, seq, d};
		struct sam3_tensor *sa_in = gh_reshape(g, arena, x_norm,
			3, sa_in_dims);
		if (!sa_in) {
			sam3_log_error("mem_attn: layer %d sa reshape failed",
				       i);
			return SAM3_ENOMEM;
		}

		/* Self-attn returns [batch*seq, d] = [seq, d] */
		struct sam3_tensor *sa_out =
			gh_multihead_attention_rope_sep(
				g, arena, sa_in,
				layer->sa_q_w, layer->sa_q_b,
				layer->sa_k_w, layer->sa_k_b,
				layer->sa_v_w, layer->sa_v_b,
				layer->sa_out_w, layer->sa_out_b,
				attn->n_heads,
				attn->rope_cos, attn->rope_sin,
				NULL, /* no attn mask */
				attn->feat_w, 1.0f);
		if (!sa_out) {
			sam3_log_error("mem_attn: layer %d self-attn failed",
				       i);
			return SAM3_ENOMEM;
		}

		/* Residual add */
		x = gh_add(g, arena, x, sa_out);
		if (!x) {
			sam3_log_error("mem_attn: layer %d sa residual failed",
				       i);
			return SAM3_ENOMEM;
		}

		/*
		 * 2. Pre-norm cross-attention.
		 *    LayerNorm -> cross-attn -> residual add.
		 */
		x_norm = gh_layernorm(g, arena, x,
			layer->norm2_w, layer->norm2_b);
		if (!x_norm) {
			sam3_log_error("mem_attn: layer %d norm2 failed", i);
			return SAM3_ENOMEM;
		}

		struct sam3_tensor *ca_out = gh_cross_attention_sep(
			g, arena, x_norm, memory, mem_keyed,
			layer->ca_q_w, layer->ca_q_b,
			layer->ca_k_w, layer->ca_k_b,
			layer->ca_v_w, layer->ca_v_b,
			layer->ca_out_w, layer->ca_out_b,
			attn->n_heads);
		if (!ca_out) {
			sam3_log_error("mem_attn: layer %d cross-attn failed",
				       i);
			return SAM3_ENOMEM;
		}

		x = gh_add(g, arena, x, ca_out);
		if (!x) {
			sam3_log_error("mem_attn: layer %d ca residual failed",
				       i);
			return SAM3_ENOMEM;
		}

		/*
		 * 3. Pre-norm FFN.
		 *    LayerNorm -> MLP(ReLU) -> residual add.
		 */
		x_norm = gh_layernorm(g, arena, x,
			layer->norm3_w, layer->norm3_b);
		if (!x_norm) {
			sam3_log_error("mem_attn: layer %d norm3 failed", i);
			return SAM3_ENOMEM;
		}

		struct sam3_tensor *ffn_out = gh_mlp(g, arena, x_norm,
			layer->ffn_fc1_w, layer->ffn_fc1_b,
			layer->ffn_fc2_w, layer->ffn_fc2_b,
			SAM3_OP_RELU);
		if (!ffn_out) {
			sam3_log_error("mem_attn: layer %d FFN failed", i);
			return SAM3_ENOMEM;
		}

		x = gh_add(g, arena, x, ffn_out);
		if (!x) {
			sam3_log_error("mem_attn: layer %d ffn residual failed",
				       i);
			return SAM3_ENOMEM;
		}

		sam3_log_debug("mem_attn: layer %d built", i);
	}

	/* Final LayerNorm */
	x = gh_layernorm(g, arena, x,
		attn->final_norm_w, attn->final_norm_b);
	if (!x) {
		sam3_log_error("mem_attn: final layernorm failed");
		return SAM3_ENOMEM;
	}

	*output = x;
	sam3_log_debug("mem_attn: output [%d,%d]", x->dims[0], x->dims[1]);
	return SAM3_OK;
}
