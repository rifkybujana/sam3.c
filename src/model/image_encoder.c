/*
 * src/model/image_encoder.c - ViT image encoder graph construction
 *
 * Implements initialization, weight loading, and compute graph
 * construction for the Vision Transformer (ViT) backbone used in
 * SAM3. The encoder takes a normalized RGB image, applies patch
 * embedding via conv2d, runs through 32 transformer blocks with
 * multi-head self-attention and GELU MLP, and produces per-patch
 * feature embeddings.
 *
 * RoPE frequencies are precomputed during init using 2D positional
 * encoding (y for first half of head dimensions, x for second half).
 * A window mask is precomputed for non-global layers: patches in the
 * same window get 0.0f, patches in different windows get -1e9f.
 *
 * Key types:  sam3_vit
 * Depends on: image_encoder.h, graph_helpers.h
 * Used by:    sam3.c (top-level API)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "image_encoder.h"
#include "graph_helpers.h"
#include "util/log.h"

/* Global attention block indices */
static const int global_blocks[] = {7, 15, 23, 31};

/*
 * precompute_rope - Precompute 2D RoPE cosine and sine tables.
 *
 * Each patch position (y, x) maps to a frequency vector. The first
 * half of head dimensions encode the y position, the second half
 * encodes the x position.
 */
static enum sam3_error precompute_rope(struct sam3_vit *vit,
					struct sam3_arena *arena)
{
	int head_dim = vit->embed_dim / vit->n_heads; /* 64 */
	int half = head_dim / 2;                       /* 32 */
	float theta = 10000.0f;

	int cos_dims[] = {vit->n_patches, half};
	vit->rope_cos = gh_alloc_tensor(arena, SAM3_DTYPE_F32,
					 2, cos_dims);
	if (!vit->rope_cos)
		return SAM3_ENOMEM;

	int sin_dims[] = {vit->n_patches, half};
	vit->rope_sin = gh_alloc_tensor(arena, SAM3_DTYPE_F32,
					 2, sin_dims);
	if (!vit->rope_sin)
		return SAM3_ENOMEM;

	float *cos_data = (float *)vit->rope_cos->data;
	float *sin_data = (float *)vit->rope_sin->data;

	/* 2D position encoding: treat each patch position as (y, x) */
	for (int py = 0; py < vit->grid_size; py++) {
		for (int px = 0; px < vit->grid_size; px++) {
			int pos = py * vit->grid_size + px;
			for (int d = 0; d < half; d++) {
				float freq = 1.0f / powf(theta,
					2.0f * (float)d / (float)head_dim);
				float angle;
				if (d < half / 2)
					angle = (float)py * freq;
				else
					angle = (float)px * freq;
				cos_data[pos * half + d] = cosf(angle);
				sin_data[pos * half + d] = sinf(angle);
			}
		}
	}

	return SAM3_OK;
}

/*
 * precompute_window_mask - Build additive mask for windowed attention.
 *
 * Allocates an [n_patches, n_patches] F32 tensor. For each pair (i, j),
 * computes grid positions and checks if they fall in the same window.
 * Same-window pairs get 0.0f; different-window pairs get -1e9f.
 */
static enum sam3_error precompute_window_mask(struct sam3_vit *vit,
					      struct sam3_arena *arena)
{
	int np = vit->n_patches;
	int gs = vit->grid_size;
	int ws = vit->window_size;

	int mask_dims[] = {np, np};
	vit->window_mask = gh_alloc_tensor(arena, SAM3_DTYPE_F32,
					   2, mask_dims);
	if (!vit->window_mask)
		return SAM3_ENOMEM;

	float *data = (float *)vit->window_mask->data;

	for (int i = 0; i < np; i++) {
		int iy = i / gs;
		int ix = i % gs;
		int wy = iy / ws;
		int wx = ix / ws;

		for (int j = 0; j < np; j++) {
			int jy = j / gs;
			int jx = j % gs;
			int same = (wy == jy / ws) && (wx == jx / ws);
			data[i * np + j] = same ? 0.0f : -1e9f;
		}
	}

	return SAM3_OK;
}

enum sam3_error sam3_vit_init(struct sam3_vit *vit,
			       int img_size, int patch_size,
			       int embed_dim, int depth, int n_heads,
			       int window_size, int mlp_dim,
			       struct sam3_arena *arena)
{
	memset(vit, 0, sizeof(*vit));

	vit->img_size = img_size;
	vit->patch_size = patch_size;
	vit->embed_dim = embed_dim;
	vit->depth = depth;
	vit->n_heads = n_heads;
	vit->window_size = window_size;
	vit->mlp_dim = mlp_dim;
	vit->grid_size = img_size / patch_size;
	vit->n_patches = vit->grid_size * vit->grid_size;

	/* Mark global attention blocks */
	for (int i = 0; i < depth; i++) {
		vit->layers[i].is_global = 0;
		for (int j = 0; j < SAM3_VIT_N_GLOBAL_BLOCKS; j++) {
			if (i == global_blocks[j]) {
				vit->layers[i].is_global = 1;
				break;
			}
		}
	}

	enum sam3_error err = precompute_rope(vit, arena);
	if (err != SAM3_OK)
		return err;

	return precompute_window_mask(vit, arena);
}

/*
 * Weight name prefix for ViT backbone weights in the .sam3 file.
 * Original PyTorch: detector_model.vision_encoder.backbone.*
 */
#define VIT_P "detector_model.vision_encoder.backbone."

/*
 * load_fuse_qkv - Load separate Q, K, V weights and fuse into [3*dim, dim].
 *
 * The .sam3 file stores Q, K, V as separate [dim, dim] tensors, but
 * the ViT compute graph expects a fused [3*dim, dim] for efficiency.
 */
static struct sam3_tensor *load_fuse_qkv(
	const struct sam3_weight_file *wf,
	struct sam3_arena *arena,
	int dim,
	const char *q_name,
	const char *k_name,
	const char *v_name,
	int n_dims, const int *single_dims)
{
	struct sam3_tensor *q_t, *k_t, *v_t, *fused;
	size_t single_bytes;

	q_t = gh_load_or_alloc(wf, q_name, arena,
			       SAM3_DTYPE_F32, n_dims, single_dims);
	k_t = gh_load_or_alloc(wf, k_name, arena,
			       SAM3_DTYPE_F32, n_dims, single_dims);
	v_t = gh_load_or_alloc(wf, v_name, arena,
			       SAM3_DTYPE_F32, n_dims, single_dims);
	if (!q_t || !k_t || !v_t)
		return NULL;

	single_bytes = q_t->nbytes;

	/* Allocate fused tensor: first dim is tripled */
	int fused_dims[4];
	for (int i = 0; i < n_dims; i++)
		fused_dims[i] = single_dims[i];
	fused_dims[0] *= 3;

	fused = gh_alloc_tensor(arena, SAM3_DTYPE_F32, n_dims, fused_dims);
	if (!fused)
		return NULL;

	memcpy((char *)fused->data, q_t->data, single_bytes);
	memcpy((char *)fused->data + single_bytes, k_t->data, single_bytes);
	memcpy((char *)fused->data + 2 * single_bytes,
	       v_t->data, single_bytes);

	return fused;
}

enum sam3_error sam3_vit_load(struct sam3_vit *vit,
			       const struct sam3_weight_file *wf,
			       struct sam3_arena *arena)
{
	int e = vit->embed_dim;
	int m = vit->mlp_dim;
	int ps = vit->patch_size;
	char name[128], q_name[128], k_name[128], v_name[128];

	/* Patch embedding: conv2d weight [embed_dim, 3, ps, ps] */
	int pe_w_dims[] = {e, 3, ps, ps};
	vit->patch_embed_w = gh_load_or_alloc(wf,
		VIT_P "embeddings.patch_embeddings.projection.weight",
		arena, SAM3_DTYPE_F32, 4, pe_w_dims);
	if (!vit->patch_embed_w)
		return SAM3_ENOMEM;

	/* Patch embedding bias [embed_dim] — may not exist in file */
	int pe_b_dims[] = {e};
	vit->patch_embed_b = gh_load_or_alloc(wf,
		VIT_P "embeddings.patch_embeddings.projection.bias",
		arena, SAM3_DTYPE_F32, 1, pe_b_dims);
	if (!vit->patch_embed_b)
		return SAM3_ENOMEM;

	/* Per-layer weights */
	int e_dims[] = {e};
	int single_w_dims[] = {e, e};
	int single_b_dims[] = {e};
	int proj_w_dims[] = {e, e};
	int fc1_w_dims[] = {m, e};
	int fc1_b_dims[] = {m};
	int fc2_w_dims[] = {e, m};

	for (int i = 0; i < vit->depth; i++) {
		/* Layer norm 1 */
		snprintf(name, sizeof(name),
			 VIT_P "layers.%d.layer_norm1.weight", i);
		vit->layers[i].ln1_w = gh_load_or_alloc(wf, name, arena,
						      SAM3_DTYPE_F32,
						      1, e_dims);
		if (!vit->layers[i].ln1_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 VIT_P "layers.%d.layer_norm1.bias", i);
		vit->layers[i].ln1_b = gh_load_or_alloc(wf, name, arena,
						      SAM3_DTYPE_F32,
						      1, e_dims);
		if (!vit->layers[i].ln1_b)
			return SAM3_ENOMEM;

		/* Attention QKV: fuse from separate Q, K, V */
		snprintf(q_name, sizeof(q_name),
			 VIT_P "layers.%d.attention.q_proj.weight", i);
		snprintf(k_name, sizeof(k_name),
			 VIT_P "layers.%d.attention.k_proj.weight", i);
		snprintf(v_name, sizeof(v_name),
			 VIT_P "layers.%d.attention.v_proj.weight", i);
		vit->layers[i].qkv_w = load_fuse_qkv(wf, arena, e,
			q_name, k_name, v_name, 2, single_w_dims);
		if (!vit->layers[i].qkv_w)
			return SAM3_ENOMEM;

		snprintf(q_name, sizeof(q_name),
			 VIT_P "layers.%d.attention.q_proj.bias", i);
		snprintf(k_name, sizeof(k_name),
			 VIT_P "layers.%d.attention.k_proj.bias", i);
		snprintf(v_name, sizeof(v_name),
			 VIT_P "layers.%d.attention.v_proj.bias", i);
		vit->layers[i].qkv_b = load_fuse_qkv(wf, arena, e,
			q_name, k_name, v_name, 1, single_b_dims);
		if (!vit->layers[i].qkv_b)
			return SAM3_ENOMEM;

		/* Attention output projection */
		snprintf(name, sizeof(name),
			 VIT_P "layers.%d.attention.o_proj.weight", i);
		vit->layers[i].proj_w = gh_load_or_alloc(wf, name, arena,
						       SAM3_DTYPE_F32,
						       2, proj_w_dims);
		if (!vit->layers[i].proj_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 VIT_P "layers.%d.attention.o_proj.bias", i);
		vit->layers[i].proj_b = gh_load_or_alloc(wf, name, arena,
						       SAM3_DTYPE_F32,
						       1, e_dims);
		if (!vit->layers[i].proj_b)
			return SAM3_ENOMEM;

		/* Layer norm 2 */
		snprintf(name, sizeof(name),
			 VIT_P "layers.%d.layer_norm2.weight", i);
		vit->layers[i].ln2_w = gh_load_or_alloc(wf, name, arena,
						      SAM3_DTYPE_F32,
						      1, e_dims);
		if (!vit->layers[i].ln2_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 VIT_P "layers.%d.layer_norm2.bias", i);
		vit->layers[i].ln2_b = gh_load_or_alloc(wf, name, arena,
						      SAM3_DTYPE_F32,
						      1, e_dims);
		if (!vit->layers[i].ln2_b)
			return SAM3_ENOMEM;

		/* MLP fc1 */
		snprintf(name, sizeof(name),
			 VIT_P "layers.%d.mlp.fc1.weight", i);
		vit->layers[i].mlp_fc1_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32, 2, fc1_w_dims);
		if (!vit->layers[i].mlp_fc1_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 VIT_P "layers.%d.mlp.fc1.bias", i);
		vit->layers[i].mlp_fc1_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32, 1, fc1_b_dims);
		if (!vit->layers[i].mlp_fc1_b)
			return SAM3_ENOMEM;

		/* MLP fc2 */
		snprintf(name, sizeof(name),
			 VIT_P "layers.%d.mlp.fc2.weight", i);
		vit->layers[i].mlp_fc2_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32, 2, fc2_w_dims);
		if (!vit->layers[i].mlp_fc2_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 VIT_P "layers.%d.mlp.fc2.bias", i);
		vit->layers[i].mlp_fc2_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32, 1, e_dims);
		if (!vit->layers[i].mlp_fc2_b)
			return SAM3_ENOMEM;
	}

	/* Final layer norm */
	vit->ln_final_w = gh_load_or_alloc(wf,
		VIT_P "layer_norm.weight",
		arena, SAM3_DTYPE_F32, 1, e_dims);
	if (!vit->ln_final_w)
		return SAM3_ENOMEM;

	vit->ln_final_b = gh_load_or_alloc(wf,
		VIT_P "layer_norm.bias",
		arena, SAM3_DTYPE_F32, 1, e_dims);
	if (!vit->ln_final_b)
		return SAM3_ENOMEM;

	return SAM3_OK;
}

struct sam3_tensor *sam3_vit_build(struct sam3_vit *vit,
				    struct sam3_backend *be,
				    struct sam3_tensor *image,
				    struct sam3_arena *scratch,
				    struct sam3_arena *persist)
{
	int gs = vit->grid_size;
	int e = vit->embed_dim;
	int np = vit->n_patches;
	size_t x_bytes = (size_t)np * e * sam3_dtype_size(SAM3_DTYPE_F32);
	struct sam3_graph g;
	enum sam3_error err;

	/*
	 * Allocate persistent buffer for the block output that
	 * survives arena resets between blocks.
	 */
	void *x_buf = sam3_arena_alloc(persist, x_bytes);
	if (!x_buf)
		return NULL;

	/*
	 * Step 1: Patch embedding via conv2d.
	 *
	 * Build a small sub-graph, evaluate it, and copy the result
	 * to the persistent buffer. Do NOT reset scratch here — the
	 * caller's image tensor lives in it. First reset happens
	 * after patch embedding is evaluated and copied to x_buf.
	 */
	sam3_graph_init(&g);

	int img4d_dims[] = {1, 3, vit->img_size, vit->img_size};
	struct sam3_tensor *image_4d;
	image_4d = gh_reshape(&g, scratch, image, 4, img4d_dims);
	if (!image_4d)
		return NULL;

	int conv_dims[] = {1, e, gs, gs};
	struct sam3_tensor *conv_out;
	conv_out = gh_alloc_tensor(scratch, SAM3_DTYPE_F32, 4, conv_dims);
	if (!conv_out)
		return NULL;

	struct sam3_tensor *conv_inputs[] = {image_4d, vit->patch_embed_w};
	conv_out = sam3_graph_add_op(&g, SAM3_OP_CONV2D,
				      conv_inputs, 2, conv_out);
	if (!conv_out)
		return NULL;

	struct sam3_node *conv_node = &g.nodes[g.n_nodes - 1];
	conv_node->params[0] = vit->patch_size;
	conv_node->params[1] = 0;

	/*
	 * Reshape to [n_patches, embed_dim].
	 * conv_out is [1, embed_dim, gs, gs].
	 * Reshape to [embed_dim, n_patches], transpose to get
	 * [n_patches, embed_dim], then add bias.
	 */
	int flat_dims[] = {e, np};
	struct sam3_tensor *x;
	x = gh_reshape(&g, scratch, conv_out, 2, flat_dims);
	if (!x)
		return NULL;

	x = gh_transpose(&g, scratch, x);
	if (!x)
		return NULL;

	x = gh_add(&g, scratch, x, vit->patch_embed_b);
	if (!x)
		return NULL;

	/* Evaluate patch embedding graph */
	err = be->ops->graph_eval(be, &g);
	if (err != SAM3_OK)
		return NULL;

	/* Copy patch embedding result to persistent buffer */
	memcpy(x_buf, x->data, x_bytes);

	sam3_log_info("vit: patch embedding evaluated (%d patches)", np);
	sam3_log_info("vit: scratch arena: %zu / %zu bytes used",
		       scratch->offset, scratch->size);

	/*
	 * Step 2: Per-block transformer evaluation.
	 *
	 * For each layer, reset scratch, build one block's graph
	 * using x_buf as input, evaluate, and copy the result back.
	 */
	for (int i = 0; i < vit->depth; i++) {
		sam3_graph_init(&g);
		sam3_arena_reset(scratch);

		/* Wrap persistent buffer as input tensor in scratch */
		int x_dims[] = {np, e};
		x = gh_tensor_wrap(scratch, SAM3_DTYPE_F32,
				    2, x_dims, x_buf);
		if (!x)
			return NULL;

		/* Pre-norm for attention */
		struct sam3_tensor *x_norm;
		x_norm = gh_layernorm(&g, scratch, x,
				       vit->layers[i].ln1_w,
				       vit->layers[i].ln1_b);
		if (!x_norm)
			return NULL;

		/* Reshape to 3D for multihead attention */
		int attn_dims[] = {1, np, e};
		struct sam3_tensor *x3d;
		x3d = gh_reshape(&g, scratch, x_norm, 3, attn_dims);
		if (!x3d)
			return NULL;

		/* Self-attention with RoPE and optional window mask */
		struct sam3_tensor *mask = vit->layers[i].is_global
					? NULL : vit->window_mask;
		struct sam3_tensor *attn;
		attn = gh_multihead_attention_rope(
			&g, scratch,
			x3d, x3d, x3d,
			vit->layers[i].qkv_w,
			vit->layers[i].qkv_b,
			vit->layers[i].proj_w,
			vit->layers[i].proj_b,
			vit->n_heads,
			vit->rope_cos,
			vit->rope_sin,
			mask);
		if (!attn) {
			sam3_log_error("vit: block %d attention OOM "
				       "(scratch %zu / %zu)",
				       i, scratch->offset, scratch->size);
			return NULL;
		}

		/* Residual: x + attn */
		x = gh_add(&g, scratch, x, attn);
		if (!x)
			return NULL;

		/* Pre-norm for MLP */
		x_norm = gh_layernorm(&g, scratch, x,
				       vit->layers[i].ln2_w,
				       vit->layers[i].ln2_b);
		if (!x_norm)
			return NULL;

		/* MLP: fc1 -> GELU -> fc2 */
		struct sam3_tensor *ff;
		ff = gh_mlp(&g, scratch, x_norm,
			     vit->layers[i].mlp_fc1_w,
			     vit->layers[i].mlp_fc1_b,
			     vit->layers[i].mlp_fc2_w,
			     vit->layers[i].mlp_fc2_b,
			     SAM3_OP_GELU);
		if (!ff) {
			sam3_log_error("vit: block %d MLP OOM "
				       "(scratch %zu / %zu)",
				       i, scratch->offset, scratch->size);
			return NULL;
		}

		/* Residual: x + ff */
		x = gh_add(&g, scratch, x, ff);
		if (!x)
			return NULL;

		/* Evaluate this block */
		err = be->ops->graph_eval(be, &g);
		if (err != SAM3_OK)
			return NULL;

		/* Save result to persistent buffer */
		memcpy(x_buf, x->data, x_bytes);

		sam3_log_debug("vit: block %d/%d evaluated", i + 1,
			       vit->depth);
	}

	sam3_log_info("vit: all %d blocks evaluated", vit->depth);

	/*
	 * Step 3: Final layer norm.
	 *
	 * Apply layer_norm to the output of the last transformer block.
	 * This normalizes the ViT features before they enter the FPN neck.
	 */
	sam3_graph_init(&g);
	sam3_arena_reset(scratch);

	int ln_dims[] = {np, e};
	x = gh_tensor_wrap(scratch, SAM3_DTYPE_F32, 2, ln_dims, x_buf);
	if (!x)
		return NULL;

	x = gh_layernorm(&g, scratch, x, vit->ln_final_w, vit->ln_final_b);
	if (!x)
		return NULL;

	err = be->ops->graph_eval(be, &g);
	if (err != SAM3_OK)
		return NULL;

	memcpy(x_buf, x->data, x_bytes);

	sam3_log_info("vit: final layer norm applied");

	/* Return a tensor in persist arena wrapping the final output */
	int out_dims[] = {np, e};
	return gh_tensor_wrap(persist, SAM3_DTYPE_F32, 2, out_dims, x_buf);
}
