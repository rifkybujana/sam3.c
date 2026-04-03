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

enum sam3_error sam3_vit_load(struct sam3_vit *vit,
			       const struct sam3_weight_file *wf,
			       struct sam3_arena *arena)
{
	int e = vit->embed_dim;
	int e3 = e * 3;
	int m = vit->mlp_dim;
	int ps = vit->patch_size;
	char name[128];

	/* Patch embedding: conv2d weight [embed_dim, 3, ps, ps] */
	int pe_w_dims[] = {e, 3, ps, ps};
	vit->patch_embed_w = gh_load_or_alloc(wf, "vit.patch_embed.weight",
					    arena, SAM3_DTYPE_F32,
					    4, pe_w_dims);
	if (!vit->patch_embed_w)
		return SAM3_ENOMEM;

	/* Patch embedding bias [embed_dim] */
	int pe_b_dims[] = {e};
	vit->patch_embed_b = gh_load_or_alloc(wf, "vit.patch_embed.bias",
					    arena, SAM3_DTYPE_F32,
					    1, pe_b_dims);
	if (!vit->patch_embed_b)
		return SAM3_ENOMEM;

	/* Per-layer weights */
	int e_dims[] = {e};
	int qkv_w_dims[] = {e3, e};
	int qkv_b_dims[] = {e3};
	int proj_w_dims[] = {e, e};
	int fc1_w_dims[] = {m, e};
	int fc1_b_dims[] = {m};
	int fc2_w_dims[] = {e, m};

	for (int i = 0; i < vit->depth; i++) {
		/* Layer norm 1 */
		snprintf(name, sizeof(name),
			 "vit.layer.%d.ln1.weight", i);
		vit->layers[i].ln1_w = gh_load_or_alloc(wf, name, arena,
						      SAM3_DTYPE_F32,
						      1, e_dims);
		if (!vit->layers[i].ln1_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "vit.layer.%d.ln1.bias", i);
		vit->layers[i].ln1_b = gh_load_or_alloc(wf, name, arena,
						      SAM3_DTYPE_F32,
						      1, e_dims);
		if (!vit->layers[i].ln1_b)
			return SAM3_ENOMEM;

		/* Attention QKV */
		snprintf(name, sizeof(name),
			 "vit.layer.%d.attn.qkv.weight", i);
		vit->layers[i].qkv_w = gh_load_or_alloc(wf, name, arena,
						       SAM3_DTYPE_F32,
						       2, qkv_w_dims);
		if (!vit->layers[i].qkv_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "vit.layer.%d.attn.qkv.bias", i);
		vit->layers[i].qkv_b = gh_load_or_alloc(wf, name, arena,
						       SAM3_DTYPE_F32,
						       1, qkv_b_dims);
		if (!vit->layers[i].qkv_b)
			return SAM3_ENOMEM;

		/* Attention output projection */
		snprintf(name, sizeof(name),
			 "vit.layer.%d.attn.proj.weight", i);
		vit->layers[i].proj_w = gh_load_or_alloc(wf, name, arena,
						       SAM3_DTYPE_F32,
						       2, proj_w_dims);
		if (!vit->layers[i].proj_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "vit.layer.%d.attn.proj.bias", i);
		vit->layers[i].proj_b = gh_load_or_alloc(wf, name, arena,
						       SAM3_DTYPE_F32,
						       1, e_dims);
		if (!vit->layers[i].proj_b)
			return SAM3_ENOMEM;

		/* Layer norm 2 */
		snprintf(name, sizeof(name),
			 "vit.layer.%d.ln2.weight", i);
		vit->layers[i].ln2_w = gh_load_or_alloc(wf, name, arena,
						      SAM3_DTYPE_F32,
						      1, e_dims);
		if (!vit->layers[i].ln2_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "vit.layer.%d.ln2.bias", i);
		vit->layers[i].ln2_b = gh_load_or_alloc(wf, name, arena,
						      SAM3_DTYPE_F32,
						      1, e_dims);
		if (!vit->layers[i].ln2_b)
			return SAM3_ENOMEM;

		/* MLP fc1 */
		snprintf(name, sizeof(name),
			 "vit.layer.%d.mlp.fc1.weight", i);
		vit->layers[i].mlp_fc1_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32, 2, fc1_w_dims);
		if (!vit->layers[i].mlp_fc1_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "vit.layer.%d.mlp.fc1.bias", i);
		vit->layers[i].mlp_fc1_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32, 1, fc1_b_dims);
		if (!vit->layers[i].mlp_fc1_b)
			return SAM3_ENOMEM;

		/* MLP fc2 */
		snprintf(name, sizeof(name),
			 "vit.layer.%d.mlp.fc2.weight", i);
		vit->layers[i].mlp_fc2_w = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32, 2, fc2_w_dims);
		if (!vit->layers[i].mlp_fc2_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "vit.layer.%d.mlp.fc2.bias", i);
		vit->layers[i].mlp_fc2_b = gh_load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32, 1, e_dims);
		if (!vit->layers[i].mlp_fc2_b)
			return SAM3_ENOMEM;
	}

	return SAM3_OK;
}

struct sam3_tensor *sam3_vit_build(struct sam3_vit *vit,
				    struct sam3_graph *g,
				    struct sam3_tensor *image,
				    struct sam3_arena *arena)
{
	int gs = vit->grid_size;
	int e = vit->embed_dim;
	int np = vit->n_patches;

	/*
	 * Step 1: Patch embedding via conv2d.
	 *
	 * Input image is [3, H, W]. Reshape to [1, 3, H, W] for conv2d.
	 * Conv2d with stride=patch_size, padding=0 produces
	 * [1, embed_dim, grid_size, grid_size].
	 */
	int img4d_dims[] = {1, 3, vit->img_size, vit->img_size};
	struct sam3_tensor *image_4d;
	image_4d = gh_reshape(g, arena, image, 4, img4d_dims);
	if (!image_4d)
		return NULL;

	int conv_dims[] = {1, e, gs, gs};
	struct sam3_tensor *conv_out;
	conv_out = gh_alloc_tensor(arena, SAM3_DTYPE_F32, 4, conv_dims);
	if (!conv_out)
		return NULL;

	struct sam3_tensor *conv_inputs[] = {image_4d, vit->patch_embed_w};
	conv_out = sam3_graph_add_op(g, SAM3_OP_CONV2D,
				      conv_inputs, 2, conv_out);
	if (!conv_out)
		return NULL;

	/* Set conv2d params: stride and padding */
	struct sam3_node *conv_node = &g->nodes[g->n_nodes - 1];
	conv_node->params[0] = vit->patch_size; /* stride */
	conv_node->params[1] = 0;               /* padding */

	/*
	 * Reshape to [n_patches, embed_dim].
	 * conv_out is [1, embed_dim, gs, gs].
	 * Reshape to [embed_dim, n_patches], transpose to get
	 * [n_patches, embed_dim], then add bias [embed_dim] which
	 * broadcasts as [n_patches, embed_dim] + [embed_dim].
	 */
	int flat_dims[] = {e, np};
	struct sam3_tensor *x;
	x = gh_reshape(g, arena, conv_out, 2, flat_dims);
	if (!x)
		return NULL;

	x = gh_transpose(g, arena, x);
	if (!x)
		return NULL;
	/* x is now [n_patches, embed_dim] */

	/* Add patch embedding bias (broadcast: [np, e] + [e]) */
	x = gh_add(g, arena, x, vit->patch_embed_b);
	if (!x)
		return NULL;

	/*
	 * Step 2: Transformer blocks.
	 *
	 * For each layer:
	 *   a. Pre-norm attention with residual
	 *   b. Pre-norm MLP with residual
	 *
	 * gh_multihead_attention expects q as [batch, seq, d_model].
	 * Our x is [n_patches, embed_dim] (2D). We reshape to
	 * [1, n_patches, embed_dim] for attention.
	 */
	for (int i = 0; i < vit->depth; i++) {
		/* Pre-norm for attention */
		struct sam3_tensor *x_norm;
		x_norm = gh_layernorm(g, arena, x,
				       vit->layers[i].ln1_w,
				       vit->layers[i].ln1_b);
		if (!x_norm)
			return NULL;

		/* Reshape to 3D for multihead attention */
		int attn_dims[] = {1, np, e};
		struct sam3_tensor *x3d;
		x3d = gh_reshape(g, arena, x_norm, 3, attn_dims);
		if (!x3d)
			return NULL;

		/* Self-attention (Q=K=V=x_norm) with RoPE */
		struct sam3_tensor *mask = vit->layers[i].is_global
					? NULL : vit->window_mask;
		struct sam3_tensor *attn;
		attn = gh_multihead_attention_rope(
			g, arena,
			x3d, x3d, x3d,
			vit->layers[i].qkv_w,
			vit->layers[i].qkv_b,
			vit->layers[i].proj_w,
			vit->layers[i].proj_b,
			vit->n_heads,
			vit->rope_cos,    /* RoPE cos */
			vit->rope_sin,    /* RoPE sin */
			mask);             /* window mask for non-global */
		if (!attn)
			return NULL;

		/*
		 * attn output is [batch*seq, embed_dim] =
		 * [n_patches, embed_dim] which matches x's shape.
		 * Add residual.
		 */
		x = gh_add(g, arena, x, attn);
		if (!x)
			return NULL;

		/* Pre-norm for MLP */
		x_norm = gh_layernorm(g, arena, x,
				       vit->layers[i].ln2_w,
				       vit->layers[i].ln2_b);
		if (!x_norm)
			return NULL;

		/* MLP: fc1 -> GELU -> fc2 */
		struct sam3_tensor *ff;
		ff = gh_mlp(g, arena, x_norm,
			     vit->layers[i].mlp_fc1_w,
			     vit->layers[i].mlp_fc1_b,
			     vit->layers[i].mlp_fc2_w,
			     vit->layers[i].mlp_fc2_b,
			     SAM3_OP_GELU);
		if (!ff)
			return NULL;

		/* Residual connection */
		x = gh_add(g, arena, x, ff);
		if (!x)
			return NULL;
	}

	/* x is [n_patches, embed_dim] */
	return x;
}
