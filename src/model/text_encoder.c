/*
 * src/model/text_encoder.c - CLIP text encoder graph construction
 *
 * Implements weight loading and compute graph construction for a
 * 24-layer CLIP-style transformer text encoder. The encoder takes
 * token IDs, embeds them with learned token and positional embeddings,
 * runs through N transformer blocks with pre-norm self-attention and
 * GELU MLP, then projects the pooled output to the final d_model
 * dimension.
 *
 * Key types:  sam3_text_encoder
 * Depends on: text_encoder.h, graph_helpers.h
 * Used by:    sam3.c (top-level API)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>

#include "text_encoder.h"
#include "graph_helpers.h"

/*
 * load_or_alloc - Load a weight tensor by name, or allocate zeroed.
 *
 * @wf:     Open weight file (may be NULL)
 * @name:   Tensor name to look up
 * @arena:  Arena for allocation
 * @dtype:  Data type for the tensor
 * @n_dims: Number of dimensions
 * @dims:   Array of dimension sizes
 *
 * If wf is non-NULL and the tensor is found, it is loaded from the
 * weight file. Otherwise a zero-initialized tensor is allocated from
 * the arena. Returns NULL if the arena is full.
 */
static struct sam3_tensor *load_or_alloc(const struct sam3_weight_file *wf,
					  const char *name,
					  struct sam3_arena *arena,
					  enum sam3_dtype dtype,
					  int n_dims, const int *dims)
{
	if (wf) {
		const struct sam3_weight_tensor_desc *desc;
		desc = sam3_weight_find(wf, name);
		if (desc) {
			struct sam3_tensor *t;
			t = gh_alloc_tensor(arena, dtype, n_dims, dims);
			if (t)
				sam3_weight_to_tensor(wf, desc, t);
			return t;
		}
	}
	/* Fallback: allocate zeroed tensor */
	return gh_alloc_tensor(arena, dtype, n_dims, dims);
}

enum sam3_error sam3_text_encoder_load(struct sam3_text_encoder *te,
				       const struct sam3_weight_file *wf,
				       struct sam3_arena *arena)
{
	int w = te->width;
	int d = te->d_model;
	int h4 = w * 4;
	int w3 = w * 3;
	char name[128];

	/* Embeddings */
	int tok_dims[] = {te->vocab_size, w};
	te->token_embedding = load_or_alloc(wf, "text.token_embedding",
					     arena, SAM3_DTYPE_F32,
					     2, tok_dims);
	if (!te->token_embedding)
		return SAM3_ENOMEM;

	int pos_dims[] = {te->context_len, w};
	te->pos_embedding = load_or_alloc(wf, "text.pos_embedding",
					   arena, SAM3_DTYPE_F32,
					   2, pos_dims);
	if (!te->pos_embedding)
		return SAM3_ENOMEM;

	/* Final layer norm */
	int w_dims[] = {w};
	te->ln_final_w = load_or_alloc(wf, "text.ln_final.weight",
					arena, SAM3_DTYPE_F32, 1, w_dims);
	if (!te->ln_final_w)
		return SAM3_ENOMEM;

	te->ln_final_b = load_or_alloc(wf, "text.ln_final.bias",
					arena, SAM3_DTYPE_F32, 1, w_dims);
	if (!te->ln_final_b)
		return SAM3_ENOMEM;

	/* Text projection */
	int proj_dims[] = {w, d};
	te->text_projection = load_or_alloc(wf, "text.text_projection",
					     arena, SAM3_DTYPE_F32,
					     2, proj_dims);
	if (!te->text_projection)
		return SAM3_ENOMEM;

	/* Per-layer weights */
	for (int i = 0; i < te->n_layers; i++) {
		/* ln1 */
		snprintf(name, sizeof(name),
			 "text.layers.%d.ln1.weight", i);
		te->layers[i].ln1_w = load_or_alloc(wf, name, arena,
						     SAM3_DTYPE_F32,
						     1, w_dims);
		if (!te->layers[i].ln1_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "text.layers.%d.ln1.bias", i);
		te->layers[i].ln1_b = load_or_alloc(wf, name, arena,
						     SAM3_DTYPE_F32,
						     1, w_dims);
		if (!te->layers[i].ln1_b)
			return SAM3_ENOMEM;

		/* Attention QKV */
		int qkv_w_dims[] = {w3, w};
		snprintf(name, sizeof(name),
			 "text.layers.%d.attn.qkv.weight", i);
		te->layers[i].attn_qkv_w = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32, 2, qkv_w_dims);
		if (!te->layers[i].attn_qkv_w)
			return SAM3_ENOMEM;

		int qkv_b_dims[] = {w3};
		snprintf(name, sizeof(name),
			 "text.layers.%d.attn.qkv.bias", i);
		te->layers[i].attn_qkv_b = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32, 1, qkv_b_dims);
		if (!te->layers[i].attn_qkv_b)
			return SAM3_ENOMEM;

		/* Attention output projection */
		int out_w_dims[] = {w, w};
		snprintf(name, sizeof(name),
			 "text.layers.%d.attn.out.weight", i);
		te->layers[i].attn_out_w = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32, 2, out_w_dims);
		if (!te->layers[i].attn_out_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "text.layers.%d.attn.out.bias", i);
		te->layers[i].attn_out_b = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32, 1, w_dims);
		if (!te->layers[i].attn_out_b)
			return SAM3_ENOMEM;

		/* ln2 */
		snprintf(name, sizeof(name),
			 "text.layers.%d.ln2.weight", i);
		te->layers[i].ln2_w = load_or_alloc(wf, name, arena,
						     SAM3_DTYPE_F32,
						     1, w_dims);
		if (!te->layers[i].ln2_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "text.layers.%d.ln2.bias", i);
		te->layers[i].ln2_b = load_or_alloc(wf, name, arena,
						     SAM3_DTYPE_F32,
						     1, w_dims);
		if (!te->layers[i].ln2_b)
			return SAM3_ENOMEM;

		/* MLP fc1 */
		int fc1_w_dims[] = {h4, w};
		snprintf(name, sizeof(name),
			 "text.layers.%d.mlp.fc1.weight", i);
		te->layers[i].mlp_fc1_w = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32, 2, fc1_w_dims);
		if (!te->layers[i].mlp_fc1_w)
			return SAM3_ENOMEM;

		int fc1_b_dims[] = {h4};
		snprintf(name, sizeof(name),
			 "text.layers.%d.mlp.fc1.bias", i);
		te->layers[i].mlp_fc1_b = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32, 1, fc1_b_dims);
		if (!te->layers[i].mlp_fc1_b)
			return SAM3_ENOMEM;

		/* MLP fc2 */
		int fc2_w_dims[] = {w, h4};
		snprintf(name, sizeof(name),
			 "text.layers.%d.mlp.fc2.weight", i);
		te->layers[i].mlp_fc2_w = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32, 2, fc2_w_dims);
		if (!te->layers[i].mlp_fc2_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 "text.layers.%d.mlp.fc2.bias", i);
		te->layers[i].mlp_fc2_b = load_or_alloc(
			wf, name, arena, SAM3_DTYPE_F32, 1, w_dims);
		if (!te->layers[i].mlp_fc2_b)
			return SAM3_ENOMEM;
	}

	return SAM3_OK;
}

struct sam3_tensor *sam3_text_encoder_build(
	struct sam3_text_encoder *te,
	struct sam3_graph *g,
	struct sam3_tensor *token_ids,
	struct sam3_tensor **pooled_out,
	struct sam3_arena *arena)
{
	int seq_len = sam3_tensor_nelems(token_ids);

	/*
	 * Step 1: Token embedding lookup.
	 * token_embedding is [vocab_size, width], token_ids is [seq_len].
	 * Result: [seq_len, width].
	 */
	struct sam3_tensor *x = gh_embed(g, arena,
					  te->token_embedding, token_ids);
	if (!x)
		return NULL;

	/*
	 * Step 2: Add positional embedding.
	 * pos_embedding is [context_len, width].  We slice it to
	 * [seq_len, width] if seq_len < context_len, then add.
	 */
	struct sam3_tensor *pos = te->pos_embedding;
	if (seq_len < te->context_len) {
		pos = gh_slice(g, arena, pos, 0, 0, seq_len);
		if (!pos)
			return NULL;
	}
	x = gh_add(g, arena, x, pos);
	if (!x)
		return NULL;

	/*
	 * Step 3: Transformer blocks.
	 * Each block: pre-norm self-attention + residual,
	 *             pre-norm MLP + residual.
	 *
	 * gh_multihead_attention expects q as [batch, seq, d_model].
	 * Our x is [seq_len, width] (2D).  We reshape to
	 * [1, seq_len, width] for attention, then back to 2D.
	 */
	for (int i = 0; i < te->n_layers; i++) {
		/* Pre-norm for attention */
		struct sam3_tensor *x_norm;
		x_norm = gh_layernorm(g, arena, x,
				       te->layers[i].ln1_w,
				       te->layers[i].ln1_b);
		if (!x_norm)
			return NULL;

		/* Reshape to 3D for multihead attention */
		int attn_dims[] = {1, seq_len, te->width};
		struct sam3_tensor *x3d;
		x3d = gh_reshape(g, arena, x_norm, 3, attn_dims);
		if (!x3d)
			return NULL;

		/* Self-attention (Q=K=V=x_norm) */
		struct sam3_tensor *attn;
		attn = gh_multihead_attention(
			g, arena,
			x3d, x3d, x3d,
			te->layers[i].attn_qkv_w,
			te->layers[i].attn_qkv_b,
			te->layers[i].attn_out_w,
			te->layers[i].attn_out_b,
			te->n_heads);
		if (!attn)
			return NULL;

		/*
		 * attn output is [batch*seq, width] = [seq_len, width]
		 * which matches x's shape.  Add residual.
		 */
		x = gh_add(g, arena, x, attn);
		if (!x)
			return NULL;

		/* Pre-norm for MLP */
		x_norm = gh_layernorm(g, arena, x,
				       te->layers[i].ln2_w,
				       te->layers[i].ln2_b);
		if (!x_norm)
			return NULL;

		/* MLP: fc1 -> GELU -> fc2 */
		struct sam3_tensor *ff;
		ff = gh_mlp(g, arena, x_norm,
			     te->layers[i].mlp_fc1_w,
			     te->layers[i].mlp_fc1_b,
			     te->layers[i].mlp_fc2_w,
			     te->layers[i].mlp_fc2_b,
			     SAM3_OP_GELU);
		if (!ff)
			return NULL;

		/* Residual connection */
		x = gh_add(g, arena, x, ff);
		if (!x)
			return NULL;
	}

	/*
	 * Step 4: Final layer norm.
	 * x is [seq_len, width].
	 */
	x = gh_layernorm(g, arena, x, te->ln_final_w, te->ln_final_b);
	if (!x)
		return NULL;

	/*
	 * Step 5: Pooled output (from last token / EOT position).
	 * Slice the last row: [1, width].
	 * Project through text_projection: matmul([1, width], [width, d_model])
	 *   -> [1, d_model].
	 * Reshape to [d_model].
	 */
	if (pooled_out) {
		struct sam3_tensor *last_tok;
		last_tok = gh_slice(g, arena, x, 0,
				     seq_len - 1, seq_len);
		if (!last_tok)
			return NULL;
		/* last_tok is [1, width] */

		struct sam3_tensor *projected;
		projected = gh_matmul(g, arena, last_tok,
				       te->text_projection);
		if (!projected)
			return NULL;
		/* projected is [1, d_model] */

		int pool_dims[] = {te->d_model};
		*pooled_out = gh_reshape(g, arena, projected,
					  1, pool_dims);
		if (!*pooled_out)
			return NULL;
	}

	/*
	 * Step 6: Project all per-token embeddings to d_model.
	 * x is [seq_len, width].
	 * matmul(x, text_projection) -> [seq_len, d_model].
	 */
	struct sam3_tensor *out;
	out = gh_matmul(g, arena, x, te->text_projection);

	return out;
}
