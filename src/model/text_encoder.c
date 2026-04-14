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
 * Weight name prefixes for CLIP text encoder weights in the .sam3 file.
 * Original PyTorch: detector_model.text_encoder.text_model.*
 */
#define TE_P "detector_model.text_encoder.text_model."
#define TE_L TE_P "encoder.layers."

/*
 * fuse_3 - Load 3 separate [d, d_in] weights and fuse into [3*d, d_in].
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

enum sam3_error sam3_text_encoder_load(struct sam3_text_encoder *te,
				       const struct sam3_weight_file *wf,
				       struct sam3_arena *arena)
{
	int w = te->width;
	int d = te->d_model;
	int h4 = w * 4;
	char name[128];

	/* Embeddings */
	int tok_dims[] = {te->vocab_size, w};
	te->token_embedding = gh_load_mmap(wf,
		TE_P "embeddings.token_embedding.weight",
		arena, SAM3_DTYPE_F32, 2, tok_dims);
	if (!te->token_embedding)
		return SAM3_ENOMEM;

	int pos_dims[] = {te->context_len, w};
	te->pos_embedding = gh_load_mmap(wf,
		TE_P "embeddings.position_embedding.weight",
		arena, SAM3_DTYPE_F32, 2, pos_dims);
	if (!te->pos_embedding)
		return SAM3_ENOMEM;

	/* Final layer norm */
	int w_dims[] = {w};
	te->ln_final_w = gh_load_mmap(wf,
		TE_P "final_layer_norm.weight",
		arena, SAM3_DTYPE_F32, 1, w_dims);
	if (!te->ln_final_w)
		return SAM3_ENOMEM;

	te->ln_final_b = gh_load_mmap(wf,
		TE_P "final_layer_norm.bias",
		arena, SAM3_DTYPE_F32, 1, w_dims);
	if (!te->ln_final_b)
		return SAM3_ENOMEM;

	/* Text projection (resizer): detector_model.text_projection.* */
	int proj_dims[] = {d, w};
	te->text_projection = gh_load_mmap(wf,
		"detector_model.text_projection.weight",
		arena, SAM3_DTYPE_F32, 2, proj_dims);
	if (!te->text_projection)
		return SAM3_ENOMEM;

	int d_dims[] = {d};
	te->text_projection_b = gh_load_mmap(wf,
		"detector_model.text_projection.bias",
		arena, SAM3_DTYPE_F32, 1, d_dims);
	if (!te->text_projection_b)
		return SAM3_ENOMEM;

	/* Per-layer weights */
	for (int i = 0; i < te->n_layers; i++) {
		/* ln1 */
		snprintf(name, sizeof(name),
			 TE_L "%d.layer_norm1.weight", i);
		te->layers[i].ln1_w = gh_load_mmap(wf, name, arena,
						     SAM3_DTYPE_F32,
						     1, w_dims);
		if (!te->layers[i].ln1_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 TE_L "%d.layer_norm1.bias", i);
		te->layers[i].ln1_b = gh_load_mmap(wf, name, arena,
						     SAM3_DTYPE_F32,
						     1, w_dims);
		if (!te->layers[i].ln1_b)
			return SAM3_ENOMEM;

		/* Attention QKV: fuse Q/K/V into [3*w, w] / [3*w] */
		{
			char q_name[128], k_name[128], v_name[128];
			int qkv_w_dims[] = {w, w};
			snprintf(q_name, sizeof(q_name),
				 TE_L "%d.self_attn.q_proj.weight", i);
			snprintf(k_name, sizeof(k_name),
				 TE_L "%d.self_attn.k_proj.weight", i);
			snprintf(v_name, sizeof(v_name),
				 TE_L "%d.self_attn.v_proj.weight", i);
			te->layers[i].qkv_w = fuse_3(wf, q_name, k_name,
						       v_name, arena, w,
						       2, qkv_w_dims);
			if (!te->layers[i].qkv_w)
				return SAM3_ENOMEM;

			snprintf(q_name, sizeof(q_name),
				 TE_L "%d.self_attn.q_proj.bias", i);
			snprintf(k_name, sizeof(k_name),
				 TE_L "%d.self_attn.k_proj.bias", i);
			snprintf(v_name, sizeof(v_name),
				 TE_L "%d.self_attn.v_proj.bias", i);
			te->layers[i].qkv_b = fuse_3(wf, q_name, k_name,
						       v_name, arena, w,
						       1, w_dims);
			if (!te->layers[i].qkv_b)
				return SAM3_ENOMEM;
		}

		/* Attention output projection */
		int out_w_dims[] = {w, w};
		snprintf(name, sizeof(name),
			 TE_L "%d.self_attn.out_proj.weight", i);
		te->layers[i].attn_out_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32, 2, out_w_dims);
		if (!te->layers[i].attn_out_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 TE_L "%d.self_attn.out_proj.bias", i);
		te->layers[i].attn_out_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32, 1, w_dims);
		if (!te->layers[i].attn_out_b)
			return SAM3_ENOMEM;

		/* ln2 */
		snprintf(name, sizeof(name),
			 TE_L "%d.layer_norm2.weight", i);
		te->layers[i].ln2_w = gh_load_mmap(wf, name, arena,
						     SAM3_DTYPE_F32,
						     1, w_dims);
		if (!te->layers[i].ln2_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 TE_L "%d.layer_norm2.bias", i);
		te->layers[i].ln2_b = gh_load_mmap(wf, name, arena,
						     SAM3_DTYPE_F32,
						     1, w_dims);
		if (!te->layers[i].ln2_b)
			return SAM3_ENOMEM;

		/* MLP fc1 */
		int fc1_w_dims[] = {h4, w};
		snprintf(name, sizeof(name),
			 TE_L "%d.mlp.fc1.weight", i);
		te->layers[i].mlp_fc1_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32, 2, fc1_w_dims);
		if (!te->layers[i].mlp_fc1_w)
			return SAM3_ENOMEM;

		int fc1_b_dims[] = {h4};
		snprintf(name, sizeof(name),
			 TE_L "%d.mlp.fc1.bias", i);
		te->layers[i].mlp_fc1_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32, 1, fc1_b_dims);
		if (!te->layers[i].mlp_fc1_b)
			return SAM3_ENOMEM;

		/* MLP fc2 */
		int fc2_w_dims[] = {w, h4};
		snprintf(name, sizeof(name),
			 TE_L "%d.mlp.fc2.weight", i);
		te->layers[i].mlp_fc2_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32, 2, fc2_w_dims);
		if (!te->layers[i].mlp_fc2_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 TE_L "%d.mlp.fc2.bias", i);
		te->layers[i].mlp_fc2_b = gh_load_mmap(
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
	 * Step 3: Build causal attention mask.
	 *
	 * Upper triangle filled with -1e9 (masked), lower triangle
	 * and diagonal = 0.0 (allowed).  Shape: [seq_len, seq_len].
	 */
	int mask_dims[] = {seq_len, seq_len};
	struct sam3_tensor *causal_mask;
	causal_mask = gh_alloc_tensor(arena, SAM3_DTYPE_F32,
				       2, mask_dims);
	if (!causal_mask)
		return NULL;

	float *mask_data = (float *)causal_mask->data;
	for (int r = 0; r < seq_len; r++) {
		for (int c = 0; c < seq_len; c++) {
			mask_data[r * seq_len + c] =
				(c > r) ? -1e9f : 0.0f;
		}
	}

	/*
	 * Step 4: Transformer blocks.
	 * Each block: pre-norm self-attention + residual,
	 *             pre-norm MLP + residual.
	 *
	 * gh_multihead_attention_rope expects q as [batch, seq, d_model].
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

		/* Self-attention with fused QKV and causal mask */
		struct sam3_tensor *attn;
		attn = gh_multihead_attention_rope(
			g, arena,
			x3d, NULL, NULL,
			te->layers[i].qkv_w,
			te->layers[i].qkv_b,
			te->layers[i].attn_out_w,
			te->layers[i].attn_out_b,
			te->n_heads,
			NULL, NULL,
			causal_mask, 0, 0.0f);
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
	 * Step 5: Final layer norm.
	 * x is [seq_len, width].
	 */
	x = gh_layernorm(g, arena, x, te->ln_final_w, te->ln_final_b);
	if (!x)
		return NULL;

	/*
	 * Transpose text_projection: file stores [d_model, width],
	 * matmul needs [width, d_model].
	 */
	struct sam3_tensor *proj_t = gh_transpose(g, arena,
						   te->text_projection);
	if (!proj_t)
		return NULL;

	/*
	 * Step 6: Pooled output (from last token / EOT position).
	 * Slice the last row: [1, width].
	 * Project: matmul([1, width], [width, d_model]) + bias -> [1, d_model].
	 * Reshape to [d_model].
	 */
	if (pooled_out) {
		struct sam3_tensor *last_tok;
		last_tok = gh_slice(g, arena, x, 0,
				     seq_len - 1, seq_len);
		if (!last_tok)
			return NULL;

		struct sam3_tensor *projected;
		projected = gh_matmul(g, arena, last_tok, proj_t);
		if (!projected)
			return NULL;

		projected = gh_add(g, arena, projected,
				   te->text_projection_b);
		if (!projected)
			return NULL;

		int pool_dims[] = {te->d_model};
		*pooled_out = gh_reshape(g, arena, projected,
					  1, pool_dims);
		if (!*pooled_out)
			return NULL;
	}

	/*
	 * Step 7: Project all per-token embeddings to d_model.
	 * x is [seq_len, width].
	 * matmul(x, proj_t) + bias -> [seq_len, d_model].
	 */
	struct sam3_tensor *out;
	out = gh_matmul(g, arena, x, proj_t);
	if (!out)
		return NULL;

	out = gh_add(g, arena, out, te->text_projection_b);

	return out;
}

/*
 * sam3_text_encoder_build_perblock - Per-block text encoder evaluation.
 *
 * Evaluates the text encoder one block at a time, persisting the
 * block output between evaluations. Dumps /tmp/dbg_te_block_XX.bin
 * and other diagnostic files for fixture comparison.
 *
 * @te:      Loaded text encoder
 * @be:      Backend for graph evaluation
 * @token_ids: Input token IDs [seq_len] (I32 tensor)
 * @scratch: Arena for per-block intermediate tensors (reset between blocks)
 * @persist: Arena for output buffer that survives across blocks
 *
 * Returns per-token embeddings [seq_len, d_model], or NULL on error.
 */
struct sam3_tensor *sam3_text_encoder_build_perblock(
	struct sam3_text_encoder *te,
	struct sam3_backend *be,
	struct sam3_tensor *token_ids,
	struct sam3_arena *scratch,
	struct sam3_arena *persist)
{
	struct sam3_graph g;
	int seq_len = sam3_tensor_nelems(token_ids);
	int w = te->width;
	size_t x_bytes = (size_t)seq_len * (size_t)w * sizeof(float);
	enum sam3_error err;

	/* Allocate persist buffer for block outputs [seq_len, width] */
	void *x_buf = sam3_arena_alloc(persist, x_bytes);
	if (!x_buf)
		return NULL;

#ifdef SAM3_DEBUG_DUMP
	/* Dump token IDs for verification */
	{
		FILE *fp = fopen("/tmp/dbg_te_token_ids.bin", "wb");
		if (fp) {
			fwrite(token_ids->data, sizeof(int32_t),
			       (size_t)seq_len, fp);
			fclose(fp);
		}
	}
#endif

	/* Step 1a: Token embedding only (for fixture comparison) */
	{
		sam3_arena_reset(scratch);
		sam3_graph_init(&g);

		struct sam3_tensor *tok;
		tok = gh_embed(&g, scratch, te->token_embedding, token_ids);
		if (!tok)
			return NULL;

		err = be->ops->graph_eval(be, &g);
		if (err != SAM3_OK)
			return NULL;

#ifdef SAM3_DEBUG_DUMP
		/* Dump token embedding only (no pos) */
		{
			FILE *fp = fopen("/tmp/dbg_te_tok_only.bin", "wb");
			if (fp) {
				fwrite(tok->data, sizeof(float),
				       (size_t)seq_len * (size_t)w, fp);
				fclose(fp);
			}
		}
#endif
	}

	/* Step 1b: Token embedding + positional embedding */
	{
		sam3_arena_reset(scratch);
		sam3_graph_init(&g);

		struct sam3_tensor *x;
		x = gh_embed(&g, scratch, te->token_embedding, token_ids);
		if (!x)
			return NULL;

		struct sam3_tensor *pos = te->pos_embedding;
		if (seq_len < te->context_len) {
			pos = gh_slice(&g, scratch, pos, 0, 0, seq_len);
			if (!pos)
				return NULL;
		}
		x = gh_add(&g, scratch, x, pos);
		if (!x)
			return NULL;

		err = be->ops->graph_eval(be, &g);
		if (err != SAM3_OK)
			return NULL;

		memcpy(x_buf, x->data, x_bytes);

#ifdef SAM3_DEBUG_DUMP
		/* Dump token+pos embedding */
		{
			FILE *fp = fopen("/tmp/dbg_te_token_embed.bin", "wb");
			if (fp) {
				fwrite(x_buf, sizeof(float),
				       (size_t)seq_len * (size_t)w, fp);
				fclose(fp);
			}
		}
#endif
	}

	/* Build causal mask (persistent across blocks) */
	int mask_dims[] = {seq_len, seq_len};
	struct sam3_tensor *causal_mask;
	causal_mask = gh_alloc_tensor(persist, SAM3_DTYPE_F32,
				       2, mask_dims);
	if (!causal_mask)
		return NULL;

	float *mask_data = (float *)causal_mask->data;
	for (int r = 0; r < seq_len; r++) {
		for (int c = 0; c < seq_len; c++) {
			mask_data[r * seq_len + c] =
				(c > r) ? -1e9f : 0.0f;
		}
	}

	/* Step 2: Batched block evaluation (up to 4 blocks per graph_eval) */
	{
		int max_batch = 4;

		for (int base = 0; base < te->n_layers; ) {
			int end = base + max_batch;
			if (end > te->n_layers)
				end = te->n_layers;

			sam3_arena_reset(scratch);
			sam3_graph_init(&g);

			int x_dims[] = {seq_len, w};
			struct sam3_tensor *x;
			x = gh_tensor_wrap(scratch, SAM3_DTYPE_F32,
					    2, x_dims, x_buf);
			if (!x)
				return NULL;

			struct sam3_tensor *mask_wrap;
			mask_wrap = gh_tensor_wrap(scratch,
				SAM3_DTYPE_F32, 2,
				mask_dims, causal_mask->data);
			if (!mask_wrap)
				return NULL;

			int actually_built = 0;

			for (int i = base; i < end; i++) {
				size_t pre = scratch->offset;

				/* Pre-norm for attention */
				struct sam3_tensor *x_norm;
				x_norm = gh_layernorm(&g, scratch, x,
					te->layers[i].ln1_w,
					te->layers[i].ln1_b);
				if (!x_norm)
					return NULL;

				/* Reshape to 3D for MHA */
				int attn_dims[] = {1, seq_len, w};
				struct sam3_tensor *x3d;
				x3d = gh_reshape(&g, scratch, x_norm,
						  3, attn_dims);
				if (!x3d)
					return NULL;

				/* Self-attention with causal mask */
				struct sam3_tensor *attn;
				attn = gh_multihead_attention_rope(
					&g, scratch,
					x3d, NULL, NULL,
					te->layers[i].qkv_w,
					te->layers[i].qkv_b,
					te->layers[i].attn_out_w,
					te->layers[i].attn_out_b,
					te->n_heads,
					NULL, NULL,
					mask_wrap, 0, 0.0f);
				if (!attn)
					return NULL;

				/* Residual */
				x = gh_add(&g, scratch, x, attn);
				if (!x)
					return NULL;

				/* Pre-norm for MLP */
				x_norm = gh_layernorm(&g, scratch, x,
					te->layers[i].ln2_w,
					te->layers[i].ln2_b);
				if (!x_norm)
					return NULL;

				/* MLP: fc1 -> GELU -> fc2 */
				struct sam3_tensor *ff;
				ff = gh_mlp(&g, scratch, x_norm,
					te->layers[i].mlp_fc1_w,
					te->layers[i].mlp_fc1_b,
					te->layers[i].mlp_fc2_w,
					te->layers[i].mlp_fc2_b,
					SAM3_OP_GELU);
				if (!ff)
					return NULL;

				/* Residual */
				x = gh_add(&g, scratch, x, ff);
				if (!x)
					return NULL;

				actually_built++;

				size_t block_cost = scratch->offset - pre;
				size_t remaining = scratch->size
					- scratch->offset;

				if (i + 1 < end
				    && remaining < block_cost)
					break;
			}

			err = be->ops->graph_eval(be, &g);
			if (err != SAM3_OK)
				return NULL;

			memcpy(x_buf, x->data, x_bytes);
			base += actually_built;

#ifdef SAM3_DEBUG_DUMP
			if (base >= te->n_layers
			    || base + max_batch >= te->n_layers) {
				char path[64];
				snprintf(path, sizeof(path),
					 "/tmp/dbg_te_block_%02d.bin",
					 end - 1);
				FILE *fp = fopen(path, "wb");
				if (fp) {
					fwrite(x_buf, sizeof(float),
					       (size_t)seq_len
					       * (size_t)w, fp);
					fclose(fp);
				}
			}
#endif
		}
	}

	/* Step 3: Final layer norm + projection */
	{
		sam3_arena_reset(scratch);
		sam3_graph_init(&g);

		int x_dims[] = {seq_len, w};
		struct sam3_tensor *x;
		x = gh_tensor_wrap(scratch, SAM3_DTYPE_F32, 2,
				    x_dims, x_buf);
		if (!x)
			return NULL;

		x = gh_layernorm(&g, scratch, x,
				  te->ln_final_w, te->ln_final_b);
		if (!x)
			return NULL;

#ifdef SAM3_DEBUG_DUMP
		struct sam3_tensor *ln_out = x;
#endif

		struct sam3_tensor *proj_t;
		proj_t = gh_transpose(&g, scratch, te->text_projection);
		if (!proj_t)
			return NULL;

		struct sam3_tensor *out;
		out = gh_matmul(&g, scratch, x, proj_t);
		if (!out)
			return NULL;

		out = gh_add(&g, scratch, out, te->text_projection_b);
		if (!out)
			return NULL;

		err = be->ops->graph_eval(be, &g);
		if (err != SAM3_OK)
			return NULL;

#ifdef SAM3_DEBUG_DUMP
		/* Dump ln_final */
		{
			FILE *fp = fopen("/tmp/dbg_te_ln_final.bin", "wb");
			if (fp) {
				fwrite(ln_out->data, sizeof(float),
				       (size_t)seq_len * (size_t)w, fp);
				fclose(fp);
			}
		}
#endif

		/* Copy result to persist arena */
		int out_dims[] = {seq_len, te->d_model};
		struct sam3_tensor *result;
		result = gh_alloc_tensor(persist, SAM3_DTYPE_F32,
					  2, out_dims);
		if (!result)
			return NULL;
		memcpy(result->data, out->data, result->nbytes);
		return result;
	}
}
