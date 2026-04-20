/*
 * src/model/mobileclip_text.c - MobileCLIP text encoder implementation
 *
 * Variant config table, iface vtable ops, and weight loader for the
 * MobileCLIP text encoder (S0/S1/L). The loader fills all embedding,
 * final-norm, projection, and per-block standard-attention tensors.
 * RepMixer blocks (S0 indices 0 and 5) are marked is_repmixer=1 and
 * skipped here; Task 6.1 fills them.
 *
 * Key types:  sam3_mobileclip_text_encoder
 * Depends on: mobileclip_text.h, graph_helpers.h, text_encoder_iface.h,
 *             util/log.h
 * Used by:    src/model/text_encoder_iface.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>

#include "mobileclip_text.h"
#include "graph_helpers.h"
#include "text_encoder_iface.h"
#include "util/log.h"

/* Weight key prefixes */
#define ENC_PFX      "detector.backbone.language_backbone.encoder."
#define BACKBONE_PFX "detector.backbone.language_backbone."

/* --- Variant configs (post-Phase-0 audit values) --- */

static const struct sam3_mobileclip_config mobileclip_s0 = {
	.text_backbone           = SAM3_TEXT_MOBILECLIP_S0,
	.n_layers                = 6,
	.width                   = 512,
	.n_heads                 = 8,
	.mlp_dim                 = 2048,
	.ctx_len                 = 16,
	.out_dim                 = 256,
	.vocab_size              = 49408,
	.pos_embed_table_len     = 77,
	.n_repmixer_blocks       = 2,
	.repmixer_block_indices  = { 0, 5, 0, 0 },
};

static const struct sam3_mobileclip_config mobileclip_s1 = {
	.text_backbone           = SAM3_TEXT_MOBILECLIP_S1,
	.n_layers                = 12,
	.width                   = 512,
	.n_heads                 = 8,
	.mlp_dim                 = 2048,
	.ctx_len                 = 16,
	.out_dim                 = 256,
	.vocab_size              = 49408,
	.pos_embed_table_len     = 77,
	.n_repmixer_blocks       = 0,
	.repmixer_block_indices  = { 0 },
};

static const struct sam3_mobileclip_config mobileclip_l = {
	.text_backbone           = SAM3_TEXT_MOBILECLIP_L,
	.n_layers                = 12,
	.width                   = 768,
	.n_heads                 = 12,
	.mlp_dim                 = 3072,
	.ctx_len                 = 16,
	.out_dim                 = 256,
	.vocab_size              = 49408,
	.pos_embed_table_len     = 77,
	.n_repmixer_blocks       = 0,
	.repmixer_block_indices  = { 0 },
};

const struct sam3_mobileclip_config *sam3_mobileclip_config_for(int text_backbone)
{
	switch (text_backbone) {
	case SAM3_TEXT_MOBILECLIP_S0: return &mobileclip_s0;
	case SAM3_TEXT_MOBILECLIP_S1: return &mobileclip_s1;
	case SAM3_TEXT_MOBILECLIP_L:  return &mobileclip_l;
	default:                      return NULL;
	}
}

/* --- iface vtable wiring --- */

static enum sam3_error mc_load(struct sam3_text_encoder_iface *iface,
			       const struct sam3_weight_file *wf,
			       struct sam3_arena *arena)
{
	return sam3_mobileclip_text_load(
		(struct sam3_mobileclip_text_encoder *)iface->impl, wf, arena);
}

static struct sam3_tensor *mc_build(struct sam3_text_encoder_iface *iface,
				    struct sam3_graph *g,
				    struct sam3_tensor *token_ids,
				    struct sam3_tensor **pooled_out,
				    struct sam3_arena *arena)
{
	return sam3_mobileclip_text_build(
		(struct sam3_mobileclip_text_encoder *)iface->impl, g,
		token_ids, pooled_out, arena);
}

static struct sam3_tensor *mc_build_perblock(
	struct sam3_text_encoder_iface *iface,
	struct sam3_backend *be,
	struct sam3_tensor *token_ids,
	struct sam3_arena *scratch,
	struct sam3_arena *persist)
{
	return sam3_mobileclip_text_build_perblock(
		(struct sam3_mobileclip_text_encoder *)iface->impl, be,
		token_ids, scratch, persist);
}

static void mc_free(struct sam3_text_encoder_iface *iface)
{
	(void)iface; /* arena-backed */
}

const struct sam3_text_encoder_iface_ops sam3_mobileclip_text_iface_ops = {
	.load           = mc_load,
	.build          = mc_build,
	.build_perblock = mc_build_perblock,
	.free           = mc_free,
};

enum sam3_error sam3_mobileclip_text_iface_init_impl(
	struct sam3_text_encoder_iface *iface,
	int text_backbone, struct sam3_arena *arena)
{
	const struct sam3_mobileclip_config *cfg;
	struct sam3_mobileclip_text_encoder *enc;

	cfg = sam3_mobileclip_config_for(text_backbone);
	if (!cfg) {
		sam3_log_error("mobileclip: unknown variant %d", text_backbone);
		return SAM3_EINVAL;
	}

	enc = sam3_arena_alloc(arena, sizeof(*enc));
	if (!enc) {
		sam3_log_error("mobileclip: arena alloc failed");
		return SAM3_ENOMEM;
	}
	memset(enc, 0, sizeof(*enc));
	enc->cfg = *cfg;

	iface->impl    = enc;
	iface->ops     = &sam3_mobileclip_text_iface_ops;
	iface->ctx_len = cfg->ctx_len;
	iface->d_model = cfg->out_dim;
	return SAM3_OK;
}

/* --- Helpers for weight loading --- */

/*
 * load_required - Lookup tensor by name; log error and return NULL on miss.
 *
 * Used by Task 6.1 for RepMixer keys that must be present. Kept here so
 * the symbol is defined before sam3_mobileclip_text_load.
 */
static struct sam3_tensor *load_required(
	const struct sam3_weight_file *wf, struct sam3_arena *arena,
	const char *name, enum sam3_dtype dtype, int n_dims, const int *dims)
{
	struct sam3_tensor *t = gh_load_mmap_optional(
		wf, name, arena, dtype, n_dims, dims);
	if (!t)
		sam3_log_error("mobileclip: required tensor %s not found", name);
	return t;
}

/* --- sam3_mobileclip_text_load --- */

enum sam3_error sam3_mobileclip_text_load(
	struct sam3_mobileclip_text_encoder *enc,
	const struct sam3_weight_file *wf,
	struct sam3_arena *arena)
{
	char name[256];
	int dims[4];
	const struct sam3_mobileclip_config *cfg;

	if (!enc || !arena)
		return SAM3_EINVAL;
	cfg = &enc->cfg;

	/* Token embedding: [vocab_size, width] */
	dims[0] = cfg->vocab_size; dims[1] = cfg->width;
	enc->token_embedding = gh_load_mmap(
		wf, ENC_PFX "embedding_layer.weight",
		arena, SAM3_DTYPE_F32, 2, dims);
	if (!enc->token_embedding)
		return SAM3_ENOMEM;

	/* Positional embedding: stored as [1, 1, 77, width] in checkpoint.
	 * Build code will slice [ctx_len, width] from this at inference. */
	dims[0] = 1; dims[1] = 1;
	dims[2] = cfg->pos_embed_table_len; dims[3] = cfg->width;
	enc->pos_embed_full = gh_load_mmap(
		wf, ENC_PFX "positional_embedding.pos_embed.pos_embed",
		arena, SAM3_DTYPE_F32, 4, dims);
	if (!enc->pos_embed_full)
		return SAM3_ENOMEM;

	/* Final layer norm: [width] */
	dims[0] = cfg->width;
	enc->ln_final_w = gh_load_mmap(
		wf, ENC_PFX "final_layer_norm.weight",
		arena, SAM3_DTYPE_F32, 1, dims);
	if (!enc->ln_final_w)
		return SAM3_ENOMEM;
	enc->ln_final_b = gh_load_mmap(
		wf, ENC_PFX "final_layer_norm.bias",
		arena, SAM3_DTYPE_F32, 1, dims);
	if (!enc->ln_final_b)
		return SAM3_ENOMEM;

	/* Inner projection (width -> width). Raw tensor, NO .weight suffix. */
	dims[0] = cfg->width; dims[1] = cfg->width;
	enc->projection_layer = gh_load_mmap(
		wf, ENC_PFX "projection_layer",
		arena, SAM3_DTYPE_F32, 2, dims);
	if (!enc->projection_layer)
		return SAM3_ENOMEM;

	/* External 256-dim projector weight: [out_dim, width] */
	dims[0] = cfg->out_dim; dims[1] = cfg->width;
	enc->out_proj_w = gh_load_mmap(
		wf, BACKBONE_PFX "projector.weight",
		arena, SAM3_DTYPE_F32, 2, dims);
	if (!enc->out_proj_w)
		return SAM3_ENOMEM;

	/* Projector bias: [out_dim]. May be absent; build code checks NULL. */
	dims[0] = cfg->out_dim;
	enc->out_proj_b = gh_load_mmap_optional(
		wf, BACKBONE_PFX "projector.bias",
		arena, SAM3_DTYPE_F32, 1, dims);

	/* Per-block tensors.
	 * RepMixer indices (S0: {0, 5}) are marked and skipped here;
	 * Task 6.1 fills them. Standard blocks load pre_norm_mha.*
	 * and pre_norm_ffn.{0,1,4} keys. */
	for (int i = 0; i < cfg->n_layers; i++) {
		int is_rep = 0;
		for (int k = 0; k < cfg->n_repmixer_blocks; k++) {
			if (cfg->repmixer_block_indices[k] == i) {
				is_rep = 1;
				break;
			}
		}
		if (is_rep) {
			enc->layers[i].is_repmixer = 1;
			continue;
		}
		enc->layers[i].is_repmixer = 0;

		struct sam3_mobileclip_layer_std *L = &enc->layers[i].u.std;

		/* LN1 (pre-norm MHA): [width] */
		dims[0] = cfg->width;
		snprintf(name, sizeof(name),
			 ENC_PFX "transformer.%d.pre_norm_mha.0.weight", i);
		L->ln1_w = gh_load_mmap(wf, name, arena,
					SAM3_DTYPE_F32, 1, dims);
		if (!L->ln1_w) return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 ENC_PFX "transformer.%d.pre_norm_mha.0.bias", i);
		L->ln1_b = gh_load_mmap(wf, name, arena,
					SAM3_DTYPE_F32, 1, dims);
		if (!L->ln1_b) return SAM3_ENOMEM;

		/* QKV projection: [3*width, width] (already fused in ckpt) */
		dims[0] = 3 * cfg->width; dims[1] = cfg->width;
		snprintf(name, sizeof(name),
			 ENC_PFX "transformer.%d"
			 ".pre_norm_mha.1.qkv_proj.weight", i);
		L->qkv_w = gh_load_mmap(wf, name, arena,
					SAM3_DTYPE_F32, 2, dims);
		if (!L->qkv_w) return SAM3_ENOMEM;

		dims[0] = 3 * cfg->width;
		snprintf(name, sizeof(name),
			 ENC_PFX "transformer.%d"
			 ".pre_norm_mha.1.qkv_proj.bias", i);
		L->qkv_b = gh_load_mmap(wf, name, arena,
					SAM3_DTYPE_F32, 1, dims);
		if (!L->qkv_b) return SAM3_ENOMEM;

		/* Out projection: [width, width] */
		dims[0] = cfg->width; dims[1] = cfg->width;
		snprintf(name, sizeof(name),
			 ENC_PFX "transformer.%d"
			 ".pre_norm_mha.1.out_proj.weight", i);
		L->out_w = gh_load_mmap(wf, name, arena,
					SAM3_DTYPE_F32, 2, dims);
		if (!L->out_w) return SAM3_ENOMEM;

		dims[0] = cfg->width;
		snprintf(name, sizeof(name),
			 ENC_PFX "transformer.%d"
			 ".pre_norm_mha.1.out_proj.bias", i);
		L->out_b = gh_load_mmap(wf, name, arena,
					SAM3_DTYPE_F32, 1, dims);
		if (!L->out_b) return SAM3_ENOMEM;

		/* LN2 (pre-norm FFN): sequential index 0 */
		dims[0] = cfg->width;
		snprintf(name, sizeof(name),
			 ENC_PFX "transformer.%d.pre_norm_ffn.0.weight", i);
		L->ln2_w = gh_load_mmap(wf, name, arena,
					SAM3_DTYPE_F32, 1, dims);
		if (!L->ln2_w) return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 ENC_PFX "transformer.%d.pre_norm_ffn.0.bias", i);
		L->ln2_b = gh_load_mmap(wf, name, arena,
					SAM3_DTYPE_F32, 1, dims);
		if (!L->ln2_b) return SAM3_ENOMEM;

		/* FC1 (sequential index 1): [mlp_dim, width] */
		dims[0] = cfg->mlp_dim; dims[1] = cfg->width;
		snprintf(name, sizeof(name),
			 ENC_PFX "transformer.%d.pre_norm_ffn.1.weight", i);
		L->fc1_w = gh_load_mmap(wf, name, arena,
					SAM3_DTYPE_F32, 2, dims);
		if (!L->fc1_w) return SAM3_ENOMEM;

		dims[0] = cfg->mlp_dim;
		snprintf(name, sizeof(name),
			 ENC_PFX "transformer.%d.pre_norm_ffn.1.bias", i);
		L->fc1_b = gh_load_mmap(wf, name, arena,
					SAM3_DTYPE_F32, 1, dims);
		if (!L->fc1_b) return SAM3_ENOMEM;

		/* FC2 (sequential index 4; indices 2-3 are GELU+Dropout):
		 * [width, mlp_dim] */
		dims[0] = cfg->width; dims[1] = cfg->mlp_dim;
		snprintf(name, sizeof(name),
			 ENC_PFX "transformer.%d.pre_norm_ffn.4.weight", i);
		L->fc2_w = gh_load_mmap(wf, name, arena,
					SAM3_DTYPE_F32, 2, dims);
		if (!L->fc2_w) return SAM3_ENOMEM;

		dims[0] = cfg->width;
		snprintf(name, sizeof(name),
			 ENC_PFX "transformer.%d.pre_norm_ffn.4.bias", i);
		L->fc2_b = gh_load_mmap(wf, name, arena,
					SAM3_DTYPE_F32, 1, dims);
		if (!L->fc2_b) return SAM3_ENOMEM;
	}

	sam3_log_info(
		"mobileclip: loaded %d-layer %s text encoder "
		"(%d RepMixer blocks)",
		cfg->n_layers,
		cfg->text_backbone == SAM3_TEXT_MOBILECLIP_S0 ? "S0" :
		cfg->text_backbone == SAM3_TEXT_MOBILECLIP_S1 ? "S1" : "L",
		cfg->n_repmixer_blocks);
	return SAM3_OK;
}

/*
 * build_std_block - One pre-norm transformer block, non-causal, no RoPE.
 *
 * Mirrors the reference pre_norm_mha + pre_norm_ffn layout.
 * QKV is pre-fused in the checkpoint (single weight [3*width, width]).
 * x must be [seq_len, width] on entry and exit (2D).
 */
static struct sam3_tensor *build_std_block(
	struct sam3_graph *g,
	const struct sam3_mobileclip_layer_std *L,
	struct sam3_tensor *x,
	int n_heads,
	int width,
	int seq_len,
	struct sam3_arena *arena)
{
	struct sam3_tensor *t, *x3d;
	int attn_dims[3];

	/* --- Attention sub-block --- */
	t = gh_layernorm(g, arena, x, L->ln1_w, L->ln1_b);
	if (!t)
		return NULL;

	/* gh_multihead_attention_rope expects [batch, seq, d_model] */
	attn_dims[0] = 1; attn_dims[1] = seq_len; attn_dims[2] = width;
	x3d = gh_reshape(g, arena, t, 3, attn_dims);
	if (!x3d)
		return NULL;

	/* Non-causal: attn_mask=NULL, no RoPE: rope_cos=rope_sin=NULL */
	t = gh_multihead_attention_rope(
		g, arena,
		x3d, NULL, NULL,
		L->qkv_w, L->qkv_b,
		L->out_w, L->out_b,
		n_heads,
		NULL, NULL,
		NULL, 0, 0.0f);
	if (!t)
		return NULL;

	/* attn output is [batch*seq, width] = [seq_len, width]; add residual */
	x = gh_add(g, arena, x, t);
	if (!x)
		return NULL;

	/* --- FFN sub-block --- */
	t = gh_layernorm(g, arena, x, L->ln2_w, L->ln2_b);
	if (!t)
		return NULL;

	t = gh_mlp(g, arena, t,
		   L->fc1_w, L->fc1_b,
		   L->fc2_w, L->fc2_b,
		   SAM3_OP_GELU);
	if (!t)
		return NULL;

	x = gh_add(g, arena, x, t);
	return x;
}

/*
 * sam3_mobileclip_text_build - Single-shot graph build for MobileCLIP S1/L.
 *
 * Builds the full compute graph: embedding + pos embed + transformer blocks
 * (std only; RepMixer errors until Phase 6) + final LN + projection.
 *
 * @enc:        Encoder with loaded weights
 * @g:          Graph to add nodes to
 * @token_ids:  [ctx_len] integer token indices
 * @pooled_out: If non-NULL, receives pooled [out_dim] EOT embedding
 * @arena:      Arena for intermediate tensors
 *
 * Returns per-token output [ctx_len, out_dim], or NULL on error.
 */
struct sam3_tensor *sam3_mobileclip_text_build(
	struct sam3_mobileclip_text_encoder *enc,
	struct sam3_graph *g,
	struct sam3_tensor *token_ids,
	struct sam3_tensor **pooled_out,
	struct sam3_arena *arena)
{
	const struct sam3_mobileclip_config *cfg = &enc->cfg;
	struct sam3_tensor *x, *pos, *out;
	int seq_len, i;

	seq_len = sam3_tensor_nelems(token_ids);

	/* 1. Token embedding lookup: [seq_len, width] */
	x = gh_embed(g, arena, enc->token_embedding, token_ids);
	if (!x)
		return NULL;

	/* 2. Positional embedding.
	 * pos_embed_full is [1, 1, 77, width] (4D). Reshape to [77, width]
	 * to make it 2D, then slice [0, seq_len) along axis 0.
	 */
	{
		int pos2d_dims[2];
		pos2d_dims[0] = cfg->pos_embed_table_len;
		pos2d_dims[1] = cfg->width;
		pos = gh_reshape(g, arena, enc->pos_embed_full,
				 2, pos2d_dims);
		if (!pos)
			return NULL;
	}
	if (seq_len < cfg->pos_embed_table_len) {
		pos = gh_slice(g, arena, pos, 0, 0, seq_len);
		if (!pos)
			return NULL;
	}
	x = gh_add(g, arena, x, pos);
	if (!x)
		return NULL;

	/* 3. Transformer blocks (no causal mask). RepMixer errors until
	 *    Phase 6 wires it. */
	for (i = 0; i < cfg->n_layers; i++) {
		const struct sam3_mobileclip_layer *Lslot = &enc->layers[i];
		if (Lslot->is_repmixer) {
			sam3_log_error(
				"mobileclip_build: RepMixer block %d "
				"reached before Phase 6 wired it", i);
			return NULL;
		}
		x = build_std_block(g, &Lslot->u.std, x,
				    cfg->n_heads, cfg->width,
				    seq_len, arena);
		if (!x)
			return NULL;
	}

	/* 4. Final layer norm */
	x = gh_layernorm(g, arena, x, enc->ln_final_w, enc->ln_final_b);
	if (!x)
		return NULL;

	/* 5. Inner projection (width -> width). projection_layer is a raw
	 *    [width, width] tensor used as a right-hand matmul factor.
	 *    x is [seq_len, width]; result is [seq_len, width]. */
	x = gh_matmul(g, arena, x, enc->projection_layer);
	if (!x)
		return NULL;

	/* 6. External 256-dim projector: out = x @ out_proj_w^T + out_proj_b
	 *    gh_linear computes input @ weight^T + bias. */
	out = gh_linear(g, arena, x, enc->out_proj_w, enc->out_proj_b);
	if (!out)
		return NULL;

	/* 7. Pooled output: last token (EOT slot = seq_len - 1).
	 *    Slice [seq_len-1, seq_len) -> [1, out_dim], reshape to [out_dim]. */
	if (pooled_out) {
		struct sam3_tensor *last_tok;
		int p_dims[1];

		last_tok = gh_slice(g, arena, out, 0,
				    seq_len - 1, seq_len);
		if (!last_tok)
			return NULL;

		p_dims[0] = cfg->out_dim;
		*pooled_out = gh_reshape(g, arena, last_tok,
					 1, p_dims);
		if (!*pooled_out)
			return NULL;
	}

	return out;
}

/*
 * sam3_mobileclip_text_build_perblock - Per-block evaluator for MobileCLIP.
 *
 * Builds the embedding once, then evaluates each transformer block in its own
 * graph, copying the result to persist and resetting scratch between blocks.
 * Mirrors the pattern used by sam3_text_encoder_build_perblock in text_encoder.c.
 *
 * @enc:      Encoder with loaded weights
 * @be:       Backend for graph evaluation
 * @token_ids: Input token IDs [seq_len] (I32 tensor)
 * @scratch:  Arena for per-block intermediate tensors (reset between blocks)
 * @persist:  Arena for output buffer that survives across blocks
 *
 * Returns per-token embeddings [seq_len, out_dim], or NULL on error.
 */
struct sam3_tensor *sam3_mobileclip_text_build_perblock(
	struct sam3_mobileclip_text_encoder *enc,
	struct sam3_backend *be,
	struct sam3_tensor *token_ids,
	struct sam3_arena *scratch,
	struct sam3_arena *persist)
{
	const struct sam3_mobileclip_config *cfg = &enc->cfg;
	struct sam3_graph g;
	enum sam3_error err;
	int seq_len, i;
	int x_dims[2];
	size_t x_bytes;
	void *x_buf;

	seq_len = sam3_tensor_nelems(token_ids);
	x_bytes = (size_t)seq_len * (size_t)cfg->width * sizeof(float);

	/* Allocate persist buffer for block outputs [seq_len, width] */
	x_buf = sam3_arena_alloc(persist, x_bytes);
	if (!x_buf)
		return NULL;

	/* === Group 0: token embed + pos embed === */
	{
		struct sam3_tensor *x, *pos;
		int pos2d_dims[2];

		sam3_arena_reset(scratch);
		sam3_graph_init(&g);

		x = gh_embed(&g, scratch, enc->token_embedding, token_ids);
		if (!x)
			return NULL;

		pos2d_dims[0] = cfg->pos_embed_table_len;
		pos2d_dims[1] = cfg->width;
		pos = gh_reshape(&g, scratch, enc->pos_embed_full,
				 2, pos2d_dims);
		if (!pos)
			return NULL;

		if (seq_len < cfg->pos_embed_table_len) {
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
	}

	/* === Block loop: one block per graph_eval === */
	x_dims[0] = seq_len;
	x_dims[1] = cfg->width;

	for (i = 0; i < cfg->n_layers; i++) {
		const struct sam3_mobileclip_layer *Lslot = &enc->layers[i];
		struct sam3_tensor *x;

		sam3_arena_reset(scratch);
		sam3_graph_init(&g);

		x = gh_tensor_wrap(scratch, SAM3_DTYPE_F32,
				   2, x_dims, x_buf);
		if (!x)
			return NULL;

		if (Lslot->is_repmixer) {
			sam3_log_error("mobileclip_perblock: RepMixer block %d "
				       "reached before Phase 6 wired it", i);
			return NULL;
		}

		x = build_std_block(&g, &Lslot->u.std, x,
				    cfg->n_heads, cfg->width,
				    seq_len, scratch);
		if (!x)
			return NULL;

		err = be->ops->graph_eval(be, &g);
		if (err != SAM3_OK)
			return NULL;

		memcpy(x_buf, x->data, x_bytes);
	}

	/* === Tail: final LN + inner projection + external projector === */
	{
		struct sam3_tensor *x, *out;
		int out_dims[2];

		sam3_arena_reset(scratch);
		sam3_graph_init(&g);

		x = gh_tensor_wrap(scratch, SAM3_DTYPE_F32,
				   2, x_dims, x_buf);
		if (!x)
			return NULL;

		x = gh_layernorm(&g, scratch, x,
				 enc->ln_final_w, enc->ln_final_b);
		if (!x)
			return NULL;

		x = gh_matmul(&g, scratch, x, enc->projection_layer);
		if (!x)
			return NULL;

		out = gh_linear(&g, scratch, x,
				enc->out_proj_w, enc->out_proj_b);
		if (!out)
			return NULL;

		err = be->ops->graph_eval(be, &g);
		if (err != SAM3_OK)
			return NULL;

		/* Copy result to persist arena */
		out_dims[0] = seq_len;
		out_dims[1] = cfg->out_dim;
		struct sam3_tensor *result;
		result = gh_alloc_tensor(persist, SAM3_DTYPE_F32,
					 2, out_dims);
		if (!result)
			return NULL;
		memcpy(result->data, out->data, result->nbytes);
		return result;
	}
}
