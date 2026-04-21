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

/* Weight key prefixes (match the key namespace written by cli_convert). */
#define ENC_PFX      "detector_model.text_encoder.encoder."
#define BACKBONE_PFX "detector_model.text_encoder."

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

/* --- load_repmixer_block --- */

/*
 * load_repmixer_block - Populate one repmixer layer slot.
 *
 * @R:           Repmixer struct to fill (zeroed by caller).
 * @wf:          Open weight file.
 * @arena:       Arena for tensor descriptors.
 * @layer_idx:   The transformer.<layer_idx> index in the source state-dict.
 * @width:       cfg.width (channel count, 512 for S0).
 * @mlp_dim:     cfg.mlp_dim (2048 for S0).
 *
 * BN params are required by the audited S0 checkpoint (no fold). All four
 * (weight, bias, running_mean, running_var) per BN must load successfully;
 * a missing key is a model-format error. The optional rbr_scale branch
 * tolerates absence — when NULL, build skips the branch.
 */
static enum sam3_error load_repmixer_block(
	struct sam3_mobileclip_layer_repmixer *R,
	const struct sam3_weight_file *wf,
	struct sam3_arena *arena,
	int layer_idx, int width, int mlp_dim)
{
	char k[256];
	int dims[4];

#define K(suffix) \
	(snprintf(k, sizeof(k), ENC_PFX "transformer.%d." suffix, \
		  layer_idx), k)

	/* Outer + token-mixer scales: shape [width, 1, 1] */
	dims[0] = width; dims[1] = 1; dims[2] = 1;
	R->outer_layer_scale = gh_load_mmap(wf, K("layer_scale"),
					    arena, SAM3_DTYPE_F32, 3, dims);
	R->tm_layer_scale    = gh_load_mmap(wf, K("token_mixer.layer_scale"),
					    arena, SAM3_DTYPE_F32, 3, dims);

	/* token_mixer.norm.rbr_skip — single BN, shape [width] each. */
	dims[0] = width;
	R->norm_skip_w  = gh_load_mmap(
		wf, K("token_mixer.norm.rbr_skip.weight"),
		arena, SAM3_DTYPE_F32, 1, dims);
	R->norm_skip_b  = gh_load_mmap(
		wf, K("token_mixer.norm.rbr_skip.bias"),
		arena, SAM3_DTYPE_F32, 1, dims);
	R->norm_skip_rm = gh_load_mmap(
		wf, K("token_mixer.norm.rbr_skip.running_mean"),
		arena, SAM3_DTYPE_F32, 1, dims);
	R->norm_skip_rv = gh_load_mmap(
		wf, K("token_mixer.norm.rbr_skip.running_var"),
		arena, SAM3_DTYPE_F32, 1, dims);

	/* token_mixer.mixer.rbr_skip — BN-only branch. */
	R->mixer_skip_w  = gh_load_mmap(
		wf, K("token_mixer.mixer.rbr_skip.weight"),
		arena, SAM3_DTYPE_F32, 1, dims);
	R->mixer_skip_b  = gh_load_mmap(
		wf, K("token_mixer.mixer.rbr_skip.bias"),
		arena, SAM3_DTYPE_F32, 1, dims);
	R->mixer_skip_rm = gh_load_mmap(
		wf, K("token_mixer.mixer.rbr_skip.running_mean"),
		arena, SAM3_DTYPE_F32, 1, dims);
	R->mixer_skip_rv = gh_load_mmap(
		wf, K("token_mixer.mixer.rbr_skip.running_var"),
		arena, SAM3_DTYPE_F32, 1, dims);

	/* token_mixer.mixer.rbr_conv[0] — depthwise conv 1×11 + BN.
	 * Weight stored as OHWI [width, 1, 11, 1] after converter permutation. */
	dims[0] = width; dims[1] = 1; dims[2] = 11; dims[3] = 1;
	R->mixer_conv_w = gh_load_mmap(
		wf, K("token_mixer.mixer.rbr_conv.0.conv.weight"),
		arena, SAM3_DTYPE_F32, 4, dims);
	dims[0] = width;
	R->mixer_conv_bn_w  = gh_load_mmap(
		wf, K("token_mixer.mixer.rbr_conv.0.bn.weight"),
		arena, SAM3_DTYPE_F32, 1, dims);
	R->mixer_conv_bn_b  = gh_load_mmap(
		wf, K("token_mixer.mixer.rbr_conv.0.bn.bias"),
		arena, SAM3_DTYPE_F32, 1, dims);
	R->mixer_conv_bn_rm = gh_load_mmap(
		wf, K("token_mixer.mixer.rbr_conv.0.bn.running_mean"),
		arena, SAM3_DTYPE_F32, 1, dims);
	R->mixer_conv_bn_rv = gh_load_mmap(
		wf, K("token_mixer.mixer.rbr_conv.0.bn.running_var"),
		arena, SAM3_DTYPE_F32, 1, dims);

	/* (Optional) token_mixer.mixer.rbr_scale — 1×1 conv + BN.
	 * Weight stored as OHWI [width, 1, 1, width] after converter permutation.
	 * NULL on miss is silent; build skips this branch when NULL. */
	dims[0] = width; dims[1] = 1; dims[2] = 1; dims[3] = width;
	R->mixer_scale_w = gh_load_mmap_optional(
		wf, K("token_mixer.mixer.rbr_scale.conv.weight"),
		arena, SAM3_DTYPE_F32, 4, dims);
	dims[0] = width;
	R->mixer_scale_bn_w  = gh_load_mmap_optional(
		wf, K("token_mixer.mixer.rbr_scale.bn.weight"),
		arena, SAM3_DTYPE_F32, 1, dims);
	R->mixer_scale_bn_b  = gh_load_mmap_optional(
		wf, K("token_mixer.mixer.rbr_scale.bn.bias"),
		arena, SAM3_DTYPE_F32, 1, dims);
	R->mixer_scale_bn_rm = gh_load_mmap_optional(
		wf, K("token_mixer.mixer.rbr_scale.bn.running_mean"),
		arena, SAM3_DTYPE_F32, 1, dims);
	R->mixer_scale_bn_rv = gh_load_mmap_optional(
		wf, K("token_mixer.mixer.rbr_scale.bn.running_var"),
		arena, SAM3_DTYPE_F32, 1, dims);

	/* convffn.conv — depthwise 1×11 + BN.
	 * Weight stored as OHWI [width, 1, 11, 1] after converter permutation. */
	dims[0] = width; dims[1] = 1; dims[2] = 11; dims[3] = 1;
	R->convffn_dw_w = gh_load_mmap(
		wf, K("convffn.conv.conv.weight"),
		arena, SAM3_DTYPE_F32, 4, dims);
	dims[0] = width;
	R->convffn_dw_bn_w  = gh_load_mmap(
		wf, K("convffn.conv.bn.weight"),
		arena, SAM3_DTYPE_F32, 1, dims);
	R->convffn_dw_bn_b  = gh_load_mmap(
		wf, K("convffn.conv.bn.bias"),
		arena, SAM3_DTYPE_F32, 1, dims);
	R->convffn_dw_bn_rm = gh_load_mmap(
		wf, K("convffn.conv.bn.running_mean"),
		arena, SAM3_DTYPE_F32, 1, dims);
	R->convffn_dw_bn_rv = gh_load_mmap(
		wf, K("convffn.conv.bn.running_var"),
		arena, SAM3_DTYPE_F32, 1, dims);

	/* convffn.fc1 / convffn.fc2 — 1×1 convs with bias.
	 * 1×1 conv weights have identical memory layout in OIHW and OHWI,
	 * so no converter permutation is needed. Declare as OHWI
	 * [OC, 1, 1, IC] so gh_conv2d reads KH=1, KW=1, C_in correctly. */
	dims[0] = mlp_dim; dims[1] = 1; dims[2] = 1; dims[3] = width;
	R->convffn_fc1_w = gh_load_mmap(
		wf, K("convffn.fc1.weight"),
		arena, SAM3_DTYPE_F32, 4, dims);
	dims[0] = mlp_dim;
	R->convffn_fc1_b = gh_load_mmap(
		wf, K("convffn.fc1.bias"),
		arena, SAM3_DTYPE_F32, 1, dims);

	dims[0] = width; dims[1] = 1; dims[2] = 1; dims[3] = mlp_dim;
	R->convffn_fc2_w = gh_load_mmap(
		wf, K("convffn.fc2.weight"),
		arena, SAM3_DTYPE_F32, 4, dims);
	dims[0] = width;
	R->convffn_fc2_b = gh_load_mmap(
		wf, K("convffn.fc2.bias"),
		arena, SAM3_DTYPE_F32, 1, dims);

#undef K

	if (!R->mixer_conv_w || !R->convffn_dw_w ||
	    !R->convffn_fc1_w || !R->convffn_fc2_w) {
		sam3_log_error("mobileclip: RepMixer block %d incomplete",
			       layer_idx);
		return SAM3_EMODEL;
	}

	sam3_log_info("mobileclip_s0: RepMixer block %d loaded%s",
		      layer_idx,
		      R->mixer_scale_w ? " (rbr_scale present)" : "");
	return SAM3_OK;
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
			enum sam3_error e = load_repmixer_block(
				&enc->layers[i].u.repmixer, wf, arena,
				i, cfg->width, cfg->mlp_dim);
			if (e != SAM3_OK)
				return e;
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
 * build_repmixer_block - One RepMixer block (S0 indices 0 and 5).
 *
 * Caller provides x as [seq_len, width] (2D). The function reshapes to
 * NHWC [1, 1, seq_len, width] for conv/BN ops and reshapes back to
 * [seq_len, width] before returning.
 *
 * Token-mixer residual: x + tm_layer_scale * (mixer_out - norm_out)
 * (SUBTRACTION, not addition). eps=1e-5 for every BN (hardcoded in op).
 *
 * @R:     RepMixer weights for this block
 * @width: cfg->width (channel count = 512 for S0)
 */
static struct sam3_tensor *build_repmixer_block(
	struct sam3_graph *g,
	const struct sam3_mobileclip_layer_repmixer *R,
	struct sam3_tensor *x,
	int width,
	int seq_len,
	struct sam3_arena *arena)
{
	struct sam3_tensor *norm_out, *y_skip, *y_conv, *mixer_out, *t;
	struct sam3_tensor *y, *scale4d;
	int nhwc[4], seq2[2];

	/* Reshape [seq_len, width] -> NHWC [1, 1, seq_len, width] for conv/BN. */
	nhwc[0] = 1; nhwc[1] = 1; nhwc[2] = seq_len; nhwc[3] = width;
	x = gh_reshape(g, arena, x, 4, nhwc);
	if (!x)
		return NULL;

	/* --- token_mixer.norm — single BN on the input. --- */
	norm_out = gh_batchnorm(g, arena, x,
				R->norm_skip_w, R->norm_skip_b,
				R->norm_skip_rm, R->norm_skip_rv);
	if (!norm_out)
		return NULL;

	/* --- token_mixer.mixer.rbr_skip — BN-only branch. --- */
	y_skip = gh_batchnorm(g, arena, x,
			      R->mixer_skip_w, R->mixer_skip_b,
			      R->mixer_skip_rm, R->mixer_skip_rv);
	if (!y_skip)
		return NULL;

	/* --- token_mixer.mixer.rbr_conv[0] — depthwise 1×11 conv + BN. ---
	 * Weight is OHWI [C, 1, 11, 1], pad_h=0 pad_w=5 to preserve seq_len. */
	t = gh_conv2d_hw(g, arena, x, R->mixer_conv_w, /*bias*/ NULL,
			 /*stride*/ 1, /*pad_h*/ 0, /*pad_w*/ 5,
			 /*groups*/ width);
	if (!t)
		return NULL;
	y_conv = gh_batchnorm(g, arena, t,
			      R->mixer_conv_bn_w, R->mixer_conv_bn_b,
			      R->mixer_conv_bn_rm, R->mixer_conv_bn_rv);
	if (!y_conv)
		return NULL;

	mixer_out = gh_add(g, arena, y_skip, y_conv);
	if (!mixer_out)
		return NULL;

	/* --- Optional rbr_scale branch — 1×1 pointwise conv + BN. --- */
	if (R->mixer_scale_w) {
		struct sam3_tensor *y_scale;

		t = gh_conv2d_hw(g, arena, x, R->mixer_scale_w, /*bias*/ NULL,
				 1, 0, 0, /*groups*/ 1);
		if (!t)
			return NULL;
		y_scale = gh_batchnorm(g, arena, t,
				       R->mixer_scale_bn_w,
				       R->mixer_scale_bn_b,
				       R->mixer_scale_bn_rm,
				       R->mixer_scale_bn_rv);
		if (!y_scale)
			return NULL;
		mixer_out = gh_add(g, arena, mixer_out, y_scale);
		if (!mixer_out)
			return NULL;
	}

	/* --- Token-mixer residual: x + tm_layer_scale * (mixer_out - norm_out). ---
	 * tm_layer_scale is [C,1,1]; reshape to [1,1,1,C] for NHWC broadcast. */
	t = gh_sub(g, arena, mixer_out, norm_out);
	if (!t)
		return NULL;

	nhwc[0] = 1; nhwc[1] = 1; nhwc[2] = 1; nhwc[3] = width;
	scale4d = gh_reshape(g, arena, R->tm_layer_scale, 4, nhwc);
	if (!scale4d)
		return NULL;

	t = gh_mul(g, arena, t, scale4d);
	if (!t)
		return NULL;

	x = gh_add(g, arena, x, t);
	if (!x)
		return NULL;

	/* --- ConvFFN: depthwise 1×11 + BN -> 1×1 fc1 -> GELU -> 1×1 fc2. --- */
	y = gh_conv2d_hw(g, arena, x, R->convffn_dw_w, /*bias*/ NULL,
			 1, 0, 5, /*groups*/ width);
	if (!y)
		return NULL;
	y = gh_batchnorm(g, arena, y,
			 R->convffn_dw_bn_w, R->convffn_dw_bn_b,
			 R->convffn_dw_bn_rm, R->convffn_dw_bn_rv);
	if (!y)
		return NULL;

	y = gh_conv2d_hw(g, arena, y, R->convffn_fc1_w, R->convffn_fc1_b,
			 1, 0, 0, /*groups*/ 1);
	if (!y)
		return NULL;

	y = gh_gelu(g, arena, y);
	if (!y)
		return NULL;

	y = gh_conv2d_hw(g, arena, y, R->convffn_fc2_w, R->convffn_fc2_b,
			 1, 0, 0, /*groups*/ 1);
	if (!y)
		return NULL;

	/* Outer block residual: x + outer_layer_scale * y.
	 * outer_layer_scale is [C,1,1]; reshape to [1,1,1,C] for NHWC broadcast. */
	nhwc[0] = 1; nhwc[1] = 1; nhwc[2] = 1; nhwc[3] = width;
	scale4d = gh_reshape(g, arena, R->outer_layer_scale, 4, nhwc);
	if (!scale4d)
		return NULL;

	y = gh_mul(g, arena, y, scale4d);
	if (!y)
		return NULL;

	x = gh_add(g, arena, x, y);
	if (!x)
		return NULL;

	/* Reshape NHWC [1, 1, seq_len, width] back to [seq_len, width]. */
	seq2[0] = seq_len; seq2[1] = width;
	return gh_reshape(g, arena, x, 2, seq2);
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

	/* 3. Transformer blocks (no causal mask). */
	for (i = 0; i < cfg->n_layers; i++) {
		const struct sam3_mobileclip_layer *Lslot = &enc->layers[i];
		if (Lslot->is_repmixer) {
			x = build_repmixer_block(g, &Lslot->u.repmixer, x,
						 cfg->width, seq_len, arena);
		} else {
			x = build_std_block(g, &Lslot->u.std, x,
					    cfg->n_heads, cfg->width,
					    seq_len, arena);
		}
		if (!x)
			return NULL;
	}

	/* 4. Final layer norm */
	x = gh_layernorm(g, arena, x, enc->ln_final_w, enc->ln_final_b);
	if (!x)
		return NULL;

	/* 5. External 256-dim projector: out = x @ out_proj_w^T + out_proj_b
	 *    gh_linear computes input @ weight^T + bias.
	 *    Note: projection_layer (width->width inner matmul) is intentionally
	 *    skipped here. It is only applied in the pooled single-token path
	 *    and is all-zeros in the MobileCLIP checkpoints when return_all_tokens
	 *    is True. */
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
			x = build_repmixer_block(&g, &Lslot->u.repmixer, x,
						 cfg->width, seq_len, scratch);
		} else {
			x = build_std_block(&g, &Lslot->u.std, x,
					    cfg->n_heads, cfg->width,
					    seq_len, scratch);
		}
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

		/* External 256-dim projector: out = x @ out_proj_w^T + out_proj_b.
		 * projection_layer (inner width->width matmul) is intentionally
		 * skipped — it is all-zeros in MobileCLIP checkpoints and is only
		 * applied in the pooled single-token path (return_all_tokens=False). */
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
