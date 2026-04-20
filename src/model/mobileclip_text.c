/*
 * src/model/mobileclip_text.c - MobileCLIP text encoder implementation
 *
 * Skeleton file: variant config table + iface vtable ops + stubs for
 * load/build that are filled in over Phases 4-6. Each variant maps to
 * a static const sam3_mobileclip_config; the iface init allocates a
 * fresh sam3_mobileclip_text_encoder, copies the config, and hands
 * back via the iface vtable.
 *
 * Key types:  sam3_mobileclip_text_encoder
 * Depends on: mobileclip_text.h, text_encoder_iface.h, util/log.h
 * Used by:    src/model/text_encoder_iface.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <string.h>

#include "mobileclip_text.h"
#include "text_encoder_iface.h"
#include "util/log.h"

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

/* --- Stubs for load/build (filled in by Tasks 4.3-4.5, 6.x) --- */

enum sam3_error sam3_mobileclip_text_load(
	struct sam3_mobileclip_text_encoder *enc,
	const struct sam3_weight_file *wf,
	struct sam3_arena *arena)
{
	(void)enc; (void)wf; (void)arena;
	sam3_log_error("mobileclip_text_load: not yet implemented");
	return SAM3_EINVAL;
}

struct sam3_tensor *sam3_mobileclip_text_build(
	struct sam3_mobileclip_text_encoder *enc,
	struct sam3_graph *g,
	struct sam3_tensor *token_ids,
	struct sam3_tensor **pooled_out,
	struct sam3_arena *arena)
{
	(void)enc; (void)g; (void)token_ids; (void)pooled_out; (void)arena;
	sam3_log_error("mobileclip_text_build: not yet implemented");
	return NULL;
}

struct sam3_tensor *sam3_mobileclip_text_build_perblock(
	struct sam3_mobileclip_text_encoder *enc,
	struct sam3_backend *be,
	struct sam3_tensor *token_ids,
	struct sam3_arena *scratch,
	struct sam3_arena *persist)
{
	(void)enc; (void)be; (void)token_ids; (void)scratch; (void)persist;
	sam3_log_error("mobileclip_text_build_perblock: not yet implemented");
	return NULL;
}
