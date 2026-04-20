/*
 * src/model/text_encoder_iface.c - Text encoder vtable wiring
 *
 * Concrete vtable implementations: one set of ops wraps the historical
 * sam3_text_encoder (CLIP); a second set wraps sam3_mobileclip_text_encoder
 * (the three MobileCLIP variants). The factory picks the right pair based
 * on the text_backbone enum and stashes it on the iface.
 *
 * Key types:  sam3_text_encoder_iface
 * Depends on: text_encoder_iface.h, text_encoder.h, core/alloc.h, util/log.h
 * Used by:    src/model/vl_combiner.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include <string.h>

#include "text_encoder_iface.h"
#include "text_encoder.h"
#include "core/alloc.h"
#include "util/log.h"

/* --- CLIP wrapper ops --- */

static enum sam3_error clip_load(struct sam3_text_encoder_iface *iface,
				 const struct sam3_weight_file *wf,
				 struct sam3_arena *arena)
{
	return sam3_text_encoder_load(
		(struct sam3_text_encoder *)iface->impl, wf, arena);
}

static struct sam3_tensor *clip_build(struct sam3_text_encoder_iface *iface,
				      struct sam3_graph *g,
				      struct sam3_tensor *token_ids,
				      struct sam3_tensor **pooled_out,
				      struct sam3_arena *arena)
{
	return sam3_text_encoder_build(
		(struct sam3_text_encoder *)iface->impl, g, token_ids,
		pooled_out, arena);
}

static struct sam3_tensor *clip_build_perblock(
	struct sam3_text_encoder_iface *iface,
	struct sam3_backend *be,
	struct sam3_tensor *token_ids,
	struct sam3_arena *scratch,
	struct sam3_arena *persist)
{
	return sam3_text_encoder_build_perblock(
		(struct sam3_text_encoder *)iface->impl, be, token_ids,
		scratch, persist);
}

static void clip_free(struct sam3_text_encoder_iface *iface)
{
	(void)iface; /* arena-backed; nothing to free */
}

static const struct sam3_text_encoder_iface_ops clip_ops = {
	.load           = clip_load,
	.build          = clip_build,
	.build_perblock = clip_build_perblock,
	.free           = clip_free,
};

/* --- MobileCLIP wrapper ops --- forward decl used after Phase 4 lands --- */

extern enum sam3_error sam3_mobileclip_text_iface_init_impl(
	struct sam3_text_encoder_iface *iface,
	int text_backbone, struct sam3_arena *arena);

/*
 * Temporary stub: removed in Phase 4 once mobileclip_text.c lands.
 * The CLIP path doesn't reach this; MobileCLIP variants will fail
 * until Phase 4.
 */
__attribute__((weak))
enum sam3_error sam3_mobileclip_text_iface_init_impl(
	struct sam3_text_encoder_iface *iface,
	int text_backbone, struct sam3_arena *arena)
{
	(void)iface; (void)text_backbone; (void)arena;
	sam3_log_error("text_iface: MobileCLIP not yet implemented");
	return SAM3_EINVAL;
}

/* --- Factory --- */

enum sam3_error sam3_text_encoder_iface_init(
	struct sam3_text_encoder_iface *iface,
	int text_backbone,
	struct sam3_arena *arena)
{
	if (!iface || !arena)
		return SAM3_EINVAL;

	memset(iface, 0, sizeof(*iface));
	iface->text_backbone = text_backbone;

	switch (text_backbone) {
	case SAM3_TEXT_CLIP: {
		struct sam3_text_encoder *te;

		te = sam3_arena_alloc(arena, sizeof(*te));
		if (!te) {
			sam3_log_error("text_iface: arena alloc failed (CLIP)");
			return SAM3_ENOMEM;
		}
		memset(te, 0, sizeof(*te));
		te->d_model     = 256;
		te->width       = 1024;
		te->n_heads     = 16;
		te->n_layers    = 24;
		te->context_len = 32;   /* matches existing vl_combiner config */
		te->vocab_size  = 49408;

		iface->impl    = te;
		iface->ops     = &clip_ops;
		iface->ctx_len = te->context_len;
		iface->d_model = te->d_model;
		return SAM3_OK;
	}
	case SAM3_TEXT_MOBILECLIP_S0:
	case SAM3_TEXT_MOBILECLIP_S1:
	case SAM3_TEXT_MOBILECLIP_L:
		return sam3_mobileclip_text_iface_init_impl(
			iface, text_backbone, arena);
	default:
		sam3_log_error("text_iface: unknown text_backbone %d",
			       text_backbone);
		return SAM3_EINVAL;
	}
}
