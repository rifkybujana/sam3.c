/*
 * src/model/text_encoder_iface.h - Text encoder vtable + factory
 *
 * Wraps either the existing CLIP encoder (sam3_text_encoder) or the new
 * MobileCLIP encoder (sam3_mobileclip_text_encoder) behind a single
 * interface so the rest of the engine (vl_combiner, sam3_processor)
 * dispatches without switching on backbone. Selected at load time by
 * the .sam3 v4 header's text_backbone field.
 *
 * Key types:  sam3_text_encoder_iface, sam3_text_encoder_iface_ops
 * Depends on: sam3/sam3_types.h, core/tensor.h, core/graph.h, core/alloc.h,
 *             core/weight.h, backend/backend.h
 * Used by:    src/model/vl_combiner.c, src/model/sam3_processor.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_TEXT_ENCODER_IFACE_H
#define SAM3_MODEL_TEXT_ENCODER_IFACE_H

#include "sam3/sam3_types.h"
#include "core/tensor.h"
#include "core/graph.h"
#include "core/alloc.h"
#include "core/weight.h"
#include "backend/backend.h"

struct sam3_text_encoder_iface;

struct sam3_text_encoder_iface_ops {
	enum sam3_error (*load)(struct sam3_text_encoder_iface *iface,
				const struct sam3_weight_file *wf,
				struct sam3_arena *arena);
	struct sam3_tensor *(*build)(struct sam3_text_encoder_iface *iface,
				     struct sam3_graph *g,
				     struct sam3_tensor *token_ids,
				     struct sam3_tensor **pooled_out,
				     struct sam3_arena *arena);
	struct sam3_tensor *(*build_perblock)(
		struct sam3_text_encoder_iface *iface,
		struct sam3_backend *be,
		struct sam3_tensor *token_ids,
		struct sam3_arena *scratch,
		struct sam3_arena *persist);
	void (*free)(struct sam3_text_encoder_iface *iface);
};

struct sam3_text_encoder_iface {
	const struct sam3_text_encoder_iface_ops *ops;
	void *impl;          /* concrete encoder pointer */
	int   text_backbone; /* enum sam3_text_backbone */
	int   ctx_len;       /* sequence length the encoder expects */
	int   d_model;       /* output embedding dimension (always 256 today) */
};

/*
 * sam3_text_encoder_iface_init - Construct an iface for the given backbone.
 *
 * @iface:         Caller-allocated iface struct (zeroed)
 * @text_backbone: enum sam3_text_backbone (CLIP or one of the MOBILECLIP_*)
 * @arena:         Arena for the concrete encoder struct + its precomputed data
 *
 * For SAM3_TEXT_CLIP, allocates a sam3_text_encoder with the historical
 * 24L/1024w/16h/ctx=32 config used today. For SAM3_TEXT_MOBILECLIP_*,
 * allocates a sam3_mobileclip_text_encoder configured per the variant
 * table in mobileclip_text.c.
 *
 * Returns SAM3_OK on success; SAM3_EINVAL on unknown backbone;
 * SAM3_ENOMEM on arena exhaustion.
 */
enum sam3_error sam3_text_encoder_iface_init(
	struct sam3_text_encoder_iface *iface,
	int text_backbone,
	struct sam3_arena *arena);

#endif /* SAM3_MODEL_TEXT_ENCODER_IFACE_H */
