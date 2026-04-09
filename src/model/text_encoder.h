/*
 * src/model/text_encoder.h - CLIP text encoder (24-layer transformer)
 *
 * Defines a transformer-based text encoder that produces per-token
 * embeddings and pooled text features from token IDs. The architecture
 * follows CLIP's text encoder: token + positional embedding, N
 * transformer blocks with pre-norm self-attention, and a final layer
 * norm plus projection to produce the pooled text feature.
 *
 * Key types:  sam3_text_encoder
 * Depends on: core/tensor.h, core/graph.h, core/alloc.h, core/weight.h
 * Used by:    sam3.c, tests/test_text_encoder.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_TEXT_ENCODER_H
#define SAM3_MODEL_TEXT_ENCODER_H

#include "core/tensor.h"
#include "core/graph.h"
#include "core/alloc.h"
#include "core/weight.h"
#include "backend/backend.h"

#define SAM3_TEXT_ENC_MAX_LAYERS 24

struct sam3_text_encoder {
	int d_model;		/* 256 -- output dimension */
	int width;		    /* 1024 -- internal transformer width */
	int n_heads;		/* 16 */
	int n_layers;		/* 24 */
	int context_len;	/* 77 */
	int vocab_size;		/* 49408 */

	/* Embeddings */
	struct sam3_tensor *token_embedding;	/* [vocab_size, width] */
	struct sam3_tensor *pos_embedding;	/* [context_len, width] */

	/* Final layer norm + projection */
	struct sam3_tensor *ln_final_w;		/* [width] */
	struct sam3_tensor *ln_final_b;		/* [width] */
	struct sam3_tensor *text_projection;	/* [d_model, width] */
	struct sam3_tensor *text_projection_b;	/* [d_model] */

	/* Per-layer weights */
	struct {
		struct sam3_tensor *ln1_w, *ln1_b;	/* [width] */
		struct sam3_tensor *attn_qkv_w;		/* [3*width, width] */
		struct sam3_tensor *attn_qkv_b;		/* [3*width] */
		struct sam3_tensor *attn_out_w;		/* [width, width] */
		struct sam3_tensor *attn_out_b;		/* [width] */
		struct sam3_tensor *ln2_w, *ln2_b;	/* [width] */
		struct sam3_tensor *mlp_fc1_w;		/* [width*4, width] */
		struct sam3_tensor *mlp_fc1_b;		/* [width*4] */
		struct sam3_tensor *mlp_fc2_w;		/* [width, width*4] */
		struct sam3_tensor *mlp_fc2_b;		/* [width] */
	} layers[SAM3_TEXT_ENC_MAX_LAYERS];
};

/*
 * sam3_text_encoder_load - Load text encoder weights from weight file.
 *
 * @te:    Text encoder struct (caller-allocated)
 * @wf:    Open weight file (may be NULL for zero-init fallback)
 * @arena: Arena for weight tensor allocation
 *
 * Looks up weight tensors by name and populates the struct.  When a
 * weight is not found (or wf is NULL), a zero-initialized tensor of
 * the correct shape is allocated as a fallback so the graph can still
 * be built and evaluated for shape testing.
 *
 * Returns SAM3_OK on success, SAM3_ENOMEM if the arena is full.
 */
enum sam3_error sam3_text_encoder_load(struct sam3_text_encoder *te,
				       const struct sam3_weight_file *wf,
				       struct sam3_arena *arena);

/*
 * sam3_text_encoder_build - Build text encoder compute graph.
 *
 * @te:         Loaded text encoder
 * @g:          Graph to add nodes to
 * @token_ids:  Input token IDs [seq_len] (I32 tensor)
 * @pooled_out: If non-NULL, receives pooled text feature [d_model]
 * @arena:      Arena for intermediate tensor allocation
 *
 * Returns per-token embeddings [seq_len, d_model], or NULL on error.
 * The pooled output is extracted from the last token position (EOT)
 * and projected through text_projection to produce a [d_model] vector.
 *
 * Note: causal attention masking is not yet implemented. For inference
 * with short prompts the outputs are still reasonable.
 */
struct sam3_tensor *sam3_text_encoder_build(
	struct sam3_text_encoder *te,
	struct sam3_graph *g,
	struct sam3_tensor *token_ids,
	struct sam3_tensor **pooled_out,
	struct sam3_arena *arena);

/*
 * sam3_text_encoder_build_perblock - Per-block text encoder evaluation.
 *
 * Evaluates one block at a time, dumping /tmp/dbg_te_block_XX.bin
 * for fixture comparison. Used for debugging text encoder divergence.
 */
struct sam3_tensor *sam3_text_encoder_build_perblock(
	struct sam3_text_encoder *te,
	struct sam3_backend *be,
	struct sam3_tensor *token_ids,
	struct sam3_arena *scratch,
	struct sam3_arena *persist);

#endif /* SAM3_MODEL_TEXT_ENCODER_H */
