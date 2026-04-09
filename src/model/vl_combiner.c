/*
 * src/model/vl_combiner.c - Vision-language backbone combiner
 *
 * Implements the composite VL backbone that wires together the ViT
 * image encoder, feature pyramid neck, CLIP text encoder, BPE
 * tokenizer, and 2D position encoding. This module does not add new
 * computation -- it organizes the sub-modules and provides unified
 * init/load/build entry points for the SAM3 inference pipeline.
 *
 * Key types:  sam3_vl_backbone
 * Depends on: vl_combiner.h, graph_helpers.h
 * Used by:    sam3.c (top-level image model)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <string.h>

#include "vl_combiner.h"
#include "graph_helpers.h"
#include "util/log.h"

enum sam3_error sam3_vl_backbone_init(struct sam3_vl_backbone *vl,
				      struct sam3_arena *arena)
{
	enum sam3_error err;

	vl->scalp = 1;

	/* Init ViT with SAM3 defaults */
	err = sam3_vit_init(&vl->vit,
			     1008,	/* img_size */
			     14,	/* patch_size */
			     1024,	/* embed_dim */
			     32,	/* depth */
			     16,	/* n_heads */
			     24,	/* window_size */
			     4736,	/* mlp_dim = 1024 * 4.625 */
			     arena);
	if (err != SAM3_OK)
		return err;

	/* Init neck: 4 scales at {4x, 2x, 1x, 0.5x} */
	float scales[] = {4.0f, 2.0f, 1.0f, 0.5f};
	err = sam3_neck_init(&vl->neck, 256, 1024,
			      vl->vit.grid_size, 4, scales);
	if (err != SAM3_OK)
		return err;

	/* Init tokenizer (byte-level fallback vocab) */
	err = sam3_tokenizer_init(&vl->tokenizer);
	if (err != SAM3_OK)
		return err;

	/* Init text encoder config (weights loaded separately) */
	vl->text_enc.d_model = 256;
	vl->text_enc.width = 1024;
	vl->text_enc.n_heads = 16;
	vl->text_enc.n_layers = 24;
	vl->text_enc.context_len = 32;
	vl->text_enc.vocab_size = 49408;

	/*
	 * Lazy 2D sinusoidal position encoding for ViT grid.
	 * Python: num_pos_feats = input // 2 = 256 // 2 = 128.
	 * Output is [H, W, 128*2] = [72, 72, 256] matching d_model.
	 * Actual computation deferred to first sam3_pos_encoding_get().
	 */
	vl->pos_enc.height = vl->vit.grid_size;
	vl->pos_enc.width = vl->vit.grid_size;
	vl->pos_enc.num_pos_feats = 128;
	vl->pos_enc.temperature = 10000.0f;
	vl->pos_enc.arena = arena;

	return SAM3_OK;
}

enum sam3_error sam3_vl_backbone_load(struct sam3_vl_backbone *vl,
				      const struct sam3_weight_file *wf,
				      struct sam3_arena *arena)
{
	enum sam3_error err;

	err = sam3_vit_load(&vl->vit, wf, arena);
	if (err != SAM3_OK)
		return err;

	err = sam3_neck_load(&vl->neck, wf, arena);
	if (err != SAM3_OK)
		return err;

	err = sam3_text_encoder_load(&vl->text_enc, wf, arena);
	if (err != SAM3_OK)
		return err;

	return SAM3_OK;
}

void sam3_vl_backbone_free(struct sam3_vl_backbone *vl)
{
	if (vl)
		sam3_tokenizer_free(&vl->tokenizer);
}

struct sam3_tensor *sam3_vl_backbone_build_vision(
	struct sam3_vl_backbone *vl,
	struct sam3_graph *g,
	struct sam3_backend *be,
	struct sam3_tensor *image,
	struct sam3_tensor *out_features[],
	struct sam3_arena *scratch,
	struct sam3_arena *persist)
{
	/*
	 * Run ViT per-block: evaluates internally, returns
	 * materialized output in persist arena.
	 */
	struct sam3_tensor *vit_out;

	vit_out = sam3_vit_build(&vl->vit, be, image, scratch, persist);
	if (!vit_out) {
		sam3_log_error("vl_backbone: vit_build returned NULL");
		return NULL;
	}

	sam3_log_debug("vl_backbone: vit done, scratch %zu/%zu, persist %zu/%zu",
		       scratch->offset, scratch->size,
		       persist->offset, persist->size);

	/*
	 * Reset scratch and build neck graph. The caller will
	 * evaluate this graph after we return.
	 */
	sam3_arena_reset(scratch);
	sam3_graph_init(g);

	enum sam3_error err;

	err = sam3_neck_build(&vl->neck, g, vit_out, out_features, scratch);
	if (err != SAM3_OK) {
		sam3_log_error("vl_backbone: neck_build failed, scratch %zu/%zu",
			       scratch->offset, scratch->size);
		return NULL;
	}

	sam3_log_debug("vl_backbone: neck built, scratch %zu/%zu, %d nodes",
		       scratch->offset, scratch->size, g->n_nodes);

	return vit_out;
}

struct sam3_tensor *sam3_vl_backbone_build_text(
	struct sam3_vl_backbone *vl,
	struct sam3_graph *g,
	const char *text,
	struct sam3_tensor **pooled_out,
	struct sam3_arena *arena)
{
	int32_t tokens[SAM3_TOKENIZER_CONTEXT_LEN];
	int n_tokens;
	struct sam3_tensor *tok_tensor;
	int dims[1];

	/* Tokenize the input text */
	n_tokens = sam3_tokenizer_encode(&vl->tokenizer, text,
					  tokens, SAM3_TOKENIZER_CONTEXT_LEN);
	if (n_tokens <= 0)
		return NULL;

	/* Create token ID tensor [context_len] */
	dims[0] = SAM3_TOKENIZER_CONTEXT_LEN;
	tok_tensor = gh_alloc_tensor(arena, SAM3_DTYPE_I32, 1, dims);
	if (!tok_tensor)
		return NULL;

	memcpy(tok_tensor->data, tokens,
	       SAM3_TOKENIZER_CONTEXT_LEN * sizeof(int32_t));

	/* Run text encoder: tokens -> per-token embeddings + pooled */
	return sam3_text_encoder_build(&vl->text_enc, g, tok_tensor,
				       pooled_out, arena);
}
