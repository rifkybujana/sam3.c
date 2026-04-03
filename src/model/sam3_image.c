/*
 * src/model/sam3_image.c - SAM3 top-level image model implementation
 *
 * Wires together all SAM3 sub-modules into a two-phase inference
 * pipeline: encode (run ViT + neck, cache image features) and segment
 * (geometry encoder + encoder fusion + decoder + segmentation head).
 * The encode phase evaluates the graph immediately to materialize cached
 * features; the segment phase only builds the graph for later evaluation.
 *
 * Key types:  sam3_image_model
 * Depends on: sam3_image.h, graph_helpers.h
 * Used by:    sam3.c (top-level context)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <string.h>

#include "sam3_image.h"
#include "graph_helpers.h"

enum sam3_error sam3_image_model_init(struct sam3_image_model *model,
				      struct sam3_arena *arena)
{
	enum sam3_error err;

	memset(model, 0, sizeof(*model));

	/* Vision-language backbone (ViT + neck + text encoder + tokenizer) */
	err = sam3_vl_backbone_init(&model->backbone, arena);
	if (err != SAM3_OK)
		return err;

	/* Encoder fusion: 6-layer transformer for image-text fusion */
	err = sam3_encoder_fusion_init(&model->encoder, 256, 8, 6, 2048);
	if (err != SAM3_OK)
		return err;

	/* Decoder: 6-layer transformer with 200 learned queries */
	err = sam3_decoder_init(&model->decoder, 256, 8, 6, 2048, 200);
	if (err != SAM3_OK)
		return err;

	/* Geometry encoder: 3-layer cross-attention for point/box prompts */
	err = sam3_geometry_encoder_init(&model->geom_enc, 256, 3);
	if (err != SAM3_OK)
		return err;

	/* Segmentation head: pixel decoder + mask prediction */
	err = sam3_seg_head_init(&model->seg_head, 256);
	if (err != SAM3_OK)
		return err;

	/* Dot-product scorer config (weights loaded separately) */
	model->scorer.input_dim = 256;
	model->scorer.hidden_dim = 2048;

	return SAM3_OK;
}

enum sam3_error sam3_image_model_load(struct sam3_image_model *model,
				      const struct sam3_weight_file *wf,
				      const char *vocab_path,
				      struct sam3_arena *arena)
{
	enum sam3_error err;

	/* Load backbone weights (ViT + neck + text encoder) */
	err = sam3_vl_backbone_load(&model->backbone, wf, arena);
	if (err != SAM3_OK)
		return err;

	/*
	 * Tokenizer vocab: the backbone init already sets up a byte-level
	 * fallback tokenizer. If a vocab path is provided, re-initialize
	 * with the full BPE vocabulary. Currently sam3_tokenizer only
	 * supports byte-level init, so vocab_path is reserved for future
	 * use when BPE file loading is implemented.
	 */
	(void)vocab_path;

	/* Load encoder fusion weights */
	err = sam3_encoder_fusion_load(&model->encoder, wf, arena);
	if (err != SAM3_OK)
		return err;

	/* Load decoder weights */
	err = sam3_decoder_load(&model->decoder, wf, arena);
	if (err != SAM3_OK)
		return err;

	/* Load geometry encoder weights */
	err = sam3_geometry_encoder_load(&model->geom_enc, wf, arena);
	if (err != SAM3_OK)
		return err;

	/* Load segmentation head weights */
	err = sam3_seg_head_load(&model->seg_head, wf, arena);
	if (err != SAM3_OK)
		return err;

	/* Load dot-product scorer weights */
	err = sam3_dot_scorer_load(&model->scorer, wf, arena);
	if (err != SAM3_OK)
		return err;

	return SAM3_OK;
}

void sam3_image_model_free(struct sam3_image_model *model)
{
	if (model)
		sam3_vl_backbone_free(&model->backbone);
}

enum sam3_error sam3_image_model_encode(struct sam3_image_model *model,
					struct sam3_graph *g,
					struct sam3_backend *be,
					struct sam3_tensor *image,
					struct sam3_arena *arena)
{
	struct sam3_tensor *features[4];
	struct sam3_tensor *vit_out;
	enum sam3_error err;

	/* Build vision pipeline: ViT -> neck -> multi-scale features */
	vit_out = sam3_vl_backbone_build_vision(&model->backbone, g,
						image, features, arena);
	if (!vit_out)
		return SAM3_ENOMEM;

	/*
	 * Cache the finest-scale feature from the neck. With scalp=1,
	 * we keep the last scale (index n_scales - scalp = 3) which
	 * is the 0.5x scale at [1, d_model, 36, 36]. For encoder
	 * fusion and downstream processing, we use this as the primary
	 * image feature representation.
	 */
	model->cached_image_features = features[3];

	/* Evaluate the vision graph to materialize cached features */
	err = be->ops->graph_eval(be, g);
	if (err != SAM3_OK)
		return err;

	model->image_encoded = 1;
	return SAM3_OK;
}

struct sam3_tensor *sam3_image_model_segment(
	struct sam3_image_model *model,
	struct sam3_graph *g,
	struct sam3_backend *be,
	struct sam3_tensor *prompt_tokens,
	struct sam3_tensor *text_features,
	struct sam3_arena *arena)
{
	struct sam3_tensor *geom_out = NULL;
	struct sam3_tensor *context;
	struct sam3_tensor *fused;
	struct sam3_tensor *queries;
	struct sam3_tensor *box_out = NULL;
	struct sam3_tensor *masks;
	int grid_h, grid_w;

	(void)be; /* reserved for future use */

	if (!model->image_encoded)
		return NULL;

	if (!prompt_tokens && !text_features)
		return NULL;

	/*
	 * Geometry encoder: cross-attend prompt tokens to cached
	 * image features. Output is [N+1, d_model].
	 */
	if (prompt_tokens) {
		geom_out = sam3_geometry_encoder_build(
			&model->geom_enc, g,
			prompt_tokens,
			model->cached_image_features,
			arena);
		if (!geom_out)
			return NULL;
	}

	/*
	 * Build context for encoder fusion and decoder.
	 * - Text only: use text_features
	 * - Geometry only: use geom_out
	 * - Both: concat(text_features, geom_out) along axis 0
	 */
	if (text_features && geom_out) {
		struct sam3_tensor *parts[2] = {
			text_features, geom_out
		};
		context = gh_concat(g, arena, parts, 2, 0);
		if (!context)
			return NULL;
	} else if (text_features) {
		context = text_features;
	} else {
		context = geom_out;
	}

	/*
	 * Encoder fusion: fuse image features with context
	 * (text and/or geometry) via self-attention and
	 * cross-attention. Output is [n_pixels, d_model].
	 */
	fused = sam3_encoder_fusion_build(&model->encoder, g,
					  model->cached_image_features,
					  context, arena);
	if (!fused)
		return NULL;

	/*
	 * Decoder: process fused features with learned queries.
	 * Cross-attends to encoder output and context tokens.
	 * Output is query embeddings [n_queries, d_model].
	 */
	queries = sam3_decoder_build(&model->decoder, g, fused,
				     context, &box_out, arena);
	if (!queries)
		return NULL;

	/*
	 * Segmentation head: upsamples fused features and computes
	 * dot product with query embeddings to produce mask logits.
	 */
	grid_h = model->backbone.vit.grid_size;
	grid_w = model->backbone.vit.grid_size;
	masks = sam3_seg_head_build(&model->seg_head, g, queries,
				    fused, grid_h, grid_w, arena);

	return masks;
}
