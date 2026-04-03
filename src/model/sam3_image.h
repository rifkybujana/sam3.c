/*
 * src/model/sam3_image.h - SAM3 top-level image model
 *
 * Defines the composite image model that wires together all SAM3
 * sub-modules: vision-language backbone, encoder fusion, transformer
 * decoder, geometry encoder, segmentation head, and dot-product scorer.
 * Provides a two-phase API: encode (run vision backbone, cache features)
 * and segment (build decoder + segmentation graph from prompts).
 *
 * Key types:  sam3_image_model
 * Depends on: vl_combiner.h, encoder.h, decoder.h, prompt_encoder.h,
 *             segmentation.h, model_misc.h, core/graph.h, core/alloc.h,
 *             backend/backend.h
 * Used by:    sam3.c (top-level context)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_SAM3_IMAGE_H
#define SAM3_MODEL_SAM3_IMAGE_H

#include "vl_combiner.h"
#include "encoder.h"
#include "decoder.h"
#include "prompt_encoder.h"
#include "segmentation.h"
#include "model_misc.h"
#include "core/graph.h"
#include "core/alloc.h"
#include "backend/backend.h"

struct sam3_image_model {
	struct sam3_vl_backbone backbone;
	struct sam3_encoder_fusion encoder;
	struct sam3_decoder decoder;
	struct sam3_geometry_encoder geom_enc;
	struct sam3_seg_head seg_head;
	struct sam3_dot_scorer scorer;

	/* Cached image features after encoding */
	struct sam3_tensor *cached_image_features; /* [n_pixels, d_model] */
	struct sam3_tensor *cached_text_features;  /* [seq_len, d_model] */
	int image_encoded;
};

/*
 * sam3_image_model_init - Initialize the image model with default SAM3
 * configuration.
 *
 * @model: Image model struct (caller-allocated)
 * @arena: Arena for precomputation (RoPE, position encoding)
 *
 * Initializes all sub-modules (backbone, encoder fusion, decoder,
 * geometry encoder, segmentation head, scorer) with SAM3 defaults.
 *
 * Returns SAM3_OK on success, or propagates the first sub-module error.
 */
enum sam3_error sam3_image_model_init(struct sam3_image_model *model,
				      struct sam3_arena *arena);

/*
 * sam3_image_model_load - Load all sub-module weights from weight file.
 *
 * @model:      Initialized image model
 * @wf:         Open weight file (may be NULL for zero-init fallback)
 * @vocab_path: Path to BPE vocab file, or NULL to keep byte-level fallback
 * @arena:      Arena for weight tensor allocation
 *
 * Loads backbone, encoder fusion, decoder, geometry encoder, segmentation
 * head, and scorer weights. Optionally re-initializes the tokenizer with
 * a full BPE vocabulary.
 *
 * Returns SAM3_OK on success, or propagates the first sub-module error.
 */
enum sam3_error sam3_image_model_load(struct sam3_image_model *model,
				      const struct sam3_weight_file *wf,
				      const char *vocab_path,
				      struct sam3_arena *arena);

/*
 * sam3_image_model_free - Free non-arena resources (tokenizer).
 *
 * @model: Image model to free (may be NULL).
 *
 * Only the VL backbone tokenizer uses heap allocation; all other
 * sub-modules use arena allocation and do not need explicit freeing.
 */
void sam3_image_model_free(struct sam3_image_model *model);

/*
 * sam3_image_model_encode - Encode image: run vision backbone + cache.
 *
 * Builds the vision pipeline graph (ViT + feature pyramid neck),
 * evaluates it on the backend, and caches the resulting image features
 * for subsequent segment calls.
 *
 * @model: Initialized and loaded image model
 * @g:     Graph to build vision pipeline into
 * @be:    Backend for graph evaluation
 * @image: Input image [3, img_size, img_size] normalized F32 tensor
 * @arena: Arena for intermediate tensors
 *
 * Returns SAM3_OK on success. Sets model->image_encoded = 1 on success.
 */
enum sam3_error sam3_image_model_encode(struct sam3_image_model *model,
					struct sam3_graph *g,
					struct sam3_backend *be,
					struct sam3_tensor *image,
					struct sam3_arena *arena);

/*
 * sam3_image_model_segment - Build segmentation graph from prompts.
 *
 * Requires image already encoded via sam3_image_model_encode().
 * Builds the geometry encoder, encoder fusion, decoder, and
 * segmentation head subgraphs. Does NOT evaluate the graph --
 * caller must call be->ops->graph_eval() after this returns.
 *
 * @model:          Initialized, loaded, and image-encoded model
 * @g:              Graph to add segmentation nodes to
 * @be:             Backend (unused, reserved for future use)
 * @prompt_tokens:  [N, d_model] pre-projected prompt embeddings, or NULL
 * @text_features:  [seq_len, d_model] text encoder output, or NULL
 * @arena:          Arena for intermediate tensors
 *
 * At least one of prompt_tokens or text_features must be non-NULL.
 * When both are provided, their features are concatenated along the
 * sequence dimension for encoder fusion and decoder cross-attention.
 *
 * Returns mask logits tensor, or NULL on error.
 */
struct sam3_tensor *sam3_image_model_segment(
	struct sam3_image_model *model,
	struct sam3_graph *g,
	struct sam3_backend *be,
	struct sam3_tensor *prompt_tokens,
	struct sam3_tensor *text_features,
	struct sam3_arena *arena);

#endif /* SAM3_MODEL_SAM3_IMAGE_H */
