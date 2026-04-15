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
#include "mask_decoder.h"
#include "model_misc.h"
#include "core/graph.h"
#include "core/alloc.h"
#include "backend/backend.h"

struct sam3_profiler;

struct sam3_image_model {
	struct sam3_vl_backbone backbone;
	struct sam3_encoder_fusion encoder;
	struct sam3_decoder decoder;
	struct sam3_geometry_encoder geom_enc;
	struct sam3_seg_head seg_head;
	struct sam3_mask_decoder mask_dec;
	struct sam3_dot_scorer scorer;

	/* Cached image features after encoding */
	struct sam3_tensor *cached_image_features; /* [1, H, W, d_model] */
	struct sam3_tensor *cached_text_features;  /* [seq_len, d_model] */
	/*
	 * Multi-scale backbone features for FPN pixel decoders. After
	 * Task 10 every consumer is NHWC, so only NHWC snapshots are
	 * kept.
	 */
	struct sam3_tensor *cached_feat_s0_nhwc; /* [1, 2H, 2W, d] 2x */
	struct sam3_tensor *cached_feat_s1_nhwc; /* [1, H,  W,  d] 1x */
	struct sam3_tensor *cached_feat_4x_nhwc; /* [1, 4H, 4W, d] 4x */
	int image_encoded;
};

/*
 * sam3_image_model_init - Initialize the image model with default SAM3
 * configuration.
 *
 * @model:         Image model struct (caller-allocated)
 * @backbone_type: SAM3_BACKBONE_HIERA or SAM3_BACKBONE_EFFICIENTVIT
 * @arena:         Arena for precomputation (RoPE, position encoding)
 *
 * Initializes all sub-modules (backbone, encoder fusion, decoder,
 * geometry encoder, segmentation head, scorer) with SAM3 defaults.
 *
 * Returns SAM3_OK on success, or propagates the first sub-module error.
 */
enum sam3_error sam3_image_model_init(struct sam3_image_model *model,
				      int backbone_type,
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
 * Evaluates the ViT per-block (resetting scratch between blocks),
 * then builds and evaluates the neck graph. Caches the resulting
 * image features for subsequent segment calls.
 *
 * @model:    Initialized and loaded image model
 * @be:       Backend for graph evaluation
 * @image:    Input image [3, img_size, img_size] normalized F32 tensor
 * @scratch:  Arena for per-block/stage intermediate tensors
 * @persist:  Arena for persistent outputs (ViT buffer, cached features)
 * @profiler: Profiler for sub-stage timing (may be NULL)
 *
 * Returns SAM3_OK on success. Sets model->image_encoded = 1 on success.
 */
enum sam3_error sam3_image_model_encode(struct sam3_image_model *model,
					struct sam3_backend *be,
					struct sam3_tensor *image,
					struct sam3_arena *scratch,
					struct sam3_arena *persist,
					struct sam3_profiler *profiler);

/*
 * sam3_image_model_segment - Run segmentation pipeline per-stage.
 *
 * Requires image already encoded via sam3_image_model_encode().
 * Evaluates each stage (geometry encoder, encoder fusion, decoder,
 * segmentation head) independently, resetting the scratch arena
 * between stages to keep peak memory bounded.
 *
 * @model:          Initialized, loaded, and image-encoded model
 * @be:             Backend for graph evaluation (GPU stages)
 * @cpu_be:         CPU backend for decoder + scorer (deterministic).
 *                  If NULL, @be is used for all stages.
 * @prompt_tokens:  [N, d_model] pre-projected prompt embeddings, or NULL
 *                  (must be materialized, i.e. already evaluated)
 * @text_features:  [seq_len, d_model] text encoder output, or NULL
 *                  (must be materialized, i.e. already evaluated)
 * @scratch:        Arena for per-stage intermediate tensors (reset between)
 * @persist:        Arena for inter-stage data (offset saved/restored)
 * @out_masks:      Receives mask logits tensor (allocated from scratch)
 * @out_scores:     Receives scorer output [n_queries, 1] after sigmoid,
 *                  or NULL if text_features is NULL. May be NULL if caller
 *                  does not need scores.
 * @profiler:       Profiler for sub-stage timing (may be NULL)
 *
 * At least one of prompt_tokens or text_features must be non-NULL.
 * Input tensors must have their data in persist or model_arena (NOT
 * scratch), since scratch is reset between stages.
 *
 * Returns SAM3_OK on success, error code otherwise.
 */
enum sam3_error sam3_image_model_segment(
	struct sam3_image_model *model,
	struct sam3_backend *be,
	struct sam3_backend *cpu_be,
	struct sam3_tensor *prompt_tokens,
	struct sam3_tensor *text_features,
	struct sam3_arena *scratch,
	struct sam3_arena *persist,
	struct sam3_tensor **out_masks,
	struct sam3_tensor **out_scores,
	struct sam3_profiler *profiler);

#endif /* SAM3_MODEL_SAM3_IMAGE_H */
