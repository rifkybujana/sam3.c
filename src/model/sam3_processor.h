/*
 * src/model/sam3_processor.h - High-level image processor API
 *
 * Provides a convenient top-level interface that owns all resources
 * needed for SAM3 inference: backend, arenas, and the image model.
 * Users call init, load, set_image, then segment. The processor
 * manages arena lifetimes, backend creation, and graph construction
 * internally so callers do not need to interact with lower layers.
 *
 * Key types:  sam3_processor
 * Depends on: sam3_image.h, backend/backend.h, sam3/sam3_types.h
 * Used by:    sam3.c (top-level context), tools/
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_SAM3_PROCESSOR_H
#define SAM3_MODEL_SAM3_PROCESSOR_H

#include "sam3_image.h"
#include "backend/backend.h"
#include "sam3/sam3_types.h"

/* Forward declaration — only used as an opaque pointer here. */
struct sam3_profiler;

struct sam3_processor {
	struct sam3_image_model model;
	struct sam3_backend *backend;
	struct sam3_arena model_arena;    /* weights + cached features */
	struct sam3_arena scratch_arena;  /* per-inference temp */
	int image_loaded;
	struct sam3_profiler *profiler;   /* NULL when profiling disabled */
};

/*
 * sam3_processor_init - Initialize processor with CPU backend and arenas.
 *
 * @proc: Processor struct (caller-allocated)
 *
 * Creates a CPU backend, allocates model and scratch arenas, and
 * initializes the image model. Must be followed by sam3_processor_load()
 * before inference.
 *
 * Returns SAM3_OK on success, or an error code on failure.
 */
enum sam3_error sam3_processor_init(struct sam3_processor *proc);

/*
 * sam3_processor_load - Load model weights from file.
 *
 * @proc:       Initialized processor
 * @model_path: Path to .sam3 weight file
 * @vocab_path: Path to BPE vocabulary file, or NULL
 *
 * Opens the weight file, loads all sub-module weights into the model
 * arena, then closes the file. The processor is ready for set_image
 * after this returns.
 *
 * Returns SAM3_OK on success, or propagates weight/model errors.
 */
enum sam3_error sam3_processor_load(struct sam3_processor *proc,
				    const char *model_path,
				    const char *vocab_path);

/*
 * sam3_processor_free - Release all processor resources.
 *
 * @proc: Processor to free (may be partially initialized).
 *
 * Frees the image model, backend, and both arenas.
 */
void sam3_processor_free(struct sam3_processor *proc);

/*
 * sam3_processor_set_image - Set image and run vision encoder.
 *
 * @proc:   Initialized and loaded processor
 * @pixels: RGB pixel data, H * W * 3 bytes (interleaved)
 * @width:  Image width in pixels
 * @height: Image height in pixels
 *
 * Converts uint8 RGB pixels to a normalized float [3, H, W] tensor,
 * builds the vision pipeline graph, and evaluates it to cache image
 * features. Resets the scratch arena before each call.
 *
 * Returns SAM3_OK on success.
 */
enum sam3_error sam3_processor_set_image(struct sam3_processor *proc,
					 const uint8_t *pixels,
					 int width, int height);

/*
 * sam3_processor_segment - Run segmentation with geometric prompts.
 *
 * @proc:      Processor with image already set
 * @prompts:   Array of point/box prompts
 * @n_prompts: Number of prompts
 * @result:    Output result (caller must call sam3_result_free)
 *
 * Projects prompt coordinates to model embeddings, builds the
 * segmentation graph (geometry encoder, fusion, decoder, seg head),
 * evaluates it, and copies mask logits into the result struct.
 * Result masks and iou_scores are malloc'd; caller frees with
 * sam3_result_free().
 *
 * Returns SAM3_OK on success, SAM3_EINVAL if no image loaded.
 */
enum sam3_error sam3_processor_segment(struct sam3_processor *proc,
				       const struct sam3_prompt *prompts,
				       int n_prompts,
				       struct sam3_result *result);

#endif /* SAM3_MODEL_SAM3_PROCESSOR_H */
