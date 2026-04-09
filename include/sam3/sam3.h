/*
 * include/sam3/sam3.h - Main public API for sam3 inference
 *
 * This is the only header users need to include. Provides functions to
 * load a SAM3 model, run segmentation inference with point/box/mask/text
 * prompts, and free resources. All functions are thread-safe with
 * respect to different sam3_ctx instances.
 *
 * Key types:  sam3_ctx (opaque handle)
 * Depends on: sam3_types.h
 * Used by:    tools/sam3_main.c, user applications
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_H
#define SAM3_H

#include "sam3_types.h"

/*
 * sam3_init - Create and initialize a sam3 context.
 *
 * Initializes the compute backend and prepares for model loading.
 * Returns NULL on failure. Call sam3_free() when done.
 */
sam3_ctx *sam3_init(void);

/*
 * sam3_free - Release all resources held by a sam3 context.
 *
 * @ctx: Context to free (may be NULL).
 */
void sam3_free(sam3_ctx *ctx);

/*
 * sam3_load_model - Load SAM3 model weights from a file.
 *
 * @ctx:  Initialized context
 * @path: Path to model weights file
 *
 * Returns SAM3_OK on success. The context takes ownership of the
 * loaded weights and frees them in sam3_free().
 */
enum sam3_error sam3_load_model(sam3_ctx *ctx, const char *path);

/*
 * sam3_load_bpe - Load BPE vocabulary for text prompt tokenization.
 *
 * @ctx:  Context with loaded model
 * @path: Path to BPE vocabulary file (e.g., bpe_simple_vocab_16e6.txt.gz)
 *
 * Without BPE vocab, the tokenizer falls back to byte-level encoding
 * which produces incorrect tokens. sam3_load_model() auto-discovers
 * BPE files next to the model, so this is only needed if the BPE
 * file is in a different location.
 */
enum sam3_error sam3_load_bpe(sam3_ctx *ctx, const char *path);

/*
 * sam3_set_image - Set the input image for segmentation.
 *
 * @ctx:    Initialized context with loaded model
 * @pixels: RGB pixel data (H * W * 3 uint8_t values)
 * @width:  Image width in pixels
 * @height: Image height in pixels
 *
 * Runs the image encoder. Subsequent sam3_segment() calls reuse the
 * encoded image until a new image is set.
 */
enum sam3_error sam3_set_image(sam3_ctx *ctx, const uint8_t *pixels,
			       int width, int height);

/*
 * sam3_set_image_file - Load an image file and set it for segmentation.
 *
 * @ctx:  Initialized context with loaded model
 * @path: Path to PNG, JPEG, or BMP image file
 *
 * Convenience function: loads the image, resizes/letterboxes to model
 * input size, then calls sam3_set_image(). Equivalent to loading and
 * resizing manually.
 */
enum sam3_error sam3_set_image_file(sam3_ctx *ctx, const char *path);

/*
 * sam3_segment - Run segmentation with the given prompts.
 *
 * @ctx:       Context with image already set
 * @prompts:   Array of prompts (points, boxes, masks, or text)
 * @n_prompts: Number of prompts
 * @result:    Output result (caller must call sam3_result_free)
 *
 * Supports text prompts (SAM3_PROMPT_TEXT) alone or mixed with
 * geometric prompts. Only the first text prompt is used if multiple
 * are provided. Returns SAM3_OK on success.
 */
enum sam3_error sam3_segment(sam3_ctx *ctx, const struct sam3_prompt *prompts,
			     int n_prompts, struct sam3_result *result);

/*
 * sam3_result_free - Free memory allocated in a sam3_result.
 *
 * @result: Result struct to free (fields set to NULL/0 after).
 */
void sam3_result_free(struct sam3_result *result);

/*
 * sam3_version - Return the sam3 version string.
 */
const char *sam3_version(void);


/* --- Profiling API --- */

/*
 * sam3_profile_enable - Enable profiling on this context.
 *
 * @ctx: Initialized context.
 *
 * Allocates profiler state if not already present. Profiling data
 * accumulates until reset or context is freed.
 * Requires SAM3_HAS_PROFILE at compile time; otherwise a no-op.
 */
enum sam3_error sam3_profile_enable(sam3_ctx *ctx);

/*
 * sam3_profile_disable - Disable profiling (data is preserved).
 *
 * @ctx: Initialized context.
 */
void sam3_profile_disable(sam3_ctx *ctx);

/*
 * sam3_profile_report - Print profiling report to stderr.
 *
 * @ctx: Initialized context.
 */
void sam3_profile_report(sam3_ctx *ctx);

/*
 * sam3_profile_reset - Clear all collected profiling data.
 *
 * @ctx: Initialized context.
 */
void sam3_profile_reset(sam3_ctx *ctx);

#endif /* SAM3_H */
