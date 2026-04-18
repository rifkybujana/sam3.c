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
 * Used by:    sam3.c (top-level context), model/sam3_video.c, tools/
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_SAM3_PROCESSOR_H
#define SAM3_MODEL_SAM3_PROCESSOR_H

#include <pthread.h>
#include <stdint.h>

#include "sam3_image.h"
#include "backend/backend.h"
#include "sam3/sam3_types.h"

/* Forward declaration — only used as an opaque pointer here. */
struct sam3_profiler;

/*
 * SAM3 text prompt token capacity. Matches the CLIP context length used
 * by the text encoder; sized as a small fixed array to keep set_text
 * lock-free with respect to the worker thread.
 */
#define SAM3_PROCESSOR_MAX_TOKENS 77

struct sam3_processor {
	struct sam3_image_model model;
	struct sam3_backend *backend;
	struct sam3_arena model_arena;    /* weights + cached features */
	struct sam3_arena scratch_arena;  /* per-inference temp */
	size_t weights_end;              /* model_arena offset after load */
	int image_loaded;
	int prompt_w;			 /* user-space image width for coord norm */
	int prompt_h;			 /* user-space image height for coord norm */
	struct sam3_profiler *profiler;   /* NULL when profiling disabled */

	/*
	 * Async text encoding state (#11). The text encoder runs on a
	 * worker thread against its own backend and arenas so that it
	 * overlaps with sam3_processor_set_image on the main thread.
	 *
	 * - text_backend:        second backend handle for the worker
	 * - text_scratch_arena:  worker's per-block scratch
	 * - text_persist_arena:  output buffer that survives the worker
	 * - text_thread:         pthread handle (valid iff active=1)
	 * - text_thread_active:  1 between pthread_create and join
	 * - text_thread_err:     last worker exit code
	 * - text_features_async: text features the worker produced
	 *                        (NULL when no async result is pending)
	 * - text_tokens:         raw token IDs the worker reads
	 * - text_n_tokens:       number of real (non-padding) tokens
	 */
	struct sam3_backend *text_backend;
	struct sam3_arena    text_scratch_arena;
	struct sam3_arena    text_persist_arena;
	pthread_t            text_thread;
	int                  text_thread_active;
	enum sam3_error      text_thread_err;
	struct sam3_tensor  *text_features_async;
	int32_t              text_tokens[SAM3_PROCESSOR_MAX_TOKENS];
	int                  text_n_tokens;
};

/*
 * sam3_processor_init - Initialize processor with CPU backend and arenas.
 *
 * @proc:          Processor struct (caller-allocated)
 * @backbone_type: SAM3_BACKBONE_HIERA or SAM3_BACKBONE_EFFICIENTVIT
 *
 * Creates a CPU backend, allocates model and scratch arenas, and
 * initializes the image model. Must be followed by sam3_processor_load()
 * before inference.
 *
 * Returns SAM3_OK on success, or an error code on failure.
 */
enum sam3_error sam3_processor_init(struct sam3_processor *proc,
				    int backbone_type);

/*
 * sam3_processor_load - Load model weights from an open weight file.
 *
 * @proc:       Initialized processor
 * @wf:         Open weight file (caller retains ownership)
 * @vocab_path: Path to BPE vocabulary file, or NULL
 *
 * Loads all sub-module weights into the model arena. The weight file
 * must remain open for the processor's lifetime since tensor data
 * points directly into the mmap region. The processor is ready for
 * set_image after this returns.
 *
 * Returns SAM3_OK on success, or propagates weight/model errors.
 */
enum sam3_error sam3_processor_load(struct sam3_processor *proc,
				    const struct sam3_weight_file *wf,
				    const char *vocab_path);

/*
 * sam3_processor_free - Release all processor resources.
 *
 * @proc: Processor to free (may be partially initialized).
 *
 * Frees the image model, backend, and both arenas. If an async text
 * encoding worker is in flight (see sam3_processor_set_text) it is
 * joined first. Callers MUST invoke this before sam3_weight_close()
 * on the underlying weight file: the worker reads tensor data
 * directly from the mmap'd weight region, and unmapping it while
 * the worker is still running will SEGV.
 */
void sam3_processor_free(struct sam3_processor *proc);

/*
 * sam3_processor_img_size - Return the backbone's actual input image size.
 *
 * @proc: Initialized processor.
 *
 * Returns the img_size the encoder was configured with (e.g. 512 for
 * EfficientViT, 1008 for Hiera/TinyViT).
 */
int sam3_processor_img_size(const struct sam3_processor *proc);

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
 * sam3_processor_set_text - Pre-tokenize and encode a text prompt.
 *
 * @proc: Initialized and loaded processor
 * @text: Null-terminated text prompt (e.g. "cat")
 *
 * Tokenizes the text on the caller thread, then spawns a worker
 * pthread that runs the text encoder against proc->text_backend
 * (CPU) using the per-text arenas. The call returns as soon as the
 * worker is spawned so the caller can proceed with
 * sam3_processor_set_image() (image encoder runs concurrently on
 * the Metal backend). The worker output is held in
 * text_persist_arena and consumed automatically by the next
 * sam3_processor_segment() call, which joins the worker first.
 *
 * Calling set_text twice without an intervening segment() joins
 * the previous worker before discarding its result. The processor
 * must have a valid text_backend (created in init); if it is NULL,
 * this returns SAM3_EBACKEND.
 *
 * Lifetime: the worker reads weight tensors directly from the mmap'd
 * .sam3 file. The weight file MUST outlive the worker — i.e.
 * sam3_processor_free() must be called before sam3_weight_close().
 *
 * Returns SAM3_OK on success.
 */
enum sam3_error sam3_processor_set_text(struct sam3_processor *proc,
					const char *text);

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

/*
 * sam3_project_prompts - CPU-side prompt coordinate projection.
 *
 * @model:        Loaded image model (for geometry encoder weights)
 * @feat_s1_nhwc: Cached 1x backbone feature [1, H, W, d_model] used
 *                for the pool-projection path. May be NULL; the
 *                pool-projection term is skipped in that case.
 * @prompts:      Array of point/box prompts
 * @n_prompts:    Number of prompts
 * @prompt_w:     Width used to normalize x coordinates
 * @prompt_h:     Height used to normalize y coordinates
 * @arena:        Arena for the output tensor and internal LN scratch
 *
 * Projects point/box coordinates to d_model embeddings purely on CPU
 * (no graph or backend involvement) and returns a freshly allocated
 * [N_total, d_model] tensor owned by @arena. Returns NULL if no point
 * or box prompts are present, or on allocation failure.
 *
 * Shared between sam3_processor_segment (single-image path) and the
 * video tracker's add_points / add_box paths.
 */
struct sam3_tensor *sam3_project_prompts(
	struct sam3_image_model *model,
	const struct sam3_tensor *feat_s1_nhwc,
	const struct sam3_prompt *prompts,
	int n_prompts,
	int prompt_w, int prompt_h,
	struct sam3_arena *arena);

#endif /* SAM3_MODEL_SAM3_PROCESSOR_H */
