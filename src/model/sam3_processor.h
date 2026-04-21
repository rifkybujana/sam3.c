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
#include "feature_cache.h"

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
	struct sam3_arena model_arena;    /* weights only (cached features in img_cache slots) */
	struct sam3_arena scratch_arena;  /* per-inference temp */
	struct sam3_arena video_scratch_arena; /* video session_encode_frame persist */
	struct sam3_image_feature_cache *img_cache;
	int current_img_slot;            /* -1 == none */
	int image_loaded;                /* 1 iff cached_* pointers are live */
	int prompt_w;			 /* user-space image width for coord norm */
	int prompt_h;			 /* user-space image height for coord norm */
	struct sam3_profiler *profiler;   /* NULL when profiling disabled */

	/*
	 * Async text encoding state (#11). The text encoder runs on a
	 * worker thread against its own backend and scratch arena so
	 * that it overlaps with sam3_processor_set_image on the main
	 * thread. Results are published through txt_cache: on a miss
	 * the worker writes its output tensor into the per-slot arena
	 * of the slot it pre-claimed (text_worker_slot); on a hit the
	 * cached bundle pointer is stashed in text_cached_bundle and
	 * no worker is spawned.
	 *
	 * - text_backend:        second backend handle for the worker
	 * - text_scratch_arena:  worker's per-block scratch
	 * - txt_cache:           LRU text feature cache
	 * - text_cached_bundle:  hit path; cleared by segment
	 * - text_worker_slot:    -1 or txt_cache slot the worker writes
	 * - text_thread:         pthread handle (valid iff active=1)
	 * - text_thread_active:  1 between pthread_create and join
	 * - text_thread_err:     last worker exit code
	 * - text_tokens:         raw token IDs the worker reads
	 * - text_n_tokens:       number of real (non-padding) tokens
	 */
	struct sam3_backend *text_backend;
	struct sam3_arena    text_scratch_arena;
	struct sam3_text_feature_cache *txt_cache;
	struct sam3_text_bundle *text_cached_bundle; /* hit path; cleared by segment */
	int                      text_worker_slot;   /* -1 or txt_cache slot */
	pthread_t            text_thread;
	int                  text_thread_active;
	enum sam3_error      text_thread_err;
	int32_t              text_tokens[SAM3_PROCESSOR_MAX_TOKENS];
	int                  text_n_tokens;
};

/*
 * sam3_processor_init - Initialize processor with CPU backend and arenas.
 *
 * @proc:          Processor struct (caller-allocated)
 * @backbone_type: SAM3_BACKBONE_HIERA or SAM3_BACKBONE_EFFICIENTVIT
 * @n_fpn_scales:  Number of FPN scales (3 for SAM 3.1, 4 for SAM 3)
 *
 * Creates a CPU backend, allocates model and scratch arenas, and
 * initializes the image model. Must be followed by sam3_processor_load()
 * before inference.
 *
 * Returns SAM3_OK on success, or an error code on failure.
 */
enum sam3_error sam3_processor_init(struct sam3_processor *proc,
				    int backbone_type,
				    int n_fpn_scales);

/*
 * sam3_processor_init_ex - Like sam3_processor_init, but with caller-
 * supplied cache slot counts. Pass 0 for either to use the defaults
 * (3 image, 16 text). Pass 1 to disable multi-slot behavior.
 */
enum sam3_error sam3_processor_init_ex(struct sam3_processor *proc,
				       int backbone_type,
				       int n_fpn_scales,
				       int n_image_slots,
				       int n_text_slots);

/*
 * sam3_processor_cache_clear - Flush feature caches.
 *
 * @proc:  Processor to flush.
 * @which: Bitmask. Bit 0 (1) clears the image cache; bit 1 (2)
 *         clears the text cache. Pass 0 to clear all caches.
 */
void sam3_processor_cache_clear(struct sam3_processor *proc, unsigned which);

/* sam3_processor_cache_stats - Read aggregate cache hit/miss counters. */
void sam3_processor_cache_stats(const struct sam3_processor *proc,
				struct sam3_cache_stats *out);

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
 * Tokenizes the text on the caller thread and looks the token
 * sequence up in proc->txt_cache. On a cache hit the cached
 * bundle pointer is stashed in proc->text_cached_bundle and the
 * call returns immediately — no worker is spawned. On a miss the
 * call claims a cache slot and spawns a worker pthread that runs
 * the text encoder against proc->text_backend (CPU), writing its
 * output tensor into the slot's per-slot arena. The call returns
 * as soon as the worker is spawned so the caller can proceed
 * with sam3_processor_set_image() (image encoder runs
 * concurrently on the Metal backend). The worker output is
 * consumed automatically by the next sam3_processor_segment()
 * call, which joins the worker first and reads the features
 * directly from the cache slot.
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
