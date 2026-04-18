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
 * sam3_set_prompt_space - Set the coordinate space for point/box prompts.
 *
 * @ctx:    Context with image set
 * @width:  Width of the image in the user's coordinate space
 * @height: Height of the image in the user's coordinate space
 *
 * sam3_set_image_file() automatically sets this to the original file
 * dimensions. Call this only when using sam3_set_image() with a
 * pre-resized image while providing prompts in the original coordinates.
 */
void sam3_set_prompt_space(sam3_ctx *ctx, int width, int height);

/*
 * sam3_set_text - Pre-tokenize and asynchronously encode a text prompt.
 *
 * @ctx:  Initialized context with loaded model
 * @text: Null-terminated prompt text (e.g. "cat")
 *
 * Optional optimization: tokenizes @text on the caller thread, then
 * spawns a worker that runs the text encoder on a CPU backend so it
 * overlaps with the next sam3_set_image() call (which runs the image
 * encoder on the main backend). The encoded features are consumed
 * automatically by the next sam3_segment() call. Calling sam3_segment
 * without a prior sam3_set_text() still works and uses the inline
 * legacy path.
 *
 * Returns SAM3_OK on success.
 */
enum sam3_error sam3_set_text(sam3_ctx *ctx, const char *text);

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
 * sam3_get_image_size - Return the model's input image size.
 *
 * @ctx: Context with loaded model.
 *
 * Returns the image size the model expects (e.g. 512 for EfficientViT,
 * 1008 for Hiera). Returns 0 if no model is loaded.
 */
int sam3_get_image_size(const sam3_ctx *ctx);

/*
 * sam3_version - Return the sam3 version string.
 */
const char *sam3_version(void);

/*
 * sam3_error_str - Return a human-readable string for an error code.
 *
 * @err: Error code to describe.
 *
 * Returns a static string. Never returns NULL.
 */
const char *sam3_error_str(enum sam3_error err);

/*
 * sam3_log_set_level - Set the minimum log level.
 *
 * @level: Messages below this level are suppressed.
 *         Default is SAM3_LOG_INFO.
 */
void sam3_log_set_level(enum sam3_log_level level);


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
 * sam3_profile_get - Internal handle to the active profiler for use
 * with the SAM3_PROF_* macros from util/profile.h.
 *
 * @ctx: Initialized context.
 *
 * Returns NULL when profiling is not enabled or the build was compiled
 * without SAM3_HAS_PROFILE. Callers pass the returned pointer into
 * SAM3_PROF_BEGIN/END so external instrumentation (CLIs, tools) can
 * contribute stages to the same report. The pointer is owned by the
 * context; do not free.
 */
struct sam3_profiler;
struct sam3_profiler *sam3_profile_get(sam3_ctx *ctx);

/*
 * sam3_profile_reset - Clear all collected profiling data.
 *
 * @ctx: Initialized context.
 */
void sam3_profile_reset(sam3_ctx *ctx);

/*
 * sam3_dump_tensors - Dump cached image features to binary files.
 *
 * @ctx:      Context with image already encoded (sam3_set_image called)
 * @out_dir:  Directory to write tensor files into (must exist)
 *
 * Writes neck feature tensors: neck_4x.bin, neck_2x.bin, neck_1x.bin,
 * neck_05x.bin. Requires image encoded. Does NOT dump mask/score tensors
 * (those are in sam3_result after segment).
 *
 * Returns SAM3_OK on success, SAM3_EINVAL if image not encoded.
 */
enum sam3_error sam3_dump_tensors(sam3_ctx *ctx, const char *out_dir);


/* --- Video Tracking API --- */

/*
 * sam3_video_object_mask - Per-object segmentation result for one frame.
 *
 * Carries the predicted mask logits, IoU score, and object presence
 * logit for one tracked object on one frame. Heap-allocated; freed
 * by sam3_video_frame_result_free.
 */
struct sam3_video_object_mask {
	int    obj_id;            /* user-supplied obj_id */
	float *mask;              /* [mask_h * mask_w] f32 logits, malloc'd */
	int    mask_h, mask_w;
	float  iou_score;
	float  obj_score_logit;   /* >0 visible, <=0 occluded */
	int    is_occluded;       /* convenience: == (obj_score_logit <= 0) */
};

/*
 * sam3_video_frame_result - Multi-object result for one video frame.
 *
 * Returned by the video propagate callback (Phase 2) and by the
 * prompt entry points after the Phase 2 refactor (n_objects=1 in
 * the prompt-entry-point case).
 */
struct sam3_video_frame_result {
	int frame_idx;
	int n_objects;
	struct sam3_video_object_mask *objects;
};

/*
 * sam3_video_frame_result_free - Release per-frame result memory.
 *
 * Frees each object_mask.mask buffer and the objects array, then
 * zeros the result. Safe to call on a zero-initialized result.
 */
void sam3_video_frame_result_free(struct sam3_video_frame_result *r);

/*
 * sam3_video_start_opts - Tunables for the video session and frame cache.
 *
 * Passed to sam3_video_start_ex. All fields accept 0 / -1 sentinels
 * that select sensible defaults; see field docs.
 */
struct sam3_video_start_opts {
	size_t frame_cache_backend_budget; /* 0 -> 4 GiB default */
	size_t frame_cache_spill_budget;   /* 0 -> 16 GiB default;
					      SIZE_MAX disables spill */
	int    clear_non_cond_window;      /* 0 -> default 7 (= num_maskmem) */
	int    iter_use_prev_mask_pred;    /* -1 -> default on (1) */
	int    multimask_via_stability;    /* -1 -> default on (1) */
	float  multimask_stability_delta;  /* 0.0 -> default 0.05 */
	float  multimask_stability_thresh; /* 0.0 -> default 0.98 */
};

/*
 * sam3_video_start - Begin a video tracking session.
 *
 * @ctx:           Initialized context with loaded model
 * @resource_path: Path to video file or frame directory
 * @out_session:   Receives the new session handle
 *
 * Decodes video frames (or loads frame images from a directory) and
 * lazily encodes frame features on demand via a tiered frame cache.
 * The session holds all tracking state. Call sam3_video_end() to free.
 * Equivalent to calling sam3_video_start_ex with NULL opts.
 *
 * Returns SAM3_OK on success.
 */
enum sam3_error sam3_video_start(sam3_ctx *ctx,
				 const char *resource_path,
				 sam3_video_session **out_session);

/*
 * sam3_video_start_ex - Begin a video session with explicit options.
 *
 * @ctx:           Initialized context with loaded model
 * @resource_path: Path to video file or frame directory
 * @opts:          Session tunables; NULL selects all defaults
 * @out_session:   Receives the new session handle
 *
 * Same contract as sam3_video_start, but accepts a tunables struct.
 * Passing NULL for @opts is equivalent to calling sam3_video_start.
 *
 * Returns SAM3_OK on success.
 */
enum sam3_error sam3_video_start_ex(sam3_ctx *ctx,
				    const char *resource_path,
				    const struct sam3_video_start_opts *opts,
				    sam3_video_session **out_session);

/*
 * sam3_video_add_points - Add point prompts for an object on a frame.
 *
 * @session:  Active video session
 * @frame_idx: Zero-based frame index
 * @obj_id:   Object identifier (user-assigned, 0..SAM3_MAX_OBJECTS-1)
 * @points:   Array of point prompts
 * @n_points: Number of points
 * @result:   Output per-frame result (caller calls sam3_video_frame_result_free).
 *            result->n_objects will be 1; result->objects[0] carries the
 *            mask logits and scores for the prompted object.
 *
 * Runs the prompt encoder and mask decoder for the given frame and
 * object. Multiple calls accumulate prompts for the same object.
 *
 * Returns SAM3_OK on success.
 */
enum sam3_error sam3_video_add_points(sam3_video_session *session,
				      int frame_idx, int obj_id,
				      const struct sam3_point *points,
				      int n_points,
				      struct sam3_video_frame_result *result);

/*
 * sam3_video_add_box - Add a bounding box prompt for an object on a frame.
 *
 * @session:   Active video session
 * @frame_idx: Zero-based frame index
 * @obj_id:    Object identifier (user-assigned, 0..SAM3_MAX_OBJECTS-1)
 * @box:       Bounding box prompt
 * @result:    Output per-frame result (caller calls sam3_video_frame_result_free).
 *             result->n_objects will be 1; result->objects[0] carries the
 *             mask logits and scores for the prompted object.
 *
 * Like sam3_video_add_points but with a box prompt. Runs prompt encoder
 * and mask decoder for the given frame and object.
 *
 * Returns SAM3_OK on success.
 */
enum sam3_error sam3_video_add_box(sam3_video_session *session,
				   int frame_idx, int obj_id,
				   const struct sam3_box *box,
				   struct sam3_video_frame_result *result);

/*
 * sam3_video_add_mask - Add a binary mask prompt for an object on a frame.
 *
 * @session:    Active video session.
 * @frame_idx:  Zero-based frame index.
 * @obj_id:     Object identifier (user-assigned).
 * @mask:       Binary mask in row-major order, [mask_h * mask_w] bytes.
 *              0 = background, non-zero = foreground.
 * @mask_h, @mask_w: Source mask dimensions. Resized nearest-neighbor
 *              to the session's high-res internal size (1152x1152).
 *              Rejected if zero or > 2 * image_size.
 * @result:     Output single-frame single-object result (n_objects=1).
 *              Caller frees with sam3_video_frame_result_free.
 *
 * Bypasses the SAM mask decoder. The mask becomes the segmentation
 * directly: resized, run through the memory encoder, and committed
 * as a conditioning entry to the object's bank. Mirrors Python
 * add_new_mask.
 *
 * Returns SAM3_OK on success.
 */
enum sam3_error sam3_video_add_mask(sam3_video_session *session,
				    int frame_idx, int obj_id,
				    const uint8_t *mask,
				    int mask_h, int mask_w,
				    struct sam3_video_frame_result *result);

/*
 * sam3_video_frame_cb - Per-frame multi-object result callback.
 *
 * @result:   Frame result containing per-object masks. Owned by the engine;
 *            valid only for the duration of the callback. Do NOT call
 *            sam3_video_frame_result_free on it.
 * @user_data: Opaque pointer from sam3_video_propagate.
 *
 * Return 0 to continue propagation, non-zero to stop early.
 */
typedef int (*sam3_video_frame_cb)(
	const struct sam3_video_frame_result *result,
	void *user_data);

/*
 * sam3_video_propagate - Propagate tracked objects across video frames.
 *
 * @session:   Active video session with at least one prompted object
 * @direction: Propagation direction (BOTH, FORWARD, or BACKWARD)
 * @callback:  Per-frame callback to receive results (may be NULL)
 * @user_data: Opaque pointer forwarded to callback
 *
 * Runs the memory attention module to propagate object masks from
 * prompted frames to all other frames in the specified direction.
 * If callback is NULL, results are computed but not reported.
 *
 * Returns SAM3_OK on success, SAM3_EVIDEO on tracking failure.
 */
enum sam3_error sam3_video_propagate(sam3_video_session *session,
				     int direction,
				     sam3_video_frame_cb callback,
				     void *user_data);

/*
 * sam3_video_remove_object - Remove a tracked object from the session.
 *
 * @session: Active video session
 * @obj_id:  Object identifier to remove
 *
 * Removes all prompts and cached masks for the given object.
 * Subsequent propagation will not include this object.
 *
 * Returns SAM3_OK on success, SAM3_EINVAL if obj_id not found.
 */
enum sam3_error sam3_video_remove_object(sam3_video_session *session,
					 int obj_id);

/*
 * sam3_video_reset - Clear all tracked objects and prompts.
 *
 * @session: Active video session
 *
 * Resets tracking state but preserves encoded frame features, so
 * new prompts can be added without re-encoding the video.
 *
 * Returns SAM3_OK on success.
 */
enum sam3_error sam3_video_reset(sam3_video_session *session);

/*
 * sam3_video_end - End a video session and free all resources.
 *
 * @session: Session to free (may be NULL).
 *
 * Releases all frame features, tracking state, and the session handle.
 */
void sam3_video_end(sam3_video_session *session);

/*
 * sam3_video_frame_count - Return the number of frames in the session.
 *
 * @session: Active video session.
 *
 * Returns the frame count, or 0 if session is NULL.
 */
int sam3_video_frame_count(const sam3_video_session *session);

#endif /* SAM3_H */
