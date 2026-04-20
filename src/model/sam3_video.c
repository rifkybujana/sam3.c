/*
 * src/model/sam3_video.c - Public video tracking API implementation
 *
 * Wires the public sam3_video_* functions declared in sam3/sam3.h to the
 * internal session, tracker, and video I/O modules. Handles session
 * lifecycle (start/end), prompt registration (add_points/add_box),
 * mask propagation, and object management. Arena allocation follows the
 * same pattern as sam3_processor: a persist arena for session-lifetime
 * data and a scratch arena reset between frames. The persist arena also
 * backs the stored prompt list and the prompted-frame bitmap.
 *
 * Key types:  sam3_video_session, sam3_ctx
 * Depends on: sam3/sam3.h, model/sam3_internal.h, model/sam3_image.h,
 *             model/sam3_processor.h, model/video_session.h,
 *             model/tracker.h, model/graph_helpers.h, model/frame_cache.h,
 *             util/video.h, core/alloc.h, core/tensor.h, core/graph.h,
 *             util/log.h
 * Used by:    user applications via sam3.h
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sam3/sam3.h"
#include "model/sam3_internal.h"
#include "model/sam3_image.h"
#include "model/sam3_processor.h"
#include "model/video_session.h"
#include "model/tracker.h"
#include "model/tracker_multiplex.h"
#include "model/mask_decoder.h"
#include "model/graph_helpers.h"
#include "model/frame_cache.h"
#include "util/video.h"
#include "core/alloc.h"
#include "core/tensor.h"
#include "core/graph.h"
#include "util/log.h"
#include "util/profile.h"
#include "util/time.h"

/* Arena capacities for video sessions */
#define VIDEO_PERSIST_SIZE  (4UL << 30)  /* 4 GiB */
#define VIDEO_SCRATCH_SIZE  (1UL << 28)  /* 256 MiB */

/*
 * Python NO_OBJ_SCORE sentinel used when the object is not visible.
 * Matches sam3_tracker_base.py:23.
 */
#define SAM3_NO_OBJ_SCORE (-1024.0f)

#ifdef SAM3_DEBUG_DUMP
/* Filled by sam3_mask_decoder_build in debug mode to let us dump
 * the pixel-decoder chain intermediate tensors. */
struct sam3_tensor *sam3_dbg_pixel_out      = NULL;
struct sam3_tensor *sam3_dbg_pix_ct1        = NULL;
struct sam3_tensor *sam3_dbg_pix_ct1_input  = NULL;
struct sam3_tensor *sam3_dbg_pix_skip1      = NULL;
struct sam3_tensor *sam3_dbg_pix_after_skip1 = NULL;
struct sam3_tensor *sam3_dbg_xformer_in_img = NULL;
struct sam3_tensor *sam3_dbg_xformer_in_tokens = NULL;
struct sam3_tensor *sam3_dbg_xformer_layer0_q = NULL;

/* Video-path tensor dumper. Used for layer-by-layer parity diffs
 * against the Python reference via scripts/dump_reference_layers.py
 * and scripts/compare_layer_dumps.py. */
static void auto_dump_tensor(const char *path, const struct sam3_tensor *t)
{
	if (!t) return;
	FILE *fp = fopen(path, "wb");
	if (!fp) return;
	int n = 1;
	for (int i = 0; i < t->n_dims; i++)
		n *= t->dims[i];
	fwrite(t->data, sizeof(float), (size_t)n, fp);
	fclose(fp);
	sam3_log_info("dump: %s (%d floats, shape [%d,%d,%d,%d])",
		      path, n,
		      t->dims[0], t->dims[1], t->dims[2], t->dims[3]);
}
#endif

/*
 * apply_occlusion_gating - Post-eval gating matching Python semantics.
 *
 * Python hard threshold: is_obj_appearing = (score > 0).
 * When the object is NOT appearing (score <= 0):
 *   - masks     -> SAM3_NO_OBJ_SCORE everywhere
 *   - obj_ptr   -> broadcast no_obj_ptr over all rows
 *   - mem_feat  -> add no_obj_embed_spatial per-channel
 *
 * All tensors must be materialised (graph-evaluated) F32. Any input
 * tensor may be NULL to skip its gating step. Returns the indicator
 * (0 or 1) so the caller can reuse it for metrics.
 */
static int apply_occlusion_gating(
	struct sam3_tensor *masks,
	struct sam3_tensor *obj_ptr,
	struct sam3_tensor *mem_feat,
	const struct sam3_tensor *obj_score_logit,
	const struct sam3_tensor *no_obj_ptr,
	const struct sam3_tensor *no_obj_embed_spatial)
{
	if (!obj_score_logit || !obj_score_logit->data)
		return 1; /* Unknown — assume appearing (backwards compat) */

	float score = ((const float *)obj_score_logit->data)[0];
	if (score > 0.0f)
		return 1;

	if (masks && masks->data) {
		float *m = (float *)masks->data;
		int n = sam3_tensor_nelems(masks);
		for (int i = 0; i < n; i++)
			m[i] = SAM3_NO_OBJ_SCORE;
	}
	if (obj_ptr && obj_ptr->data && no_obj_ptr && no_obj_ptr->data) {
		float *dst = (float *)obj_ptr->data;
		const float *src = (const float *)no_obj_ptr->data;
		int rows = obj_ptr->n_dims >= 1 ? obj_ptr->dims[0] : 1;
		int cols = obj_ptr->n_dims >= 2 ? obj_ptr->dims[1] :
				 sam3_tensor_nelems(obj_ptr);
		int src_cols = no_obj_ptr->n_dims >= 2 ?
				 no_obj_ptr->dims[1] :
				 sam3_tensor_nelems(no_obj_ptr);
		if (cols == src_cols) {
			for (int r = 0; r < rows; r++)
				memcpy(dst + (size_t)r * cols, src,
				       (size_t)cols * sizeof(float));
		}
	}
	if (mem_feat && mem_feat->data && mem_feat->n_dims == 4 &&
	    no_obj_embed_spatial && no_obj_embed_spatial->data) {
		float *f = (float *)mem_feat->data;
		const float *add = (const float *)no_obj_embed_spatial->data;
		int H = mem_feat->dims[1];
		int W = mem_feat->dims[2];
		int C = mem_feat->dims[3];
		int add_n = sam3_tensor_nelems(no_obj_embed_spatial);
		if (add_n == C) {
			int HW = H * W;
			for (int i = 0; i < HW; i++)
				for (int c = 0; c < C; c++)
					f[(size_t)i * C + c] += add[c];
		}
	}
	return 0;
}

/*
 * preprocess_mask_for_mem_enc - Python mask_for_mem preprocessing.
 *
 * Matches sam3_tracker_base.py:820-830 (with skip_mask_sigmoid=True
 * downstream):
 *
 *   if is_mask_from_pts and not training:
 *       mask_for_mem = (mask > 0).float()
 *   else:
 *       mask_for_mem = sigmoid(mask)
 *   mask_for_mem = mask_for_mem * sigmoid_scale_for_mem_enc
 *                + sigmoid_bias_for_mem_enc
 *
 * Applied in-place on an F32 NHWC tensor before passing to
 * sam3_memory_encoder_build, which no longer applies sigmoid internally.
 * `is_mask_from_pts` should be 1 for cond frames where the user
 * supplied prompts on this frame and 0 for propagation frames.
 */
static void preprocess_mask_for_mem_enc(
	struct sam3_tensor *mask,
	int is_mask_from_pts,
	float sigmoid_scale,
	float sigmoid_bias)
{
	if (!mask || !mask->data)
		return;

	float *data = (float *)mask->data;
	int n = sam3_tensor_nelems(mask);

	if (is_mask_from_pts) {
		float one_val  = sigmoid_scale + sigmoid_bias;
		float zero_val = sigmoid_bias;
		for (int i = 0; i < n; i++)
			data[i] = (data[i] > 0.0f) ? one_val : zero_val;
	} else {
		for (int i = 0; i < n; i++) {
			float s = 1.0f / (1.0f + expf(-data[i]));
			data[i] = s * sigmoid_scale + sigmoid_bias;
		}
	}
}

/*
 * compute_eff_iou_score - Python cal_mem_score for memory selection.
 *
 *   object_score_norm = score > 0 ? sigmoid(score) * 2 - 1 : 0
 *   eff_iou           = object_score_norm * iou_score
 *
 * Returns the raw iou_score if obj_score_logit is unavailable.
 */
static float compute_eff_iou_score(
	const struct sam3_tensor *obj_score_logit,
	float iou_score)
{
	if (!obj_score_logit || !obj_score_logit->data)
		return iou_score;
	float s = ((const float *)obj_score_logit->data)[0];
	if (s <= 0.0f)
		return 0.0f;
	float norm = 2.0f / (1.0f + expf(-s)) - 1.0f;
	return norm * iou_score;
}

/*
 * apply_opts_defaults - Fill zero/sentinel fields with sensible defaults.
 *
 * Called after copying the caller's opts struct (or a zeroed struct when
 * opts is NULL) into session->opts.
 */
static void
apply_opts_defaults(struct sam3_video_start_opts *o)
{
	if (o->clear_non_cond_window <= 0)
		o->clear_non_cond_window = 7;
	if (o->iter_use_prev_mask_pred < 0)
		o->iter_use_prev_mask_pred = 1;
	if (o->multimask_via_stability < 0)
		o->multimask_via_stability = 1;
	if (o->multimask_stability_delta == 0.0f)
		o->multimask_stability_delta = 0.05f;
	if (o->multimask_stability_thresh == 0.0f)
		o->multimask_stability_thresh = 0.98f;
}

/*
 * session_encode_frame - Frame cache miss hook.
 *
 * Runs the image encoder for one frame and clones the three NHWC feature
 * tensors into @arena. Called by sam3_frame_cache_get on cache misses.
 *
 * @session:   Owning video session (for ctx and frames access).
 * @frame_idx: Zero-based frame to encode.
 * @arena:     Bump arena for tensor allocation (cache's backend arena).
 * @out:       Receives pointers to the three encoded feature tensors.
 *
 * Returns SAM3_OK on success; SAM3_EINVAL on bad arguments;
 * SAM3_ENOMEM if tensor cloning fails.
 */
static enum sam3_error
session_encode_frame(struct sam3_video_session *session,
		     int frame_idx,
		     struct sam3_arena *arena,
		     struct sam3_frame_features *out)
{
	if (!session || !arena || !out)
		return SAM3_EINVAL;

	struct sam3_ctx *ctx = session->ctx;
	if (!ctx || !ctx->proc_ready || !ctx->proc.backend)
		return SAM3_EINVAL;

	if (frame_idx < 0 || frame_idx >= session->frames.n_frames)
		return SAM3_EINVAL;

	struct sam3_tensor *frame = session->frames.pixels[frame_idx];
	if (!frame) {
		sam3_log_error("session_encode_frame: frame %d pixels NULL",
			       frame_idx);
		return SAM3_EINVAL;
	}

	SAM3_PROF_BEGIN(ctx->proc.profiler, "video_encode_frame");

	/*
	 * Roll back the model_arena to the post-weight-load offset before
	 * each frame encode. Without this, the cached features from prior
	 * frames pile up in the persist arena and eventually exhaust it.
	 * The frame_cache has already cloned the relevant tensors into its
	 * own backend arena, so model_arena's per-frame outputs are safe to
	 * discard.
	 */
	if (ctx->proc.model_arena.offset > ctx->proc.weights_end) {
		char *base = (char *)ctx->proc.model_arena.base;
		size_t old_off = ctx->proc.model_arena.offset;
		ctx->proc.model_arena.offset = ctx->proc.weights_end;
		if (ctx->proc.backend->ops->cache_invalidate) {
			ctx->proc.backend->ops->cache_invalidate(
				ctx->proc.backend,
				base + ctx->proc.weights_end,
				old_off - ctx->proc.weights_end);
		}
	}

	/* Reset scratch arena so the per-frame ViT/neck eval gets a clean
	 * working area. (proc.set_image resets it for the single-image
	 * path; the video path encodes via session_encode_frame which
	 * bypasses that helper.) */
	sam3_arena_reset(&ctx->proc.scratch_arena);

	enum sam3_error err =
		sam3_image_model_encode(&ctx->proc.model,
					ctx->proc.backend,
					frame,
					&ctx->proc.scratch_arena,
					&ctx->proc.model_arena,
					ctx->proc.profiler);
	if (err != SAM3_OK) {
		sam3_log_error("session_encode_frame: encode frame %d "
			       "failed (%d)", frame_idx, err);
		SAM3_PROF_END(ctx->proc.profiler, "video_encode_frame");
		return err;
	}

	/*
	 * Prefer the sam2-side neck features when available. The video
	 * tracker's SAM mask decoder consumes sam2 features as its image
	 * embeddings (Python pix_feat_with_mem path). Fall back to the
	 * sam3-side features for older checkpoints where the sam2 neck
	 * isn't present.
	 */
	struct sam3_tensor *src_05x  = ctx->proc.model.cached_sam2_05x_nhwc
		? ctx->proc.model.cached_sam2_05x_nhwc
		: ctx->proc.model.cached_image_features;
	struct sam3_tensor *src_2x   = ctx->proc.model.cached_sam2_2x_nhwc
		? ctx->proc.model.cached_sam2_2x_nhwc
		: ctx->proc.model.cached_feat_s0_nhwc;
	struct sam3_tensor *src_1x   = ctx->proc.model.cached_sam2_1x_nhwc
		? ctx->proc.model.cached_sam2_1x_nhwc
		: ctx->proc.model.cached_feat_s1_nhwc;
	struct sam3_tensor *src_4x   = ctx->proc.model.cached_sam2_4x_nhwc
		? ctx->proc.model.cached_sam2_4x_nhwc
		: ctx->proc.model.cached_feat_4x_nhwc;

	/*
	 * image_features (0.5x) is only populated for SAM 3's dual-neck
	 * checkpoints (where the sam2 neck contributes a 4th FPN scale).
	 * SAM 3.1 ships a tri-neck with no 0.5x output, and its tracker_multiplex
	 * reads feat_s1 (1x) as the main backbone embedding — the 0.5x slot
	 * stays NULL and is not consumed downstream. Tolerate that here.
	 */
	out->image_features = src_05x
		? sam3_tensor_clone_persist(arena, src_05x)
		: NULL;
	out->feat_s0 = sam3_tensor_clone_persist(arena, src_2x);
	out->feat_s1 = sam3_tensor_clone_persist(arena, src_1x);
	/* feat_4x (288x288 4x skip) tolerates NULL for older models. */
	out->feat_4x = src_4x
		? sam3_tensor_clone_persist(arena, src_4x)
		: NULL;

	if (!out->feat_s0 || !out->feat_s1) {
		sam3_log_error("session_encode_frame: clone frame %d failed "
			       "(feat_s0=%p feat_s1=%p)",
			       frame_idx, (void *)out->feat_s0,
			       (void *)out->feat_s1);
		SAM3_PROF_END(ctx->proc.profiler, "video_encode_frame");
		return SAM3_ENOMEM;
	}
	if (!out->image_features &&
	    session->variant != SAM3_VARIANT_SAM3_1) {
		sam3_log_error("session_encode_frame: image_features missing "
			       "on non-SAM3.1 frame %d", frame_idx);
		SAM3_PROF_END(ctx->proc.profiler, "video_encode_frame");
		return SAM3_ENOMEM;
	}
	/*
	 * SAM 3.1's multiplex tracker reads feat_4x as the decoder's skip feed —
	 * reject clone failures here instead of silently returning a NULL
	 * feat_4x that trips the multiplex pipeline later.
	 */
	if (!out->feat_4x && src_4x) {
		sam3_log_error("session_encode_frame: feat_4x clone failed "
			       "frame %d (arena exhausted?)", frame_idx);
		SAM3_PROF_END(ctx->proc.profiler, "video_encode_frame");
		return SAM3_ENOMEM;
	}
	SAM3_PROF_END(ctx->proc.profiler, "video_encode_frame");
	return SAM3_OK;
}

/*
 * sam3_video_start_ex - Begin a video tracking session with explicit options.
 *
 * Allocates a session, initializes arenas, loads tracker weights, loads
 * video frames, and initializes the tiered frame cache. Frames are encoded
 * lazily on first access rather than eagerly up front. The ctx must have a
 * loaded model (sam3_load_model called successfully).
 *
 * @ctx:           Initialized context with loaded model
 * @resource_path: Path to video file or frame directory
 * @opts:          Tunables; NULL selects all defaults
 * @out_session:   Receives the new session handle
 */
enum sam3_error sam3_video_start_ex(sam3_ctx *ctx,
				    const char *resource_path,
				    const struct sam3_video_start_opts *opts,
				    sam3_video_session **out_session)
{
	struct sam3_video_session *session = NULL;
	enum sam3_error err;

	if (!ctx || !resource_path || !out_session) {
		sam3_log_error("video_start: NULL argument");
		return SAM3_EINVAL;
	}

	*out_session = NULL;

	if (!ctx->loaded || !ctx->proc_ready) {
		sam3_log_error("video_start: model not loaded");
		return SAM3_EINVAL;
	}

	SAM3_PROF_BEGIN(ctx->proc.profiler, "video_start");

	session = calloc(1, sizeof(*session));
	if (!session) {
		sam3_log_error("video_start: session alloc failed");
		return SAM3_ENOMEM;
	}

	/* Store opts (or zeros) and fill defaults before any subsystem
	 * uses them. session->ctx is set later, after all arenas/weights
	 * are ready, so the cache init (which sets owner) comes after. */
	if (opts)
		session->opts = *opts;
	else
		memset(&session->opts, 0, sizeof(session->opts));
	apply_opts_defaults(&session->opts);

	/* Initialize persist arena */
	err = sam3_arena_init(&session->persist, VIDEO_PERSIST_SIZE);
	if (err != SAM3_OK) {
		sam3_log_error("video_start: persist arena init failed");
		goto cleanup;
	}

	/* Initialize scratch arena */
	err = sam3_arena_init(&session->scratch, VIDEO_SCRATCH_SIZE);
	if (err != SAM3_OK) {
		sam3_log_error("video_start: scratch arena init failed");
		goto cleanup;
	}

	/* Initialize and load the variant-specific tracker. */
	session->variant = ctx->config.variant;
	if (session->variant == SAM3_VARIANT_SAM3_1) {
		err = sam3_tracker_multiplex_init(&session->tracker_multiplex);
		if (err != SAM3_OK) {
			sam3_log_error("video_start: tracker_multiplex init failed (%d)",
				       err);
			goto cleanup;
		}
		err = sam3_tracker_multiplex_load(&session->tracker_multiplex, &ctx->weights,
					   &session->persist);
		if (err != SAM3_OK) {
			sam3_log_error("video_start: tracker_multiplex load failed (%d)",
				       err);
			goto cleanup;
		}
	} else {
		err = sam3_tracker_init(&session->tracker);
		if (err != SAM3_OK) {
			sam3_log_error("video_start: tracker init failed (%d)",
				       err);
			goto cleanup;
		}
		err = sam3_tracker_load(&session->tracker, &ctx->weights,
					&session->persist);
		if (err != SAM3_OK) {
			sam3_log_error("video_start: tracker load failed (%d)",
				       err);
			goto cleanup;
		}
	}

	/* Load video frames */
	err = sam3_video_load(resource_path, ctx->config.image_size,
			      &session->frames, &session->persist);
	if (err != SAM3_OK) {
		sam3_log_error("video_start: frame load failed (%d)", err);
		goto cleanup;
	}

	int nf = session->frames.n_frames;
	if (nf <= 0) {
		sam3_log_error("video_start: no frames loaded");
		err = SAM3_EVIDEO;
		goto cleanup;
	}

	/* Allocate per-frame tracking status array */
	session->frames_tracked = sam3_arena_alloc(
		&session->persist, (size_t)nf * sizeof(int));
	if (!session->frames_tracked) {
		sam3_log_error("video_start: frames_tracked alloc failed");
		err = SAM3_ENOMEM;
		goto cleanup;
	}

	/* Prompted-frame bitmap (1 byte per frame).
	 * sam3_arena_alloc() zero-fills, so no explicit memset needed. */
	session->prompted_frames = sam3_arena_alloc(&session->persist,
						    (size_t)nf);
	if (!session->prompted_frames) {
		sam3_log_error("video_start: prompted_frames alloc failed");
		err = SAM3_ENOMEM;
		goto cleanup;
	}

	/* Prompt list: bounded by SAM3_MAX_OBJECTS * nf. Clamp to INT_MAX
	 * so cap_prompts (stored as int) never truncates. */
	size_t cap = (size_t)SAM3_MAX_OBJECTS * (size_t)nf;
	if (cap > (size_t)INT_MAX)
		cap = (size_t)INT_MAX;
	session->prompts = sam3_arena_alloc(
		&session->persist, cap * sizeof(*session->prompts));
	if (!session->prompts) {
		sam3_log_error("video_start: prompts alloc failed");
		err = SAM3_ENOMEM;
		goto cleanup;
	}
	session->cap_prompts = (int)cap;
	session->n_prompts   = 0;

	/*
	 * Assign ctx before cache init: the encode hook (session_encode_frame)
	 * reads session->ctx to reach the image encoder. ctx must be valid
	 * before the first cache miss can occur.
	 */
	session->ctx = ctx;

	/*
	 * Initialize the tiered frame cache. Frames are encoded lazily on
	 * first access by session_encode_frame; no eager encode loop.
	 */
	err = sam3_frame_cache_init(
		&session->frame_cache, session, session_encode_frame,
		nf, session->opts.frame_cache_backend_budget,
		session->opts.frame_cache_spill_budget);
	if (err != SAM3_OK) {
		sam3_log_error("video_start: frame_cache_init failed (%d)",
			       err);
		goto cleanup;
	}

	session->loaded = 1;
	*out_session = session;

	sam3_log_info("video session started (%d frames, %dx%d)",
		      nf, session->frames.frame_size,
		      session->frames.frame_size);
	SAM3_PROF_END(ctx->proc.profiler, "video_start");
	return SAM3_OK;

cleanup:
	if (session) {
		sam3_frame_cache_release(&session->frame_cache);
		sam3_arena_free(&session->scratch);
		sam3_arena_free(&session->persist);
		free(session);
	}
	SAM3_PROF_END(ctx->proc.profiler, "video_start");
	return err;
}

/*
 * sam3_video_start - Begin a video tracking session.
 *
 * Thin wrapper that calls sam3_video_start_ex with NULL opts (all defaults).
 */
enum sam3_error sam3_video_start(sam3_ctx *ctx,
				 const char *resource_path,
				 sam3_video_session **out_session)
{
	return sam3_video_start_ex(ctx, resource_path, NULL, out_session);
}

/*
 * propagate_frame_ctx - Per-frame shared state for multi-object
 *                       propagation. Built once in propagate_one and
 *                       passed to each per-object call so the backbone
 *                       feature flatten runs only once per frame instead
 *                       of once per object. NULL means "no sharing, do
 *                       the per-object work locally" (legacy path used
 *                       by add_points / add_box).
 *
 * @cf:           Frame cache features (result of sam3_frame_cache_get).
 * @img_2d:       [HW, d] flatten of cf.feat_s1 for the tracker graph.
 * @grid_h/grid_w: Spatial dims of cf.feat_s1 (e.g. 72x72 on Hiera).
 * @scratch_mark: Scratch arena offset immediately after @img_2d. Each
 *                per-object call resets the scratch to this mark instead
 *                of 0, so the shared img_2d survives across objects
 *                while per-object intermediates (prompt tokens, graph
 *                scratch) are recycled between iterations.
 */
struct propagate_frame_ctx {
	struct sam3_frame_features cf;
	struct sam3_tensor *img_2d;
	int grid_h;
	int grid_w;
	size_t scratch_mark;
};

/*
 * video_track_one_obj - Run the per-frame tracker pipeline for one object.
 *
 * Shared pipeline used by every (variant, is_cond) combination. Runs the
 * variant's tracker + memory encoder, commits a memory-bank entry, and
 * fills @obj_mask with the selected multimask prediction. Variant-specific
 * logic is bounded to the tracker call (step 4) and the memory encoder
 * (step 8); everything else is variant-agnostic. See pipeline steps in
 * the function body for the exact flow.
 *
 * @session:        Video session (ctx must be ready; caller pre-validated).
 * @obj_idx:        Internal object index into session->objects[].
 * @frame_idx:      Frame to process.
 * @is_cond:        1 for conditioning frames (add_points / add_box);
 *                  0 for propagation.
 * @prompts_opt:    Public prompt array — consumed only by the SAM 3
 *                  conditioning path. Pass NULL for propagation and for
 *                  all SAM 3.1 calls.
 * @n_prompts_opt:  Length of @prompts_opt. 0 when @prompts_opt is NULL.
 * @shared:         Optional per-frame shared state from propagate_one;
 *                  NULL selects the per-call path (used by add_points /
 *                  add_box). When non-NULL, @shared->cf and the arena
 *                  mark are reused, and for SAM 3 the img_2d flatten is
 *                  shared across objects on the same frame.
 * @obj_mask:       Caller-owned output. Zeroed before work starts; on
 *                  success @obj_mask->mask is heap-allocated and owned
 *                  by the caller.
 *
 * Returns SAM3_OK on success. On error returns SAM3_EINVAL, SAM3_ENOMEM,
 * or a backend error; @obj_mask->mask is left NULL and counter fields
 * zeroed so callers can call free(obj_mask->mask) unconditionally.
 *
 * The bank entry is committed before this function returns; callers that
 * need to roll it back must do so explicitly (see
 * video_add_prompts_pipeline's capacity pre-check).
 */
static enum sam3_error
video_track_one_obj(struct sam3_video_session *session,
		    int obj_idx, int frame_idx, int is_cond,
		    const struct sam3_prompt *prompts_opt,
		    int n_prompts_opt,
		    const struct propagate_frame_ctx *shared,
		    struct sam3_video_object_mask *obj_mask)
{
	struct sam3_ctx *ctx = session->ctx;
	if (!ctx || !ctx->proc_ready || !ctx->proc.backend) {
		sam3_log_error("video_track: context not ready");
		return SAM3_EINVAL;
	}

	/*
	 * Invariant: `trk` and `trk_mux` are mutually exclusive — exactly
	 * one is non-NULL based on `is_multiplex`. Every variant-dispatch branch
	 * below relies on this to dereference the right tracker without
	 * another NULL check.
	 */
	const int is_multiplex = (session->variant == SAM3_VARIANT_SAM3_1);
	struct sam3_tracker     *trk    = is_multiplex ? NULL : &session->tracker;
	struct sam3_tracker_multiplex  *trk_mux = is_multiplex ? &session->tracker_multiplex : NULL;
	struct sam3_memory_bank *bank   = &session->objects[obj_idx].bank;

	SAM3_PROF_BEGIN(ctx->proc.profiler,
			is_cond ? "video_track_cond" : "video_track_prop");

	enum sam3_error err = SAM3_OK;
	struct sam3_frame_features cf;
	struct sam3_tensor *spatial_persist = NULL;
	struct sam3_tensor *obj_ptr_persist = NULL;

	/* --- 1. Features ------------------------------------------------- */
	if (shared) {
		cf = shared->cf;
	} else {
		memset(&cf, 0, sizeof(cf));
		SAM3_PROF_BEGIN(ctx->proc.profiler, "frame_cache_get");
		err = sam3_frame_cache_get(&session->frame_cache,
					   frame_idx, &cf);
		SAM3_PROF_END(ctx->proc.profiler, "frame_cache_get");
		if (err != SAM3_OK) {
			sam3_log_error("video_track: frame %d cache get "
				       "failed (%d)", frame_idx, err);
			goto cleanup;
		}
	}
	if (!cf.feat_s0 || !cf.feat_s1 ||
	    (is_multiplex && !cf.feat_4x) ||
	    (!is_multiplex && !cf.image_features)) {
		sam3_log_error("video_track: frame %d missing features "
			       "(variant=%d)", frame_idx, session->variant);
		err = SAM3_EINVAL;
		goto cleanup;
	}

	/* --- 2. Clear stale non-cond memory on re-prompt ---------------- */
	if (is_cond) {
		int w = session->opts.clear_non_cond_window;
		if (w <= 0)
			w = 7;
		sam3_memory_bank_clear_around_frame(bank, frame_idx, w);
	}

	/* --- 3. Scratch arena ------------------------------------------- */
	/*
	 * Use the processor's large scratch arena for tracker + memory-
	 * encoder intermediates — the session scratch is too small. When
	 * propagate_one has flattened feat_s1 into a shared img_2d, reset
	 * to the post-flatten mark so the shared tensor survives across
	 * objects while per-object work recycles.
	 */
	struct sam3_arena *gfx = &ctx->proc.scratch_arena;
	if (shared)
		gfx->offset = shared->scratch_mark;
	else
		sam3_arena_reset(gfx);

	struct sam3_graph g;
	sam3_graph_init(&g);

	/* --- 4. Variant-dispatched tracker call ------------------------- */
	struct sam3_tensor *track_masks   = NULL; /* [N_mask, H, W]            */
	struct sam3_tensor *track_iou     = NULL; /* [N_mask]                  */
	struct sam3_tensor *track_obj_ptr = NULL; /* sam3:[1,256] mux:[N_mask,256] */
	struct sam3_tensor *track_score   = NULL; /* [1]                        */

	SAM3_PROF_BEGIN(ctx->proc.profiler, "tracker_build");
	if (is_multiplex) {
		err = sam3_tracker_multiplex_track_frame(
			trk_mux, &g, bank,
			cf.feat_s1,   /* 1x = main grid, 72x72 */
			cf.feat_s0,   /* 2x skip, 144x144       */
			cf.feat_4x,   /* 4x skip, 288x288       */
			frame_idx, is_cond, gfx,
			&track_masks, &track_iou,
			&track_obj_ptr, &track_score);
	} else {
		/*
		 * SAM 3 takes sparse prompt tokens only on conditioning
		 * frames. Pass NULL on propagation.
		 */
		struct sam3_tensor *prompt_tokens = NULL;
		if (is_cond && prompts_opt && n_prompts_opt > 0) {
			int frame_size = session->frames.frame_size;
			SAM3_PROF_BEGIN(ctx->proc.profiler, "prompt_project");
			prompt_tokens = sam3_tracker_sam_project_prompts(
				trk, prompts_opt, n_prompts_opt,
				frame_size, frame_size, gfx);
			if (!prompt_tokens) {
				prompt_tokens = sam3_project_prompts(
					&ctx->proc.model, cf.feat_s1,
					prompts_opt, n_prompts_opt,
					frame_size, frame_size, gfx);
			}
			SAM3_PROF_END(ctx->proc.profiler, "prompt_project");
			if (!prompt_tokens) {
				sam3_log_error("video_track: prompt projection "
					       "failed (frame %d, obj %d)",
					       frame_idx, obj_idx);
				err = SAM3_ENOMEM;
				goto cleanup;
			}
		}

		/*
		 * SAM 3's tracker takes the 1x feature as a 2D tensor
		 * [HW, d]. Reuse the flatten propagate_one prepared
		 * across objects when available.
		 */
		struct sam3_tensor *img_2d = NULL;
		int grid_h = 0, grid_w = 0;
		if (shared) {
			img_2d = shared->img_2d;
			grid_h = shared->grid_h;
			grid_w = shared->grid_w;
		} else {
			struct sam3_tensor *img = cf.feat_s1;
			if (img->n_dims != 4) {
				sam3_log_error("video_track: feat_s1 not 4D");
				err = SAM3_EINVAL;
				goto cleanup;
			}
			grid_h = img->dims[1];
			grid_w = img->dims[2];
			int d2[2] = {grid_h * grid_w, img->dims[3]};
			img_2d = gh_alloc_tensor(gfx, img->dtype, 2, d2);
			if (!img_2d) {
				err = SAM3_ENOMEM;
				goto cleanup;
			}
			memcpy(img_2d->data, img->data, img->nbytes);
		}

		err = sam3_tracker_track_frame(
			trk, &g, bank,
			img_2d, grid_h, grid_w,
			prompt_tokens,
			cf.feat_4x, cf.feat_s0,
			frame_idx, is_cond, gfx,
			&track_masks, &track_iou,
			&track_obj_ptr, &track_score);
	}
	SAM3_PROF_END(ctx->proc.profiler, "tracker_build");

	if (err != SAM3_OK) {
		sam3_log_error("video_track: tracker build failed "
			       "(frame %d, obj %d, err %d)",
			       frame_idx, obj_idx, err);
		goto cleanup;
	}

	SAM3_PROF_BEGIN(ctx->proc.profiler, "tracker_eval");
	err = ctx->proc.backend->ops->graph_eval(ctx->proc.backend, &g);
	SAM3_PROF_END(ctx->proc.profiler, "tracker_eval");
	if (err != SAM3_OK) {
		sam3_log_error("video_track: tracker eval failed "
			       "(frame %d, obj %d, err %d)",
			       frame_idx, obj_idx, err);
		goto cleanup;
	}

	if (!track_masks || track_masks->n_dims != 3) {
		sam3_log_error("video_track: bad track_masks shape");
		err = SAM3_EINVAL;
		goto cleanup;
	}
	const int n_masks = track_masks->dims[0];
	const int final_h = track_masks->dims[1];
	const int final_w = track_masks->dims[2];

	/* --- 5. Best mask pick ----------------------------------------- */
	int best_idx = 0;
	if (track_iou && track_iou->n_dims >= 1 &&
	    track_iou->dims[0] >= n_masks) {
		if (is_multiplex) {
			/*
			 * Argmax IoU. Stability-aware selection for SAM 3.1
			 * lives in sub-project 3's interactive decoder once
			 * real sparse prompts drive the multimask logits.
			 */
			const float *io = (const float *)track_iou->data;
			float best = io[0];
			for (int i = 1; i < n_masks; i++) {
				if (io[i] > best) {
					best = io[i];
					best_idx = i;
				}
			}
		} else {
			float delta  = session->opts.multimask_via_stability
				       ? session->opts.multimask_stability_delta
				       : 0.0f;
			float thresh = session->opts.multimask_via_stability
				       ? session->opts.multimask_stability_thresh
				       : 0.0f;
			best_idx = sam3_mask_decoder_select_with_stability(
				(const float *)track_masks->data,
				(const float *)track_iou->data,
				n_masks, final_h, final_w, delta, thresh);
		}
	}

	const float best_iou_value = (track_iou && track_iou->n_dims >= 1 &&
				      track_iou->dims[0] > best_idx)
		? ((const float *)track_iou->data)[best_idx]
		: 1.0f;

	/* --- 6. Stash prev_mask (iter_use_prev_mask_pred on cond) ------- */
	if (is_cond && session->opts.iter_use_prev_mask_pred &&
	    track_masks->dims[0] > best_idx) {
		int slice_dims[3] = {1, final_h, final_w};
		struct sam3_tensor *prev = gh_alloc_tensor(
			&session->persist, SAM3_DTYPE_F32, 3, slice_dims);
		if (prev) {
			size_t per_mask =
				(size_t)final_h * final_w * sizeof(float);
			memcpy(prev->data,
			       (const float *)track_masks->data
			         + (size_t)best_idx * final_h * final_w,
			       per_mask);
			session->objects[obj_idx].prev_mask_logits = prev;
			session->objects[obj_idx].prev_mask_frame  = frame_idx;
		} else {
			sam3_log_warn("video_track: prev_mask clone failed "
				      "(frame %d, obj %d)", frame_idx, obj_idx);
		}
	}

	/*
	 * --- 7. Occlusion gating ----------------------------------------
	 *
	 * SAM 3: apply_occlusion_gating broadcasts no_obj_ptr over the
	 *   full [1, 256] obj_ptr tensor in-place; the gated tensor is
	 *   what we memcpy into the persist obj_ptr below.
	 *
	 * SAM 3.1: obj_ptr is [N_mask, 256] and we pick best_idx after the
	 *   fact, so we skip the obj_ptr broadcast here and zero the
	 *   persist copy when not appearing in step 8. The mask gating to
	 *   NO_OBJ_SCORE is identical.
	 */
	struct sam3_tensor *no_obj_ptr = is_multiplex
		? NULL
		: trk->no_obj_ptr;
	struct sam3_tensor *no_obj_embed = is_multiplex
		? NULL  /* multiplex no_obj_embed_spatial is [16, 256], applied later */
		: trk->no_obj_embed_spatial;
	int is_appearing = apply_occlusion_gating(
		track_masks,
		is_multiplex ? NULL : track_obj_ptr,
		NULL, track_score, no_obj_ptr, no_obj_embed);

	/* --- 8. Persist the best obj_ptr ------------------------------- */
	struct sam3_tensor *best_obj_ptr_scratch = NULL;
	{
		int op_dims[2] = {1, SAM3_MULTIPLEX_HIDDEN_DIM};
		best_obj_ptr_scratch = gh_alloc_tensor(gfx, SAM3_DTYPE_F32,
						      2, op_dims);
		if (!best_obj_ptr_scratch) {
			err = SAM3_ENOMEM;
			goto cleanup;
		}
		const size_t D = (size_t)SAM3_MULTIPLEX_HIDDEN_DIM * sizeof(float);
		if (!is_multiplex) {
			/* Already pre-selected by sam3_tracker_track_frame
			 * and gated in-place by apply_occlusion_gating. */
			memcpy(best_obj_ptr_scratch->data,
			       track_obj_ptr->data, D);
		} else if (is_appearing) {
			memcpy(best_obj_ptr_scratch->data,
			       (const float *)track_obj_ptr->data
			         + (size_t)best_idx * SAM3_MULTIPLEX_HIDDEN_DIM,
			       D);
		} else {
			memset(best_obj_ptr_scratch->data, 0, D);
		}
	}

	/* --- 9. Best mask → [1, H, W, 1] NHWC scratch tensor ----------- */
	struct sam3_tensor *best_mask_nhwc;
	{
		int dims_nhwc[4] = {1, final_h, final_w, 1};
		best_mask_nhwc = gh_alloc_tensor(gfx, SAM3_DTYPE_F32, 4,
						 dims_nhwc);
		if (!best_mask_nhwc) {
			err = SAM3_ENOMEM;
			goto cleanup;
		}
		size_t per_mask = (size_t)final_h * final_w;
		memcpy(best_mask_nhwc->data,
		       (const float *)track_masks->data
		         + (size_t)best_idx * per_mask,
		       per_mask * sizeof(float));
	}

	/* --- 10. Variant-dispatched memory encoder --------------------- */
	sam3_graph_init(&g);
	struct sam3_tensor *mem_feat = NULL;
	struct sam3_tensor *mem_pos  = NULL;

	SAM3_PROF_BEGIN(ctx->proc.profiler, "memenc_build");
	if (is_multiplex) {
		/*
		 * Multiplex packing: put the best mask into slot-0 channel 0
		 * and zero the remaining 31 channels. Proper multiplex
		 * layout (channels 0..15 active, 16..31 no-obj complements)
		 * is sub-project 4's concern.
		 */
		int pf_H  = cf.feat_s1->dims[1];
		int pf_W  = cf.feat_s1->dims[2];
		int big_H = pf_H * 16;
		int big_W = pf_W * 16;
		int mm_C  = SAM3_MULTIPLEX_IN_CHANNELS;
		int mm_dims[4] = {1, big_H, big_W, mm_C};
		struct sam3_tensor *multi_mask = gh_alloc_tensor(
			gfx, SAM3_DTYPE_F32, 4, mm_dims);
		if (!multi_mask) {
			err = SAM3_ENOMEM;
		} else {
			memset(multi_mask->data, 0, multi_mask->nbytes);
			int up_h = big_H / final_h;
			int up_w = big_W / final_w;
			if (up_h < 1) up_h = 1;
			if (up_w < 1) up_w = 1;
			const float *src =
				(const float *)track_masks->data
				  + (size_t)best_idx * final_h * final_w;
			float *dst = (float *)multi_mask->data;
			for (int y = 0; y < big_H; y++) {
				int sy = y / up_h;
				if (sy >= final_h) sy = final_h - 1;
				for (int x = 0; x < big_W; x++) {
					int sx = x / up_w;
					if (sx >= final_w) sx = final_w - 1;
					dst[((size_t)y * big_W + x)
					    * mm_C + 0] =
						src[sy * final_w + sx];
				}
			}
			mem_feat = sam3_multiplex_maskmem_forward(
				&g, gfx, &trk_mux->maskmem,
				cf.feat_s1, multi_mask,
				/*skip_mask_sigmoid=*/0);
			if (!mem_feat)
				err = SAM3_ENOMEM;
		}
	} else {
		/* Python mask_for_mem: threshold (cond) vs sigmoid (prop),
		 * then scale/bias. Done in-place on best_mask_nhwc. */
		preprocess_mask_for_mem_enc(best_mask_nhwc,
					    /*is_mask_from_pts=*/is_cond,
					    trk->sigmoid_scale,
					    trk->sigmoid_bias);
		int up_factor = trk->mem_encoder.interpol_h / final_h;
		if (up_factor < 1) up_factor = 1;
		struct sam3_tensor *mask_up = gh_upsample(
			&g, gfx, best_mask_nhwc, up_factor);
		if (!mask_up) {
			err = SAM3_ENOMEM;
		} else {
			err = sam3_memory_encoder_build(
				&trk->mem_encoder, &g,
				cf.feat_s1, mask_up, gfx,
				&mem_feat, &mem_pos);
		}
	}
	SAM3_PROF_END(ctx->proc.profiler, "memenc_build");
	if (err != SAM3_OK) {
		sam3_log_error("video_track: memenc build failed "
			       "(frame %d, obj %d, err %d)",
			       frame_idx, obj_idx, err);
		goto cleanup;
	}

	SAM3_PROF_BEGIN(ctx->proc.profiler, "memenc_eval");
	err = ctx->proc.backend->ops->graph_eval(ctx->proc.backend, &g);
	SAM3_PROF_END(ctx->proc.profiler, "memenc_eval");
	if (err != SAM3_OK) {
		sam3_log_error("video_track: memenc eval failed "
			       "(frame %d, obj %d, err %d)",
			       frame_idx, obj_idx, err);
		goto cleanup;
	}
	if (!mem_feat || mem_feat->n_dims != 4) {
		sam3_log_error("video_track: bad mem_feat shape");
		err = SAM3_EINVAL;
		goto cleanup;
	}

	/*
	 * SAM 3 post-encoder gating: add no_obj_embed_spatial when the
	 * object isn't appearing so the stored memory reflects occlusion.
	 * The multiplex no_obj_embed_spatial has a different shape (16, 256) and
	 * is applied via a different mechanism; skipped here.
	 */
	if (!is_multiplex) {
		apply_occlusion_gating(NULL, NULL, mem_feat,
				       track_score, no_obj_ptr, no_obj_embed);
	}

	/* --- 11. Flatten + clone to persist ---------------------------- */
	{
		int mH = mem_feat->dims[1];
		int mW = mem_feat->dims[2];
		int mC = mem_feat->dims[3];
		struct sam3_tensor flat_view = *mem_feat;
		flat_view.n_dims  = 2;
		flat_view.dims[0] = mH * mW;
		flat_view.dims[1] = mC;
		flat_view.dims[2] = 0;
		flat_view.dims[3] = 0;
		sam3_tensor_compute_strides(&flat_view);
		flat_view.ephemeral = 0;
		spatial_persist = sam3_tensor_clone_persist(
			&session->persist, &flat_view);
	}
	if (!spatial_persist) {
		sam3_log_error("video_track: spatial clone failed");
		err = SAM3_ENOMEM;
		goto cleanup;
	}
	obj_ptr_persist = sam3_tensor_clone_persist(&session->persist,
						    best_obj_ptr_scratch);
	if (!obj_ptr_persist) {
		sam3_log_error("video_track: obj_ptr clone failed");
		err = SAM3_ENOMEM;
		goto cleanup;
	}

	/* --- 12. Commit bank entry ------------------------------------- */
	{
		struct sam3_memory_entry entry;
		memset(&entry, 0, sizeof(entry));
		entry.spatial_features = spatial_persist;
		entry.obj_pointer      = obj_ptr_persist;
		entry.frame_idx        = frame_idx;
		entry.is_conditioning  = is_cond;
		entry.obj_score        = compute_eff_iou_score(
			track_score, best_iou_value);
		sam3_memory_bank_add(bank, &entry);
	}

	/* --- 13. Fill caller's obj_mask -------------------------------- */
	obj_mask->mask_h    = final_h;
	obj_mask->mask_w    = final_w;
	obj_mask->iou_score = best_iou_value;
	obj_mask->obj_score_logit = (track_score && track_score->data)
		? ((const float *)track_score->data)[0]
		: 0.0f;
	obj_mask->is_occluded = !is_appearing;
	{
		size_t mask_bytes =
			(size_t)final_h * final_w * sizeof(float);
		obj_mask->mask = malloc(mask_bytes);
		if (!obj_mask->mask) {
			err = SAM3_ENOMEM;
			goto cleanup;
		}
		memcpy(obj_mask->mask,
		       (const float *)track_masks->data
		         + (size_t)best_idx * final_h * final_w,
		       mask_bytes);
	}

	session->frames_tracked[frame_idx] = 1;

	sam3_log_debug("video_track: frame %d obj %d variant=%d is_cond=%d "
		       "iou=%.3f mask=%dx%d appearing=%d",
		       frame_idx, obj_idx, session->variant, is_cond,
		       (double)obj_mask->iou_score, final_h, final_w,
		       is_appearing);
	(void)mem_pos;  /* SAM 3 memory encoder returns but caller ignores */

cleanup:
	if (err != SAM3_OK && obj_mask) {
		free(obj_mask->mask);
		obj_mask->mask = NULL;
		obj_mask->mask_h = 0;
		obj_mask->mask_w = 0;
	}
	SAM3_PROF_END(ctx->proc.profiler,
		       is_cond ? "video_track_cond" : "video_track_prop");
	return err;
}

/*
 * video_add_prompts_pipeline - Conditioning wrapper around video_track_one_obj.
 *
 * Runs the shared per-object pipeline with is_cond=1, then commits the
 * stored prompt record and marks the per-object prompted-frame bitmap.
 *
 * @session:       Video session.
 * @frame_idx:     Frame the prompt applies to.
 * @obj_idx:       Internal object index into session->objects[].
 * @prompts:       Public prompt array (points / box) forwarded to the
 *                 shared pipeline.
 * @n_prompts:     Length of @prompts (>= 1).
 * @stored_prompt: Prompt record to append to session->prompts; frame_idx
 *                 and obj_internal_idx are overwritten here so the
 *                 stored record cannot drift from the pipeline args.
 * @obj_id:        User-facing obj id, used for log messages only.
 * @shared:        Optional per-frame shared state from propagate_one.
 * @result:        Caller-owned result. Zeroed before work starts; on
 *                 success result->objects is calloc'd with one entry.
 *
 * Returns SAM3_OK on success, SAM3_ENOMEM if the prompt list or the
 * conditioning ring is at capacity, or any error from the shared
 * pipeline.
 *
 * Capacity pre-check: the shared pipeline commits the bank entry
 * before returning, so rolling back on a later sam3_session_add_prompt
 * failure is unsafe (sam3_memory_bank_add silently drops at the
 * 16-entry cond cap; see src/model/memory_bank.c:35-47). Pre-checking
 * both the prompts[] capacity and the cond-ring capacity up front
 * avoids the race: once both checks pass, the post-pipeline append is
 * guaranteed to succeed.
 */
static enum sam3_error
video_add_prompts_pipeline(struct sam3_video_session *session,
			   int frame_idx, int obj_idx,
			   const struct sam3_prompt *prompts,
			   int n_prompts,
			   const struct sam3_video_prompt *stored_prompt,
			   int obj_id,
			   const struct propagate_frame_ctx *shared,
			   struct sam3_video_frame_result *result)
{
	if (session->n_prompts >= session->cap_prompts) {
		sam3_log_error("video_add_prompts: prompt list at capacity "
			       "(%d/%d), refusing to run tracker",
			       session->n_prompts, session->cap_prompts);
		return SAM3_ENOMEM;
	}
	{
		const struct sam3_memory_bank *b =
			&session->objects[obj_idx].bank;
		if (b->n_cond >= SAM3_MAX_MEMORY_FRAMES) {
			sam3_log_error("video_add_prompts: cond bank full "
				       "(%d/%d) for obj %d, refusing to run "
				       "tracker", b->n_cond,
				       SAM3_MAX_MEMORY_FRAMES, obj_id);
			return SAM3_ENOMEM;
		}
	}

	struct sam3_video_object_mask om;
	memset(&om, 0, sizeof(om));
	om.obj_id = obj_id;

	enum sam3_error err = video_track_one_obj(
		session, obj_idx, frame_idx, /*is_cond=*/1,
		prompts, n_prompts, shared, &om);
	if (err != SAM3_OK) {
		free(om.mask);
		return err;
	}

	struct sam3_video_prompt sp = *stored_prompt;
	sp.frame_idx        = frame_idx;
	sp.obj_internal_idx = obj_idx;
	err = sam3_session_add_prompt(session, &sp);
	if (err != SAM3_OK) {
		/* Should not trigger given the pre-check above; logged as an
		 * invariant violation rather than a recoverable path. The
		 * bank entry is already committed and cannot be rolled back
		 * safely (silent-drop case in sam3_memory_bank_add). */
		sam3_log_error("video_add_prompts: prompt store failed (%d) "
			       "after capacity pre-check — bank entry leaked",
			       err);
		free(om.mask);
		return err;
	}

	sam3_session_obj_mark_prompted(session, obj_idx, frame_idx);

	result->frame_idx = frame_idx;
	result->n_objects = 1;
	result->objects   = calloc(1, sizeof(*result->objects));
	if (!result->objects) {
		free(om.mask);
		return SAM3_ENOMEM;
	}
	result->objects[0] = om;
	return SAM3_OK;
}

/*
 * sam3_video_add_points - Add point prompts for an object on a frame.
 *
 * Thin wrapper: validates args, registers the object, packs the points
 * into public and internal prompt records, and delegates to
 * video_add_prompts_pipeline for the full prompt -> tracker -> memory
 * pipeline.
 */
enum sam3_error sam3_video_add_points(sam3_video_session *session,
				      int frame_idx, int obj_id,
				      const struct sam3_point *points,
				      int n_points,
				      struct sam3_video_frame_result *result)
{
	struct sam3_prompt prompts[SAM3_MAX_POINTS_PER_OBJ];
	struct sam3_video_prompt stored;
	int obj_idx;

	if (!session || !points || n_points <= 0 || !result) {
		sam3_log_error("sam3_video_add_points: NULL session/points/"
			       "result or n_points <= 0 (got %d)", n_points);
		return SAM3_EINVAL;
	}

	/*
	 * Zero the result up front so the pipeline's cleanup path
	 * can safely free() the malloc'd buffers without touching
	 * uninitialized fields.
	 */
	memset(result, 0, sizeof(*result));

	if (frame_idx < 0 || frame_idx >= session->frames.n_frames) {
		sam3_log_error("video_add_points: frame %d out of range "
			       "[0,%d)", frame_idx,
			       session->frames.n_frames);
		return SAM3_EINVAL;
	}

	if (n_points > SAM3_MAX_POINTS_PER_OBJ) {
		sam3_log_error("video_add_points: %d points exceeds max %d",
			       n_points, SAM3_MAX_POINTS_PER_OBJ);
		return SAM3_EINVAL;
	}

	obj_idx = sam3_session_get_or_add_obj(session, obj_id);
	if (obj_idx < 0) {
		sam3_log_error("video_add_points: cannot add obj %d", obj_id);
		return SAM3_EINVAL;
	}

	/* Public-API prompt array for sam3_project_prompts. */
	for (int i = 0; i < n_points; i++) {
		prompts[i].type  = SAM3_PROMPT_POINT;
		prompts[i].point = points[i];
	}

	/* Internal stored-prompt record (frame_idx / obj_idx filled by
	 * the pipeline helper to guarantee they match the actual run). */
	memset(&stored, 0, sizeof(stored));
	stored.kind          = SAM3_PROMPT_POINTS;
	stored.data.points.n = n_points;
	for (int i = 0; i < n_points; i++) {
		stored.data.points.xys[i * 2 + 0] = points[i].x;
		stored.data.points.xys[i * 2 + 1] = points[i].y;
		stored.data.points.labels[i]      = points[i].label;
	}

	return video_add_prompts_pipeline(session, frame_idx, obj_idx,
					  prompts, n_points,
					  &stored, obj_id, /*shared=*/NULL,
					  result);
}

/*
 * sam3_video_add_box - Add a bounding box prompt for an object on a frame.
 *
 * Thin wrapper: validates args, registers the object, packs the box
 * into public and internal prompt records, and delegates to
 * video_add_prompts_pipeline for the full prompt -> tracker -> memory
 * pipeline.
 */
enum sam3_error sam3_video_add_box(sam3_video_session *session,
				   int frame_idx, int obj_id,
				   const struct sam3_box *box,
				   struct sam3_video_frame_result *result)
{
	struct sam3_prompt prompt;
	struct sam3_video_prompt stored;
	int obj_idx;

	if (!session || !box || !result) {
		sam3_log_error("sam3_video_add_box: NULL session/box/result");
		return SAM3_EINVAL;
	}

	/* Zero result up-front so the pipeline cleanup path is safe. */
	memset(result, 0, sizeof(*result));

	if (frame_idx < 0 || frame_idx >= session->frames.n_frames) {
		sam3_log_error("video_add_box: frame %d out of range [0,%d)",
			       frame_idx, session->frames.n_frames);
		return SAM3_EINVAL;
	}

	if (!(box->x2 > box->x1) || !(box->y2 > box->y1)) {
		sam3_log_error("video_add_box: invalid box [%.1f,%.1f,"
			       "%.1f,%.1f] (x2>x1 and y2>y1 required)",
			       box->x1, box->y1, box->x2, box->y2);
		return SAM3_EINVAL;
	}

	obj_idx = sam3_session_get_or_add_obj(session, obj_id);
	if (obj_idx < 0) {
		sam3_log_error("video_add_box: cannot add obj %d", obj_id);
		return SAM3_EINVAL;
	}

	/* Public-API prompt record for sam3_project_prompts. */
	memset(&prompt, 0, sizeof(prompt));
	prompt.type = SAM3_PROMPT_BOX;
	prompt.box  = *box;

	/* Internal stored-prompt record (frame_idx / obj_idx filled by
	 * the pipeline helper to guarantee they match the actual run). */
	memset(&stored, 0, sizeof(stored));
	stored.kind     = SAM3_PROMPT_KIND_BOX;
	stored.data.box = *box;

	return video_add_prompts_pipeline(session, frame_idx, obj_idx,
					  &prompt, 1,
					  &stored, obj_id, /*shared=*/NULL,
					  result);
}

/*
 * video_replay_stored_prompt - Rebuild a public prompt array from a
 *                              stored prompt and dispatch it through
 *                              the shared add-prompts pipeline.
 *
 * Used by propagate to replay conditioning frames: the stored prompt
 * already knows the frame index and object, so we just translate
 * data.points / data.box back into a sam3_prompt[] and invoke the
 * same helper that add_points / add_box use.
 *
 * @session: Video session (validated by caller)
 * @sp:      Stored prompt to replay
 * @result:  Output result (caller zeros + frees via
 *           sam3_video_frame_result_free)
 */
static enum sam3_error
video_replay_stored_prompt(struct sam3_video_session *session,
			   const struct sam3_video_prompt *sp,
			   const struct propagate_frame_ctx *shared,
			   struct sam3_video_frame_result *result)
{
	struct sam3_prompt prompts[SAM3_MAX_POINTS_PER_OBJ];
	int n_prompts = 0;
	int obj_id;

	if (sp->obj_internal_idx < 0 ||
	    sp->obj_internal_idx >= session->n_objects) {
		sam3_log_error("video_propagate: stored prompt obj %d out of "
			       "range [0,%d)", sp->obj_internal_idx,
			       session->n_objects);
		return SAM3_EINVAL;
	}
	obj_id = session->objects[sp->obj_internal_idx].obj_id;

	if (sp->kind == SAM3_PROMPT_POINTS) {
		n_prompts = sp->data.points.n;
		if (n_prompts <= 0 ||
		    n_prompts > SAM3_MAX_POINTS_PER_OBJ) {
			sam3_log_error("video_propagate: bad stored point "
				       "count %d", n_prompts);
			return SAM3_EINVAL;
		}
		for (int i = 0; i < n_prompts; i++) {
			prompts[i].type        = SAM3_PROMPT_POINT;
			prompts[i].point.x     =
				sp->data.points.xys[i * 2 + 0];
			prompts[i].point.y     =
				sp->data.points.xys[i * 2 + 1];
			prompts[i].point.label =
				sp->data.points.labels[i];
		}
	} else if (sp->kind == SAM3_PROMPT_KIND_BOX) {
		prompts[0].type = SAM3_PROMPT_BOX;
		prompts[0].box  = sp->data.box;
		n_prompts = 1;
	} else {
		sam3_log_error("video_propagate: unknown stored prompt kind "
			       "%d", (int)sp->kind);
		return SAM3_EINVAL;
	}

	/*
	 * Drop any prior cond entry this object has at sp->frame_idx
	 * before re-running the pipeline. add_points / add_box already
	 * committed a cond entry when the user first invoked them; if we
	 * just called the pipeline again, memory attention on the replay
	 * would pull in the stale cond entry and produce occluded
	 * outputs (obj_score <= 0, masks zero). Matches Python, which
	 * runs add_new_points in a temp dict and only consolidates into
	 * the bank during propagate_preflight — the replay path here
	 * stands in for that consolidation.
	 */
	{
		struct sam3_memory_bank *bank =
			&session->objects[sp->obj_internal_idx].bank;
		int write = 0;
		for (int read = 0; read < bank->n_cond; read++) {
			if (bank->cond[read].frame_idx == sp->frame_idx)
				continue;
			if (write != read)
				bank->cond[write] = bank->cond[read];
			write++;
		}
		bank->n_cond = write;
	}

	return video_add_prompts_pipeline(session, sp->frame_idx,
					  sp->obj_internal_idx,
					  prompts, n_prompts,
					  sp, obj_id, shared, result);
}

/*
 * video_propagate_pure_tracking_obj - Propagation wrapper around
 *                                     video_track_one_obj.
 *
 * Thin pass-through: runs the shared per-object pipeline with is_cond=0
 * and no sparse prompt, committing a non-conditioning bank entry and
 * filling @obj_mask.
 *
 * @session:  Video session.
 * @obj_idx:  Internal object index into session->objects[].
 * @f:        Frame index (validated by caller).
 * @shared:   Optional per-frame shared state from propagate_one.
 * @obj_mask: Caller-owned output; on success obj_mask->mask is
 *            heap-allocated and owned by the caller.
 *
 * Returns SAM3_OK on success or any error from video_track_one_obj.
 */
static enum sam3_error
video_propagate_pure_tracking_obj(struct sam3_video_session *session,
				  int obj_idx, int f,
				  const struct propagate_frame_ctx *shared,
				  struct sam3_video_object_mask *obj_mask)
{
	return video_track_one_obj(session, obj_idx, f, /*is_cond=*/0,
				   /*prompts_opt=*/NULL, /*n_prompts_opt=*/0,
				   shared, obj_mask);
}

/*
 * video_replay_obj_prompt - Replay all stored prompts for one object
 *                           on one frame, writing the final mask into
 *                           obj_mask.
 *
 * Walks session->prompts[] filtering for entries matching
 * (frame_idx == f && obj_internal_idx == obj_idx). For each match,
 * calls video_replay_stored_prompt with a temporary frame result.
 * If multiple prompts match (e.g. repeated add_points), they are
 * applied in order; only the LAST result is kept in obj_mask.
 *
 * Returns SAM3_EMODEL if obj_is_prompted is true but no prompts are
 * found (bitmap/list desync). Returns SAM3_OK on success. On error,
 * obj_mask->mask is left NULL (never partially allocated by this
 * function).
 */
static enum sam3_error
video_replay_obj_prompt(struct sam3_video_session *session,
			int obj_idx, int f,
			const struct propagate_frame_ctx *shared,
			struct sam3_video_object_mask *obj_mask)
{
	/*
	 * Snapshot n_prompts before the loop: each replay call goes
	 * through video_add_prompts_pipeline, which appends the replayed
	 * prompt back onto session->prompts[]. Using the live n_prompts
	 * would cause newly-appended duplicates to be re-replayed.
	 */
	const int n_stored = session->n_prompts;
	int matched = 0;
	enum sam3_error err = SAM3_OK;

	for (int i = 0; i < n_stored; i++) {
		struct sam3_video_prompt *sp = &session->prompts[i];
		if (sp->frame_idx != f || sp->obj_internal_idx != obj_idx)
			continue;

		struct sam3_video_frame_result tmp;
		memset(&tmp, 0, sizeof(tmp));

		err = video_replay_stored_prompt(session, sp, shared, &tmp);
		if (err != SAM3_OK) {
			sam3_video_frame_result_free(&tmp);
			return err;
		}

		/* Keep only the LAST replay's mask: free any prior
		 * result allocated in previous iterations. */
		free(obj_mask->mask);
		obj_mask->mask = NULL;

		if (tmp.n_objects >= 1 && tmp.objects) {
			/* Transfer ownership of the mask buffer. */
			*obj_mask = tmp.objects[0];
			/* Prevent frame_result_free from double-freeing
			 * the mask we just transferred. */
			tmp.objects[0].mask = NULL;
		}
		sam3_video_frame_result_free(&tmp);
		matched++;
	}

	if (!matched) {
		sam3_log_error("video_propagate: frame %d obj %d flagged "
			       "prompted but no prompt found", f, obj_idx);
		return SAM3_EMODEL;
	}
	return SAM3_OK;
}

/*
 * propagate_one - Run propagation for a single frame across all objects.
 *
 * Loops over session->objects[]. For each object, dispatches to either
 * the conditioning-replay path (per-object prompted bit) or the pure-
 * tracking path. Assembles a sam3_video_frame_result with one entry per
 * object. On error the function frees any partially built result and
 * returns the error.
 *
 * @session: Video session
 * @f:       Frame index
 * @out:     Output result; zeroed on entry, filled on success.
 */
static enum sam3_error
propagate_one(struct sam3_video_session *session, int f,
	      struct sam3_video_frame_result *out)
{
	memset(out, 0, sizeof(*out));
	out->frame_idx = f;
	out->n_objects = session->n_objects;

	if (session->n_objects <= 0)
		return SAM3_OK;

	struct sam3_profiler *prof = session->ctx
		? session->ctx->proc.profiler
		: NULL;
	(void)prof;  /* Silence unused warning when SAM3_HAS_PROFILE is off. */
	SAM3_PROF_BEGIN(prof, "video_frame_track");
	uint64_t t_start = sam3_time_ns();

	out->objects = calloc((size_t)session->n_objects,
			      sizeof(struct sam3_video_object_mask));
	if (!out->objects) {
		SAM3_PROF_END(prof, "video_frame_track");
		return SAM3_ENOMEM;
	}

	/*
	 * Per-frame shared-state prefetch. Flatten feat_s1 (SAM 3 path
	 * only) once and stash the arena mark; each per-object call resets
	 * scratch to this mark instead of 0, so the flatten is paid once
	 * per frame rather than once per object. On multi-object clips
	 * (8 objs) this drops 7 redundant ~1.3 MiB memcpys plus the
	 * matching arena churn.
	 *
	 * Variant gating: only the SAM 3 tracker consumes the img_2d
	 * flatten — the SAM 3.1 multiplex tracker takes NHWC features
	 * directly. Tri-neck SAM 3.1 encodes also leave `image_features`
	 * NULL by design (no 0.5x scale), so gating on it would
	 * accidentally reject valid SAM 3.1 frames. SAM 3.1 still
	 * prefetches the frame cache and stashes the arena mark, but skips
	 * the 1.3 MiB img_2d memcpy that nobody reads.
	 *
	 * Fall back to per-object flatten (shared = NULL) if context is
	 * not ready, cache lookup fails, or feat_s1 has an unexpected
	 * shape — keeps correctness paths identical.
	 */
	struct sam3_ctx *ctx = session->ctx;
	struct propagate_frame_ctx fctx;
	const struct propagate_frame_ctx *shared = NULL;
	if (ctx && ctx->proc_ready && ctx->proc.backend) {
		const int is_multiplex =
			(session->variant == SAM3_VARIANT_SAM3_1);
		memset(&fctx, 0, sizeof(fctx));
		SAM3_PROF_BEGIN(prof, "frame_cache_get");
		enum sam3_error cf_err = sam3_frame_cache_get(
			&session->frame_cache, f, &fctx.cf);
		SAM3_PROF_END(prof, "frame_cache_get");
		int feats_ok = (cf_err == SAM3_OK &&
				fctx.cf.feat_s0 &&
				fctx.cf.feat_s1 &&
				fctx.cf.feat_s1->n_dims == 4 &&
				(is_multiplex
				 ? (fctx.cf.feat_4x != NULL)
				 : (fctx.cf.image_features != NULL)));
		if (feats_ok) {
			struct sam3_arena *scratch = &ctx->proc.scratch_arena;
			sam3_arena_reset(scratch);

			struct sam3_tensor *img = fctx.cf.feat_s1;
			fctx.grid_h = img->dims[1];
			fctx.grid_w = img->dims[2];

			int shared_ok = 1;
			if (!is_multiplex) {
				int dims2[2] = {fctx.grid_h * fctx.grid_w,
						img->dims[3]};
				fctx.img_2d = gh_alloc_tensor(scratch,
							      img->dtype,
							      2, dims2);
				if (fctx.img_2d) {
					memcpy(fctx.img_2d->data, img->data,
					       img->nbytes);
				} else {
					shared_ok = 0;
				}
			} else {
				fctx.img_2d = NULL;  /* unused by multiplex tracker */
			}
			if (shared_ok) {
				fctx.scratch_mark = scratch->offset;
				shared = &fctx;
			}
		}
	}

	for (int i = 0; i < session->n_objects; i++) {
		out->objects[i].obj_id = session->objects[i].obj_id;

		enum sam3_error err;
		if (sam3_session_obj_is_prompted(session, i, f)) {
			SAM3_PROF_BEGIN(prof, "video_replay_prompt");
			err = video_replay_obj_prompt(session, i, f,
						      shared,
						      &out->objects[i]);
			SAM3_PROF_END(prof, "video_replay_prompt");
		} else {
			SAM3_PROF_BEGIN(prof, "video_pure_tracking");
			err = video_propagate_pure_tracking_obj(
				session, i, f, shared, &out->objects[i]);
			SAM3_PROF_END(prof, "video_pure_tracking");
		}
		if (err != SAM3_OK) {
			sam3_video_frame_result_free(out);
			SAM3_PROF_END(prof, "video_frame_track");
			return err;
		}
	}

	uint64_t t_end = sam3_time_ns();
	sam3_log_info("frame %d tracked: %d objects, %.1f ms",
		      f, session->n_objects,
		      (double)(t_end - t_start) / 1.0e6);
	SAM3_PROF_END(prof, "video_frame_track");
	return SAM3_OK;
}

/*
 * do_propagate_inner - Inner sweep for sam3_video_propagate.
 *
 * Performs the actual frame iteration (forward, backward, or both).
 * Called with session->in_propagate already set to 1; does not manage
 * the flag itself. Returns SAM3_OK on success, or error code on failure
 * (failed frame processing or callback-initiated stop).
 */
static enum sam3_error do_propagate_inner(sam3_video_session *session,
					   int direction,
					   sam3_video_frame_cb callback,
					   void *user_data)
{
	struct sam3_video_frame_result r;
	enum sam3_error err;
	int nf;

	nf = session->frames.n_frames;

	sam3_log_info("video_propagate: dir=%d, %d frames, %d objects",
		      direction, nf, session->n_objects);

	/*
	 * Per-object memory banks are rebuilt from scratch on every propagate
	 * call — clear all objects' banks on entry. Conditioning entries for
	 * prompted frames are re-added by video_add_prompts_pipeline as those
	 * frames are visited during the sweep. Without this, repeat propagate
	 * calls would stack non-conditioning entries for every frame and drive
	 * the banks toward memory exhaustion.
	 *
	 * Task 4.1: memory persistence enabled. Propagate is now resumable:
	 * a second call extends tracking instead of recomputing from scratch.
	 * When a new prompt arrives on a previously-tracked frame,
	 * clear_non_cond_mem_around_input wipes only the stale window.
	 */

	/* Forward pass (FORWARD and BOTH) */
	if (direction == SAM3_PROPAGATE_FORWARD ||
	    direction == SAM3_PROPAGATE_BOTH) {
		for (int f = 0; f < nf; f++) {
			memset(&r, 0, sizeof(r));
			err = propagate_one(session, f, &r);
			if (err != SAM3_OK) {
				sam3_video_frame_result_free(&r);
				return err;
			}

			if (callback) {
				int stop = callback(&r, user_data);
				sam3_video_frame_result_free(&r);
				if (stop) {
					sam3_log_info("video_propagate: "
						      "stopped at frame %d",
						      f);
					return SAM3_OK;
				}
			} else {
				sam3_video_frame_result_free(&r);
			}

			sam3_arena_reset(&session->scratch);
		}
	}

	/*
	 * Backward pass (BACKWARD and BOTH).
	 * Per-object banks now persist across both passes, allowing the
	 * reverse sweep to extend from the forward pass rather than rebuild
	 * from scratch. Conditioning frames still re-add themselves on revisit.
	 */

	if (direction == SAM3_PROPAGATE_BACKWARD ||
	    direction == SAM3_PROPAGATE_BOTH) {
		for (int f = nf - 1; f >= 0; f--) {
			memset(&r, 0, sizeof(r));
			err = propagate_one(session, f, &r);
			if (err != SAM3_OK) {
				sam3_video_frame_result_free(&r);
				return err;
			}

			if (callback) {
				int stop = callback(&r, user_data);
				sam3_video_frame_result_free(&r);
				if (stop) {
					sam3_log_info("video_propagate: "
						      "stopped at frame %d",
						      f);
					return SAM3_OK;
				}
			} else {
				sam3_video_frame_result_free(&r);
			}

			sam3_arena_reset(&session->scratch);
		}
	}

	return SAM3_OK;
}

/*
 * sam3_video_propagate - Propagate tracked objects across video frames.
 *
 * Iterates frames in the requested direction. For each frame, either
 * replays the stored prompt pipeline (conditioning) or runs the
 * tracker with the memory bank only (pure tracking), then invokes
 * @callback. Returning non-zero from the callback halts propagation.
 *
 * The memory bank is cleared on entry for every direction, so each
 * propagate call rebuilds memory from scratch. Conditioning entries
 * for prompted frames are re-committed by the replay pipeline as
 * those frames are visited. For SAM3_PROPAGATE_BOTH the bank is also
 * cleared between the forward and backward passes so the reverse
 * sweep starts fresh.
 */
enum sam3_error sam3_video_propagate(sam3_video_session *session,
				     int direction,
				     sam3_video_frame_cb callback,
				     void *user_data)
{
	enum sam3_error err;

	if (!session) {
		sam3_log_error("video_propagate: NULL session");
		return SAM3_EINVAL;
	}

	if (session->n_objects <= 0) {
		sam3_log_error("video_propagate: no objects to propagate");
		return SAM3_EINVAL;
	}

	if (direction != SAM3_PROPAGATE_FORWARD &&
	    direction != SAM3_PROPAGATE_BACKWARD &&
	    direction != SAM3_PROPAGATE_BOTH) {
		sam3_log_error("video_propagate: invalid direction %d",
			       direction);
		return SAM3_EINVAL;
	}

	struct sam3_profiler *prof = session->ctx
		? session->ctx->proc.profiler
		: NULL;
	(void)prof;
	SAM3_PROF_BEGIN(prof, "video_propagate");
	uint64_t t_start = sam3_time_ns();

	session->in_propagate = 1;
	err = do_propagate_inner(session, direction, callback, user_data);
	session->in_propagate = 0;

	uint64_t t_end = sam3_time_ns();
	int nf = sam3_video_frame_count(session);
	double total_ms = (double)(t_end - t_start) / 1.0e6;
	double per_frame_ms = nf > 0 ? total_ms / nf : 0.0;
	sam3_log_info("propagation done: %d frames, %.1f ms total "
		      "(%.1f ms/frame, %.2f fps)",
		      nf, total_ms, per_frame_ms,
		      per_frame_ms > 0 ? 1000.0 / per_frame_ms : 0.0);
	uint64_t cache_total = session->frame_cache.n_hits +
			       session->frame_cache.n_misses +
			       session->frame_cache.n_spill_promotes;
	if (cache_total > 0) {
		double hit_pct = 100.0 * (double)session->frame_cache.n_hits /
				 (double)cache_total;
		sam3_log_info("frame cache: %llu hits, %llu misses, "
			      "%llu spill_promotes (%.1f%% hit rate)",
			      (unsigned long long)session->frame_cache.n_hits,
			      (unsigned long long)session->frame_cache.n_misses,
			      (unsigned long long)
			      session->frame_cache.n_spill_promotes,
			      hit_pct);
	}
	SAM3_PROF_END(prof, "video_propagate");

	return err;
}

/*
 * sam3_video_remove_object - Remove a tracked object from the session.
 */
enum sam3_error sam3_video_remove_object(sam3_video_session *session,
					 int obj_id)
{
	if (!session) {
		sam3_log_error("video_remove_object: NULL session");
		return SAM3_EINVAL;
	}

	if (session->in_propagate) {
		sam3_log_error("video_remove_object: cannot call from inside "
			       "propagate callback (use callback return value "
			       "to stop first)");
		return SAM3_EINVAL;
	}

	return sam3_session_remove_obj(session, obj_id);
}

/*
 * sam3_video_reset - Clear all tracked objects and prompts.
 *
 * Resets tracking state but preserves encoded frame features so
 * new prompts can be added without re-encoding the video.
 */
enum sam3_error sam3_video_reset(sam3_video_session *session)
{
	if (!session) {
		sam3_log_error("video_reset: NULL session");
		return SAM3_EINVAL;
	}

	if (session->in_propagate) {
		sam3_log_error("video_reset: cannot call from inside propagate "
			       "callback (use callback return value to stop "
			       "first)");
		return SAM3_EINVAL;
	}

	/*
	 * Clear per-object memory banks before zeroing the object array.
	 * sam3_tracker_reset is still called for any future tracker-level
	 * cleanup, but since mem_bank was removed from sam3_tracker in
	 * Task 2.2 it is a no-op.
	 */
	for (int i = 0; i < session->n_objects; i++) {
		sam3_memory_bank_clear(&session->objects[i].bank);
		free(session->objects[i].prompted_frames);
	}
	memset(session->objects, 0, sizeof(session->objects));
	sam3_tracker_reset(&session->tracker);
	session->n_objects = 0;

	/* Clear per-frame tracking status */
	if (session->frames_tracked && session->frames.n_frames > 0) {
		memset(session->frames_tracked, 0,
		       (size_t)session->frames.n_frames * sizeof(int));
	}

	/* Drop stored prompts and clear prompted-frame bitmap */
	sam3_session_clear_prompts(session);

	/* Reset per-frame scratch arena */
	sam3_arena_reset(&session->scratch);

	sam3_log_info("video session reset");
	return SAM3_OK;
}

/*
 * sam3_video_end - End a video session and free all resources.
 *
 * Releases arenas and the session struct. Safe to call with NULL.
 */
void sam3_video_end(sam3_video_session *session)
{
	if (!session)
		return;

	sam3_tracker_reset(&session->tracker);
	sam3_frame_cache_release(&session->frame_cache);
	sam3_arena_free(&session->scratch);
	sam3_arena_free(&session->persist);
	free(session);

	sam3_log_info("video session ended");
}

/*
 * sam3_video_add_mask - Add a binary mask prompt for an object on a frame.
 *
 * Bypasses the SAM mask decoder. The caller-supplied binary mask IS the
 * segmentation: it is resized nearest-neighbor to 1152x1152, preprocessed
 * with mask_for_mem (is_mask_from_pts=1 semantics), run through the memory
 * encoder, and committed as a conditioning entry to the object's bank.
 * Mirrors Python add_new_mask semantics.
 *
 * obj_pointer in the committed entry is NULL because no decoder ran; the
 * graph_helpers concat path already tolerates NULL pointers (skips them).
 */
enum sam3_error sam3_video_add_mask(sam3_video_session *session,
				    int frame_idx, int obj_id,
				    const uint8_t *mask,
				    int mask_h, int mask_w,
				    struct sam3_video_frame_result *result)
{
	struct sam3_ctx *ctx;
	struct sam3_tracker *trk;
	struct sam3_frame_features cf;
	struct sam3_arena *gfx;
	struct sam3_tensor *hires = NULL;
	struct sam3_tensor *mem_feat = NULL;
	struct sam3_tensor *mem_pos = NULL;
	struct sam3_tensor *spatial_persist = NULL;
	struct sam3_graph g;
	int obj_idx;
	int clear_window;
	int mem_H, mem_W, mem_C, mem_HW;
	enum sam3_error err = SAM3_OK;

	if (!session || !mask || !result)
		return SAM3_EINVAL;
	if (mask_h <= 0 || mask_w <= 0)
		return SAM3_EINVAL;

	memset(result, 0, sizeof(*result));

	/*
	 * SAM 3.1 mask prompts route through the multiplex maskmem backbone
	 * instead of the SAM 3 memory encoder. Sub-project 3's interactive
	 * decoder + the multiplex mask downsampler would wire this cleanly; until
	 * then, reject with a clear error rather than dereferencing the
	 * zero-initialized SAM 3 tracker below.
	 */
	if (session->variant == SAM3_VARIANT_SAM3_1) {
		sam3_log_error("video_add_mask: not yet supported for SAM 3.1 "
			       "(use add_points / propagate; sub-project 3 "
			       "will add mask prompts)");
		return SAM3_EVIDEO;
	}

	ctx = session->ctx;
	if (!ctx || !ctx->proc_ready || !ctx->proc.backend) {
		sam3_log_error("video_add_mask: context not ready");
		return SAM3_EINVAL;
	}

	trk = &session->tracker;

	/*
	 * Reject absurd dims: allow up to 2 * image_size on each axis
	 * (same leniency as the Python reference).
	 */
	if (mask_h > 2 * trk->image_size || mask_w > 2 * trk->image_size) {
		sam3_log_error("video_add_mask: mask dims %dx%d exceed "
			       "2*image_size (%d)", mask_h, mask_w,
			       2 * trk->image_size);
		return SAM3_EINVAL;
	}

	if (frame_idx < 0 || frame_idx >= session->frames.n_frames) {
		sam3_log_error("video_add_mask: frame %d out of range [0,%d)",
			       frame_idx, session->frames.n_frames);
		return SAM3_EINVAL;
	}

	obj_idx = sam3_session_get_or_add_obj(session, obj_id);
	if (obj_idx < 0) {
		sam3_log_error("video_add_mask: object cap reached for obj %d",
			       obj_id);
		return SAM3_EINVAL;
	}

	/*
	 * Clear stale non-conditioning entries in the memory window around
	 * this frame (Python clear_non_cond_mem_around_input).
	 */
	clear_window = session->opts.clear_non_cond_window;
	if (clear_window <= 0)
		clear_window = 7;
	sam3_memory_bank_clear_around_frame(
		&session->objects[obj_idx].bank, frame_idx, clear_window);

	/* Use the processor's large scratch arena for graph intermediates. */
	gfx = &ctx->proc.scratch_arena;
	sam3_arena_reset(gfx);

	/* Fetch frame features via the cache (encodes on miss). */
	memset(&cf, 0, sizeof(cf));
	SAM3_PROF_BEGIN(ctx->proc.profiler, "frame_cache_get");
	err = sam3_frame_cache_get(&session->frame_cache, frame_idx, &cf);
	SAM3_PROF_END(ctx->proc.profiler, "frame_cache_get");
	if (err != SAM3_OK) {
		sam3_log_error("video_add_mask: frame_cache_get failed "
			       "(frame %d, err %d)", frame_idx, err);
		return err;
	}
	if (!cf.image_features) {
		sam3_log_error("video_add_mask: frame %d features missing",
			       frame_idx);
		return SAM3_EINVAL;
	}

	/*
	 * Allocate high-res mask [1, 1152, 1152, 1] NHWC f32 in scratch.
	 * The memory encoder's interpol_h/w is 1152 (SAM3_MEMENC_DS_LAYERS
	 * stride-2 convs expect this exact size as input to the downsampler).
	 */
	{
		const int HIRES = trk->mem_encoder.interpol_h;
		int hires_dims[4] = {1, HIRES, HIRES, 1};

		hires = gh_alloc_tensor(gfx, SAM3_DTYPE_F32, 4, hires_dims);
		if (!hires) {
			sam3_log_error("video_add_mask: hires alloc failed");
			return SAM3_ENOMEM;
		}

		/*
		 * Nearest-neighbor resize: map each output pixel to the
		 * nearest source pixel, then apply mask_for_mem preprocessing
		 * inline (is_mask_from_pts=1 branch: fg -> scale+bias,
		 * bg -> bias). This is equivalent to calling
		 * preprocess_mask_for_mem_enc after resizing.
		 */
		float fg_val = trk->sigmoid_scale + trk->sigmoid_bias;
		float bg_val = trk->sigmoid_bias;
		float *d = (float *)hires->data;
		for (int y = 0; y < HIRES; y++) {
			int sy = (int)((int64_t)y * mask_h / HIRES);
			if (sy >= mask_h) sy = mask_h - 1;
			for (int x = 0; x < HIRES; x++) {
				int sx = (int)((int64_t)x * mask_w / HIRES);
				if (sx >= mask_w) sx = mask_w - 1;
				d[y * HIRES + x] =
					mask[sy * mask_w + sx] ? fg_val : bg_val;
			}
		}
	}

	/* Build and evaluate the memory encoder graph. */
	sam3_graph_init(&g);
	err = sam3_memory_encoder_build(&trk->mem_encoder, &g,
					cf.image_features, hires, gfx,
					&mem_feat, &mem_pos);
	if (err != SAM3_OK) {
		sam3_log_error("video_add_mask: memory encoder build failed "
			       "(err %d)", err);
		return err;
	}

	err = ctx->proc.backend->ops->graph_eval(ctx->proc.backend, &g);
	if (err != SAM3_OK) {
		sam3_log_error("video_add_mask: memory encoder eval failed "
			       "(err %d)", err);
		return err;
	}

	if (!mem_feat || mem_feat->n_dims != 4) {
		sam3_log_error("video_add_mask: bad mem_feat shape");
		return SAM3_EINVAL;
	}

	/*
	 * Flatten [1, H, W, C] -> [HW, C] and clone into the session persist
	 * arena so the entry survives the next scratch-arena reset.
	 */
	mem_H  = mem_feat->dims[1];
	mem_W  = mem_feat->dims[2];
	mem_C  = mem_feat->dims[3];
	mem_HW = mem_H * mem_W;
	{
		struct sam3_tensor flat_view = *mem_feat;
		flat_view.n_dims   = 2;
		flat_view.dims[0]  = mem_HW;
		flat_view.dims[1]  = mem_C;
		flat_view.dims[2]  = 0;
		flat_view.dims[3]  = 0;
		sam3_tensor_compute_strides(&flat_view);
		flat_view.ephemeral = 0;

		spatial_persist = sam3_tensor_clone_persist(
			&session->persist, &flat_view);
	}
	if (!spatial_persist) {
		sam3_log_error("video_add_mask: spatial clone failed");
		return SAM3_ENOMEM;
	}

	/*
	 * Commit a conditioning entry. obj_pointer is NULL because no decoder
	 * ran; graph_helpers.c (gh_concat_obj_ptrs) already skips NULL
	 * obj_pointer entries with `if (!op) continue`.
	 */
	{
		struct sam3_memory_entry entry;
		memset(&entry, 0, sizeof(entry));
		entry.spatial_features = spatial_persist;
		entry.obj_pointer      = NULL;
		entry.frame_idx        = frame_idx;
		entry.is_conditioning  = 1;
		entry.obj_score        = 1.0f; /* user-supplied = full confidence */
		sam3_memory_bank_add(&session->objects[obj_idx].bank, &entry);
	}

	/* Mark the prompted-frame bitmap. */
	sam3_session_obj_mark_prompted(session, obj_idx, frame_idx);
	session->frames_tracked[frame_idx] = 1;

	/*
	 * Fill the result: single-object, single-frame.
	 * The mask payload carries the input mask as ±10 logits so callers
	 * can use the same threshold (0) as the decoder output.
	 */
	result->frame_idx = frame_idx;
	result->n_objects = 1;
	result->objects = calloc(1, sizeof(*result->objects));
	if (!result->objects) {
		sam3_log_error("video_add_mask: result objects alloc failed");
		err = SAM3_ENOMEM;
		goto cleanup;
	}
	result->objects[0].obj_id          = obj_id;
	result->objects[0].mask_h          = mask_h;
	result->objects[0].mask_w          = mask_w;
	result->objects[0].iou_score       = 1.0f;
	result->objects[0].obj_score_logit = 10.0f;
	result->objects[0].is_occluded     = 0;
	result->objects[0].mask = malloc(
		(size_t)mask_h * (size_t)mask_w * sizeof(float));
	if (!result->objects[0].mask) {
		sam3_log_error("video_add_mask: result mask alloc failed");
		err = SAM3_ENOMEM;
		goto cleanup;
	}
	{
		float *rm = result->objects[0].mask;
		int npix = mask_h * mask_w;
		for (int i = 0; i < npix; i++)
			rm[i] = mask[i] ? 10.0f : -10.0f;
	}

	sam3_log_info("video_add_mask: committed mask frame=%d obj=%d "
		      "src=%dx%d", frame_idx, obj_id, mask_h, mask_w);

	(void)mem_pos;
	return SAM3_OK;

cleanup:
	sam3_video_frame_result_free(result);
	(void)mem_pos;
	return err;
}

/*
 * sam3_video_frame_result_free - Release per-frame result memory.
 */
void sam3_video_frame_result_free(struct sam3_video_frame_result *r)
{
	if (!r)
		return;
	if (r->objects) {
		for (int i = 0; i < r->n_objects; i++)
			free(r->objects[i].mask);
		free(r->objects);
	}
	r->objects   = NULL;
	r->n_objects = 0;
	r->frame_idx = 0;
}

/*
 * sam3_video_frame_count - Return the number of frames in the session.
 */
int sam3_video_frame_count(const sam3_video_session *session)
{
	if (!session)
		return 0;
	return session->frames.n_frames;
}
