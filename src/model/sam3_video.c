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

	out->image_features = sam3_tensor_clone_persist(arena, src_05x);
	out->feat_s0 = sam3_tensor_clone_persist(arena, src_2x);
	out->feat_s1 = sam3_tensor_clone_persist(arena, src_1x);
	/* feat_4x (288x288 4x skip) tolerates NULL for older models. */
	out->feat_4x = src_4x
		? sam3_tensor_clone_persist(arena, src_4x)
		: NULL;

	if (!out->image_features || !out->feat_s0 || !out->feat_s1) {
		sam3_log_error("session_encode_frame: clone frame %d failed",
			       frame_idx);
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

	/* Initialize and load the tracker with model weights */
	err = sam3_tracker_init(&session->tracker);
	if (err != SAM3_OK) {
		sam3_log_error("video_start: tracker init failed (%d)", err);
		goto cleanup;
	}

	err = sam3_tracker_load(&session->tracker, &ctx->weights,
				&session->persist);
	if (err != SAM3_OK) {
		sam3_log_error("video_start: tracker load failed (%d)", err);
		goto cleanup;
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
 * video_add_prompts_pipeline - Shared prompt -> tracker -> memory pipeline.
 *
 * @session:        Video session (already validated non-NULL by caller)
 * @frame_idx:      Frame the prompts apply to (must be in range)
 * @obj_idx:        Internal object index (from sam3_session_get_or_add_obj)
 * @prompts:        Public-API prompt array for sam3_project_prompts
 * @n_prompts:      Length of @prompts (>= 1)
 * @stored_prompt:  Prompt entry to commit to session->prompts before
 *                  the memory bank is updated, so a capacity failure
 *                  cannot leave the two structures out of sync. Its
 *                  frame_idx / obj_internal_idx are overwritten here to
 *                  guarantee consistency with the pipeline arguments.
 * @obj_id:         User-facing obj id, used for log messages only.
 * @result:         Caller-owned result. Zeroed before work starts and on
 *                  failure; filled with heap-allocated masks + IoU on
 *                  success.
 * @shared:         Optional per-frame shared state from propagate_one.
 *                  When non-NULL, frame_cache_get and the img_2d flatten
 *                  are skipped — the pipeline reuses @shared->cf and
 *                  @shared->img_2d and resets the scratch arena to
 *                  @shared->scratch_mark between objects. NULL = run the
 *                  legacy per-call path (used by add_points / add_box).
 *
 * Runs the same five-step pipeline used by add_points / add_box:
 *   1. Project prompts to [N, d_model] tokens (sam3_project_prompts).
 *   2. Reshape cached NHWC image features to [HW, d] for the tracker.
 *   3. Build + evaluate the tracker graph (is_cond=1). Select argmax
 *      IoU as best mask.
 *   4. Upsample the best mask to the memory encoder's interpol_size,
 *      run the memory encoder graph, and clone the spatial features
 *      into the session persist arena.
 *   5. Commit the stored prompt to session->prompts via
 *      sam3_session_add_prompt — this can fail (SAM3_ENOMEM) if the
 *      list is at capacity, in which case we bail out before touching
 *      the memory bank so the two structures stay consistent. On
 *      success, append a conditioning entry to the memory bank and
 *      copy masks + IoU into the caller's result.
 *
 * On error the function logs the failure, frees any heap fields it had
 * attached to @result, and returns the appropriate sam3_error code.
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
	struct sam3_ctx *ctx;
	struct sam3_tracker *trk;
	struct sam3_frame_features cf;
	struct sam3_arena *gfx_scratch;
	struct sam3_tensor *img_2d = NULL;
	struct sam3_tensor *prompt_tokens = NULL;
	struct sam3_tensor *track_masks = NULL;
	struct sam3_tensor *track_iou = NULL;
	struct sam3_tensor *best_mask_nhwc = NULL;
	struct sam3_tensor *mask_up = NULL;
	struct sam3_tensor *mem_feat = NULL;
	struct sam3_tensor *mem_pos = NULL;
	struct sam3_tensor *spatial_persist = NULL;
	struct sam3_graph g;
	int grid_h = 0, grid_w = 0;
	int frame_size;
	int n_masks = 0, final_h = 0, final_w = 0;
	int best_idx = 0;
	int mem_H, mem_W, mem_C, mem_HW;
	size_t mask_bytes = 0;
	enum sam3_error err = SAM3_OK;

	ctx = session->ctx;
	if (!ctx || !ctx->proc_ready || !ctx->proc.backend) {
		sam3_log_error("video_add_prompts: context not ready");
		err = SAM3_EINVAL;
		goto cleanup;
	}

	SAM3_PROF_BEGIN(ctx->proc.profiler, "video_add_prompts");

	trk = &session->tracker;
	if (shared) {
		cf = shared->cf;
	} else {
		memset(&cf, 0, sizeof(cf));
		SAM3_PROF_BEGIN(ctx->proc.profiler, "frame_cache_get");
		err = sam3_frame_cache_get(&session->frame_cache,
					   frame_idx, &cf);
		SAM3_PROF_END(ctx->proc.profiler, "frame_cache_get");
		if (err != SAM3_OK) {
			sam3_log_error("video_add_prompts: frame %d cache get "
				       "failed (%d)", frame_idx, err);
			goto cleanup;
		}
	}
	if (!cf.image_features || !cf.feat_s0 || !cf.feat_s1) {
		sam3_log_error("video_add_prompts: frame %d features missing",
			       frame_idx);
		err = SAM3_EINVAL;
		goto cleanup;
	}

	/*
	 * Use the processor's large (3 GiB) scratch arena for graph
	 * intermediates rather than the session's 256 MiB scratch.
	 * The tracker graph (4-layer memory attention + full mask
	 * decoder) needs several hundred MiB of intermediates at the
	 * 5184-patch grid size. The session scratch is reserved for
	 * small per-call bookkeeping.
	 *
	 * Shared-frame path: propagate_one already allocated @shared->img_2d
	 * in this arena and captured the post-flatten offset. Resetting to
	 * that mark (instead of 0) preserves the shared flatten across the
	 * per-object loop while still recycling per-object prompt / graph
	 * scratch between iterations.
	 */
	gfx_scratch = &ctx->proc.scratch_arena;
	if (shared)
		gfx_scratch->offset = shared->scratch_mark;
	else
		sam3_arena_reset(gfx_scratch);

	/*
	 * Python clear_non_cond_mem_around_input: when a new conditioning
	 * prompt arrives on a previously-tracked frame, the propagated
	 * non-cond entries within the memory window are stale. Drop them
	 * from this object's bank before re-running the decoder.
	 */
	int clear_window = session->opts.clear_non_cond_window;
	if (clear_window <= 0)
		clear_window = 7;
	sam3_memory_bank_clear_around_frame(
		&session->objects[obj_idx].bank,
		frame_idx, clear_window);

	/* Project prompt coordinates to [N, d_model] tokens on CPU.
	 * The video tracker uses SAM's prompt encoder (sam_prompt_encoder
	 * weights) when available. The SAM path produces 2 tokens per
	 * point (pe_layer + point_embeddings[label] + not_a_point
	 * padding) which matches the two-way transformer's expected
	 * input format. Fall back to the SAM3 geometry encoder when the
	 * weights aren't loaded (older checkpoints). */
	frame_size = session->frames.frame_size;
	SAM3_PROF_BEGIN(ctx->proc.profiler, "prompt_project");
	prompt_tokens = sam3_tracker_sam_project_prompts(
		trk, prompts, n_prompts, frame_size, frame_size,
		gfx_scratch);
	if (!prompt_tokens) {
		prompt_tokens = sam3_project_prompts(&ctx->proc.model,
						     cf.feat_s1,
						     prompts, n_prompts,
						     frame_size, frame_size,
						     gfx_scratch);
	}
	SAM3_PROF_END(ctx->proc.profiler, "prompt_project");
	if (!prompt_tokens) {
		sam3_log_error("video_add_prompts: prompt projection failed");
		err = SAM3_ENOMEM;
		goto cleanup;
	}

	/*
	 * Python reference uses main = 72x72 grid (image_size 1008 /
	 * stride 14). Our cache layout is:
	 *   image_features = neck_05x (36x36 = 0.5x)
	 *   feat_s1        = neck_1x  (72x72 = 1x = main)
	 *   feat_s0        = neck_2x  (144x144 = 2x)
	 *   feat_4x        = neck_4x  (288x288 = 4x)
	 * So the tracker's main-grid input is cf.feat_s1, not
	 * image_features. (An earlier revert to image_features kept
	 * frame-0 outputs non-zero but at the wrong 36x36 scale — masks
	 * came out with structural bias but not clean silhouettes. The
	 * weights are trained for 72x72; stick with that here.)
	 *
	 * Shared-frame path: reuse the flatten propagate_one already did.
	 */
	if (shared) {
		img_2d = shared->img_2d;
		grid_h = shared->grid_h;
		grid_w = shared->grid_w;
	} else {
		struct sam3_tensor *img = cf.feat_s1;
		int HW, d;
		int dims2[2];

		if (img->n_dims != 4) {
			sam3_log_error("video_add_prompts: feat_s1 expected "
				       "4D, got %d", img->n_dims);
			err = SAM3_EINVAL;
			goto cleanup;
		}
		grid_h = img->dims[1];
		grid_w = img->dims[2];
		HW = grid_h * grid_w;
		d  = img->dims[3];
		dims2[0] = HW;
		dims2[1] = d;
		img_2d = gh_alloc_tensor(gfx_scratch, img->dtype,
					 2, dims2);
		if (!img_2d) {
			sam3_log_error("video_add_prompts: img_2d alloc "
				       "failed");
			err = SAM3_ENOMEM;
			goto cleanup;
		}
		memcpy(img_2d->data, img->data, img->nbytes);
	}

	/*
	 * Build and evaluate the tracker graph (memory attention +
	 * mask decoder). Mask-decoder skip params expect (when main=72):
	 *   feat_s0 param → 4x main scale (288x288) = cf.feat_4x
	 *   feat_s1 param → 2x main scale (144x144) = cf.feat_s0
	 */
	struct sam3_tensor *cond_obj_ptr = NULL;
	struct sam3_tensor *cond_obj_score = NULL;
	sam3_graph_init(&g);
	SAM3_PROF_BEGIN(ctx->proc.profiler, "tracker_build");
	err = sam3_tracker_track_frame(trk, &g,
				       &session->objects[obj_idx].bank,
				       img_2d, grid_h, grid_w,
				       prompt_tokens,
				       cf.feat_4x, cf.feat_s0,
				       frame_idx, /*is_cond=*/1,
				       gfx_scratch,
				       &track_masks, &track_iou,
				       &cond_obj_ptr, &cond_obj_score);
	SAM3_PROF_END(ctx->proc.profiler, "tracker_build");
	if (err != SAM3_OK) {
		sam3_log_error("video_add_prompts: track_frame failed (%d)",
			       err);
		goto cleanup;
	}

	SAM3_PROF_BEGIN(ctx->proc.profiler, "tracker_eval");
	err = ctx->proc.backend->ops->graph_eval(ctx->proc.backend, &g);
	SAM3_PROF_END(ctx->proc.profiler, "tracker_eval");
	if (err != SAM3_OK) {
		sam3_log_error("video_add_prompts: tracker graph eval "
			       "failed (%d)", err);
		goto cleanup;
	}

#ifdef SAM3_DEBUG_DUMP
	{
		auto_dump_tensor("/tmp/dbg_tracker_img2d.bin", img_2d);
		auto_dump_tensor("/tmp/dbg_tracker_prompt.bin", prompt_tokens);
		auto_dump_tensor("/tmp/dbg_tracker_masks.bin", track_masks);
		auto_dump_tensor("/tmp/dbg_tracker_iou.bin", track_iou);
		auto_dump_tensor("/tmp/dbg_tracker_obj_score.bin",
				 cond_obj_score);
		extern struct sam3_tensor *sam3_dbg_pixel_out;
		extern struct sam3_tensor *sam3_dbg_pix_ct1;
		extern struct sam3_tensor *sam3_dbg_pix_ct1_input;
		extern struct sam3_tensor *sam3_dbg_pix_skip1;
		extern struct sam3_tensor *sam3_dbg_pix_after_skip1;
		extern struct sam3_tensor *sam3_dbg_xformer_in_img;
		extern struct sam3_tensor *sam3_dbg_xformer_in_tokens;
		extern struct sam3_tensor *sam3_dbg_xformer_layer0_q;
		auto_dump_tensor("/tmp/dbg_xformer_in_img.bin",
				 sam3_dbg_xformer_in_img);
		auto_dump_tensor("/tmp/dbg_xformer_in_tokens.bin",
				 sam3_dbg_xformer_in_tokens);
		auto_dump_tensor("/tmp/dbg_xformer_layer0_q.bin",
				 sam3_dbg_xformer_layer0_q);
		auto_dump_tensor("/tmp/dbg_pix_ct1_input.bin",
				 sam3_dbg_pix_ct1_input);
		auto_dump_tensor("/tmp/dbg_pix_ct1.bin", sam3_dbg_pix_ct1);
		auto_dump_tensor("/tmp/dbg_pix_skip1_proj.bin",
				 sam3_dbg_pix_skip1);
		auto_dump_tensor("/tmp/dbg_pix_after_skip1.bin",
				 sam3_dbg_pix_after_skip1);
		auto_dump_tensor("/tmp/dbg_tracker_pixel_out.bin",
				 sam3_dbg_pixel_out);
	}
#endif

	/*
	 * Decoder output is [n_masks, final_h, final_w] with
	 * final_h = grid_h * 4 = 288. Pick the argmax-IoU mask and
	 * reshape that single slice to NHWC [1, final_h, final_w, 1]
	 * so it can feed gh_upsample.
	 */
	if (!track_masks || track_masks->n_dims != 3) {
		sam3_log_error("video_add_prompts: bad track_masks shape");
		err = SAM3_EINVAL;
		goto cleanup;
	}
	n_masks = track_masks->dims[0];
	final_h = track_masks->dims[1];
	final_w = track_masks->dims[2];

	/*
	 * Select best mask: when IoU scores are available, use
	 * stability-aware selection (or fall back to argmax when
	 * stability opts are disabled). Done BEFORE occlusion gating
	 * so the result->best_mask field reflects the raw prediction
	 * (Python semantics).
	 */
	if (track_iou && track_iou->n_dims >= 1 &&
	    track_iou->dims[0] >= n_masks &&
	    track_masks && track_masks->n_dims == 3) {
		float delta  = session->opts.multimask_via_stability
			       ? session->opts.multimask_stability_delta
			       : 0.0f;
		float thresh = session->opts.multimask_via_stability
			       ? session->opts.multimask_stability_thresh
			       : 0.0f;
		best_idx = sam3_mask_decoder_select_with_stability(
			(const float *)track_masks->data,
			(const float *)track_iou->data,
			n_masks,
			track_masks->dims[1], /* H */
			track_masks->dims[2], /* W */
			delta, thresh);
	}

	/*
	 * iter_use_prev_mask_pred: stash the selected conditioning mask
	 * logits for this (obj, frame) so a future re-prompt on the same
	 * pair can feed them into the decoder as a dense prompt input.
	 * The decoder-side feed is a follow-up; the stash is cheap and
	 * keeps session->objects[].prev_mask_* consistent with the last
	 * conditioning evaluation.
	 */
	if (session->opts.iter_use_prev_mask_pred &&
	    track_masks && track_masks->n_dims == 3 &&
	    track_masks->dims[0] > best_idx) {
		int H = track_masks->dims[1];
		int W = track_masks->dims[2];
		int slice_dims[3] = {1, H, W};
		struct sam3_tensor *prev = gh_alloc_tensor(
			&session->persist, SAM3_DTYPE_F32, 3, slice_dims);
		if (prev) {
			size_t per_mask = (size_t)H * W * sizeof(float);
			const float *src = (const float *)track_masks->data;
			memcpy(prev->data,
			       src + (size_t)best_idx * (size_t)H * W,
			       per_mask);
			session->objects[obj_idx].prev_mask_logits = prev;
			session->objects[obj_idx].prev_mask_frame = frame_idx;
		} else {
			sam3_log_warn("video_add_prompts: prev_mask clone "
				      "failed (frame %d, obj %d)", frame_idx,
				      obj_idx);
		}
	}

	/*
	 * Python occlusion gating: when object_score_logit <= 0,
	 *   masks     -> NO_OBJ_SCORE
	 *   obj_ptr   -> no_obj_ptr
	 * Done BEFORE the mask upsample/memory-encoder pass so that
	 * the stored memory features see the gated mask.
	 */
	apply_occlusion_gating(track_masks, cond_obj_ptr, NULL,
			       cond_obj_score, trk->no_obj_ptr,
			       trk->no_obj_embed_spatial);

	{
		int dims_nhwc[4] = {1, final_h, final_w, 1};
		const float *src;
		size_t per_mask;

		best_mask_nhwc = gh_alloc_tensor(gfx_scratch,
						 SAM3_DTYPE_F32, 4, dims_nhwc);
		if (!best_mask_nhwc) {
			sam3_log_error("video_add_prompts: best_mask alloc "
				       "failed");
			err = SAM3_ENOMEM;
			goto cleanup;
		}
		src = (const float *)track_masks->data;
		per_mask = (size_t)final_h * (size_t)final_w;
		memcpy(best_mask_nhwc->data,
		       src + (size_t)best_idx * per_mask,
		       per_mask * sizeof(float));
	}

	/*
	 * Python mask_for_mem preprocessing (is_mask_from_pts=True here:
	 * this frame carries a user prompt). sam3_memory_encoder_build
	 * runs with skip_mask_sigmoid-style semantics — no internal
	 * sigmoid — so the caller is responsible for the threshold/scale/
	 * bias transform below.
	 */
	SAM3_PROF_BEGIN(ctx->proc.profiler, "mask_postprocess");
	preprocess_mask_for_mem_enc(best_mask_nhwc,
				    /*is_mask_from_pts=*/1,
				    trk->sigmoid_scale,
				    trk->sigmoid_bias);
	SAM3_PROF_END(ctx->proc.profiler, "mask_postprocess");

	/*
	 * Memory encoder expects masks at interpol_size (1152x1152).
	 * The actual tracker grid size varies with backbone stride, so
	 * compute the upsample factor dynamically. Typical values:
	 *   tracker grid 72 → mask 288 → factor 4
	 *   tracker grid 36 → mask 144 → factor 8
	 */
	sam3_graph_init(&g);
	SAM3_PROF_BEGIN(ctx->proc.profiler, "mask_upsample");
	{
		int up_factor = trk->mem_encoder.interpol_h / final_h;
		if (up_factor < 1)
			up_factor = 1;
		mask_up = gh_upsample(&g, gfx_scratch, best_mask_nhwc, up_factor);
	}
	SAM3_PROF_END(ctx->proc.profiler, "mask_upsample");
	if (!mask_up) {
		sam3_log_error("video_add_prompts: mask upsample failed");
		err = SAM3_ENOMEM;
		goto cleanup;
	}

	/* Memory encoder fuses the downsampled mask (72x72) with the
	 * backbone 1x feature (feat_s1), NOT image_features which is the
	 * 0.5x (36x36) neck output. Python reference uses the highest-
	 * resolution backbone feature here. */
	SAM3_PROF_BEGIN(ctx->proc.profiler, "memenc_build");
	err = sam3_memory_encoder_build(&trk->mem_encoder, &g,
					cf.feat_s1, mask_up,
					gfx_scratch,
					&mem_feat, &mem_pos);
	SAM3_PROF_END(ctx->proc.profiler, "memenc_build");
	if (err != SAM3_OK) {
		sam3_log_error("video_add_prompts: memory encoder build "
			       "failed (%d)", err);
		goto cleanup;
	}

	SAM3_PROF_BEGIN(ctx->proc.profiler, "memenc_eval");
	err = ctx->proc.backend->ops->graph_eval(ctx->proc.backend, &g);
	SAM3_PROF_END(ctx->proc.profiler, "memenc_eval");
	if (err != SAM3_OK) {
		sam3_log_error("video_add_prompts: memory encoder eval "
			       "failed (%d)", err);
		goto cleanup;
	}

	/*
	 * Memory bank expects spatial_features as [HW, mem_dim].
	 * mem_feat from the encoder is [1, H, W, out_dim]; flatten
	 * in-place (same nbytes, just new dims/strides) and clone
	 * into the session persist arena so it survives after the
	 * scratch arena is reset.
	 */
	if (!mem_feat || mem_feat->n_dims != 4) {
		sam3_log_error("video_add_prompts: bad mem_feat shape");
		err = SAM3_EINVAL;
		goto cleanup;
	}

	/*
	 * Python: maskmem_features += (1 - is_obj_appearing) *
	 *                             no_obj_embed_spatial
	 * (sam3_tracker_base.py:843-848). Apply after memory encoder
	 * so the stored entry reflects the occlusion state.
	 */
	apply_occlusion_gating(NULL, NULL, mem_feat,
			       cond_obj_score, trk->no_obj_ptr,
			       trk->no_obj_embed_spatial);
	mem_H  = mem_feat->dims[1];
	mem_W  = mem_feat->dims[2];
	mem_C  = mem_feat->dims[3];
	mem_HW = mem_H * mem_W;

	{
		struct sam3_tensor flat_view = *mem_feat;
		flat_view.n_dims = 2;
		flat_view.dims[0] = mem_HW;
		flat_view.dims[1] = mem_C;
		flat_view.dims[2] = 0;
		flat_view.dims[3] = 0;
		sam3_tensor_compute_strides(&flat_view);
		flat_view.ephemeral = 0;

		spatial_persist = sam3_tensor_clone_persist(
			&session->persist, &flat_view);
	}
	if (!spatial_persist) {
		sam3_log_error("video_add_prompts: spatial clone failed");
		err = SAM3_ENOMEM;
		goto cleanup;
	}

	/*
	 * Commit the prompt to the session FIRST. If the prompt list
	 * is at capacity, sam3_session_add_prompt returns SAM3_ENOMEM
	 * and we bail out before touching the memory bank, so the
	 * session and bank cannot drift out of sync. We overwrite
	 * frame_idx and obj_internal_idx from the pipeline args so
	 * the stored record cannot drift from what actually ran.
	 */
	{
		struct sam3_video_prompt sp = *stored_prompt;
		sp.frame_idx        = frame_idx;
		sp.obj_internal_idx = obj_idx;
		err = sam3_session_add_prompt(session, &sp);
		if (err != SAM3_OK) {
			sam3_log_error("video_add_prompts: prompt store "
				       "failed (%d)", err);
			goto cleanup;
		}
	}

	/*
	 * Prompt stored — now commit the conditioning entry to the
	 * memory bank. Any failure after this point leaves the bank
	 * and prompt list in agreement (one entry each per prompt).
	 *
	 * Persist the obj_ptr (already gated) so it flows into
	 * subsequent frames' memory attention. obj_score uses the
	 * Python eff_iou_score (cal_mem_score) so SAM2Long-style
	 * memory selection consults the occlusion-aware metric.
	 */
	struct sam3_tensor *cond_obj_ptr_persist = NULL;
	if (cond_obj_ptr) {
		cond_obj_ptr_persist = sam3_tensor_clone_persist(
			&session->persist, cond_obj_ptr);
		if (!cond_obj_ptr_persist) {
			sam3_log_error("video_add_prompts: obj_ptr clone failed");
			err = SAM3_ENOMEM;
			goto cleanup;
		}
	}
	{
		float raw_iou = (track_iou && track_iou->data &&
				 track_iou->n_dims >= 1 &&
				 track_iou->dims[0] > best_idx)
				? ((const float *)track_iou->data)[best_idx]
				: 1.0f;
		struct sam3_memory_entry entry;
		memset(&entry, 0, sizeof(entry));
		entry.spatial_features = spatial_persist;
		entry.obj_pointer     = cond_obj_ptr_persist;
		entry.frame_idx        = frame_idx;
		entry.is_conditioning  = 1;
		entry.obj_score        = compute_eff_iou_score(
			cond_obj_score, raw_iou);
		sam3_memory_bank_add(&session->objects[obj_idx].bank, &entry);
	}

	/*
	 * Populate result as sam3_video_frame_result with n_objects=1.
	 * result->objects[0] carries the mask logits for the prompted object.
	 * We surface the best-IoU mask in objects[0].mask (single plane).
	 */
	result->frame_idx = frame_idx;
	result->n_objects = 1;
	result->objects = calloc(1, sizeof(struct sam3_video_object_mask));
	if (!result->objects) {
		sam3_log_error("video_add_prompts: result objects alloc failed");
		err = SAM3_ENOMEM;
		goto cleanup;
	}
	result->objects[0].obj_id = obj_id;
	result->objects[0].mask_h = final_h;
	result->objects[0].mask_w = final_w;
	result->objects[0].iou_score = (track_iou && track_iou->n_dims >= 1 &&
					track_iou->dims[0] > best_idx)
				       ? ((const float *)track_iou->data)[best_idx]
				       : 1.0f;
	result->objects[0].obj_score_logit = (cond_obj_score && cond_obj_score->data)
					     ? ((const float *)cond_obj_score->data)[0]
					     : 0.0f;
	result->objects[0].is_occluded =
		(result->objects[0].obj_score_logit <= 0.0f) ? 1 : 0;

	mask_bytes = (size_t)final_h * (size_t)final_w * sizeof(float);
	result->objects[0].mask = malloc(mask_bytes);
	if (!result->objects[0].mask) {
		sam3_log_error("video_add_prompts: result mask alloc failed");
		err = SAM3_ENOMEM;
		goto cleanup;
	}
	{
		const float *src = (const float *)track_masks->data;
		size_t per_mask = (size_t)final_h * (size_t)final_w;
		memcpy(result->objects[0].mask,
		       src + (size_t)best_idx * per_mask,
		       mask_bytes);
	}

	/* Mark this object's prompted-frame bitmap */
	sam3_session_obj_mark_prompted(session, obj_idx, frame_idx);

	session->frames_tracked[frame_idx] = 1;

	sam3_log_debug("video_add_prompts: frame %d, obj %d, %d prompts, "
		       "mask %dx%d, bank n_cond=%d",
		       frame_idx, obj_id, n_prompts, final_h, final_w,
		       session->objects[obj_idx].bank.n_cond);

cleanup:
	if (err != SAM3_OK)
		sam3_video_frame_result_free(result);
	(void)mem_pos;  /* returned by encoder but not used here */
	if (ctx)
		SAM3_PROF_END(ctx->proc.profiler, "video_add_prompts");
	return err;
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
 * video_propagate_pure_tracking_obj - Run the tracker on a non-prompted
 *                                     frame for one specific object and
 *                                     commit a non-conditioning memory
 *                                     bank entry to its bank.
 *
 * @session:  Video session
 * @obj_idx:  Internal object index (index into session->objects[])
 * @f:        Frame index (0..nf-1, validated by caller)
 * @obj_mask: Output per-object mask. Caller zeros up-front; on success
 *            obj_mask->mask is malloc'd and the caller owns it.
 *
 * The flow mirrors video_add_prompts_pipeline but passes NULL for the
 * sparse prompt (pure tracking), appends the memory entry with
 * is_conditioning=0 to session->objects[obj_idx].bank, and writes
 * the best-IoU mask into obj_mask. Object pointers are cloned into
 * session->persist so the bank survives the next graph arena reset.
 */
static enum sam3_error
video_propagate_pure_tracking_obj(struct sam3_video_session *session,
				  int obj_idx, int f,
				  const struct propagate_frame_ctx *shared,
				  struct sam3_video_object_mask *obj_mask)
{
	struct sam3_ctx *ctx;
	struct sam3_tracker *trk;
	struct sam3_frame_features cf;
	struct sam3_arena *gfx_scratch;
	struct sam3_tensor *img_2d = NULL;
	struct sam3_tensor *track_masks = NULL;
	struct sam3_tensor *track_iou = NULL;
	struct sam3_tensor *obj_ptr = NULL;
	struct sam3_tensor *best_mask_nhwc = NULL;
	struct sam3_tensor *mask_up = NULL;
	struct sam3_tensor *mem_feat = NULL;
	struct sam3_tensor *mem_pos = NULL;
	struct sam3_tensor *spatial_persist = NULL;
	struct sam3_tensor *obj_ptr_persist = NULL;
	struct sam3_graph g;
	int grid_h = 0, grid_w = 0;
	int n_masks = 0, final_h = 0, final_w = 0;
	int best_idx = 0;
	float best_iou_value = 1.0f;
	int mem_H, mem_W, mem_C, mem_HW;
	enum sam3_error err = SAM3_OK;

	ctx = session->ctx;
	if (!ctx || !ctx->proc_ready || !ctx->proc.backend) {
		sam3_log_error("video_propagate_obj: context not ready");
		err = SAM3_EINVAL;
		goto cleanup;
	}

	trk = &session->tracker;
	if (shared) {
		cf = shared->cf;
	} else {
		memset(&cf, 0, sizeof(cf));
		SAM3_PROF_BEGIN(ctx->proc.profiler, "frame_cache_get");
		err = sam3_frame_cache_get(&session->frame_cache, f, &cf);
		SAM3_PROF_END(ctx->proc.profiler, "frame_cache_get");
		if (err != SAM3_OK) {
			sam3_log_error("video_propagate_obj: frame %d cache get "
				       "failed (%d)", f, err);
			goto cleanup;
		}
	}
	if (!cf.image_features || !cf.feat_s0 || !cf.feat_s1) {
		sam3_log_error("video_propagate_obj: frame %d features missing",
			       f);
		err = SAM3_EINVAL;
		goto cleanup;
	}

	/* Same 3 GiB scratch as the conditioning path. When @shared is set,
	 * propagate_one already flattened feat_s1 into this arena and marked
	 * the tail — reset to that mark so the shared img_2d survives. */
	gfx_scratch = &ctx->proc.scratch_arena;
	if (shared)
		gfx_scratch->offset = shared->scratch_mark;
	else
		sam3_arena_reset(gfx_scratch);

	/* Flatten NHWC backbone features → [HW, d] for tracker main.
	 * Python reference operates at main=72x72 (image_size/stride).
	 * Cache layout: feat_s1 = neck_1x = 72x72 1x feature.
	 *
	 * Shared-frame path: reuse the flatten propagate_one already did. */
	if (shared) {
		img_2d = shared->img_2d;
		grid_h = shared->grid_h;
		grid_w = shared->grid_w;
	} else {
		struct sam3_tensor *img = cf.feat_s1;
		int HW, d;
		int dims2[2];

		if (img->n_dims != 4) {
			sam3_log_error("video_propagate_obj: feat_s1 expected "
				       "4D, got %d", img->n_dims);
			err = SAM3_EINVAL;
			goto cleanup;
		}
		grid_h = img->dims[1];
		grid_w = img->dims[2];
		HW = grid_h * grid_w;
		d  = img->dims[3];
		dims2[0] = HW;
		dims2[1] = d;
		img_2d = gh_alloc_tensor(gfx_scratch, img->dtype, 2, dims2);
		if (!img_2d) {
			sam3_log_error("video_propagate_obj: img_2d alloc failed");
			err = SAM3_ENOMEM;
			goto cleanup;
		}
		memcpy(img_2d->data, img->data, img->nbytes);
	}

	/* Tracker graph: pure tracking — prompt=NULL, is_cond=0.
	 * Skip-feature argument order matches the conditioning path
	 * (main=feat_s1 at 72x72). */
	struct sam3_tensor *track_obj_score = NULL;
	sam3_graph_init(&g);
	SAM3_PROF_BEGIN(ctx->proc.profiler, "tracker_build");
	err = sam3_tracker_track_frame(trk, &g,
				       &session->objects[obj_idx].bank,
				       img_2d, grid_h, grid_w,
				       /*prompt=*/NULL,
				       cf.feat_4x, cf.feat_s0,
				       f, /*is_cond=*/0,
				       gfx_scratch,
				       &track_masks, &track_iou,
				       &obj_ptr, &track_obj_score);
	SAM3_PROF_END(ctx->proc.profiler, "tracker_build");
	if (err != SAM3_OK) {
		sam3_log_error("video_propagate_obj: track_frame failed "
			       "(frame %d, obj %d, err=%d)", f, obj_idx, err);
		goto cleanup;
	}

	SAM3_PROF_BEGIN(ctx->proc.profiler, "tracker_eval");
	err = ctx->proc.backend->ops->graph_eval(ctx->proc.backend, &g);
	SAM3_PROF_END(ctx->proc.profiler, "tracker_eval");
	if (err != SAM3_OK) {
		sam3_log_error("video_propagate_obj: tracker eval failed "
			       "(frame %d, obj %d, err=%d)", f, obj_idx, err);
		goto cleanup;
	}

	if (!track_masks || track_masks->n_dims != 3) {
		sam3_log_error("video_propagate_obj: bad track_masks shape "
			       "(frame %d, obj %d)", f, obj_idx);
		err = SAM3_EINVAL;
		goto cleanup;
	}
	n_masks = track_masks->dims[0];
	final_h = track_masks->dims[1];
	final_w = track_masks->dims[2];

	/* Select best mask using stability-aware pick — before gating. */
	if (track_iou && track_iou->n_dims >= 1 &&
	    track_iou->dims[0] >= n_masks &&
	    track_masks && track_masks->n_dims == 3) {
		float delta  = session->opts.multimask_via_stability
			       ? session->opts.multimask_stability_delta
			       : 0.0f;
		float thresh = session->opts.multimask_via_stability
			       ? session->opts.multimask_stability_thresh
			       : 0.0f;
		best_idx = sam3_mask_decoder_select_with_stability(
			(const float *)track_masks->data,
			(const float *)track_iou->data,
			n_masks,
			track_masks->dims[1], /* H */
			track_masks->dims[2], /* W */
			delta, thresh);
		/* best_iou_value reflects the selected mask's IoU, regardless
		 * of whether it was picked via stability or argmax fallback. */
		best_iou_value = ((const float *)track_iou->data)[best_idx];
	}

	/*
	 * Python occlusion gating: masks -> NO_OBJ_SCORE, obj_ptr ->
	 * no_obj_ptr when object_score_logit <= 0. Done before the
	 * upsample/memory-encoder pass.
	 */
	apply_occlusion_gating(track_masks, obj_ptr, NULL,
			       track_obj_score, trk->no_obj_ptr,
			       trk->no_obj_embed_spatial);

	/* Extract the best mask into [1, final_h, final_w, 1] NHWC. */
	{
		int dims_nhwc[4] = {1, final_h, final_w, 1};
		const float *src;
		size_t per_mask;

		best_mask_nhwc = gh_alloc_tensor(gfx_scratch,
						 SAM3_DTYPE_F32, 4, dims_nhwc);
		if (!best_mask_nhwc) {
			sam3_log_error("video_propagate_obj: best_mask alloc "
				       "failed (frame %d, obj %d)", f, obj_idx);
			err = SAM3_ENOMEM;
			goto cleanup;
		}
		src = (const float *)track_masks->data;
		per_mask = (size_t)final_h * (size_t)final_w;
		memcpy(best_mask_nhwc->data,
		       src + (size_t)best_idx * per_mask,
		       per_mask * sizeof(float));
	}

	/*
	 * Python mask_for_mem preprocessing (is_mask_from_pts=False here:
	 * propagation frames never carry user inputs in the C API). The
	 * encoder runs without an internal sigmoid.
	 */
	SAM3_PROF_BEGIN(ctx->proc.profiler, "mask_postprocess");
	preprocess_mask_for_mem_enc(best_mask_nhwc,
				    /*is_mask_from_pts=*/0,
				    trk->sigmoid_scale,
				    trk->sigmoid_bias);
	SAM3_PROF_END(ctx->proc.profiler, "mask_postprocess");

	/* Memory encoder: upsample to interpol_size then encoder graph. */
	sam3_graph_init(&g);
	SAM3_PROF_BEGIN(ctx->proc.profiler, "mask_upsample");
	{
		int up_factor = trk->mem_encoder.interpol_h / final_h;
		if (up_factor < 1)
			up_factor = 1;
		mask_up = gh_upsample(&g, gfx_scratch, best_mask_nhwc, up_factor);
	}
	SAM3_PROF_END(ctx->proc.profiler, "mask_upsample");
	if (!mask_up) {
		sam3_log_error("video_propagate_obj: mask upsample failed "
			       "(frame %d, obj %d)", f, obj_idx);
		err = SAM3_ENOMEM;
		goto cleanup;
	}

	/* Memory encoder uses 1x feature (feat_s1) — see note on the
	 * conditioning path above. */
	SAM3_PROF_BEGIN(ctx->proc.profiler, "memenc_build");
	err = sam3_memory_encoder_build(&trk->mem_encoder, &g,
					cf.feat_s1, mask_up,
					gfx_scratch,
					&mem_feat, &mem_pos);
	SAM3_PROF_END(ctx->proc.profiler, "memenc_build");
	if (err != SAM3_OK) {
		sam3_log_error("video_propagate_obj: memory encoder build "
			       "failed (frame %d, obj %d, err=%d)",
			       f, obj_idx, err);
		goto cleanup;
	}

	SAM3_PROF_BEGIN(ctx->proc.profiler, "memenc_eval");
	err = ctx->proc.backend->ops->graph_eval(ctx->proc.backend, &g);
	SAM3_PROF_END(ctx->proc.profiler, "memenc_eval");
	if (err != SAM3_OK) {
		sam3_log_error("video_propagate_obj: memory encoder eval "
			       "failed (frame %d, obj %d, err=%d)",
			       f, obj_idx, err);
		goto cleanup;
	}

	if (!mem_feat || mem_feat->n_dims != 4) {
		sam3_log_error("video_propagate_obj: bad mem_feat shape "
			       "(frame %d, obj %d)", f, obj_idx);
		err = SAM3_EINVAL;
		goto cleanup;
	}

	/* Python: maskmem_features += (1 - is_obj_appearing) *
	 *                             no_obj_embed_spatial */
	apply_occlusion_gating(NULL, NULL, mem_feat,
			       track_obj_score, trk->no_obj_ptr,
			       trk->no_obj_embed_spatial);
	mem_H  = mem_feat->dims[1];
	mem_W  = mem_feat->dims[2];
	mem_C  = mem_feat->dims[3];
	mem_HW = mem_H * mem_W;

	/* Flatten [1,H,W,C] -> [HW,C] and clone to session->persist so
	 * the entry survives the next scratch reset. */
	{
		struct sam3_tensor flat_view = *mem_feat;
		flat_view.n_dims = 2;
		flat_view.dims[0] = mem_HW;
		flat_view.dims[1] = mem_C;
		flat_view.dims[2] = 0;
		flat_view.dims[3] = 0;
		sam3_tensor_compute_strides(&flat_view);
		flat_view.ephemeral = 0;

		spatial_persist = sam3_tensor_clone_persist(
			&session->persist, &flat_view);
	}
	if (!spatial_persist) {
		sam3_log_error("video_propagate_obj: spatial clone failed "
			       "(frame %d, obj %d)", f, obj_idx);
		err = SAM3_ENOMEM;
		goto cleanup;
	}

	/* Clone the object pointer too — tracker returns it on the
	 * scratch arena so we must copy it before committing to the bank. */
	if (obj_ptr) {
		obj_ptr_persist = sam3_tensor_clone_persist(
			&session->persist, obj_ptr);
		if (!obj_ptr_persist) {
			sam3_log_error("video_propagate_obj: obj_ptr clone "
				       "failed (frame %d, obj %d)",
				       f, obj_idx);
			err = SAM3_ENOMEM;
			goto cleanup;
		}
	}

	{
		struct sam3_memory_entry entry;
		memset(&entry, 0, sizeof(entry));
		entry.spatial_features = spatial_persist;
		entry.obj_pointer     = obj_ptr_persist;
		entry.frame_idx        = f;
		entry.is_conditioning  = 0;
		/*
		 * Python cal_mem_score:
		 *   obj_score_norm = (sigmoid(s)*2 - 1 if s > 0 else 0)
		 *   eff_iou        = obj_score_norm * iou_score
		 * Used by SAM3-Long memory selection to threshold frames.
		 */
		entry.obj_score        = compute_eff_iou_score(
			track_obj_score, best_iou_value);
		sam3_memory_bank_add(&session->objects[obj_idx].bank, &entry);
	}

	/* Populate the per-object output mask. */
	obj_mask->mask_h = final_h;
	obj_mask->mask_w = final_w;
	obj_mask->iou_score = best_iou_value;
	obj_mask->obj_score_logit = (track_obj_score && track_obj_score->data)
				    ? ((const float *)track_obj_score->data)[0]
				    : 0.0f;
	obj_mask->is_occluded = (obj_mask->obj_score_logit <= 0.0f) ? 1 : 0;
	{
		size_t mask_bytes = (size_t)final_h * (size_t)final_w *
				    sizeof(float);
		obj_mask->mask = malloc(mask_bytes);
		if (!obj_mask->mask) {
			sam3_log_error("video_propagate_obj: obj_mask alloc "
				       "failed (frame %d, obj %d)",
				       f, obj_idx);
			err = SAM3_ENOMEM;
			goto cleanup;
		}
		const float *src = (const float *)track_masks->data;
		size_t per_mask = (size_t)final_h * (size_t)final_w;
		memcpy(obj_mask->mask,
		       src + (size_t)best_idx * per_mask,
		       mask_bytes);
	}

	session->frames_tracked[f] = 1;

	sam3_log_debug("video_propagate_obj: frame %d obj %d tracked, "
		       "mask %dx%d, best_iou %.3f, bank n_cond=%d n_nc=%d",
		       f, obj_idx, final_h, final_w,
		       (double)best_iou_value,
		       session->objects[obj_idx].bank.n_cond,
		       session->objects[obj_idx].bank.n_non_cond);

cleanup:
	if (err != SAM3_OK) {
		free(obj_mask->mask);
		obj_mask->mask = NULL;
		obj_mask->mask_h = 0;
		obj_mask->mask_w = 0;
	}
	(void)mem_pos;
	return err;
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
	 * Build per-frame shared state: flatten feat_s1 once and capture
	 * the arena offset. Each per-object call resets scratch to this
	 * mark instead of 0, so the flatten is paid once per frame rather
	 * than once per object. On multi-object clips (8 objs) this drops
	 * 7 redundant ~1.3 MB memcpys plus the matching arena churn.
	 *
	 * Fall back to per-object flatten (shared = NULL) if context is
	 * not ready, cache lookup fails, or feat_s1 has an unexpected
	 * shape — keeps correctness paths identical to the legacy code.
	 */
	struct sam3_ctx *ctx = session->ctx;
	struct propagate_frame_ctx fctx;
	const struct propagate_frame_ctx *shared = NULL;
	if (ctx && ctx->proc_ready && ctx->proc.backend) {
		memset(&fctx, 0, sizeof(fctx));
		SAM3_PROF_BEGIN(prof, "frame_cache_get");
		enum sam3_error cf_err = sam3_frame_cache_get(
			&session->frame_cache, f, &fctx.cf);
		SAM3_PROF_END(prof, "frame_cache_get");
		if (cf_err == SAM3_OK &&
		    fctx.cf.image_features &&
		    fctx.cf.feat_s0 &&
		    fctx.cf.feat_s1 &&
		    fctx.cf.feat_s1->n_dims == 4) {
			struct sam3_arena *scratch = &ctx->proc.scratch_arena;
			sam3_arena_reset(scratch);

			struct sam3_tensor *img = fctx.cf.feat_s1;
			fctx.grid_h = img->dims[1];
			fctx.grid_w = img->dims[2];
			int dims2[2] = {fctx.grid_h * fctx.grid_w,
					img->dims[3]};
			fctx.img_2d = gh_alloc_tensor(scratch, img->dtype,
						      2, dims2);
			if (fctx.img_2d) {
				memcpy(fctx.img_2d->data, img->data,
				       img->nbytes);
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
