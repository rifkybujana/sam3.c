/*
 * src/model/video_session.h - Video tracking session state
 *
 * Manages the internal state for a video tracking session: tracker
 * instance, video frames, per-object state array, per-frame feature
 * cache, prompt list + prompted-frame bitmap, and memory arenas. This
 * is the implementation behind the public sam3_video_* API declared in
 * sam3.h.
 *
 * Key types:  sam3_video_session, sam3_video_object, sam3_video_prompt
 * Depends on: model/tracker.h, model/tracker_multiplex.h, model/memory_bank.h,
 *             model/frame_cache.h, util/video.h, sam3/sam3.h,
 *             sam3/sam3_types.h, core/alloc.h
 * Used by:    model/sam3_video.c, tests/test_video_session.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_VIDEO_SESSION_H
#define SAM3_MODEL_VIDEO_SESSION_H

#include <stdint.h>

#include "sam3/sam3.h"
#include "sam3/sam3_types.h"
#include "model/tracker.h"
#include "model/tracker_multiplex.h"
#include "model/memory_bank.h"
#include "model/frame_cache.h"
#include "util/video.h"
#include "core/alloc.h"

/* Maximum number of frames we support per session */
#define SAM3_MAX_SESSION_FRAMES 4096

struct sam3_ctx;  /* forward decl (defined in model/sam3_internal.h) */

/*
 * Internal prompt-kind enum for stored video prompts.
 *
 * Note: the public header sam3/sam3_types.h already defines
 * enum sam3_prompt_type { SAM3_PROMPT_POINT, SAM3_PROMPT_BOX, ... }.
 * C enumerators are not scoped inside their enum, so this internal
 * enum deliberately uses SAM3_PROMPT_KIND_BOX (not SAM3_PROMPT_BOX)
 * to avoid a redeclaration conflict. SAM3_PROMPT_POINTS (plural) does
 * not collide with the public SAM3_PROMPT_POINT (singular), so the
 * naming is intentionally asymmetric.
 */
enum sam3_prompt_kind {
	SAM3_PROMPT_POINTS   = 0,
	SAM3_PROMPT_KIND_BOX = 1,
};

/* A single stored prompt for a given (frame, object) pair. */
struct sam3_video_prompt {
	int frame_idx;
	int obj_internal_idx;
	enum sam3_prompt_kind kind;
	union {
		struct {
			int   n;
			float xys[SAM3_MAX_POINTS_PER_OBJ * 2];
			int   labels[SAM3_MAX_POINTS_PER_OBJ];
		} points;
		struct sam3_box box;
	} data;
};

/*
 * sam3_video_object - Per-object tracking state.
 *
 * Each user-added object owns its own memory bank, per-object prompt
 * bitmap, and prev-mask cache slot (the latter used by
 * iter_use_prev_mask_pred in Phase 5). Banks are independent so one
 * object's occlusion cannot pollute another's tracking.
 *
 * prompted_frames is a lazy-allocated bitmap sized [(n_frames+7)/8].
 * NULL means "no frames prompted for this object yet."
 *
 * prev_mask_logits is an arena-allocated tensor (persist arena), valid
 * when prev_mask_frame >= 0 and equals the frame for which the logits
 * were produced. Phase 5 wires it into the mask decoder's dense-prompt
 * input on re-prompt.
 */
struct sam3_video_object {
	int                      obj_id;           /* user-supplied id */
	struct sam3_memory_bank  bank;
	uint8_t                 *prompted_frames;  /* bitmap, [(n_frames+7)/8] */
	struct sam3_tensor      *prev_mask_logits; /* [1,H,W,n_masks] or NULL */
	int                      prev_mask_frame;  /* -1 if no prev mask */
};

struct sam3_video_session {
	/* Parent context (for image encoder access during tracking) */
	struct sam3_ctx *ctx;

	/* Tracker module (owns sub-modules and memory bank) */
	struct sam3_tracker tracker;

	/*
	 * SAM 3.1 variant-specific tracker. Exactly one of `tracker` (SAM 3)
	 * or `tracker_multiplex` (SAM 3.1) is populated for the session lifetime;
	 * `variant` selects which. Kept as parallel fields rather than a
	 * C union so the per-frame pipelines can reference sub-module
	 * addresses without type punning.
	 */
	struct sam3_tracker_multiplex tracker_multiplex;
	int                    variant;  /* enum sam3_variant */

	/* Loaded video frames */
	struct sam3_video_frames frames;

	/*
	 * Per-object tracking state. Filled by sam3_session_get_or_add_obj;
	 * compacted by sam3_session_remove_obj. n_objects is the count of
	 * live objects (indices 0..n_objects-1).
	 *
	 * Each object owns its bank (objects[i].bank). The tracker no longer
	 * has a global bank (Task 2.2 removed sam3_tracker::mem_bank).
	 */
	struct sam3_video_object objects[SAM3_MAX_OBJECTS];
	int n_objects;

	/*
	 * Tiered LRU frame cache: encodes on miss, evicts LRU to spill or
	 * recompute. Replaces the old eager cached_features[] array.
	 * Initialized by sam3_video_start_ex; released by sam3_video_end.
	 * sam3_video_reset PRESERVES this cache (spec §3.3).
	 */
	struct sam3_frame_cache frame_cache;

	/* Copy of start_ex opts (defaults applied; set before cache init). */
	struct sam3_video_start_opts opts;

	/* Per-frame tracking status */
	int *frames_tracked;  /* 1 = already tracked, 0 = not yet */

	/* Stored prompts (arena-owned; sized by sam3_video_start) */
	struct sam3_video_prompt *prompts;
	int                       n_prompts;
	int                       cap_prompts;

	/* Prompted-frame bitmap: 1 byte per frame (arena-owned) */
	uint8_t *prompted_frames;

	/* Arenas */
	struct sam3_arena persist;  /* session lifetime: weights, features, memory */
	struct sam3_arena scratch;  /* per-frame: reset between frames */

	/* Whether the tracker has been loaded with weights */
	int loaded;

	/*
	 * Set to 1 while sam3_video_propagate is executing its sweep;
	 * cleared on return. Guards sam3_video_remove_object and
	 * sam3_video_reset against being called from inside a callback
	 * (spec §6.2 cases 2 and 14).
	 */
	int in_propagate;
};

/*
 * sam3_session_get_or_add_obj - Map a user-facing object ID to an internal index.
 *
 * @session: Video session
 * @obj_id:  User-provided object identifier
 *
 * If obj_id already exists, returns its index. Otherwise adds it and
 * returns the new index. Returns -1 if the session is full (SAM3_MAX_OBJECTS)
 * or if session is NULL.
 */
int sam3_session_get_or_add_obj(struct sam3_video_session *session, int obj_id);

/*
 * sam3_session_remove_obj - Remove a tracked object by ID.
 *
 * @session: Video session
 * @obj_id:  Object ID to remove
 *
 * Shifts remaining objects down to fill the gap.
 * Returns SAM3_OK on success, SAM3_EINVAL if obj_id not found or
 * session is NULL.
 */
enum sam3_error sam3_session_remove_obj(struct sam3_video_session *session,
					int obj_id);

/*
 * sam3_session_add_prompt - Append a prompt to the session and mark its frame.
 *
 * @s: Video session. Must have a non-NULL `prompts` array with
 *     `cap_prompts > 0`. When `prompted_frames` is non-NULL, the
 *     bitmap entry for `p->frame_idx` is set to 1.
 * @p: Prompt to copy in (by value).
 *
 * Contract: the caller is responsible for ensuring `p->obj_internal_idx`
 * is valid for the session (e.g. obtained via `sam3_session_get_or_add_obj`
 * before calling this helper). This helper does not verify it, to keep
 * the append path usable in test contexts that pre-allocate storage
 * without populating `n_objects`.
 *
 * Returns SAM3_OK on success, SAM3_EINVAL on NULL args or out-of-range
 * frame_idx, SAM3_ENOMEM if the prompt list is at capacity.
 */
int sam3_session_add_prompt(struct sam3_video_session *s,
			    const struct sam3_video_prompt *p);

/*
 * sam3_session_clear_prompts - Drop all stored prompts and reset the bitmap.
 *
 * NULL-safe. Zeroes n_prompts and, if both prompted_frames and
 * frames.n_frames are set, zeroes the bitmap.
 */
void sam3_session_clear_prompts(struct sam3_video_session *s);

/*
 * sam3_session_is_prompted - Check whether a frame has at least one prompt.
 *
 * @s:         Video session (NULL returns 0)
 * @frame_idx: Frame index (out-of-range returns 0)
 *
 * Returns 1 if the prompted-frame bitmap marks this frame, 0 otherwise.
 * Treats a NULL bitmap as "no frames prompted" so stack-initialized
 * sessions in tests keep working.
 */
int sam3_session_is_prompted(const struct sam3_video_session *s,
			     int frame_idx);

/*
 * sam3_session_obj_is_prompted - Per-object prompted-frame check.
 *
 * @s:         Video session (NULL returns 0)
 * @obj_idx:   Internal object index (out-of-range returns 0)
 * @frame_idx: Frame index (out-of-range returns 0)
 *
 * Returns 1 if this object has at least one stored prompt on this frame,
 * 0 otherwise. Mirrors sam3_session_is_prompted but at object granularity
 * for Phase 2 per-object propagation.
 */
int sam3_session_obj_is_prompted(const struct sam3_video_session *s,
				 int obj_idx, int frame_idx);

/*
 * sam3_session_obj_mark_prompted - Set the per-object prompted bit.
 *
 * @s:         Video session (NULL -> no-op)
 * @obj_idx:   Internal object index (out-of-range -> no-op)
 * @frame_idx: Frame index (out-of-range -> no-op)
 *
 * Lazily allocates the per-object bitmap on first call. The bitmap is
 * sized at frames.n_frames bits rounded up to bytes; allocations happen
 * via malloc and are freed in sam3_session_remove_obj.
 */
int sam3_session_obj_mark_prompted(struct sam3_video_session *s,
				   int obj_idx, int frame_idx);

#endif /* SAM3_MODEL_VIDEO_SESSION_H */
