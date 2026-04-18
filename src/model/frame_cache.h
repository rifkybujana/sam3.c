/*
 * src/model/frame_cache.h - Tiered LRU cache for encoded frame features
 *
 * Replaces eager full-video encoding. Each frame's encoded features
 * (image_features + feat_s0 + feat_s1) live in one of three tiers:
 *
 *   1. Backend tier — held in a dedicated bump arena, ready for
 *      tracker consumption. Default budget 4 GiB.
 *   2. CPU spill — raw byte copies on the host (malloc'd). Default
 *      budget 16 GiB; 0 disables the tier. Promotion back to backend
 *      is currently done by recompute (memcpy fast path is a future
 *      optimization).
 *   3. Recompute — runs the image encoder on a miss. Automatic
 *      fallback when both tiers cannot hold the frame.
 *
 * LRU eviction within each tier. The cache is invisible to correctness:
 * callers receive fully-encoded features regardless of which tier
 * served the request.
 *
 * Key types:  sam3_frame_cache, sam3_frame_features
 * Depends on: core/tensor.h, core/alloc.h, sam3/sam3_types.h
 * Used by:    model/video_session.h, model/sam3_video.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_FRAME_CACHE_H
#define SAM3_MODEL_FRAME_CACHE_H

#include <stddef.h>
#include <stdint.h>

#include "sam3/sam3_types.h"
#include "core/tensor.h"
#include "core/alloc.h"

struct sam3_video_session; /* forward decl */

/*
 * sam3_frame_features - Encoded features for one frame.
 *
 * All three tensors are owned by the cache when the serving tier is
 * backend; the caller must not free them. When served from a miss
 * path, the tensors are allocated in the cache's arena for the
 * lifetime of the cache slot.
 */
struct sam3_frame_features {
	struct sam3_tensor *image_features; /* neck_05x: [1, 36, 36, 256] (0.5x) */
	struct sam3_tensor *feat_s0;        /* neck_2x:  [1, 144, 144, 256] (2x)  */
	struct sam3_tensor *feat_s1;        /* neck_1x:  [1, 72,  72,  256] (1x)  */
	struct sam3_tensor *feat_4x;        /* neck_4x:  [1, 288, 288, 256] (4x)  */
};

/*
 * sam3_frame_cache_encode_fn - Image encoder hook.
 *
 * Called on cache misses. Runs the image encoder for the requested
 * frame and writes the three tensors into @out. Tensor allocation
 * MUST use @arena.
 *
 * @session:   Session pointer passed through for callback state.
 * @frame_idx: Frame index to encode.
 * @arena:     Arena for tensor allocation.
 * @out:       Output features.
 *
 * Returns SAM3_OK on success; SAM3_ENOMEM if arena is exhausted;
 * SAM3_EMODEL if the graph eval fails.
 */
typedef enum sam3_error (*sam3_frame_cache_encode_fn)(
	struct sam3_video_session *session,
	int frame_idx,
	struct sam3_arena *arena,
	struct sam3_frame_features *out);

enum sam3_frame_tier {
	SAM3_FRAME_TIER_NONE      = 0,
	SAM3_FRAME_TIER_BACKEND   = 1,
	SAM3_FRAME_TIER_CPU_SPILL = 2,
};

struct sam3_frame_cache_slot {
	int                  frame_idx;       /* equals slot array index */
	enum sam3_frame_tier tier;
	/* Backend tier tensor pointers. NULL when tier != BACKEND. */
	struct sam3_tensor  *image_features;
	struct sam3_tensor  *feat_s0;
	struct sam3_tensor  *feat_s1;
	struct sam3_tensor  *feat_4x;
	/* Spill tier byte buffers (same layout and nbytes as backend). */
	void                *spill_image_features;
	void                *spill_feat_s0;
	void                *spill_feat_s1;
	void                *spill_feat_4x;
	size_t               spill_bytes;     /* total bytes across all 4 */
	uint64_t             last_access_seq; /* LRU bookkeeping */
};

struct sam3_frame_cache {
	struct sam3_frame_cache_slot *slots;   /* [n_frames] */
	int                           n_frames;
	size_t                        backend_budget;
	size_t                        backend_used;
	size_t                        spill_budget;
	size_t                        spill_used;
	uint64_t                      access_counter;
	/*
	 * Hit/miss accounting. A "hit" means the requested frame is in the
	 * backend tier and required no encode; a "miss" means the encode
	 * hook ran (cold or post-eviction); a "spill_promote" means the
	 * frame was on CPU spill and was promoted back to the backend.
	 * Reset to 0 in sam3_frame_cache_init.
	 */
	uint64_t                      n_hits;
	uint64_t                      n_misses;
	uint64_t                      n_spill_promotes;
	struct sam3_arena             backend_arena;
	struct sam3_video_session    *owner;
	sam3_frame_cache_encode_fn    encode;
};

/*
 * sam3_frame_cache_init - Allocate slots + backend arena.
 *
 * @cache:           Output cache (caller-allocated, zeroed before call).
 * @owner:           Session, passed to @encode on misses.
 * @encode:          Encoder hook (must be non-NULL).
 * @n_frames:        Total frames in the video (> 0).
 * @backend_budget:  Bytes for backend arena. 0 → 4 GiB default.
 * @spill_budget:    Bytes for CPU spill. 0 → 16 GiB default; SIZE_MAX
 *                   to disable spill (evicted slots become NONE).
 *
 * Returns SAM3_OK on success; SAM3_EINVAL for bad args; SAM3_ENOMEM
 * if arena allocation fails.
 */
enum sam3_error sam3_frame_cache_init(struct sam3_frame_cache *cache,
				      struct sam3_video_session *owner,
				      sam3_frame_cache_encode_fn encode,
				      int n_frames,
				      size_t backend_budget,
				      size_t spill_budget);

/*
 * sam3_frame_cache_get - Fetch features for a frame.
 *
 * @cache:  Initialized cache.
 * @frame:  Frame index (0 <= frame < n_frames).
 * @out:    Output features (pointers into cache-owned tensors).
 *
 * On miss, runs @cache->encode and promotes the result to the backend
 * tier (evicting LRU slots to spill as needed). Updates LRU on every
 * call regardless of tier.
 *
 * Returns SAM3_OK on success; SAM3_EINVAL on bad args; SAM3_ENOMEM
 * if neither tier nor recompute can satisfy the request.
 */
enum sam3_error sam3_frame_cache_get(struct sam3_frame_cache *cache,
				     int frame,
				     struct sam3_frame_features *out);

/*
 * sam3_frame_cache_release - Free slots, spill buffers, backend arena.
 *
 * @cache: Cache to release (safe on NULL or already-zeroed cache).
 *
 * After return the struct is zeroed; calling again is safe.
 */
void sam3_frame_cache_release(struct sam3_frame_cache *cache);

/*
 * sam3_frame_cache_invalidate - Drop all cached features in place.
 *
 * Resets every slot to TIER_NONE, frees spill buffers, and resets
 * the backend arena. Configuration (budgets, encode hook, n_frames)
 * is preserved. Not called by sam3_video_reset — video reset
 * preserves the cache per spec §3.3.
 */
void sam3_frame_cache_invalidate(struct sam3_frame_cache *cache);

#endif /* SAM3_MODEL_FRAME_CACHE_H */
