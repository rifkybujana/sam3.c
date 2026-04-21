/*
 * src/model/frame_cache.c - Tiered LRU cache for encoded frame features
 *
 * Implements the public API declared in model/frame_cache.h. Backend
 * tier uses a dedicated bump arena; CPU spill keeps evicted bytes on
 * the host. Promotion from spill to backend is a direct memcpy into a
 * fresh arena allocation — tensor header metadata captured at spill
 * time lets the cache rebuild the backend tensors without invoking
 * the encoder again.
 *
 * Key types:  sam3_frame_cache, sam3_frame_features
 * Depends on: model/frame_cache.h, util/log.h
 * Used by:    model/sam3_video.c (via the session)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "model/frame_cache.h"
#include "util/log.h"

#define DEFAULT_BACKEND_BUDGET ((size_t)4  * 1024 * 1024 * 1024) /* 4 GiB  */
#define DEFAULT_SPILL_BUDGET   ((size_t)16 * 1024 * 1024 * 1024) /* 16 GiB */

/* --- sam3_frame_cache_init --- */

enum sam3_error sam3_frame_cache_init(struct sam3_frame_cache *cache,
				      struct sam3_video_session *owner,
				      sam3_frame_cache_encode_fn encode,
				      int n_frames,
				      size_t backend_budget,
				      size_t spill_budget)
{
	if (!cache || !encode || n_frames <= 0)
		return SAM3_EINVAL;

	memset(cache, 0, sizeof(*cache));
	cache->owner  = owner;
	cache->encode = encode;
	cache->n_frames = n_frames;
	cache->backend_budget = backend_budget ? backend_budget
					       : DEFAULT_BACKEND_BUDGET;
	cache->spill_budget   = spill_budget   ? spill_budget
					       : DEFAULT_SPILL_BUDGET;

	cache->slots = calloc((size_t)n_frames, sizeof(*cache->slots));
	if (!cache->slots) {
		sam3_log_error("frame_cache_init: slots alloc failed (%d)",
			       n_frames);
		return SAM3_ENOMEM;
	}
	for (int i = 0; i < n_frames; i++)
		cache->slots[i].frame_idx = i;

	enum sam3_error err = sam3_arena_init(&cache->backend_arena,
					      cache->backend_budget);
	if (err != SAM3_OK) {
		free(cache->slots);
		cache->slots = NULL;
		sam3_log_error("frame_cache_init: arena init failed (%zu B)",
			       cache->backend_budget);
		return err;
	}
	return SAM3_OK;
}

/* --- Internal helpers --- */

static size_t slot_backend_bytes(const struct sam3_frame_cache_slot *s)
{
	size_t b = 0;
	if (s->image_features)        b += s->image_features->nbytes;
	if (s->feat_s0)               b += s->feat_s0->nbytes;
	if (s->feat_s1)               b += s->feat_s1->nbytes;
	if (s->feat_4x)               b += s->feat_4x->nbytes;
	if (s->interactive_feat_s0)   b += s->interactive_feat_s0->nbytes;
	if (s->interactive_feat_s1)   b += s->interactive_feat_s1->nbytes;
	if (s->interactive_feat_4x)   b += s->interactive_feat_4x->nbytes;
	return b;
}

static void slot_free_spill(struct sam3_frame_cache_slot *s,
			    struct sam3_frame_cache *cache)
{
	free(s->spill_image_features);       s->spill_image_features       = NULL;
	free(s->spill_feat_s0);              s->spill_feat_s0              = NULL;
	free(s->spill_feat_s1);              s->spill_feat_s1              = NULL;
	free(s->spill_feat_4x);              s->spill_feat_4x              = NULL;
	free(s->spill_interactive_feat_s0);  s->spill_interactive_feat_s0  = NULL;
	free(s->spill_interactive_feat_s1);  s->spill_interactive_feat_s1  = NULL;
	free(s->spill_interactive_feat_4x);  s->spill_interactive_feat_4x  = NULL;
	if (s->spill_bytes <= cache->spill_used)
		cache->spill_used -= s->spill_bytes;
	else
		cache->spill_used = 0;
	s->spill_bytes = 0;
}

/* --- sam3_frame_cache_invalidate --- */

void sam3_frame_cache_invalidate(struct sam3_frame_cache *cache)
{
	if (!cache || !cache->slots)
		return;
	for (int i = 0; i < cache->n_frames; i++) {
		struct sam3_frame_cache_slot *s = &cache->slots[i];
		if (s->tier == SAM3_FRAME_TIER_CPU_SPILL)
			slot_free_spill(s, cache);
		s->image_features       = NULL;
		s->feat_s0              = NULL;
		s->feat_s1              = NULL;
		s->feat_4x              = NULL;
		s->interactive_feat_s0  = NULL;
		s->interactive_feat_s1  = NULL;
		s->interactive_feat_4x  = NULL;
		s->tier                 = SAM3_FRAME_TIER_NONE;
		s->last_access_seq      = 0;
	}
	cache->backend_used   = 0;
	cache->spill_used     = 0;
	cache->access_counter = 0;
	sam3_arena_reset(&cache->backend_arena);
}

/* sam3_frame_cache_release --- */

void sam3_frame_cache_release(struct sam3_frame_cache *cache)
{
	if (!cache)
		return;
	if (cache->slots) {
		sam3_frame_cache_invalidate(cache);
		free(cache->slots);
		cache->slots = NULL;
	}
	sam3_arena_free(&cache->backend_arena);
	memset(cache, 0, sizeof(*cache));
}

/* --- Eviction helpers --- */

/*
 * spill_slot_to_cpu - Copy a backend-tier slot's tensor data to host memory.
 *
 * If spill is disabled (spill_budget == SIZE_MAX) or the budget is
 * exhausted, or malloc fails, the slot is dropped to TIER_NONE instead.
 * Either way the slot's backend tensor pointers are NULLed and the tier
 * is updated — the caller must account for the freed bytes separately.
 */
static enum sam3_error
spill_slot_to_cpu(struct sam3_frame_cache_slot *s,
		  struct sam3_frame_cache *cache)
{
	if (s->tier != SAM3_FRAME_TIER_BACKEND)
		return SAM3_EINVAL;

	size_t total = slot_backend_bytes(s);

	/* Spill disabled or insufficient room: drop to NONE. */
	if (cache->spill_budget == SIZE_MAX ||
	    cache->spill_used + total > cache->spill_budget) {
		s->image_features      = NULL;
		s->feat_s0             = NULL;
		s->feat_s1             = NULL;
		s->feat_4x             = NULL;
		s->interactive_feat_s0 = NULL;
		s->interactive_feat_s1 = NULL;
		s->interactive_feat_4x = NULL;
		s->tier                = SAM3_FRAME_TIER_NONE;
		return SAM3_OK;
	}

	/*
	 * image_features is NULL for SAM 3.1 tri-neck encodes (no 0.5x
	 * scale in that checkpoint). feat_4x is optional for older
	 * single-scale backbones. interactive_feat_* are NULL on SAM 3
	 * checkpoints. Only feat_s0 / feat_s1 are required.
	 */
	void *im = s->image_features ? malloc(s->image_features->nbytes) : NULL;
	void *s0 = malloc(s->feat_s0->nbytes);
	void *s1 = malloc(s->feat_s1->nbytes);
	void *s4 = s->feat_4x ? malloc(s->feat_4x->nbytes) : NULL;
	void *i0 = s->interactive_feat_s0 ? malloc(s->interactive_feat_s0->nbytes) : NULL;
	void *i1 = s->interactive_feat_s1 ? malloc(s->interactive_feat_s1->nbytes) : NULL;
	void *i4 = s->interactive_feat_4x ? malloc(s->interactive_feat_4x->nbytes) : NULL;
	int im_ok = (!s->image_features) || (im != NULL);
	int s4_ok = (!s->feat_4x)        || (s4 != NULL);
	int i0_ok = (!s->interactive_feat_s0) || (i0 != NULL);
	int i1_ok = (!s->interactive_feat_s1) || (i1 != NULL);
	int i4_ok = (!s->interactive_feat_4x) || (i4 != NULL);
	if (!im_ok || !s0 || !s1 || !s4_ok || !i0_ok || !i1_ok || !i4_ok) {
		free(im); free(s0); free(s1); free(s4);
		free(i0); free(i1); free(i4);
		/* Spill malloc failed; drop the slot instead. */
		s->image_features      = NULL;
		s->feat_s0             = NULL;
		s->feat_s1             = NULL;
		s->feat_4x             = NULL;
		s->interactive_feat_s0 = NULL;
		s->interactive_feat_s1 = NULL;
		s->interactive_feat_4x = NULL;
		s->tier                = SAM3_FRAME_TIER_NONE;
		sam3_log_warn("frame_cache: spill malloc failed frame %d",
			      s->frame_idx);
		return SAM3_OK;
	}

	if (s->image_features)
		memcpy(im, s->image_features->data,
		       s->image_features->nbytes);
	memcpy(s0, s->feat_s0->data, s->feat_s0->nbytes);
	memcpy(s1, s->feat_s1->data, s->feat_s1->nbytes);
	if (s->feat_4x)
		memcpy(s4, s->feat_4x->data, s->feat_4x->nbytes);
	if (s->interactive_feat_s0)
		memcpy(i0, s->interactive_feat_s0->data,
		       s->interactive_feat_s0->nbytes);
	if (s->interactive_feat_s1)
		memcpy(i1, s->interactive_feat_s1->data,
		       s->interactive_feat_s1->nbytes);
	if (s->interactive_feat_4x)
		memcpy(i4, s->interactive_feat_4x->data,
		       s->interactive_feat_4x->nbytes);

	/* Snapshot tensor headers so a later promote-via-memcpy can
	 * rebuild the backend tensors without the encoder. Clearing the
	 * data pointer keeps these copies from being mistaken for live
	 * arena-backed tensors. */
	if (s->image_features) {
		s->spill_hdr_image_features      = *s->image_features;
		s->spill_hdr_image_features.data = NULL;
	}
	s->spill_hdr_feat_s0             = *s->feat_s0;
	s->spill_hdr_feat_s0.data        = NULL;
	s->spill_hdr_feat_s1             = *s->feat_s1;
	s->spill_hdr_feat_s1.data        = NULL;
	if (s->feat_4x) {
		s->spill_hdr_feat_4x      = *s->feat_4x;
		s->spill_hdr_feat_4x.data = NULL;
	}
	if (s->interactive_feat_s0) {
		s->spill_hdr_interactive_feat_s0      = *s->interactive_feat_s0;
		s->spill_hdr_interactive_feat_s0.data = NULL;
	}
	if (s->interactive_feat_s1) {
		s->spill_hdr_interactive_feat_s1      = *s->interactive_feat_s1;
		s->spill_hdr_interactive_feat_s1.data = NULL;
	}
	if (s->interactive_feat_4x) {
		s->spill_hdr_interactive_feat_4x      = *s->interactive_feat_4x;
		s->spill_hdr_interactive_feat_4x.data = NULL;
	}

	s->spill_image_features        = im;
	s->spill_feat_s0               = s0;
	s->spill_feat_s1               = s1;
	s->spill_feat_4x               = s4;
	s->spill_interactive_feat_s0   = i0;
	s->spill_interactive_feat_s1   = i1;
	s->spill_interactive_feat_4x   = i4;
	s->spill_bytes                 = total;
	cache->spill_used             += total;

	s->image_features      = NULL;
	s->feat_s0             = NULL;
	s->feat_s1             = NULL;
	s->feat_4x             = NULL;
	s->interactive_feat_s0 = NULL;
	s->interactive_feat_s1 = NULL;
	s->interactive_feat_4x = NULL;
	s->tier                = SAM3_FRAME_TIER_CPU_SPILL;
	return SAM3_OK;
}

/*
 * make_backend_room - Evict LRU backend slots until @need bytes are available.
 *
 * Uses the bump arena's all-or-nothing model: we cannot free individual
 * tensors. We track backend_used to know how much is logically in use, and
 * reset the arena only when all backend slots have been evicted (backend_used
 * reaches 0). This is the cost of bump allocation — deterministic and cheap
 * per alloc, but no per-slot reclaim.
 */
static enum sam3_error
make_backend_room(struct sam3_frame_cache *cache, size_t need)
{
	while (cache->backend_used + need > cache->backend_budget) {
		struct sam3_frame_cache_slot *victim = NULL;
		uint64_t oldest = UINT64_MAX;

		for (int i = 0; i < cache->n_frames; i++) {
			struct sam3_frame_cache_slot *s = &cache->slots[i];
			if (s->tier != SAM3_FRAME_TIER_BACKEND)
				continue;
			if (s->last_access_seq < oldest) {
				oldest = s->last_access_seq;
				victim = s;
			}
		}
		if (!victim) {
			sam3_log_error("frame_cache: no backend victims "
				       "(need %zu B, used %zu/%zu)",
				       need, cache->backend_used,
				       cache->backend_budget);
			return SAM3_ENOMEM;
		}

		size_t freed = slot_backend_bytes(victim);
		enum sam3_error err = spill_slot_to_cpu(victim, cache);
		if (err != SAM3_OK)
			return err;

		if (freed <= cache->backend_used)
			cache->backend_used -= freed;
		else
			cache->backend_used = 0;

		/* If backend tier is now empty, reset the arena to clear
		 * fragmentation and allow the full budget to be reused. */
		if (cache->backend_used == 0)
			sam3_arena_reset(&cache->backend_arena);
	}
	return SAM3_OK;
}

/*
 * drain_backend_tier - Evict every backend slot and reset the arena.
 *
 * The bump arena cannot reclaim individual allocations, so when a
 * budget-scoped make_backend_room() leaves the arena offset too close
 * to its capacity for the next encode/promote to fit, the only path
 * forward is to evict everything in the backend tier. Each evicted
 * slot moves to spill when budget allows, or to NONE otherwise.
 * Called from sam3_frame_cache_get as a last-resort retry step.
 */
static void drain_backend_tier(struct sam3_frame_cache *cache)
{
	for (int i = 0; i < cache->n_frames; i++) {
		struct sam3_frame_cache_slot *s = &cache->slots[i];
		if (s->tier != SAM3_FRAME_TIER_BACKEND)
			continue;
		size_t freed = slot_backend_bytes(s);
		(void)spill_slot_to_cpu(s, cache);
		if (freed <= cache->backend_used)
			cache->backend_used -= freed;
		else
			cache->backend_used = 0;
	}
	if (cache->backend_used == 0)
		sam3_arena_reset(&cache->backend_arena);
}

/*
 * spill_promote - Promote a CPU-spill slot back to the backend tier
 *                 by memcpy'ing bytes into fresh arena allocations,
 *                 bypassing the image encoder entirely.
 *
 * Precondition: s->tier == SAM3_FRAME_TIER_CPU_SPILL with valid
 * spill_image_features / spill_feat_s0 / spill_feat_s1 pointers
 * and matching spill_hdr_* metadata captured by spill_slot_to_cpu.
 *
 * On success, @out points at tensors allocated from cache->backend_arena
 * whose contents are byte-identical to the pre-eviction state. Returns
 * SAM3_ENOMEM if any arena allocation fails (the partial allocations
 * are left in the arena; the bump offset is not rewound).
 */
static enum sam3_error
spill_promote(struct sam3_frame_cache *cache,
	      struct sam3_frame_cache_slot *s,
	      struct sam3_frame_features *out)
{
	struct sam3_arena *arena = &cache->backend_arena;

	struct sam3_tensor tmp;

	/* image_features is optional (NULL on SAM 3.1 tri-neck). */
	if (s->spill_image_features) {
		tmp      = s->spill_hdr_image_features;
		tmp.data = s->spill_image_features;
		out->image_features = sam3_tensor_clone_persist(arena, &tmp);
		if (!out->image_features)
			return SAM3_ENOMEM;
	} else {
		out->image_features = NULL;
	}

	tmp      = s->spill_hdr_feat_s0;
	tmp.data = s->spill_feat_s0;
	out->feat_s0 = sam3_tensor_clone_persist(arena, &tmp);
	if (!out->feat_s0)
		return SAM3_ENOMEM;

	tmp      = s->spill_hdr_feat_s1;
	tmp.data = s->spill_feat_s1;
	out->feat_s1 = sam3_tensor_clone_persist(arena, &tmp);
	if (!out->feat_s1)
		return SAM3_ENOMEM;

	if (s->spill_feat_4x) {
		tmp      = s->spill_hdr_feat_4x;
		tmp.data = s->spill_feat_4x;
		out->feat_4x = sam3_tensor_clone_persist(arena, &tmp);
		if (!out->feat_4x)
			return SAM3_ENOMEM;
	} else {
		out->feat_4x = NULL;
	}

	if (s->spill_interactive_feat_s0) {
		tmp      = s->spill_hdr_interactive_feat_s0;
		tmp.data = s->spill_interactive_feat_s0;
		out->interactive_feat_s0 = sam3_tensor_clone_persist(arena, &tmp);
		if (!out->interactive_feat_s0)
			return SAM3_ENOMEM;
	} else {
		out->interactive_feat_s0 = NULL;
	}
	if (s->spill_interactive_feat_s1) {
		tmp      = s->spill_hdr_interactive_feat_s1;
		tmp.data = s->spill_interactive_feat_s1;
		out->interactive_feat_s1 = sam3_tensor_clone_persist(arena, &tmp);
		if (!out->interactive_feat_s1)
			return SAM3_ENOMEM;
	} else {
		out->interactive_feat_s1 = NULL;
	}
	if (s->spill_interactive_feat_4x) {
		tmp      = s->spill_hdr_interactive_feat_4x;
		tmp.data = s->spill_interactive_feat_4x;
		out->interactive_feat_4x = sam3_tensor_clone_persist(arena, &tmp);
		if (!out->interactive_feat_4x)
			return SAM3_ENOMEM;
	} else {
		out->interactive_feat_4x = NULL;
	}
	return SAM3_OK;
}

/* --- sam3_frame_cache_get --- */

enum sam3_error sam3_frame_cache_get(struct sam3_frame_cache *cache,
				     int frame,
				     struct sam3_frame_features *out)
{
	if (!cache || !out || frame < 0 || frame >= cache->n_frames)
		return SAM3_EINVAL;

	struct sam3_frame_cache_slot *s = &cache->slots[frame];
	cache->access_counter++;
	s->last_access_seq = cache->access_counter;

	/* Fast path: already in backend tier. */
	if (s->tier == SAM3_FRAME_TIER_BACKEND) {
		cache->n_hits++;
		out->image_features      = s->image_features;
		out->feat_s0             = s->feat_s0;
		out->feat_s1             = s->feat_s1;
		out->feat_4x             = s->feat_4x;
		out->interactive_feat_s0 = s->interactive_feat_s0;
		out->interactive_feat_s1 = s->interactive_feat_s1;
		out->interactive_feat_4x = s->interactive_feat_4x;
		return SAM3_OK;
	}

	/*
	 * spill_image_features is NULL for SAM 3.1 tri-neck encodes; treat
	 * it as optional and only require feat_s0 + feat_s1 for the
	 * promote fast path.
	 */
	int use_promote = (s->tier == SAM3_FRAME_TIER_CPU_SPILL &&
			   s->spill_feat_s0 != NULL &&
			   s->spill_feat_s1 != NULL);
	if (use_promote)
		cache->n_spill_promotes++;
	else if (s->tier == SAM3_FRAME_TIER_CPU_SPILL)
		cache->n_misses++;  /* spill buffers dropped — must re-encode */
	else
		cache->n_misses++;

	/*
	 * Miss, or spill-hit promote path. Both allocate into the backend
	 * arena and share the same SAM3_ENOMEM recovery: evict LRU slots
	 * via make_backend_room, and if the bump arena is still stuck full
	 * (partial eviction leaves the offset high since the allocator
	 * cannot reclaim individual slots), drain the backend tier to
	 * force a reset before the final retry.
	 */
	struct sam3_frame_features fresh = {0};
	enum sam3_error err;

	if (use_promote)
		err = spill_promote(cache, s, &fresh);
	else
		err = cache->encode(cache->owner, frame,
				    &cache->backend_arena, &fresh);

	if (err == SAM3_ENOMEM) {
		enum sam3_error room_err =
			make_backend_room(cache, cache->backend_budget / 2);
		if (room_err != SAM3_OK)
			return room_err;
		memset(&fresh, 0, sizeof(fresh));
		if (use_promote)
			err = spill_promote(cache, s, &fresh);
		else
			err = cache->encode(cache->owner, frame,
					    &cache->backend_arena, &fresh);
	}
	if (err == SAM3_ENOMEM) {
		/* Budget-based eviction left the bump arena too full for
		 * this encode to fit. Drain every backend slot so the
		 * arena can reset, then try one last time. */
		drain_backend_tier(cache);
		memset(&fresh, 0, sizeof(fresh));
		if (use_promote)
			err = spill_promote(cache, s, &fresh);
		else
			err = cache->encode(cache->owner, frame,
					    &cache->backend_arena, &fresh);
	}
	if (err != SAM3_OK)
		return err;

	size_t total = 0;
	if (fresh.image_features)       total += fresh.image_features->nbytes;
	if (fresh.feat_s0)              total += fresh.feat_s0->nbytes;
	if (fresh.feat_s1)              total += fresh.feat_s1->nbytes;
	if (fresh.feat_4x)              total += fresh.feat_4x->nbytes;
	if (fresh.interactive_feat_s0)  total += fresh.interactive_feat_s0->nbytes;
	if (fresh.interactive_feat_s1)  total += fresh.interactive_feat_s1->nbytes;
	if (fresh.interactive_feat_4x)  total += fresh.interactive_feat_4x->nbytes;

	/* Free prior spill bytes now that we have a backend copy. */
	if (s->tier == SAM3_FRAME_TIER_CPU_SPILL)
		slot_free_spill(s, cache);

	s->image_features      = fresh.image_features;
	s->feat_s0             = fresh.feat_s0;
	s->feat_s1             = fresh.feat_s1;
	s->feat_4x             = fresh.feat_4x;
	s->interactive_feat_s0 = fresh.interactive_feat_s0;
	s->interactive_feat_s1 = fresh.interactive_feat_s1;
	s->interactive_feat_4x = fresh.interactive_feat_4x;
	s->tier                = SAM3_FRAME_TIER_BACKEND;
	cache->backend_used   += total;

	*out = fresh;
	return SAM3_OK;
}
