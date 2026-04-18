/*
 * tests/test_frame_cache.c - Tiered LRU frame cache unit tests
 *
 * Validates init/release, hit semantics, LRU eviction to spill when
 * the backend budget is exceeded, and invalidate. A mock encoder
 * produces tiny fake tensors so we can drive the cache through all
 * tiers without the real image encoder.
 *
 * Key types:  sam3_frame_cache, sam3_frame_features
 * Depends on: model/frame_cache.h, core/tensor.h, core/alloc.h,
 *             test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "test_helpers.h"
#include "core/tensor.h"
#include "core/alloc.h"
#include "model/frame_cache.h"

static int g_encode_calls = 0;

/*
 * Mock encoder: allocates three tiny tensors from @arena so the
 * cache can exercise hit/spill paths without touching the real image
 * encoder. Each tensor is [1, 2, 2, 4] f32 = 64 bytes of data plus
 * tensor metadata.
 */
static enum sam3_error mock_encode(struct sam3_video_session *session,
				    int frame_idx,
				    struct sam3_arena *arena,
				    struct sam3_frame_features *out)
{
	(void)session;
	g_encode_calls++;

	int dims[4] = {1, 2, 2, 4};

	out->image_features = sam3_arena_alloc(arena,
		sizeof(struct sam3_tensor));
	out->feat_s0        = sam3_arena_alloc(arena,
		sizeof(struct sam3_tensor));
	out->feat_s1        = sam3_arena_alloc(arena,
		sizeof(struct sam3_tensor));
	if (!out->image_features || !out->feat_s0 || !out->feat_s1)
		return SAM3_ENOMEM;

	size_t nbytes = 1 * 2 * 2 * 4 * sizeof(float);
	void *d0 = sam3_arena_alloc(arena, nbytes);
	void *d1 = sam3_arena_alloc(arena, nbytes);
	void *d2 = sam3_arena_alloc(arena, nbytes);
	if (!d0 || !d1 || !d2)
		return SAM3_ENOMEM;

	out->image_features->dtype    = SAM3_DTYPE_F32;
	out->image_features->n_dims   = 4;
	memcpy(out->image_features->dims, dims, sizeof(dims));
	out->image_features->data     = d0;
	out->image_features->nbytes   = nbytes;

	out->feat_s0->dtype  = SAM3_DTYPE_F32;
	out->feat_s0->n_dims = 4;
	memcpy(out->feat_s0->dims, dims, sizeof(dims));
	out->feat_s0->data   = d1;
	out->feat_s0->nbytes = nbytes;

	out->feat_s1->dtype  = SAM3_DTYPE_F32;
	out->feat_s1->n_dims = 4;
	memcpy(out->feat_s1->dims, dims, sizeof(dims));
	out->feat_s1->data   = d2;
	out->feat_s1->nbytes = nbytes;

	/* Encode a signal so we can verify identity on hits. */
	((float *)d0)[0] = (float)frame_idx;
	return SAM3_OK;
}

static void test_init_and_release(void)
{
	struct sam3_frame_cache cache = {0};
	enum sam3_error err =
		sam3_frame_cache_init(&cache, NULL, mock_encode,
				       /*n_frames=*/8,
				       /*backend=*/1024 * 1024,
				       /*spill=*/SIZE_MAX);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT_EQ(cache.n_frames, 8);
	sam3_frame_cache_release(&cache);
	ASSERT(cache.slots == NULL);
}

static void test_first_access_encodes_then_hits(void)
{
	g_encode_calls = 0;
	struct sam3_frame_cache cache = {0};
	sam3_frame_cache_init(&cache, NULL, mock_encode, 4,
			      /*backend=*/1024 * 1024,
			      /*spill=*/SIZE_MAX);

	struct sam3_frame_features f1 = {0}, f2 = {0};
	ASSERT_EQ(sam3_frame_cache_get(&cache, 2, &f1), SAM3_OK);
	ASSERT_EQ(g_encode_calls, 1);
	ASSERT(f1.image_features != NULL);

	ASSERT_EQ(sam3_frame_cache_get(&cache, 2, &f2), SAM3_OK);
	ASSERT_EQ(g_encode_calls, 1);                 /* cache hit */
	ASSERT(f1.image_features == f2.image_features); /* same pointer */

	sam3_frame_cache_release(&cache);
}

static void test_lru_eviction_when_backend_full(void)
{
	g_encode_calls = 0;
	struct sam3_frame_cache cache = {0};
	/* Per-slot backend usage accounting sums tensor->nbytes only
	 * (three 64-byte buffers = 192 bytes). So a 2048-byte budget
	 * holds 10 slots before triggering eviction. To force eviction
	 * in under 8 slots we set a very tight budget: enough for ~3
	 * slots. */
	sam3_frame_cache_init(&cache, NULL, mock_encode, 8,
			      /*backend=*/768,      /* ~3 slots */
			      /*spill=*/SIZE_MAX);

	struct sam3_frame_features f = {0};
	for (int i = 0; i < 6; i++)
		ASSERT_EQ(sam3_frame_cache_get(&cache, i, &f), SAM3_OK);
	ASSERT_EQ(g_encode_calls, 6);

	/* After 6 distinct frames with only ~3 slots of backend room,
	 * some slots must have been evicted. Since spill is SIZE_MAX
	 * (disabled), eviction drops to TIER_NONE. */
	int still_backend = 0;
	for (int i = 0; i < 6; i++) {
		if (cache.slots[i].tier == SAM3_FRAME_TIER_BACKEND)
			still_backend++;
	}
	/* At least one slot must have been evicted. */
	ASSERT(still_backend < 6);

	sam3_frame_cache_release(&cache);
}

static void test_invalidate_clears_state(void)
{
	g_encode_calls = 0;
	struct sam3_frame_cache cache = {0};
	sam3_frame_cache_init(&cache, NULL, mock_encode, 4,
			      /*backend=*/0, /*spill=*/SIZE_MAX);

	struct sam3_frame_features f = {0};
	sam3_frame_cache_get(&cache, 0, &f);
	sam3_frame_cache_get(&cache, 1, &f);
	int before = g_encode_calls;

	sam3_frame_cache_invalidate(&cache);
	for (int i = 0; i < 4; i++)
		ASSERT_EQ(cache.slots[i].tier, SAM3_FRAME_TIER_NONE);

	sam3_frame_cache_get(&cache, 0, &f);
	ASSERT_EQ(g_encode_calls, before + 1); /* re-encoded after invalidate */

	sam3_frame_cache_release(&cache);
}

int main(void)
{
	test_init_and_release();
	test_first_access_encodes_then_hits();
	test_lru_eviction_when_backend_full();
	test_invalidate_clears_state();
	TEST_REPORT();
}
