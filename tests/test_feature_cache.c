/*
 * tests/test_feature_cache.c - Encoder feature cache unit tests
 *
 * Tests the standalone feature cache module: init/free, slot allocation,
 * LRU eviction, hit/miss accounting, and clear semantics. Uses synthetic
 * tensors so the tests do not require a loaded model.
 *
 * Key types:  sam3_image_feature_cache, sam3_text_feature_cache
 * Depends on: test_helpers.h, model/feature_cache.h, core/alloc.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include <string.h>
#include "model/feature_cache.h"

static void test_image_cache_init_default_slots(void)
{
	struct sam3_image_feature_cache *c = sam3_image_cache_create(0, 1024);
	ASSERT_NOT_NULL(c);
	ASSERT_EQ(sam3_image_cache_n_slots(c), 3);
	struct sam3_cache_stats s = {0};
	sam3_image_cache_stats(c, &s);
	ASSERT_EQ(s.image_hits, 0u);
	ASSERT_EQ(s.image_misses, 0u);
	sam3_image_cache_destroy(c);
}

static void test_image_cache_clamps_slot_count(void)
{
	struct sam3_image_feature_cache *c = sam3_image_cache_create(99, 1024);
	ASSERT_NOT_NULL(c);
	ASSERT_EQ(sam3_image_cache_n_slots(c), 16);
	sam3_image_cache_destroy(c);
}

static void test_text_cache_init_default_slots(void)
{
	struct sam3_text_feature_cache *c = sam3_text_cache_create(0, 4096);
	ASSERT_NOT_NULL(c);
	ASSERT_EQ(sam3_text_cache_n_slots(c), 16);
	sam3_text_cache_destroy(c);
}

static struct sam3_tensor *make_dummy_tensor(struct sam3_arena *a, int v)
{
	struct sam3_tensor *t = sam3_arena_alloc(a, sizeof(*t));
	if (!t) return NULL;
	t->dtype = SAM3_DTYPE_F32;
	t->n_dims = 1;
	t->dims[0] = 1;
	t->data = sam3_arena_alloc(a, sizeof(float));
	if (!t->data) return NULL;
	((float *)t->data)[0] = (float)v;
	return t;
}

static void test_image_cache_miss_then_hit(void)
{
	struct sam3_image_feature_cache *c = sam3_image_cache_create(2,
								     4 * 1024);
	int idx = sam3_image_cache_lookup(c, 0xdeadbeefULL, NULL, 0);
	ASSERT_EQ(idx, -1);

	int slot = sam3_image_cache_claim_slot(c);
	ASSERT(slot >= 0);
	struct sam3_image_bundle b = {0};
	b.image_features = make_dummy_tensor(&c->slots[slot].arena, 42);
	sam3_image_cache_register(c, slot, 0xdeadbeefULL, NULL, 0, &b);

	int hit = sam3_image_cache_lookup(c, 0xdeadbeefULL, NULL, 0);
	ASSERT_EQ(hit, slot);
	ASSERT_EQ(((float *)c->slots[hit].bundle.image_features->data)[0],
		  42.0f);

	struct sam3_cache_stats s = {0};
	sam3_image_cache_stats(c, &s);
	ASSERT_EQ(s.image_hits, 1u);
	ASSERT_EQ(s.image_misses, 1u);

	sam3_image_cache_destroy(c);
}

static void test_image_cache_lru_eviction(void)
{
	struct sam3_image_feature_cache *c = sam3_image_cache_create(2,
								     4 * 1024);

	/* Insert A and B. */
	int sa = sam3_image_cache_claim_slot(c);
	struct sam3_image_bundle ba = {0};
	ba.image_features = make_dummy_tensor(&c->slots[sa].arena, 1);
	sam3_image_cache_register(c, sa, 0xAAULL, NULL, 0, &ba);

	int sb = sam3_image_cache_claim_slot(c);
	struct sam3_image_bundle bb = {0};
	bb.image_features = make_dummy_tensor(&c->slots[sb].arena, 2);
	sam3_image_cache_register(c, sb, 0xBBULL, NULL, 0, &bb);

	/* Touch A so B becomes the LRU victim. */
	int touch = sam3_image_cache_lookup(c, 0xAAULL, NULL, 0);
	ASSERT_EQ(touch, sa);

	/* Insert C. It should evict B, not A. */
	int sc = sam3_image_cache_claim_slot(c);
	ASSERT_EQ(sc, sb);
	struct sam3_image_bundle bc = {0};
	bc.image_features = make_dummy_tensor(&c->slots[sc].arena, 3);
	sam3_image_cache_register(c, sc, 0xCCULL, NULL, 0, &bc);

	ASSERT_EQ(sam3_image_cache_lookup(c, 0xAAULL, NULL, 0), sa);
	ASSERT_EQ(sam3_image_cache_lookup(c, 0xBBULL, NULL, 0), -1);
	ASSERT_EQ(sam3_image_cache_lookup(c, 0xCCULL, NULL, 0), sc);

	struct sam3_cache_stats s = {0};
	sam3_image_cache_stats(c, &s);
	ASSERT_EQ(s.image_evictions, 1u);

	sam3_image_cache_destroy(c);
}

static void test_image_cache_collision_verifies_prefix(void)
{
	struct sam3_image_feature_cache *c = sam3_image_cache_create(1,
								     4 * 1024);
	int slot = sam3_image_cache_claim_slot(c);
	struct sam3_image_bundle b = {0};
	b.image_features = make_dummy_tensor(&c->slots[slot].arena, 7);
	uint8_t pref_a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
	sam3_image_cache_register(c, slot, 0x1234ULL, pref_a, 8, &b);

	/* Same hash, different prefix: must NOT hit. */
	uint8_t pref_b[8] = {9, 9, 9, 9, 9, 9, 9, 9};
	int collide = sam3_image_cache_lookup(c, 0x1234ULL, pref_b, 8);
	ASSERT_EQ(collide, -1);

	sam3_image_cache_destroy(c);
}

static void test_image_cache_clear(void)
{
	struct sam3_image_feature_cache *c = sam3_image_cache_create(2,
								     4 * 1024);
	int slot = sam3_image_cache_claim_slot(c);
	struct sam3_image_bundle b = {0};
	b.image_features = make_dummy_tensor(&c->slots[slot].arena, 1);
	sam3_image_cache_register(c, slot, 0xFEULL, NULL, 0, &b);

	sam3_image_cache_clear(c);
	ASSERT_EQ(sam3_image_cache_lookup(c, 0xFEULL, NULL, 0), -1);

	sam3_image_cache_destroy(c);
}

int main(void)
{
	test_image_cache_init_default_slots();
	test_image_cache_clamps_slot_count();
	test_text_cache_init_default_slots();
	test_image_cache_miss_then_hit();
	test_image_cache_lru_eviction();
	test_image_cache_collision_verifies_prefix();
	test_image_cache_clear();
	TEST_REPORT();
}
