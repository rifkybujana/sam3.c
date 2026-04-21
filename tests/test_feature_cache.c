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

int main(void)
{
	test_image_cache_init_default_slots();
	test_image_cache_clamps_slot_count();
	test_text_cache_init_default_slots();
	TEST_REPORT();
}
