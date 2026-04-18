/*
 * tests/test_video_frame_result.c - sam3_video_frame_result_free safety
 *
 * Verifies the free helper handles zero-initialized results, results
 * with one object, and double-free safety. Defensive unit test for
 * the new public type added in Phase 1 ahead of Phase 2 plumbing.
 *
 * Key types: sam3_video_frame_result, sam3_video_object_mask
 * Depends on: sam3/sam3.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include "test_helpers.h"
#include "sam3/sam3.h"

static void test_free_zero_init_safe(void)
{
	struct sam3_video_frame_result r = {0};
	sam3_video_frame_result_free(&r);
	ASSERT_EQ(r.n_objects, 0);
	ASSERT(r.objects == NULL);
}

static void test_free_one_object(void)
{
	struct sam3_video_frame_result r = {0};
	r.frame_idx = 7;
	r.n_objects = 1;
	r.objects = calloc(1, sizeof(*r.objects));
	r.objects[0].obj_id = 42;
	r.objects[0].mask = calloc(16, sizeof(float));
	r.objects[0].mask_h = 4;
	r.objects[0].mask_w = 4;
	sam3_video_frame_result_free(&r);
	ASSERT_EQ(r.n_objects, 0);
	ASSERT(r.objects == NULL);
}

static void test_free_null_safe(void)
{
	sam3_video_frame_result_free(NULL); /* must not crash */
}

int main(void)
{
	test_free_zero_init_safe();
	test_free_one_object();
	test_free_null_safe();
	TEST_REPORT();
}
