/*
 * tests/test_video_multi_object.c - Per-object propagation correctness
 *
 * Unit tests for per-object memory bank independence, object compaction
 * on remove, and the SAM3_MAX_OBJECTS=16 cap. End-to-end multi-object
 * tracking IoU vs Python lives in the Phase 5 parity test.
 *
 * Key types:  sam3_video_session, sam3_video_frame_result
 * Depends on: sam3/sam3.h, model/video_session.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include <string.h>
#include "test_helpers.h"
#include "sam3/sam3.h"
#include "model/video_session.h"

/*
 * Two objects added to a session each get their own memory bank.
 * Adding a conditioning entry to obj A's bank must not touch obj B's bank.
 */
static void test_two_objects_have_independent_banks(void)
{
	struct sam3_video_session session = {0};
	session.frames.n_frames = 4;

	int idx_a = sam3_session_get_or_add_obj(&session, /*obj_id=*/100);
	int idx_b = sam3_session_get_or_add_obj(&session, /*obj_id=*/101);
	ASSERT_EQ(idx_a, 0);
	ASSERT_EQ(idx_b, 1);
	ASSERT_EQ(session.n_objects, 2);

	struct sam3_memory_entry e = {
		.frame_idx = 0,
		.is_conditioning = 1,
		.obj_score = 1.0f,
	};
	sam3_memory_bank_add(&session.objects[idx_a].bank, &e);

	ASSERT_EQ(session.objects[idx_a].bank.n_cond, 1);
	ASSERT_EQ(session.objects[idx_b].bank.n_cond, 0);

	sam3_session_remove_obj(&session, 100);
	sam3_session_remove_obj(&session, 101);
}

/*
 * Removing an object compacts objects[] down without wiping
 * sibling state. The obj that shifted into the removed slot must
 * keep its bank contents intact.
 */
static void test_remove_compacts_without_pollution(void)
{
	struct sam3_video_session session = {0};
	session.frames.n_frames = 4;

	sam3_session_get_or_add_obj(&session, 10);
	sam3_session_get_or_add_obj(&session, 20);
	sam3_session_get_or_add_obj(&session, 30);
	ASSERT_EQ(session.n_objects, 3);

	struct sam3_memory_entry e = {
		.frame_idx = 1,
		.is_conditioning = 1,
		.obj_score = 1.0f,
	};
	sam3_memory_bank_add(&session.objects[1].bank, &e);
	ASSERT_EQ(session.objects[1].bank.n_cond, 1);

	sam3_session_remove_obj(&session, /*obj_id=*/10);
	ASSERT_EQ(session.n_objects, 2);

	/* obj 20 shifts into slot 0; its bank must still carry the entry. */
	ASSERT_EQ(session.objects[0].obj_id, 20);
	ASSERT_EQ(session.objects[0].bank.n_cond, 1);
	ASSERT_EQ(session.objects[0].bank.cond[0].frame_idx, 1);

	ASSERT_EQ(session.objects[1].obj_id, 30);

	sam3_session_remove_obj(&session, 20);
	sam3_session_remove_obj(&session, 30);
}

/*
 * 17th object add must be rejected now that SAM3_MAX_OBJECTS = 16.
 * Expected return is a negative index (sam3_session_get_or_add_obj
 * returns -1 when the table is full).
 */
static void test_full_object_cap_returns_error(void)
{
	struct sam3_video_session session = {0};
	session.frames.n_frames = 4;

	for (int i = 0; i < SAM3_MAX_OBJECTS; i++) {
		int idx = sam3_session_get_or_add_obj(&session, 100 + i);
		ASSERT(idx >= 0);
	}
	ASSERT_EQ(session.n_objects, SAM3_MAX_OBJECTS);

	int idx = sam3_session_get_or_add_obj(&session, /*obj_id=*/9999);
	ASSERT(idx < 0);

	for (int i = 0; i < SAM3_MAX_OBJECTS; i++)
		sam3_session_remove_obj(&session, 100 + i);
}

/*
 * Per-object prompted-frame bitmap is lazy: NULL before any mark,
 * allocated on first mark, unaffected by sibling objects.
 */
static void test_per_object_prompted_bitmap(void)
{
	struct sam3_video_session session = {0};
	session.frames.n_frames = 16;

	sam3_session_get_or_add_obj(&session, 1);
	sam3_session_get_or_add_obj(&session, 2);

	ASSERT(session.objects[0].prompted_frames == NULL);
	ASSERT(session.objects[1].prompted_frames == NULL);

	ASSERT_EQ(sam3_session_obj_is_prompted(&session, 0, 3), 0);
	ASSERT_EQ(sam3_session_obj_mark_prompted(&session, 0, 3), SAM3_OK);
	ASSERT_EQ(sam3_session_obj_is_prompted(&session, 0, 3), 1);
	/* Sibling bitmap untouched. */
	ASSERT_EQ(sam3_session_obj_is_prompted(&session, 1, 3), 0);

	sam3_session_remove_obj(&session, 1);
	sam3_session_remove_obj(&session, 2);
}

int main(void)
{
	test_two_objects_have_independent_banks();
	test_remove_compacts_without_pollution();
	test_full_object_cap_returns_error();
	test_per_object_prompted_bitmap();
	TEST_REPORT();
}
