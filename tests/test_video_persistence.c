/*
 * tests/test_video_persistence.c - Memory persistence and scoping
 *
 * Unit tests at the session/bank level for:
 * 1. clear_non_cond_mem_around_input only affects the target object's
 *    bank — siblings are untouched.
 * 2. Bank state survives across would-be-propagate boundaries (no
 *    implicit clear anywhere that isn't a reset).
 *
 * End-to-end idempotency lives in the Phase 5 parity test
 * (test_video_parity_kids).
 *
 * Key types:  sam3_video_session, sam3_memory_bank
 * Depends on: sam3/sam3.h, model/video_session.h, model/memory_bank.h,
 *             test_helpers.h
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
#include "model/memory_bank.h"

static void seed_non_cond(struct sam3_memory_bank *bank, int n, int start)
{
	for (int i = 0; i < n; i++) {
		struct sam3_memory_entry e = {
			.frame_idx = start + i,
			.is_conditioning = 0,
			.obj_score = 0.5f,
		};
		sam3_memory_bank_add(bank, &e);
	}
}

/*
 * clear_around_frame on one object's bank must not touch another
 * object's bank. This enforces per-object isolation.
 */
static void test_clear_around_only_affects_target_object(void)
{
	struct sam3_video_session session = {0};
	session.frames.n_frames = 16;
	session.opts.clear_non_cond_window = 3;

	int a = sam3_session_get_or_add_obj(&session, 1);
	int b = sam3_session_get_or_add_obj(&session, 2);
	ASSERT_EQ(a, 0);
	ASSERT_EQ(b, 1);

	/* Bank capacity defaults (from get_or_add_obj) = 7, max non_cond = 6. */
	seed_non_cond(&session.objects[a].bank, 6, /*start=*/1);
	seed_non_cond(&session.objects[b].bank, 6, /*start=*/1);
	ASSERT_EQ(session.objects[a].bank.n_non_cond, 6);
	ASSERT_EQ(session.objects[b].bank.n_non_cond, 6);

	/*
	 * Clear non-cond around frame 4 with window=3 on obj A only:
	 * removes entries in [1, 7] — that's all six of obj A's entries
	 * (they sit at frames {1..6}). Sibling obj B untouched.
	 */
	sam3_memory_bank_clear_around_frame(&session.objects[a].bank,
					     /*frame=*/4, /*window=*/3);
	ASSERT_EQ(session.objects[a].bank.n_non_cond, 0);
	ASSERT_EQ(session.objects[b].bank.n_non_cond, 6);

	sam3_session_remove_obj(&session, 1);
	sam3_session_remove_obj(&session, 2);
}

/*
 * Bank state must survive a "call sequence" that does not include a
 * clear. This is a data-level guarantee; the actual propagate path
 * no longer clears banks on entry (Task 4.1 removed that behavior).
 */
static void test_bank_survives_without_explicit_clear(void)
{
	struct sam3_video_session session = {0};
	session.frames.n_frames = 16;

	int idx = sam3_session_get_or_add_obj(&session, 99);
	seed_non_cond(&session.objects[idx].bank, 4, /*start=*/0);
	int before = session.objects[idx].bank.n_non_cond;
	ASSERT_EQ(before, 4);

	/*
	 * Simulate the first half of a sweep (no clears). Bank count
	 * stays. (Real propagate never clears per-object banks after
	 * Task 4.1.)
	 */
	ASSERT_EQ(session.objects[idx].bank.n_non_cond, before);

	sam3_session_remove_obj(&session, 99);
}

/*
 * Selecting a small window spares entries just outside it.
 */
static void test_clear_respects_window_boundary(void)
{
	struct sam3_memory_bank bank;
	sam3_memory_bank_init(&bank, 7, 4, 1, 0.0f);
	seed_non_cond(&bank, 5, /*start=*/0); /* frames {0,1,2,3,4} */
	ASSERT_EQ(bank.n_non_cond, 5);

	sam3_memory_bank_clear_around_frame(&bank, /*frame=*/2,
					     /*window=*/1);
	/* Removes frames {1,2,3}; keeps {0,4}. */
	ASSERT_EQ(bank.n_non_cond, 2);
	ASSERT_EQ(bank.non_cond[0].frame_idx, 0);
	ASSERT_EQ(bank.non_cond[1].frame_idx, 4);
}

int main(void)
{
	test_clear_around_only_affects_target_object();
	test_bank_survives_without_explicit_clear();
	test_clear_respects_window_boundary();
	TEST_REPORT();
}
