/*
 * tests/test_memory_bank.c - Memory bank ring buffer tests
 *
 * Tests the memory bank data structure for correct initialization,
 * conditioning/non-conditioning frame insertion, FIFO eviction,
 * SAM3-Long selection filtering, closest-frame queries, and clearing.
 *
 * Key types:  sam3_memory_bank, sam3_memory_entry
 * Depends on: model/memory_bank.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "model/memory_bank.h"

static void test_bank_init(void)
{
	struct sam3_memory_bank bank;
	sam3_memory_bank_init(&bank, 7, 4, 1, 0.01f);
	ASSERT_EQ(bank.n_non_cond, 0);
	ASSERT_EQ(bank.n_cond, 0);
	ASSERT_EQ(bank.capacity, 7);
	ASSERT_EQ(bank.max_cond_frames_in_attn, 4);
}

static void test_bank_add_cond(void)
{
	struct sam3_memory_bank bank;
	sam3_memory_bank_init(&bank, 7, 4, 1, 0.01f);

	struct sam3_memory_entry e = {
		.spatial_features = NULL,
		.obj_pointer = NULL,
		.frame_idx = 0,
		.is_conditioning = 1,
		.obj_score = 1.0f,
	};
	sam3_memory_bank_add(&bank, &e);
	ASSERT_EQ(bank.n_cond, 1);
	ASSERT_EQ(bank.n_non_cond, 0);
	ASSERT_EQ(sam3_memory_bank_total(&bank), 1);
}

static void test_bank_add_non_cond(void)
{
	struct sam3_memory_bank bank;
	sam3_memory_bank_init(&bank, 7, 4, 1, 0.01f);

	struct sam3_memory_entry e = {
		.frame_idx = 5,
		.is_conditioning = 0,
		.obj_score = 0.5f,
	};
	sam3_memory_bank_add(&bank, &e);
	ASSERT_EQ(bank.n_non_cond, 1);
}

static void test_bank_evict_oldest_non_cond(void)
{
	struct sam3_memory_bank bank;
	sam3_memory_bank_init(&bank, 3, 1, 1, 0.01f);

	struct sam3_memory_entry cond = {
		.frame_idx = 0, .is_conditioning = 1, .obj_score = 1.0f
	};
	struct sam3_memory_entry nc1 = {
		.frame_idx = 1, .is_conditioning = 0, .obj_score = 0.5f
	};
	struct sam3_memory_entry nc2 = {
		.frame_idx = 2, .is_conditioning = 0, .obj_score = 0.5f
	};
	struct sam3_memory_entry nc3 = {
		.frame_idx = 3, .is_conditioning = 0, .obj_score = 0.5f
	};

	sam3_memory_bank_add(&bank, &cond);
	sam3_memory_bank_add(&bank, &nc1);
	sam3_memory_bank_add(&bank, &nc2);
	ASSERT_EQ(bank.n_non_cond, 2);

	/* Adding nc3 should evict nc1 (oldest) */
	sam3_memory_bank_add(&bank, &nc3);
	ASSERT_EQ(bank.n_non_cond, 2);
	ASSERT_EQ(bank.non_cond[0].frame_idx, 2);
	ASSERT_EQ(bank.non_cond[1].frame_idx, 3);
}

static void test_bank_sam3long_selection(void)
{
	struct sam3_memory_bank bank;
	sam3_memory_bank_init(&bank, 7, 4, 1, 0.01f);

	/* Frame with score below threshold should be rejected */
	struct sam3_memory_entry low = {
		.frame_idx = 5, .is_conditioning = 0, .obj_score = 0.005f
	};
	sam3_memory_bank_add(&bank, &low);
	ASSERT_EQ(bank.n_non_cond, 0);

	/* Frame with score above threshold should be accepted */
	struct sam3_memory_entry high = {
		.frame_idx = 6, .is_conditioning = 0, .obj_score = 0.5f
	};
	sam3_memory_bank_add(&bank, &high);
	ASSERT_EQ(bank.n_non_cond, 1);
}

static void test_bank_select_closest(void)
{
	struct sam3_memory_bank bank;
	sam3_memory_bank_init(&bank, 7, 4, 1, 0.01f);

	struct sam3_memory_entry e0 = {
		.frame_idx = 0, .is_conditioning = 1, .obj_score = 1.0f
	};
	struct sam3_memory_entry e5 = {
		.frame_idx = 5, .is_conditioning = 1, .obj_score = 1.0f
	};
	struct sam3_memory_entry e10 = {
		.frame_idx = 10, .is_conditioning = 1, .obj_score = 1.0f
	};
	sam3_memory_bank_add(&bank, &e0);
	sam3_memory_bank_add(&bank, &e5);
	sam3_memory_bank_add(&bank, &e10);

	int indices[2];
	int n = sam3_memory_bank_select_closest_cond(&bank, 7, indices, 2);
	ASSERT_EQ(n, 2);
	ASSERT_EQ(bank.cond[indices[0]].frame_idx, 5);
	ASSERT_EQ(bank.cond[indices[1]].frame_idx, 10);
}

/*
 * Cond storage previously capped at max_cond_frames_in_attn; now all
 * cond frames are stored (up to SAM3_MAX_MEMORY_FRAMES) and the cap
 * only applies when building a per-frame view. Matches Python.
 */
static void test_bank_cond_storage_beyond_attn_cap(void)
{
	struct sam3_memory_bank bank;
	sam3_memory_bank_init(&bank, 7, 4, 1, 0.01f);

	for (int i = 0; i < 6; i++) {
		struct sam3_memory_entry e = {
			.frame_idx = i * 10,
			.is_conditioning = 1,
			.obj_score = 1.0f
		};
		sam3_memory_bank_add(&bank, &e);
	}
	ASSERT_EQ(bank.n_cond, 6); /* exceeds max_cond_frames_in_attn=4 */
}

static void test_bank_build_view_selects_closest(void)
{
	struct sam3_memory_bank bank;
	sam3_memory_bank_init(&bank, 7, 4, 1, 0.01f);

	int frames[] = {0, 5, 10, 20, 30, 100};
	for (int i = 0; i < 6; i++) {
		struct sam3_memory_entry e = {
			.frame_idx = frames[i],
			.is_conditioning = 1,
			.obj_score = 1.0f
		};
		sam3_memory_bank_add(&bank, &e);
	}

	struct sam3_memory_bank_view view;
	sam3_memory_bank_build_view(&bank, /*frame_idx=*/12, &view);

	/*
	 * max_cond_frames_in_attn=4; per-frame distances from 12:
	 *   0->12, 5->7, 10->2, 20->8, 30->18, 100->88.
	 * Closest four are {10, 5, 20, 0}; sorted ascending for a
	 * deterministic row order gives {0, 5, 10, 20}.
	 */
	ASSERT_EQ(view.n_cond, 4);
	ASSERT_EQ(bank.cond[view.cond_idx[0]].frame_idx, 0);
	ASSERT_EQ(bank.cond[view.cond_idx[1]].frame_idx, 5);
	ASSERT_EQ(bank.cond[view.cond_idx[2]].frame_idx, 10);
	ASSERT_EQ(bank.cond[view.cond_idx[3]].frame_idx, 20);
}

static void test_bank_build_view_empty(void)
{
	struct sam3_memory_bank bank;
	sam3_memory_bank_init(&bank, 7, 4, 1, 0.01f);

	struct sam3_memory_bank_view view;
	sam3_memory_bank_build_view(&bank, 0, &view);
	ASSERT_EQ(view.n_cond, 0);
	ASSERT(view.bank == &bank);
}

static void test_bank_clear(void)
{
	struct sam3_memory_bank bank;
	sam3_memory_bank_init(&bank, 7, 4, 1, 0.01f);

	struct sam3_memory_entry e = {
		.frame_idx = 0, .is_conditioning = 1, .obj_score = 1.0f
	};
	sam3_memory_bank_add(&bank, &e);
	ASSERT_EQ(bank.n_cond, 1);

	sam3_memory_bank_clear(&bank);
	ASSERT_EQ(bank.n_cond, 0);
	ASSERT_EQ(bank.n_non_cond, 0);
}

/*
 * Clear non-cond entries within a frame window. Cond entries and
 * non-cond entries outside the window must survive.
 */
static void test_bank_clear_around_frame(void)
{
	struct sam3_memory_bank bank;
	sam3_memory_bank_init(&bank, 7, 4, 1, 0.0f);

	struct sam3_memory_entry cond = {
		.frame_idx = 5, .is_conditioning = 1, .obj_score = 1.0f
	};
	sam3_memory_bank_add(&bank, &cond);

	int frames[] = {0, 3, 5, 7, 10};
	for (int i = 0; i < 5; i++) {
		struct sam3_memory_entry e = {
			.frame_idx = frames[i],
			.is_conditioning = 0,
			.obj_score = 0.5f,
		};
		sam3_memory_bank_add(&bank, &e);
	}
	ASSERT_EQ(bank.n_non_cond, 5);
	ASSERT_EQ(bank.n_cond, 1);

	/* Clear non-cond within window=2 around frame 5: removes 3,5,7. */
	sam3_memory_bank_clear_around_frame(&bank, /*frame=*/5, /*window=*/2);
	ASSERT_EQ(bank.n_non_cond, 2);
	ASSERT_EQ(bank.non_cond[0].frame_idx, 0);
	ASSERT_EQ(bank.non_cond[1].frame_idx, 10);
	ASSERT_EQ(bank.n_cond, 1); /* cond untouched */
}

/*
 * Select non-cond entries for the current frame attention pass.
 * With num_maskmem=4 and temporal_stride=2, expect: most recent
 * past frame (target=current-stride*1), then current-stride*2, etc.
 * Capped at num_maskmem-1=3 picks.
 */
static void test_bank_select_non_cond_for_frame(void)
{
	struct sam3_memory_bank bank;
	sam3_memory_bank_init(&bank, /*capacity=*/7,
			       /*max_cond_in_attn=*/4,
			       /*temporal_stride=*/2,
			       /*mf_threshold=*/0.0f);

	int frames[] = {1, 2, 3, 4, 5, 6};
	for (int i = 0; i < 6; i++) {
		struct sam3_memory_entry e = {
			.frame_idx = frames[i],
			.is_conditioning = 0,
			.obj_score = 0.5f,
		};
		sam3_memory_bank_add(&bank, &e);
	}
	/* capacity=7 → max_non_cond = capacity-1 = 6, so all six fit. */

	int selected[8];
	int n = sam3_memory_bank_select_non_cond_for_frame(
		&bank, /*current_frame=*/10,
		/*num_maskmem=*/4, selected, 8);
	/*
	 * num_maskmem-1=3 picks with simple-skip-on-dup:
	 *   t=1: target=10-2*1=8 → closest past entry is frame 6 (dist 2) at
	 *        index 5 → pick 5.
	 *   t=2: target=10-2*2=6 → closest past entry is frame 6 (dist 0) at
	 *        index 5, but already selected → skip (no retry/fallback).
	 *   t=3: target=10-2*3=4 → closest past entry is frame 4 (dist 0) at
	 *        index 3 → pick 3.
	 *
	 * Expected: 2 picks, frames {6, 4} at indices {5, 3}.
	 */
	ASSERT_EQ(n, 2);
	ASSERT_EQ(bank.non_cond[selected[0]].frame_idx, 6);
	ASSERT_EQ(bank.non_cond[selected[1]].frame_idx, 4);
}

/*
 * Exercise the simple-skip-on-dup behavior (no retry-fallback).
 *
 * Bank has only 2 non-cond entries: frames {1, 5}. Current=10,
 * stride=3, num_maskmem=4 → 3 targets {7, 4, 1}.
 *   t=1 target=7: closest past entry is frame 5 (dist 2). Pick.
 *   t=2 target=4: closest past entry is frame 5 (dist 1), but already
 *                 picked → skip. Plan algorithm: writes nothing for t=2.
 *                 (Retry-fallback would have picked frame 1 instead.)
 *   t=3 target=1: closest past entry is frame 1 (dist 0). Pick.
 *
 * Expected: 2 picks, frames {5, 1}.
 * If retry-fallback semantics had been implemented, t=2 would fallback
 * to frame 1, potentially giving 3 picks {5, 1, 1} or similar.
 */
static void test_bank_select_non_cond_simple_skip_on_dup(void)
{
	struct sam3_memory_bank bank;
	sam3_memory_bank_init(&bank, /*capacity=*/7,
			       /*max_cond_in_attn=*/4,
			       /*temporal_stride=*/3,
			       /*mf_threshold=*/0.0f);

	struct sam3_memory_entry e1 = {
		.frame_idx = 1, .is_conditioning = 0, .obj_score = 0.5f
	};
	struct sam3_memory_entry e5 = {
		.frame_idx = 5, .is_conditioning = 0, .obj_score = 0.5f
	};
	sam3_memory_bank_add(&bank, &e1);
	sam3_memory_bank_add(&bank, &e5);

	int selected[8];
	int n = sam3_memory_bank_select_non_cond_for_frame(
		&bank, /*current_frame=*/10,
		/*num_maskmem=*/4, selected, 8);

	ASSERT_EQ(n, 2);
	ASSERT_EQ(bank.non_cond[selected[0]].frame_idx, 5);
	ASSERT_EQ(bank.non_cond[selected[1]].frame_idx, 1);
}

int main(void)
{
	test_bank_init();
	test_bank_add_cond();
	test_bank_add_non_cond();
	test_bank_evict_oldest_non_cond();
	test_bank_sam3long_selection();
	test_bank_select_closest();
	test_bank_cond_storage_beyond_attn_cap();
	test_bank_build_view_selects_closest();
	test_bank_build_view_empty();
	test_bank_clear();
	test_bank_clear_around_frame();
	test_bank_select_non_cond_for_frame();
	test_bank_select_non_cond_simple_skip_on_dup();
	TEST_REPORT();
}
