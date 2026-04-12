/*
 * tests/test_mask_nms.c - Unit tests for mask NMS post-processing.
 *
 * Verifies greedy non-maximum suppression on binary masks with
 * score-based prefiltering. Uses tiny hand-constructed masks so the
 * correctness is easy to reason about.
 *
 * Key types:  none
 * Depends on: sam3/internal/mask_nms.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <string.h>
#include "test_helpers.h"
#include "sam3/internal/mask_nms.h"

/*
 * Build a tiny 4x4 mask where pixels in the half-open rectangle
 * [y0, y1) x [x0, x1) are +1.0 (active under thresh 0.0) and all
 * other pixels are -1.0.
 */
static void fill_rect_mask(float *m, int x0, int y0, int x1, int y1)
{
	int x, y;
	for (y = 0; y < 4; y++) {
		for (x = 0; x < 4; x++) {
			int active = (x >= x0 && x < x1 &&
				      y >= y0 && y < y1);
			m[y * 4 + x] = active ? 1.0f : -1.0f;
		}
	}
}

static void test_nms_keeps_highest_when_identical(void)
{
	/* Two identical full masks, scores 0.9 and 0.5.
	 * Expect only the higher-score one kept. */
	float masks[2 * 16];
	float scores[2] = { 0.9f, 0.5f };
	int   kept[2];
	int   n_kept;

	fill_rect_mask(&masks[0 * 16], 0, 0, 4, 4);
	fill_rect_mask(&masks[1 * 16], 0, 0, 4, 4);

	n_kept = sam3_mask_nms(masks, scores, 2, 4, 4,
			       0.0f /* prob_thresh */,
			       0.5f /* iou_thresh */,
			       0.0f, kept);

	ASSERT_EQ(n_kept, 1);
	ASSERT_EQ(kept[0], 0);
}

static void test_nms_keeps_disjoint(void)
{
	/* Two disjoint masks, both high score. Expect both kept. */
	float masks[2 * 16];
	float scores[2] = { 0.9f, 0.8f };
	int   kept[2];
	int   n_kept;

	fill_rect_mask(&masks[0 * 16], 0, 0, 2, 2); /* top-left 2x2 */
	fill_rect_mask(&masks[1 * 16], 2, 2, 4, 4); /* bot-right 2x2 */

	n_kept = sam3_mask_nms(masks, scores, 2, 4, 4,
			       0.0f, 0.5f, 0.0f, kept);

	ASSERT_EQ(n_kept, 2);
}

static void test_nms_prefilters_by_score(void)
{
	/* One mask with score 0.9, one with score 0.1.
	 * prob_thresh=0.5 drops the second. */
	float masks[2 * 16];
	float scores[2] = { 0.9f, 0.1f };
	int   kept[2];
	int   n_kept;

	fill_rect_mask(&masks[0 * 16], 0, 0, 2, 2);
	fill_rect_mask(&masks[1 * 16], 2, 2, 4, 4);

	n_kept = sam3_mask_nms(masks, scores, 2, 4, 4,
			       0.5f, 0.5f, 0.0f, kept);

	ASSERT_EQ(n_kept, 1);
	ASSERT_EQ(kept[0], 0);
}

static void test_nms_orders_by_score(void)
{
	/* Three disjoint high-score masks, unsorted input scores.
	 * kept[] should be in descending-score order. */
	float masks[3 * 16];
	float scores[3] = { 0.5f, 0.9f, 0.7f };
	int   kept[3];
	int   n_kept;

	fill_rect_mask(&masks[0 * 16], 0, 0, 1, 1);
	fill_rect_mask(&masks[1 * 16], 1, 1, 2, 2);
	fill_rect_mask(&masks[2 * 16], 2, 2, 3, 3);

	n_kept = sam3_mask_nms(masks, scores, 3, 4, 4,
			       0.0f, 0.5f, 0.0f, kept);

	ASSERT_EQ(n_kept, 3);
	ASSERT_EQ(kept[0], 1); /* 0.9 */
	ASSERT_EQ(kept[1], 2); /* 0.7 */
	ASSERT_EQ(kept[2], 0); /* 0.5 */
}

static void test_nms_drops_overlapping_lower_score(void)
{
	/* mask A covers whole image score 0.9; mask B overlaps fully
	 * but score 0.7 → dropped. mask C disjoint score 0.5 → kept. */
	float masks[3 * 16];
	float scores[3] = { 0.9f, 0.7f, 0.5f };
	int   kept[3];
	int   n_kept;

	fill_rect_mask(&masks[0 * 16], 0, 0, 4, 4); /* whole 4x4 */
	fill_rect_mask(&masks[1 * 16], 1, 1, 3, 3); /* inside A */
	fill_rect_mask(&masks[2 * 16], 0, 0, 1, 1); /* corner, inside A */

	/* IoU(A,B) = 4/16 = 0.25, IoU(A,C) = 1/16 = 0.0625,
	 * IoU(B,C) = 0/5 = 0. With iou_thresh=0.2, B dropped. */
	n_kept = sam3_mask_nms(masks, scores, 3, 4, 4,
			       0.0f, 0.2f, 0.0f, kept);

	ASSERT_EQ(n_kept, 2);
	ASSERT_EQ(kept[0], 0);
	ASSERT_EQ(kept[1], 2);
}

static void test_nms_invalid_args_returns_minus_one(void)
{
	/* Invalid arguments must return -1. */
	float masks[16];
	float scores[1] = { 0.5f };
	int   kept[1];

	ASSERT_EQ(sam3_mask_nms(NULL, scores, 1, 4, 4, 0.0f, 0.5f,
				0.0f, kept), -1);
	ASSERT_EQ(sam3_mask_nms(masks, NULL, 1, 4, 4, 0.0f, 0.5f,
				0.0f, kept), -1);
	ASSERT_EQ(sam3_mask_nms(masks, scores, 1, 4, 4, 0.0f, 0.5f,
				0.0f, NULL), -1);
	ASSERT_EQ(sam3_mask_nms(masks, scores, 0, 4, 4, 0.0f, 0.5f,
				0.0f, kept), -1);
	ASSERT_EQ(sam3_mask_nms(masks, scores, 1, 0, 4, 0.0f, 0.5f,
				0.0f, kept), -1);
	ASSERT_EQ(sam3_mask_nms(masks, scores, 1, 4, 0, 0.0f, 0.5f,
				0.0f, kept), -1);
	ASSERT_EQ(sam3_mask_nms(masks, scores, 513, 4, 4, 0.0f, 0.5f,
				0.0f, kept), -1);
}

static void test_nms_all_filtered_returns_zero(void)
{
	/* All scores below prob_thresh → kept set is empty. */
	float masks[2 * 16];
	float scores[2] = { 0.1f, 0.2f };
	int   kept[2];
	int   n_kept;

	fill_rect_mask(&masks[0 * 16], 0, 0, 2, 2);
	fill_rect_mask(&masks[1 * 16], 2, 2, 4, 4);

	n_kept = sam3_mask_nms(masks, scores, 2, 4, 4,
			       0.5f, 0.5f, 0.0f, kept);

	ASSERT_EQ(n_kept, 0);
}

static void test_nms_single_candidate(void)
{
	/* One mask above prob_thresh → kept. */
	float masks[1 * 16];
	float scores[1] = { 0.9f };
	int   kept[1];
	int   n_kept;

	fill_rect_mask(&masks[0 * 16], 0, 0, 2, 2);

	n_kept = sam3_mask_nms(masks, scores, 1, 4, 4,
			       0.0f, 0.5f, 0.0f, kept);

	ASSERT_EQ(n_kept, 1);
	ASSERT_EQ(kept[0], 0);
}

static void test_nms_quality_floor_rejects_noisy_mask(void)
{
	/* Mask 0: all pixels confident (logit=5.0). Score=0.9.
	 * Mask 1: all pixels near-zero (logit=0.01). Score=0.8.
	 * With min_quality=0.1, mask 1 should be rejected. */
	float masks[2 * 16];
	float scores[2] = { 0.9f, 0.8f };
	int kept[2];
	int n_kept;

	/* Mask 0: confident (top half positive, bottom negative) */
	for (int i = 0; i < 16; i++)
		masks[0 * 16 + i] = (i < 8) ? 5.0f : -5.0f;

	/* Mask 1: noisy (all near zero) */
	for (int i = 0; i < 16; i++)
		masks[1 * 16 + i] = (i % 2) ? 0.01f : -0.01f;

	n_kept = sam3_mask_nms(masks, scores, 2, 4, 4,
			       0.0f, 0.5f, 0.1f, kept);

	ASSERT_EQ(n_kept, 1);
	ASSERT_EQ(kept[0], 0);
}

static void test_nms_quality_zero_is_noop(void)
{
	/* Same setup but min_quality=0 should keep both */
	float masks[2 * 16];
	float scores[2] = { 0.9f, 0.8f };
	int kept[2];
	int n_kept;

	for (int i = 0; i < 16; i++)
		masks[0 * 16 + i] = (i < 8) ? 5.0f : -5.0f;
	for (int i = 0; i < 16; i++)
		masks[1 * 16 + i] = (i % 2) ? 0.01f : -0.01f;

	n_kept = sam3_mask_nms(masks, scores, 2, 4, 4,
			       0.0f, 0.5f, 0.0f, kept);

	ASSERT_EQ(n_kept, 2);
}

int main(void)
{
	test_nms_keeps_highest_when_identical();
	test_nms_keeps_disjoint();
	test_nms_prefilters_by_score();
	test_nms_orders_by_score();
	test_nms_drops_overlapping_lower_score();
	test_nms_invalid_args_returns_minus_one();
	test_nms_all_filtered_returns_zero();
	test_nms_single_candidate();
	test_nms_quality_floor_rejects_noisy_mask();
	test_nms_quality_zero_is_noop();

	TEST_REPORT();
}
