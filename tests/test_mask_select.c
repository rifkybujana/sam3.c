/*
 * tests/test_mask_select.c - Unit tests for stability-based mask selection.
 *
 * Verifies the stability score algorithm selects token 0 when stable,
 * and falls back to the best multimask when unstable.
 *
 * Key types:  none
 * Depends on: sam3/internal/mask_select.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <string.h>
#include "test_helpers.h"
#include "sam3/internal/mask_select.h"

static void fill_uniform(float *m, float val)
{
	for (int i = 0; i < 16; i++)
		m[i] = val;
}

static void test_select_stable_token0(void)
{
	float masks[4 * 16];
	float scores[4] = { 0.8f, 0.9f, 0.7f, 0.6f };

	fill_uniform(&masks[0 * 16], 2.0f);
	fill_uniform(&masks[1 * 16], 1.0f);
	fill_uniform(&masks[2 * 16], 1.0f);
	fill_uniform(&masks[3 * 16], 1.0f);

	int best = sam3_mask_select_best(masks, scores, 4, 4, 4,
					  SAM3_STABILITY_DELTA,
					  SAM3_STABILITY_THRESH);
	ASSERT_EQ(best, 0);
}

static void test_select_unstable_picks_best_multimask(void)
{
	float masks[4 * 16];
	float scores[4] = { 0.5f, 0.7f, 0.95f, 0.6f };

	for (int i = 0; i < 8; i++)
		masks[i] = 0.03f;
	for (int i = 8; i < 16; i++)
		masks[i] = 0.10f;

	fill_uniform(&masks[1 * 16], 1.0f);
	fill_uniform(&masks[2 * 16], 1.0f);
	fill_uniform(&masks[3 * 16], 1.0f);

	int best = sam3_mask_select_best(masks, scores, 4, 4, 4,
					  SAM3_STABILITY_DELTA,
					  SAM3_STABILITY_THRESH);
	ASSERT_EQ(best, 2);
}

static void test_select_empty_token0_selects_multimask(void)
{
	float masks[4 * 16];
	float scores[4] = { 0.3f, 0.9f, 0.8f, 0.7f };

	fill_uniform(&masks[0 * 16], -1.0f);
	fill_uniform(&masks[1 * 16],  1.0f);
	fill_uniform(&masks[2 * 16],  1.0f);
	fill_uniform(&masks[3 * 16],  1.0f);

	int best = sam3_mask_select_best(masks, scores, 4, 4, 4,
					  SAM3_STABILITY_DELTA,
					  SAM3_STABILITY_THRESH);
	ASSERT_EQ(best, 0); /* empty mask is "stable" per reference */
}

static void test_select_single_mask(void)
{
	float masks[16];
	float scores[1] = { 0.5f };
	fill_uniform(masks, 1.0f);

	int best = sam3_mask_select_best(masks, scores, 1, 4, 4,
					  SAM3_STABILITY_DELTA,
					  SAM3_STABILITY_THRESH);
	ASSERT_EQ(best, 0);
}

static void test_select_invalid_args(void)
{
	float masks[16], scores[1] = { 0.5f };
	ASSERT_EQ(sam3_mask_select_best(NULL, scores, 1, 4, 4, 0.05f, 0.98f), -1);
	ASSERT_EQ(sam3_mask_select_best(masks, NULL, 1, 4, 4, 0.05f, 0.98f), -1);
	ASSERT_EQ(sam3_mask_select_best(masks, scores, 0, 4, 4, 0.05f, 0.98f), -1);
	ASSERT_EQ(sam3_mask_select_best(masks, scores, 1, 0, 4, 0.05f, 0.98f), -1);
}

int main(void)
{
	test_select_stable_token0();
	test_select_unstable_picks_best_multimask();
	test_select_empty_token0_selects_multimask();
	test_select_single_mask();
	test_select_invalid_args();

	TEST_REPORT();
}
