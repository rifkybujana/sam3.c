/*
 * tests/test_dynamic_multimask.c - dynamic_multimask_via_stability
 *
 * Unit tests for the stability-aware mask selector:
 * - unstable mask with high IoU is rejected in favor of a stable
 *   one with slightly lower IoU
 * - single-mask input bypasses stability (returns index 0)
 * - two-mask input bypasses stability (n_masks < 3)
 * - zero delta / zero thresh fall back to plain argmax IoU
 *
 * Key types: (none; function-level)
 * Depends on: model/mask_decoder.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include <stdlib.h>

#include "test_helpers.h"
#include "model/mask_decoder.h"

static void test_unstable_mask_rejected(void)
{
	int H = 4, W = 4;
	float logits[3 * 16];

	/* Mask 0: every pixel near zero (|logit| < delta=0.05) → both
	 * areas tiny, ratio unstable as a fraction of noise. Area at
	 * +0.05 is ~0; area at -0.05 is ~16 → stab = 0/16 = 0.
	 * Note: exactly 0 < thresh=0.98, so rejected. */
	for (int i = 0; i < 16; i++)
		logits[0 * 16 + i] = 0.001f;
	/* Mask 1: half of pixels above +0.05, half below -0.05. */
	for (int i = 0; i < 16; i++)
		logits[1 * 16 + i] = (i < 8) ? 1.0f : -1.0f;
	/* Mask 2: similar solid pattern. */
	for (int i = 0; i < 16; i++)
		logits[2 * 16 + i] = (i < 7) ? 1.0f : -1.0f;

	/* Stable areas: mask 1 → area_hi=8, area_lo=8, stab=1.0
	 *               mask 2 → area_hi=7, area_lo=7, stab=1.0
	 *               mask 0 → area_hi=0, area_lo=16, stab=0 */
	float iou[3] = {0.95f, 0.90f, 0.92f};

	/* Best IoU is mask 0 (unstable). Best stable IoU is mask 2. */
	int picked = sam3_mask_decoder_select_with_stability(
		logits, iou, 3, H, W, 0.05f, 0.98f);
	ASSERT_EQ(picked, 2);
}

static void test_single_mask_bypasses_stability(void)
{
	float logits[16] = {0};
	float iou[1] = {0.9f};
	int picked = sam3_mask_decoder_select_with_stability(
		logits, iou, 1, 4, 4, 0.05f, 0.98f);
	ASSERT_EQ(picked, 0);
}

static void test_two_masks_bypass_stability(void)
{
	float logits[2 * 16] = {0};
	float iou[2] = {0.3f, 0.9f};
	int picked = sam3_mask_decoder_select_with_stability(
		logits, iou, 2, 4, 4, 0.05f, 0.98f);
	ASSERT_EQ(picked, 1);
}

static void test_zero_thresh_falls_back_to_argmax_iou(void)
{
	float logits[3 * 16] = {0};
	float iou[3] = {0.1f, 0.9f, 0.2f};
	int picked = sam3_mask_decoder_select_with_stability(
		logits, iou, 3, 4, 4, /*delta=*/0.0f, /*thresh=*/0.0f);
	ASSERT_EQ(picked, 1);
}

int main(void)
{
	test_unstable_mask_rejected();
	test_single_mask_bypasses_stability();
	test_two_masks_bypass_stability();
	test_zero_thresh_falls_back_to_argmax_iou();
	TEST_REPORT();
}
