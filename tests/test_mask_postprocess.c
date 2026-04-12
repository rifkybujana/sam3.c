/*
 * tests/test_mask_postprocess.c - Tests for mask post-processing utilities.
 *
 * Key types:  none
 * Depends on: sam3/internal/mask_postprocess.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <string.h>
#include "test_helpers.h"
#include "sam3/internal/mask_postprocess.h"

static void test_sigmoid_basic(void)
{
	float src[] = {0.0f, 10.0f, -10.0f};
	float dst[3];

	ASSERT_EQ(sam3_mask_sigmoid(src, dst, 3), 0);
	ASSERT_NEAR(dst[0], 0.5f, 1e-5);
	ASSERT(dst[1] > 0.999f);
	ASSERT(dst[2] < 0.001f);
}

static void test_sigmoid_rejects_null(void)
{
	float buf[1];
	ASSERT_EQ(sam3_mask_sigmoid(NULL, buf, 1), -1);
	ASSERT_EQ(sam3_mask_sigmoid(buf, NULL, 1), -1);
}

static void test_morpho_removes_isolated_pixel(void)
{
	/* 5x5 mask with a single isolated pixel at (2,2).
	 * Erode removes it, dilate does not restore it. */
	unsigned char mask[25] = {0};
	unsigned char out[25];
	unsigned char work[25];

	mask[2 * 5 + 2] = 1;

	ASSERT_EQ(sam3_mask_morpho_open(mask, out, 5, 5, work), 0);
	for (int i = 0; i < 25; i++)
		ASSERT_EQ(out[i], 0);
}

static void test_morpho_keeps_solid_block(void)
{
	/* 5x5 mask with a solid 3x3 block in center.
	 * Erode shrinks to 1x1, dilate restores to ~3x3. */
	unsigned char mask[25] = {0};
	unsigned char out[25];
	unsigned char work[25];

	for (int y = 1; y <= 3; y++)
		for (int x = 1; x <= 3; x++)
			mask[y * 5 + x] = 1;

	ASSERT_EQ(sam3_mask_morpho_open(mask, out, 5, 5, work), 0);
	ASSERT_EQ(out[2 * 5 + 2], 1);
}

static void test_remove_small_filters_tiny_blob(void)
{
	/* 6x6 mask: large component (9 pixels) + small component (2 pixels) */
	unsigned char mask[36] = {0};
	int labels[36];
	int stack[36];

	/* Large blob: 3x3 at top-left */
	for (int y = 0; y < 3; y++)
		for (int x = 0; x < 3; x++)
			mask[y * 6 + x] = 1;

	/* Small blob: 2 pixels at bottom-right */
	mask[5 * 6 + 4] = 1;
	mask[5 * 6 + 5] = 1;

	ASSERT_EQ(sam3_mask_remove_small(mask, 6, 6, 5, labels, stack), 0);

	/* Large blob should survive */
	ASSERT_EQ(mask[0], 1);
	ASSERT_EQ(mask[1 * 6 + 1], 1);

	/* Small blob should be removed */
	ASSERT_EQ(mask[5 * 6 + 4], 0);
	ASSERT_EQ(mask[5 * 6 + 5], 0);
}

int main(void)
{
	test_sigmoid_basic();
	test_sigmoid_rejects_null();
	test_morpho_removes_isolated_pixel();
	test_morpho_keeps_solid_block();
	test_remove_small_filters_tiny_blob();
	TEST_REPORT();
}
