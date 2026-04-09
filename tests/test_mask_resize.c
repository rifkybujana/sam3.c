/*
 * tests/test_mask_resize.c - Unit tests for bilinear mask resize.
 *
 * Verifies bilinear interpolation against hand-computed values and
 * known properties (identity resize, uniform value preservation).
 *
 * Key types:  none
 * Depends on: sam3/internal/mask_resize.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "sam3/internal/mask_resize.h"

static void test_resize_identity(void)
{
	/* 2x2 -> 2x2: output must equal input. */
	float src[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
	float dst[4];
	int rc = sam3_mask_resize_bilinear(src, 2, 2, dst, 2, 2);
	ASSERT_EQ(rc, 0);
	ASSERT_NEAR(dst[0], 1.0f, 1e-5f);
	ASSERT_NEAR(dst[1], 2.0f, 1e-5f);
	ASSERT_NEAR(dst[2], 3.0f, 1e-5f);
	ASSERT_NEAR(dst[3], 4.0f, 1e-5f);
}

static void test_resize_uniform(void)
{
	/* A uniform mask should stay uniform after resize. */
	float src[4] = { 5.0f, 5.0f, 5.0f, 5.0f };
	float dst[16]; /* 4x4 */
	int rc = sam3_mask_resize_bilinear(src, 2, 2, dst, 4, 4);
	ASSERT_EQ(rc, 0);
	for (int i = 0; i < 16; i++)
		ASSERT_NEAR(dst[i], 5.0f, 1e-5f);
}

static void test_resize_2x2_to_4x4(void)
{
	/*
	 * 2x2 source:
	 *   0  1
	 *   0  1
	 *
	 * Upsample to 4x4. align_corners=False means:
	 *   src_x = (dst_x + 0.5) * (2/4) - 0.5
	 *
	 * For dst_x=0: src_x = 0.25 - 0.5 = -0.25 -> clamped to 0.0
	 * For dst_x=1: src_x = 0.75 - 0.5 = 0.25
	 * For dst_x=2: src_x = 1.25 - 0.5 = 0.75
	 * For dst_x=3: src_x = 1.75 - 0.5 = 1.25 -> clamped to 1.0
	 *
	 * With src row = [0, 1]:
	 *   x=0: lerp(0,1, 0.0) = 0.0
	 *   x=1: lerp(0,1, 0.25) = 0.25
	 *   x=2: lerp(0,1, 0.75) = 0.75
	 *   x=3: lerp(0,1, 1.0) = 1.0
	 *
	 * Both rows identical -> all y-lerps preserve x values.
	 */
	float src[4] = { 0.0f, 1.0f, 0.0f, 1.0f };
	float dst[16];
	int rc = sam3_mask_resize_bilinear(src, 2, 2, dst, 4, 4);
	ASSERT_EQ(rc, 0);

	/* Each row should be [0.0, 0.25, 0.75, 1.0] */
	for (int y = 0; y < 4; y++) {
		ASSERT_NEAR(dst[y * 4 + 0], 0.0f,  1e-5f);
		ASSERT_NEAR(dst[y * 4 + 1], 0.25f, 1e-5f);
		ASSERT_NEAR(dst[y * 4 + 2], 0.75f, 1e-5f);
		ASSERT_NEAR(dst[y * 4 + 3], 1.0f,  1e-5f);
	}
}

static void test_resize_downsample_4x4_to_2x2(void)
{
	float src[16];
	for (int y = 0; y < 4; y++)
		for (int x = 0; x < 4; x++)
			src[y * 4 + x] = (float)x;
	float dst[4];
	int rc = sam3_mask_resize_bilinear(src, 4, 4, dst, 2, 2);
	ASSERT_EQ(rc, 0);
	ASSERT_NEAR(dst[0], 0.5f, 1e-5f);
	ASSERT_NEAR(dst[1], 2.5f, 1e-5f);
	ASSERT_NEAR(dst[2], 0.5f, 1e-5f);
	ASSERT_NEAR(dst[3], 2.5f, 1e-5f);
}

static void test_resize_1x1_to_3x3(void)
{
	float src[1] = { 7.0f };
	float dst[9];
	int rc = sam3_mask_resize_bilinear(src, 1, 1, dst, 3, 3);
	ASSERT_EQ(rc, 0);
	for (int i = 0; i < 9; i++)
		ASSERT_NEAR(dst[i], 7.0f, 1e-5f);
}

static void test_resize_invalid_args(void)
{
	float src[4], dst[4];
	ASSERT_EQ(sam3_mask_resize_bilinear(NULL, 2, 2, dst, 2, 2), -1);
	ASSERT_EQ(sam3_mask_resize_bilinear(src, 2, 2, NULL, 2, 2), -1);
	ASSERT_EQ(sam3_mask_resize_bilinear(src, 0, 2, dst, 2, 2), -1);
	ASSERT_EQ(sam3_mask_resize_bilinear(src, 2, 0, dst, 2, 2), -1);
	ASSERT_EQ(sam3_mask_resize_bilinear(src, 2, 2, dst, 0, 2), -1);
	ASSERT_EQ(sam3_mask_resize_bilinear(src, 2, 2, dst, 2, 0), -1);
}

int main(void)
{
	test_resize_identity();
	test_resize_uniform();
	test_resize_2x2_to_4x4();
	test_resize_downsample_4x4_to_2x2();
	test_resize_1x1_to_3x3();
	test_resize_invalid_args();

	TEST_REPORT();
}
