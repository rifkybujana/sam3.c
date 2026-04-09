/*
 * tests/test_processor_normalize.c - Unit tests for pixel normalization.
 *
 * Verifies the uint8 HWC to float CHW normalization used by the image
 * processor matches the Python reference (x/255 - 0.5) / 0.5, producing
 * values in [-1, 1].
 *
 * Key types:  none
 * Depends on: sam3/internal/processor_normalize.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include <stdint.h>
#include "test_helpers.h"
#include "sam3/internal/processor_normalize.h"

static void test_normalize_zero_maps_to_minus_one(void)
{
	uint8_t src[3] = { 0, 0, 0 };
	float dst[3] = { 0 };
	sam3_normalize_rgb_chw(src, dst, 1, 1);
	ASSERT_NEAR(dst[0], -1.0f, 1e-6f);
	ASSERT_NEAR(dst[1], -1.0f, 1e-6f);
	ASSERT_NEAR(dst[2], -1.0f, 1e-6f);
}

static void test_normalize_255_maps_to_plus_one(void)
{
	uint8_t src[3] = { 255, 255, 255 };
	float dst[3] = { 0 };
	sam3_normalize_rgb_chw(src, dst, 1, 1);
	ASSERT_NEAR(dst[0], 1.0f, 1e-6f);
	ASSERT_NEAR(dst[1], 1.0f, 1e-6f);
	ASSERT_NEAR(dst[2], 1.0f, 1e-6f);
}

static void test_normalize_midpoint_maps_to_zero(void)
{
	/* 127.5 is the midpoint; uint8 127 and 128 straddle it. */
	uint8_t src[6] = { 127, 127, 127, 128, 128, 128 };
	float dst[6] = { 0 };
	sam3_normalize_rgb_chw(src, dst, 2, 1);
	/* pixel 0 (uint8=127) -> 127/127.5 - 1 = -0.003921568... */
	ASSERT_NEAR(dst[0], -0.003921568f, 1e-6f);
	/* pixel 1 (uint8=128) -> 128/127.5 - 1 = +0.003921568... */
	ASSERT_NEAR(dst[3], 0.003921568f, 1e-6f);
}

static void test_normalize_chw_layout(void)
{
	/* 2x1 image, interleaved HWC: [R0 G0 B0 R1 G1 B1]
	 * CHW output order: [R0 R1 | G0 G1 | B0 B1] */
	uint8_t src[6] = { 255, 0, 0,   0, 255, 0 };
	float dst[6] = { 0 };
	sam3_normalize_rgb_chw(src, dst, 2, 1);
	/* channel R (offset 0): pixel0=255->+1, pixel1=0->-1 */
	ASSERT_NEAR(dst[0], 1.0f, 1e-6f);
	ASSERT_NEAR(dst[1], -1.0f, 1e-6f);
	/* channel G (offset 2): pixel0=0->-1, pixel1=255->+1 */
	ASSERT_NEAR(dst[2], -1.0f, 1e-6f);
	ASSERT_NEAR(dst[3], 1.0f, 1e-6f);
	/* channel B (offset 4): pixel0=0->-1, pixel1=0->-1 */
	ASSERT_NEAR(dst[4], -1.0f, 1e-6f);
	ASSERT_NEAR(dst[5], -1.0f, 1e-6f);
}

int main(void)
{
	test_normalize_zero_maps_to_minus_one();
	test_normalize_255_maps_to_plus_one();
	test_normalize_midpoint_maps_to_zero();
	test_normalize_chw_layout();

	TEST_REPORT();
}
