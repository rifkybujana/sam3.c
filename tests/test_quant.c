/*
 * tests/test_quant.c - Unit tests for Q8_0 quantization
 *
 * Tests dtype registration, block size helpers, round-trip quantize/
 * dequantize accuracy, and edge cases (zeros, max values, non-aligned
 * tail elements).
 *
 * Key types:  sam3_q8_block
 * Depends on: core/tensor.h, sam3/sam3_types.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "core/tensor.h"
#include "core/quant.h"
#include "sam3/sam3_types.h"

#include <math.h>
#include <string.h>

static void test_q8_dtype_basics(void)
{
	/* Q8_0 is a block type, not per-element */
	ASSERT_EQ(sam3_dtype_size(SAM3_DTYPE_Q8_0), (size_t)0);

	/* String representation */
	ASSERT(strcmp(sam3_dtype_str(SAM3_DTYPE_Q8_0), "Q8_0") == 0);

	/* DTYPE_COUNT updated */
	ASSERT(SAM3_DTYPE_Q8_0 < SAM3_DTYPE_COUNT);
}

static void test_q8_block_count(void)
{
	/* Exact multiple of 32 */
	ASSERT_EQ(sam3_q8_block_count(32), 1);
	ASSERT_EQ(sam3_q8_block_count(64), 2);
	ASSERT_EQ(sam3_q8_block_count(1024), 32);

	/* Non-multiple: rounds up */
	ASSERT_EQ(sam3_q8_block_count(1), 1);
	ASSERT_EQ(sam3_q8_block_count(33), 2);
	ASSERT_EQ(sam3_q8_block_count(63), 2);
}

static void test_q8_nbytes(void)
{
	/* 32 elements = 1 block = 36 bytes */
	ASSERT_EQ(sam3_q8_nbytes(32), (size_t)36);

	/* 64 elements = 2 blocks = 72 bytes */
	ASSERT_EQ(sam3_q8_nbytes(64), (size_t)72);

	/* 33 elements = 2 blocks (tail padded) = 72 bytes */
	ASSERT_EQ(sam3_q8_nbytes(33), (size_t)72);
}

static void test_q8_round_trip_simple(void)
{
	/* 32 values exactly = 1 block */
	float src[32];
	float dst[32];
	struct sam3_q8_block blk[1];

	for (int i = 0; i < 32; i++)
		src[i] = (float)(i - 16) * 0.5f;

	sam3_q8_quantize(src, blk, 32);
	sam3_q8_dequantize(blk, dst, 32);

	/* Tolerance: scale = max(|val|)/127 = 8.0/127 ~ 0.063 */
	/* Max quantization error = scale/2 ~ 0.031 */
	for (int i = 0; i < 32; i++)
		ASSERT_NEAR(dst[i], src[i], 0.04f);
}

static void test_q8_round_trip_zeros(void)
{
	float src[32];
	float dst[32];
	struct sam3_q8_block blk[1];

	memset(src, 0, sizeof(src));
	sam3_q8_quantize(src, blk, 32);

	ASSERT_NEAR(blk[0].scale, 0.0f, 1e-10f);

	sam3_q8_dequantize(blk, dst, 32);
	for (int i = 0; i < 32; i++)
		ASSERT_NEAR(dst[i], 0.0f, 1e-10f);
}

static void test_q8_round_trip_tail(void)
{
	/* 40 elements = 2 blocks, second block has 8 real + 24 padded */
	float src[40];
	float dst[40];
	struct sam3_q8_block blk[2];

	for (int i = 0; i < 40; i++)
		src[i] = (float)i * 0.1f;

	sam3_q8_quantize(src, blk, 40);
	sam3_q8_dequantize(blk, dst, 40);

	for (int i = 0; i < 40; i++)
		ASSERT_NEAR(dst[i], src[i], 0.04f);
}

static void test_q8_round_trip_large(void)
{
	/* 1024 elements = 32 blocks, larger magnitude */
	float src[1024];
	float dst[1024];
	struct sam3_q8_block blk[32];

	for (int i = 0; i < 1024; i++)
		src[i] = sinf((float)i * 0.1f) * 10.0f;

	sam3_q8_quantize(src, blk, 1024);
	sam3_q8_dequantize(blk, dst, 1024);

	/* Per-block scale ~ 10/127 ~ 0.079, max error ~ 0.04 */
	for (int i = 0; i < 1024; i++)
		ASSERT_NEAR(dst[i], src[i], 0.08f);
}

int main(void)
{
	test_q8_dtype_basics();
	test_q8_block_count();
	test_q8_nbytes();
	test_q8_round_trip_simple();
	test_q8_round_trip_zeros();
	test_q8_round_trip_tail();
	test_q8_round_trip_large();
	TEST_REPORT();
}
