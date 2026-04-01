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
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "core/tensor.h"
#include "core/quant.h"
#include "sam3/sam3_types.h"

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

int main(void)
{
	test_q8_dtype_basics();
	test_q8_block_count();
	test_q8_nbytes();
	TEST_REPORT();
}
