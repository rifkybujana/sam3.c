/*
 * tests/test_tensor.c - Unit tests for core tensor operations
 *
 * Tests tensor element counting, dtype sizing, and stride computation.
 * Run via: ctest --output-on-failure
 *
 * Key types:  sam3_tensor
 * Depends on: core/tensor.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "core/tensor.h"

static void test_tensor_nelems_1d(void)
{
	struct sam3_tensor t = { .n_dims = 1, .dims = {10} };
	ASSERT_EQ(sam3_tensor_nelems(&t), 10);
}

static void test_tensor_nelems_3d(void)
{
	struct sam3_tensor t = { .n_dims = 3, .dims = {3, 224, 224} };
	ASSERT_EQ(sam3_tensor_nelems(&t), 3 * 224 * 224);
}

static void test_dtype_size(void)
{
	ASSERT_EQ(sam3_dtype_size(SAM3_DTYPE_F32), 4);
	ASSERT_EQ(sam3_dtype_size(SAM3_DTYPE_F16), 2);
	ASSERT_EQ(sam3_dtype_size(SAM3_DTYPE_I8), 1);
}

static void test_tensor_strides(void)
{
	struct sam3_tensor t = { .n_dims = 3, .dims = {3, 4, 5} };
	sam3_tensor_compute_strides(&t);

	ASSERT_EQ(t.strides[2], 1);
	ASSERT_EQ(t.strides[1], 5);
	ASSERT_EQ(t.strides[0], 20);
}

int main(void)
{
	test_tensor_nelems_1d();
	test_tensor_nelems_3d();
	test_dtype_size();
	test_tensor_strides();

	TEST_REPORT();
}
