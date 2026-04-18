/*
 * tests/test_helpers_test.c - Self-tests for test_helpers.h
 *
 * Exercises the tolerance-math helper underpinning ASSERT_TENSOR_CLOSE
 * so regressions in the comparison logic are caught without requiring a
 * disk fixture. The mismatch branch calls exit(1) and is therefore not
 * covered here; this file only verifies passing paths (exact match and
 * within-tolerance) plus the shape/dtype checks layered on top via the
 * ASSERT_TENSOR_CLOSE macro itself.
 *
 * Key types:  sam3_tensor
 * Depends on: test_helpers.h, core/tensor.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"

#include "core/tensor.h"

static void test_close_exact_match(void)
{
	float a[4] = {0.0f, 1.0f, -2.5f, 3.14159f};
	float b[4] = {0.0f, 1.0f, -2.5f, 3.14159f};

	assert_tensor_close_f32(a, b, 4, 0.0f, 0.0f, "exact");
	ASSERT(1);
}

static void test_close_within_atol(void)
{
	float a[3] = {1.000001f, 2.0f, -3.0f};
	float b[3] = {1.000000f, 2.0f, -3.0f};

	/* atol = 1e-5 is plenty for a 1e-6 deviation. */
	assert_tensor_close_f32(a, b, 3, 0.0f, 1e-5f, "atol");
	ASSERT(1);
}

static void test_close_within_rtol(void)
{
	float a[2] = {1000.0f + 0.5f, -2000.0f};
	float b[2] = {1000.0f,          -2000.0f};

	/* rtol = 1e-3 * 1000 = 1.0 > 0.5 diff. */
	assert_tensor_close_f32(a, b, 2, 1e-3f, 0.0f, "rtol");
	ASSERT(1);
}

static void test_assert_tensor_close_macro(void)
{
	float adata[6] = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
	float edata[6] = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f};

	struct sam3_tensor actual = {
		.dtype  = SAM3_DTYPE_F32,
		.n_dims = 2,
		.dims   = {2, 3},
		.data   = adata,
	};
	struct sam3_tensor expected = {
		.dtype  = SAM3_DTYPE_F32,
		.n_dims = 2,
		.dims   = {2, 3},
		.data   = edata,
	};

	ASSERT_EQ(sam3_tensor_nelems(&actual), 6);
	ASSERT_TENSOR_CLOSE(&actual, &expected, 0.0f, 0.0f);
	ASSERT(1);
}

int main(void)
{
	test_close_exact_match();
	test_close_within_atol();
	test_close_within_rtol();
	test_assert_tensor_close_macro();
	TEST_REPORT();
}
