/*
 * tests/test_trace.c - Unit tests for the trace system
 *
 * Tests the always-available compute functions: sam3_trace_compute_stats
 * and sam3_trace_compute_compare. Forces SAM3_HAS_TRACE on so that the
 * logging functions are compiled and exercised without crashing.
 *
 * Key types:  sam3_numeric_stats, sam3_compare_result
 * Depends on: core/trace.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_HAS_TRACE
#define SAM3_HAS_TRACE
#endif

#include "test_helpers.h"
#include "core/trace.h"

/* --- Helpers to build a simple F32 tensor on the stack --- */

static void
make_f32_tensor(struct sam3_tensor *t, float *data, int n)
{
	t->dtype     = SAM3_DTYPE_F32;
	t->n_dims    = 1;
	t->dims[0]   = n;
	t->dims[1]   = 1;
	t->dims[2]   = 1;
	t->dims[3]   = 1;
	t->strides[0] = 1;
	t->strides[1] = n;
	t->strides[2] = n;
	t->strides[3] = n;
	t->data      = data;
	t->nbytes    = (size_t)n * sizeof(float);
}

/* --- Tests  --- */

static void
test_numeric_stats_f32(void)
{
	float data[4] = {1.0f, -2.0f, 3.0f, 0.5f};
	struct sam3_tensor t;
	struct sam3_numeric_stats stats;

	make_f32_tensor(&t, data, 4);
	sam3_trace_compute_stats(&t, &stats);

	ASSERT_EQ(stats.total_elems, 4);
	ASSERT_EQ(stats.nan_count,   0);
	ASSERT_EQ(stats.inf_count,   0);
	ASSERT_NEAR(stats.min,  -2.0f, 1e-6f);
	ASSERT_NEAR(stats.max,   3.0f, 1e-6f);
	/* mean = (1 + (-2) + 3 + 0.5) / 4 = 2.5 / 4 = 0.625 */
	ASSERT_NEAR(stats.mean, 0.625f, 1e-6f);
}

static void
test_numeric_stats_nan_inf(void)
{
	float data[4];
	struct sam3_tensor t;
	struct sam3_numeric_stats stats;

	data[0] = 1.0f;
	data[1] = 0.0f / 0.0f;   /* NaN */
	data[2] = 1.0f / 0.0f;   /* +Inf */
	data[3] = -1.0f / 0.0f;  /* -Inf */

	make_f32_tensor(&t, data, 4);
	sam3_trace_compute_stats(&t, &stats);

	ASSERT_EQ(stats.total_elems, 4);
	ASSERT_EQ(stats.nan_count,   1);
	ASSERT_EQ(stats.inf_count,   2);
}

static void
test_compare_identical(void)
{
	float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
	struct sam3_tensor a, b;
	struct sam3_compare_result cmp;

	make_f32_tensor(&a, data, 4);
	make_f32_tensor(&b, data, 4);

	sam3_trace_compute_compare(&a, &b, 1e-5f, &cmp);

	ASSERT_NEAR(cmp.max_abs_error, 0.0f, 1e-9f);
	ASSERT_EQ(cmp.mismatches, 0);
}

int main(void)
{
	test_numeric_stats_f32();
	test_numeric_stats_nan_inf();
	test_compare_identical();

	TEST_REPORT();
}
