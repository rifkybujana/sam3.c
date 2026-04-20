/*
 * tests/test_multiplex_helpers.c - Unit coverage for the pure-math
 *                                  helpers in tracker_multiplex.c.
 *
 * Exercises multiplex_sine_tpos_256, multiplex_apply_linear_256, and
 * multiplex_maskmem_tpos_slot against hand-computed references. None of these
 * touch the backend, the arena, or the weight file, so the test is
 * deterministic and runs without the SAM 3.1 model.
 *
 * Key types:  (none)
 * Depends on: model/tracker_multiplex_internal.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "model/tracker_multiplex_internal.h"
#include "test_helpers.h"

/* Matches SAM3_MULTIPLEX_HIDDEN_DIM / SAM3_MULTIPLEX_NUM_MASKMEM without
 * dragging in the full tracker header — these are fundamental to the
 * helpers' contract and duplicated intentionally. */
#define D 256
#define PE_DIM (D / 2)
#define NUM_MASKMEM 7

/*
 * At norm_pos=0 the sine PE collapses to sin(0)=0 (first half) and
 * cos(0)=1 (second half) regardless of frequency. This is the simplest
 * possible sanity check that the layout is `[sin_part, cos_part]` and
 * not the reverse.
 */
static void test_sine_tpos_identity_at_zero(void)
{
	float row[D];
	memset(row, 0xAA, sizeof(row));  /* ensure the helper actually writes */
	multiplex_sine_tpos_256(row, 0.0f);
	for (int j = 0; j < PE_DIM; j++)
		ASSERT_NEAR(row[j], 0.0f, 1e-6);
	for (int j = 0; j < PE_DIM; j++)
		ASSERT_NEAR(row[PE_DIM + j], 1.0f, 1e-6);
}

/*
 * Reproduce the Python formula element-for-element at a nontrivial
 * position. Temperature is 10000, dim_t[j] = temperature^(2*(j/2)/128),
 * so adjacent (j, j+1) pairs share the same dim_t, i.e. sin and cos of
 * the same argument.
 */
static void test_sine_tpos_matches_python_formula(void)
{
	const float norm_pos = 0.25f;
	float row[D];
	multiplex_sine_tpos_256(row, norm_pos);

	for (int j = 0; j < PE_DIM; j++) {
		float exponent = (float)(2 * (j / 2)) / (float)PE_DIM;
		float dim_t = powf(10000.0f, exponent);
		float v = norm_pos / dim_t;
		ASSERT_NEAR(row[j],          sinf(v), 1e-6);
		ASSERT_NEAR(row[PE_DIM + j], cosf(v), 1e-6);
	}
}

/*
 * W = I (identity row-major [D, D]), b = some vector, x = random-ish.
 * Expected: y = x + b. Confirms the torch layout (W[i, j] accessed as
 * W + i*D + j) and the bias add.
 */
static void test_apply_linear_identity(void)
{
	static float W[D * D];
	static float b[D];
	static float x[D];
	static float y[D];

	memset(W, 0, sizeof(W));
	for (int i = 0; i < D; i++) {
		W[i * D + i] = 1.0f;   /* identity */
		b[i] = (float)(i % 7) * 0.1f;
		x[i] = (float)(i % 5) - 2.0f;
	}

	multiplex_apply_linear_256(y, x, 1, W, b);

	for (int i = 0; i < D; i++)
		ASSERT_NEAR(y[i], x[i] + b[i], 1e-5);
}

/*
 * b = NULL path: y should equal W @ x. Use a weight with a single
 * nonzero entry per row (permutation matrix) so the answer is a fixed
 * permutation of x.
 */
static void test_apply_linear_permutation_no_bias(void)
{
	static float W[D * D];
	static float x[D];
	static float y[D];

	memset(W, 0, sizeof(W));
	for (int i = 0; i < D; i++) {
		int j = (i + 3) % D;
		W[i * D + j] = 1.0f;
		x[i] = (float)i * 0.5f - 1.0f;
	}

	multiplex_apply_linear_256(y, x, 1, W, NULL);

	for (int i = 0; i < D; i++) {
		int j = (i + 3) % D;
		ASSERT_NEAR(y[i], x[j], 1e-5);
	}
}

/*
 * Linear-combination check: with W filled by sums of two inputs per
 * output, verify the matmul's arithmetic at a random row. Small coeffs
 * so accumulated error stays well under tolerance.
 */
static void test_apply_linear_two_nonzero_per_row(void)
{
	static float W[D * D];
	static float b[D];
	static float x[D];
	static float y[D];

	memset(W, 0, sizeof(W));
	for (int i = 0; i < D; i++) {
		int j1 = i;
		int j2 = (i + 17) % D;
		W[i * D + j1] = 0.5f;
		W[i * D + j2] = -0.25f;
		b[i] = 1.0f;
		x[i] = (float)(i % 11) - 5.0f;
	}

	multiplex_apply_linear_256(y, x, 1, W, b);

	for (int i = 0; i < D; i++) {
		int j1 = i;
		int j2 = (i + 17) % D;
		float expect = 0.5f * x[j1] + -0.25f * x[j2] + b[i];
		ASSERT_NEAR(y[i], expect, 1e-5);
	}
}

/*
 * Multi-row broadcast: two input rows produce two independent outputs.
 * Catches a regression where the helper forgot to stride src/dst per
 * row or accidentally aliased them.
 */
static void test_apply_linear_multi_row_independence(void)
{
	static float W[D * D];
	static float b[D];
	static float src[2 * D];
	static float dst[2 * D];

	memset(W, 0, sizeof(W));
	for (int i = 0; i < D; i++) {
		W[i * D + i] = 2.0f;
		b[i] = 0.5f;
	}
	for (int i = 0; i < D; i++) {
		src[i]     = (float)i;
		src[D + i] = -(float)i;
	}

	multiplex_apply_linear_256(dst, src, 2, W, b);

	for (int i = 0; i < D; i++) {
		ASSERT_NEAR(dst[i],     2.0f * (float)i  + 0.5f, 1e-5);
		ASSERT_NEAR(dst[D + i], 2.0f * -(float)i + 0.5f, 1e-5);
	}
}

/*
 * multiplex_maskmem_tpos_slot - explicit table of Python's use_maskmem_tpos_v2
 * rule for num_maskmem = 7.
 *   in-range (0 < t < 7): slot = 7 - t - 1
 *   out-of-range         : slot = 6
 */
static void test_maskmem_tpos_slot_table(void)
{
	/* In range. */
	ASSERT_EQ(multiplex_maskmem_tpos_slot(1), NUM_MASKMEM - 1 - 1);  /* 5 */
	ASSERT_EQ(multiplex_maskmem_tpos_slot(2), NUM_MASKMEM - 2 - 1);  /* 4 */
	ASSERT_EQ(multiplex_maskmem_tpos_slot(3), NUM_MASKMEM - 3 - 1);  /* 3 */
	ASSERT_EQ(multiplex_maskmem_tpos_slot(6), NUM_MASKMEM - 6 - 1);  /* 0 */

	/* Out-of-range edges. */
	ASSERT_EQ(multiplex_maskmem_tpos_slot(0),  NUM_MASKMEM - 1);      /* t_rel == 0 */
	ASSERT_EQ(multiplex_maskmem_tpos_slot(-1), NUM_MASKMEM - 1);      /* negative */
	ASSERT_EQ(multiplex_maskmem_tpos_slot(NUM_MASKMEM),     NUM_MASKMEM - 1);
	ASSERT_EQ(multiplex_maskmem_tpos_slot(NUM_MASKMEM + 1), NUM_MASKMEM - 1);
	ASSERT_EQ(multiplex_maskmem_tpos_slot(1000),            NUM_MASKMEM - 1);
}

int main(void)
{
	test_sine_tpos_identity_at_zero();
	test_sine_tpos_matches_python_formula();
	test_apply_linear_identity();
	test_apply_linear_permutation_no_bias();
	test_apply_linear_two_nonzero_per_row();
	test_apply_linear_multi_row_independence();
	test_maskmem_tpos_slot_table();

	TEST_REPORT();
}
