/*
 * tests/test_half.c - Unit tests for fp16/bf16 conversion utilities
 *
 * Tests scalar round-trip conversions, special-value handling (zero,
 * inf, NaN, denormals), and round-to-nearest-even behaviour for both
 * fp16 and bf16 formats. Each test function exercises one behavioural
 * aspect and uses macros from test_helpers.h.
 *
 * Key types:  uint16_t (fp16/bf16 bit patterns)
 * Depends on: core/half.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "core/half.h"

#include <stdint.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* fp16 round-trip tests                                               */
/* ------------------------------------------------------------------ */

static void
test_fp16_round_trip(void)
{
	/*
	 * For each test value: convert f32->fp16->f32 and check that
	 * the result is within the relative error budget of fp16
	 * (~0.1% for normal values). We use a generous eps here because
	 * fp16 has only 10 mantissa bits.
	 */
	static const float vals[] = {
		0.0f, 1.0f, -1.0f, 0.5f, 65504.0f,
		6.1e-5f, 3.14f
	};
	static const float eps[] = {
		0.0f, 1e-3f, 1e-3f, 1e-4f, 1.0f,
		1e-5f, 5e-3f
	};
	int n = (int)(sizeof(vals) / sizeof(vals[0]));

	for (int i = 0; i < n; i++) {
		uint16_t h = f32_to_fp16(vals[i]);
		float    r = fp16_to_f32(h);
		ASSERT_NEAR(r, vals[i], eps[i]);
	}
}

/* ------------------------------------------------------------------ */
/* fp16 special-value tests                                            */
/* ------------------------------------------------------------------ */

static void
test_fp16_special_values(void)
{
	/* +0.0 */
	uint16_t pos_zero = f32_to_fp16(0.0f);
	ASSERT_EQ(pos_zero, (uint16_t)0x0000u);
	ASSERT_EQ(fp16_is_nan(pos_zero), 0);
	ASSERT_EQ(fp16_is_inf(pos_zero), 0);

	/* -0.0: sign bit set, exp and mantissa zero */
	uint16_t neg_zero = f32_to_fp16(-0.0f);
	ASSERT_EQ(neg_zero, (uint16_t)0x8000u);
	ASSERT_EQ(fp16_is_nan(neg_zero), 0);
	ASSERT_EQ(fp16_is_inf(neg_zero), 0);

	/* +inf */
	uint16_t pos_inf = f32_to_fp16(1.0f / 0.0f);
	ASSERT_EQ(fp16_is_inf(pos_inf), 1);
	ASSERT_EQ(fp16_is_nan(pos_inf), 0);
	ASSERT_EQ(pos_inf, (uint16_t)0x7C00u);

	/* -inf */
	uint16_t neg_inf = f32_to_fp16(-1.0f / 0.0f);
	ASSERT_EQ(fp16_is_inf(neg_inf), 1);
	ASSERT_EQ(fp16_is_nan(neg_inf), 0);
	ASSERT_EQ(neg_inf, (uint16_t)0xFC00u);

	/* NaN round-trip: the result must still be NaN */
	float    nan_f   = 0.0f / 0.0f;
	uint16_t nan_h   = f32_to_fp16(nan_f);
	ASSERT_EQ(fp16_is_nan(nan_h), 1);
	ASSERT_EQ(fp16_is_inf(nan_h), 0);

	/* fp16_to_f32 of a NaN bit pattern must produce a NaN */
	uint16_t nan_bits = 0x7E00u; /* quiet NaN */
	float    nan_r    = fp16_to_f32(nan_bits);
	ASSERT(isnan(nan_r));

	/* fp16_to_f32 of +inf bit pattern */
	float inf_r = fp16_to_f32(0x7C00u);
	ASSERT(isinf(inf_r));
	ASSERT(inf_r > 0.0f);

	/* fp16_to_f32 of -inf bit pattern */
	float neginf_r = fp16_to_f32(0xFC00u);
	ASSERT(isinf(neginf_r));
	ASSERT(neginf_r < 0.0f);
}

/* ------------------------------------------------------------------ */
/* bf16 round-trip tests                                               */
/* ------------------------------------------------------------------ */

static void
test_bf16_round_trip(void)
{
	/*
	 * bf16 has 7 mantissa bits (vs 23 for f32), giving ~0.8%
	 * relative error. eps values below are chosen to be tight
	 * enough to catch wrong sign/exponent but loose enough to
	 * pass with correct RNE rounding.
	 */
	static const float vals[] = {
		0.0f, 1.0f, -1.0f, 3.14f,
		1e10f, -1e10f, 1e-20f
	};
	static const float eps[] = {
		0.0f, 1e-2f, 1e-2f, 2e-2f,
		1e6f, 1e6f, 1e-22f
	};
	int n = (int)(sizeof(vals) / sizeof(vals[0]));

	for (int i = 0; i < n; i++) {
		uint16_t b = f32_to_bf16(vals[i]);
		float    r = bf16_to_f32(b);
		ASSERT_NEAR(r, vals[i], eps[i]);
	}
}

/* ------------------------------------------------------------------ */
/* bf16 special-value tests                                            */
/* ------------------------------------------------------------------ */

static void
test_bf16_special_values(void)
{
	/* +inf */
	uint16_t pos_inf = f32_to_bf16(1.0f / 0.0f);
	ASSERT_EQ(bf16_is_inf(pos_inf), 1);
	ASSERT_EQ(bf16_is_nan(pos_inf), 0);
	ASSERT_EQ(pos_inf, (uint16_t)0x7F80u);

	/* -inf */
	uint16_t neg_inf = f32_to_bf16(-1.0f / 0.0f);
	ASSERT_EQ(bf16_is_inf(neg_inf), 1);
	ASSERT_EQ(bf16_is_nan(neg_inf), 0);
	ASSERT_EQ(neg_inf, (uint16_t)0xFF80u);

	/* NaN: conversion must produce a NaN bit pattern */
	float    nan_f  = 0.0f / 0.0f;
	uint16_t nan_b  = f32_to_bf16(nan_f);
	ASSERT_EQ(bf16_is_nan(nan_b), 1);
	ASSERT_EQ(bf16_is_inf(nan_b), 0);

	/* bf16_to_f32 of a NaN bit pattern must produce a NaN */
	uint16_t nan_bits = 0x7FC0u; /* quiet NaN */
	float    nan_r    = bf16_to_f32(nan_bits);
	ASSERT(isnan(nan_r));

	/* bf16_to_f32 of +inf bit pattern */
	float inf_r = bf16_to_f32(0x7F80u);
	ASSERT(isinf(inf_r));
	ASSERT(inf_r > 0.0f);

	/* +0.0 */
	uint16_t pos_zero = f32_to_bf16(0.0f);
	ASSERT_EQ(pos_zero, (uint16_t)0x0000u);
	ASSERT_EQ(bf16_is_nan(pos_zero), 0);
	ASSERT_EQ(bf16_is_inf(pos_zero), 0);

	/* -0.0 */
	uint16_t neg_zero = f32_to_bf16(-0.0f);
	ASSERT_EQ(neg_zero, (uint16_t)0x8000u);
}

/* ------------------------------------------------------------------ */
/* bf16 round-to-nearest-even                                         */
/* ------------------------------------------------------------------ */

static void
test_bf16_round_to_nearest_even(void)
{
	/*
	 * 1.5f in f32 is 0x3FC00000.
	 * The lower 16 bits are 0x0000, so the round bit is 0; the result
	 * is simply the truncation: 0x3FC0.
	 *
	 * The task spec states 1.5f → 0x3FC0, which is correct because
	 * 1.5 = 1.1b in binary and fits exactly in bf16.
	 */
	uint16_t b = f32_to_bf16(1.5f);
	ASSERT_EQ(b, (uint16_t)0x3FC0u);
	float r = bf16_to_f32(b);
	ASSERT_NEAR(r, 1.5f, 1e-6f);

	/*
	 * Verify round-to-nearest-even: pick a value whose 17th bit is 1
	 * and lower bits create a tie, forcing even rounding.
	 *
	 * Use 1.0f + 2^-7 (0x3F810000):
	 *   f32:  0 01111111 00000010 00000000000000
	 *   bf16 would truncate to 0x3F81 (odd), but bit[15] = 0 and
	 *   all sticky bits = 0, so it is a halfway case → round to even
	 *   → 0x3F82? No. Let us work carefully.
	 *
	 * 0x3F810000: bits[15..0] = 0x0000, round_bit (bit15) = 0.
	 * No rounding needed → 0x3F81. Not a tie.
	 *
	 * Use a value where bits[15..0] = 0x8000 (exact halfway):
	 * 0x3F818000 = 1.0 + 2^-7 + 2^-24 (approx).
	 * upper16 = 0x3F81 (odd), round_bit = 1, sticky = 0 → tie.
	 * RNE: round up to even → 0x3F82.
	 *
	 * Construct that bit pattern directly.
	 */
	uint32_t u_tie = 0x3F818000u;
	float    f_tie;
	memcpy(&f_tie, &u_tie, sizeof(f_tie));
	uint16_t b_tie = f32_to_bf16(f_tie);
	/* 0x3F81 is odd, 0x3F82 is even → expect 0x3F82 */
	ASSERT_EQ(b_tie, (uint16_t)0x3F82u);

	/*
	 * Confirm the "round-down on tie to even" case:
	 * 0x3F808000 → upper16 = 0x3F80 (even), round_bit = 1,
	 * sticky = 0 → tie, already even → stays 0x3F80.
	 */
	uint32_t u_even = 0x3F808000u;
	float    f_even;
	memcpy(&f_even, &u_even, sizeof(f_even));
	uint16_t b_even = f32_to_bf16(f_even);
	ASSERT_EQ(b_even, (uint16_t)0x3F80u);
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

int main(void)
{
	test_fp16_round_trip();
	test_fp16_special_values();
	test_bf16_round_trip();
	test_bf16_special_values();
	test_bf16_round_to_nearest_even();

	TEST_REPORT();
}
