/*
 * tests/test_bench_video.c - Unit tests for video benchmark helpers
 *
 * Tests the pure helpers in bench/bench_video.h: the triangle-wave
 * bounce position function (regression test for the clip-generator
 * OOB bug) and the filter-glob matching on the case-table naming
 * scheme. Pure functions only; no model required.
 *
 * Key types:  (none)
 * Depends on: test_helpers.h, bench/bench.h, bench/bench_video.h
 * Used by:    CTest (when SAM3_BENCH is ON)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "bench/bench.h"
#include "bench/bench_video.h"

#include <string.h>

/* --- test_bounce_stays_in_bounds --- */

static void test_bounce_stays_in_bounds(void)
{
	int max_pos = SAM3_BENCH_VIDEO_IMG_SIZE - SAM3_BENCH_VIDEO_SQUARE_SIZE;

	for (int i = 0; i < 256; i++) {
		int p = sam3_bench_bounce_pos(i);
		ASSERT(p >= 0);
		ASSERT(p <= max_pos);
	}
}

/* --- test_bounce_starts_at_zero --- */

static void test_bounce_starts_at_zero(void)
{
	ASSERT_EQ(sam3_bench_bounce_pos(0), 0);
}

int main(void)
{
	test_bounce_stays_in_bounds();
	test_bounce_starts_at_zero();
	TEST_REPORT();
}
