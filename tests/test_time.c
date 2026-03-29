/*
 * tests/test_time.c - Unit tests for nanosecond clock utility
 *
 * Tests that sam3_time_ns returns monotonically increasing values
 * and that elapsed time is non-negative.
 *
 * Key types:  (none)
 * Depends on: util/time.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "util/time.h"

static void test_time_ns_returns_nonzero(void)
{
	uint64_t t = sam3_time_ns();
	ASSERT(t > 0);
}

static void test_time_ns_monotonic(void)
{
	uint64_t t1 = sam3_time_ns();
	/* Burn some CPU cycles */
	volatile int x = 0;
	for (int i = 0; i < 10000; i++)
		x += i;
	(void)x;
	uint64_t t2 = sam3_time_ns();
	ASSERT(t2 >= t1);
}

static void test_time_elapsed_positive(void)
{
	uint64_t start = sam3_time_ns();
	volatile int x = 0;
	for (int i = 0; i < 100000; i++)
		x += i;
	(void)x;
	uint64_t end = sam3_time_ns();
	ASSERT(end - start > 0);
}

int main(void)
{
	test_time_ns_returns_nonzero();
	test_time_ns_monotonic();
	test_time_elapsed_positive();

	TEST_REPORT();
}
