/*
 * tests/test_helpers.h - Minimal test assertion macros
 *
 * Provides ASSERT and ASSERT_EQ macros for unit tests. Each test file
 * is a standalone executable with a main() that calls test functions.
 * CTest discovers and runs them.
 *
 * Key types:  (macros only)
 * Depends on: <stdio.h>, <stdlib.h>
 * Used by:    all test_*.c files
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_TEST_HELPERS_H
#define SAM3_TEST_HELPERS_H

#include <stdio.h>
#include <stdlib.h>

static int tests_run = 0;
static int tests_failed = 0;

#define ASSERT(cond) do {                                           \
	tests_run++;                                                    \
	if (!(cond)) {                                                  \
		fprintf(stderr, "FAIL %s:%d: %s\n",                         \
			__FILE__, __LINE__, #cond);                             \
		tests_failed++;                                             \
	}                                                               \
} while (0)

#define ASSERT_EQ(a, b) do {                                        \
	tests_run++;                                                    \
	if ((a) != (b)) {                                               \
		fprintf(stderr, "FAIL %s:%d: %s != %s\n",                   \
			__FILE__, __LINE__, #a, #b);                            \
		tests_failed++;                                             \
	}                                                               \
} while (0)

#define TEST_REPORT() do {                                          \
	printf("%d tests, %d failures\n", tests_run, tests_failed);     \
	return tests_failed ? 1 : 0;                                    \
} while (0)

#endif /* SAM3_TEST_HELPERS_H */
