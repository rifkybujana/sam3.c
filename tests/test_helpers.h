/*
 * tests/test_helpers.h - Minimal test assertion macros
 *
 * Provides ASSERT, ASSERT_EQ, ASSERT_NEAR, ASSERT_NOT_NULL, and
 * ASSERT_TENSOR_CLOSE macros for unit tests. Each test file is a
 * standalone executable with a main() that calls test functions.
 * CTest discovers and runs them.
 *
 * ASSERT_TENSOR_CLOSE compares two F32 sam3_tensors element-wise using
 * a numpy-style tolerance: |actual - expected| <= atol + rtol*|expected|.
 * On mismatch, the first offending element is printed to stderr and the
 * test process exits with status 1 (matching the existing stderr/exit
 * pattern used by the other macros in this file). Neither tensor is
 * owned or freed by the macro: callers keep full responsibility for
 * the tensor storage they pass in (arena, stack, or heap).
 *
 * Key types:  (macros + one inline helper)
 * Depends on: <stdio.h>, <stdlib.h>, <math.h>, <stddef.h>,
 *             sam3/sam3_types.h, core/tensor.h
 * Used by:    all test_*.c files
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_TEST_HELPERS_H
#define SAM3_TEST_HELPERS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>

#include "sam3/sam3_types.h"
#include "core/tensor.h"

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

#define ASSERT_NEAR(a, b, eps) do {                                  \
	tests_run++;                                                    \
	if (fabs((double)(a) - (double)(b)) > (double)(eps)) {          \
		fprintf(stderr, "FAIL %s:%d: %s=%g != %s=%g (eps=%g)\n",    \
			__FILE__, __LINE__, #a, (double)(a),                    \
			#b, (double)(b), (double)(eps));                         \
		tests_failed++;                                             \
	}                                                               \
} while (0)

#define ASSERT_NOT_NULL(x) ASSERT((x) != NULL)

/*
 * assert_tensor_close_f32 - Tolerance-based F32 array comparison.
 *
 * @actual:   Array of size @n from the engine under test.
 * @expected: Reference array of size @n (e.g. loaded from a fixture).
 * @n:        Number of float elements in both arrays.
 * @rtol:     Relative tolerance (multiplied by |expected[k]|).
 * @atol:     Absolute tolerance (added to the relative component).
 * @where:    Context string printed on mismatch (typically the fixture
 *            path or a label identifying the call site).
 *
 * Element-wise check: |actual[k] - expected[k]| <= atol + rtol*|expected[k]|.
 * On the first mismatch, prints a single diagnostic line to stderr and
 * calls exit(1) — matching how other assertions in this header fail hard
 * rather than accumulate into tests_failed. Returns normally on success.
 *
 * marked static inline to keep the symbol local to each test translation
 * unit (the header defines tests_run / tests_failed the same way).
 */
static inline void
assert_tensor_close_f32(const float *actual, const float *expected,
			size_t n, float rtol, float atol,
			const char *where)
{
	for (size_t k = 0; k < n; k++) {
		float tol = atol + rtol * fabsf(expected[k]);
		float diff = fabsf(actual[k] - expected[k]);

		if (diff > tol) {
			fprintf(stderr,
				"%s: mismatch at %zu: "
				"actual=%g expected=%g tol=%g diff=%g\n",
				where ? where : "(tensor)", k,
				(double)actual[k], (double)expected[k],
				(double)tol, (double)diff);
			exit(1);
		}
	}
}

/*
 * ASSERT_TENSOR_CLOSE - Compare two F32 sam3_tensors element-wise.
 *
 * @actual:   struct sam3_tensor * produced by the code under test.
 * @expected: struct sam3_tensor * holding the reference values.
 * @rtol:     Relative tolerance (float).
 * @atol:     Absolute tolerance (float).
 *
 * Both tensors must be F32 with identical n_dims and dims[]. On any
 * shape, dtype, or value mismatch the macro prints to stderr and exits
 * the process with status 1. Neither tensor is freed.
 *
 * Call sites are expected to have loaded @expected themselves (e.g. via
 * the test-local load_fixture_tensor helper over weight_reader) and
 * wrapped the resulting buffer in a struct sam3_tensor. Keeping the
 * loader out of this header avoids pulling the SafeTensors reader into
 * every test translation unit.
 */
#define ASSERT_TENSOR_CLOSE(actual, expected, rtol, atol) do {         \
	const struct sam3_tensor *__act = (actual);                    \
	const struct sam3_tensor *__exp = (expected);                  \
	ASSERT_NOT_NULL(__act);                                        \
	ASSERT_NOT_NULL(__exp);                                        \
	if (__act->dtype != SAM3_DTYPE_F32 ||                          \
	    __exp->dtype != SAM3_DTYPE_F32) {                          \
		fprintf(stderr,                                        \
			"ASSERT_TENSOR_CLOSE %s:%d: F32 required "     \
			"(actual=%d expected=%d)\n",                   \
			__FILE__, __LINE__,                            \
			(int)__act->dtype, (int)__exp->dtype);         \
		exit(1);                                               \
	}                                                              \
	if (__act->n_dims != __exp->n_dims) {                          \
		fprintf(stderr,                                        \
			"ASSERT_TENSOR_CLOSE %s:%d: n_dims "           \
			"actual=%d expected=%d\n",                     \
			__FILE__, __LINE__,                            \
			__act->n_dims, __exp->n_dims);                 \
		exit(1);                                               \
	}                                                              \
	for (int __i = 0; __i < __act->n_dims; __i++) {                \
		if (__act->dims[__i] != __exp->dims[__i]) {            \
			fprintf(stderr,                                \
				"ASSERT_TENSOR_CLOSE %s:%d: dim %d "   \
				"actual=%d expected=%d\n",             \
				__FILE__, __LINE__, __i,               \
				__act->dims[__i],                      \
				__exp->dims[__i]);                     \
			exit(1);                                       \
		}                                                      \
	}                                                              \
	assert_tensor_close_f32((const float *)__act->data,            \
				(const float *)__exp->data,            \
				(size_t)sam3_tensor_nelems(__act),     \
				(float)(rtol), (float)(atol),          \
				#actual " vs " #expected);             \
} while (0)

#define TEST_REPORT() do {                                          \
	printf("%d tests, %d failures\n", tests_run, tests_failed);     \
	return tests_failed ? 1 : 0;                                    \
} while (0)

#endif /* SAM3_TEST_HELPERS_H */
