/*
 * tests/test_cpu_blas.c - Unit tests for cpu_blas.h / cpu_parallel.h
 *
 * Verifies:
 *   - sam3_blas_sgemm matches a naive matmul reference for a few
 *     non-square shapes and trans flag combinations.
 *   - sam3_blas_sgemm_batched produces identical results to a manual
 *     loop of single-batch sgemms.
 *   - sam3_parallel_for_range visits every index in [0, total) exactly
 *     once with no overlaps or gaps, for a few (total, grain) cases.
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "test_helpers.h"

#include "backend/cpu/cpu_blas.h"
#include "backend/cpu/cpu_parallel.h"
#include "util/threadpool.h"

/* BLAS does blocked summation, naive reference does sequential.
 * Use a relative+absolute tolerance to avoid spurious failures from
 * FP rounding-order drift on large accumulations. */
#define ASSERT_NEAR_REL(a, b)                                          \
	do {                                                           \
		float _av = (a), _bv = (b);                            \
		float _d = _av > _bv ? _av - _bv : _bv - _av;          \
		float _m = _bv < 0 ? -_bv : _bv;                       \
		float _tol = 1e-4f + 1e-5f * _m;                       \
		ASSERT(_d <= _tol);                                    \
	} while (0)

/* --- Naive reference matmul (row-major). --- */
static void naive_sgemm(int trans_a, int trans_b,
			int M, int N, int K,
			const float *A, int lda,
			const float *B, int ldb,
			float *C, int ldc)
{
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			float acc = 0.0f;
			for (int k = 0; k < K; ++k) {
				float a = trans_a ? A[k * lda + i]
						  : A[i * lda + k];
				float b = trans_b ? B[j * ldb + k]
						  : B[k * ldb + j];
				acc += a * b;
			}
			C[i * ldc + j] = acc;
		}
	}
}

static void fill_seq(float *p, int n, float scale, float offset)
{
	for (int i = 0; i < n; ++i)
		p[i] = (float)i * scale + offset;
}

static void test_sgemm_notrans(void)
{
	enum { M = 7, N = 11, K = 13 };
	float A[M * K], B[K * N], C[M * N], R[M * N];
	fill_seq(A, M * K, 0.1f, 0.5f);
	fill_seq(B, K * N, 0.2f, -0.3f);

	memset(C, 0, sizeof(C));
	memset(R, 0, sizeof(R));
	sam3_blas_sgemm(false, false, M, N, K, 1.0f,
			A, K, B, N, 0.0f, C, N);
	naive_sgemm(0, 0, M, N, K, A, K, B, N, R, N);

	for (int i = 0; i < M * N; ++i)
		ASSERT_NEAR_REL(C[i], R[i]);
}

static void test_sgemm_trans_b(void)
{
	enum { M = 5, N = 9, K = 6 };
	float A[M * K], B[N * K], C[M * N], R[M * N];
	fill_seq(A, M * K, 0.05f, 0.1f);
	fill_seq(B, N * K, 0.07f, -0.2f);

	memset(C, 0, sizeof(C));
	memset(R, 0, sizeof(R));
	sam3_blas_sgemm(false, true, M, N, K, 1.0f,
			A, K, B, K, 0.0f, C, N);
	naive_sgemm(0, 1, M, N, K, A, K, B, K, R, N);

	for (int i = 0; i < M * N; ++i)
		ASSERT_NEAR_REL(C[i], R[i]);
}

static void test_sgemm_alpha_beta(void)
{
	enum { M = 4, N = 4, K = 4 };
	float A[M * K], B[K * N], C[M * N], R[M * N];
	fill_seq(A, M * K, 0.5f, 1.0f);
	fill_seq(B, K * N, -0.5f, 2.0f);
	for (int i = 0; i < M * N; ++i) {
		C[i] = 1.0f;
		R[i] = 1.0f;
	}
	sam3_blas_sgemm(false, false, M, N, K,
			2.0f, A, K, B, N, 3.0f, C, N);

	float tmp[M * N];
	memset(tmp, 0, sizeof(tmp));
	naive_sgemm(0, 0, M, N, K, A, K, B, N, tmp, N);
	for (int i = 0; i < M * N; ++i)
		R[i] = 2.0f * tmp[i] + 3.0f * R[i];

	for (int i = 0; i < M * N; ++i)
		ASSERT_NEAR_REL(C[i], R[i]);
}

static void test_sgemm_batched(void)
{
	enum { B_ = 4, M = 8, N = 8, K = 8 };
	float A[B_ * M * K], B[B_ * K * N], C[B_ * M * N], R[B_ * M * N];
	fill_seq(A, B_ * M * K, 0.01f, 0.0f);
	fill_seq(B, B_ * K * N, 0.02f, 0.5f);

	memset(C, 0, sizeof(C));
	memset(R, 0, sizeof(R));

	struct sam3_threadpool *pool = sam3_threadpool_create(4);
	sam3_blas_sgemm_batched(pool, B_, false, false, M, N, K,
				1.0f,
				A, K, (ptrdiff_t)(M * K),
				B, N, (ptrdiff_t)(K * N),
				0.0f,
				C, N, (ptrdiff_t)(M * N));

	for (int b = 0; b < B_; ++b) {
		naive_sgemm(0, 0, M, N, K,
			    A + b * M * K, K,
			    B + b * K * N, N,
			    R + b * M * N, N);
	}
	for (int i = 0; i < B_ * M * N; ++i)
		ASSERT_NEAR_REL(C[i], R[i]);

	sam3_threadpool_free(pool);
}

/* --- parallel_for_range coverage --- */

struct cover_ctx {
	int *hits;
	size_t total;
};

static void cover_fn(size_t begin, size_t end, void *ctx)
{
	struct cover_ctx *c = ctx;
	for (size_t i = begin; i < end; ++i)
		__atomic_add_fetch(&c->hits[i], 1, __ATOMIC_RELAXED);
}

static void run_cover(struct sam3_threadpool *pool, size_t total, size_t grain)
{
	int *hits = calloc(total, sizeof(int));
	struct cover_ctx ctx = { hits, total };
	sam3_parallel_for_range(pool, total, grain, cover_fn, &ctx);
	for (size_t i = 0; i < total; ++i)
		ASSERT_EQ(hits[i], 1);
	free(hits);
}

static void test_parallel_for_range_basic(void)
{
	struct sam3_threadpool *pool = sam3_threadpool_create(4);
	run_cover(pool, 10000, 128);
	run_cover(pool, 1, 1);
	run_cover(pool, 17, 4);
	run_cover(pool, 1024, 1024); /* total < grain*2 -> serial */
	sam3_threadpool_free(pool);
}

static void test_parallel_for_range_serial(void)
{
	run_cover(NULL, 500, 16);
}

int main(void)
{
	test_sgemm_notrans();
	test_sgemm_trans_b();
	test_sgemm_alpha_beta();
	test_sgemm_batched();
	test_parallel_for_range_basic();
	test_parallel_for_range_serial();
	TEST_REPORT();
}
