/*
 * src/backend/cpu/cpu_blas.c - BLAS (sgemm) wrapper implementation
 *
 * Forwards sam3_blas_sgemm to cblas_sgemm when SAM3_HAS_BLAS is set;
 * otherwise runs a naive scalar f32 matmul as a portable fallback.
 * The batched helper fans out across the batch dimension via the
 * threadpool; it sets BLAS local thread count to 1 inside the fan-out
 * (when supported) so BLAS threads don't oversubscribe with our pool.
 *
 * fp16 / bf16 helpers cast inputs to f32 in scratch, run sgemm, and
 * cast back. Threshold for routing through this path is exposed in
 * cpu_blas.h (SAM3_BLAS_F16_GEMM_THRESHOLD).
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */
#include "cpu_blas.h"
#include "cpu_backend.h"
#include "cpu_parallel.h"
#include "core/alloc.h"
#include "core/half.h"
#include "util/threadpool.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifdef SAM3_HAS_BLAS
# ifdef __APPLE__
#  include <Accelerate/Accelerate.h>
# else
#  include <cblas.h>
# endif
#endif

/* OpenBLAS exposes a per-thread thread-count hint we use during the
 * batched fan-out to avoid (pool_threads * blas_threads) oversubscription.
 * Accelerate / generic BLAS have no equivalent — silently skip.
 * The OpenBLAS prototypes come from <cblas.h>. */
#if defined(SAM3_HAS_BLAS) && !defined(__APPLE__) && \
	defined(OPENBLAS_VERSION)
# define SAM3_HAS_OPENBLAS_LOCAL_THREADS 1
#endif

static inline void sam3_blas_local_threads(int n)
{
#ifdef SAM3_HAS_OPENBLAS_LOCAL_THREADS
	(void)openblas_set_num_threads_local(n);
#else
	(void)n;
#endif
}

void sam3_blas_sgemm(bool trans_a, bool trans_b,
		     int M, int N, int K,
		     float alpha,
		     const float *A, int lda,
		     const float *B, int ldb,
		     float beta,
		     float *C, int ldc)
{
#ifdef SAM3_HAS_BLAS
	cblas_sgemm(CblasRowMajor,
		    trans_a ? CblasTrans : CblasNoTrans,
		    trans_b ? CblasTrans : CblasNoTrans,
		    M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	/* Naive scalar fallback. Supports the four trans combinations
	 * needed by sdpa (notrans/trans, etc.). */
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
			C[i * ldc + j] = alpha * acc + beta * C[i * ldc + j];
		}
	}
#endif
}

struct sam3_blas_batched_ctx {
	bool trans_a, trans_b;
	int  M, N, K;
	int  lda, ldb, ldc;
	float alpha, beta;
	const float *A;
	const float *B;
	float       *C;
	ptrdiff_t sa, sb, sc;
};

static void sam3_blas_batched_chunk(size_t begin, size_t end, void *ctx)
{
	const struct sam3_blas_batched_ctx *c = ctx;
	sam3_blas_local_threads(1);
	for (size_t i = begin; i < end; ++i) {
		sam3_blas_sgemm(c->trans_a, c->trans_b,
				c->M, c->N, c->K,
				c->alpha,
				c->A + (ptrdiff_t)i * c->sa, c->lda,
				c->B + (ptrdiff_t)i * c->sb, c->ldb,
				c->beta,
				c->C + (ptrdiff_t)i * c->sc, c->ldc);
	}
}

void sam3_blas_sgemm_batched(struct sam3_threadpool *pool,
			     int batch,
			     bool trans_a, bool trans_b,
			     int M, int N, int K,
			     float alpha,
			     const float *A, int lda, ptrdiff_t stride_a,
			     const float *B, int ldb, ptrdiff_t stride_b,
			     float beta,
			     float *C, int ldc, ptrdiff_t stride_c)
{
	if (batch <= 0)
		return;

	struct sam3_blas_batched_ctx ctx = {
		.trans_a = trans_a, .trans_b = trans_b,
		.M = M, .N = N, .K = K,
		.lda = lda, .ldb = ldb, .ldc = ldc,
		.alpha = alpha, .beta = beta,
		.A = A, .B = B, .C = C,
		.sa = stride_a, .sb = stride_b, .sc = stride_c,
	};

	if (batch == 1 || !pool) {
		/* Single GEMM: let BLAS use all its threads. */
		sam3_blas_sgemm(trans_a, trans_b, M, N, K, alpha,
				A, lda, B, ldb, beta, C, ldc);
		return;
	}

	/* Fan out across batch; each task does its slice serially. */
	sam3_parallel_for_range(pool, (size_t)batch, 1,
				sam3_blas_batched_chunk, &ctx);
	sam3_blas_local_threads(0); /* restore default thread policy */
}

static void f16_to_f32_n(const uint16_t *src, float *dst, size_t n)
{
	for (size_t i = 0; i < n; ++i)
		dst[i] = fp16_to_f32(src[i]);
}

static void f32_to_f16_n(const float *src, uint16_t *dst, size_t n)
{
	for (size_t i = 0; i < n; ++i)
		dst[i] = f32_to_fp16(src[i]);
}

static void bf16_to_f32_n(const uint16_t *src, float *dst, size_t n)
{
	for (size_t i = 0; i < n; ++i)
		dst[i] = bf16_to_f32(src[i]);
}

static void f32_to_bf16_n(const float *src, uint16_t *dst, size_t n)
{
	for (size_t i = 0; i < n; ++i)
		dst[i] = f32_to_bf16(src[i]);
}

int sam3_blas_gemm_f16(struct sam3_threadpool *pool,
		       bool trans_a, bool trans_b,
		       int M, int N, int K,
		       const uint16_t *A_f16, const uint16_t *B_f16,
		       uint16_t *C_f16)
{
	(void)pool; /* reserved for future parallel cast */
	size_t nA = (size_t)M * (size_t)K;
	size_t nB = (size_t)K * (size_t)N;
	size_t nC = (size_t)M * (size_t)N;
	float *Af = malloc(nA * sizeof(float));
	float *Bf = malloc(nB * sizeof(float));
	float *Cf = malloc(nC * sizeof(float));
	if (!Af || !Bf || !Cf) {
		free(Af); free(Bf); free(Cf);
		return -1;
	}

	f16_to_f32_n(A_f16, Af, nA);
	f16_to_f32_n(B_f16, Bf, nB);
	int lda = trans_a ? M : K;
	int ldb = trans_b ? K : N;
	sam3_blas_sgemm(trans_a, trans_b, M, N, K,
			1.0f, Af, lda, Bf, ldb, 0.0f, Cf, N);
	f32_to_f16_n(Cf, C_f16, nC);

	free(Af); free(Bf); free(Cf);
	return 0;
}

int sam3_blas_gemm_bf16(struct sam3_threadpool *pool,
			bool trans_a, bool trans_b,
			int M, int N, int K,
			const uint16_t *A_bf16, const uint16_t *B_bf16,
			uint16_t *C_bf16)
{
	(void)pool;
	size_t nA = (size_t)M * (size_t)K;
	size_t nB = (size_t)K * (size_t)N;
	size_t nC = (size_t)M * (size_t)N;
	float *Af = malloc(nA * sizeof(float));
	float *Bf = malloc(nB * sizeof(float));
	float *Cf = malloc(nC * sizeof(float));
	if (!Af || !Bf || !Cf) {
		free(Af); free(Bf); free(Cf);
		return -1;
	}

	bf16_to_f32_n(A_bf16, Af, nA);
	bf16_to_f32_n(B_bf16, Bf, nB);
	int lda = trans_a ? M : K;
	int ldb = trans_b ? K : N;
	sam3_blas_sgemm(trans_a, trans_b, M, N, K,
			1.0f, Af, lda, Bf, ldb, 0.0f, Cf, N);
	f32_to_bf16_n(Cf, C_bf16, nC);

	free(Af); free(Bf); free(Cf);
	return 0;
}
