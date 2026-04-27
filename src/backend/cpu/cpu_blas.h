/*
 * src/backend/cpu/cpu_blas.h - BLAS (sgemm) wrapper for CPU kernels
 *
 * Thin layer over Accelerate / OpenBLAS / MKL exposing a single sgemm,
 * a batched sgemm fan-out via the threadpool, and convenience cast
 * helpers for fp16/bf16 inputs (cast to f32, sgemm, cast back). When
 * SAM3_HAS_BLAS is undefined, sam3_blas_sgemm falls back to a naive
 * f32 matmul so callers can use a single code path.
 *
 * Key types:  (none)
 * Depends on: backend/cpu/cpu_backend.h, util/threadpool.h
 * Used by:    cpu_matmul*.c, cpu_conv2d*.c, cpu_sdpa.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */
#ifndef SAM3_CPU_BLAS_H
#define SAM3_CPU_BLAS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct sam3_threadpool;
struct sam3_cpu_backend;

/*
 * Threshold (M*N*K) above which fp16/bf16 matmul should route through
 * BLAS via the cast-up-cast-down path. Below this, the existing
 * hand-rolled SIMD kernel wins because the cast overhead dominates.
 */
#define SAM3_BLAS_F16_GEMM_THRESHOLD ((size_t)64 * 64 * 64)

/* Row-major C = alpha * op(A) * op(B) + beta * C. */
void sam3_blas_sgemm(bool trans_a, bool trans_b,
		     int M, int N, int K,
		     float alpha,
		     const float *A, int lda,
		     const float *B, int ldb,
		     float beta,
		     float *C, int ldc);

/*
 * Batched sgemm. Performs `batch` independent (M,N,K) sgemms. When
 * batch>1 and pool!=NULL, fans out across the batch dimension via
 * sam3_parallel_for_range; sets BLAS local thread count to 1 inside
 * the fan-out to avoid oversubscription.
 */
void sam3_blas_sgemm_batched(struct sam3_threadpool *pool,
			     int batch,
			     bool trans_a, bool trans_b,
			     int M, int N, int K,
			     float alpha,
			     const float *A, int lda, ptrdiff_t stride_a,
			     const float *B, int ldb, ptrdiff_t stride_b,
			     float beta,
			     float *C, int ldc, ptrdiff_t stride_c);

/*
 * Cast-up-cast-down GEMM helpers for fp16/bf16. Inputs/outputs are
 * uint16_t bit patterns. The helpers allocate (and free) their own
 * f32 working buffers via malloc; alloc cost is negligible relative
 * to the underlying GEMM at sizes above SAM3_BLAS_F16_GEMM_THRESHOLD.
 * Returns 0 on success, -1 on allocation failure.
 */
int sam3_blas_gemm_f16(struct sam3_threadpool *pool,
		       bool trans_a, bool trans_b,
		       int M, int N, int K,
		       const uint16_t *A_f16, const uint16_t *B_f16,
		       uint16_t *C_f16);

int sam3_blas_gemm_bf16(struct sam3_threadpool *pool,
			bool trans_a, bool trans_b,
			int M, int N, int K,
			const uint16_t *A_bf16, const uint16_t *B_bf16,
			uint16_t *C_bf16);

#endif /* SAM3_CPU_BLAS_H */
