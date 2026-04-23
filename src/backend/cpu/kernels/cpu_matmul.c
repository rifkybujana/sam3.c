/*
 * src/backend/cpu/kernels/cpu_matmul.c - Tiled matrix multiply kernel
 *
 * Matrix multiplication C = A @ B with 8x8x64 tiling for M-series L1
 * cache. A is [M, K], B is [K, N], C is [M, N]. NEON path uses
 * vfmaq_f32 for fused multiply-add. Scalar path uses same tiling.
 *
 * Key types:  sam3_node, sam3_tensor
 * Depends on: cpu_kernels.h, cpu_simd.h, core/tensor.h, util/threadpool.h
 * Used by:    cpu_dispatch.c, cpu_conv2d.c (im2col reuse)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "cpu_kernels.h"
#include "cpu_simd.h"
#include "core/tensor.h"
#include "util/log.h"
#include "util/threadpool.h"

#include <string.h>

#ifdef SAM3_HAS_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#define TILE_M 8
#define TILE_N 8
#define TILE_K 64

/* --- Scalar path (when no SIMD available) --- */

#if !SAM3_HAS_NEON && !SAM3_HAS_AVX2

static void matmul_f32_scalar(const float *a, const float *b, float *c,
			      int M, int K, int N,
			      int m_start, int m_end)
{
	memset(c + (size_t)m_start * N, 0,
	       (size_t)(m_end - m_start) * N * sizeof(float));

	for (int i0 = m_start; i0 < m_end; i0 += TILE_M) {
		int imax = (i0 + TILE_M < m_end) ? i0 + TILE_M : m_end;
		for (int j0 = 0; j0 < N; j0 += TILE_N) {
			int jmax = (j0 + TILE_N < N) ? j0 + TILE_N : N;
			for (int k0 = 0; k0 < K; k0 += TILE_K) {
				int kmax = (k0 + TILE_K < K) ? k0 + TILE_K : K;
				for (int i = i0; i < imax; i++) {
					for (int k = k0; k < kmax; k++) {
						float aik = a[i * K + k];
						for (int j = j0; j < jmax; j++)
							c[i * N + j] += aik * b[k * N + j];
					}
				}
			}
		}
	}
}

#endif /* !SAM3_HAS_NEON && !SAM3_HAS_AVX2 */

/* --- NEON path --- */

#if SAM3_HAS_NEON

static void matmul_f32_neon(const float *a, const float *b, float *c,
			    int M, int K, int N,
			    int m_start, int m_end)
{
	memset(c + (size_t)m_start * N, 0,
	       (size_t)(m_end - m_start) * N * sizeof(float));

	for (int i0 = m_start; i0 < m_end; i0 += TILE_M) {
		int imax = (i0 + TILE_M < m_end) ? i0 + TILE_M : m_end;
		for (int j0 = 0; j0 < N; j0 += TILE_N) {
			int jmax = (j0 + TILE_N < N) ? j0 + TILE_N : N;
			for (int k0 = 0; k0 < K; k0 += TILE_K) {
				int kmax = (k0 + TILE_K < K) ? k0 + TILE_K : K;
				for (int i = i0; i < imax; i++) {
					for (int k = k0; k < kmax; k++) {
						float32x4_t va = vdupq_n_f32(a[i * K + k]);
						int j = j0;
						for (; j + 4 <= jmax; j += 4) {
							float32x4_t vc = vld1q_f32(c + i * N + j);
							float32x4_t vb = vld1q_f32(b + k * N + j);
							vst1q_f32(c + i * N + j,
								  vfmaq_f32(vc, va, vb));
						}
						float aik = a[i * K + k];
						for (; j < jmax; j++)
							c[i * N + j] += aik * b[k * N + j];
					}
				}
			}
		}
	}
}

#endif /* SAM3_HAS_NEON */

/* --- AVX2 path --- */

#if SAM3_HAS_AVX2

static void matmul_f32_avx2(const float *a, const float *b, float *c,
			    int M, int K, int N,
			    int m_start, int m_end)
{
	memset(c + (size_t)m_start * N, 0,
	       (size_t)(m_end - m_start) * N * sizeof(float));

	for (int i0 = m_start; i0 < m_end; i0 += TILE_M) {
		int imax = (i0 + TILE_M < m_end) ? i0 + TILE_M : m_end;
		for (int j0 = 0; j0 < N; j0 += TILE_N) {
			int jmax = (j0 + TILE_N < N) ? j0 + TILE_N : N;
			for (int k0 = 0; k0 < K; k0 += TILE_K) {
				int kmax = (k0 + TILE_K < K) ? k0 + TILE_K : K;
				for (int i = i0; i < imax; i++) {
					for (int k = k0; k < kmax; k++) {
						__m256 va = _mm256_set1_ps(a[i * K + k]);
						int j = j0;
						for (; j + 8 <= jmax; j += 8) {
							__m256 vc = _mm256_loadu_ps(c + i * N + j);
							__m256 vb = _mm256_loadu_ps(b + k * N + j);
							_mm256_storeu_ps(c + i * N + j,
								_mm256_fmadd_ps(va, vb, vc));
						}
						float aik = a[i * K + k];
						for (; j < jmax; j++)
							c[i * N + j] += aik * b[k * N + j];
					}
				}
			}
		}
	}
}

#endif /* SAM3_HAS_AVX2 */

/* --- Parallel dispatch context --- */

struct matmul_par_ctx {
	const float *a;
	const float *b;
	float       *c;
	int          M;
	int          K;
	int          N;
};

static void matmul_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct matmul_par_ctx *ctx = (struct matmul_par_ctx *)arg;
	int chunk = ctx->M / n_tasks;
	int m_start = task_id * chunk;
	int m_end = (task_id == n_tasks - 1) ? ctx->M : m_start + chunk;

	if (m_start >= m_end)
		return;

#if SAM3_HAS_NEON
	matmul_f32_neon(ctx->a, ctx->b, ctx->c,
			ctx->M, ctx->K, ctx->N, m_start, m_end);
#elif SAM3_HAS_AVX2
	matmul_f32_avx2(ctx->a, ctx->b, ctx->c,
			ctx->M, ctx->K, ctx->N, m_start, m_end);
#else
	matmul_f32_scalar(ctx->a, ctx->b, ctx->c,
			  ctx->M, ctx->K, ctx->N, m_start, m_end);
#endif
}

enum sam3_error cpu_kernel_matmul(const struct sam3_node *node,
				  struct sam3_threadpool *pool)
{
	if (node->n_inputs < 2 || !node->inputs[0] || !node->inputs[1] ||
	    !node->output) {
		sam3_log_error("matmul: NULL tensor");
		return SAM3_EINVAL;
	}

	struct sam3_tensor *a = node->inputs[0];
	struct sam3_tensor *b = node->inputs[1];
	struct sam3_tensor *c = node->output;

	if (a->dtype != SAM3_DTYPE_F32 || b->dtype != SAM3_DTYPE_F32) {
		sam3_log_error("matmul: unsupported dtype");
		return SAM3_EINVAL;
	}

	/* A: [..., M, K], B: [..., K, N], C: [..., M, N] */
	if (a->n_dims < 1 || b->n_dims < 1) {
		sam3_log_error("matmul: need at least 1D tensors");
		return SAM3_EINVAL;
	}

	int M, K_a, K_b, N;

	if (a->n_dims == 1) {
		M = 1;
		K_a = a->dims[0];
	} else {
		M = a->dims[a->n_dims - 2];
		K_a = a->dims[a->n_dims - 1];
	}

	if (b->n_dims == 1) {
		K_b = b->dims[0];
		N = 1;
	} else {
		K_b = b->dims[b->n_dims - 2];
		N = b->dims[b->n_dims - 1];
	}

	if (K_a != K_b) {
		sam3_log_error("matmul: K mismatch %d != %d", K_a, K_b);
		return SAM3_EINVAL;
	}

	/*
	 * Batched matmul: when A carries leading batch dims (A has rank >= 3),
	 * iterate the 2D GEMM over the flattened batch. Supports the common
	 * SAM3 pattern where B is either:
	 *   - rank-2 (broadcast to all batch slots), or
	 *   - same rank as A (per-slot weights, e.g. batched scorer prompt^T).
	 * Other broadcasting patterns are rejected.
	 */
	int batch = 1;
	int a_nb = a->n_dims - 2;
	int b_nb = b->n_dims - 2;

	if (a_nb > 0) {
		for (int i = 0; i < a_nb; i++) {
			if (a->dims[i] <= 0) {
				sam3_log_error("matmul: non-positive batch "
					       "dim %d = %d",
					       i, a->dims[i]);
				return SAM3_EINVAL;
			}
			batch *= a->dims[i];
		}
	}

	int b_batch = 1;
	if (b_nb > 0) {
		for (int i = 0; i < b_nb; i++)
			b_batch *= b->dims[i];
	}
	if (b_nb > 0 && b_batch != batch) {
		sam3_log_error("matmul: batch mismatch a=%d b=%d",
			       batch, b_batch);
		return SAM3_EINVAL;
	}

	if (sam3_tensor_nelems(c) != batch * M * N) {
		sam3_log_error("matmul: output size mismatch %d != %d",
			       sam3_tensor_nelems(c), batch * M * N);
		return SAM3_EINVAL;
	}

	size_t a_stride = (size_t)M * (size_t)K_a;
	size_t b_stride = (b_nb > 0) ? (size_t)K_b * (size_t)N : 0;
	size_t c_stride = (size_t)M * (size_t)N;

	for (int g = 0; g < batch; g++) {
		const float *a_ptr = (const float *)a->data +
				     (size_t)g * a_stride;
		const float *b_ptr = (const float *)b->data +
				     (size_t)g * b_stride;
		float *c_ptr = (float *)c->data + (size_t)g * c_stride;

#ifdef SAM3_HAS_BLAS
		(void)pool;
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			    M, N, K_a,
			    1.0f,
			    a_ptr, K_a,
			    b_ptr, N,
			    0.0f,
			    c_ptr, N);
#else
		struct matmul_par_ctx ctx = {
			.a = a_ptr,
			.b = b_ptr,
			.c = c_ptr,
			.M = M, .K = K_a, .N = N,
		};

		int n_tasks = sam3_threadpool_n_threads(pool);
		if (n_tasks < 1)
			n_tasks = 1;

		sam3_threadpool_parallel_for(pool, matmul_parallel_fn, &ctx,
					     n_tasks);
#endif
	}

	return SAM3_OK;
}
