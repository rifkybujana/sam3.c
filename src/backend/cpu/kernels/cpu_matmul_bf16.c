/*
 * src/backend/cpu/kernels/cpu_matmul_bf16.c - BF16 tiled matrix multiply
 *
 * Matrix multiplication C = A @ B for bf16 tensors. All arithmetic is
 * performed in f32 for precision; bf16 inputs are upcast via bf16_to_f32
 * and results are converted back via f32_to_bf16. NEON path uses 4-wide
 * float32x4_t with vfmaq_f32, with scalar fallback on other platforms.
 *
 * Key types:  sam3_node, sam3_tensor
 * Depends on: cpu_kernels.h, cpu_simd.h, core/half.h, core/tensor.h
 * Used by:    cpu_dispatch.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "cpu_kernels.h"
#include "cpu_simd.h"
#include "backend/cpu/cpu_blas.h"
#include "core/half.h"
#include "core/tensor.h"
#include "util/log.h"
#include "util/threadpool.h"

#include <string.h>

#define TILE_M 8
#define TILE_N 8
#define TILE_K 64

/* --- NEON bf16 path (f32 arithmetic, 4-wide) --- */

#if SAM3_HAS_NEON

static void matmul_bf16_neon(const uint16_t *a, const uint16_t *b,
			     uint16_t *c, int M, int K, int N,
			     int m_start, int m_end)
{
	for (int i0 = m_start; i0 < m_end; i0 += TILE_M) {
		int imax = (i0 + TILE_M < m_end) ? i0 + TILE_M : m_end;
		for (int j0 = 0; j0 < N; j0 += TILE_N) {
			int jmax = (j0 + TILE_N < N) ? j0 + TILE_N : N;
			float acc[TILE_M][TILE_N];
			memset(acc, 0, sizeof(acc));

			for (int k0 = 0; k0 < K; k0 += TILE_K) {
				int kmax = (k0 + TILE_K < K)
					? k0 + TILE_K : K;
				for (int i = i0; i < imax; i++) {
					int ai = i - i0;
					for (int k = k0; k < kmax; k++) {
						float aik = bf16_to_f32(
							a[i * K + k]);
						float32x4_t va =
							vdupq_n_f32(aik);
						int j = j0;
						for (; j + 4 <= jmax;
						     j += 4) {
							int aj = j - j0;
							float32x4_t vc =
								vld1q_f32(
								&acc[ai][aj]);
							float32x4_t vb =
								bf16x4_to_f32x4(
								b + k * N + j);
							vst1q_f32(
								&acc[ai][aj],
								vfmaq_f32(
								vc, va, vb));
						}
						for (; j < jmax; j++) {
							acc[ai][j - j0] +=
								aik *
								bf16_to_f32(
								b[k * N + j]);
						}
					}
				}
			}

			/* Convert accumulated f32 tile to bf16 output */
			for (int i = i0; i < imax; i++) {
				int ai = i - i0;
				int j = j0;
				for (; j + 4 <= jmax; j += 4) {
					f32x4_to_bf16x4(
						c + i * N + j,
						vld1q_f32(
						&acc[ai][j - j0]));
				}
				for (; j < jmax; j++) {
					c[i * N + j] = f32_to_bf16(
						acc[ai][j - j0]);
				}
			}
		}
	}
}

#else /* !SAM3_HAS_NEON */

/* --- Scalar fallback path --- */

static void matmul_bf16_scalar(const uint16_t *a, const uint16_t *b,
			       uint16_t *c, int M, int K, int N,
			       int m_start, int m_end)
{
	for (int i0 = m_start; i0 < m_end; i0 += TILE_M) {
		int imax = (i0 + TILE_M < m_end) ? i0 + TILE_M : m_end;
		for (int j0 = 0; j0 < N; j0 += TILE_N) {
			int jmax = (j0 + TILE_N < N) ? j0 + TILE_N : N;
			float acc[TILE_M][TILE_N];
			memset(acc, 0, sizeof(acc));

			for (int k0 = 0; k0 < K; k0 += TILE_K) {
				int kmax = (k0 + TILE_K < K)
					? k0 + TILE_K : K;
				for (int i = i0; i < imax; i++) {
					for (int k = k0; k < kmax; k++) {
						float aik = bf16_to_f32(
							a[i * K + k]);
						for (int j = j0; j < jmax;
						     j++) {
							acc[i - i0][j - j0]
								+= aik *
								bf16_to_f32(
								b[k * N + j]);
						}
					}
				}
			}

			/* Convert accumulated f32 tile to bf16 output */
			for (int i = i0; i < imax; i++) {
				for (int j = j0; j < jmax; j++) {
					c[i * N + j] = f32_to_bf16(
						acc[i - i0][j - j0]);
				}
			}
		}
	}
}

#endif /* SAM3_HAS_NEON */

/* --- Parallel dispatch context --- */

struct matmul_par_ctx_bf16 {
	const uint16_t *a;
	const uint16_t *b;
	uint16_t       *c;
	int             M;
	int             K;
	int             N;
};

static void matmul_bf16_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct matmul_par_ctx_bf16 *ctx = (struct matmul_par_ctx_bf16 *)arg;
	int chunk   = ctx->M / n_tasks;
	int m_start = task_id * chunk;
	int m_end   = (task_id == n_tasks - 1) ? ctx->M : m_start + chunk;

	if (m_start >= m_end)
		return;

#if SAM3_HAS_NEON
	matmul_bf16_neon(ctx->a, ctx->b, ctx->c,
			 ctx->M, ctx->K, ctx->N, m_start, m_end);
#else
	matmul_bf16_scalar(ctx->a, ctx->b, ctx->c,
			   ctx->M, ctx->K, ctx->N, m_start, m_end);
#endif
}

/*
 * cpu_kernel_matmul_bf16 - BF16 matrix multiply: A [M,K] @ B [K,N] -> C [M,N].
 *
 * @node: Node with n_inputs>=2, all SAM3_DTYPE_BF16.
 * @pool: Thread pool for parallel execution.
 *
 * Returns SAM3_OK on success, SAM3_EINVAL on bad inputs or dtype mismatch.
 */
enum sam3_error cpu_kernel_matmul_bf16(const struct sam3_node *node,
				       struct sam3_threadpool *pool)
{
	if (node->n_inputs < 2 || !node->inputs[0] || !node->inputs[1] ||
	    !node->output) {
		sam3_log_error("matmul_bf16: NULL tensor");
		return SAM3_EINVAL;
	}

	struct sam3_tensor *a = node->inputs[0];
	struct sam3_tensor *b = node->inputs[1];
	struct sam3_tensor *c = node->output;

	if (a->dtype != SAM3_DTYPE_BF16 || b->dtype != SAM3_DTYPE_BF16) {
		sam3_log_error("matmul_bf16: unsupported dtype");
		return SAM3_EINVAL;
	}

	/* A: [M, K], B: [K, N], C: [M, N] */
	if (a->n_dims < 1 || b->n_dims < 1) {
		sam3_log_error("matmul_bf16: need at least 1D tensors");
		return SAM3_EINVAL;
	}

	int M, K_a, K_b, N;

	if (a->n_dims == 1) {
		M   = 1;
		K_a = a->dims[0];
	} else {
		M   = a->dims[a->n_dims - 2];
		K_a = a->dims[a->n_dims - 1];
	}

	if (b->n_dims == 1) {
		K_b = b->dims[0];
		N   = 1;
	} else {
		K_b = b->dims[b->n_dims - 2];
		N   = b->dims[b->n_dims - 1];
	}

	if (K_a != K_b) {
		sam3_log_error("matmul_bf16: K mismatch %d != %d", K_a, K_b);
		return SAM3_EINVAL;
	}

	if (sam3_tensor_nelems(c) != M * N) {
		sam3_log_error("matmul_bf16: output size mismatch %d != %d",
			       sam3_tensor_nelems(c), M * N);
		return SAM3_EINVAL;
	}

#ifdef SAM3_HAS_BLAS
	if ((size_t)M * (size_t)N * (size_t)K_a >= SAM3_BLAS_F16_GEMM_THRESHOLD) {
		if (sam3_blas_gemm_bf16(pool, false, false, M, N, K_a,
					(const uint16_t *)a->data,
					(const uint16_t *)b->data,
					(uint16_t *)c->data) == 0)
			return SAM3_OK;
	}
#endif

	struct matmul_par_ctx_bf16 ctx = {
		.a = (const uint16_t *)a->data,
		.b = (const uint16_t *)b->data,
		.c = (uint16_t *)c->data,
		.M = M, .K = K_a, .N = N,
	};

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

	sam3_threadpool_parallel_for(pool, matmul_bf16_parallel_fn, &ctx,
				     n_tasks);

	return SAM3_OK;
}
