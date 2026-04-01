/*
 * src/backend/cpu/kernels/cpu_matmul_f16.c - FP16 tiled matrix multiply
 *
 * Matrix multiplication C = A @ B for fp16 tensors with 8x8x64 tiling.
 * Uses native NEON float16x8_t with vfmaq_f16 on ARMv8.2-A+, with
 * scalar fallback via f32 conversion on other platforms.
 *
 * Key types:  sam3_node, sam3_tensor
 * Depends on: cpu_kernels.h, cpu_simd_f16.h, core/half.h, core/tensor.h
 * Used by:    cpu_dispatch.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "cpu_kernels.h"
#include "cpu_simd_f16.h"
#include "core/half.h"
#include "core/tensor.h"
#include "util/log.h"
#include "util/threadpool.h"

#include <string.h>

#define TILE_M 8
#define TILE_N 8
#define TILE_K 64

/* --- NEON fp16 path --- */

#if SAM3_HAS_NEON_FP16

static void matmul_f16_neon(const _Float16 *a, const _Float16 *b,
			    _Float16 *c, int M, int K, int N,
			    int m_start, int m_end)
{
	memset(c + (size_t)m_start * N, 0,
	       (size_t)(m_end - m_start) * N * sizeof(_Float16));

	for (int i0 = m_start; i0 < m_end; i0 += TILE_M) {
		int imax = (i0 + TILE_M < m_end) ? i0 + TILE_M : m_end;
		for (int j0 = 0; j0 < N; j0 += TILE_N) {
			int jmax = (j0 + TILE_N < N) ? j0 + TILE_N : N;
			for (int k0 = 0; k0 < K; k0 += TILE_K) {
				int kmax = (k0 + TILE_K < K) ? k0 + TILE_K : K;
				for (int i = i0; i < imax; i++) {
					for (int k = k0; k < kmax; k++) {
						float16x8_t va = vdupq_n_f16(
							a[i * K + k]);
						int j = j0;
						for (; j + 8 <= jmax; j += 8) {
							float16x8_t vc = vld1q_f16(
								(const __fp16 *)(c + i * N + j));
							float16x8_t vb = vld1q_f16(
								(const __fp16 *)(b + k * N + j));
							vst1q_f16((__fp16 *)(c + i * N + j),
								  vfmaq_f16(vc, va, vb));
						}
						_Float16 aik = a[i * K + k];
						for (; j < jmax; j++)
							c[i * N + j] += aik * b[k * N + j];
					}
				}
			}
		}
	}
}

#else /* !SAM3_HAS_NEON_FP16 */

/* --- Scalar fallback path --- */

static void matmul_f16_scalar(const uint16_t *a, const uint16_t *b,
			      uint16_t *c, int M, int K, int N,
			      int m_start, int m_end)
{
	/* Zero output rows — uint16_t zero is fp16 +0.0 */
	memset(c + (size_t)m_start * N, 0,
	       (size_t)(m_end - m_start) * N * sizeof(uint16_t));

	for (int i0 = m_start; i0 < m_end; i0 += TILE_M) {
		int imax = (i0 + TILE_M < m_end) ? i0 + TILE_M : m_end;
		for (int j0 = 0; j0 < N; j0 += TILE_N) {
			int jmax = (j0 + TILE_N < N) ? j0 + TILE_N : N;
			for (int k0 = 0; k0 < K; k0 += TILE_K) {
				int kmax = (k0 + TILE_K < K) ? k0 + TILE_K : K;
				for (int i = i0; i < imax; i++) {
					for (int k = k0; k < kmax; k++) {
						float aik = fp16_to_f32(a[i * K + k]);
						for (int j = j0; j < jmax; j++) {
							float cur = fp16_to_f32(c[i * N + j]);
							float bkj = fp16_to_f32(b[k * N + j]);
							c[i * N + j] = f32_to_fp16(
								cur + aik * bkj);
						}
					}
				}
			}
		}
	}
}

#endif /* SAM3_HAS_NEON_FP16 */

/* --- Parallel dispatch context --- */

#if SAM3_HAS_NEON_FP16

struct matmul_par_ctx_f16 {
	const _Float16 *a;
	const _Float16 *b;
	_Float16       *c;
	int             M;
	int             K;
	int             N;
};

static void matmul_f16_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct matmul_par_ctx_f16 *ctx = (struct matmul_par_ctx_f16 *)arg;
	int chunk   = ctx->M / n_tasks;
	int m_start = task_id * chunk;
	int m_end   = (task_id == n_tasks - 1) ? ctx->M : m_start + chunk;

	if (m_start >= m_end)
		return;

	matmul_f16_neon(ctx->a, ctx->b, ctx->c,
			ctx->M, ctx->K, ctx->N, m_start, m_end);
}

#else /* !SAM3_HAS_NEON_FP16 */

struct matmul_par_ctx_f16 {
	const uint16_t *a;
	const uint16_t *b;
	uint16_t       *c;
	int             M;
	int             K;
	int             N;
};

static void matmul_f16_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct matmul_par_ctx_f16 *ctx = (struct matmul_par_ctx_f16 *)arg;
	int chunk   = ctx->M / n_tasks;
	int m_start = task_id * chunk;
	int m_end   = (task_id == n_tasks - 1) ? ctx->M : m_start + chunk;

	if (m_start >= m_end)
		return;

	matmul_f16_scalar(ctx->a, ctx->b, ctx->c,
			  ctx->M, ctx->K, ctx->N, m_start, m_end);
}

#endif /* SAM3_HAS_NEON_FP16 */

/*
 * cpu_kernel_matmul_f16 - FP16 matrix multiply: A [M,K] @ B [K,N] -> C [M,N].
 *
 * @node: Node with n_inputs>=2, all SAM3_DTYPE_F16.
 * @pool: Thread pool for parallel execution.
 *
 * Returns SAM3_OK on success, SAM3_EINVAL on bad inputs or dtype mismatch.
 */
enum sam3_error cpu_kernel_matmul_f16(const struct sam3_node *node,
				      struct sam3_threadpool *pool)
{
	if (node->n_inputs < 2 || !node->inputs[0] || !node->inputs[1] ||
	    !node->output) {
		sam3_log_error("matmul_f16: NULL tensor");
		return SAM3_EINVAL;
	}

	struct sam3_tensor *a = node->inputs[0];
	struct sam3_tensor *b = node->inputs[1];
	struct sam3_tensor *c = node->output;

	if (a->dtype != SAM3_DTYPE_F16 || b->dtype != SAM3_DTYPE_F16) {
		sam3_log_error("matmul_f16: unsupported dtype");
		return SAM3_EINVAL;
	}

	/* A: [M, K], B: [K, N], C: [M, N] */
	if (a->n_dims < 1 || b->n_dims < 1) {
		sam3_log_error("matmul_f16: need at least 1D tensors");
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
		sam3_log_error("matmul_f16: K mismatch %d != %d", K_a, K_b);
		return SAM3_EINVAL;
	}

	if (sam3_tensor_nelems(c) != M * N) {
		sam3_log_error("matmul_f16: output size mismatch %d != %d",
			       sam3_tensor_nelems(c), M * N);
		return SAM3_EINVAL;
	}

#if SAM3_HAS_NEON_FP16
	struct matmul_par_ctx_f16 ctx = {
		.a = (const _Float16 *)a->data,
		.b = (const _Float16 *)b->data,
		.c = (_Float16 *)c->data,
		.M = M, .K = K_a, .N = N,
	};
#else
	struct matmul_par_ctx_f16 ctx = {
		.a = (const uint16_t *)a->data,
		.b = (const uint16_t *)b->data,
		.c = (uint16_t *)c->data,
		.M = M, .K = K_a, .N = N,
	};
#endif

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

	sam3_threadpool_parallel_for(pool, matmul_f16_parallel_fn, &ctx,
				     n_tasks);

	return SAM3_OK;
}
