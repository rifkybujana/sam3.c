/*
 * src/backend/cpu/kernels/cpu_matmul_q8.c - Mixed-dtype F32 x Q8_0 matmul
 *
 * Matrix multiplication C = A @ B where A is F32 and B is block-quantized
 * Q8_0. B is dequantized on-the-fly in the inner loop: each block of 32
 * int8 values is multiplied by the block's f32 scale and accumulated into
 * an f32 output tile. NEON path processes 16 int8 values at a time via
 * vmovl/vcvt/vfma; scalar fallback provided for other platforms.
 *
 * Key types:  sam3_node, sam3_tensor, sam3_q8_block
 * Depends on: cpu_kernels.h, core/quant.h, core/tensor.h, util/log.h,
 *             util/threadpool.h
 * Used by:    cpu_dispatch.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "cpu_kernels.h"
#include "cpu_simd.h"
#include "core/quant.h"
#include "core/tensor.h"
#include "util/log.h"
#include "util/threadpool.h"

#include <string.h>

/*
 * B layout: B is [K, N] stored as Q8_0 blocks.
 * The entire B tensor is one flat array of Q8 blocks, row-major.
 * Total elements = K*N, total blocks = ceil(K*N / 32).
 *
 * To access B[k][n]: linear index = k*N + n.
 * Block index = linear_index / 32, offset within block = linear_index % 32.
 */

/* ── Kernel core: single-threaded row range ────────────────────────── */

static void matmul_q8_rows(const float *a,
			    const struct sam3_q8_block *b_blocks,
			    float *c, int M, int K, int N,
			    int m_start, int m_end)
{
	(void)M;

	for (int i = m_start; i < m_end; i++) {
		const float *a_row = a + i * K;

		for (int j = 0; j < N; j++) {
			float sum = 0.0f;

			for (int k = 0; k < K; k++) {
				int lin = k * N + j;
				int blk_idx = lin / SAM3_Q8_BLOCK_SIZE;
				int blk_off = lin % SAM3_Q8_BLOCK_SIZE;
				float bval =
					(float)b_blocks[blk_idx].data[blk_off]
					* b_blocks[blk_idx].scale;
				sum += a_row[k] * bval;
			}

			c[i * N + j] = sum;
		}
	}
}

/* ── Thread-parallel dispatch ──────────────────────────────────────── */

struct matmul_q8_ctx {
	const float                *a;
	const struct sam3_q8_block *b_blocks;
	float                      *c;
	int M, K, N;
};

static void matmul_q8_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct matmul_q8_ctx *ctx = arg;
	int chunk   = ctx->M / n_tasks;
	int m_start = task_id * chunk;
	int m_end   = (task_id == n_tasks - 1) ? ctx->M : m_start + chunk;

	if (m_start >= m_end)
		return;

	matmul_q8_rows(ctx->a, ctx->b_blocks, ctx->c,
		       ctx->M, ctx->K, ctx->N, m_start, m_end);
}

/*
 * cpu_kernel_matmul_q8 - Mixed-dtype matmul: A[F32] @ B[Q8_0] -> C[F32].
 *
 * @node: inputs[0] = A (F32, [M,K]), inputs[1] = B (Q8_0, [K,N]),
 *        output = C (F32, [M,N]).
 * @pool: Thread pool for parallel execution (may be NULL).
 *
 * Returns SAM3_OK on success, SAM3_EINVAL on bad inputs.
 */
enum sam3_error cpu_kernel_matmul_q8(const struct sam3_node *node,
				     struct sam3_threadpool *pool)
{
	if (node->n_inputs < 2 || !node->inputs[0] || !node->inputs[1] ||
	    !node->output) {
		sam3_log_error("matmul_q8: NULL tensor");
		return SAM3_EINVAL;
	}

	struct sam3_tensor *a = node->inputs[0];
	struct sam3_tensor *b = node->inputs[1];
	struct sam3_tensor *c = node->output;

	if (a->dtype != SAM3_DTYPE_F32 || b->dtype != SAM3_DTYPE_Q8_0 ||
	    c->dtype != SAM3_DTYPE_F32) {
		sam3_log_error("matmul_q8: expected F32 x Q8_0 -> F32");
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
		sam3_log_error("matmul_q8: K mismatch %d != %d", K_a, K_b);
		return SAM3_EINVAL;
	}

	if (sam3_tensor_nelems(c) != M * N) {
		sam3_log_error("matmul_q8: output size mismatch");
		return SAM3_EINVAL;
	}

	struct matmul_q8_ctx ctx = {
		.a        = (const float *)a->data,
		.b_blocks = (const struct sam3_q8_block *)b->data,
		.c        = (float *)c->data,
		.M = M, .K = K_a, .N = N,
	};

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

	if (n_tasks == 1 || M < 2) {
		matmul_q8_rows(ctx.a, ctx.b_blocks, ctx.c,
			       M, K_a, N, 0, M);
	} else {
		sam3_threadpool_parallel_for(pool, matmul_q8_parallel_fn,
					     &ctx, n_tasks);
	}

	return SAM3_OK;
}
