/*
 * src/backend/cpu/kernels/cpu_gelu_f16.c - FP16 GELU activation kernel
 *
 * Fast GELU approximation for fp16 tensors using the tanh formula.
 * Uses neon_f16_gelu_approx from cpu_simd_f16.h on ARMv8.2-A+, with
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

#include <math.h>

/* --- NEON fp16 path --- */

#if SAM3_HAS_NEON_FP16

static void gelu_f16_neon(const _Float16 *in, _Float16 *out,
			  int start, int end)
{
	int i = start;

	for (; i + 8 <= end; i += 8) {
		float16x8_t x = vld1q_f16((const __fp16 *)(in + i));
		vst1q_f16((__fp16 *)(out + i), neon_f16_gelu_approx(x));
	}
	/* Scalar tail */
	for (; i < end; i++) {
		float x = (float)in[i];
		float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
		out[i] = (_Float16)(0.5f * x * (1.0f + tanhf(inner)));
	}
}

#else /* !SAM3_HAS_NEON_FP16 */

static void gelu_f16_scalar(const uint16_t *in, uint16_t *out,
			    int start, int end)
{
	for (int i = start; i < end; i++) {
		float x = fp16_to_f32(in[i]);
		float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
		out[i] = f32_to_fp16(0.5f * x * (1.0f + tanhf(inner)));
	}
}

#endif /* SAM3_HAS_NEON_FP16 */

/* --- Parallel dispatch --- */

#if SAM3_HAS_NEON_FP16

struct gelu_par_ctx_f16 {
	const _Float16 *in;
	_Float16       *out;
	int             n;
};

static void gelu_f16_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct gelu_par_ctx_f16 *ctx = (struct gelu_par_ctx_f16 *)arg;
	int chunk = ctx->n / n_tasks;
	int start = task_id * chunk;
	int end = (task_id == n_tasks - 1) ? ctx->n : start + chunk;

	if (start >= end)
		return;

	gelu_f16_neon(ctx->in, ctx->out, start, end);
}

#else /* !SAM3_HAS_NEON_FP16 */

struct gelu_par_ctx_f16 {
	const uint16_t *in;
	uint16_t       *out;
	int             n;
};

static void gelu_f16_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct gelu_par_ctx_f16 *ctx = (struct gelu_par_ctx_f16 *)arg;
	int chunk = ctx->n / n_tasks;
	int start = task_id * chunk;
	int end = (task_id == n_tasks - 1) ? ctx->n : start + chunk;

	if (start >= end)
		return;

	gelu_f16_scalar(ctx->in, ctx->out, start, end);
}

#endif /* SAM3_HAS_NEON_FP16 */

/*
 * cpu_kernel_gelu_f16 - Element-wise GELU for fp16 tensors.
 *
 * @node: Node with n_inputs>=1, input SAM3_DTYPE_F16.
 * @pool: Thread pool for parallel execution.
 *
 * Returns SAM3_OK on success, SAM3_EINVAL on bad inputs.
 */
enum sam3_error cpu_kernel_gelu_f16(const struct sam3_node *node,
				    struct sam3_threadpool *pool)
{
	if (!node->inputs[0] || !node->output) {
		sam3_log_error("gelu_f16: NULL tensor");
		return SAM3_EINVAL;
	}

	if (node->inputs[0]->dtype != SAM3_DTYPE_F16) {
		sam3_log_error("gelu_f16: unsupported dtype");
		return SAM3_EINVAL;
	}

#if SAM3_HAS_NEON_FP16
	struct gelu_par_ctx_f16 ctx = {
		.in  = (const _Float16 *)node->inputs[0]->data,
		.out = (_Float16 *)node->output->data,
		.n   = sam3_tensor_nelems(node->inputs[0]),
	};
#else
	struct gelu_par_ctx_f16 ctx = {
		.in  = (const uint16_t *)node->inputs[0]->data,
		.out = (uint16_t *)node->output->data,
		.n   = sam3_tensor_nelems(node->inputs[0]),
	};
#endif

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

	sam3_threadpool_parallel_for(pool, gelu_f16_parallel_fn, &ctx, n_tasks);

	return SAM3_OK;
}
