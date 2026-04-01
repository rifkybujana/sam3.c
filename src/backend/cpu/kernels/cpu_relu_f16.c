/*
 * src/backend/cpu/kernels/cpu_relu_f16.c - FP16 elementwise ReLU kernel
 *
 * Implements element-wise ReLU (max(0, x)) for fp16 tensors. Uses native
 * NEON fp16 arithmetic on ARMv8.2-A+, with scalar fallback via fp16_to_f32
 * and f32_to_fp16 conversion on other platforms.
 *
 * Key types:  sam3_node, sam3_tensor
 * Depends on: cpu_kernels.h, cpu_simd_f16.h, core/half.h, core/tensor.h
 * Used by:    cpu_dispatch.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "cpu_kernels.h"
#include "cpu_simd_f16.h"
#include "core/half.h"
#include "core/tensor.h"
#include "util/log.h"
#include "util/threadpool.h"

/* --- NEON fp16 path --- */

#if SAM3_HAS_NEON_FP16

static void relu_f16_neon(const _Float16 *in, _Float16 *out,
			  int start, int end)
{
	float16x8_t zero = neon_f16_zero();
	int i = start;

	for (; i + 8 <= end; i += 8) {
		float16x8_t v = vld1q_f16((const __fp16 *)(in + i));
		vst1q_f16((__fp16 *)(out + i), vmaxq_f16(v, zero));
	}
	for (; i < end; i++)
		out[i] = in[i] > (_Float16)0.0f ? in[i] : (_Float16)0.0f;
}

#else /* !SAM3_HAS_NEON_FP16 */

static void relu_f16_scalar(const uint16_t *in, uint16_t *out,
			    int start, int end)
{
	for (int i = start; i < end; i++) {
		float val = fp16_to_f32(in[i]);
		out[i] = (val > 0.0f) ? in[i] : f32_to_fp16(0.0f);
	}
}

#endif /* SAM3_HAS_NEON_FP16 */

/* --- Parallel dispatch --- */

#if SAM3_HAS_NEON_FP16

struct relu_par_ctx_f16 {
	const _Float16 *in;
	_Float16       *out;
	int             n;
};

static void relu_f16_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct relu_par_ctx_f16 *ctx = (struct relu_par_ctx_f16 *)arg;
	int chunk = ctx->n / n_tasks;
	int start = task_id * chunk;
	int end = (task_id == n_tasks - 1) ? ctx->n : start + chunk;

	if (start >= end)
		return;

	relu_f16_neon(ctx->in, ctx->out, start, end);
}

#else /* !SAM3_HAS_NEON_FP16 */

struct relu_par_ctx_f16 {
	const uint16_t *in;
	uint16_t       *out;
	int             n;
};

static void relu_f16_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct relu_par_ctx_f16 *ctx = (struct relu_par_ctx_f16 *)arg;
	int chunk = ctx->n / n_tasks;
	int start = task_id * chunk;
	int end = (task_id == n_tasks - 1) ? ctx->n : start + chunk;

	if (start >= end)
		return;

	relu_f16_scalar(ctx->in, ctx->out, start, end);
}

#endif /* SAM3_HAS_NEON_FP16 */

/*
 * cpu_kernel_relu_f16 - Element-wise ReLU for fp16 tensors.
 *
 * @node: Node with n_inputs>=1, input SAM3_DTYPE_F16.
 * @pool: Thread pool for parallel execution.
 *
 * Returns SAM3_OK on success, SAM3_EINVAL on bad inputs.
 */
enum sam3_error cpu_kernel_relu_f16(const struct sam3_node *node,
				    struct sam3_threadpool *pool)
{
	if (!node->inputs[0] || !node->output) {
		sam3_log_error("relu_f16: NULL tensor");
		return SAM3_EINVAL;
	}

	if (node->inputs[0]->dtype != SAM3_DTYPE_F16) {
		sam3_log_error("relu_f16: unsupported dtype");
		return SAM3_EINVAL;
	}

#if SAM3_HAS_NEON_FP16
	struct relu_par_ctx_f16 ctx = {
		.in  = (const _Float16 *)node->inputs[0]->data,
		.out = (_Float16 *)node->output->data,
		.n   = sam3_tensor_nelems(node->inputs[0]),
	};
#else
	struct relu_par_ctx_f16 ctx = {
		.in  = (const uint16_t *)node->inputs[0]->data,
		.out = (uint16_t *)node->output->data,
		.n   = sam3_tensor_nelems(node->inputs[0]),
	};
#endif

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

	sam3_threadpool_parallel_for(pool, relu_f16_parallel_fn, &ctx, n_tasks);

	return SAM3_OK;
}
