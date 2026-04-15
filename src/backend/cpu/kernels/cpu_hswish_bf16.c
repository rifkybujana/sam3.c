/*
 * src/backend/cpu/kernels/cpu_hswish_bf16.c - BF16 Hard Swish activation kernel
 *
 * Element-wise Hard Swish for bf16 tensors: y = x * clamp(x+3, 0, 6) / 6.
 * BF16 has no native arithmetic, so all computation is done in f32.
 * NEON path uses 4-wide float32x4_t with bf16<->f32 conversions from
 * core/half.h. Fully vectorizable — no transcendentals.
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
#include "core/half.h"
#include "core/tensor.h"
#include "util/log.h"
#include "util/threadpool.h"

#include <math.h>

/* --- NEON bf16 path (f32 arithmetic, 4-wide) --- */

#if SAM3_HAS_NEON

static void hswish_bf16_neon(const uint16_t *in, uint16_t *out,
			     int start, int end)
{
	float32x4_t v3 = vdupq_n_f32(3.0f);
	float32x4_t v6 = vdupq_n_f32(6.0f);
	float32x4_t v0 = vdupq_n_f32(0.0f);
	float32x4_t vinv6 = vdupq_n_f32(1.0f / 6.0f);
	int i = start;

	for (; i + 4 <= end; i += 4) {
		float32x4_t vx = bf16x4_to_f32x4(in + i);
		float32x4_t vc = vminq_f32(
			vmaxq_f32(vaddq_f32(vx, v3), v0), v6);
		float32x4_t result = vmulq_f32(
			vmulq_f32(vx, vc), vinv6);
		f32x4_to_bf16x4(out + i, result);
	}
	/* Scalar tail */
	for (; i < end; i++) {
		float v = bf16_to_f32(in[i]);
		out[i] = f32_to_bf16(
			v * fminf(fmaxf(v + 3.0f, 0.0f), 6.0f)
			* (1.0f / 6.0f));
	}
}

#else /* !SAM3_HAS_NEON */

static void hswish_bf16_scalar(const uint16_t *in, uint16_t *out,
			       int start, int end)
{
	for (int i = start; i < end; i++) {
		float v = bf16_to_f32(in[i]);
		out[i] = f32_to_bf16(
			v * fminf(fmaxf(v + 3.0f, 0.0f), 6.0f)
			* (1.0f / 6.0f));
	}
}

#endif /* SAM3_HAS_NEON */

/* --- Parallel dispatch --- */

struct hswish_par_ctx_bf16 {
	const uint16_t *in;
	uint16_t       *out;
	int             n;
};

static void hswish_bf16_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct hswish_par_ctx_bf16 *ctx = (struct hswish_par_ctx_bf16 *)arg;
	int chunk = ctx->n / n_tasks;
	int start = task_id * chunk;
	int end = (task_id == n_tasks - 1) ? ctx->n : start + chunk;

	if (start >= end)
		return;

#if SAM3_HAS_NEON
	hswish_bf16_neon(ctx->in, ctx->out, start, end);
#else
	hswish_bf16_scalar(ctx->in, ctx->out, start, end);
#endif
}

/*
 * cpu_kernel_hswish_bf16 - Element-wise Hard Swish for bf16 tensors.
 *
 * @node: Node with n_inputs>=1, input SAM3_DTYPE_BF16.
 * @pool: Thread pool for parallel execution.
 *
 * Returns SAM3_OK on success, SAM3_EINVAL on bad inputs.
 */
enum sam3_error cpu_kernel_hswish_bf16(const struct sam3_node *node,
				       struct sam3_threadpool *pool)
{
	if (!node->inputs[0] || !node->output) {
		sam3_log_error("hswish_bf16: NULL tensor");
		return SAM3_EINVAL;
	}

	if (node->inputs[0]->dtype != SAM3_DTYPE_BF16) {
		sam3_log_error("hswish_bf16: unsupported dtype");
		return SAM3_EINVAL;
	}

	struct hswish_par_ctx_bf16 ctx = {
		.in  = (const uint16_t *)node->inputs[0]->data,
		.out = (uint16_t *)node->output->data,
		.n   = sam3_tensor_nelems(node->inputs[0]),
	};

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

	sam3_threadpool_parallel_for(pool, hswish_bf16_parallel_fn, &ctx,
				     n_tasks);

	return SAM3_OK;
}
