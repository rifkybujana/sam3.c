/*
 * src/backend/cpu/kernels/cpu_hswish.c - Hard Swish activation kernel
 *
 * Element-wise Hard Swish: y[i] = x[i] * clamp(x[i]+3, 0, 6) / 6.
 * This is a piecewise-linear approximation of Swish used by MobileNetV3
 * and EfficientViT.  Uses thread pool parallelism for large tensors.
 * Scalar path only; SIMD can be added when profiling shows this is a
 * bottleneck.
 *
 * Key types:  sam3_node, sam3_tensor
 * Depends on: cpu_kernels.h, core/tensor.h, util/threadpool.h
 * Used by:    cpu_dispatch.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "cpu_kernels.h"
#include "core/tensor.h"
#include "util/log.h"
#include "util/threadpool.h"

#include <math.h>

/* --- Scalar path --- */

static void hswish_f32_scalar(const float *in, float *out,
			      int start, int end)
{
	for (int i = start; i < end; i++) {
		float v = in[i];
		out[i] = v * fminf(fmaxf(v + 3.0f, 0.0f), 6.0f) * (1.0f / 6.0f);
	}
}

/* --- Parallel dispatch --- */

struct hswish_par_ctx {
	const float *in;
	float       *out;
	int          n;
};

static void hswish_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct hswish_par_ctx *ctx = (struct hswish_par_ctx *)arg;
	int chunk = ctx->n / n_tasks;
	int start = task_id * chunk;
	int end = (task_id == n_tasks - 1) ? ctx->n : start + chunk;

	if (start >= end)
		return;

	hswish_f32_scalar(ctx->in, ctx->out, start, end);
}

enum sam3_error cpu_kernel_hswish(const struct sam3_node *node,
				  struct sam3_threadpool *pool)
{
	if (!node->inputs[0] || !node->output) {
		sam3_log_error("hswish: NULL tensor");
		return SAM3_EINVAL;
	}

	if (node->inputs[0]->dtype != SAM3_DTYPE_F32) {
		sam3_log_error("hswish: unsupported dtype");
		return SAM3_EINVAL;
	}

	struct hswish_par_ctx ctx = {
		.in  = (const float *)node->inputs[0]->data,
		.out = (float *)node->output->data,
		.n   = sam3_tensor_nelems(node->inputs[0]),
	};

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

	sam3_threadpool_parallel_for(pool, hswish_parallel_fn, &ctx, n_tasks);

	return SAM3_OK;
}
