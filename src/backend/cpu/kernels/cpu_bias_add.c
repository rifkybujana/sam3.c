/*
 * src/backend/cpu/kernels/cpu_bias_add.c - NHWC bias add kernel
 *
 * Adds a per-channel bias vector to every spatial position of an NHWC
 * tensor: out[n,h,w,c] = x[n,h,w,c] + bias[c]. Parallelises over
 * N*H*W rows with NEON 4-wide acceleration on the channel dimension.
 * Also provides a BF16 variant (bf16 load -> f32 add -> bf16 store).
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

/* ── F32 bias add ──────────────────────────────────────────────────── */

struct bias_add_par_ctx {
	const float *x;
	const float *bias;
	float       *out;
	int          C;
	int          total_rows; /* N * H * W */
};

static void bias_add_f32_fn(void *arg, int task_id, int n_tasks)
{
	struct bias_add_par_ctx *ctx = (struct bias_add_par_ctx *)arg;
	int chunk = ctx->total_rows / n_tasks;
	int start = task_id * chunk;
	int end   = (task_id == n_tasks - 1)
		    ? ctx->total_rows : start + chunk;
	int C = ctx->C;
	const float *bias = ctx->bias;

	for (int p = start; p < end; p++) {
		size_t off = (size_t)p * C;
		const float *xr = ctx->x + off;
		float *or_ = ctx->out + off;
#if SAM3_HAS_NEON
		int c = 0;
		for (; c + 4 <= C; c += 4) {
			float32x4_t vx = vld1q_f32(xr + c);
			float32x4_t vb = vld1q_f32(bias + c);
			vst1q_f32(or_ + c, vaddq_f32(vx, vb));
		}
		for (; c < C; c++)
			or_[c] = xr[c] + bias[c];
#else
		for (int c = 0; c < C; c++)
			or_[c] = xr[c] + bias[c];
#endif
	}
}

/*
 * cpu_kernel_bias_add - NHWC bias add (F32).
 *
 * inputs[0] = x[N,H,W,C], inputs[1] = bias[C], output = out[N,H,W,C].
 */
enum sam3_error cpu_kernel_bias_add(const struct sam3_node *node,
				    struct sam3_threadpool *pool)
{
	const struct sam3_tensor *x    = node->inputs[0];
	const struct sam3_tensor *bias = node->inputs[1];
	struct sam3_tensor *out = node->output;

	if (!x || !bias || !out) {
		sam3_log_error("bias_add: NULL tensor");
		return SAM3_EINVAL;
	}

	int N = x->dims[0];
	int H = x->dims[1];
	int W = x->dims[2];
	int C = x->dims[3];

	struct bias_add_par_ctx ctx = {
		.x          = (const float *)x->data,
		.bias       = (const float *)bias->data,
		.out        = (float *)out->data,
		.C          = C,
		.total_rows = N * H * W,
	};

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

	sam3_threadpool_parallel_for(pool, bias_add_f32_fn, &ctx, n_tasks);

	return SAM3_OK;
}

/* ── BF16 bias add ─────────────────────────────────────────────────── */

struct bias_add_par_ctx_bf16 {
	const uint16_t *x;
	const uint16_t *bias;
	uint16_t       *out;
	int              C;
	int              total_rows;
};

static void bias_add_bf16_fn(void *arg, int task_id, int n_tasks)
{
	struct bias_add_par_ctx_bf16 *ctx =
		(struct bias_add_par_ctx_bf16 *)arg;
	int chunk = ctx->total_rows / n_tasks;
	int start = task_id * chunk;
	int end   = (task_id == n_tasks - 1)
		    ? ctx->total_rows : start + chunk;
	int C = ctx->C;
	const uint16_t *bias = ctx->bias;

	for (int p = start; p < end; p++) {
		size_t off = (size_t)p * C;
		const uint16_t *xr = ctx->x + off;
		uint16_t *or_ = ctx->out + off;
#if SAM3_HAS_NEON
		int c = 0;
		for (; c + 4 <= C; c += 4) {
			float32x4_t vx = bf16x4_to_f32x4(xr + c);
			float32x4_t vb = bf16x4_to_f32x4(bias + c);
			f32x4_to_bf16x4(or_ + c,
					 vaddq_f32(vx, vb));
		}
		for (; c < C; c++)
			or_[c] = f32_to_bf16(
				bf16_to_f32(xr[c]) +
				bf16_to_f32(bias[c]));
#else
		for (int c = 0; c < C; c++)
			or_[c] = f32_to_bf16(
				bf16_to_f32(xr[c]) +
				bf16_to_f32(bias[c]));
#endif
	}
}

/*
 * cpu_kernel_bias_add_bf16 - NHWC bias add (BF16).
 *
 * inputs[0] = x[N,H,W,C], inputs[1] = bias[C], output = out[N,H,W,C].
 * BF16 -> F32 add -> BF16 store.
 */
enum sam3_error cpu_kernel_bias_add_bf16(const struct sam3_node *node,
					 struct sam3_threadpool *pool)
{
	const struct sam3_tensor *x    = node->inputs[0];
	const struct sam3_tensor *bias = node->inputs[1];
	struct sam3_tensor *out = node->output;

	if (!x || !bias || !out) {
		sam3_log_error("bias_add_bf16: NULL tensor");
		return SAM3_EINVAL;
	}

	int N = x->dims[0];
	int H = x->dims[1];
	int W = x->dims[2];
	int C = x->dims[3];

	struct bias_add_par_ctx_bf16 ctx = {
		.x          = (const uint16_t *)x->data,
		.bias       = (const uint16_t *)bias->data,
		.out        = (uint16_t *)out->data,
		.C          = C,
		.total_rows = N * H * W,
	};

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

	sam3_threadpool_parallel_for(pool, bias_add_bf16_fn, &ctx, n_tasks);

	return SAM3_OK;
}
