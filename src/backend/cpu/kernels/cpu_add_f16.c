/*
 * src/backend/cpu/kernels/cpu_add_f16.c - FP16 elementwise add kernel
 *
 * Implements element-wise addition for fp16 tensors with broadcasting.
 * Uses native NEON fp16 arithmetic on ARMv8.2-A+, with scalar fallback
 * via fp16_to_f32/f32_to_fp16 conversion on other platforms.
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

/*
 * check_broadcast_f16 - Check if b broadcasts onto a: a is [M, N], b is [N].
 *
 * Returns the inner dimension N, or 0 if shapes match exactly,
 * or -1 on error. Does not check dtype.
 */
static int check_broadcast_f16(const struct sam3_tensor *a,
				const struct sam3_tensor *b)
{
	int na = sam3_tensor_nelems(a);
	int nb = sam3_tensor_nelems(b);

	if (na == nb)
		return 0;

	/* b is [N], a's last dim is N */
	if (b->n_dims == 1 && a->dims[a->n_dims - 1] == b->dims[0])
		return b->dims[0];

	return -1;
}

/* --- NEON fp16 path --- */

#if SAM3_HAS_NEON_FP16

static void add_f16_neon(const _Float16 *a, const _Float16 *b, _Float16 *out,
			 int broadcast_n, int start, int end)
{
	if (broadcast_n <= 0) {
		int i = start;
		for (; i + 8 <= end; i += 8) {
			float16x8_t va = vld1q_f16((const __fp16 *)(a + i));
			float16x8_t vb = vld1q_f16((const __fp16 *)(b + i));
			vst1q_f16((__fp16 *)(out + i), vaddq_f16(va, vb));
		}
		for (; i < end; i++)
			out[i] = a[i] + b[i];
	} else {
		for (int r = start; r < end; r++) {
			int base = r * broadcast_n;
			int j = 0;
			for (; j + 8 <= broadcast_n; j += 8) {
				float16x8_t va = vld1q_f16(
					(const __fp16 *)(a + base + j));
				float16x8_t vb = vld1q_f16(
					(const __fp16 *)(b + j));
				vst1q_f16((__fp16 *)(out + base + j),
					  vaddq_f16(va, vb));
			}
			for (; j < broadcast_n; j++)
				out[base + j] = a[base + j] + b[j];
		}
	}
}

#else /* !SAM3_HAS_NEON_FP16 */

static void add_f16_scalar(const uint16_t *a, const uint16_t *b,
			   uint16_t *out, int broadcast_n,
			   int start, int end)
{
	if (broadcast_n <= 0) {
		for (int i = start; i < end; i++) {
			out[i] = f32_to_fp16(
				fp16_to_f32(a[i]) + fp16_to_f32(b[i]));
		}
	} else {
		for (int i = start; i < end; i++) {
			out[i] = f32_to_fp16(
				fp16_to_f32(a[i]) +
				fp16_to_f32(b[i % broadcast_n]));
		}
	}
}

#endif /* SAM3_HAS_NEON_FP16 */

/* --- Parallel dispatch --- */

#if SAM3_HAS_NEON_FP16

struct binop_par_ctx_f16 {
	const _Float16 *a;
	const _Float16 *b;
	_Float16       *out;
	int             n;
	int             broadcast_n;
};

static void add_f16_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct binop_par_ctx_f16 *ctx = (struct binop_par_ctx_f16 *)arg;
	int total, start, end;

	if (ctx->broadcast_n <= 0) {
		total = ctx->n;
	} else {
		total = ctx->n / ctx->broadcast_n;
	}

	int chunk = total / n_tasks;
	start = task_id * chunk;
	end = (task_id == n_tasks - 1) ? total : start + chunk;

	if (start >= end)
		return;

	add_f16_neon(ctx->a, ctx->b, ctx->out,
		     ctx->broadcast_n, start, end);
}

#else /* !SAM3_HAS_NEON_FP16 */

struct binop_par_ctx_f16 {
	const uint16_t *a;
	const uint16_t *b;
	uint16_t       *out;
	int             n;
	int             broadcast_n;
};

static void add_f16_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct binop_par_ctx_f16 *ctx = (struct binop_par_ctx_f16 *)arg;
	int total, start, end;

	if (ctx->broadcast_n <= 0) {
		total = ctx->n;
	} else {
		total = ctx->n / ctx->broadcast_n;
	}

	int chunk = total / n_tasks;
	start = task_id * chunk;
	end = (task_id == n_tasks - 1) ? total : start + chunk;

	if (start >= end)
		return;

	add_f16_scalar(ctx->a, ctx->b, ctx->out,
		       ctx->broadcast_n, start, end);
}

#endif /* SAM3_HAS_NEON_FP16 */

/*
 * cpu_kernel_add_f16 - Element-wise add for fp16 tensors with broadcasting.
 *
 * @node: Node with n_inputs>=2, all SAM3_DTYPE_F16.
 * @pool: Thread pool for parallel execution.
 *
 * Returns SAM3_OK on success, SAM3_EINVAL on bad inputs.
 */
enum sam3_error cpu_kernel_add_f16(const struct sam3_node *node,
				   struct sam3_threadpool *pool)
{
	if (node->n_inputs < 2 || !node->inputs[0] || !node->inputs[1] ||
	    !node->output) {
		sam3_log_error("add_f16: NULL tensor");
		return SAM3_EINVAL;
	}

	if (node->inputs[0]->dtype != SAM3_DTYPE_F16 ||
	    node->inputs[1]->dtype != SAM3_DTYPE_F16) {
		sam3_log_error("add_f16: unsupported dtype");
		return SAM3_EINVAL;
	}

	int bc = check_broadcast_f16(node->inputs[0], node->inputs[1]);
	if (bc < 0) {
		sam3_log_error("add_f16: shape mismatch");
		return SAM3_EINVAL;
	}

#if SAM3_HAS_NEON_FP16
	struct binop_par_ctx_f16 ctx = {
		.a           = (const _Float16 *)node->inputs[0]->data,
		.b           = (const _Float16 *)node->inputs[1]->data,
		.out         = (_Float16 *)node->output->data,
		.n           = sam3_tensor_nelems(node->inputs[0]),
		.broadcast_n = bc,
	};
#else
	struct binop_par_ctx_f16 ctx = {
		.a           = (const uint16_t *)node->inputs[0]->data,
		.b           = (const uint16_t *)node->inputs[1]->data,
		.out         = (uint16_t *)node->output->data,
		.n           = sam3_tensor_nelems(node->inputs[0]),
		.broadcast_n = bc,
	};
#endif

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

	sam3_threadpool_parallel_for(pool, add_f16_parallel_fn, &ctx, n_tasks);

	return SAM3_OK;
}
