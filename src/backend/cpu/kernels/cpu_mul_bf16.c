/*
 * src/backend/cpu/kernels/cpu_mul_bf16.c - BF16 elementwise multiply kernel
 *
 * Implements element-wise multiplication for bf16 tensors with broadcasting.
 * BF16 has no native arithmetic on any platform, so we always upcast
 * to f32 for computation. NEON path uses 4-wide float32x4_t vectors
 * with bf16<->f32 conversion helpers from core/half.h.
 *
 * Key types:  sam3_node, sam3_tensor
 * Depends on: cpu_kernels.h, cpu_simd.h, core/half.h, core/tensor.h
 * Used by:    cpu_dispatch.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "cpu_kernels.h"
#include "cpu_simd.h"
#include "core/half.h"
#include "core/tensor.h"
#include "util/log.h"
#include "util/threadpool.h"

/*
 * check_broadcast_bf16_mul - Check if b broadcasts onto a: a is [M, N], b is [N].
 *
 * Returns the inner dimension N, or 0 if shapes match exactly,
 * or -1 on error. Does not check dtype.
 */
static int check_broadcast_bf16_mul(const struct sam3_tensor *a,
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

/* --- NEON bf16 path (f32 arithmetic, 4-wide) --- */

#if SAM3_HAS_NEON

static void mul_bf16_neon(const uint16_t *a, const uint16_t *b,
			  uint16_t *out, int broadcast_n,
			  int start, int end)
{
	if (broadcast_n <= 0) {
		int i = start;
		for (; i + 4 <= end; i += 4) {
			float32x4_t va = bf16x4_to_f32x4(a + i);
			float32x4_t vb = bf16x4_to_f32x4(b + i);
			f32x4_to_bf16x4(out + i, vmulq_f32(va, vb));
		}
		for (; i < end; i++)
			out[i] = f32_to_bf16(bf16_to_f32(a[i]) *
					     bf16_to_f32(b[i]));
	} else {
		for (int r = start; r < end; r++) {
			int base = r * broadcast_n;
			int j = 0;
			for (; j + 4 <= broadcast_n; j += 4) {
				float32x4_t va = bf16x4_to_f32x4(
					a + base + j);
				float32x4_t vb = bf16x4_to_f32x4(b + j);
				f32x4_to_bf16x4(out + base + j,
						vmulq_f32(va, vb));
			}
			for (; j < broadcast_n; j++)
				out[base + j] = f32_to_bf16(
					bf16_to_f32(a[base + j]) *
					bf16_to_f32(b[j]));
		}
	}
}

#else /* !SAM3_HAS_NEON */

static void mul_bf16_scalar(const uint16_t *a, const uint16_t *b,
			    uint16_t *out, int broadcast_n,
			    int start, int end)
{
	if (broadcast_n <= 0) {
		for (int i = start; i < end; i++) {
			out[i] = f32_to_bf16(
				bf16_to_f32(a[i]) * bf16_to_f32(b[i]));
		}
	} else {
		for (int i = start; i < end; i++) {
			out[i] = f32_to_bf16(
				bf16_to_f32(a[i]) *
				bf16_to_f32(b[i % broadcast_n]));
		}
	}
}

#endif /* SAM3_HAS_NEON */

/* --- Parallel dispatch --- */

struct binop_par_ctx_bf16_mul {
	const uint16_t *a;
	const uint16_t *b;
	uint16_t       *out;
	int             n;
	int             broadcast_n;
};

static void mul_bf16_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct binop_par_ctx_bf16_mul *ctx =
		(struct binop_par_ctx_bf16_mul *)arg;
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

#if SAM3_HAS_NEON
	mul_bf16_neon(ctx->a, ctx->b, ctx->out,
		      ctx->broadcast_n, start, end);
#else
	mul_bf16_scalar(ctx->a, ctx->b, ctx->out,
			ctx->broadcast_n, start, end);
#endif
}

/*
 * cpu_kernel_mul_bf16 - Element-wise multiply for bf16 tensors with broadcasting.
 *
 * @node: Node with n_inputs>=2, all SAM3_DTYPE_BF16.
 * @pool: Thread pool for parallel execution.
 *
 * Returns SAM3_OK on success, SAM3_EINVAL on bad inputs.
 */
enum sam3_error cpu_kernel_mul_bf16(const struct sam3_node *node,
				    struct sam3_threadpool *pool)
{
	if (node->n_inputs < 2 || !node->inputs[0] || !node->inputs[1] ||
	    !node->output) {
		sam3_log_error("mul_bf16: NULL tensor");
		return SAM3_EINVAL;
	}

	if (node->inputs[0]->dtype != SAM3_DTYPE_BF16 ||
	    node->inputs[1]->dtype != SAM3_DTYPE_BF16) {
		sam3_log_error("mul_bf16: unsupported dtype");
		return SAM3_EINVAL;
	}

	int bc = check_broadcast_bf16_mul(node->inputs[0], node->inputs[1]);
	if (bc < 0) {
		sam3_log_error("mul_bf16: shape mismatch");
		return SAM3_EINVAL;
	}

	struct binop_par_ctx_bf16_mul ctx = {
		.a           = (const uint16_t *)node->inputs[0]->data,
		.b           = (const uint16_t *)node->inputs[1]->data,
		.out         = (uint16_t *)node->output->data,
		.n           = sam3_tensor_nelems(node->inputs[0]),
		.broadcast_n = bc,
	};

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

	sam3_threadpool_parallel_for(pool, mul_bf16_parallel_fn, &ctx, n_tasks);

	return SAM3_OK;
}
