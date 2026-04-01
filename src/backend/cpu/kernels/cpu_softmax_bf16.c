/*
 * src/backend/cpu/kernels/cpu_softmax_bf16.c - BF16 row-wise softmax
 *
 * Computes softmax along the last dimension for bf16 tensors. Uses the
 * numerically stable max-subtract trick. All arithmetic is done in f32
 * since bf16 has no native arithmetic. NEON path uses 4-wide float32x4_t
 * with bf16<->f32 conversion helpers from core/half.h.
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
#include <float.h>

/* --- NEON bf16 path (f32 arithmetic, 4-wide) --- */

#if SAM3_HAS_NEON

static void softmax_row_bf16_neon(const uint16_t *in, uint16_t *out, int cols)
{
	int j = 0;

	/* Find max — accumulate in f32 */
	float32x4_t vmax = vdupq_n_f32(-FLT_MAX);
	for (; j + 4 <= cols; j += 4)
		vmax = vmaxq_f32(vmax, bf16x4_to_f32x4(in + j));
	float max_val = neon_hmax_f32(vmax);
	for (; j < cols; j++) {
		float v = bf16_to_f32(in[j]);
		if (v > max_val)
			max_val = v;
	}

	/* exp(x - max) and sum — all in f32 */
	float sum_f32 = 0.0f;
	j = 0;
	for (; j + 4 <= cols; j += 4) {
		float32x4_t v = vsubq_f32(bf16x4_to_f32x4(in + j),
					   vdupq_n_f32(max_val));
		float tmp[4];
		vst1q_f32(tmp, v);
		for (int k = 0; k < 4; k++)
			tmp[k] = expf(tmp[k]);
		float32x4_t ve = vld1q_f32(tmp);
		f32x4_to_bf16x4(out + j, ve);
		sum_f32 += neon_hsum_f32(ve);
	}
	for (; j < cols; j++) {
		float e = expf(bf16_to_f32(in[j]) - max_val);
		out[j] = f32_to_bf16(e);
		sum_f32 += e;
	}

	/* Normalize */
	float inv_sum = 1.0f / sum_f32;
	float32x4_t vinv = vdupq_n_f32(inv_sum);
	j = 0;
	for (; j + 4 <= cols; j += 4) {
		float32x4_t v = bf16x4_to_f32x4(out + j);
		f32x4_to_bf16x4(out + j, vmulq_f32(v, vinv));
	}
	for (; j < cols; j++)
		out[j] = f32_to_bf16(bf16_to_f32(out[j]) * inv_sum);
}

#else /* !SAM3_HAS_NEON */

static void softmax_row_bf16_scalar(const uint16_t *in, uint16_t *out,
				    int cols)
{
	/* Find max */
	float max_val = -FLT_MAX;
	for (int j = 0; j < cols; j++) {
		float v = bf16_to_f32(in[j]);
		if (v > max_val)
			max_val = v;
	}

	/* exp(x - max) and sum */
	float sum = 0.0f;
	for (int j = 0; j < cols; j++) {
		float e = expf(bf16_to_f32(in[j]) - max_val);
		out[j] = f32_to_bf16(e);
		sum += e;
	}

	/* Normalize */
	float inv_sum = 1.0f / sum;
	for (int j = 0; j < cols; j++)
		out[j] = f32_to_bf16(bf16_to_f32(out[j]) * inv_sum);
}

#endif /* SAM3_HAS_NEON */

/* --- Parallel dispatch --- */

struct softmax_par_ctx_bf16 {
	const uint16_t *in;
	uint16_t       *out;
	int             cols;
	int             rows;
};

static void softmax_bf16_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct softmax_par_ctx_bf16 *ctx =
		(struct softmax_par_ctx_bf16 *)arg;
	int chunk   = ctx->rows / n_tasks;
	int r_start = task_id * chunk;
	int r_end   = (task_id == n_tasks - 1) ? ctx->rows : r_start + chunk;

	if (r_start >= r_end)
		return;

	for (int r = r_start; r < r_end; r++) {
#if SAM3_HAS_NEON
		softmax_row_bf16_neon(ctx->in  + r * ctx->cols,
				      ctx->out + r * ctx->cols,
				      ctx->cols);
#else
		softmax_row_bf16_scalar(ctx->in  + r * ctx->cols,
					ctx->out + r * ctx->cols,
					ctx->cols);
#endif
	}
}

/*
 * cpu_kernel_softmax_bf16 - Row-wise softmax for bf16 tensors.
 *
 * @node: Node with n_inputs>=1, input SAM3_DTYPE_BF16. Softmax is
 *        applied along the last dimension.
 * @pool: Thread pool for parallel row execution.
 *
 * Returns SAM3_OK on success, SAM3_EINVAL on bad inputs.
 */
enum sam3_error cpu_kernel_softmax_bf16(const struct sam3_node *node,
					struct sam3_threadpool *pool)
{
	if (!node->inputs[0] || !node->output) {
		sam3_log_error("softmax_bf16: NULL tensor");
		return SAM3_EINVAL;
	}

	struct sam3_tensor *in  = node->inputs[0];
	struct sam3_tensor *out = node->output;

	if (in->dtype != SAM3_DTYPE_BF16) {
		sam3_log_error("softmax_bf16: unsupported dtype");
		return SAM3_EINVAL;
	}

	int cols = in->dims[in->n_dims - 1];
	int rows = sam3_tensor_nelems(in) / cols;

	struct softmax_par_ctx_bf16 ctx = {
		.in   = (const uint16_t *)in->data,
		.out  = (uint16_t *)out->data,
		.cols = cols,
		.rows = rows,
	};

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

	sam3_threadpool_parallel_for(pool, softmax_bf16_parallel_fn,
				     &ctx, n_tasks);

	return SAM3_OK;
}
