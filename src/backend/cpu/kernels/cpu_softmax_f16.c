/*
 * src/backend/cpu/kernels/cpu_softmax_f16.c - FP16 row-wise softmax
 *
 * Computes softmax along the last dimension for fp16 tensors. Uses the
 * numerically stable max-subtract trick. NEON fp16 path accumulates in
 * float16x8_t, upcasting only for expf calls. Scalar fallback uses f32.
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

#include <math.h>
#include <float.h>

/* --- NEON fp16 path --- */

#if SAM3_HAS_NEON_FP16

static void softmax_row_f16_neon(const _Float16 *in, _Float16 *out, int cols)
{
	int j = 0;

	/* Find max */
	float16x8_t vmax = vdupq_n_f16((_Float16)(-65504.0f));
	for (; j + 8 <= cols; j += 8)
		vmax = vmaxq_f16(vmax, vld1q_f16((const __fp16 *)(in + j)));
	_Float16 max_val = neon_f16_hmax(vmax);
	for (; j < cols; j++) {
		if (in[j] > max_val)
			max_val = in[j];
	}

	/* exp(x - max) and sum — expf needs f32 */
	float16x8_t vmax_s = vdupq_n_f16(max_val);
	float sum_f32 = 0.0f;
	j = 0;
	for (; j + 8 <= cols; j += 8) {
		float16x8_t v = vsubq_f16(
			vld1q_f16((const __fp16 *)(in + j)), vmax_s);
		/* Upcast to f32 for expf */
		float tmp[8];
		float32x4_t lo = vcvt_f32_f16(vget_low_f16(v));
		float32x4_t hi = vcvt_f32_f16(vget_high_f16(v));
		vst1q_f32(tmp,     lo);
		vst1q_f32(tmp + 4, hi);
		for (int k = 0; k < 8; k++)
			tmp[k] = expf(tmp[k]);
		lo = vld1q_f32(tmp);
		hi = vld1q_f32(tmp + 4);
		/* Store as fp16 */
		float16x8_t ve = vcombine_f16(vcvt_f16_f32(lo),
					       vcvt_f16_f32(hi));
		vst1q_f16((__fp16 *)(out + j), ve);
		sum_f32 += neon_hsum_f32(vaddq_f32(lo, hi));
	}
	for (; j < cols; j++) {
		float e = expf((float)(in[j] - max_val));
		out[j] = (_Float16)e;
		sum_f32 += e;
	}

	/* Normalize */
	_Float16 inv_sum = (_Float16)(1.0f / sum_f32);
	float16x8_t vinv = vdupq_n_f16(inv_sum);
	j = 0;
	for (; j + 8 <= cols; j += 8) {
		float16x8_t v = vld1q_f16((const __fp16 *)(out + j));
		vst1q_f16((__fp16 *)(out + j), vmulq_f16(v, vinv));
	}
	for (; j < cols; j++)
		out[j] = out[j] * inv_sum;
}

#else /* !SAM3_HAS_NEON_FP16 */

static void softmax_row_f16_scalar(const uint16_t *in, uint16_t *out, int cols)
{
	/* Find max */
	float max_val = -FLT_MAX;
	for (int j = 0; j < cols; j++) {
		float v = fp16_to_f32(in[j]);
		if (v > max_val)
			max_val = v;
	}

	/* exp(x - max) and sum */
	float sum = 0.0f;
	for (int j = 0; j < cols; j++) {
		float e = expf(fp16_to_f32(in[j]) - max_val);
		out[j] = f32_to_fp16(e);
		sum += e;
	}

	/* Normalize */
	float inv_sum = 1.0f / sum;
	for (int j = 0; j < cols; j++)
		out[j] = f32_to_fp16(fp16_to_f32(out[j]) * inv_sum);
}

#endif /* SAM3_HAS_NEON_FP16 */

/* --- Parallel dispatch --- */

#if SAM3_HAS_NEON_FP16

struct softmax_par_ctx_f16 {
	const _Float16 *in;
	_Float16       *out;
	int             cols;
	int             rows;
};

static void softmax_f16_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct softmax_par_ctx_f16 *ctx = (struct softmax_par_ctx_f16 *)arg;
	int chunk   = ctx->rows / n_tasks;
	int r_start = task_id * chunk;
	int r_end   = (task_id == n_tasks - 1) ? ctx->rows : r_start + chunk;

	if (r_start >= r_end)
		return;

	for (int r = r_start; r < r_end; r++) {
		softmax_row_f16_neon(ctx->in  + r * ctx->cols,
				     ctx->out + r * ctx->cols,
				     ctx->cols);
	}
}

#else /* !SAM3_HAS_NEON_FP16 */

struct softmax_par_ctx_f16 {
	const uint16_t *in;
	uint16_t       *out;
	int             cols;
	int             rows;
};

static void softmax_f16_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct softmax_par_ctx_f16 *ctx = (struct softmax_par_ctx_f16 *)arg;
	int chunk   = ctx->rows / n_tasks;
	int r_start = task_id * chunk;
	int r_end   = (task_id == n_tasks - 1) ? ctx->rows : r_start + chunk;

	if (r_start >= r_end)
		return;

	for (int r = r_start; r < r_end; r++) {
		softmax_row_f16_scalar(ctx->in  + r * ctx->cols,
				       ctx->out + r * ctx->cols,
				       ctx->cols);
	}
}

#endif /* SAM3_HAS_NEON_FP16 */

/*
 * cpu_kernel_softmax_f16 - Row-wise softmax for fp16 tensors.
 *
 * @node: Node with n_inputs>=1, input SAM3_DTYPE_F16. Softmax is
 *        applied along the last dimension.
 * @pool: Thread pool for parallel row execution.
 *
 * Returns SAM3_OK on success, SAM3_EINVAL on bad inputs.
 */
enum sam3_error cpu_kernel_softmax_f16(const struct sam3_node *node,
				       struct sam3_threadpool *pool)
{
	if (!node->inputs[0] || !node->output) {
		sam3_log_error("softmax_f16: NULL tensor");
		return SAM3_EINVAL;
	}

	struct sam3_tensor *in  = node->inputs[0];
	struct sam3_tensor *out = node->output;

	if (in->dtype != SAM3_DTYPE_F16) {
		sam3_log_error("softmax_f16: unsupported dtype");
		return SAM3_EINVAL;
	}

	int cols = in->dims[in->n_dims - 1];
	int rows = sam3_tensor_nelems(in) / cols;

#if SAM3_HAS_NEON_FP16
	struct softmax_par_ctx_f16 ctx = {
		.in   = (const _Float16 *)in->data,
		.out  = (_Float16 *)out->data,
		.cols = cols,
		.rows = rows,
	};
#else
	struct softmax_par_ctx_f16 ctx = {
		.in   = (const uint16_t *)in->data,
		.out  = (uint16_t *)out->data,
		.cols = cols,
		.rows = rows,
	};
#endif

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

	sam3_threadpool_parallel_for(pool, softmax_f16_parallel_fn,
				     &ctx, n_tasks);

	return SAM3_OK;
}
