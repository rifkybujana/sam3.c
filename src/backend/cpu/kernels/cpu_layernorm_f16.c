/*
 * src/backend/cpu/kernels/cpu_layernorm_f16.c - FP16 layer normalization
 *
 * Layer normalization for fp16 tensors along the last dimension.
 * NEON fp16 path uses float16x8_t for elementwise ops, upcasting
 * to f32 for mean/variance reduction and sqrtf. Scalar fallback
 * does everything in f32.
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

#define LAYERNORM_EPS 1e-5f

/* --- NEON fp16 path --- */

#if SAM3_HAS_NEON_FP16

static void layernorm_row_f16_neon(const _Float16 *in, _Float16 *out,
				   int cols, const _Float16 *gamma,
				   const _Float16 *beta)
{
	int j = 0;

	/* Mean — accumulate in f32 for precision */
	float32x4_t vsum = vdupq_n_f32(0.0f);
	for (; j + 8 <= cols; j += 8) {
		float16x8_t v = vld1q_f16((const __fp16 *)(in + j));
		vsum = vaddq_f32(vsum, vcvt_f32_f16(vget_low_f16(v)));
		vsum = vaddq_f32(vsum, vcvt_f32_f16(vget_high_f16(v)));
	}
	float sum = neon_hsum_f32(vsum);
	for (; j < cols; j++)
		sum += (float)in[j];
	float mean = sum / (float)cols;

	/* Variance — accumulate in f32 */
	float32x4_t vvar  = vdupq_n_f32(0.0f);
	float32x4_t vmean = vdupq_n_f32(mean);
	j = 0;
	for (; j + 8 <= cols; j += 8) {
		float16x8_t v  = vld1q_f16((const __fp16 *)(in + j));
		float32x4_t lo = vsubq_f32(vcvt_f32_f16(vget_low_f16(v)),
					    vmean);
		float32x4_t hi = vsubq_f32(vcvt_f32_f16(vget_high_f16(v)),
					    vmean);
		vvar = vfmaq_f32(vvar, lo, lo);
		vvar = vfmaq_f32(vvar, hi, hi);
	}
	float var_sum = neon_hsum_f32(vvar);
	for (; j < cols; j++) {
		float d = (float)in[j] - mean;
		var_sum += d * d;
	}
	float inv_std = 1.0f / sqrtf(var_sum / (float)cols + LAYERNORM_EPS);

	/* Normalize — fp16 operations, scale/shift in fp16 */
	_Float16 h_mean   = (_Float16)mean;
	_Float16 h_inv_std = (_Float16)inv_std;
	float16x8_t vmean_h = vdupq_n_f16(h_mean);
	float16x8_t vinv_h  = vdupq_n_f16(h_inv_std);
	j = 0;
	for (; j + 8 <= cols; j += 8) {
		float16x8_t v = vmulq_f16(
			vsubq_f16(vld1q_f16((const __fp16 *)(in + j)),
				  vmean_h),
			vinv_h);
		if (gamma)
			v = vmulq_f16(v, vld1q_f16((const __fp16 *)(gamma + j)));
		if (beta)
			v = vaddq_f16(v, vld1q_f16((const __fp16 *)(beta + j)));
		vst1q_f16((__fp16 *)(out + j), v);
	}
	for (; j < cols; j++) {
		_Float16 val = (in[j] - h_mean) * h_inv_std;
		if (gamma)
			val = val * gamma[j];
		if (beta)
			val = val + beta[j];
		out[j] = val;
	}
}

#else /* !SAM3_HAS_NEON_FP16 */

static void layernorm_row_f16_scalar(const uint16_t *in, uint16_t *out,
				     int cols, const uint16_t *gamma,
				     const uint16_t *beta)
{
	/* Compute mean in f32 */
	float sum = 0.0f;
	for (int j = 0; j < cols; j++)
		sum += fp16_to_f32(in[j]);
	float mean = sum / (float)cols;

	/* Compute variance in f32 */
	float var_sum = 0.0f;
	for (int j = 0; j < cols; j++) {
		float d = fp16_to_f32(in[j]) - mean;
		var_sum += d * d;
	}
	float inv_std = 1.0f / sqrtf(var_sum / (float)cols + LAYERNORM_EPS);

	/* Normalize, scale, shift */
	for (int j = 0; j < cols; j++) {
		float val = (fp16_to_f32(in[j]) - mean) * inv_std;
		if (gamma)
			val *= fp16_to_f32(gamma[j]);
		if (beta)
			val += fp16_to_f32(beta[j]);
		out[j] = f32_to_fp16(val);
	}
}

#endif /* SAM3_HAS_NEON_FP16 */

/* --- Parallel dispatch --- */

#if SAM3_HAS_NEON_FP16

struct layernorm_par_ctx_f16 {
	const _Float16 *in;
	_Float16       *out;
	int             cols;
	int             rows;
	const _Float16 *gamma;
	const _Float16 *beta;
};

static void layernorm_f16_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct layernorm_par_ctx_f16 *ctx =
		(struct layernorm_par_ctx_f16 *)arg;
	int chunk   = ctx->rows / n_tasks;
	int r_start = task_id * chunk;
	int r_end   = (task_id == n_tasks - 1) ? ctx->rows : r_start + chunk;

	if (r_start >= r_end)
		return;

	for (int r = r_start; r < r_end; r++) {
		layernorm_row_f16_neon(ctx->in  + r * ctx->cols,
				       ctx->out + r * ctx->cols,
				       ctx->cols,
				       ctx->gamma, ctx->beta);
	}
}

#else /* !SAM3_HAS_NEON_FP16 */

struct layernorm_par_ctx_f16 {
	const uint16_t *in;
	uint16_t       *out;
	int             cols;
	int             rows;
	const uint16_t *gamma;
	const uint16_t *beta;
};

static void layernorm_f16_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct layernorm_par_ctx_f16 *ctx =
		(struct layernorm_par_ctx_f16 *)arg;
	int chunk   = ctx->rows / n_tasks;
	int r_start = task_id * chunk;
	int r_end   = (task_id == n_tasks - 1) ? ctx->rows : r_start + chunk;

	if (r_start >= r_end)
		return;

	for (int r = r_start; r < r_end; r++) {
		layernorm_row_f16_scalar(ctx->in  + r * ctx->cols,
					 ctx->out + r * ctx->cols,
					 ctx->cols,
					 ctx->gamma, ctx->beta);
	}
}

#endif /* SAM3_HAS_NEON_FP16 */

/*
 * cpu_kernel_layernorm_f16 - Layer normalization for fp16 tensors.
 *
 * @node: Node with n_inputs>=1, input SAM3_DTYPE_F16. Optional
 *        inputs[1]=gamma, inputs[2]=beta (both F16 if present).
 * @pool: Thread pool for parallel row execution.
 *
 * Returns SAM3_OK on success, SAM3_EINVAL on bad inputs.
 */
enum sam3_error cpu_kernel_layernorm_f16(const struct sam3_node *node,
					 struct sam3_threadpool *pool)
{
	if (!node->inputs[0] || !node->output) {
		sam3_log_error("layernorm_f16: NULL tensor");
		return SAM3_EINVAL;
	}

	struct sam3_tensor *in  = node->inputs[0];
	struct sam3_tensor *out = node->output;

	if (in->dtype != SAM3_DTYPE_F16) {
		sam3_log_error("layernorm_f16: unsupported dtype");
		return SAM3_EINVAL;
	}

	int cols = in->dims[in->n_dims - 1];
	int rows = sam3_tensor_nelems(in) / cols;

#if SAM3_HAS_NEON_FP16
	const _Float16 *gamma = NULL;
	const _Float16 *beta  = NULL;

	if (node->n_inputs > 1 && node->inputs[1])
		gamma = (const _Float16 *)node->inputs[1]->data;
	if (node->n_inputs > 2 && node->inputs[2])
		beta  = (const _Float16 *)node->inputs[2]->data;

	struct layernorm_par_ctx_f16 ctx = {
		.in    = (const _Float16 *)in->data,
		.out   = (_Float16 *)out->data,
		.cols  = cols,
		.rows  = rows,
		.gamma = gamma,
		.beta  = beta,
	};
#else
	const uint16_t *gamma = NULL;
	const uint16_t *beta  = NULL;

	if (node->n_inputs > 1 && node->inputs[1])
		gamma = (const uint16_t *)node->inputs[1]->data;
	if (node->n_inputs > 2 && node->inputs[2])
		beta  = (const uint16_t *)node->inputs[2]->data;

	struct layernorm_par_ctx_f16 ctx = {
		.in    = (const uint16_t *)in->data,
		.out   = (uint16_t *)out->data,
		.cols  = cols,
		.rows  = rows,
		.gamma = gamma,
		.beta  = beta,
	};
#endif

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

	sam3_threadpool_parallel_for(pool, layernorm_f16_parallel_fn,
				     &ctx, n_tasks);

	return SAM3_OK;
}
