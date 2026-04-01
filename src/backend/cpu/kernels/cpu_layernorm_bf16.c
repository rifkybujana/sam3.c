/*
 * src/backend/cpu/kernels/cpu_layernorm_bf16.c - BF16 layer normalization
 *
 * Layer normalization for bf16 tensors along the last dimension. All
 * arithmetic is done in f32 since bf16 has no native arithmetic. NEON
 * path uses 4-wide float32x4_t with bf16<->f32 conversion helpers from
 * core/half.h. Scalar fallback does everything in f32.
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

#define LAYERNORM_EPS 1e-5f

/* --- NEON bf16 path (f32 arithmetic, 4-wide) --- */

#if SAM3_HAS_NEON

static void layernorm_row_bf16_neon(const uint16_t *in, uint16_t *out,
				    int cols, const uint16_t *gamma,
				    const uint16_t *beta)
{
	int j = 0;

	/* Mean — accumulate in f32 */
	float32x4_t vsum = vdupq_n_f32(0.0f);
	for (; j + 4 <= cols; j += 4)
		vsum = vaddq_f32(vsum, bf16x4_to_f32x4(in + j));
	float sum = neon_hsum_f32(vsum);
	for (; j < cols; j++)
		sum += bf16_to_f32(in[j]);
	float mean = sum / (float)cols;

	/* Variance — accumulate in f32 */
	float32x4_t vvar  = vdupq_n_f32(0.0f);
	float32x4_t vmean = vdupq_n_f32(mean);
	j = 0;
	for (; j + 4 <= cols; j += 4) {
		float32x4_t v = vsubq_f32(bf16x4_to_f32x4(in + j), vmean);
		vvar = vfmaq_f32(vvar, v, v);
	}
	float var_sum = neon_hsum_f32(vvar);
	for (; j < cols; j++) {
		float d = bf16_to_f32(in[j]) - mean;
		var_sum += d * d;
	}
	float inv_std = 1.0f / sqrtf(var_sum / (float)cols + LAYERNORM_EPS);

	/* Normalize, scale, shift — all in f32 */
	float32x4_t vmean_f = vdupq_n_f32(mean);
	float32x4_t vinv_f  = vdupq_n_f32(inv_std);
	j = 0;
	for (; j + 4 <= cols; j += 4) {
		float32x4_t v = vmulq_f32(
			vsubq_f32(bf16x4_to_f32x4(in + j), vmean_f),
			vinv_f);
		if (gamma)
			v = vmulq_f32(v, bf16x4_to_f32x4(gamma + j));
		if (beta)
			v = vaddq_f32(v, bf16x4_to_f32x4(beta + j));
		f32x4_to_bf16x4(out + j, v);
	}
	for (; j < cols; j++) {
		float val = (bf16_to_f32(in[j]) - mean) * inv_std;
		if (gamma)
			val *= bf16_to_f32(gamma[j]);
		if (beta)
			val += bf16_to_f32(beta[j]);
		out[j] = f32_to_bf16(val);
	}
}

#else /* !SAM3_HAS_NEON */

static void layernorm_row_bf16_scalar(const uint16_t *in, uint16_t *out,
				      int cols, const uint16_t *gamma,
				      const uint16_t *beta)
{
	/* Compute mean in f32 */
	float sum = 0.0f;
	for (int j = 0; j < cols; j++)
		sum += bf16_to_f32(in[j]);
	float mean = sum / (float)cols;

	/* Compute variance in f32 */
	float var_sum = 0.0f;
	for (int j = 0; j < cols; j++) {
		float d = bf16_to_f32(in[j]) - mean;
		var_sum += d * d;
	}
	float inv_std = 1.0f / sqrtf(var_sum / (float)cols + LAYERNORM_EPS);

	/* Normalize, scale, shift */
	for (int j = 0; j < cols; j++) {
		float val = (bf16_to_f32(in[j]) - mean) * inv_std;
		if (gamma)
			val *= bf16_to_f32(gamma[j]);
		if (beta)
			val += bf16_to_f32(beta[j]);
		out[j] = f32_to_bf16(val);
	}
}

#endif /* SAM3_HAS_NEON */

/* --- Parallel dispatch --- */

struct layernorm_par_ctx_bf16 {
	const uint16_t *in;
	uint16_t       *out;
	int             cols;
	int             rows;
	const uint16_t *gamma;
	const uint16_t *beta;
};

static void layernorm_bf16_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct layernorm_par_ctx_bf16 *ctx =
		(struct layernorm_par_ctx_bf16 *)arg;
	int chunk   = ctx->rows / n_tasks;
	int r_start = task_id * chunk;
	int r_end   = (task_id == n_tasks - 1) ? ctx->rows : r_start + chunk;

	if (r_start >= r_end)
		return;

	for (int r = r_start; r < r_end; r++) {
#if SAM3_HAS_NEON
		layernorm_row_bf16_neon(ctx->in  + r * ctx->cols,
					ctx->out + r * ctx->cols,
					ctx->cols,
					ctx->gamma, ctx->beta);
#else
		layernorm_row_bf16_scalar(ctx->in  + r * ctx->cols,
					  ctx->out + r * ctx->cols,
					  ctx->cols,
					  ctx->gamma, ctx->beta);
#endif
	}
}

/*
 * cpu_kernel_layernorm_bf16 - Layer normalization for bf16 tensors.
 *
 * @node: Node with n_inputs>=1, input SAM3_DTYPE_BF16. Optional
 *        inputs[1]=gamma, inputs[2]=beta (both BF16 if present).
 * @pool: Thread pool for parallel row execution.
 *
 * Returns SAM3_OK on success, SAM3_EINVAL on bad inputs.
 */
enum sam3_error cpu_kernel_layernorm_bf16(const struct sam3_node *node,
					  struct sam3_threadpool *pool)
{
	if (!node->inputs[0] || !node->output) {
		sam3_log_error("layernorm_bf16: NULL tensor");
		return SAM3_EINVAL;
	}

	struct sam3_tensor *in  = node->inputs[0];
	struct sam3_tensor *out = node->output;

	if (in->dtype != SAM3_DTYPE_BF16) {
		sam3_log_error("layernorm_bf16: unsupported dtype");
		return SAM3_EINVAL;
	}

	int cols = in->dims[in->n_dims - 1];
	int rows = sam3_tensor_nelems(in) / cols;

	const uint16_t *gamma = NULL;
	const uint16_t *beta  = NULL;

	if (node->n_inputs > 1 && node->inputs[1])
		gamma = (const uint16_t *)node->inputs[1]->data;
	if (node->n_inputs > 2 && node->inputs[2])
		beta  = (const uint16_t *)node->inputs[2]->data;

	struct layernorm_par_ctx_bf16 ctx = {
		.in    = (const uint16_t *)in->data,
		.out   = (uint16_t *)out->data,
		.cols  = cols,
		.rows  = rows,
		.gamma = gamma,
		.beta  = beta,
	};

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

	sam3_threadpool_parallel_for(pool, layernorm_bf16_parallel_fn,
				     &ctx, n_tasks);

	return SAM3_OK;
}
