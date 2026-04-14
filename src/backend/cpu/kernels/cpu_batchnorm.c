/*
 * src/backend/cpu/kernels/cpu_batchnorm.c - Batch normalization kernel
 *
 * Computes batch normalization (eval mode) on NHWC tensors using
 * pre-computed running statistics:
 *   out = (x - running_mean) / sqrt(running_var + eps) * gamma + beta
 * inputs[0]=input, inputs[1]=gamma[C], inputs[2]=beta[C],
 * inputs[3]=running_mean[C], inputs[4]=running_var[C].
 * eps is fixed at 1e-5.
 *
 * Key types:  sam3_node, sam3_tensor
 * Depends on: cpu_kernels.h, cpu_simd.h, core/tensor.h, util/threadpool.h
 * Used by:    cpu_dispatch.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "cpu_kernels.h"
#include "cpu_simd.h"
#include "core/tensor.h"
#include "util/log.h"
#include "util/threadpool.h"

#include <math.h>

#define BATCHNORM_EPS 1e-5f

/*
 * BatchNorm inference is a per-channel affine transform:
 *   out[c] = (x[c] - mean[c]) * inv_std[c] * gamma[c] + beta[c]
 * where inv_std[c] = 1/sqrt(var[c] + eps).
 *
 * The row functions normalise one spatial row of C channels.
 */

/* --- Scalar path --- */

#if !SAM3_HAS_NEON && !SAM3_HAS_AVX2

static void batchnorm_row_scalar(const float *in, float *out, int C,
				 const float *gamma, const float *beta,
				 const float *mean, const float *inv_std)
{
	for (int c = 0; c < C; c++) {
		float v = (in[c] - mean[c]) * inv_std[c];
		if (gamma)
			v *= gamma[c];
		if (beta)
			v += beta[c];
		out[c] = v;
	}
}

#endif /* !SAM3_HAS_NEON && !SAM3_HAS_AVX2 */

/* --- NEON path --- */

#if SAM3_HAS_NEON

static void batchnorm_row_neon(const float *in, float *out, int C,
			       const float *gamma, const float *beta,
			       const float *mean, const float *inv_std)
{
	int c = 0;

	for (; c + 4 <= C; c += 4) {
		float32x4_t v = vld1q_f32(in + c);
		v = vmulq_f32(vsubq_f32(v, vld1q_f32(mean + c)),
			      vld1q_f32(inv_std + c));
		if (gamma)
			v = vmulq_f32(v, vld1q_f32(gamma + c));
		if (beta)
			v = vaddq_f32(v, vld1q_f32(beta + c));
		vst1q_f32(out + c, v);
	}
	for (; c < C; c++) {
		float v = (in[c] - mean[c]) * inv_std[c];
		if (gamma)
			v *= gamma[c];
		if (beta)
			v += beta[c];
		out[c] = v;
	}
}

#endif /* SAM3_HAS_NEON */

/* --- AVX2 path --- */

#if SAM3_HAS_AVX2

static void batchnorm_row_avx2(const float *in, float *out, int C,
			       const float *gamma, const float *beta,
			       const float *mean, const float *inv_std)
{
	int c = 0;

	for (; c + 8 <= C; c += 8) {
		__m256 v = _mm256_loadu_ps(in + c);
		v = _mm256_mul_ps(
			_mm256_sub_ps(v, _mm256_loadu_ps(mean + c)),
			_mm256_loadu_ps(inv_std + c));
		if (gamma)
			v = _mm256_mul_ps(v, _mm256_loadu_ps(gamma + c));
		if (beta)
			v = _mm256_add_ps(v, _mm256_loadu_ps(beta + c));
		_mm256_storeu_ps(out + c, v);
	}
	for (; c < C; c++) {
		float v = (in[c] - mean[c]) * inv_std[c];
		if (gamma)
			v *= gamma[c];
		if (beta)
			v += beta[c];
		out[c] = v;
	}
}

#endif /* SAM3_HAS_AVX2 */

/* --- Parallel dispatch --- */

struct batchnorm_par_ctx {
	const float *in;
	float       *out;
	int          C;
	int          pixels;        /* N * H * W */
	const float *gamma;
	const float *beta;
	const float *mean;
	const float *inv_std;       /* precomputed 1/sqrt(var+eps) */
};

static void batchnorm_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct batchnorm_par_ctx *ctx = (struct batchnorm_par_ctx *)arg;
	int chunk = ctx->pixels / n_tasks;
	int start = task_id * chunk;
	int end = (task_id == n_tasks - 1) ? ctx->pixels : start + chunk;

	if (start >= end)
		return;

	for (int p = start; p < end; p++) {
#if SAM3_HAS_NEON
		batchnorm_row_neon(ctx->in + p * ctx->C,
				   ctx->out + p * ctx->C,
				   ctx->C, ctx->gamma, ctx->beta,
				   ctx->mean, ctx->inv_std);
#elif SAM3_HAS_AVX2
		batchnorm_row_avx2(ctx->in + p * ctx->C,
				   ctx->out + p * ctx->C,
				   ctx->C, ctx->gamma, ctx->beta,
				   ctx->mean, ctx->inv_std);
#else
		batchnorm_row_scalar(ctx->in + p * ctx->C,
				     ctx->out + p * ctx->C,
				     ctx->C, ctx->gamma, ctx->beta,
				     ctx->mean, ctx->inv_std);
#endif
	}
}

enum sam3_error cpu_kernel_batchnorm(const struct sam3_node *node,
				     struct sam3_threadpool *pool)
{
	if (!node->inputs[0] || !node->output) {
		sam3_log_error("batchnorm: NULL tensor");
		return SAM3_EINVAL;
	}

	struct sam3_tensor *in = node->inputs[0];

	if (in->dtype != SAM3_DTYPE_F32) {
		sam3_log_error("batchnorm: unsupported dtype %d", in->dtype);
		return SAM3_EINVAL;
	}

	if (node->n_inputs < 5 || !node->inputs[3] || !node->inputs[4]) {
		sam3_log_error("batchnorm: need running_mean and running_var");
		return SAM3_EINVAL;
	}

	int C = in->dims[in->n_dims - 1];
	int pixels = sam3_tensor_nelems(in) / C;

	const float *gamma = NULL;
	const float *beta = NULL;

	if (node->n_inputs > 1 && node->inputs[1])
		gamma = (const float *)node->inputs[1]->data;
	if (node->n_inputs > 2 && node->inputs[2])
		beta = (const float *)node->inputs[2]->data;

	const float *running_mean = (const float *)node->inputs[3]->data;
	const float *running_var = (const float *)node->inputs[4]->data;

	/*
	 * Precompute inv_std = 1/sqrt(var + eps) once per channel.
	 * C is bounded by model width (typically <= 1024), so a
	 * stack buffer is safe.
	 */
	float inv_std_buf[4096];
	if (C > 4096) {
		sam3_log_error("batchnorm: C=%d exceeds stack buffer", C);
		return SAM3_EINVAL;
	}
	for (int c = 0; c < C; c++)
		inv_std_buf[c] = 1.0f / sqrtf(running_var[c] + BATCHNORM_EPS);

	struct batchnorm_par_ctx ctx = {
		.in      = (const float *)in->data,
		.out     = (float *)node->output->data,
		.C       = C,
		.pixels  = pixels,
		.gamma   = gamma,
		.beta    = beta,
		.mean    = running_mean,
		.inv_std = inv_std_buf,
	};

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

	sam3_threadpool_parallel_for(pool, batchnorm_parallel_fn, &ctx,
				     n_tasks);

	return SAM3_OK;
}
