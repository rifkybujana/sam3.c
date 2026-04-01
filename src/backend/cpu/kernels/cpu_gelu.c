/*
 * src/backend/cpu/kernels/cpu_gelu.c - GELU activation kernel
 *
 * Fast GELU approximation using the tanh formula:
 *   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * Key types:  sam3_node, sam3_tensor
 * Depends on: cpu_kernels.h, cpu_simd.h, core/tensor.h, util/threadpool.h
 * Used by:    cpu_backend.c (dispatch)
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

#define GELU_SQRT_2_PI  0.7978845608f  /* sqrt(2/pi) */
#define GELU_COEFF      0.044715f

/* --- Scalar path (when no SIMD available) --- */

#if !SAM3_HAS_NEON && !SAM3_HAS_AVX2

static void gelu_f32_scalar(const float *in, float *out,
			    int start, int end)
{
	for (int i = start; i < end; i++) {
		float x = in[i];
		float inner = GELU_SQRT_2_PI * (x + GELU_COEFF * x * x * x);
		out[i] = 0.5f * x * (1.0f + tanhf(inner));
	}
}

#endif /* !SAM3_HAS_NEON && !SAM3_HAS_AVX2 */

/* --- NEON path --- */

#if SAM3_HAS_NEON

static void gelu_f32_neon(const float *in, float *out,
			  int start, int end)
{
	float32x4_t half = vdupq_n_f32(0.5f);
	float32x4_t one = vdupq_n_f32(1.0f);
	float32x4_t coeff = vdupq_n_f32(GELU_COEFF);
	float32x4_t sqrt2pi = vdupq_n_f32(GELU_SQRT_2_PI);
	int i = start;

	for (; i + 4 <= end; i += 4) {
		float32x4_t x = vld1q_f32(in + i);
		float32x4_t x3 = vmulq_f32(vmulq_f32(x, x), x);
		float32x4_t inner = vmulq_f32(sqrt2pi,
					       vfmaq_f32(x, coeff, x3));

		/* tanhf per-element — no NEON tanh intrinsic */
		float tmp[4];
		vst1q_f32(tmp, inner);
		tmp[0] = tanhf(tmp[0]);
		tmp[1] = tanhf(tmp[1]);
		tmp[2] = tanhf(tmp[2]);
		tmp[3] = tanhf(tmp[3]);
		float32x4_t tanh_v = vld1q_f32(tmp);

		float32x4_t result = vmulq_f32(half,
						vmulq_f32(x, vaddq_f32(one,
								       tanh_v)));
		vst1q_f32(out + i, result);
	}

	for (; i < end; i++) {
		float x = in[i];
		float inner = GELU_SQRT_2_PI * (x + GELU_COEFF * x * x * x);
		out[i] = 0.5f * x * (1.0f + tanhf(inner));
	}
}

#endif /* SAM3_HAS_NEON */

/* --- AVX2 path --- */

#if SAM3_HAS_AVX2

static void gelu_f32_avx2(const float *in, float *out,
			  int start, int end)
{
	__m256 half = _mm256_set1_ps(0.5f);
	__m256 one = _mm256_set1_ps(1.0f);
	__m256 coeff = _mm256_set1_ps(GELU_COEFF);
	__m256 sqrt2pi = _mm256_set1_ps(GELU_SQRT_2_PI);
	int i = start;

	for (; i + 8 <= end; i += 8) {
		__m256 x = _mm256_loadu_ps(in + i);
		__m256 x2 = _mm256_mul_ps(x, x);
		__m256 x3 = _mm256_mul_ps(x2, x);
		__m256 inner = _mm256_mul_ps(sqrt2pi,
			_mm256_fmadd_ps(coeff, x3, x));

		float tmp[8];
		_mm256_storeu_ps(tmp, inner);
		for (int k = 0; k < 8; k++)
			tmp[k] = tanhf(tmp[k]);
		__m256 tanh_v = _mm256_loadu_ps(tmp);

		__m256 result = _mm256_mul_ps(half,
			_mm256_mul_ps(x, _mm256_add_ps(one, tanh_v)));
		_mm256_storeu_ps(out + i, result);
	}

	for (; i < end; i++) {
		float x = in[i];
		float inner = GELU_SQRT_2_PI * (x + GELU_COEFF * x * x * x);
		out[i] = 0.5f * x * (1.0f + tanhf(inner));
	}
}

#endif /* SAM3_HAS_AVX2 */

/* --- Parallel dispatch --- */

struct gelu_par_ctx {
	const float *in;
	float       *out;
	int          n;
};

static void gelu_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct gelu_par_ctx *ctx = (struct gelu_par_ctx *)arg;
	int chunk = ctx->n / n_tasks;
	int start = task_id * chunk;
	int end = (task_id == n_tasks - 1) ? ctx->n : start + chunk;

	if (start >= end)
		return;

#if SAM3_HAS_NEON
	gelu_f32_neon(ctx->in, ctx->out, start, end);
#elif SAM3_HAS_AVX2
	gelu_f32_avx2(ctx->in, ctx->out, start, end);
#else
	gelu_f32_scalar(ctx->in, ctx->out, start, end);
#endif
}

enum sam3_error cpu_kernel_gelu(const struct sam3_node *node,
				struct sam3_threadpool *pool)
{
	if (!node->inputs[0] || !node->output) {
		sam3_log_error("gelu: NULL tensor");
		return SAM3_EINVAL;
	}

	if (node->inputs[0]->dtype != SAM3_DTYPE_F32) {
		sam3_log_error("gelu: unsupported dtype");
		return SAM3_EINVAL;
	}

	struct gelu_par_ctx ctx = {
		.in  = (const float *)node->inputs[0]->data,
		.out = (float *)node->output->data,
		.n   = sam3_tensor_nelems(node->inputs[0]),
	};

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

	sam3_threadpool_parallel_for(pool, gelu_parallel_fn, &ctx, n_tasks);

	return SAM3_OK;
}
