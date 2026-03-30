/*
 * src/backend/cpu/kernels/cpu_gelu.c - GELU activation kernel
 *
 * Fast GELU approximation using the tanh formula:
 *   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * Key types:  sam3_node, sam3_tensor
 * Depends on: cpu_kernels.h, cpu_simd.h, core/tensor.h
 * Used by:    cpu_backend.c (dispatch)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "cpu_kernels.h"
#include "cpu_simd.h"
#include "core/tensor.h"
#include "util/log.h"

#include <math.h>

#define GELU_SQRT_2_PI  0.7978845608f  /* sqrt(2/pi) */
#define GELU_COEFF      0.044715f

/* --- Scalar path (when no SIMD available) --- */

#if !SAM3_HAS_NEON && !SAM3_HAS_AVX2

static void gelu_f32_scalar(const float *in, float *out, int n)
{
	for (int i = 0; i < n; i++) {
		float x = in[i];
		float inner = GELU_SQRT_2_PI * (x + GELU_COEFF * x * x * x);
		out[i] = 0.5f * x * (1.0f + tanhf(inner));
	}
}

#endif /* !SAM3_HAS_NEON && !SAM3_HAS_AVX2 */

/* --- NEON path --- */

#if SAM3_HAS_NEON

static void gelu_f32_neon(const float *in, float *out, int n)
{
	float32x4_t half = vdupq_n_f32(0.5f);
	float32x4_t one = vdupq_n_f32(1.0f);
	float32x4_t coeff = vdupq_n_f32(GELU_COEFF);
	float32x4_t sqrt2pi = vdupq_n_f32(GELU_SQRT_2_PI);
	int i = 0;

	for (; i + 4 <= n; i += 4) {
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

	for (; i < n; i++) {
		float x = in[i];
		float inner = GELU_SQRT_2_PI * (x + GELU_COEFF * x * x * x);
		out[i] = 0.5f * x * (1.0f + tanhf(inner));
	}
}

#endif /* SAM3_HAS_NEON */

/* --- AVX2 path --- */

#if SAM3_HAS_AVX2

static void gelu_f32_avx2(const float *in, float *out, int n)
{
	__m256 half = _mm256_set1_ps(0.5f);
	__m256 one = _mm256_set1_ps(1.0f);
	__m256 coeff = _mm256_set1_ps(GELU_COEFF);
	__m256 sqrt2pi = _mm256_set1_ps(GELU_SQRT_2_PI);
	int i = 0;

	for (; i + 8 <= n; i += 8) {
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

	for (; i < n; i++) {
		float x = in[i];
		float inner = GELU_SQRT_2_PI * (x + GELU_COEFF * x * x * x);
		out[i] = 0.5f * x * (1.0f + tanhf(inner));
	}
}

#endif /* SAM3_HAS_AVX2 */

enum sam3_error cpu_kernel_gelu(const struct sam3_node *node)
{
	if (!node->inputs[0] || !node->output) {
		sam3_log_error("gelu: NULL tensor");
		return SAM3_EINVAL;
	}

	if (node->inputs[0]->dtype != SAM3_DTYPE_F32) {
		sam3_log_error("gelu: unsupported dtype");
		return SAM3_EINVAL;
	}

	const float *in = (const float *)node->inputs[0]->data;
	float *out = (float *)node->output->data;
	int n = sam3_tensor_nelems(node->inputs[0]);

#if SAM3_HAS_NEON
	gelu_f32_neon(in, out, n);
#elif SAM3_HAS_AVX2
	gelu_f32_avx2(in, out, n);
#else
	gelu_f32_scalar(in, out, n);
#endif

	return SAM3_OK;
}
