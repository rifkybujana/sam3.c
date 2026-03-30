/*
 * src/backend/cpu/kernels/cpu_layernorm.c - Layer normalization kernel
 *
 * Computes layer normalization along the last dimension:
 *   out = (x - mean) / sqrt(var + eps) * gamma + beta
 * inputs[0]=input, inputs[1]=gamma (optional), inputs[2]=beta (optional).
 * eps is fixed at 1e-5.
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

#define LAYERNORM_EPS 1e-5f

/* --- Scalar path (when no SIMD available) --- */

#if !SAM3_HAS_NEON && !SAM3_HAS_AVX2

static void layernorm_row_scalar(const float *in, float *out, int cols,
				 const float *gamma, const float *beta)
{
	/* Compute mean */
	float sum = 0.0f;
	for (int j = 0; j < cols; j++)
		sum += in[j];
	float mean = sum / (float)cols;

	/* Compute variance */
	float var_sum = 0.0f;
	for (int j = 0; j < cols; j++) {
		float d = in[j] - mean;
		var_sum += d * d;
	}
	float inv_std = 1.0f / sqrtf(var_sum / (float)cols + LAYERNORM_EPS);

	/* Normalize, scale, shift */
	for (int j = 0; j < cols; j++) {
		float val = (in[j] - mean) * inv_std;
		if (gamma)
			val *= gamma[j];
		if (beta)
			val += beta[j];
		out[j] = val;
	}
}

#endif /* !SAM3_HAS_NEON && !SAM3_HAS_AVX2 */

/* --- NEON path --- */

#if SAM3_HAS_NEON

static void layernorm_row_neon(const float *in, float *out, int cols,
			       const float *gamma, const float *beta)
{
	int j = 0;

	/* Mean */
	float32x4_t vsum = vdupq_n_f32(0.0f);
	for (; j + 4 <= cols; j += 4)
		vsum = vaddq_f32(vsum, vld1q_f32(in + j));
	float sum = neon_hsum_f32(vsum);
	for (; j < cols; j++)
		sum += in[j];
	float mean = sum / (float)cols;

	/* Variance */
	float32x4_t vmean = vdupq_n_f32(mean);
	float32x4_t vvar = vdupq_n_f32(0.0f);
	j = 0;
	for (; j + 4 <= cols; j += 4) {
		float32x4_t d = vsubq_f32(vld1q_f32(in + j), vmean);
		vvar = vfmaq_f32(vvar, d, d);
	}
	float var_sum = neon_hsum_f32(vvar);
	for (; j < cols; j++) {
		float d = in[j] - mean;
		var_sum += d * d;
	}
	float inv_std = 1.0f / sqrtf(var_sum / (float)cols + LAYERNORM_EPS);

	/* Normalize + scale + shift */
	float32x4_t vinv = vdupq_n_f32(inv_std);
	j = 0;
	for (; j + 4 <= cols; j += 4) {
		float32x4_t v = vmulq_f32(
			vsubq_f32(vld1q_f32(in + j), vmean), vinv);
		if (gamma)
			v = vmulq_f32(v, vld1q_f32(gamma + j));
		if (beta)
			v = vaddq_f32(v, vld1q_f32(beta + j));
		vst1q_f32(out + j, v);
	}
	for (; j < cols; j++) {
		float val = (in[j] - mean) * inv_std;
		if (gamma)
			val *= gamma[j];
		if (beta)
			val += beta[j];
		out[j] = val;
	}
}

#endif /* SAM3_HAS_NEON */

/* --- AVX2 path --- */

#if SAM3_HAS_AVX2

static void layernorm_row_avx2(const float *in, float *out, int cols,
			       const float *gamma, const float *beta)
{
	int j = 0;

	__m256 vsum = _mm256_setzero_ps();
	for (; j + 8 <= cols; j += 8)
		vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(in + j));
	float sum = avx2_hsum_f32(vsum);
	for (; j < cols; j++)
		sum += in[j];
	float mean = sum / (float)cols;

	__m256 vmean = _mm256_set1_ps(mean);
	__m256 vvar = _mm256_setzero_ps();
	j = 0;
	for (; j + 8 <= cols; j += 8) {
		__m256 d = _mm256_sub_ps(_mm256_loadu_ps(in + j), vmean);
		vvar = _mm256_fmadd_ps(d, d, vvar);
	}
	float var_sum = avx2_hsum_f32(vvar);
	for (; j < cols; j++) {
		float d = in[j] - mean;
		var_sum += d * d;
	}
	float inv_std = 1.0f / sqrtf(var_sum / (float)cols + LAYERNORM_EPS);

	__m256 vinv = _mm256_set1_ps(inv_std);
	j = 0;
	for (; j + 8 <= cols; j += 8) {
		__m256 v = _mm256_mul_ps(
			_mm256_sub_ps(_mm256_loadu_ps(in + j), vmean), vinv);
		if (gamma)
			v = _mm256_mul_ps(v, _mm256_loadu_ps(gamma + j));
		if (beta)
			v = _mm256_add_ps(v, _mm256_loadu_ps(beta + j));
		_mm256_storeu_ps(out + j, v);
	}
	for (; j < cols; j++) {
		float val = (in[j] - mean) * inv_std;
		if (gamma)
			val *= gamma[j];
		if (beta)
			val += beta[j];
		out[j] = val;
	}
}

#endif /* SAM3_HAS_AVX2 */

enum sam3_error cpu_kernel_layernorm(const struct sam3_node *node)
{
	if (!node->inputs[0] || !node->output) {
		sam3_log_error("layernorm: NULL tensor");
		return SAM3_EINVAL;
	}

	struct sam3_tensor *in = node->inputs[0];
	struct sam3_tensor *out = node->output;

	if (in->dtype != SAM3_DTYPE_F32) {
		sam3_log_error("layernorm: unsupported dtype");
		return SAM3_EINVAL;
	}

	int cols = in->dims[in->n_dims - 1];
	int rows = sam3_tensor_nelems(in) / cols;

	const float *gamma = NULL;
	const float *beta = NULL;

	if (node->n_inputs > 1 && node->inputs[1])
		gamma = (const float *)node->inputs[1]->data;
	if (node->n_inputs > 2 && node->inputs[2])
		beta = (const float *)node->inputs[2]->data;

	const float *in_data = (const float *)in->data;
	float *out_data = (float *)out->data;

	for (int r = 0; r < rows; r++) {
#if SAM3_HAS_NEON
		layernorm_row_neon(in_data + r * cols, out_data + r * cols,
				   cols, gamma, beta);
#elif SAM3_HAS_AVX2
		layernorm_row_avx2(in_data + r * cols, out_data + r * cols,
				   cols, gamma, beta);
#else
		layernorm_row_scalar(in_data + r * cols, out_data + r * cols,
				     cols, gamma, beta);
#endif
	}

	return SAM3_OK;
}
