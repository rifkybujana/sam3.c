/*
 * src/backend/cpu/kernels/cpu_softmax.c - Row-wise softmax kernel
 *
 * Computes softmax along the last dimension. The tensor is viewed as
 * [rows, cols] where cols is the last dimension. Uses the numerically
 * stable max-subtract trick: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x))).
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
#include <float.h>

/* --- Scalar path (when no SIMD available) --- */

#if !SAM3_HAS_NEON && !SAM3_HAS_AVX2

static void softmax_row_scalar(const float *in, float *out, int cols)
{
	float max_val = -FLT_MAX;

	for (int j = 0; j < cols; j++) {
		if (in[j] > max_val)
			max_val = in[j];
	}

	float sum = 0.0f;
	for (int j = 0; j < cols; j++) {
		out[j] = expf(in[j] - max_val);
		sum += out[j];
	}

	float inv_sum = 1.0f / sum;
	for (int j = 0; j < cols; j++)
		out[j] *= inv_sum;
}

#endif /* !SAM3_HAS_NEON && !SAM3_HAS_AVX2 */

/* --- NEON path --- */

#if SAM3_HAS_NEON

static void softmax_row_neon(const float *in, float *out, int cols)
{
	int j = 0;

	/* Find max */
	float32x4_t vmax = vdupq_n_f32(-FLT_MAX);
	for (; j + 4 <= cols; j += 4)
		vmax = vmaxq_f32(vmax, vld1q_f32(in + j));

	float max_val = neon_hmax_f32(vmax);
	for (; j < cols; j++) {
		if (in[j] > max_val)
			max_val = in[j];
	}

	/* exp(x - max) and sum */
	float32x4_t vmax_scalar = vdupq_n_f32(max_val);
	float32x4_t vsum = vdupq_n_f32(0.0f);
	j = 0;
	for (; j + 4 <= cols; j += 4) {
		float32x4_t v = vsubq_f32(vld1q_f32(in + j), vmax_scalar);
		/* expf per-element — no NEON exp intrinsic */
		float tmp[4];
		vst1q_f32(tmp, v);
		tmp[0] = expf(tmp[0]);
		tmp[1] = expf(tmp[1]);
		tmp[2] = expf(tmp[2]);
		tmp[3] = expf(tmp[3]);
		float32x4_t ve = vld1q_f32(tmp);
		vst1q_f32(out + j, ve);
		vsum = vaddq_f32(vsum, ve);
	}

	float sum = neon_hsum_f32(vsum);
	for (; j < cols; j++) {
		out[j] = expf(in[j] - max_val);
		sum += out[j];
	}

	/* Normalize */
	float inv_sum = 1.0f / sum;
	float32x4_t vinv = vdupq_n_f32(inv_sum);
	j = 0;
	for (; j + 4 <= cols; j += 4)
		vst1q_f32(out + j, vmulq_f32(vld1q_f32(out + j), vinv));
	for (; j < cols; j++)
		out[j] *= inv_sum;
}

#endif /* SAM3_HAS_NEON */

/* --- AVX2 path --- */

#if SAM3_HAS_AVX2

static void softmax_row_avx2(const float *in, float *out, int cols)
{
	int j = 0;

	__m256 vmax = _mm256_set1_ps(-FLT_MAX);
	for (; j + 8 <= cols; j += 8)
		vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(in + j));

	float max_val = avx2_hmax_f32(vmax);
	for (; j < cols; j++) {
		if (in[j] > max_val)
			max_val = in[j];
	}

	__m256 vmax_scalar = _mm256_set1_ps(max_val);
	__m256 vsum = _mm256_setzero_ps();
	j = 0;
	for (; j + 8 <= cols; j += 8) {
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(in + j), vmax_scalar);
		float tmp[8];
		_mm256_storeu_ps(tmp, v);
		for (int k = 0; k < 8; k++)
			tmp[k] = expf(tmp[k]);
		__m256 ve = _mm256_loadu_ps(tmp);
		_mm256_storeu_ps(out + j, ve);
		vsum = _mm256_add_ps(vsum, ve);
	}

	float sum = avx2_hsum_f32(vsum);
	for (; j < cols; j++) {
		out[j] = expf(in[j] - max_val);
		sum += out[j];
	}

	float inv_sum = 1.0f / sum;
	__m256 vinv = _mm256_set1_ps(inv_sum);
	j = 0;
	for (; j + 8 <= cols; j += 8)
		_mm256_storeu_ps(out + j,
				 _mm256_mul_ps(_mm256_loadu_ps(out + j), vinv));
	for (; j < cols; j++)
		out[j] *= inv_sum;
}

#endif /* SAM3_HAS_AVX2 */

enum sam3_error cpu_kernel_softmax(const struct sam3_node *node)
{
	if (!node->inputs[0] || !node->output) {
		sam3_log_error("softmax: NULL tensor");
		return SAM3_EINVAL;
	}

	struct sam3_tensor *in = node->inputs[0];
	struct sam3_tensor *out = node->output;

	if (in->dtype != SAM3_DTYPE_F32) {
		sam3_log_error("softmax: unsupported dtype");
		return SAM3_EINVAL;
	}

	int cols = in->dims[in->n_dims - 1];
	int rows = sam3_tensor_nelems(in) / cols;

	const float *in_data = (const float *)in->data;
	float *out_data = (float *)out->data;

	for (int r = 0; r < rows; r++) {
#if SAM3_HAS_NEON
		softmax_row_neon(in_data + r * cols, out_data + r * cols, cols);
#elif SAM3_HAS_AVX2
		softmax_row_avx2(in_data + r * cols, out_data + r * cols, cols);
#else
		softmax_row_scalar(in_data + r * cols, out_data + r * cols,
				   cols);
#endif
	}

	return SAM3_OK;
}
