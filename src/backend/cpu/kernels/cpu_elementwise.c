/*
 * src/backend/cpu/kernels/cpu_elementwise.c - Elementwise add, mul, relu
 *
 * Implements add, mul with [M,N]+[N] broadcasting (bias add pattern),
 * and ReLU activation. Each op has scalar and NEON/AVX2 paths.
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

#include <string.h>

/*
 * Check if b broadcasts onto a: a is [M, N], b is [N].
 * Returns the inner dimension N, or 0 if shapes match exactly,
 * or -1 on error.
 */
static int check_broadcast(const struct sam3_tensor *a,
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

/* --- Scalar paths (when no SIMD available) --- */

#if !SAM3_HAS_NEON && !SAM3_HAS_AVX2

static void add_f32_scalar(const float *a, const float *b, float *out,
			   int n, int broadcast_n)
{
	if (broadcast_n <= 0) {
		for (int i = 0; i < n; i++)
			out[i] = a[i] + b[i];
	} else {
		for (int i = 0; i < n; i++)
			out[i] = a[i] + b[i % broadcast_n];
	}
}

static void mul_f32_scalar(const float *a, const float *b, float *out,
			   int n, int broadcast_n)
{
	if (broadcast_n <= 0) {
		for (int i = 0; i < n; i++)
			out[i] = a[i] * b[i];
	} else {
		for (int i = 0; i < n; i++)
			out[i] = a[i] * b[i % broadcast_n];
	}
}

static void relu_f32_scalar(const float *in, float *out, int n)
{
	for (int i = 0; i < n; i++)
		out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

#endif /* !SAM3_HAS_NEON && !SAM3_HAS_AVX2 */

/* --- NEON paths --- */

#if SAM3_HAS_NEON

static void add_f32_neon(const float *a, const float *b, float *out,
			 int n, int broadcast_n)
{
	int i = 0;

	if (broadcast_n <= 0) {
		for (; i + 4 <= n; i += 4) {
			float32x4_t va = vld1q_f32(a + i);
			float32x4_t vb = vld1q_f32(b + i);
			vst1q_f32(out + i, vaddq_f32(va, vb));
		}
		for (; i < n; i++)
			out[i] = a[i] + b[i];
	} else {
		/* Broadcasting: a is [rows, N], b is [N] */
		int rows = n / broadcast_n;
		for (int r = 0; r < rows; r++) {
			int base = r * broadcast_n;
			int j = 0;
			for (; j + 4 <= broadcast_n; j += 4) {
				float32x4_t va = vld1q_f32(a + base + j);
				float32x4_t vb = vld1q_f32(b + j);
				vst1q_f32(out + base + j, vaddq_f32(va, vb));
			}
			for (; j < broadcast_n; j++)
				out[base + j] = a[base + j] + b[j];
		}
	}
}

static void mul_f32_neon(const float *a, const float *b, float *out,
			 int n, int broadcast_n)
{
	int i = 0;

	if (broadcast_n <= 0) {
		for (; i + 4 <= n; i += 4) {
			float32x4_t va = vld1q_f32(a + i);
			float32x4_t vb = vld1q_f32(b + i);
			vst1q_f32(out + i, vmulq_f32(va, vb));
		}
		for (; i < n; i++)
			out[i] = a[i] * b[i];
	} else {
		int rows = n / broadcast_n;
		for (int r = 0; r < rows; r++) {
			int base = r * broadcast_n;
			int j = 0;
			for (; j + 4 <= broadcast_n; j += 4) {
				float32x4_t va = vld1q_f32(a + base + j);
				float32x4_t vb = vld1q_f32(b + j);
				vst1q_f32(out + base + j, vmulq_f32(va, vb));
			}
			for (; j < broadcast_n; j++)
				out[base + j] = a[base + j] * b[j];
		}
	}
}

static void relu_f32_neon(const float *in, float *out, int n)
{
	float32x4_t zero = vdupq_n_f32(0.0f);
	int i = 0;

	for (; i + 4 <= n; i += 4) {
		float32x4_t v = vld1q_f32(in + i);
		vst1q_f32(out + i, vmaxq_f32(v, zero));
	}
	for (; i < n; i++)
		out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

#endif /* SAM3_HAS_NEON */

/* --- AVX2 paths --- */

#if SAM3_HAS_AVX2

static void add_f32_avx2(const float *a, const float *b, float *out,
			  int n, int broadcast_n)
{
	int i = 0;

	if (broadcast_n <= 0) {
		for (; i + 8 <= n; i += 8) {
			__m256 va = _mm256_loadu_ps(a + i);
			__m256 vb = _mm256_loadu_ps(b + i);
			_mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
		}
		for (; i < n; i++)
			out[i] = a[i] + b[i];
	} else {
		int rows = n / broadcast_n;
		for (int r = 0; r < rows; r++) {
			int base = r * broadcast_n;
			int j = 0;
			for (; j + 8 <= broadcast_n; j += 8) {
				__m256 va = _mm256_loadu_ps(a + base + j);
				__m256 vb = _mm256_loadu_ps(b + j);
				_mm256_storeu_ps(out + base + j,
						 _mm256_add_ps(va, vb));
			}
			for (; j < broadcast_n; j++)
				out[base + j] = a[base + j] + b[j];
		}
	}
}

static void mul_f32_avx2(const float *a, const float *b, float *out,
			  int n, int broadcast_n)
{
	int i = 0;

	if (broadcast_n <= 0) {
		for (; i + 8 <= n; i += 8) {
			__m256 va = _mm256_loadu_ps(a + i);
			__m256 vb = _mm256_loadu_ps(b + i);
			_mm256_storeu_ps(out + i, _mm256_mul_ps(va, vb));
		}
		for (; i < n; i++)
			out[i] = a[i] * b[i];
	} else {
		int rows = n / broadcast_n;
		for (int r = 0; r < rows; r++) {
			int base = r * broadcast_n;
			int j = 0;
			for (; j + 8 <= broadcast_n; j += 8) {
				__m256 va = _mm256_loadu_ps(a + base + j);
				__m256 vb = _mm256_loadu_ps(b + j);
				_mm256_storeu_ps(out + base + j,
						 _mm256_mul_ps(va, vb));
			}
			for (; j < broadcast_n; j++)
				out[base + j] = a[base + j] * b[j];
		}
	}
}

static void relu_f32_avx2(const float *in, float *out, int n)
{
	__m256 zero = _mm256_setzero_ps();
	int i = 0;

	for (; i + 8 <= n; i += 8) {
		__m256 v = _mm256_loadu_ps(in + i);
		_mm256_storeu_ps(out + i, _mm256_max_ps(v, zero));
	}
	for (; i < n; i++)
		out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

#endif /* SAM3_HAS_AVX2 */

/* --- Public kernel functions --- */

static enum sam3_error validate_binary_op(const struct sam3_node *node,
					  int *broadcast_n)
{
	if (node->n_inputs < 2 || !node->inputs[0] || !node->inputs[1] ||
	    !node->output) {
		sam3_log_error("elementwise: NULL tensor");
		return SAM3_EINVAL;
	}

	if (node->inputs[0]->dtype != SAM3_DTYPE_F32 ||
	    node->inputs[1]->dtype != SAM3_DTYPE_F32) {
		sam3_log_error("elementwise: unsupported dtype");
		return SAM3_EINVAL;
	}

	int bc = check_broadcast(node->inputs[0], node->inputs[1]);
	if (bc < 0) {
		sam3_log_error("elementwise: shape mismatch");
		return SAM3_EINVAL;
	}

	*broadcast_n = bc;
	return SAM3_OK;
}

enum sam3_error cpu_kernel_add(const struct sam3_node *node)
{
	int broadcast_n;
	enum sam3_error err = validate_binary_op(node, &broadcast_n);
	if (err != SAM3_OK)
		return err;

	const float *a = (const float *)node->inputs[0]->data;
	const float *b = (const float *)node->inputs[1]->data;
	float *out = (float *)node->output->data;
	int n = sam3_tensor_nelems(node->inputs[0]);

#if SAM3_HAS_NEON
	add_f32_neon(a, b, out, n, broadcast_n);
#elif SAM3_HAS_AVX2
	add_f32_avx2(a, b, out, n, broadcast_n);
#else
	add_f32_scalar(a, b, out, n, broadcast_n);
#endif

	return SAM3_OK;
}

enum sam3_error cpu_kernel_mul(const struct sam3_node *node)
{
	int broadcast_n;
	enum sam3_error err = validate_binary_op(node, &broadcast_n);
	if (err != SAM3_OK)
		return err;

	const float *a = (const float *)node->inputs[0]->data;
	const float *b = (const float *)node->inputs[1]->data;
	float *out = (float *)node->output->data;
	int n = sam3_tensor_nelems(node->inputs[0]);

#if SAM3_HAS_NEON
	mul_f32_neon(a, b, out, n, broadcast_n);
#elif SAM3_HAS_AVX2
	mul_f32_avx2(a, b, out, n, broadcast_n);
#else
	mul_f32_scalar(a, b, out, n, broadcast_n);
#endif

	return SAM3_OK;
}

enum sam3_error cpu_kernel_relu(const struct sam3_node *node)
{
	if (!node->inputs[0] || !node->output) {
		sam3_log_error("relu: NULL tensor");
		return SAM3_EINVAL;
	}

	if (node->inputs[0]->dtype != SAM3_DTYPE_F32) {
		sam3_log_error("relu: unsupported dtype");
		return SAM3_EINVAL;
	}

	const float *in = (const float *)node->inputs[0]->data;
	float *out = (float *)node->output->data;
	int n = sam3_tensor_nelems(node->inputs[0]);

#if SAM3_HAS_NEON
	relu_f32_neon(in, out, n);
#elif SAM3_HAS_AVX2
	relu_f32_avx2(in, out, n);
#else
	relu_f32_scalar(in, out, n);
#endif

	return SAM3_OK;
}
