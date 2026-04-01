/*
 * src/backend/cpu/kernels/cpu_transpose.c - 2D transpose kernel
 *
 * Transposes a 2D matrix [rows, cols] -> [cols, rows] by copying.
 * NEON/AVX2 paths use block transpose for cache efficiency (f32 only).
 * Non-f32 dtypes use a generic byte-level path via sam3_dtype_size().
 * Always copies data (no lazy stride swap).
 *
 * Key types:  sam3_node, sam3_tensor
 * Depends on: cpu_kernels.h, cpu_simd.h, core/tensor.h, util/threadpool.h
 * Used by:    cpu_backend.c (dispatch)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <string.h>

#include "cpu_kernels.h"
#include "cpu_simd.h"
#include "core/tensor.h"
#include "util/log.h"
#include "util/threadpool.h"

/* --- Scalar path (when no SIMD available) --- */

#if !SAM3_HAS_NEON && !SAM3_HAS_AVX2

static void transpose_f32_scalar(const float *in, float *out,
				 int rows, int cols,
				 int row_start, int row_end)
{
	for (int i = row_start; i < row_end; i++)
		for (int j = 0; j < cols; j++)
			out[j * rows + i] = in[i * cols + j];
}

#endif /* !SAM3_HAS_NEON && !SAM3_HAS_AVX2 */

/* --- Generic byte-level path (any dtype) --- */

static void transpose_generic(const void *in, void *out,
			       int rows, int cols, size_t elem_size,
			       int row_start, int row_end)
{
	const unsigned char *src = (const unsigned char *)in;
	unsigned char *dst = (unsigned char *)out;

	for (int i = row_start; i < row_end; i++) {
		for (int j = 0; j < cols; j++) {
			memcpy(dst + ((size_t)j * rows + i) * elem_size,
			       src + ((size_t)i * cols + j) * elem_size,
			       elem_size);
		}
	}
}

/* --- NEON path --- */

#if SAM3_HAS_NEON

static void transpose_4x4_neon(const float *in, float *out,
			       int in_stride, int out_stride)
{
	float32x4_t r0 = vld1q_f32(in);
	float32x4_t r1 = vld1q_f32(in + in_stride);
	float32x4_t r2 = vld1q_f32(in + 2 * in_stride);
	float32x4_t r3 = vld1q_f32(in + 3 * in_stride);

	/* Interleave pairs */
	float32x4x2_t t01 = vzipq_f32(r0, r2);
	float32x4x2_t t23 = vzipq_f32(r1, r3);

	float32x4x2_t o0 = vzipq_f32(t01.val[0], t23.val[0]);
	float32x4x2_t o1 = vzipq_f32(t01.val[1], t23.val[1]);

	vst1q_f32(out,                    o0.val[0]);
	vst1q_f32(out + out_stride,       o0.val[1]);
	vst1q_f32(out + 2 * out_stride,   o1.val[0]);
	vst1q_f32(out + 3 * out_stride,   o1.val[1]);
}

static void transpose_f32_neon(const float *in, float *out,
			       int rows, int cols,
			       int row_start, int row_end)
{
	int i = row_start;

	/* Process 4x4 blocks */
	for (; i + 4 <= row_end; i += 4) {
		int j = 0;
		for (; j + 4 <= cols; j += 4)
			transpose_4x4_neon(in + i * cols + j,
					   out + j * rows + i,
					   cols, rows);
		/* Remainder columns */
		for (; j < cols; j++)
			for (int ii = i; ii < i + 4; ii++)
				out[j * rows + ii] = in[ii * cols + j];
	}

	/* Remainder rows */
	for (; i < row_end; i++)
		for (int j = 0; j < cols; j++)
			out[j * rows + i] = in[i * cols + j];
}

#endif /* SAM3_HAS_NEON */

/* --- AVX2 path --- */

#if SAM3_HAS_AVX2

static void transpose_f32_avx2(const float *in, float *out,
			       int rows, int cols,
			       int row_start, int row_end)
{
	/* Use 8x8 blocks for AVX2 */
	int i = row_start;

	for (; i + 8 <= row_end; i += 8) {
		int j = 0;
		for (; j + 8 <= cols; j += 8) {
			/* Load 8 rows of 8 floats */
			__m256 r0 = _mm256_loadu_ps(in + (i + 0) * cols + j);
			__m256 r1 = _mm256_loadu_ps(in + (i + 1) * cols + j);
			__m256 r2 = _mm256_loadu_ps(in + (i + 2) * cols + j);
			__m256 r3 = _mm256_loadu_ps(in + (i + 3) * cols + j);
			__m256 r4 = _mm256_loadu_ps(in + (i + 4) * cols + j);
			__m256 r5 = _mm256_loadu_ps(in + (i + 5) * cols + j);
			__m256 r6 = _mm256_loadu_ps(in + (i + 6) * cols + j);
			__m256 r7 = _mm256_loadu_ps(in + (i + 7) * cols + j);

			/* 8x8 in-register transpose */
			__m256 t0 = _mm256_unpacklo_ps(r0, r1);
			__m256 t1 = _mm256_unpackhi_ps(r0, r1);
			__m256 t2 = _mm256_unpacklo_ps(r2, r3);
			__m256 t3 = _mm256_unpackhi_ps(r2, r3);
			__m256 t4 = _mm256_unpacklo_ps(r4, r5);
			__m256 t5 = _mm256_unpackhi_ps(r4, r5);
			__m256 t6 = _mm256_unpacklo_ps(r6, r7);
			__m256 t7 = _mm256_unpackhi_ps(r6, r7);

			r0 = _mm256_shuffle_ps(t0, t2, 0x44);
			r1 = _mm256_shuffle_ps(t0, t2, 0xEE);
			r2 = _mm256_shuffle_ps(t1, t3, 0x44);
			r3 = _mm256_shuffle_ps(t1, t3, 0xEE);
			r4 = _mm256_shuffle_ps(t4, t6, 0x44);
			r5 = _mm256_shuffle_ps(t4, t6, 0xEE);
			r6 = _mm256_shuffle_ps(t5, t7, 0x44);
			r7 = _mm256_shuffle_ps(t5, t7, 0xEE);

			t0 = _mm256_permute2f128_ps(r0, r4, 0x20);
			t1 = _mm256_permute2f128_ps(r1, r5, 0x20);
			t2 = _mm256_permute2f128_ps(r2, r6, 0x20);
			t3 = _mm256_permute2f128_ps(r3, r7, 0x20);
			t4 = _mm256_permute2f128_ps(r0, r4, 0x31);
			t5 = _mm256_permute2f128_ps(r1, r5, 0x31);
			t6 = _mm256_permute2f128_ps(r2, r6, 0x31);
			t7 = _mm256_permute2f128_ps(r3, r7, 0x31);

			_mm256_storeu_ps(out + (j + 0) * rows + i, t0);
			_mm256_storeu_ps(out + (j + 1) * rows + i, t1);
			_mm256_storeu_ps(out + (j + 2) * rows + i, t2);
			_mm256_storeu_ps(out + (j + 3) * rows + i, t3);
			_mm256_storeu_ps(out + (j + 4) * rows + i, t4);
			_mm256_storeu_ps(out + (j + 5) * rows + i, t5);
			_mm256_storeu_ps(out + (j + 6) * rows + i, t6);
			_mm256_storeu_ps(out + (j + 7) * rows + i, t7);
		}
		for (; j < cols; j++)
			for (int ii = i; ii < i + 8; ii++)
				out[j * rows + ii] = in[ii * cols + j];
	}

	for (; i < row_end; i++)
		for (int j = 0; j < cols; j++)
			out[j * rows + i] = in[i * cols + j];
}

#endif /* SAM3_HAS_AVX2 */

/* --- Parallel dispatch --- */

struct transpose_par_ctx {
	const void *in;
	void       *out;
	int         rows;
	int         cols;
	size_t      elem_size;
};

static void transpose_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct transpose_par_ctx *ctx = (struct transpose_par_ctx *)arg;
	int chunk = ctx->rows / n_tasks;
	int row_start = task_id * chunk;
	int row_end = (task_id == n_tasks - 1) ? ctx->rows : row_start + chunk;

	if (row_start >= row_end)
		return;

	if (ctx->elem_size == 4) {
#if SAM3_HAS_NEON
		transpose_f32_neon((const float *)ctx->in, (float *)ctx->out,
				   ctx->rows, ctx->cols,
				   row_start, row_end);
#elif SAM3_HAS_AVX2
		transpose_f32_avx2((const float *)ctx->in, (float *)ctx->out,
				   ctx->rows, ctx->cols,
				   row_start, row_end);
#else
		transpose_f32_scalar((const float *)ctx->in,
				     (float *)ctx->out,
				     ctx->rows, ctx->cols,
				     row_start, row_end);
#endif
	} else {
		transpose_generic(ctx->in, ctx->out,
				  ctx->rows, ctx->cols, ctx->elem_size,
				  row_start, row_end);
	}
}

enum sam3_error cpu_kernel_transpose(const struct sam3_node *node,
				     struct sam3_threadpool *pool)
{
	if (!node->inputs[0] || !node->output) {
		sam3_log_error("transpose: NULL tensor");
		return SAM3_EINVAL;
	}

	struct sam3_tensor *in = node->inputs[0];
	struct sam3_tensor *out = node->output;

	if (in->n_dims != 2) {
		sam3_log_error("transpose: expected 2D tensor, got %dD",
			       in->n_dims);
		return SAM3_EINVAL;
	}

	struct transpose_par_ctx ctx = {
		.in        = in->data,
		.out       = out->data,
		.rows      = in->dims[0],
		.cols      = in->dims[1],
		.elem_size = sam3_dtype_size(in->dtype),
	};

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

	sam3_threadpool_parallel_for(pool, transpose_parallel_fn, &ctx,
				     n_tasks);

	return SAM3_OK;
}
