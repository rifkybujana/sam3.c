/*
 * src/backend/cpu/kernels/cpu_elementwise.c - Elementwise add, mul, div, relu
 *
 * Implements add, mul, div with [M,N]+[N] broadcasting (bias add pattern),
 * and ReLU activation. Each op has scalar and NEON/AVX2 paths.
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

#include <string.h>

/*
 * Check if b broadcasts onto a.
 * Returns the inner dimension N (for bias-style [N] broadcast),
 * 1 for scalar broadcast, 0 if shapes match exactly, or -1 on error.
 */
static int check_broadcast(const struct sam3_tensor *a,
			    const struct sam3_tensor *b)
{
	int na = sam3_tensor_nelems(a);
	int nb = sam3_tensor_nelems(b);

	if (na == nb)
		return 0;

	/* Scalar broadcast: b has 1 element → multiply all of a */
	if (nb == 1)
		return 1;

	/* Last-dim broadcast: b has as many elements as a's last dim,
	 * and those elements form b's trailing dim. Accepts
	 *   b=[N]         (1-D bias)
	 *   b=[1, N]      (row vector, e.g. prompt_encoder.no_mask_embed)
	 *   b=[1, 1, N]   (interactivity_no_mem_embed)
	 * as long as all non-trailing dims of b are 1 so the tile pattern
	 * matches a straight modulo-N bias add.
	 */
	int last = a->dims[a->n_dims - 1];
	if (nb == last && b->dims[b->n_dims - 1] == last) {
		int leading_ok = 1;
		for (int i = 0; i < b->n_dims - 1; i++) {
			if (b->dims[i] != 1) {
				leading_ok = 0;
				break;
			}
		}
		if (leading_ok)
			return last;
	}

	return -1;
}

/* --- Scalar paths (when no SIMD available) --- */

#if !SAM3_HAS_NEON && !SAM3_HAS_AVX2

static void add_f32_scalar(const float *a, const float *b, float *out,
			   int broadcast_n, int start, int end)
{
	if (broadcast_n <= 0) {
		for (int i = start; i < end; i++)
			out[i] = a[i] + b[i];
	} else {
		for (int i = start; i < end; i++)
			out[i] = a[i] + b[i % broadcast_n];
	}
}

static void mul_f32_scalar(const float *a, const float *b, float *out,
			   int broadcast_n, int start, int end)
{
	if (broadcast_n <= 0) {
		for (int i = start; i < end; i++)
			out[i] = a[i] * b[i];
	} else {
		for (int i = start; i < end; i++)
			out[i] = a[i] * b[i % broadcast_n];
	}
}

static void div_f32_scalar(const float *a, const float *b, float *out,
			   int broadcast_n, int start, int end)
{
	if (broadcast_n <= 0) {
		for (int i = start; i < end; i++)
			out[i] = a[i] / b[i];
	} else {
		for (int i = start; i < end; i++)
			out[i] = a[i] / b[i % broadcast_n];
	}
}

static void relu_f32_scalar(const float *in, float *out,
			    int start, int end)
{
	for (int i = start; i < end; i++)
		out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

#endif /* !SAM3_HAS_NEON && !SAM3_HAS_AVX2 */

/* --- NEON paths --- */

#if SAM3_HAS_NEON

static void add_f32_neon(const float *a, const float *b, float *out,
			 int broadcast_n, int start, int end)
{
	if (broadcast_n <= 0) {
		int i = start;
		for (; i + 4 <= end; i += 4) {
			float32x4_t va = vld1q_f32(a + i);
			float32x4_t vb = vld1q_f32(b + i);
			vst1q_f32(out + i, vaddq_f32(va, vb));
		}
		for (; i < end; i++)
			out[i] = a[i] + b[i];
	} else {
		/* Broadcasting: start/end are row indices */
		for (int r = start; r < end; r++) {
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
			 int broadcast_n, int start, int end)
{
	if (broadcast_n <= 0) {
		int i = start;
		for (; i + 4 <= end; i += 4) {
			float32x4_t va = vld1q_f32(a + i);
			float32x4_t vb = vld1q_f32(b + i);
			vst1q_f32(out + i, vmulq_f32(va, vb));
		}
		for (; i < end; i++)
			out[i] = a[i] * b[i];
	} else {
		for (int r = start; r < end; r++) {
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

static void div_f32_neon(const float *a, const float *b, float *out,
			 int broadcast_n, int start, int end)
{
	if (broadcast_n <= 0) {
		int i = start;
		for (; i + 4 <= end; i += 4) {
			float32x4_t va = vld1q_f32(a + i);
			float32x4_t vb = vld1q_f32(b + i);
			vst1q_f32(out + i, vdivq_f32(va, vb));
		}
		for (; i < end; i++)
			out[i] = a[i] / b[i];
	} else {
		for (int r = start; r < end; r++) {
			int base = r * broadcast_n;
			int j = 0;
			for (; j + 4 <= broadcast_n; j += 4) {
				float32x4_t va = vld1q_f32(a + base + j);
				float32x4_t vb = vld1q_f32(b + j);
				vst1q_f32(out + base + j, vdivq_f32(va, vb));
			}
			for (; j < broadcast_n; j++)
				out[base + j] = a[base + j] / b[j];
		}
	}
}

static void relu_f32_neon(const float *in, float *out,
			  int start, int end)
{
	float32x4_t zero = vdupq_n_f32(0.0f);
	int i = start;

	for (; i + 4 <= end; i += 4) {
		float32x4_t v = vld1q_f32(in + i);
		vst1q_f32(out + i, vmaxq_f32(v, zero));
	}
	for (; i < end; i++)
		out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

#endif /* SAM3_HAS_NEON */

/* --- AVX2 paths --- */

#if SAM3_HAS_AVX2

static void add_f32_avx2(const float *a, const float *b, float *out,
			  int broadcast_n, int start, int end)
{
	if (broadcast_n <= 0) {
		int i = start;
		for (; i + 8 <= end; i += 8) {
			__m256 va = _mm256_loadu_ps(a + i);
			__m256 vb = _mm256_loadu_ps(b + i);
			_mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
		}
		for (; i < end; i++)
			out[i] = a[i] + b[i];
	} else {
		for (int r = start; r < end; r++) {
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
			  int broadcast_n, int start, int end)
{
	if (broadcast_n <= 0) {
		int i = start;
		for (; i + 8 <= end; i += 8) {
			__m256 va = _mm256_loadu_ps(a + i);
			__m256 vb = _mm256_loadu_ps(b + i);
			_mm256_storeu_ps(out + i, _mm256_mul_ps(va, vb));
		}
		for (; i < end; i++)
			out[i] = a[i] * b[i];
	} else {
		for (int r = start; r < end; r++) {
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

static void div_f32_avx2(const float *a, const float *b, float *out,
			  int broadcast_n, int start, int end)
{
	if (broadcast_n <= 0) {
		int i = start;
		for (; i + 8 <= end; i += 8) {
			__m256 va = _mm256_loadu_ps(a + i);
			__m256 vb = _mm256_loadu_ps(b + i);
			_mm256_storeu_ps(out + i, _mm256_div_ps(va, vb));
		}
		for (; i < end; i++)
			out[i] = a[i] / b[i];
	} else {
		for (int r = start; r < end; r++) {
			int base = r * broadcast_n;
			int j = 0;
			for (; j + 8 <= broadcast_n; j += 8) {
				__m256 va = _mm256_loadu_ps(a + base + j);
				__m256 vb = _mm256_loadu_ps(b + j);
				_mm256_storeu_ps(out + base + j,
						 _mm256_div_ps(va, vb));
			}
			for (; j < broadcast_n; j++)
				out[base + j] = a[base + j] / b[j];
		}
	}
}

static void relu_f32_avx2(const float *in, float *out,
			  int start, int end)
{
	__m256 zero = _mm256_setzero_ps();
	int i = start;

	for (; i + 8 <= end; i += 8) {
		__m256 v = _mm256_loadu_ps(in + i);
		_mm256_storeu_ps(out + i, _mm256_max_ps(v, zero));
	}
	for (; i < end; i++)
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

/* --- Parallel dispatch contexts --- */

struct binop_par_ctx {
	const float *a;
	const float *b;
	float       *out;
	int          n;
	int          broadcast_n;
};

static void add_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct binop_par_ctx *ctx = (struct binop_par_ctx *)arg;
	int total, start, end;

	if (ctx->broadcast_n <= 0) {
		total = ctx->n;
	} else {
		total = ctx->n / ctx->broadcast_n;
	}

	int chunk = total / n_tasks;
	start = task_id * chunk;
	end = (task_id == n_tasks - 1) ? total : start + chunk;

	if (start >= end)
		return;

#if SAM3_HAS_NEON
	add_f32_neon(ctx->a, ctx->b, ctx->out,
		     ctx->broadcast_n, start, end);
#elif SAM3_HAS_AVX2
	add_f32_avx2(ctx->a, ctx->b, ctx->out,
		     ctx->broadcast_n, start, end);
#else
	add_f32_scalar(ctx->a, ctx->b, ctx->out,
		       ctx->broadcast_n, start, end);
#endif
}

static void mul_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct binop_par_ctx *ctx = (struct binop_par_ctx *)arg;
	int total, start, end;

	if (ctx->broadcast_n <= 0) {
		total = ctx->n;
	} else {
		total = ctx->n / ctx->broadcast_n;
	}

	int chunk = total / n_tasks;
	start = task_id * chunk;
	end = (task_id == n_tasks - 1) ? total : start + chunk;

	if (start >= end)
		return;

#if SAM3_HAS_NEON
	mul_f32_neon(ctx->a, ctx->b, ctx->out,
		     ctx->broadcast_n, start, end);
#elif SAM3_HAS_AVX2
	mul_f32_avx2(ctx->a, ctx->b, ctx->out,
		     ctx->broadcast_n, start, end);
#else
	mul_f32_scalar(ctx->a, ctx->b, ctx->out,
		       ctx->broadcast_n, start, end);
#endif
}

static void div_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct binop_par_ctx *ctx = (struct binop_par_ctx *)arg;
	int total, start, end;

	if (ctx->broadcast_n <= 0) {
		total = ctx->n;
	} else {
		total = ctx->n / ctx->broadcast_n;
	}

	int chunk = total / n_tasks;
	start = task_id * chunk;
	end = (task_id == n_tasks - 1) ? total : start + chunk;

	if (start >= end)
		return;

#if SAM3_HAS_NEON
	div_f32_neon(ctx->a, ctx->b, ctx->out,
		     ctx->broadcast_n, start, end);
#elif SAM3_HAS_AVX2
	div_f32_avx2(ctx->a, ctx->b, ctx->out,
		     ctx->broadcast_n, start, end);
#else
	div_f32_scalar(ctx->a, ctx->b, ctx->out,
		       ctx->broadcast_n, start, end);
#endif
}

struct relu_par_ctx {
	const float *in;
	float       *out;
	int          n;
};

static void relu_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct relu_par_ctx *ctx = (struct relu_par_ctx *)arg;
	int chunk = ctx->n / n_tasks;
	int start = task_id * chunk;
	int end = (task_id == n_tasks - 1) ? ctx->n : start + chunk;

	if (start >= end)
		return;

#if SAM3_HAS_NEON
	relu_f32_neon(ctx->in, ctx->out, start, end);
#elif SAM3_HAS_AVX2
	relu_f32_avx2(ctx->in, ctx->out, start, end);
#else
	relu_f32_scalar(ctx->in, ctx->out, start, end);
#endif
}

enum sam3_error cpu_kernel_add(const struct sam3_node *node,
			       struct sam3_threadpool *pool)
{
	int broadcast_n;
	enum sam3_error err = validate_binary_op(node, &broadcast_n);
	if (err != SAM3_OK)
		return err;

	struct binop_par_ctx ctx = {
		.a           = (const float *)node->inputs[0]->data,
		.b           = (const float *)node->inputs[1]->data,
		.out         = (float *)node->output->data,
		.n           = sam3_tensor_nelems(node->inputs[0]),
		.broadcast_n = broadcast_n,
	};

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

	sam3_threadpool_parallel_for(pool, add_parallel_fn, &ctx, n_tasks);

	return SAM3_OK;
}

enum sam3_error cpu_kernel_mul(const struct sam3_node *node,
			       struct sam3_threadpool *pool)
{
	int broadcast_n;
	enum sam3_error err = validate_binary_op(node, &broadcast_n);
	if (err != SAM3_OK)
		return err;

	struct binop_par_ctx ctx = {
		.a           = (const float *)node->inputs[0]->data,
		.b           = (const float *)node->inputs[1]->data,
		.out         = (float *)node->output->data,
		.n           = sam3_tensor_nelems(node->inputs[0]),
		.broadcast_n = broadcast_n,
	};

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

	sam3_threadpool_parallel_for(pool, mul_parallel_fn, &ctx, n_tasks);

	return SAM3_OK;
}

enum sam3_error cpu_kernel_div(const struct sam3_node *node,
			       struct sam3_threadpool *pool)
{
	int broadcast_n;
	enum sam3_error err = validate_binary_op(node, &broadcast_n);
	if (err != SAM3_OK)
		return err;

	struct binop_par_ctx ctx = {
		.a           = (const float *)node->inputs[0]->data,
		.b           = (const float *)node->inputs[1]->data,
		.out         = (float *)node->output->data,
		.n           = sam3_tensor_nelems(node->inputs[0]),
		.broadcast_n = broadcast_n,
	};

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

	sam3_threadpool_parallel_for(pool, div_parallel_fn, &ctx, n_tasks);

	return SAM3_OK;
}

enum sam3_error cpu_kernel_relu(const struct sam3_node *node,
				struct sam3_threadpool *pool)
{
	if (!node->inputs[0] || !node->output) {
		sam3_log_error("relu: NULL tensor");
		return SAM3_EINVAL;
	}

	if (node->inputs[0]->dtype != SAM3_DTYPE_F32) {
		sam3_log_error("relu: unsupported dtype");
		return SAM3_EINVAL;
	}

	struct relu_par_ctx ctx = {
		.in  = (const float *)node->inputs[0]->data,
		.out = (float *)node->output->data,
		.n   = sam3_tensor_nelems(node->inputs[0]),
	};

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

	sam3_threadpool_parallel_for(pool, relu_parallel_fn, &ctx, n_tasks);

	return SAM3_OK;
}
