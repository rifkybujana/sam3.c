/*
 * src/backend/cpu/kernels/cpu_conv2d_f16.c - FP16 Conv2D via im2col + matmul
 *
 * 2D convolution for fp16 tensors using im2col + matrix multiply. NEON fp16
 * path uses native float16x8_t arithmetic. Scalar fallback upcasts to f32.
 * Scratch arena for im2col buffer.
 *
 * Key types:  sam3_node, sam3_tensor, sam3_arena
 * Depends on: cpu_kernels.h, cpu_simd_f16.h, core/half.h, core/tensor.h, core/alloc.h
 * Used by:    cpu_dispatch.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "cpu_kernels.h"
#include "cpu_simd_f16.h"
#include "core/half.h"
#include "core/tensor.h"
#include "core/alloc.h"
#include "util/log.h"
#include "util/threadpool.h"

#include <string.h>

/* ── NEON fp16 path ─────────────────────────────────────────────────── */

#if SAM3_HAS_NEON_FP16

/*
 * im2col_f16 - Unroll fp16 input patches into a column matrix.
 *
 * Input:  [C, H, W] as _Float16*
 * Output: [C*KH*KW, OH*OW] column-major patches
 */
static void im2col_f16(const _Float16 *in, _Float16 *col,
		       int C, int H, int W, int KH, int KW,
		       int stride, int pad, int OH, int OW)
{
	int col_cols = OH * OW;

	for (int c = 0; c < C; c++) {
		for (int kh = 0; kh < KH; kh++) {
			for (int kw = 0; kw < KW; kw++) {
				int col_row = c * KH * KW + kh * KW + kw;
				for (int oh = 0; oh < OH; oh++) {
					for (int ow = 0; ow < OW; ow++) {
						int ih = oh * stride + kh - pad;
						int iw = ow * stride + kw - pad;
						int idx = col_row * col_cols
							  + oh * OW + ow;
						if (ih >= 0 && ih < H &&
						    iw >= 0 && iw < W)
							col[idx] = in[
								c * H * W
								+ ih * W + iw];
						else
							col[idx] =
								(_Float16)0.0f;
					}
				}
			}
		}
	}
}

struct conv2d_matmul_ctx_f16 {
	const _Float16 *a;  /* weight [OC, C*KH*KW] */
	const _Float16 *b;  /* col    [C*KH*KW, OH*OW] */
	_Float16       *c;  /* output [OC, OH*OW] */
	int             M;
	int             K;
	int             N;
};

static void conv2d_matmul_f16_fn(void *arg, int task_id, int n_tasks)
{
	struct conv2d_matmul_ctx_f16 *ctx =
		(struct conv2d_matmul_ctx_f16 *)arg;
	int chunk   = ctx->M / n_tasks;
	int m_start = task_id * chunk;
	int m_end   = (task_id == n_tasks - 1) ? ctx->M : m_start + chunk;

	if (m_start >= m_end)
		return;

	memset(ctx->c + (size_t)m_start * ctx->N, 0,
	       (size_t)(m_end - m_start) * ctx->N * sizeof(_Float16));

	for (int i = m_start; i < m_end; i++) {
		for (int k = 0; k < ctx->K; k++) {
			float16x8_t va = vdupq_n_f16(ctx->a[i * ctx->K + k]);
			int j = 0;
			for (; j + 8 <= ctx->N; j += 8) {
				float16x8_t vc = vld1q_f16(
					(const __fp16 *)(ctx->c + i * ctx->N
							 + j));
				float16x8_t vb = vld1q_f16(
					(const __fp16 *)(ctx->b + k * ctx->N
							 + j));
				vst1q_f16((__fp16 *)(ctx->c + i * ctx->N + j),
					  vfmaq_f16(vc, va, vb));
			}
			_Float16 aik = ctx->a[i * ctx->K + k];
			for (; j < ctx->N; j++)
				ctx->c[i * ctx->N + j] +=
					aik * ctx->b[k * ctx->N + j];
		}
	}
}

#else /* !SAM3_HAS_NEON_FP16 */

/* ── Scalar fallback path ───────────────────────────────────────────── */

/*
 * im2col_f16_scalar - Unroll fp16 input patches (scalar, via uint16_t).
 *
 * Input:  [C, H, W] as uint16_t*
 * Output: [C*KH*KW, OH*OW] column-major patches
 */
static void im2col_f16_scalar(const uint16_t *in, uint16_t *col,
			      int C, int H, int W, int KH, int KW,
			      int stride, int pad, int OH, int OW)
{
	int col_cols = OH * OW;

	for (int c = 0; c < C; c++) {
		for (int kh = 0; kh < KH; kh++) {
			for (int kw = 0; kw < KW; kw++) {
				int col_row = c * KH * KW + kh * KW + kw;
				for (int oh = 0; oh < OH; oh++) {
					for (int ow = 0; ow < OW; ow++) {
						int ih = oh * stride + kh - pad;
						int iw = ow * stride + kw - pad;
						int idx = col_row * col_cols
							  + oh * OW + ow;
						if (ih >= 0 && ih < H &&
						    iw >= 0 && iw < W)
							col[idx] = in[
								c * H * W
								+ ih * W + iw];
						else
							col[idx] =
								f32_to_fp16(0.0f);
					}
				}
			}
		}
	}
}

struct conv2d_matmul_ctx_f16 {
	const uint16_t *a;  /* weight [OC, C*KH*KW] */
	const uint16_t *b;  /* col    [C*KH*KW, OH*OW] */
	uint16_t       *c;  /* output [OC, OH*OW] */
	int             M;
	int             K;
	int             N;
};

static void conv2d_matmul_f16_fn(void *arg, int task_id, int n_tasks)
{
	struct conv2d_matmul_ctx_f16 *ctx =
		(struct conv2d_matmul_ctx_f16 *)arg;
	int chunk   = ctx->M / n_tasks;
	int m_start = task_id * chunk;
	int m_end   = (task_id == n_tasks - 1) ? ctx->M : m_start + chunk;

	if (m_start >= m_end)
		return;

	/* Zero output — uint16_t zero is fp16 +0.0 */
	memset(ctx->c + (size_t)m_start * ctx->N, 0,
	       (size_t)(m_end - m_start) * ctx->N * sizeof(uint16_t));

	for (int i = m_start; i < m_end; i++) {
		for (int k = 0; k < ctx->K; k++) {
			float aik = fp16_to_f32(ctx->a[i * ctx->K + k]);
			for (int j = 0; j < ctx->N; j++) {
				float cur = fp16_to_f32(
					ctx->c[i * ctx->N + j]);
				float bkj = fp16_to_f32(
					ctx->b[k * ctx->N + j]);
				ctx->c[i * ctx->N + j] =
					f32_to_fp16(cur + aik * bkj);
			}
		}
	}
}

#endif /* SAM3_HAS_NEON_FP16 */

/*
 * cpu_kernel_conv2d_f16 - FP16 Conv2D via im2col + matmul.
 *
 * @node:    Node with n_inputs>=2: input [N,C,H,W] and weight [OC,C,KH,KW],
 *           both SAM3_DTYPE_F16. node->params[0]=stride, params[1]=padding.
 * @scratch: Scratch arena for im2col temp buffer. Offset is saved/restored.
 * @pool:    Thread pool for parallel matmul over output channels.
 *
 * Returns SAM3_OK on success, SAM3_EINVAL on bad inputs, SAM3_ENOMEM if
 * the scratch arena is too small.
 */
enum sam3_error cpu_kernel_conv2d_f16(const struct sam3_node *node,
				      struct sam3_arena *scratch,
				      struct sam3_threadpool *pool)
{
	if (node->n_inputs < 2 || !node->inputs[0] || !node->inputs[1] ||
	    !node->output) {
		sam3_log_error("conv2d_f16: NULL tensor");
		return SAM3_EINVAL;
	}

	if (!scratch) {
		sam3_log_error("conv2d_f16: NULL scratch arena");
		return SAM3_EINVAL;
	}

	struct sam3_tensor *input  = node->inputs[0];
	struct sam3_tensor *weight = node->inputs[1];
	struct sam3_tensor *output = node->output;

	if (input->dtype != SAM3_DTYPE_F16 ||
	    weight->dtype != SAM3_DTYPE_F16) {
		sam3_log_error("conv2d_f16: unsupported dtype");
		return SAM3_EINVAL;
	}

	if (input->n_dims != 4 || weight->n_dims != 4) {
		sam3_log_error("conv2d_f16: expected 4D tensors");
		return SAM3_EINVAL;
	}

	int N_batch = input->dims[0];
	int C       = input->dims[1];
	int H       = input->dims[2];
	int W       = input->dims[3];

	int OC  = weight->dims[0];
	int KC  = weight->dims[1];
	int KH  = weight->dims[2];
	int KW  = weight->dims[3];

	if (KC != C) {
		sam3_log_error("conv2d_f16: channel mismatch %d != %d", KC, C);
		return SAM3_EINVAL;
	}

	int stride = node->params[0] > 0 ? node->params[0] : 1;
	int pad    = node->params[1];

	int OH = (H + 2 * pad - KH) / stride + 1;
	int OW = (W + 2 * pad - KW) / stride + 1;

	/* Save scratch offset for restore */
	size_t saved_offset = scratch->offset;

	/* Allocate im2col buffer: [C*KH*KW, OH*OW] in fp16 */
	size_t col_size = (size_t)(C * KH * KW) * (OH * OW) * sizeof(uint16_t);
	void *col_buf = sam3_arena_alloc(scratch, col_size);
	if (!col_buf) {
		sam3_log_error("conv2d_f16: scratch OOM (%zu bytes)", col_size);
		scratch->offset = saved_offset;
		return SAM3_ENOMEM;
	}

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

#if SAM3_HAS_NEON_FP16
	const _Float16 *w_data = (const _Float16 *)weight->data;
	_Float16       *col    = (_Float16 *)col_buf;

	for (int n = 0; n < N_batch; n++) {
		const _Float16 *in_n = (const _Float16 *)input->data
				       + (size_t)n * C * H * W;
		_Float16 *out_n = (_Float16 *)output->data
				  + (size_t)n * OC * OH * OW;

		im2col_f16(in_n, col, C, H, W, KH, KW,
			   stride, pad, OH, OW);

		struct conv2d_matmul_ctx_f16 mctx = {
			.a = w_data,
			.b = col,
			.c = out_n,
			.M = OC,
			.K = C * KH * KW,
			.N = OH * OW,
		};
		sam3_threadpool_parallel_for(pool, conv2d_matmul_f16_fn,
					     &mctx, n_tasks);
	}
#else
	const uint16_t *w_data = (const uint16_t *)weight->data;
	uint16_t       *col    = (uint16_t *)col_buf;

	for (int n = 0; n < N_batch; n++) {
		const uint16_t *in_n = (const uint16_t *)input->data
				       + (size_t)n * C * H * W;
		uint16_t *out_n = (uint16_t *)output->data
				  + (size_t)n * OC * OH * OW;

		im2col_f16_scalar(in_n, col, C, H, W, KH, KW,
				  stride, pad, OH, OW);

		struct conv2d_matmul_ctx_f16 mctx = {
			.a = w_data,
			.b = col,
			.c = out_n,
			.M = OC,
			.K = C * KH * KW,
			.N = OH * OW,
		};
		sam3_threadpool_parallel_for(pool, conv2d_matmul_f16_fn,
					     &mctx, n_tasks);
	}
#endif

	/* Restore scratch offset — frees the im2col buffer */
	scratch->offset = saved_offset;

	return SAM3_OK;
}
