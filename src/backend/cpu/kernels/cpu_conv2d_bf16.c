/*
 * src/backend/cpu/kernels/cpu_conv2d_bf16.c - BF16 Conv2D via im2col + matmul
 *
 * 2D convolution for bf16 tensors using im2col + matrix multiply. All
 * arithmetic is done in f32 since bf16 has no native arithmetic. NEON
 * path uses 4-wide float32x4_t with bf16<->f32 conversion helpers from
 * core/half.h. Scratch arena for im2col buffer.
 *
 * Key types:  sam3_node, sam3_tensor, sam3_arena
 * Depends on: cpu_kernels.h, cpu_simd.h, core/half.h, core/tensor.h, core/alloc.h
 * Used by:    cpu_dispatch.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "cpu_kernels.h"
#include "cpu_simd.h"
#include "core/half.h"
#include "core/tensor.h"
#include "core/alloc.h"
#include "util/log.h"
#include "util/threadpool.h"

#include <string.h>

/*
 * im2col_bf16 - Unroll bf16 input patches into a column matrix.
 *
 * Input:  [C, H, W] as uint16_t*
 * Output: [C*KH*KW, OH*OW] column-major patches, kept as bf16.
 */
static void im2col_bf16(const uint16_t *in, uint16_t *col,
			 int C, int H, int W, int KH, int KW,
			 int stride, int pad, int OH, int OW)
{
	int col_cols = OH * OW;
	uint16_t zero_bf16 = f32_to_bf16(0.0f);

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
							col[idx] = zero_bf16;
					}
				}
			}
		}
	}
}

/* --- NEON bf16 matmul path (f32 arithmetic, 4-wide) --- */

#if SAM3_HAS_NEON

struct conv2d_matmul_ctx_bf16 {
	const uint16_t *a;  /* weight [OC, C*KH*KW] */
	const uint16_t *b;  /* col    [C*KH*KW, OH*OW] */
	uint16_t       *c;  /* output [OC, OH*OW] */
	int             M;
	int             K;
	int             N;
};

static void conv2d_matmul_bf16_fn(void *arg, int task_id, int n_tasks)
{
	struct conv2d_matmul_ctx_bf16 *ctx =
		(struct conv2d_matmul_ctx_bf16 *)arg;
	int chunk   = ctx->M / n_tasks;
	int m_start = task_id * chunk;
	int m_end   = (task_id == n_tasks - 1) ? ctx->M : m_start + chunk;

	if (m_start >= m_end)
		return;

	/* Zero output — uint16_t zero is bf16 +0.0 */
	memset(ctx->c + (size_t)m_start * ctx->N, 0,
	       (size_t)(m_end - m_start) * ctx->N * sizeof(uint16_t));

	for (int i = m_start; i < m_end; i++) {
		for (int k = 0; k < ctx->K; k++) {
			float aik = bf16_to_f32(ctx->a[i * ctx->K + k]);
			float32x4_t va = vdupq_n_f32(aik);
			int j = 0;
			for (; j + 4 <= ctx->N; j += 4) {
				float32x4_t vc = bf16x4_to_f32x4(
					ctx->c + i * ctx->N + j);
				float32x4_t vb = bf16x4_to_f32x4(
					ctx->b + k * ctx->N + j);
				f32x4_to_bf16x4(
					ctx->c + i * ctx->N + j,
					vfmaq_f32(vc, va, vb));
			}
			for (; j < ctx->N; j++) {
				float cur = bf16_to_f32(
					ctx->c[i * ctx->N + j]);
				float bkj = bf16_to_f32(
					ctx->b[k * ctx->N + j]);
				ctx->c[i * ctx->N + j] =
					f32_to_bf16(cur + aik * bkj);
			}
		}
	}
}

#else /* !SAM3_HAS_NEON */

/* --- Scalar fallback path --- */

struct conv2d_matmul_ctx_bf16 {
	const uint16_t *a;  /* weight [OC, C*KH*KW] */
	const uint16_t *b;  /* col    [C*KH*KW, OH*OW] */
	uint16_t       *c;  /* output [OC, OH*OW] */
	int             M;
	int             K;
	int             N;
};

static void conv2d_matmul_bf16_fn(void *arg, int task_id, int n_tasks)
{
	struct conv2d_matmul_ctx_bf16 *ctx =
		(struct conv2d_matmul_ctx_bf16 *)arg;
	int chunk   = ctx->M / n_tasks;
	int m_start = task_id * chunk;
	int m_end   = (task_id == n_tasks - 1) ? ctx->M : m_start + chunk;

	if (m_start >= m_end)
		return;

	/* Zero output — uint16_t zero is bf16 +0.0 */
	memset(ctx->c + (size_t)m_start * ctx->N, 0,
	       (size_t)(m_end - m_start) * ctx->N * sizeof(uint16_t));

	for (int i = m_start; i < m_end; i++) {
		for (int k = 0; k < ctx->K; k++) {
			float aik = bf16_to_f32(ctx->a[i * ctx->K + k]);
			for (int j = 0; j < ctx->N; j++) {
				float cur = bf16_to_f32(
					ctx->c[i * ctx->N + j]);
				float bkj = bf16_to_f32(
					ctx->b[k * ctx->N + j]);
				ctx->c[i * ctx->N + j] =
					f32_to_bf16(cur + aik * bkj);
			}
		}
	}
}

#endif /* SAM3_HAS_NEON */

/*
 * cpu_kernel_conv2d_bf16 - BF16 Conv2D via im2col + matmul.
 *
 * @node:    Node with n_inputs>=2: input [N,C,H,W] and weight [OC,C,KH,KW],
 *           both SAM3_DTYPE_BF16. node->params[0]=stride, params[1]=padding.
 * @scratch: Scratch arena for im2col temp buffer. Offset is saved/restored.
 * @pool:    Thread pool for parallel matmul over output channels.
 *
 * Returns SAM3_OK on success, SAM3_EINVAL on bad inputs, SAM3_ENOMEM if
 * the scratch arena is too small.
 */
enum sam3_error cpu_kernel_conv2d_bf16(const struct sam3_node *node,
				       struct sam3_arena *scratch,
				       struct sam3_threadpool *pool)
{
	if (node->n_inputs < 2 || !node->inputs[0] || !node->inputs[1] ||
	    !node->output) {
		sam3_log_error("conv2d_bf16: NULL tensor");
		return SAM3_EINVAL;
	}

	if (!scratch) {
		sam3_log_error("conv2d_bf16: NULL scratch arena");
		return SAM3_EINVAL;
	}

	struct sam3_tensor *input  = node->inputs[0];
	struct sam3_tensor *weight = node->inputs[1];
	struct sam3_tensor *output = node->output;

	if (input->dtype != SAM3_DTYPE_BF16 ||
	    weight->dtype != SAM3_DTYPE_BF16) {
		sam3_log_error("conv2d_bf16: unsupported dtype");
		return SAM3_EINVAL;
	}

	if (input->n_dims != 4 || weight->n_dims != 4) {
		sam3_log_error("conv2d_bf16: expected 4D tensors");
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
		sam3_log_error("conv2d_bf16: channel mismatch %d != %d",
			       KC, C);
		return SAM3_EINVAL;
	}

	int stride = node->params[0] > 0 ? node->params[0] : 1;
	int pad    = node->params[1];

	int OH = (H + 2 * pad - KH) / stride + 1;
	int OW = (W + 2 * pad - KW) / stride + 1;

	/* Save scratch offset for restore */
	size_t saved_offset = scratch->offset;

	/* Allocate im2col buffer: [C*KH*KW, OH*OW] in bf16 */
	size_t col_size = (size_t)(C * KH * KW) * (OH * OW)
			  * sizeof(uint16_t);
	void *col_buf = sam3_arena_alloc(scratch, col_size);
	if (!col_buf) {
		sam3_log_error("conv2d_bf16: scratch OOM (%zu bytes)",
			       col_size);
		scratch->offset = saved_offset;
		return SAM3_ENOMEM;
	}

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

	const uint16_t *w_data = (const uint16_t *)weight->data;
	uint16_t       *col    = (uint16_t *)col_buf;

	for (int n = 0; n < N_batch; n++) {
		const uint16_t *in_n = (const uint16_t *)input->data
				       + (size_t)n * C * H * W;
		uint16_t *out_n = (uint16_t *)output->data
				  + (size_t)n * OC * OH * OW;

		im2col_bf16(in_n, col, C, H, W, KH, KW,
			    stride, pad, OH, OW);

		struct conv2d_matmul_ctx_bf16 mctx = {
			.a = w_data,
			.b = col,
			.c = out_n,
			.M = OC,
			.K = C * KH * KW,
			.N = OH * OW,
		};
		sam3_threadpool_parallel_for(pool, conv2d_matmul_bf16_fn,
					     &mctx, n_tasks);
	}

	/* Restore scratch offset — frees the im2col buffer */
	scratch->offset = saved_offset;

	return SAM3_OK;
}
