/*
 * src/backend/cpu/kernels/cpu_conv2d.c - Conv2D via im2col + matmul
 *
 * Implements 2D convolution using the im2col approach: unroll input
 * patches into a column matrix, then matrix-multiply with the weight
 * matrix. Scratch arena is used for the im2col buffer with save/restore
 * of offset so the temp buffer is freed automatically.
 *
 * Layout: input [N,C,H,W], weight [OC,C,KH,KW], output [N,OC,OH,OW].
 * node->params[0]=stride, params[1]=padding.
 *
 * Key types:  sam3_node, sam3_tensor, sam3_arena
 * Depends on: cpu_kernels.h, core/tensor.h, core/alloc.h
 * Used by:    cpu_backend.c (dispatch)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "cpu_kernels.h"
#include "core/tensor.h"
#include "core/alloc.h"
#include "util/log.h"
#include "util/threadpool.h"

#include <string.h>

/*
 * im2col - Unroll input patches into a column matrix.
 *
 * Input: [C, H, W]
 * Output: [C*KH*KW, OH*OW] (column-major patches)
 */
static void im2col_f32(const float *in, float *col,
		       int C, int H, int W,
		       int KH, int KW,
		       int stride, int pad,
		       int OH, int OW)
{
	int col_rows = C * KH * KW;
	int col_cols = OH * OW;

	for (int c = 0; c < C; c++) {
		for (int kh = 0; kh < KH; kh++) {
			for (int kw = 0; kw < KW; kw++) {
				int col_row = c * KH * KW + kh * KW + kw;
				for (int oh = 0; oh < OH; oh++) {
					for (int ow = 0; ow < OW; ow++) {
						int ih = oh * stride + kh - pad;
						int iw = ow * stride + kw - pad;
						int col_idx = col_row * col_cols + oh * OW + ow;
						if (ih >= 0 && ih < H && iw >= 0 && iw < W)
							col[col_idx] = in[c * H * W + ih * W + iw];
						else
							col[col_idx] = 0.0f;
					}
				}
			}
		}
	}

	(void)col_rows;
}

struct conv2d_matmul_ctx {
	const float *a;
	const float *b;
	float       *c;
	int          M;
	int          K;
	int          N;
};

static void conv2d_matmul_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct conv2d_matmul_ctx *ctx = (struct conv2d_matmul_ctx *)arg;
	int chunk = ctx->M / n_tasks;
	int m_start = task_id * chunk;
	int m_end = (task_id == n_tasks - 1) ? ctx->M : m_start + chunk;

	if (m_start >= m_end)
		return;

	memset(ctx->c + (size_t)m_start * ctx->N, 0,
	       (size_t)(m_end - m_start) * ctx->N * sizeof(float));

	for (int i = m_start; i < m_end; i++) {
		for (int k = 0; k < ctx->K; k++) {
			float aik = ctx->a[i * ctx->K + k];
			for (int j = 0; j < ctx->N; j++)
				ctx->c[i * ctx->N + j] += aik * ctx->b[k * ctx->N + j];
		}
	}
}

enum sam3_error cpu_kernel_conv2d(const struct sam3_node *node,
				  struct sam3_arena *scratch,
				  struct sam3_threadpool *pool)
{
	if (node->n_inputs < 2 || !node->inputs[0] || !node->inputs[1] ||
	    !node->output) {
		sam3_log_error("conv2d: NULL tensor");
		return SAM3_EINVAL;
	}

	if (!scratch) {
		sam3_log_error("conv2d: NULL scratch arena");
		return SAM3_EINVAL;
	}

	struct sam3_tensor *input = node->inputs[0];
	struct sam3_tensor *weight = node->inputs[1];
	struct sam3_tensor *output = node->output;

	if (input->dtype != SAM3_DTYPE_F32 || weight->dtype != SAM3_DTYPE_F32) {
		sam3_log_error("conv2d: unsupported dtype");
		return SAM3_EINVAL;
	}

	if (input->n_dims != 4 || weight->n_dims != 4) {
		sam3_log_error("conv2d: expected 4D tensors");
		return SAM3_EINVAL;
	}

	int N_batch = input->dims[0];
	int C = input->dims[1];
	int H = input->dims[2];
	int W = input->dims[3];

	int OC = weight->dims[0];
	int KC = weight->dims[1];
	int KH = weight->dims[2];
	int KW = weight->dims[3];

	if (KC != C) {
		sam3_log_error("conv2d: channel mismatch %d != %d", KC, C);
		return SAM3_EINVAL;
	}

	int stride = node->params[0] > 0 ? node->params[0] : 1;
	int pad = node->params[1];

	int OH = (H + 2 * pad - KH) / stride + 1;
	int OW = (W + 2 * pad - KW) / stride + 1;

	/* Save scratch offset for restore */
	size_t saved_offset = scratch->offset;

	/* Allocate im2col buffer: [C*KH*KW, OH*OW] */
	size_t col_size = (size_t)(C * KH * KW) * (OH * OW) * sizeof(float);
	float *col = (float *)sam3_arena_alloc(scratch, col_size);
	if (!col) {
		sam3_log_error("conv2d: scratch OOM (%zu bytes)", col_size);
		scratch->offset = saved_offset;
		return SAM3_ENOMEM;
	}

	const float *w_data = (const float *)weight->data;

	for (int n = 0; n < N_batch; n++) {
		const float *in_n = (const float *)input->data +
				    n * C * H * W;
		float *out_n = (float *)output->data +
			       n * OC * OH * OW;

		im2col_f32(in_n, col, C, H, W, KH, KW, stride, pad, OH, OW);

		/* Parallel matmul: weight [OC, C*KH*KW] @ col -> out [OC, OH*OW] */
		struct conv2d_matmul_ctx mctx = {
			.a = w_data, .b = col, .c = out_n,
			.M = OC, .K = C * KH * KW, .N = OH * OW,
		};
		int n_tasks = sam3_threadpool_n_threads(pool);
		if (n_tasks < 1)
			n_tasks = 1;
		sam3_threadpool_parallel_for(pool, conv2d_matmul_parallel_fn,
					     &mctx, n_tasks);
	}

	/* Restore scratch offset — frees the im2col buffer */
	scratch->offset = saved_offset;

	return SAM3_OK;
}
