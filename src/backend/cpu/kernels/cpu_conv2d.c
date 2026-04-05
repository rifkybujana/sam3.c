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
 * Copyright (c) 2026 Rifky Bujana Bisri
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
 *
 * Splits oh/ow iteration into boundary (needs padding) and interior
 * (no bounds check) regions.  Interior rows with stride=1 use memcpy.
 */
static void im2col_f32(const float *in, float *col,
		       int C, int H, int W,
		       int KH, int KW,
		       int stride, int pad,
		       int OH, int OW)
{
	int col_cols = OH * OW;

	for (int c = 0; c < C; c++) {
		const float *in_c = in + c * H * W;
		for (int kh = 0; kh < KH; kh++) {
			for (int kw = 0; kw < KW; kw++) {
				int col_row = (c * KH + kh) * KW + kw;
				float *dst_row = col + col_row * col_cols;

				/* oh range where ih = oh*stride+kh-pad is in [0,H) */
				int oh_start = 0;
				int oh_end = OH;
				if (kh - pad < 0)
					oh_start = (-kh + pad + stride - 1) / stride;
				if (kh - pad + (OH - 1) * stride >= H)
					oh_end = (H - kh + pad + stride - 1) / stride;
				if (oh_end > OH)
					oh_end = OH;
				if (oh_start > OH)
					oh_start = OH;

				/* ow range where iw = ow*stride+kw-pad is in [0,W) */
				int ow_start = 0;
				int ow_end = OW;
				if (kw - pad < 0)
					ow_start = (-kw + pad + stride - 1) / stride;
				if (kw - pad + (OW - 1) * stride >= W)
					ow_end = (W - kw + pad + stride - 1) / stride;
				if (ow_end > OW)
					ow_end = OW;
				if (ow_start > OW)
					ow_start = OW;

				/* Top boundary rows: all zeros */
				for (int oh = 0; oh < oh_start; oh++)
					memset(dst_row + oh * OW, 0,
					       (size_t)OW * sizeof(float));

				/* Interior rows */
				for (int oh = oh_start; oh < oh_end; oh++) {
					int ih = oh * stride + kh - pad;
					float *dst = dst_row + oh * OW;

					/* Left pad */
					if (ow_start > 0)
						memset(dst, 0,
						       (size_t)ow_start * sizeof(float));

					/* Interior: no bounds check needed */
					int iw_base = ow_start * stride + kw - pad;
					if (stride == 1) {
						memcpy(dst + ow_start,
						       in_c + ih * W + iw_base,
						       (size_t)(ow_end - ow_start) * sizeof(float));
					} else {
						for (int ow = ow_start; ow < ow_end; ow++) {
							int iw = ow * stride + kw - pad;
							dst[ow] = in_c[ih * W + iw];
						}
					}

					/* Right pad */
					if (ow_end < OW)
						memset(dst + ow_end, 0,
						       (size_t)(OW - ow_end) * sizeof(float));
				}

				/* Bottom boundary rows: all zeros */
				for (int oh = oh_end; oh < OH; oh++)
					memset(dst_row + oh * OW, 0,
					       (size_t)OW * sizeof(float));
			}
		}
	}
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
