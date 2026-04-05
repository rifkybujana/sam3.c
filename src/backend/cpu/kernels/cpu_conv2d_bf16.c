/*
 * src/backend/cpu/kernels/cpu_conv2d_bf16.c - BF16 Conv2D via im2col + matmul
 *
 * 2D convolution for bf16 tensors using im2col + matrix multiply. All
 * arithmetic is done in f32 since bf16 has no native arithmetic. NEON
 * path uses 4-wide float32x4_t with bf16<->f32 conversion helpers from
 * core/half.h. Scratch arena for im2col buffer and f32 accumulation.
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
 *
 * Pre-computes valid oh/ow ranges to eliminate per-element bounds checks.
 * Uses memcpy for contiguous interior when stride==1, memset for padding.
 */
static void im2col_bf16(const uint16_t *in, uint16_t *col,
			 int C, int H, int W, int KH, int KW,
			 int stride, int pad, int OH, int OW)
{
	int col_cols = OH * OW;

	for (int c = 0; c < C; c++) {
		const uint16_t *in_c = in + c * H * W;
		for (int kh = 0; kh < KH; kh++) {
			/* Valid oh range: ih = oh*stride+kh-pad in [0,H) */
			int oh_s = (pad > kh)
				 ? (pad - kh + stride - 1) / stride : 0;
			int oh_e = (H + pad - kh - 1 >= 0)
				 ? (H + pad - kh - 1) / stride + 1 : 0;
			if (oh_e > OH) oh_e = OH;
			if (oh_s > oh_e) oh_s = oh_e;

			for (int kw = 0; kw < KW; kw++) {
				int col_row = c * KH * KW
					    + kh * KW + kw;
				uint16_t *dst = col
					      + col_row * col_cols;

				/* Valid ow range */
				int ow_s = (pad > kw)
					 ? (pad - kw + stride - 1)
					   / stride : 0;
				int ow_e = (W + pad - kw - 1 >= 0)
					 ? (W + pad - kw - 1)
					   / stride + 1 : 0;
				if (ow_e > OW) ow_e = OW;
				if (ow_s > ow_e) ow_s = ow_e;

				/* Top padding rows */
				if (oh_s > 0)
					memset(dst, 0,
					       (size_t)oh_s * OW
					       * sizeof(uint16_t));

				for (int oh = oh_s; oh < oh_e; oh++) {
					int ih = oh * stride + kh - pad;
					uint16_t *row = dst + oh * OW;

					if (ow_s > 0)
						memset(row, 0,
						       ow_s
						       * sizeof(uint16_t));

					int nv = ow_e - ow_s;
					if (nv > 0) {
						int iw0 = ow_s * stride
							 + kw - pad;
						if (stride == 1) {
							memcpy(row + ow_s,
							       in_c + ih * W
							       + iw0,
							       nv * sizeof(
							       uint16_t));
						} else {
							for (int ow = ow_s;
							     ow < ow_e;
							     ow++) {
								int iw = ow * stride
									 + kw - pad;
								row[ow] = in_c[
									ih * W
									+ iw];
							}
						}
					}

					if (ow_e < OW)
						memset(row + ow_e, 0,
						       (OW - ow_e)
						       * sizeof(uint16_t));
				}

				/* Bottom padding rows */
				if (oh_e < OH)
					memset(dst + (size_t)oh_e * OW,
					       0,
					       (size_t)(OH - oh_e) * OW
					       * sizeof(uint16_t));
			}
		}
	}
}

/* --- NEON bf16 matmul path (f32 accumulation, 4-wide) --- */

#if SAM3_HAS_NEON

struct conv2d_matmul_ctx_bf16 {
	const uint16_t *a;   /* weight [OC, C*KH*KW] */
	const uint16_t *b;   /* col    [C*KH*KW, OH*OW] */
	uint16_t       *c;   /* output [OC, OH*OW] */
	float          *acc; /* f32 accum [n_tasks, OH*OW] */
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

	float *acc = ctx->acc + (size_t)task_id * ctx->N;

	for (int i = m_start; i < m_end; i++) {
		memset(acc, 0, ctx->N * sizeof(float));

		for (int k = 0; k < ctx->K; k++) {
			float aik = bf16_to_f32(
				ctx->a[i * ctx->K + k]);
			float32x4_t va = vdupq_n_f32(aik);
			int j = 0;
			for (; j + 4 <= ctx->N; j += 4) {
				float32x4_t vc = vld1q_f32(acc + j);
				float32x4_t vb = bf16x4_to_f32x4(
					ctx->b + k * ctx->N + j);
				vst1q_f32(acc + j,
					  vfmaq_f32(vc, va, vb));
			}
			for (; j < ctx->N; j++)
				acc[j] += aik * bf16_to_f32(
					ctx->b[k * ctx->N + j]);
		}

		/* Convert f32 accumulator -> bf16 output once */
		uint16_t *out_row = ctx->c + (size_t)i * ctx->N;
		int j = 0;
		for (; j + 4 <= ctx->N; j += 4)
			f32x4_to_bf16x4(out_row + j,
					 vld1q_f32(acc + j));
		for (; j < ctx->N; j++)
			out_row[j] = f32_to_bf16(acc[j]);
	}
}

#else /* !SAM3_HAS_NEON */

/* --- Scalar fallback path (f32 accumulation) --- */

struct conv2d_matmul_ctx_bf16 {
	const uint16_t *a;   /* weight [OC, C*KH*KW] */
	const uint16_t *b;   /* col    [C*KH*KW, OH*OW] */
	uint16_t       *c;   /* output [OC, OH*OW] */
	float          *acc; /* f32 accum [n_tasks, OH*OW] */
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

	float *acc = ctx->acc + (size_t)task_id * ctx->N;

	for (int i = m_start; i < m_end; i++) {
		memset(acc, 0, ctx->N * sizeof(float));

		for (int k = 0; k < ctx->K; k++) {
			float aik = bf16_to_f32(
				ctx->a[i * ctx->K + k]);
			for (int j = 0; j < ctx->N; j++)
				acc[j] += aik * bf16_to_f32(
					ctx->b[k * ctx->N + j]);
		}

		uint16_t *out_row = ctx->c + (size_t)i * ctx->N;
		for (int j = 0; j < ctx->N; j++)
			out_row[j] = f32_to_bf16(acc[j]);
	}
}

#endif /* SAM3_HAS_NEON */

/*
 * cpu_kernel_conv2d_bf16 - BF16 Conv2D via im2col + matmul.
 *
 * @node:    Node with n_inputs>=2: input [N,C,H,W] and weight [OC,C,KH,KW],
 *           both SAM3_DTYPE_BF16. node->params[0]=stride, params[1]=padding.
 * @scratch: Scratch arena for im2col temp buffer and f32 accumulation.
 *           Offset is saved/restored.
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
	int N_out = OH * OW;

	/* Save scratch offset for restore */
	size_t saved_offset = scratch->offset;

	/* Allocate im2col buffer: [C*KH*KW, OH*OW] in bf16 */
	size_t col_size = (size_t)(C * KH * KW) * N_out
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

	/* Allocate f32 accumulation buffer: one row per thread */
	size_t acc_size = (size_t)n_tasks * N_out * sizeof(float);
	float *acc_buf = (float *)sam3_arena_alloc(scratch, acc_size);
	if (!acc_buf) {
		sam3_log_error("conv2d_bf16: scratch OOM for acc "
			       "(%zu bytes)", acc_size);
		scratch->offset = saved_offset;
		return SAM3_ENOMEM;
	}

	const uint16_t *w_data = (const uint16_t *)weight->data;
	uint16_t       *col    = (uint16_t *)col_buf;

	for (int n = 0; n < N_batch; n++) {
		const uint16_t *in_n = (const uint16_t *)input->data
				       + (size_t)n * C * H * W;
		uint16_t *out_n = (uint16_t *)output->data
				  + (size_t)n * OC * N_out;

		im2col_bf16(in_n, col, C, H, W, KH, KW,
			    stride, pad, OH, OW);

		struct conv2d_matmul_ctx_bf16 mctx = {
			.a   = w_data,
			.b   = col,
			.c   = out_n,
			.acc = acc_buf,
			.M   = OC,
			.K   = C * KH * KW,
			.N   = N_out,
		};
		sam3_threadpool_parallel_for(pool, conv2d_matmul_bf16_fn,
					     &mctx, n_tasks);
	}

	/* Restore scratch offset — frees im2col + acc buffers */
	scratch->offset = saved_offset;

	return SAM3_OK;
}
