/*
 * src/backend/cpu/kernels/cpu_conv2d_bf16.c - Native NHWC BF16 Conv2D
 *
 * 2D convolution for bf16 tensors directly in NHWC layout. Unrolls
 * input patches from channels-last memory, transposes the small OHWI
 * weight to [K,N] for vectorization-friendly access, then matmuls
 * col × weight_T to produce NHWC output with zero input/output layout
 * conversions. All arithmetic is done in f32 since bf16 has no native
 * arithmetic. NEON path uses 4-wide float32x4_t with bf16<->f32
 * conversion helpers from core/half.h.
 *
 * Public layout: input [N,H,W,C], weight [OC,KH,KW,IC] (OHWI),
 * output [N,OH,OW,OC]. node->params[0]=stride, params[1]=padding.
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
#include "backend/cpu/cpu_blas.h"
#include "core/half.h"
#include "core/tensor.h"
#include "core/alloc.h"
#include "util/log.h"
#include "util/threadpool.h"

#include <stdint.h>
#include <string.h>

/*
 * im2col_nhwc_bf16 - Unroll NHWC bf16 input patches into a column matrix.
 *
 * Input:  [H, W, C] (one batch slice, channels-last, as uint16_t)
 * Output: [OH*OW, C_grp*KH*KW] (one row per output position)
 *
 * For grouped convolutions, c_off selects the starting channel.
 */
static void im2col_nhwc_bf16(const uint16_t *in, uint16_t *col,
			      int C, int H, int W,
			      int KH, int KW,
			      int stride, int pad,
			      int OH, int OW,
			      int c_off, int C_grp)
{
	size_t K = (size_t)C_grp * KH * KW;
	size_t grp_bytes = (size_t)C_grp * sizeof(uint16_t);

	for (int oh = 0; oh < OH; oh++) {
		for (int ow = 0; ow < OW; ow++) {
			uint16_t *dst = col +
				(size_t)(oh * OW + ow) * K;
			for (int kh = 0; kh < KH; kh++) {
				int ih = oh * stride + kh - pad;
				for (int kw = 0; kw < KW; kw++) {
					int iw = ow * stride + kw - pad;
					uint16_t *d = dst +
						((size_t)kh * KW + kw) *
						C_grp;
					if ((unsigned)ih < (unsigned)H &&
					    (unsigned)iw < (unsigned)W) {
						memcpy(d,
						       in + ((size_t)ih * W +
							     iw) * C +
						       c_off,
						       grp_bytes);
					} else {
						memset(d, 0, grp_bytes);
					}
				}
			}
		}
	}
}

/*
 * transpose_u16 - Transpose [rows, cols] → [cols, rows] for uint16_t.
 * Only used by the bespoke (non-BLAS) matmul path below.
 */
#ifndef SAM3_HAS_BLAS
static void transpose_u16(const uint16_t *src, uint16_t *dst,
			   int rows, int cols)
{
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			dst[(size_t)j * rows + i] =
				src[(size_t)i * cols + j];
}

/* --- NEON bf16 matmul path (f32 accumulation, 4-wide) --- */

#if SAM3_HAS_NEON

struct conv2d_nhwc_matmul_ctx_bf16 {
	const uint16_t *col;       /* [M, K] im2col output */
	const uint16_t *weight_t;  /* [K, N] transposed weight */
	uint16_t       *output;    /* [M, ldc] NHWC output */
	float          *acc;       /* [n_tasks, N] f32 accumulator */
	int             M;         /* OH * OW */
	int             K;         /* cpg_in * KH * KW */
	int             N;         /* cpg_out */
	int             ldc;       /* C_out (output row stride) */
};

/*
 * NEON bf16 matmul: output = col × weight_T (f32 accumulation).
 *
 * col [M, K], weight_T [K, N], output [M, ldc] with N cols written.
 * Accumulates in f32 using float32x4_t, converts to bf16 at the end.
 */
static void conv2d_nhwc_matmul_bf16_fn(void *arg, int task_id,
					int n_tasks)
{
	struct conv2d_nhwc_matmul_ctx_bf16 *ctx =
		(struct conv2d_nhwc_matmul_ctx_bf16 *)arg;
	int chunk = ctx->M / n_tasks;
	int m_start = task_id * chunk;
	int m_end = (task_id == n_tasks - 1)
		  ? ctx->M : m_start + chunk;

	if (m_start >= m_end)
		return;

	float *acc = ctx->acc + (size_t)task_id * ctx->N;
	int K = ctx->K;
	int N = ctx->N;
	int ldc = ctx->ldc;

	for (int i = m_start; i < m_end; i++) {
		memset(acc, 0, (size_t)N * sizeof(float));

		for (int k = 0; k < K; k++) {
			float aik = bf16_to_f32(
				ctx->col[(size_t)i * K + k]);
			float32x4_t va = vdupq_n_f32(aik);
			int j = 0;
			for (; j + 4 <= N; j += 4) {
				float32x4_t vc = vld1q_f32(acc + j);
				float32x4_t vb = bf16x4_to_f32x4(
					ctx->weight_t +
					(size_t)k * N + j);
				vst1q_f32(acc + j,
					  vfmaq_f32(vc, va, vb));
			}
			for (; j < N; j++)
				acc[j] += aik * bf16_to_f32(
					ctx->weight_t[
					(size_t)k * N + j]);
		}

		/* Convert f32 accumulator → bf16 output */
		uint16_t *out_row = ctx->output +
			(size_t)i * ldc;
		int j = 0;
		for (; j + 4 <= N; j += 4)
			f32x4_to_bf16x4(out_row + j,
					 vld1q_f32(acc + j));
		for (; j < N; j++)
			out_row[j] = f32_to_bf16(acc[j]);
	}
}

#else /* !SAM3_HAS_NEON */

/* --- Scalar fallback path (f32 accumulation) --- */

struct conv2d_nhwc_matmul_ctx_bf16 {
	const uint16_t *col;       /* [M, K] im2col output */
	const uint16_t *weight_t;  /* [K, N] transposed weight */
	uint16_t       *output;    /* [M, ldc] NHWC output */
	float          *acc;       /* [n_tasks, N] f32 accumulator */
	int             M;
	int             K;
	int             N;
	int             ldc;
};

static void conv2d_nhwc_matmul_bf16_fn(void *arg, int task_id,
					int n_tasks)
{
	struct conv2d_nhwc_matmul_ctx_bf16 *ctx =
		(struct conv2d_nhwc_matmul_ctx_bf16 *)arg;
	int chunk = ctx->M / n_tasks;
	int m_start = task_id * chunk;
	int m_end = (task_id == n_tasks - 1)
		  ? ctx->M : m_start + chunk;

	if (m_start >= m_end)
		return;

	float *acc = ctx->acc + (size_t)task_id * ctx->N;
	int K = ctx->K;
	int N = ctx->N;
	int ldc = ctx->ldc;

	for (int i = m_start; i < m_end; i++) {
		memset(acc, 0, (size_t)N * sizeof(float));

		for (int k = 0; k < K; k++) {
			float aik = bf16_to_f32(
				ctx->col[(size_t)i * K + k]);
			for (int j = 0; j < N; j++)
				acc[j] += aik * bf16_to_f32(
					ctx->weight_t[
					(size_t)k * N + j]);
		}

		uint16_t *out_row = ctx->output +
			(size_t)i * ldc;
		for (int j = 0; j < N; j++)
			out_row[j] = f32_to_bf16(acc[j]);
	}
}

#endif /* SAM3_HAS_NEON */
#endif /* !SAM3_HAS_BLAS */

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

	if (input->n_dims != 4 || weight->n_dims != 4 ||
	    output->n_dims != 4) {
		sam3_log_error("conv2d_bf16: expected 4D tensors");
		return SAM3_EINVAL;
	}

	/*
	 * Native NHWC conv2d: im2col from NHWC input, transpose the
	 * small OHWI weight to [K,N], then matmul col × weight_T
	 * → [OH*OW, OC] which is already NHWC.
	 */
	int N_batch = input->dims[0];
	int H       = input->dims[1];
	int W       = input->dims[2];
	int C_in    = input->dims[3];

	int C_out = weight->dims[0];
	int KH    = weight->dims[1];
	int KW    = weight->dims[2];
	int KIC   = weight->dims[3];

	int stride = node->params[0] > 0 ? node->params[0] : 1;
	int pad    = node->params[1];
	int groups = node->params[2] > 0 ? node->params[2] : 1;

	if (KIC * groups != C_in) {
		sam3_log_error("conv2d_bf16: channel mismatch %d*%d != %d",
			       KIC, groups, C_in);
		return SAM3_EINVAL;
	}

	if (C_in % groups != 0 || C_out % groups != 0) {
		sam3_log_error("conv2d_bf16: channels not divisible "
			       "by groups");
		return SAM3_EINVAL;
	}

	int OH = output->dims[1];
	int OW = output->dims[2];
	if (output->dims[0] != N_batch || output->dims[3] != C_out) {
		sam3_log_error("conv2d_bf16: output shape mismatch");
		return SAM3_EINVAL;
	}

	int cpg_in  = C_in / groups;
	int cpg_out = C_out / groups;

	size_t saved_offset = scratch->offset;

	/* im2col buffer: [OH*OW, cpg_in*KH*KW] in bf16 */
	size_t K = (size_t)cpg_in * KH * KW;
	size_t col_size = (size_t)(OH * OW) * K * sizeof(uint16_t);
	uint16_t *col = (uint16_t *)sam3_arena_alloc(scratch, col_size);
	if (!col) {
		sam3_log_error("conv2d_bf16: scratch OOM for im2col "
			       "(%zu bytes)", col_size);
		scratch->offset = saved_offset;
		return SAM3_ENOMEM;
	}

#ifdef SAM3_HAS_BLAS
	/* BLAS path: cast col & weight to f32, sgemm with trans_b=true,
	 * cast result back to bf16 with NHWC stride. The transpose
	 * fuses into the GEMM so we skip the wt_buf copy. */
	{
		size_t M = (size_t)OH * OW;
		float *col_f = (float *)sam3_arena_alloc(scratch,
				M * K * sizeof(float));
		float *w_f   = (float *)sam3_arena_alloc(scratch,
				(size_t)cpg_out * K * sizeof(float));
		float *out_f = (float *)sam3_arena_alloc(scratch,
				M * (size_t)cpg_out * sizeof(float));
		if (!col_f || !w_f || !out_f) {
			sam3_log_error("conv2d_bf16: scratch OOM (BLAS)");
			scratch->offset = saved_offset;
			return SAM3_ENOMEM;
		}

		const uint16_t *w_data = (const uint16_t *)weight->data;

		for (int n = 0; n < N_batch; n++) {
			const uint16_t *in_n =
				(const uint16_t *)input->data +
				(size_t)n * H * W * C_in;
			uint16_t *out_n = (uint16_t *)output->data +
				(size_t)n * OH * OW * C_out;

			for (int g = 0; g < groups; g++) {
				im2col_nhwc_bf16(in_n, col,
						 C_in, H, W, KH, KW,
						 stride, pad, OH, OW,
						 g * cpg_in, cpg_in);

				const uint16_t *w_g = w_data +
					(size_t)g * cpg_out * KH * KW * KIC;

				size_t nC = M * K;
				size_t nW = (size_t)cpg_out * K;
				for (size_t i = 0; i < nC; ++i)
					col_f[i] = bf16_to_f32(col[i]);
				for (size_t i = 0; i < nW; ++i)
					w_f[i] = bf16_to_f32(w_g[i]);

				sam3_blas_sgemm(false, true,
						(int)M, cpg_out, (int)K,
						1.0f,
						col_f, (int)K,
						w_f, (int)K,
						0.0f,
						out_f, cpg_out);

				uint16_t *out_g = out_n + g * cpg_out;
				for (size_t i = 0; i < M; ++i) {
					for (int j = 0; j < cpg_out; ++j)
						out_g[i * C_out + j] =
							f32_to_bf16(
							  out_f[i * cpg_out + j]);
				}
			}
		}

		scratch->offset = saved_offset;
		(void)pool;
		return SAM3_OK;
	}
#endif

#ifndef SAM3_HAS_BLAS
	/* Weight transpose buffer: [K, cpg_out] in bf16 */
	size_t wt_size = K * (size_t)cpg_out * sizeof(uint16_t);
	uint16_t *wt_buf = (uint16_t *)sam3_arena_alloc_raw(scratch,
							     wt_size);
	if (!wt_buf) {
		sam3_log_error("conv2d_bf16: scratch OOM for weight_T "
			       "(%zu bytes)", wt_size);
		scratch->offset = saved_offset;
		return SAM3_ENOMEM;
	}

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

	/* f32 accumulator: one row per thread */
	size_t acc_size = (size_t)n_tasks * cpg_out * sizeof(float);
	float *acc_buf = (float *)sam3_arena_alloc(scratch, acc_size);
	if (!acc_buf) {
		sam3_log_error("conv2d_bf16: scratch OOM for acc "
			       "(%zu bytes)", acc_size);
		scratch->offset = saved_offset;
		return SAM3_ENOMEM;
	}

	const uint16_t *w_data = (const uint16_t *)weight->data;
	int M = OH * OW;

	for (int n = 0; n < N_batch; n++) {
		const uint16_t *in_n =
			(const uint16_t *)input->data +
			(size_t)n * H * W * C_in;
		uint16_t *out_n = (uint16_t *)output->data +
			(size_t)n * OH * OW * C_out;

		for (int g = 0; g < groups; g++) {
			im2col_nhwc_bf16(in_n, col,
					 C_in, H, W, KH, KW,
					 stride, pad, OH, OW,
					 g * cpg_in, cpg_in);

			const uint16_t *w_g = w_data +
				(size_t)g * cpg_out * KH * KW * KIC;
			transpose_u16(w_g, wt_buf,
				      cpg_out, (int)K);

			struct conv2d_nhwc_matmul_ctx_bf16 mctx = {
				.col = col,
				.weight_t = wt_buf,
				.output = out_n + g * cpg_out,
				.acc = acc_buf,
				.M = M, .K = (int)K,
				.N = cpg_out, .ldc = C_out,
			};
			sam3_threadpool_parallel_for(
				pool,
				conv2d_nhwc_matmul_bf16_fn,
				&mctx, n_tasks);
		}
	}

	scratch->offset = saved_offset;
	return SAM3_OK;
#endif /* !SAM3_HAS_BLAS */
}
