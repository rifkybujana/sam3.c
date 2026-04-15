/*
 * src/backend/cpu/kernels/cpu_conv2d_f16.c - Native NHWC FP16 Conv2D
 *
 * 2D convolution for fp16 tensors directly in NHWC layout. Unrolls
 * input patches from channels-last memory, transposes the small OHWI
 * weight to [K,N] for vectorization-friendly access, then matmuls
 * col × weight_T to produce NHWC output with zero input/output layout
 * conversions. NEON fp16 path uses native float16x8_t with 8x8x64
 * tiling. Scalar fallback accumulates in f32.
 *
 * Public layout: input [N,H,W,C], weight [OC,KH,KW,IC] (OHWI),
 * output [N,OH,OW,OC]. node->params[0]=stride, params[1]=padding.
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

#include <stdint.h>
#include <string.h>

#define TILE_M 8
#define TILE_N 8
#define TILE_K 64

/*
 * im2col_nhwc_f16 - Unroll NHWC fp16 input patches into a column matrix.
 *
 * Input:  [H, W, C] (one batch slice, channels-last, as uint16_t)
 * Output: [OH*OW, C_grp*KH*KW] (one row per output position)
 *
 * For grouped convolutions, c_off selects the starting channel.
 */
static void im2col_nhwc_f16(const uint16_t *in, uint16_t *col,
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
 *
 * Used to convert weight [N, K] → weight_T [K, N] so the matmul
 * inner loop accesses weight_T[k][j:j+8] contiguously for NEON.
 */
static void transpose_u16(const uint16_t *src, uint16_t *dst,
			   int rows, int cols)
{
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			dst[(size_t)j * rows + i] =
				src[(size_t)i * cols + j];
}

/* ── NEON fp16 path ─────────────────────────────────────────────────── */

#if SAM3_HAS_NEON_FP16

struct conv2d_nhwc_matmul_ctx_f16 {
	const _Float16 *col;       /* [M, K] im2col output */
	const _Float16 *weight_t;  /* [K, N] transposed weight */
	_Float16       *output;    /* [M, ldc] NHWC output */
	int             M;         /* OH * OW */
	int             K;         /* cpg_in * KH * KW */
	int             N;         /* cpg_out */
	int             ldc;       /* C_out (output row stride) */
};

/*
 * NEON fp16 matmul: output = col × weight_T.
 *
 * col [M, K], weight_T [K, N], output [M, ldc] with N cols written.
 * Uses float16x8_t with 8×8×64 tiling. Splits over M rows.
 */
static void conv2d_nhwc_matmul_f16_fn(void *arg, int task_id,
				       int n_tasks)
{
	struct conv2d_nhwc_matmul_ctx_f16 *ctx =
		(struct conv2d_nhwc_matmul_ctx_f16 *)arg;
	int chunk = ctx->M / n_tasks;
	int m_start = task_id * chunk;
	int m_end = (task_id == n_tasks - 1)
		  ? ctx->M : m_start + chunk;

	if (m_start >= m_end)
		return;

	int K = ctx->K;
	int N = ctx->N;
	int ldc = ctx->ldc;

	for (int i = m_start; i < m_end; i++)
		memset(ctx->output + (size_t)i * ldc, 0,
		       (size_t)N * sizeof(_Float16));

	for (int i0 = m_start; i0 < m_end; i0 += TILE_M) {
		int imax = (i0 + TILE_M < m_end)
			 ? i0 + TILE_M : m_end;
		for (int j0 = 0; j0 < N; j0 += TILE_N) {
			int jmax = (j0 + TILE_N < N)
				 ? j0 + TILE_N : N;
			for (int k0 = 0; k0 < K; k0 += TILE_K) {
				int kmax = (k0 + TILE_K < K)
					 ? k0 + TILE_K : K;
				for (int i = i0; i < imax; i++) {
					for (int k = k0; k < kmax;
					     k++) {
						float16x8_t va =
							vdupq_n_f16(
							ctx->col[
							(size_t)i * K
							+ k]);
						int j = j0;
						for (; j + 8 <= jmax;
						     j += 8) {
							float16x8_t vc =
								vld1q_f16(
								(const __fp16 *)
								(ctx->output +
								 (size_t)i *
								 ldc + j));
							float16x8_t vb =
								vld1q_f16(
								(const __fp16 *)
								(ctx->weight_t +
								 (size_t)k *
								 N + j));
							vst1q_f16(
								(__fp16 *)
								(ctx->output +
								 (size_t)i *
								 ldc + j),
								vfmaq_f16(
								vc, va, vb));
						}
						_Float16 aik =
							ctx->col[
							(size_t)i * K
							+ k];
						for (; j < jmax; j++)
							ctx->output[
							(size_t)i * ldc
							+ j] += aik *
							ctx->weight_t[
							(size_t)k * N
							+ j];
					}
				}
			}
		}
	}
}

#else /* !SAM3_HAS_NEON_FP16 */

/* ── Scalar fallback path (f32 accumulation) ───────────────────────── */

struct conv2d_nhwc_matmul_ctx_f16 {
	const uint16_t *col;       /* [M, K] im2col output */
	const uint16_t *weight_t;  /* [K, N] transposed weight */
	uint16_t       *output;    /* [M, ldc] NHWC output */
	float          *acc;       /* [n_tasks, N] f32 accumulator */
	int             M;         /* OH * OW */
	int             K;         /* cpg_in * KH * KW */
	int             N;         /* cpg_out */
	int             ldc;       /* C_out (output row stride) */
};

static void conv2d_nhwc_matmul_f16_fn(void *arg, int task_id,
				       int n_tasks)
{
	struct conv2d_nhwc_matmul_ctx_f16 *ctx =
		(struct conv2d_nhwc_matmul_ctx_f16 *)arg;
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

		for (int k0 = 0; k0 < K; k0 += TILE_K) {
			int kmax = (k0 + TILE_K < K)
				 ? k0 + TILE_K : K;
			for (int k = k0; k < kmax; k++) {
				float aik = fp16_to_f32(
					ctx->col[(size_t)i * K + k]);
				for (int j = 0; j < N; j++)
					acc[j] += aik * fp16_to_f32(
						ctx->weight_t[
						(size_t)k * N + j]);
			}
		}

		uint16_t *out_row = ctx->output +
			(size_t)i * ldc;
		for (int j = 0; j < N; j++)
			out_row[j] = f32_to_fp16(acc[j]);
	}
}

#endif /* SAM3_HAS_NEON_FP16 */

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

	if (input->n_dims != 4 || weight->n_dims != 4 ||
	    output->n_dims != 4) {
		sam3_log_error("conv2d_f16: expected 4D tensors");
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
		sam3_log_error("conv2d_f16: channel mismatch %d*%d != %d",
			       KIC, groups, C_in);
		return SAM3_EINVAL;
	}

	if (C_in % groups != 0 || C_out % groups != 0) {
		sam3_log_error("conv2d_f16: channels not divisible "
			       "by groups");
		return SAM3_EINVAL;
	}

	int OH = output->dims[1];
	int OW = output->dims[2];
	if (output->dims[0] != N_batch || output->dims[3] != C_out) {
		sam3_log_error("conv2d_f16: output shape mismatch");
		return SAM3_EINVAL;
	}

	int cpg_in  = C_in / groups;
	int cpg_out = C_out / groups;

	size_t saved_offset = scratch->offset;

	/* im2col buffer: [OH*OW, cpg_in*KH*KW] in fp16 */
	size_t K = (size_t)cpg_in * KH * KW;
	size_t col_size = (size_t)(OH * OW) * K * sizeof(uint16_t);
	uint16_t *col = (uint16_t *)sam3_arena_alloc(scratch, col_size);
	if (!col) {
		sam3_log_error("conv2d_f16: scratch OOM for im2col "
			       "(%zu bytes)", col_size);
		scratch->offset = saved_offset;
		return SAM3_ENOMEM;
	}

	/* Weight transpose buffer: [K, cpg_out] in fp16 */
	size_t wt_size = K * (size_t)cpg_out * sizeof(uint16_t);
	uint16_t *wt_buf = (uint16_t *)sam3_arena_alloc_raw(scratch,
							     wt_size);
	if (!wt_buf) {
		sam3_log_error("conv2d_f16: scratch OOM for weight_T "
			       "(%zu bytes)", wt_size);
		scratch->offset = saved_offset;
		return SAM3_ENOMEM;
	}

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

#if !SAM3_HAS_NEON_FP16
	/* f32 accumulator: one row per thread */
	size_t acc_size = (size_t)n_tasks * cpg_out * sizeof(float);
	float *acc_buf = (float *)sam3_arena_alloc(scratch, acc_size);
	if (!acc_buf) {
		sam3_log_error("conv2d_f16: scratch OOM for acc "
			       "(%zu bytes)", acc_size);
		scratch->offset = saved_offset;
		return SAM3_ENOMEM;
	}
#endif

	const uint16_t *w_data = (const uint16_t *)weight->data;
	int M = OH * OW;

	for (int n = 0; n < N_batch; n++) {
		const uint16_t *in_n =
			(const uint16_t *)input->data +
			(size_t)n * H * W * C_in;
		uint16_t *out_n = (uint16_t *)output->data +
			(size_t)n * OH * OW * C_out;

		for (int g = 0; g < groups; g++) {
			im2col_nhwc_f16(in_n, col,
					C_in, H, W, KH, KW,
					stride, pad, OH, OW,
					g * cpg_in, cpg_in);

			const uint16_t *w_g = w_data +
				(size_t)g * cpg_out * KH * KW * KIC;
			transpose_u16(w_g, wt_buf,
				      cpg_out, (int)K);

#if SAM3_HAS_NEON_FP16
			struct conv2d_nhwc_matmul_ctx_f16 mctx = {
				.col = (const _Float16 *)col,
				.weight_t = (const _Float16 *)wt_buf,
				.output = (_Float16 *)(out_n +
					g * cpg_out),
				.M = M, .K = (int)K,
				.N = cpg_out, .ldc = C_out,
			};
#else
			struct conv2d_nhwc_matmul_ctx_f16 mctx = {
				.col = col,
				.weight_t = wt_buf,
				.output = out_n + g * cpg_out,
				.acc = acc_buf,
				.M = M, .K = (int)K,
				.N = cpg_out, .ldc = C_out,
			};
#endif
			sam3_threadpool_parallel_for(
				pool,
				conv2d_nhwc_matmul_f16_fn,
				&mctx, n_tasks);
		}
	}

	scratch->offset = saved_offset;
	return SAM3_OK;
}
