/*
 * src/backend/cpu/kernels/cpu_conv2d.c - Native NHWC Conv2D via im2col + matmul
 *
 * Implements 2D convolution directly in NHWC layout: unrolls input
 * patches from channels-last memory into a column matrix, then
 * matrix-multiplies with the OHWI weight to produce NHWC output
 * with zero layout conversions. The OHWI weight [OC,KH,KW,IC]
 * is treated as [OC, KH*KW*IC] and the matmul computes col × W^T.
 *
 * Public layout: input [N,H,W,C], weight [OC,KH,KW,IC] (OHWI),
 * output [N,OH,OW,OC]. node->params[0]=stride, params[1]=padding.
 *
 * Byte-level transpose helpers are retained for the f16 / bf16 /
 * conv_transpose2d / maxpool kernels which still use the NCHW shim.
 *
 * Key types:  sam3_node, sam3_tensor, sam3_arena
 * Depends on: cpu_kernels.h, core/tensor.h, core/alloc.h
 * Used by:    cpu_backend.c (dispatch), cpu_conv2d_f16.c,
 *             cpu_conv2d_bf16.c, cpu_conv_transpose2d.c,
 *             cpu_maxpool2d.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "cpu_kernels.h"
#include "core/tensor.h"
#include "core/alloc.h"
#include "util/log.h"
#include "util/threadpool.h"

#include <stdint.h>
#include <string.h>

#ifdef SAM3_HAS_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

/*
 * im2col_nhwc_f32 - Unroll NHWC input patches into a column matrix.
 *
 * Input:  [H, W, C] (one batch slice, channels-last)
 * Output: [OH*OW, C_grp*KH*KW] (one row per output position)
 *
 * For grouped convolutions, c_off selects the starting channel
 * and C_grp copies only that group's channels per pixel. Each
 * copy moves C_grp contiguous floats — cache-friendly for NHWC
 * since channels are innermost.
 */
static void im2col_nhwc_f32(const float *in, float *col,
			     int C, int H, int W,
			     int KH, int KW,
			     int stride, int pad,
			     int OH, int OW,
			     int c_off, int C_grp)
{
	size_t K = (size_t)C_grp * KH * KW;
	size_t grp_bytes = (size_t)C_grp * sizeof(float);

	for (int oh = 0; oh < OH; oh++) {
		for (int ow = 0; ow < OW; ow++) {
			float *dst = col + (size_t)(oh * OW + ow) * K;
			for (int kh = 0; kh < KH; kh++) {
				int ih = oh * stride + kh - pad;
				for (int kw = 0; kw < KW; kw++) {
					int iw = ow * stride + kw - pad;
					float *d = dst +
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

struct conv2d_nhwc_matmul_ctx {
	const float *col;     /* [M, K] im2col output */
	const float *weight;  /* [N, K] OHWI reshaped per-group */
	float       *output;  /* NHWC output base (may be offset for groups) */
	int          M;       /* OH * OW */
	int          K;       /* cpg_in * KH * KW */
	int          N;       /* cpg_out */
	int          ldc;     /* C_out (output row stride for groups) */
};

/*
 * Parallel matmul: output = col × weight^T.
 *
 * col [M, K], weight [N, K] → output [M, ldc] with N cols written.
 * Splits work over M rows (output spatial positions).
 */
static void conv2d_nhwc_matmul_fn(void *arg, int task_id, int n_tasks)
{
	struct conv2d_nhwc_matmul_ctx *ctx =
		(struct conv2d_nhwc_matmul_ctx *)arg;
	int chunk = ctx->M / n_tasks;
	int m_start = task_id * chunk;
	int m_end = (task_id == n_tasks - 1)
		  ? ctx->M : m_start + chunk;

	if (m_start >= m_end)
		return;

	int K = ctx->K;
	int N = ctx->N;
	int ldc = ctx->ldc;

	for (int i = m_start; i < m_end; i++) {
		float *out = ctx->output + (size_t)i * ldc;
		memset(out, 0, (size_t)N * sizeof(float));
		for (int k = 0; k < K; k++) {
			float cik = ctx->col[(size_t)i * K + k];
			for (int j = 0; j < N; j++)
				out[j] += cik *
					ctx->weight[(size_t)j * K + k];
		}
	}
}

/* --- NHWC transpose helpers (dtype-generic, file-scope linkage) --- */

/*
 * These helpers are used by cpu_kernel_conv2d_f16 / _bf16 /
 * cpu_kernel_conv_transpose2d / cpu_kernel_maxpool2d which still use
 * the NCHW shim. They have external linkage under the sam3_cpu_nhwc_*
 * prefix so sibling kernel files can reuse them without duplication.
 * They are not declared in cpu_kernels.h because they are an
 * implementation detail of the CPU spatial-op NCHW paths.
 */

/*
 * Transpose NHWC -> NCHW at byte granularity.
 *
 * src is [N, H, W, C] with `elem_sz` bytes per element. dst is
 * [N, C, H, W]. This is test-only scaffolding for the CPU conv
 * path, so a straight scalar loop with per-element memcpy is
 * sufficient (single-byte / two-byte / four-byte dtypes).
 */
void sam3_cpu_nhwc_to_nchw_bytes(const uint8_t *src, uint8_t *dst,
				 int N, int H, int W, int C,
				 size_t elem_sz);

void sam3_cpu_nchw_to_nhwc_bytes(const uint8_t *src, uint8_t *dst,
				 int N, int C, int H, int W,
				 size_t elem_sz);

void sam3_cpu_ohwi_to_oihw_bytes(const uint8_t *src, uint8_t *dst,
				 int OC, int KH, int KW, int IC,
				 size_t elem_sz);

void sam3_cpu_ohwi_to_iohw_bytes(const uint8_t *src, uint8_t *dst,
				 int OC, int KH, int KW, int IC,
				 size_t elem_sz);

void sam3_cpu_nhwc_to_nchw_bytes(const uint8_t *src, uint8_t *dst,
				 int N, int H, int W, int C,
				 size_t elem_sz)
{
	size_t hw = (size_t)H * W;

	for (int n = 0; n < N; n++) {
		for (int h = 0; h < H; h++) {
			for (int w = 0; w < W; w++) {
				const uint8_t *src_px = src +
					(((size_t)n * H + h) * W + w) *
					C * elem_sz;
				for (int c = 0; c < C; c++) {
					uint8_t *dst_px = dst +
						(((size_t)n * C + c) * hw +
						 (size_t)h * W + w) *
						elem_sz;
					memcpy(dst_px,
					       src_px + (size_t)c * elem_sz,
					       elem_sz);
				}
			}
		}
	}
}

/*
 * Transpose NCHW -> NHWC at byte granularity. Inverse of the above.
 */
void sam3_cpu_nchw_to_nhwc_bytes(const uint8_t *src, uint8_t *dst,
				 int N, int C, int H, int W,
				 size_t elem_sz)
{
	size_t hw = (size_t)H * W;

	for (int n = 0; n < N; n++) {
		for (int h = 0; h < H; h++) {
			for (int w = 0; w < W; w++) {
				uint8_t *dst_px = dst +
					(((size_t)n * H + h) * W + w) *
					C * elem_sz;
				for (int c = 0; c < C; c++) {
					const uint8_t *src_px = src +
						(((size_t)n * C + c) * hw +
						 (size_t)h * W + w) *
						elem_sz;
					memcpy(dst_px + (size_t)c * elem_sz,
					       src_px, elem_sz);
				}
			}
		}
	}
}

/*
 * Transpose OHWI -> OIHW at byte granularity for conv2d weights.
 *
 * src is [OC, KH, KW, IC]. dst is [OC, IC, KH, KW].
 */
void sam3_cpu_ohwi_to_oihw_bytes(const uint8_t *src, uint8_t *dst,
				 int OC, int KH, int KW, int IC,
				 size_t elem_sz)
{
	size_t khkw = (size_t)KH * KW;

	for (int oc = 0; oc < OC; oc++) {
		for (int kh = 0; kh < KH; kh++) {
			for (int kw = 0; kw < KW; kw++) {
				const uint8_t *src_px = src +
					(((size_t)oc * KH + kh) * KW + kw) *
					IC * elem_sz;
				for (int ic = 0; ic < IC; ic++) {
					uint8_t *dst_px = dst +
						(((size_t)oc * IC + ic) *
						 khkw + (size_t)kh * KW +
						 kw) * elem_sz;
					memcpy(dst_px,
					       src_px +
					       (size_t)ic * elem_sz,
					       elem_sz);
				}
			}
		}
	}
}

/*
 * Transpose OHWI -> IOHW at byte granularity for conv_transpose2d
 * weights.
 *
 * src is [OC, KH, KW, IC]. dst is [IC, OC, KH, KW].
 */
void sam3_cpu_ohwi_to_iohw_bytes(const uint8_t *src, uint8_t *dst,
				 int OC, int KH, int KW, int IC,
				 size_t elem_sz)
{
	size_t khkw = (size_t)KH * KW;

	for (int oc = 0; oc < OC; oc++) {
		for (int kh = 0; kh < KH; kh++) {
			for (int kw = 0; kw < KW; kw++) {
				const uint8_t *src_px = src +
					(((size_t)oc * KH + kh) * KW + kw) *
					IC * elem_sz;
				for (int ic = 0; ic < IC; ic++) {
					uint8_t *dst_px = dst +
						(((size_t)ic * OC + oc) *
						 khkw + (size_t)kh * KW +
						 kw) * elem_sz;
					memcpy(dst_px,
					       src_px +
					       (size_t)ic * elem_sz,
					       elem_sz);
				}
			}
		}
	}
}

enum sam3_error cpu_kernel_conv2d(const struct sam3_node *node,
				  struct sam3_arena *scratch,
				  struct sam3_threadpool *pool)
{
#ifdef SAM3_HAS_BLAS
	(void)pool;
#endif

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

	if (input->dtype != SAM3_DTYPE_F32 ||
	    weight->dtype != SAM3_DTYPE_F32) {
		sam3_log_error("conv2d: unsupported dtype");
		return SAM3_EINVAL;
	}

	if (input->n_dims != 4 || weight->n_dims != 4 ||
	    output->n_dims != 4) {
		sam3_log_error("conv2d: expected 4D tensors");
		return SAM3_EINVAL;
	}

	/*
	 * Native NHWC conv2d: input [N,H,W,C_in], weight [OC,KH,KW,IC]
	 * (OHWI), output [N,OH,OW,OC]. im2col unrolls directly from
	 * NHWC producing [OH*OW, IC*KH*KW], then matmul computes
	 * col × W^T → [OH*OW, OC] which is already NHWC. Zero layout
	 * conversions.
	 */
	int N_batch = input->dims[0];
	int H = input->dims[1];
	int W = input->dims[2];
	int C_in = input->dims[3];

	int C_out = weight->dims[0];
	int KH = weight->dims[1];
	int KW = weight->dims[2];
	int KIC = weight->dims[3];

	int stride = node->params[0] > 0 ? node->params[0] : 1;
	int pad = node->params[1];
	int groups = node->params[2] > 0 ? node->params[2] : 1;

	if (KIC * groups != C_in) {
		sam3_log_error("conv2d: channel mismatch %d*%d != %d",
			       KIC, groups, C_in);
		return SAM3_EINVAL;
	}

	if (C_in % groups != 0 || C_out % groups != 0) {
		sam3_log_error("conv2d: channels not divisible by groups");
		return SAM3_EINVAL;
	}

	int OH = output->dims[1];
	int OW = output->dims[2];
	if (output->dims[0] != N_batch || output->dims[3] != C_out) {
		sam3_log_error("conv2d: output shape mismatch");
		return SAM3_EINVAL;
	}

	int cpg_in = C_in / groups;
	int cpg_out = C_out / groups;

	size_t saved_offset = scratch->offset;

	/* im2col buffer: [OH*OW, cpg_in*KH*KW] */
	size_t K = (size_t)cpg_in * KH * KW;
	size_t col_size = (size_t)(OH * OW) * K * sizeof(float);
	float *col = (float *)sam3_arena_alloc(scratch, col_size);
	if (!col) {
		sam3_log_error("conv2d: scratch OOM for im2col "
			       "(%zu bytes)", col_size);
		scratch->offset = saved_offset;
		return SAM3_ENOMEM;
	}

	int M = OH * OW;
	const float *w_data = (const float *)weight->data;

	for (int n = 0; n < N_batch; n++) {
		const float *in_n = (const float *)input->data +
				    (size_t)n * H * W * C_in;
		float *out_n = (float *)output->data +
			       (size_t)n * OH * OW * C_out;

		for (int g = 0; g < groups; g++) {
			im2col_nhwc_f32(in_n, col,
					C_in, H, W, KH, KW,
					stride, pad, OH, OW,
					g * cpg_in, cpg_in);

			/* OHWI weight for group g: contiguous
			 * [cpg_out, KH*KW*KIC] block */
			const float *w_g = w_data +
				(size_t)g * cpg_out * KH * KW * KIC;

#ifdef SAM3_HAS_BLAS
			/* output = col × weight^T
			 * col [M, K], weight [cpg_out, K]
			 * → output [M, cpg_out] with ldc = C_out */
			cblas_sgemm(CblasRowMajor,
				    CblasNoTrans, CblasTrans,
				    M, cpg_out, (int)K,
				    1.0f,
				    col, (int)K,
				    w_g, (int)K,
				    0.0f,
				    out_n + g * cpg_out, C_out);
#else
			struct conv2d_nhwc_matmul_ctx mctx = {
				.col = col,
				.weight = w_g,
				.output = out_n + g * cpg_out,
				.M = M,
				.K = (int)K,
				.N = cpg_out,
				.ldc = C_out,
			};
			int n_tasks = sam3_threadpool_n_threads(pool);
			if (n_tasks < 1)
				n_tasks = 1;
			sam3_threadpool_parallel_for(pool,
						     conv2d_nhwc_matmul_fn,
						     &mctx, n_tasks);
#endif
		}
	}

	scratch->offset = saved_offset;
	return SAM3_OK;
}
