/*
 * src/backend/cpu/kernels/cpu_conv2d.c - Conv2D via im2col + matmul
 *
 * Implements 2D convolution using the im2col approach: unroll input
 * patches into a column matrix, then matrix-multiply with the weight
 * matrix. Scratch arena is used for the im2col buffer with save/restore
 * of offset so the temp buffer is freed automatically.
 *
 * Public layout (after the NHWC migration): input [N,H,W,C],
 * weight [OC,KH,KW,IC] (OHWI), output [N,OH,OW,OC].
 * node->params[0]=stride, params[1]=padding.
 *
 * The public entry point transposes NHWC inputs and OHWI weights into
 * NCHW/OIHW scratch buffers, runs the legacy NCHW im2col body, and
 * transposes the result back to NHWC. The CPU backend is test-only;
 * Metal handles the NHWC path natively via MLX. Byte-level transpose
 * helpers are shared across the f16 / bf16 / conv_transpose2d /
 * maxpool kernels via file-scope exports.
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

/* ── NHWC transpose helpers (dtype-generic, file-scope linkage) ──── */

/*
 * These helpers are used by cpu_kernel_conv2d below and also by the
 * NHWC shim in cpu_kernel_conv2d_f16 / _bf16 / cpu_kernel_conv_transpose2d.
 * They have external linkage under the sam3_cpu_nhwc_* prefix so sibling
 * kernel files can reuse them without duplication. They are not declared
 * in cpu_kernels.h because they are an implementation detail of the
 * test-only CPU NHWC path; other callers must not rely on them.
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

/*
 * conv2d_nchw_body_f32 - Legacy NCHW F32 conv2d via im2col + matmul.
 *
 * Operates on stack tensors prepared by the NHWC shim: input NCHW
 * [N,C,H,W], weight OIHW [OC,C,KH,KW], output NCHW [N,OC,OH,OW].
 * Allocates an im2col buffer from scratch and restores the offset
 * before returning.
 */
static enum sam3_error
conv2d_nchw_body_f32(const struct sam3_tensor *input,
		     const struct sam3_tensor *weight,
		     struct sam3_tensor *output,
		     int stride, int pad, int groups,
		     struct sam3_arena *scratch,
		     struct sam3_threadpool *pool)
{
#ifdef SAM3_HAS_BLAS
	(void)pool;
#endif

	int N_batch = input->dims[0];
	int C = input->dims[1];
	int H = input->dims[2];
	int W = input->dims[3];

	int OC = weight->dims[0];
	int KH = weight->dims[2];
	int KW = weight->dims[3];

	int OH = (H + 2 * pad - KH) / stride + 1;
	int OW = (W + 2 * pad - KW) / stride + 1;

	int cpg_in = C / groups;
	int cpg_out = OC / groups;

	size_t saved_offset = scratch->offset;

	/* Allocate im2col buffer: [cpg_in*KH*KW, OH*OW] */
	size_t col_size = (size_t)(cpg_in * KH * KW) * (OH * OW) *
			  sizeof(float);
	float *col = (float *)sam3_arena_alloc(scratch, col_size);
	if (!col) {
		sam3_log_error("conv2d: scratch OOM (%zu bytes)", col_size);
		scratch->offset = saved_offset;
		return SAM3_ENOMEM;
	}

	const float *w_data = (const float *)weight->data;

	for (int n = 0; n < N_batch; n++) {
		for (int g = 0; g < groups; g++) {
			const float *in_n = (const float *)input->data +
					    n * C * H * W +
					    g * cpg_in * H * W;
			float *out_n = (float *)output->data +
				       n * OC * OH * OW +
				       g * cpg_out * OH * OW;
			const float *w_g = w_data +
					   g * cpg_out * cpg_in * KH * KW;

			im2col_f32(in_n, col, cpg_in, H, W, KH, KW,
				   stride, pad, OH, OW);

#ifdef SAM3_HAS_BLAS
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				    cpg_out, OH * OW, cpg_in * KH * KW,
				    1.0f,
				    w_g, cpg_in * KH * KW,
				    col, OH * OW,
				    0.0f,
				    out_n, OH * OW);
#else
			struct conv2d_matmul_ctx mctx = {
				.a = w_g, .b = col, .c = out_n,
				.M = cpg_out,
				.K = cpg_in * KH * KW,
				.N = OH * OW,
			};
			int n_tasks = sam3_threadpool_n_threads(pool);
			if (n_tasks < 1)
				n_tasks = 1;
			sam3_threadpool_parallel_for(pool,
						     conv2d_matmul_parallel_fn,
						     &mctx, n_tasks);
#endif
		}
	}

	scratch->offset = saved_offset;
	return SAM3_OK;
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
	 * NHWC input [N,H,W,C_in]; OHWI weight [C_out,KH,KW,C_in];
	 * NHWC output [N,OH,OW,C_out]. The CPU body is NCHW-only, so
	 * we transpose both inputs into scratch, run the body, and
	 * transpose the result back to NHWC. No production path runs
	 * here — Metal handles conv2d natively via MLX.
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

	size_t elem_sz = sizeof(float);
	size_t saved_offset = scratch->offset;

	size_t in_bytes = (size_t)N_batch * C_in * H * W * elem_sz;
	size_t wt_bytes = (size_t)C_out * KIC * KH * KW * elem_sz;
	size_t out_bytes = (size_t)N_batch * C_out * OH * OW * elem_sz;

	uint8_t *in_nchw = (uint8_t *)sam3_arena_alloc_raw(scratch,
							   in_bytes);
	uint8_t *wt_oihw = (uint8_t *)sam3_arena_alloc_raw(scratch,
							   wt_bytes);
	uint8_t *out_nchw = (uint8_t *)sam3_arena_alloc_raw(scratch,
							    out_bytes);
	if (!in_nchw || !wt_oihw || !out_nchw) {
		sam3_log_error("conv2d: scratch OOM (need %zu bytes)",
			       in_bytes + wt_bytes + out_bytes);
		scratch->offset = saved_offset;
		return SAM3_ENOMEM;
	}

	sam3_cpu_nhwc_to_nchw_bytes((const uint8_t *)input->data, in_nchw,
				    N_batch, H, W, C_in, elem_sz);
	sam3_cpu_ohwi_to_oihw_bytes((const uint8_t *)weight->data, wt_oihw,
				    C_out, KH, KW, KIC, elem_sz);

	struct sam3_tensor input_nchw = *input;
	input_nchw.dims[0] = N_batch;
	input_nchw.dims[1] = C_in;
	input_nchw.dims[2] = H;
	input_nchw.dims[3] = W;
	input_nchw.data = in_nchw;
	input_nchw.nbytes = in_bytes;
	sam3_tensor_compute_strides(&input_nchw);

	struct sam3_tensor weight_oihw = *weight;
	weight_oihw.dims[0] = C_out;
	weight_oihw.dims[1] = KIC;
	weight_oihw.dims[2] = KH;
	weight_oihw.dims[3] = KW;
	weight_oihw.data = wt_oihw;
	weight_oihw.nbytes = wt_bytes;
	sam3_tensor_compute_strides(&weight_oihw);

	struct sam3_tensor output_nchw = *output;
	output_nchw.dims[0] = N_batch;
	output_nchw.dims[1] = C_out;
	output_nchw.dims[2] = OH;
	output_nchw.dims[3] = OW;
	output_nchw.data = out_nchw;
	output_nchw.nbytes = out_bytes;
	sam3_tensor_compute_strides(&output_nchw);

	enum sam3_error err = conv2d_nchw_body_f32(&input_nchw,
						   &weight_oihw,
						   &output_nchw,
						   stride, pad, groups,
						   scratch, pool);
	if (err != SAM3_OK) {
		scratch->offset = saved_offset;
		return err;
	}

	sam3_cpu_nchw_to_nhwc_bytes(out_nchw, (uint8_t *)output->data,
				    N_batch, C_out, OH, OW, elem_sz);

	scratch->offset = saved_offset;
	return SAM3_OK;
}
