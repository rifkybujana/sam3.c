/*
 * src/backend/cpu/kernels/cpu_conv_transpose2d.c - Native NHWC Transposed Conv2D
 *
 * Implements 2D transposed convolution directly in NHWC layout using
 * a gather approach: for each output pixel, find contributing input
 * pixels and accumulate via dot products with the OHWI weight. No
 * layout conversions and no scratch memory required.
 *
 * Public layout: input [N,H,W,C_in], weight [C_out,KH,KW,C_in]
 * (OHWI), output [N,OH,OW,C_out] where OH = (H-1)*stride - 2*pad + KH.
 * node->params[0]=stride, params[1]=padding.
 *
 * Key types:  sam3_node, sam3_tensor, sam3_arena
 * Depends on: cpu_kernels.h, core/tensor.h
 * Used by:    cpu_dispatch.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "cpu_kernels.h"
#include "core/tensor.h"
#include "util/log.h"
#include "util/threadpool.h"

#include <string.h>

struct conv_transpose2d_nhwc_ctx {
	const float *input;   /* [N, H, W, C_in] */
	const float *weight;  /* [C_out, KH, KW, C_in] OHWI */
	float       *output;  /* [N, OH, OW, C_out] */
	int          N_batch;
	int          C_in;
	int          C_out;
	int          H;
	int          W;
	int          KH;
	int          KW;
	int          OH;
	int          OW;
	int          stride;
	int          pad;
};

/*
 * Gather-based NHWC transposed conv2d. For each output pixel (oh, ow),
 * find input pixels that contribute via the inverse stride formula:
 *   ih = (oh + pad - kh) / stride  (when evenly divisible)
 *   iw = (ow + pad - kw) / stride
 *
 * Then accumulate: out[co] += dot(input_pixel[:C_in], weight[co,kh,kw,:])
 *
 * Parallelized over N_batch * OH output rows.
 */
static void conv_transpose2d_nhwc_fn(void *arg, int task_id,
				      int n_tasks)
{
	struct conv_transpose2d_nhwc_ctx *ctx =
		(struct conv_transpose2d_nhwc_ctx *)arg;
	int total_rows = ctx->N_batch * ctx->OH;
	int chunk = total_rows / n_tasks;
	int start = task_id * chunk;
	int end = (task_id == n_tasks - 1)
		? total_rows : start + chunk;

	if (start >= end)
		return;

	int C_in = ctx->C_in;
	int C_out = ctx->C_out;
	int H = ctx->H;
	int W = ctx->W;
	int KH = ctx->KH;
	int KW = ctx->KW;
	int OW = ctx->OW;
	int stride = ctx->stride;
	int pad = ctx->pad;
	size_t kk_cin = (size_t)KH * KW * C_in;

	for (int row = start; row < end; row++) {
		int n = row / ctx->OH;
		int oh = row % ctx->OH;

		for (int ow = 0; ow < OW; ow++) {
			float *out_px = ctx->output +
				((size_t)n * ctx->OH * OW +
				 (size_t)oh * OW + ow) * C_out;
			memset(out_px, 0,
			       (size_t)C_out * sizeof(float));

			for (int kh = 0; kh < KH; kh++) {
				int oh_off = oh + pad - kh;
				if (oh_off < 0 ||
				    oh_off % stride != 0)
					continue;
				int ih = oh_off / stride;
				if (ih >= H)
					continue;

				for (int kw = 0; kw < KW; kw++) {
					int ow_off = ow + pad - kw;
					if (ow_off < 0 ||
					    ow_off % stride != 0)
						continue;
					int iw = ow_off / stride;
					if (iw >= W)
						continue;

					const float *in_px =
						ctx->input +
						((size_t)n * H * W +
						 (size_t)ih * W + iw) *
						C_in;

					for (int co = 0; co < C_out;
					     co++) {
						const float *w =
							ctx->weight +
							(size_t)co *
							kk_cin +
							((size_t)kh * KW
							 + kw) * C_in;
						float dot = 0;
						for (int ci = 0;
						     ci < C_in; ci++)
							dot += in_px[ci] *
								w[ci];
						out_px[co] += dot;
					}
				}
			}
		}
	}
}

enum sam3_error cpu_kernel_conv_transpose2d(const struct sam3_node *node,
					    struct sam3_arena *scratch,
					    struct sam3_threadpool *pool)
{
	(void)scratch;

	if (node->n_inputs < 2 || !node->inputs[0] || !node->inputs[1] ||
	    !node->output) {
		sam3_log_error("conv_transpose2d: NULL tensor");
		return SAM3_EINVAL;
	}

	struct sam3_tensor *input = node->inputs[0];
	struct sam3_tensor *weight = node->inputs[1];
	struct sam3_tensor *output = node->output;

	if (input->dtype != SAM3_DTYPE_F32 ||
	    weight->dtype != SAM3_DTYPE_F32) {
		sam3_log_error("conv_transpose2d: unsupported dtype");
		return SAM3_EINVAL;
	}

	if (input->n_dims != 4 || weight->n_dims != 4 ||
	    output->n_dims != 4) {
		sam3_log_error("conv_transpose2d: expected 4D tensors");
		return SAM3_EINVAL;
	}

	/*
	 * Native NHWC conv_transpose2d: gather approach. For each
	 * output pixel, find contributing input pixels and accumulate
	 * dot products with OHWI weight. Zero scratch needed.
	 */
	int N_batch = input->dims[0];
	int H = input->dims[1];
	int W = input->dims[2];
	int C_in = input->dims[3];

	int C_out = weight->dims[0];
	int KH = weight->dims[1];
	int KW = weight->dims[2];
	int KIC = weight->dims[3];

	if (KIC != C_in) {
		sam3_log_error("conv_transpose2d: channel mismatch "
			       "%d != %d", KIC, C_in);
		return SAM3_EINVAL;
	}

	int stride = node->params[0] > 0 ? node->params[0] : 1;
	int pad = node->params[1];

	int OH = (H - 1) * stride - 2 * pad + KH;
	int OW = (W - 1) * stride - 2 * pad + KW;

	if (output->dims[0] != N_batch || output->dims[1] != OH ||
	    output->dims[2] != OW || output->dims[3] != C_out) {
		sam3_log_error("conv_transpose2d: output shape mismatch");
		return SAM3_EINVAL;
	}

	struct conv_transpose2d_nhwc_ctx ctx = {
		.input = (const float *)input->data,
		.weight = (const float *)weight->data,
		.output = (float *)output->data,
		.N_batch = N_batch,
		.C_in = C_in,
		.C_out = C_out,
		.H = H, .W = W,
		.KH = KH, .KW = KW,
		.OH = OH, .OW = OW,
		.stride = stride,
		.pad = pad,
	};

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;
	sam3_threadpool_parallel_for(pool, conv_transpose2d_nhwc_fn,
				     &ctx, n_tasks);
	return SAM3_OK;
}
