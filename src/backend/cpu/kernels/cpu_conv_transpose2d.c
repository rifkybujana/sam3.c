/*
 * src/backend/cpu/kernels/cpu_conv_transpose2d.c - Transposed Conv2D kernel
 *
 * Implements 2D transposed convolution (deconvolution) using the
 * scatter-accumulate approach: for each input pixel, scatter its
 * contribution through the kernel into the output. The output buffer
 * is zeroed before accumulation begins.
 *
 * Layout: input [N,C_in,H,W], weight [C_in,C_out,KH,KW] (PyTorch layout),
 * output [N,C_out,OH,OW] where OH = (H-1)*stride - 2*pad + KH.
 * node->params[0]=stride, params[1]=padding, params[2]=NHWC flag.
 * When params[2]=1 the kernel forwards to cpu_conv_transpose2d_nhwc_wrap
 * (defined in cpu_conv2d.c) which transposes NHWC/OHWI to NCHW/IOHW
 * in scratch and recursively re-enters this kernel with params[2]=0.
 *
 * Key types:  sam3_node, sam3_tensor, sam3_arena
 * Depends on: cpu_kernels.h, core/tensor.h, core/alloc.h
 * Used by:    cpu_dispatch.c
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

struct conv_transpose2d_ctx {
	const float *input;
	const float *weight;
	float       *output;
	float       *scratch;
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

static void conv_transpose2d_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct conv_transpose2d_ctx *ctx =
		(struct conv_transpose2d_ctx *)arg;
	int total_planes = ctx->N_batch * ctx->C_out;
	int chunk = total_planes / n_tasks;
	int start = task_id * chunk;
	int end = (task_id == n_tasks - 1) ? total_planes : start + chunk;

	if (start >= end)
		return;

	int C_in = ctx->C_in;
	int C_out = ctx->C_out;
	int H = ctx->H;
	int W = ctx->W;
	int KH = ctx->KH;
	int KW = ctx->KW;
	int OH = ctx->OH;
	int OW = ctx->OW;
	int stride = ctx->stride;
	int pad = ctx->pad;

	for (int plane = start; plane < end; plane++) {
		int n = plane / C_out;
		int co = plane % C_out;

		float *out_plane = ctx->output +
			(size_t)n * C_out * OH * OW + (size_t)co * OH * OW;
		memset(out_plane, 0, (size_t)OH * OW * sizeof(float));

		for (int ci = 0; ci < C_in; ci++) {
			const float *in_plane = ctx->input +
				(size_t)n * C_in * H * W +
				(size_t)ci * H * W;
			const float *kern = ctx->weight +
				(size_t)ci * C_out * KH * KW +
				(size_t)co * KH * KW;

			for (int ih = 0; ih < H; ih++) {
				int oh_base = ih * stride - pad;
				int kh_start = -oh_base > 0 ? -oh_base : 0;
				int kh_end = OH - oh_base < KH
					? OH - oh_base : KH;

				for (int iw = 0; iw < W; iw++) {
					float val = in_plane[ih * W + iw];
					int ow_base = iw * stride - pad;
					int kw_start = -ow_base > 0
						? -ow_base : 0;
					int kw_end = OW - ow_base < KW
						? OW - ow_base : KW;

					for (int kh = kh_start; kh < kh_end; kh++) {
						int oh = oh_base + kh;
						float *out_row =
							out_plane + oh * OW;
						const float *kern_row =
							kern + kh * KW;
						for (int kw = kw_start; kw < kw_end; kw++) {
							out_row[ow_base + kw] +=
								val * kern_row[kw];
						}
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
	if (node->n_inputs < 2 || !node->inputs[0] || !node->inputs[1] ||
	    !node->output) {
		sam3_log_error("conv_transpose2d: NULL tensor");
		return SAM3_EINVAL;
	}

	if (node->params[2]) {
		if (!scratch) {
			sam3_log_error("conv_transpose2d: NULL scratch "
				       "arena for NHWC wrap");
			return SAM3_EINVAL;
		}
		return cpu_conv_transpose2d_nhwc_wrap(
			cpu_kernel_conv_transpose2d, node, scratch, pool);
	}

	(void)scratch;

	struct sam3_tensor *input = node->inputs[0];
	struct sam3_tensor *weight = node->inputs[1];
	struct sam3_tensor *output = node->output;

	if (input->dtype != SAM3_DTYPE_F32 ||
	    weight->dtype != SAM3_DTYPE_F32) {
		sam3_log_error("conv_transpose2d: unsupported dtype");
		return SAM3_EINVAL;
	}

	if (input->n_dims != 4 || weight->n_dims != 4) {
		sam3_log_error("conv_transpose2d: expected 4D tensors");
		return SAM3_EINVAL;
	}

	int N_batch = input->dims[0];
	int C_in = input->dims[1];
	int H = input->dims[2];
	int W = input->dims[3];

	/* Weight: [C_in, C_out, KH, KW] (PyTorch ConvTranspose2d layout) */
	int C_out = weight->dims[1];
	int KH = weight->dims[2];
	int KW = weight->dims[3];

	if (weight->dims[0] != C_in) {
		sam3_log_error("conv_transpose2d: C_in mismatch %d != %d",
			       weight->dims[0], C_in);
		return SAM3_EINVAL;
	}

	int stride = node->params[0] > 0 ? node->params[0] : 1;
	int pad = node->params[1];

	int OH = (H - 1) * stride - 2 * pad + KH;
	int OW = (W - 1) * stride - 2 * pad + KW;

	if (output->dims[2] != OH || output->dims[3] != OW) {
		sam3_log_error("conv_transpose2d: output size mismatch "
			       "expected [%d,%d] got [%d,%d]",
			       OH, OW, output->dims[2], output->dims[3]);
		return SAM3_EINVAL;
	}

	struct conv_transpose2d_ctx ctx = {
		.input = (const float *)input->data,
		.weight = (const float *)weight->data,
		.output = (float *)output->data,
		.scratch = NULL,
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
	sam3_threadpool_parallel_for(pool, conv_transpose2d_parallel_fn,
				     &ctx, n_tasks);

	return SAM3_OK;
}
