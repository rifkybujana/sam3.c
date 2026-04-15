/*
 * src/backend/cpu/kernels/cpu_maxpool2d.c - Native NHWC MaxPool2D
 *
 * Implements 2D max pooling directly in NHWC layout. For each output
 * position, computes the maximum value in the corresponding kernel
 * window across all channels simultaneously. No layout conversions
 * and no scratch memory required.
 *
 * Public layout: input [N,H,W,C], output [N,OH,OW,C] where
 * OH = (H - kernel) / stride + 1.
 * node->params[0]=kernel_size, params[1]=stride.
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

#include <float.h>
#include <string.h>

struct maxpool2d_nhwc_ctx {
	const float *input;   /* [N, H, W, C] */
	float       *output;  /* [N, OH, OW, C] */
	int          N_batch;
	int          C;
	int          H;
	int          W;
	int          OH;
	int          OW;
	int          kernel;
	int          stride;
};

/*
 * Native NHWC max pool. For each output pixel (oh, ow), scan the
 * kernel window and keep the per-channel maximum. Channels are
 * contiguous in NHWC, so the inner loop over C is cache-friendly.
 *
 * Parallelized over N_batch * OH output rows.
 */
static void maxpool2d_nhwc_fn(void *arg, int task_id, int n_tasks)
{
	struct maxpool2d_nhwc_ctx *ctx =
		(struct maxpool2d_nhwc_ctx *)arg;
	int total_rows = ctx->N_batch * ctx->OH;
	int chunk = total_rows / n_tasks;
	int start = task_id * chunk;
	int end = (task_id == n_tasks - 1)
		? total_rows : start + chunk;

	if (start >= end)
		return;

	int C = ctx->C;
	int H = ctx->H;
	int W = ctx->W;
	int OW = ctx->OW;
	int kernel = ctx->kernel;
	int stride = ctx->stride;

	for (int row = start; row < end; row++) {
		int n = row / ctx->OH;
		int oh = row % ctx->OH;

		for (int ow = 0; ow < OW; ow++) {
			float *out_px = ctx->output +
				((size_t)n * ctx->OH * OW +
				 (size_t)oh * OW + ow) * C;

			/* Initialize to first kernel position */
			int ih0 = oh * stride;
			int iw0 = ow * stride;
			const float *first = ctx->input +
				((size_t)n * H * W +
				 (size_t)ih0 * W + iw0) * C;
			memcpy(out_px, first,
			       (size_t)C * sizeof(float));

			/* Scan remaining kernel positions */
			for (int kh = 0; kh < kernel; kh++) {
				int ih = ih0 + kh;
				for (int kw = 0; kw < kernel; kw++) {
					int iw = iw0 + kw;
					if (kh == 0 && kw == 0)
						continue;
					if (ih >= H || iw >= W)
						continue;

					const float *in_px =
						ctx->input +
						((size_t)n * H * W +
						 (size_t)ih * W +
						 iw) * C;
					for (int c = 0; c < C; c++) {
						if (in_px[c] > out_px[c])
							out_px[c] =
								in_px[c];
					}
				}
			}
		}
	}
}

enum sam3_error cpu_kernel_maxpool2d(const struct sam3_node *node,
				     struct sam3_arena *scratch,
				     struct sam3_threadpool *pool)
{
	(void)scratch;

	if (!node->inputs[0] || !node->output) {
		sam3_log_error("maxpool2d: NULL tensor");
		return SAM3_EINVAL;
	}

	struct sam3_tensor *input = node->inputs[0];
	struct sam3_tensor *output = node->output;

	if (input->dtype != SAM3_DTYPE_F32) {
		sam3_log_error("maxpool2d: unsupported dtype");
		return SAM3_EINVAL;
	}

	if (input->n_dims != 4 || output->n_dims != 4) {
		sam3_log_error("maxpool2d: expected 4D tensors");
		return SAM3_EINVAL;
	}

	/*
	 * Native NHWC max pool: for each output pixel, scan the
	 * kernel window and keep per-channel maximum. Channels are
	 * contiguous in memory, giving good cache behavior. Zero
	 * scratch memory needed.
	 */
	int N_batch = input->dims[0];
	int H = input->dims[1];
	int W = input->dims[2];
	int C = input->dims[3];

	int kernel = node->params[0] > 0 ? node->params[0] : 2;
	int stride = node->params[1] > 0 ? node->params[1] : kernel;

	int OH = (H - kernel) / stride + 1;
	int OW = (W - kernel) / stride + 1;

	if (output->dims[0] != N_batch || output->dims[1] != OH ||
	    output->dims[2] != OW || output->dims[3] != C) {
		sam3_log_error("maxpool2d: output shape mismatch");
		return SAM3_EINVAL;
	}

	struct maxpool2d_nhwc_ctx ctx = {
		.input = (const float *)input->data,
		.output = (float *)output->data,
		.N_batch = N_batch,
		.C = C,
		.H = H, .W = W,
		.OH = OH, .OW = OW,
		.kernel = kernel,
		.stride = stride,
	};

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;
	sam3_threadpool_parallel_for(pool, maxpool2d_nhwc_fn,
				     &ctx, n_tasks);
	return SAM3_OK;
}
