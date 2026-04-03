/*
 * src/backend/cpu/kernels/cpu_maxpool2d.c - MaxPool2D kernel
 *
 * Implements 2D max pooling with a sliding window. For each output
 * position, computes the maximum value in the corresponding kernel
 * window of the input. Parallelized over N*C planes.
 *
 * Layout: input [N,C,H,W], output [N,C,OH,OW]
 * where OH = (H - kernel) / stride + 1.
 * node->params[0]=kernel_size, params[1]=stride.
 *
 * Key types:  sam3_node, sam3_tensor
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

struct maxpool2d_ctx {
	const float *input;
	float       *output;
	int          N_batch;
	int          C;
	int          H;
	int          W;
	int          OH;
	int          OW;
	int          kernel;
	int          stride;
};

static void maxpool2d_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct maxpool2d_ctx *ctx = (struct maxpool2d_ctx *)arg;
	int total_planes = ctx->N_batch * ctx->C;
	int chunk = total_planes / n_tasks;
	int start = task_id * chunk;
	int end = (task_id == n_tasks - 1) ? total_planes : start + chunk;

	if (start >= end)
		return;

	int H = ctx->H;
	int W = ctx->W;
	int OH = ctx->OH;
	int OW = ctx->OW;
	int kernel = ctx->kernel;
	int stride = ctx->stride;

	for (int plane = start; plane < end; plane++) {
		const float *in_p = ctx->input + (size_t)plane * H * W;
		float *out_p = ctx->output + (size_t)plane * OH * OW;

		for (int oh = 0; oh < OH; oh++) {
			for (int ow = 0; ow < OW; ow++) {
				float maxval = -FLT_MAX;
				int ih_start = oh * stride;
				int iw_start = ow * stride;

				for (int kh = 0; kh < kernel; kh++) {
					for (int kw = 0; kw < kernel; kw++) {
						float v = in_p[(ih_start + kh) * W +
							       (iw_start + kw)];
						if (v > maxval)
							maxval = v;
					}
				}
				out_p[oh * OW + ow] = maxval;
			}
		}
	}
}

enum sam3_error cpu_kernel_maxpool2d(const struct sam3_node *node,
				     struct sam3_threadpool *pool)
{
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

	if (input->n_dims != 4) {
		sam3_log_error("maxpool2d: expected 4D input");
		return SAM3_EINVAL;
	}

	int N_batch = input->dims[0];
	int C = input->dims[1];
	int H = input->dims[2];
	int W = input->dims[3];

	int kernel = node->params[0] > 0 ? node->params[0] : 2;
	int stride = node->params[1] > 0 ? node->params[1] : kernel;

	int OH = (H - kernel) / stride + 1;
	int OW = (W - kernel) / stride + 1;

	if (output->dims[2] != OH || output->dims[3] != OW) {
		sam3_log_error("maxpool2d: output size mismatch "
			       "expected [%d,%d] got [%d,%d]",
			       OH, OW, output->dims[2], output->dims[3]);
		return SAM3_EINVAL;
	}

	struct maxpool2d_ctx ctx = {
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
	sam3_threadpool_parallel_for(pool, maxpool2d_parallel_fn,
				     &ctx, n_tasks);

	return SAM3_OK;
}
