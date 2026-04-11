/*
 * src/backend/cpu/kernels/cpu_maxpool2d.c - MaxPool2D kernel
 *
 * Implements 2D max pooling with a sliding window. For each output
 * position, computes the maximum value in the corresponding kernel
 * window of the input. Parallelized over N*C planes.
 *
 * Public layout (after the NHWC migration): input [N,H,W,C],
 * output [N,OH,OW,C] where OH = (H - kernel) / stride + 1.
 * node->params[0]=kernel_size, params[1]=stride.
 *
 * The public entry point transposes NHWC input into an NCHW scratch
 * buffer, runs the legacy NCHW body, and transposes the result back
 * to NHWC. The CPU backend is test-only; Metal handles NHWC natively.
 * Byte-level transpose helpers are shared with cpu_conv2d.c via
 * file-scope exports.
 *
 * Key types:  sam3_node, sam3_tensor
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

#include <float.h>
#include <stdint.h>
#include <string.h>

/* Shared NHWC transpose helpers (defined in cpu_conv2d.c). */
void sam3_cpu_nhwc_to_nchw_bytes(const uint8_t *src, uint8_t *dst,
				 int N, int H, int W, int C,
				 size_t elem_sz);
void sam3_cpu_nchw_to_nhwc_bytes(const uint8_t *src, uint8_t *dst,
				 int N, int C, int H, int W,
				 size_t elem_sz);

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

/*
 * maxpool2d_nchw_body_f32 - Legacy NCHW F32 max pool body.
 *
 * Operates on stack tensors prepared by the NHWC shim: input
 * [N,C,H,W], output [N,C,OH,OW]. Parallelized across N*C planes.
 */
static void maxpool2d_nchw_body_f32(const struct sam3_tensor *input,
				    struct sam3_tensor *output,
				    int kernel, int stride,
				    struct sam3_threadpool *pool)
{
	struct maxpool2d_ctx ctx = {
		.input = (const float *)input->data,
		.output = (float *)output->data,
		.N_batch = input->dims[0],
		.C = input->dims[1],
		.H = input->dims[2],
		.W = input->dims[3],
		.OH = output->dims[2],
		.OW = output->dims[3],
		.kernel = kernel,
		.stride = stride,
	};

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;
	sam3_threadpool_parallel_for(pool, maxpool2d_parallel_fn,
				     &ctx, n_tasks);
}

enum sam3_error cpu_kernel_maxpool2d(const struct sam3_node *node,
				     struct sam3_arena *scratch,
				     struct sam3_threadpool *pool)
{
	if (!node->inputs[0] || !node->output) {
		sam3_log_error("maxpool2d: NULL tensor");
		return SAM3_EINVAL;
	}

	if (!scratch) {
		sam3_log_error("maxpool2d: NULL scratch arena");
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
	 * NHWC input [N,H,W,C]; NHWC output [N,OH,OW,C]. The body is
	 * NCHW-only, so we transpose into scratch, run the body, and
	 * transpose the result back. No production path runs here --
	 * Metal handles maxpool2d natively via MLX reduction.
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

	size_t elem_sz = sizeof(float);
	size_t saved_offset = scratch->offset;

	size_t in_bytes = (size_t)N_batch * C * H * W * elem_sz;
	size_t out_bytes = (size_t)N_batch * C * OH * OW * elem_sz;

	uint8_t *in_nchw = (uint8_t *)sam3_arena_alloc_raw(scratch,
							   in_bytes);
	uint8_t *out_nchw = (uint8_t *)sam3_arena_alloc_raw(scratch,
							    out_bytes);
	if (!in_nchw || !out_nchw) {
		sam3_log_error("maxpool2d: scratch OOM (need %zu bytes)",
			       in_bytes + out_bytes);
		scratch->offset = saved_offset;
		return SAM3_ENOMEM;
	}

	sam3_cpu_nhwc_to_nchw_bytes((const uint8_t *)input->data, in_nchw,
				    N_batch, H, W, C, elem_sz);

	struct sam3_tensor input_nchw = *input;
	input_nchw.dims[0] = N_batch;
	input_nchw.dims[1] = C;
	input_nchw.dims[2] = H;
	input_nchw.dims[3] = W;
	input_nchw.data = in_nchw;
	input_nchw.nbytes = in_bytes;
	sam3_tensor_compute_strides(&input_nchw);

	struct sam3_tensor output_nchw = *output;
	output_nchw.dims[0] = N_batch;
	output_nchw.dims[1] = C;
	output_nchw.dims[2] = OH;
	output_nchw.dims[3] = OW;
	output_nchw.data = out_nchw;
	output_nchw.nbytes = out_bytes;
	sam3_tensor_compute_strides(&output_nchw);

	maxpool2d_nchw_body_f32(&input_nchw, &output_nchw,
				kernel, stride, pool);

	sam3_cpu_nchw_to_nhwc_bytes(out_nchw, (uint8_t *)output->data,
				    N_batch, C, OH, OW, elem_sz);

	scratch->offset = saved_offset;
	return SAM3_OK;
}
