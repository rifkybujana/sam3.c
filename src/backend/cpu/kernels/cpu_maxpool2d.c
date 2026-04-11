/*
 * src/backend/cpu/kernels/cpu_maxpool2d.c - MaxPool2D kernel
 *
 * Implements 2D max pooling with a sliding window. For each output
 * position, computes the maximum value in the corresponding kernel
 * window of the input. Parallelized over N*C planes.
 *
 * Layout: input [N,C,H,W], output [N,C,OH,OW]
 * where OH = (H - kernel) / stride + 1.
 * node->params[0]=kernel_size, params[1]=stride, params[2]=NHWC flag.
 *
 * When params[2]==1 the NHWC dispatch wrapper hands the node to
 * cpu_maxpool2d_nhwc_wrap (defined below), which permutes [N,H,W,C]
 * to [N,C,H,W] into the scratch arena, calls cpu_kernel_maxpool2d,
 * and permutes the result back. Used only by the FPN neck's 0.5x
 * stage on the CPU backend; Metal handles NHWC natively.
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

	/*
	 * params[2] == 1 marks an NHWC input. The CPU kernel only
	 * implements NCHW today; no CPU test exercises the NHWC path
	 * so rejecting here is enough to satisfy the layout migration
	 * plan ("any test that was green before must still be green").
	 */
	if (node->params[2]) {
		sam3_log_error("maxpool2d: NHWC path not implemented on CPU");
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

/* ── NHWC transpose wrap ───────────────────────────────────────────── */

/*
 * Byte-level NHWC -> NCHW transpose. Mirrors the helper in
 * cpu_conv2d.c but is duplicated here to keep this file's linkage
 * unit self-contained; both helpers are scalar scaffolding for the
 * test-only CPU NHWC path.
 */
static void nhwc_to_nchw_bytes(const uint8_t *src, uint8_t *dst,
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
						(((size_t)n * C + c) * hw
						 + (size_t)h * W + w) *
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
 * Byte-level NCHW -> NHWC transpose (inverse of the above).
 */
static void nchw_to_nhwc_bytes(const uint8_t *src, uint8_t *dst,
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
						(((size_t)n * C + c) * hw
						 + (size_t)h * W + w) *
						elem_sz;
					memcpy(dst_px + (size_t)c * elem_sz,
					       src_px, elem_sz);
				}
			}
		}
	}
}

enum sam3_error cpu_maxpool2d_nhwc_wrap(const struct sam3_node *node,
					struct sam3_arena *scratch,
					struct sam3_threadpool *pool)
{
	if (!node || !scratch) {
		sam3_log_error("maxpool2d_nhwc_wrap: NULL argument");
		return SAM3_EINVAL;
	}

	if (node->n_inputs < 1 || !node->inputs[0] || !node->output) {
		sam3_log_error("maxpool2d_nhwc_wrap: NULL tensor");
		return SAM3_EINVAL;
	}

	struct sam3_tensor *input = node->inputs[0];
	struct sam3_tensor *output = node->output;

	if (input->n_dims != 4 || output->n_dims != 4) {
		sam3_log_error("maxpool2d_nhwc_wrap: expected 4D tensors");
		return SAM3_EINVAL;
	}

	size_t elem_sz = sam3_dtype_size(input->dtype);
	if (elem_sz == 0) {
		sam3_log_error("maxpool2d_nhwc_wrap: bad dtype %d",
			       input->dtype);
		return SAM3_EINVAL;
	}

	/* NHWC input [N, H, W, C] */
	int N_batch = input->dims[0];
	int H = input->dims[1];
	int W = input->dims[2];
	int C = input->dims[3];

	/* NHWC output [N, OH, OW, C] */
	int OH = output->dims[1];
	int OW = output->dims[2];
	if (output->dims[0] != N_batch || output->dims[3] != C) {
		sam3_log_error("maxpool2d_nhwc_wrap: output shape mismatch");
		return SAM3_EINVAL;
	}

	size_t saved_offset = scratch->offset;

	size_t in_bytes = (size_t)N_batch * C * H * W * elem_sz;
	size_t out_bytes = (size_t)N_batch * C * OH * OW * elem_sz;

	uint8_t *in_nchw = (uint8_t *)sam3_arena_alloc_raw(scratch,
							   in_bytes);
	uint8_t *out_nchw = (uint8_t *)sam3_arena_alloc_raw(scratch,
							    out_bytes);
	if (!in_nchw || !out_nchw) {
		sam3_log_error("maxpool2d_nhwc_wrap: scratch OOM "
			       "(need %zu bytes)",
			       in_bytes + out_bytes);
		scratch->offset = saved_offset;
		return SAM3_ENOMEM;
	}

	nhwc_to_nchw_bytes((const uint8_t *)input->data, in_nchw,
			   N_batch, H, W, C, elem_sz);

	/* Build stack tensors with NCHW dims pointing at scratch. */
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

	/*
	 * Build a local node copy with params[2]=0 so the kernel
	 * takes the legacy NCHW path on the scratch buffers. Only
	 * inputs/output and params[2] are rewritten; kernel_size and
	 * stride (params[0]/[1]) stay identical.
	 */
	struct sam3_node nchw_node = *node;
	nchw_node.inputs[0] = &input_nchw;
	nchw_node.output = &output_nchw;
	nchw_node.params[2] = 0;

	enum sam3_error err = cpu_kernel_maxpool2d(&nchw_node, pool);
	if (err != SAM3_OK) {
		scratch->offset = saved_offset;
		return err;
	}

	nchw_to_nhwc_bytes(out_nchw, (uint8_t *)output->data,
			   N_batch, C, OH, OW, elem_sz);

	scratch->offset = saved_offset;
	return SAM3_OK;
}
