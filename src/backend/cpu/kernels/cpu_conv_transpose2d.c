/*
 * src/backend/cpu/kernels/cpu_conv_transpose2d.c - Transposed Conv2D kernel
 *
 * Implements 2D transposed convolution (deconvolution) using the
 * scatter-accumulate approach: for each input pixel, scatter its
 * contribution through the kernel into the output. The output buffer
 * is zeroed before accumulation begins.
 *
 * Public layout (after the NHWC migration): input [N,H,W,C_in],
 * weight [C_out,KH,KW,C_in] (OHWI), output [N,OH,OW,C_out] where
 * OH = (H-1)*stride - 2*pad + KH. node->params[0]=stride,
 * params[1]=padding.
 *
 * The public entry point transposes NHWC inputs and OHWI weights into
 * NCHW/IOHW scratch buffers, runs the legacy NCHW scatter body, and
 * transposes the result back to NHWC. The CPU backend is test-only;
 * Metal handles the NHWC path natively via MLX. Byte-level transpose
 * helpers are shared with cpu_conv2d.c via file-scope exports.
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

#include <stdint.h>
#include <string.h>

/* Shared NHWC transpose helpers (defined in cpu_conv2d.c). */
void sam3_cpu_nhwc_to_nchw_bytes(const uint8_t *src, uint8_t *dst,
				 int N, int H, int W, int C,
				 size_t elem_sz);
void sam3_cpu_nchw_to_nhwc_bytes(const uint8_t *src, uint8_t *dst,
				 int N, int C, int H, int W,
				 size_t elem_sz);
void sam3_cpu_ohwi_to_iohw_bytes(const uint8_t *src, uint8_t *dst,
				 int OC, int KH, int KW, int IC,
				 size_t elem_sz);

struct conv_transpose2d_ctx {
	const float *input;
	const float *weight;
	float       *output;
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

/*
 * conv_transpose2d_nchw_body_f32 - Legacy NCHW F32 body.
 *
 * Operates on stack tensors prepared by the NHWC shim: input NCHW
 * [N,C_in,H,W], weight IOHW [C_in,C_out,KH,KW], output NCHW
 * [N,C_out,OH,OW]. The parallel scatter-accumulate body zeros each
 * output plane before writing so no prior clear is required.
 */
static enum sam3_error
conv_transpose2d_nchw_body_f32(const struct sam3_tensor *input,
			       const struct sam3_tensor *weight,
			       struct sam3_tensor *output,
			       int stride, int pad,
			       struct sam3_threadpool *pool)
{
	int N_batch = input->dims[0];
	int C_in = input->dims[1];
	int H = input->dims[2];
	int W = input->dims[3];

	int C_out = weight->dims[1];
	int KH = weight->dims[2];
	int KW = weight->dims[3];

	int OH = output->dims[2];
	int OW = output->dims[3];

	struct conv_transpose2d_ctx ctx = {
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
	sam3_threadpool_parallel_for(pool, conv_transpose2d_parallel_fn,
				     &ctx, n_tasks);
	return SAM3_OK;
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

	if (!scratch) {
		sam3_log_error("conv_transpose2d: NULL scratch arena");
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
	 * NHWC input [N,H,W,C_in]; OHWI weight [C_out,KH,KW,C_in];
	 * NHWC output [N,OH,OW,C_out]. The CPU body is NCHW-only, so
	 * we transpose both inputs into scratch (weight to IOHW via
	 * sam3_cpu_ohwi_to_iohw_bytes to match the PyTorch scatter
	 * layout), run the body, and transpose the result back to
	 * NHWC. No production path runs here — Metal handles
	 * conv_transpose2d natively via MLX.
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
		sam3_log_error("conv_transpose2d: channel mismatch %d != %d",
			       KIC, C_in);
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

	size_t elem_sz = sizeof(float);
	size_t saved_offset = scratch->offset;

	size_t in_bytes = (size_t)N_batch * C_in * H * W * elem_sz;
	size_t wt_bytes = (size_t)C_in * C_out * KH * KW * elem_sz;
	size_t out_bytes = (size_t)N_batch * C_out * OH * OW * elem_sz;

	uint8_t *in_nchw = (uint8_t *)sam3_arena_alloc_raw(scratch,
							   in_bytes);
	uint8_t *wt_iohw = (uint8_t *)sam3_arena_alloc_raw(scratch,
							   wt_bytes);
	uint8_t *out_nchw = (uint8_t *)sam3_arena_alloc_raw(scratch,
							    out_bytes);
	if (!in_nchw || !wt_iohw || !out_nchw) {
		sam3_log_error("conv_transpose2d: scratch OOM (need %zu)",
			       in_bytes + wt_bytes + out_bytes);
		scratch->offset = saved_offset;
		return SAM3_ENOMEM;
	}

	sam3_cpu_nhwc_to_nchw_bytes((const uint8_t *)input->data, in_nchw,
				    N_batch, H, W, C_in, elem_sz);
	sam3_cpu_ohwi_to_iohw_bytes((const uint8_t *)weight->data, wt_iohw,
				    C_out, KH, KW, C_in, elem_sz);

	struct sam3_tensor input_nchw = *input;
	input_nchw.dims[0] = N_batch;
	input_nchw.dims[1] = C_in;
	input_nchw.dims[2] = H;
	input_nchw.dims[3] = W;
	input_nchw.data = in_nchw;
	input_nchw.nbytes = in_bytes;
	sam3_tensor_compute_strides(&input_nchw);

	struct sam3_tensor weight_iohw = *weight;
	weight_iohw.dims[0] = C_in;
	weight_iohw.dims[1] = C_out;
	weight_iohw.dims[2] = KH;
	weight_iohw.dims[3] = KW;
	weight_iohw.data = wt_iohw;
	weight_iohw.nbytes = wt_bytes;
	sam3_tensor_compute_strides(&weight_iohw);

	struct sam3_tensor output_nchw = *output;
	output_nchw.dims[0] = N_batch;
	output_nchw.dims[1] = C_out;
	output_nchw.dims[2] = OH;
	output_nchw.dims[3] = OW;
	output_nchw.data = out_nchw;
	output_nchw.nbytes = out_bytes;
	sam3_tensor_compute_strides(&output_nchw);

	enum sam3_error err = conv_transpose2d_nchw_body_f32(&input_nchw,
							     &weight_iohw,
							     &output_nchw,
							     stride, pad,
							     pool);
	if (err != SAM3_OK) {
		scratch->offset = saved_offset;
		return err;
	}

	sam3_cpu_nchw_to_nhwc_bytes(out_nchw, (uint8_t *)output->data,
				    N_batch, C_out, OH, OW, elem_sz);

	scratch->offset = saved_offset;
	return SAM3_OK;
}
