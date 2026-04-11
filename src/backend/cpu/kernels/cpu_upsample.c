/*
 * src/backend/cpu/kernels/cpu_upsample.c - Nearest-neighbor upsampling kernel
 *
 * Upsamples a 4D tensor by an integer scale factor using nearest-
 * neighbor interpolation. Supports both NCHW [N, C, H, W] and NHWC
 * [N, H, W, C] inputs: params[1] selects the layout. Each output pixel
 * at (y, x) copies the input pixel at (y/scale, x/scale). The outer
 * channel/row dimension is parallelised across threads via the pool.
 *
 * Key types:  sam3_node, sam3_tensor
 * Depends on: cpu_kernels.h, core/tensor.h, util/log.h, util/threadpool.h
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

struct upsample_ctx {
	const void *src;
	void       *dst;
	size_t      esz;      /* bytes per element */
	int         src_h;
	int         src_w;
	int         dst_h;
	int         dst_w;
	int         scale;
	int         nc;       /* N * C (NCHW) or N (NHWC) */
	int         c;        /* channels (NHWC only, else 0) */
};

/* NCHW nearest-neighbor upsample: parallel over N*C rows. */
static void upsample_nchw_fn(void *arg, int task_id, int n_tasks)
{
	struct upsample_ctx *ctx = (struct upsample_ctx *)arg;
	size_t esz = ctx->esz;
	int chunk = ctx->nc / n_tasks;
	int start = task_id * chunk;
	int end = (task_id == n_tasks - 1) ? ctx->nc : start + chunk;

	int src_plane = ctx->src_h * ctx->src_w;
	int dst_plane = ctx->dst_h * ctx->dst_w;

	for (int nc = start; nc < end; nc++) {
		const char *sp = (const char *)ctx->src +
				 (size_t)nc * src_plane * esz;
		char *dp = (char *)ctx->dst +
			   (size_t)nc * dst_plane * esz;

		for (int y = 0; y < ctx->dst_h; y++) {
			int sy = y / ctx->scale;
			for (int x = 0; x < ctx->dst_w; x++) {
				int sx = x / ctx->scale;
				memcpy(dp + (size_t)(y * ctx->dst_w + x) * esz,
				       sp + (size_t)(sy * ctx->src_w + sx) * esz,
				       esz);
			}
		}
	}
}

/*
 * NHWC nearest-neighbor upsample: parallel over output rows (N*dst_h).
 * For each output (y, x) pixel, copy the full C-wide channel vector
 * from the source pixel at (y/scale, x/scale).
 */
static void upsample_nhwc_fn(void *arg, int task_id, int n_tasks)
{
	struct upsample_ctx *ctx = (struct upsample_ctx *)arg;
	size_t esz = ctx->esz;
	size_t row_bytes = (size_t)ctx->c * esz;
	int total_rows = ctx->nc * ctx->dst_h;
	int chunk = total_rows / n_tasks;
	int start = task_id * chunk;
	int end = (task_id == n_tasks - 1) ? total_rows : start + chunk;

	size_t src_row_stride = (size_t)ctx->src_w * row_bytes;
	size_t dst_row_stride = (size_t)ctx->dst_w * row_bytes;
	size_t src_img_stride = (size_t)ctx->src_h * src_row_stride;
	size_t dst_img_stride = (size_t)ctx->dst_h * dst_row_stride;

	for (int r = start; r < end; r++) {
		int n = r / ctx->dst_h;
		int y = r % ctx->dst_h;
		int sy = y / ctx->scale;

		const char *src_row = (const char *)ctx->src +
				      (size_t)n * src_img_stride +
				      (size_t)sy * src_row_stride;
		char *dst_row = (char *)ctx->dst +
				(size_t)n * dst_img_stride +
				(size_t)y * dst_row_stride;

		for (int x = 0; x < ctx->dst_w; x++) {
			int sx = x / ctx->scale;
			memcpy(dst_row + (size_t)x * row_bytes,
			       src_row + (size_t)sx * row_bytes,
			       row_bytes);
		}
	}
}

enum sam3_error cpu_kernel_upsample(const struct sam3_node *node,
				     struct sam3_threadpool *pool)
{
	if (!node->inputs[0] || !node->output) {
		sam3_log_error("upsample: NULL tensor");
		return SAM3_EINVAL;
	}

	const struct sam3_tensor *inp = node->inputs[0];
	struct sam3_tensor *out = node->output;
	int scale = node->params[0];
	int nhwc = node->params[1];

	if (inp->n_dims != 4) {
		sam3_log_error("upsample: input must be 4D, got %dD",
			       inp->n_dims);
		return SAM3_EINVAL;
	}

	if (scale < 1) {
		sam3_log_error("upsample: scale must be >= 1, got %d",
			       scale);
		return SAM3_EINVAL;
	}

	int n, c, src_h, src_w;
	if (nhwc) {
		n     = inp->dims[0];
		src_h = inp->dims[1];
		src_w = inp->dims[2];
		c     = inp->dims[3];
	} else {
		n     = inp->dims[0];
		c     = inp->dims[1];
		src_h = inp->dims[2];
		src_w = inp->dims[3];
	}
	int dst_h = src_h * scale;
	int dst_w = src_w * scale;

	/* Validate output shape */
	if (nhwc) {
		if (out->dims[0] != n || out->dims[1] != dst_h ||
		    out->dims[2] != dst_w || out->dims[3] != c) {
			sam3_log_error("upsample: NHWC output shape "
				       "mismatch, expected [%d,%d,%d,%d]",
				       n, dst_h, dst_w, c);
			return SAM3_EINVAL;
		}
	} else {
		if (out->dims[0] != n || out->dims[1] != c ||
		    out->dims[2] != dst_h || out->dims[3] != dst_w) {
			sam3_log_error("upsample: NCHW output shape "
				       "mismatch, expected [%d,%d,%d,%d]",
				       n, c, dst_h, dst_w);
			return SAM3_EINVAL;
		}
	}

	struct upsample_ctx ctx = {
		.src   = inp->data,
		.dst   = out->data,
		.esz   = sam3_dtype_size(inp->dtype),
		.src_h = src_h,
		.src_w = src_w,
		.dst_h = dst_h,
		.dst_w = dst_w,
		.scale = scale,
		.nc    = nhwc ? n : (n * c),
		.c     = nhwc ? c : 0,
	};

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

	if (nhwc) {
		int rows = ctx.nc * ctx.dst_h;
		if (n_tasks > rows)
			n_tasks = rows;
		if (n_tasks < 1)
			n_tasks = 1;
		sam3_threadpool_parallel_for(pool, upsample_nhwc_fn,
					      &ctx, n_tasks);
	} else {
		if (n_tasks > ctx.nc)
			n_tasks = ctx.nc;
		if (n_tasks < 1)
			n_tasks = 1;
		sam3_threadpool_parallel_for(pool, upsample_nchw_fn,
					      &ctx, n_tasks);
	}

	return SAM3_OK;
}
