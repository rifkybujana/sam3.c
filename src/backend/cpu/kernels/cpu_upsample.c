/*
 * src/backend/cpu/kernels/cpu_upsample.c - Nearest-neighbor upsampling kernel
 *
 * Upsamples a 4D tensor [N, C, H, W] by an integer scale factor using
 * nearest-neighbor interpolation.  Each output pixel at (y, x) copies
 * the input pixel at (y/scale, x/scale).  The outer N*C dimension can
 * be parallelised across threads via the thread pool.
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
	int         nc;       /* N * C */
};

static void upsample_parallel_fn(void *arg, int task_id, int n_tasks)
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

	int n = inp->dims[0];
	int c = inp->dims[1];
	int src_h = inp->dims[2];
	int src_w = inp->dims[3];
	int dst_h = src_h * scale;
	int dst_w = src_w * scale;

	/* Validate output shape */
	if (out->dims[0] != n || out->dims[1] != c ||
	    out->dims[2] != dst_h || out->dims[3] != dst_w) {
		sam3_log_error("upsample: output shape mismatch, "
			       "expected [%d,%d,%d,%d]",
			       n, c, dst_h, dst_w);
		return SAM3_EINVAL;
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
		.nc    = n * c,
	};

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;
	if (n_tasks > ctx.nc)
		n_tasks = ctx.nc;

	sam3_threadpool_parallel_for(pool, upsample_parallel_fn, &ctx,
				     n_tasks);

	return SAM3_OK;
}
