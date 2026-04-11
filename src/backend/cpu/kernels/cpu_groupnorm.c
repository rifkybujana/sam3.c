/*
 * src/backend/cpu/kernels/cpu_groupnorm.c - Group normalization kernel
 *
 * Computes group normalization on 4D NHWC tensors [N, H, W, C]. For
 * each group of C/num_groups channels, normalize across (channels,
 * H, W):
 *   out = (x - mean) / sqrt(var + eps) * gamma + beta
 * inputs[0]=input, inputs[1]=gamma [C], inputs[2]=beta [C].
 * params[0]=num_groups. eps is fixed at 1e-5.
 *
 * Key types:  sam3_node, sam3_tensor
 * Depends on: cpu_kernels.h, core/tensor.h, util/threadpool.h
 * Used by:    cpu_dispatch.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "cpu_kernels.h"
#include "core/tensor.h"
#include "util/log.h"
#include "util/threadpool.h"

#include <math.h>

#define GROUPNORM_EPS 1e-5f

/*
 * groupnorm_group_nhwc - Normalize one group for one batch element on
 *                        NHWC data.
 *
 * NHWC storage interleaves channels, so reductions and writes use
 * stride C between spatial samples of the same channel.
 *
 * @in:                 Input data pointer at start of this batch
 *                      element [H, W, C]
 * @out:                Output data pointer at start of this batch
 *                      element
 * @group:              Group index (0..num_groups-1)
 * @channels_per_group: C / num_groups
 * @hw:                 H * W (spatial elements per channel)
 * @C:                  Total channels
 * @gamma:              Per-channel scale [C] or NULL
 * @beta:               Per-channel bias [C] or NULL
 */
static void groupnorm_group_nhwc(const float *in, float *out,
				  int group, int channels_per_group,
				  int hw, int C,
				  const float *gamma, const float *beta)
{
	int c_start = group * channels_per_group;
	int group_size = channels_per_group * hw;

	/* Mean over the group */
	float sum = 0.0f;
	for (int p = 0; p < hw; p++) {
		const float *pix = in + (size_t)p * C + c_start;
		for (int c = 0; c < channels_per_group; c++)
			sum += pix[c];
	}
	float mean = sum / (float)group_size;

	/* Variance over the group */
	float var_sum = 0.0f;
	for (int p = 0; p < hw; p++) {
		const float *pix = in + (size_t)p * C + c_start;
		for (int c = 0; c < channels_per_group; c++) {
			float d = pix[c] - mean;
			var_sum += d * d;
		}
	}
	float inv_std = 1.0f / sqrtf(var_sum / (float)group_size +
				      GROUPNORM_EPS);

	/* Normalize, scale, shift */
	for (int p = 0; p < hw; p++) {
		const float *pix_in = in + (size_t)p * C + c_start;
		float *pix_out = out + (size_t)p * C + c_start;
		for (int c = 0; c < channels_per_group; c++) {
			int abs_c = c_start + c;
			float g = gamma ? gamma[abs_c] : 1.0f;
			float b = beta ? beta[abs_c] : 0.0f;
			pix_out[c] = (pix_in[c] - mean) * inv_std * g + b;
		}
	}
}

/* --- Parallel dispatch --- */

struct groupnorm_par_ctx {
	const float *in;
	float       *out;
	int          num_groups;
	int          channels_per_group;
	int          hw;
	int          batch_size;
	const float *gamma;
	const float *beta;
};

static void groupnorm_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct groupnorm_par_ctx *ctx = (struct groupnorm_par_ctx *)arg;
	int total_groups = ctx->batch_size * ctx->num_groups;
	int chunk = total_groups / n_tasks;
	int start = task_id * chunk;
	int end = (task_id == n_tasks - 1) ? total_groups : start + chunk;

	if (start >= end)
		return;

	int C = ctx->num_groups * ctx->channels_per_group;
	int batch_stride = C * ctx->hw;

	for (int idx = start; idx < end; idx++) {
		int n = idx / ctx->num_groups;
		int g = idx % ctx->num_groups;
		groupnorm_group_nhwc(
			ctx->in + (size_t)n * batch_stride,
			ctx->out + (size_t)n * batch_stride,
			g, ctx->channels_per_group, ctx->hw, C,
			ctx->gamma, ctx->beta);
	}
}

enum sam3_error cpu_kernel_groupnorm(const struct sam3_node *node,
				     struct sam3_threadpool *pool)
{
	if (!node->inputs[0] || !node->output) {
		sam3_log_error("groupnorm: NULL tensor");
		return SAM3_EINVAL;
	}

	struct sam3_tensor *in = node->inputs[0];
	int num_groups = node->params[0];

	if (in->n_dims != 4) {
		sam3_log_error("groupnorm: expected 4D input, got %dD",
			       in->n_dims);
		return SAM3_EINVAL;
	}

	if (in->dtype != SAM3_DTYPE_F32) {
		sam3_log_error("groupnorm: unsupported dtype %d", in->dtype);
		return SAM3_EINVAL;
	}

	int N = in->dims[0];
	int H = in->dims[1];
	int W = in->dims[2];
	int C = in->dims[3];

	if (C % num_groups != 0) {
		sam3_log_error("groupnorm: C=%d not divisible by groups=%d",
			       C, num_groups);
		return SAM3_EINVAL;
	}

	const float *gamma = NULL;
	const float *beta = NULL;

	if (node->n_inputs > 1 && node->inputs[1])
		gamma = (const float *)node->inputs[1]->data;
	if (node->n_inputs > 2 && node->inputs[2])
		beta = (const float *)node->inputs[2]->data;

	struct groupnorm_par_ctx ctx = {
		.in                 = (const float *)in->data,
		.out                = (float *)node->output->data,
		.num_groups         = num_groups,
		.channels_per_group = C / num_groups,
		.hw                 = H * W,
		.batch_size         = N,
		.gamma              = gamma,
		.beta               = beta,
	};

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

	sam3_threadpool_parallel_for(pool, groupnorm_parallel_fn, &ctx,
				     n_tasks);

	return SAM3_OK;
}
