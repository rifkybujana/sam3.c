/*
 * src/backend/cpu/kernels/cpu_sdpa.c - Tiled scaled dot-product attention
 *
 * Computes softmax(Q @ K^T / sqrt(head_dim) + mask) @ V without
 * materializing the full [seq, seq] attention matrix. Uses online
 * softmax (tracking running max and log-sum-exp) to process K/V in
 * tiles, accumulating the output incrementally.
 *
 * For SAM3's ViT with 5184 patches and 64-dim heads, this reduces
 * per-head memory from ~200 MB (full [5184, 5184] matrix) to ~32 KB
 * (one tile of scores).
 *
 * Key types:  sam3_node, sam3_tensor
 * Depends on: cpu_kernels.h, cpu_simd.h, core/tensor.h
 * Used by:    cpu_dispatch.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "cpu_kernels.h"
#include "cpu_simd.h"
#include "core/tensor.h"
#include "util/log.h"
#include "util/threadpool.h"

#include <math.h>
#include <float.h>
#include <string.h>

#define SDPA_TILE_K 64

/*
 * sdpa_tiled_f32 - Tiled SDPA for a single query row.
 *
 * For one query row q[head_dim], computes the attention output by
 * iterating over K/V in tiles of SDPA_TILE_K rows. Uses online
 * softmax to avoid a second pass.
 *
 * @q:        Query vector [head_dim]
 * @K:        Key matrix [seq_k, head_dim], row-major
 * @V:        Value matrix [seq_k, head_dim], row-major
 * @mask_row: Mask row [seq_k] (additive, pre-softmax), or NULL
 * @out:      Output vector [head_dim]
 * @seq_k:    Number of key/value positions
 * @head_dim: Dimension per head
 * @scale:    1/sqrt(head_dim)
 */
static void sdpa_row_f32(const float *q, const float *K, const float *V,
			  const float *mask_row, float *out,
			  int seq_k, int head_dim, float scale)
{
	float row_max = -FLT_MAX;
	float row_sum = 0.0f;
	int t, j, d;

	memset(out, 0, (size_t)head_dim * sizeof(float));

	for (t = 0; t < seq_k; t += SDPA_TILE_K) {
		int tile_end = t + SDPA_TILE_K;
		if (tile_end > seq_k)
			tile_end = seq_k;

		for (j = t; j < tile_end; j++) {
			/* Compute dot(q, K[j]) * scale */
			const float *kj = K + j * head_dim;
			float score;

#if SAM3_HAS_NEON
			float32x4_t vsum = vdupq_n_f32(0);
			d = 0;
			for (; d + 4 <= head_dim; d += 4) {
				float32x4_t vq = vld1q_f32(q + d);
				float32x4_t vk = vld1q_f32(kj + d);
				vsum = vfmaq_f32(vsum, vq, vk);
			}
			score = neon_hsum_f32(vsum);
			for (; d < head_dim; d++)
				score += q[d] * kj[d];
#else
			score = 0.0f;
			for (d = 0; d < head_dim; d++)
				score += q[d] * kj[d];
#endif
			score *= scale;

			/* Add mask if present */
			if (mask_row)
				score += mask_row[j];

			/*
			 * Online softmax update (Milakov & Gimelshein):
			 *   new_max = max(old_max, score)
			 *   correction = exp(old_max - new_max)
			 *   old_sum *= correction
			 *   old_out *= correction
			 *   w = exp(score - new_max)
			 *   out += w * V[j]
			 *   sum += w
			 */
			if (score > row_max) {
				float correction = expf(row_max - score);
				row_sum *= correction;
#if SAM3_HAS_NEON
				float32x4_t vcorr = vdupq_n_f32(correction);
				d = 0;
				for (; d + 4 <= head_dim; d += 4) {
					float32x4_t vo = vld1q_f32(out + d);
					vst1q_f32(out + d,
						  vmulq_f32(vo, vcorr));
				}
				for (; d < head_dim; d++)
					out[d] *= correction;
#else
				for (d = 0; d < head_dim; d++)
					out[d] *= correction;
#endif
				row_max = score;
			}

			float w = expf(score - row_max);
			const float *vj = V + j * head_dim;
#if SAM3_HAS_NEON
			float32x4_t vw = vdupq_n_f32(w);
			d = 0;
			for (; d + 4 <= head_dim; d += 4) {
				float32x4_t va = vld1q_f32(out + d);
				float32x4_t vv = vld1q_f32(vj + d);
				vst1q_f32(out + d,
					  vfmaq_f32(va, vw, vv));
			}
			for (; d < head_dim; d++)
				out[d] += w * vj[d];
#else
			for (d = 0; d < head_dim; d++)
				out[d] += w * vj[d];
#endif
			row_sum += w;
		}
	}

	/* Normalize by sum */
	if (row_sum > 0.0f) {
		float inv_sum = 1.0f / row_sum;
#if SAM3_HAS_NEON
		float32x4_t vinv = vdupq_n_f32(inv_sum);
		d = 0;
		for (; d + 4 <= head_dim; d += 4) {
			float32x4_t vo = vld1q_f32(out + d);
			vst1q_f32(out + d, vmulq_f32(vo, vinv));
		}
		for (; d < head_dim; d++)
			out[d] *= inv_sum;
#else
		for (d = 0; d < head_dim; d++)
			out[d] *= inv_sum;
#endif
	}
}

/* --- Parallel dispatch --- */

struct sdpa_par_ctx {
	const float *qbase;
	const float *kbase;
	const float *vbase;
	const float *mdata;
	float       *obase;
	int          BH;             /* B * H (4D) or 1 (2D) */
	int          seq_q;
	int          seq_k;
	int          head_dim;
	float        scale;
	int          head_stride_q;  /* seq_q * head_dim */
	int          head_stride_k;  /* seq_k * head_dim */
};

static void sdpa_parallel_fn(void *arg, int task_id, int n_tasks)
{
	struct sdpa_par_ctx *ctx = (struct sdpa_par_ctx *)arg;
	int total = ctx->BH * ctx->seq_q;
	int chunk = total / n_tasks;
	int start = task_id * chunk;
	int end   = (task_id == n_tasks - 1) ? total : start + chunk;

	if (start >= end)
		return;

	for (int idx = start; idx < end; idx++) {
		int bh = idx / ctx->seq_q;
		int i  = idx % ctx->seq_q;

		const float *qd = ctx->qbase + (size_t)bh * ctx->head_stride_q;
		const float *kd = ctx->kbase + (size_t)bh * ctx->head_stride_k;
		const float *vd = ctx->vbase + (size_t)bh * ctx->head_stride_k;
		float *od       = ctx->obase + (size_t)bh * ctx->head_stride_q;

		const float *mask_row = ctx->mdata
			? ctx->mdata + (size_t)i * ctx->seq_k : NULL;

		sdpa_row_f32(qd + i * ctx->head_dim,
			     kd, vd, mask_row,
			     od + i * ctx->head_dim,
			     ctx->seq_k, ctx->head_dim, ctx->scale);
	}
}

/*
 * cpu_kernel_sdpa - Tiled SDPA kernel (2D and 4D).
 *
 * 2D path: Q[seq_q, hd], K[seq_k, hd], V[seq_k, hd]
 * 4D path: Q[B, H, seq_q, hd], K[B, H, seq_k, hd], V[B, H, seq_k, hd]
 * inputs[3]: mask [seq_q, seq_k] (optional, shared across heads)
 * params[0]: head_dim
 */
enum sam3_error cpu_kernel_sdpa(const struct sam3_node *node,
				struct sam3_threadpool *pool)
{
	const struct sam3_tensor *Q = node->inputs[0];
	const struct sam3_tensor *K = node->inputs[1];
	const struct sam3_tensor *V = node->inputs[2];
	const struct sam3_tensor *mask = (node->n_inputs > 3)
					  ? node->inputs[3] : NULL;
	struct sam3_tensor *out = node->output;

	if (!Q || !K || !V || !out) {
		sam3_log_error("sdpa: NULL tensor");
		return SAM3_EINVAL;
	}

	int head_dim = node->params[0];
	float scale = 1.0f / sqrtf((float)head_dim);

	int n_tasks = sam3_threadpool_n_threads(pool);
	if (n_tasks < 1)
		n_tasks = 1;

	if (Q->n_dims == 4) {
		int B = Q->dims[0];
		int H = Q->dims[1];
		int seq_q = Q->dims[2];
		int seq_k = K->dims[2];

		struct sdpa_par_ctx ctx = {
			.qbase         = (const float *)Q->data,
			.kbase         = (const float *)K->data,
			.vbase         = (const float *)V->data,
			.mdata         = mask ? (const float *)mask->data
					      : NULL,
			.obase         = (float *)out->data,
			.BH            = B * H,
			.seq_q         = seq_q,
			.seq_k         = seq_k,
			.head_dim      = head_dim,
			.scale         = scale,
			.head_stride_q = seq_q * head_dim,
			.head_stride_k = seq_k * head_dim,
		};

		sam3_threadpool_parallel_for(pool, sdpa_parallel_fn,
					     &ctx, n_tasks);
		return SAM3_OK;
	}

	/* 2D path: Q[seq_q, hd] — reuse same parallel dispatch */
	int seq_q = Q->dims[0];
	int seq_k = K->dims[0];

	struct sdpa_par_ctx ctx = {
		.qbase         = (const float *)Q->data,
		.kbase         = (const float *)K->data,
		.vbase         = (const float *)V->data,
		.mdata         = mask ? (const float *)mask->data : NULL,
		.obase         = (float *)out->data,
		.BH            = 1,
		.seq_q         = seq_q,
		.seq_k         = seq_k,
		.head_dim      = head_dim,
		.scale         = scale,
		.head_stride_q = seq_q * head_dim,
		.head_stride_k = seq_k * head_dim,
	};

	sam3_threadpool_parallel_for(pool, sdpa_parallel_fn,
				     &ctx, n_tasks);

	return SAM3_OK;
}
