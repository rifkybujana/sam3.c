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
			float score = 0.0f;
			const float *kj = K + j * head_dim;

			for (d = 0; d < head_dim; d++)
				score += q[d] * kj[d];
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
				for (d = 0; d < head_dim; d++)
					out[d] *= correction;
				row_max = score;
			}

			float w = expf(score - row_max);
			const float *vj = V + j * head_dim;
			for (d = 0; d < head_dim; d++)
				out[d] += w * vj[d];
			row_sum += w;
		}
	}

	/* Normalize by sum */
	if (row_sum > 0.0f) {
		float inv_sum = 1.0f / row_sum;
		for (d = 0; d < head_dim; d++)
			out[d] *= inv_sum;
	}
}

/*
 * cpu_kernel_sdpa - Tiled SDPA kernel.
 *
 * inputs[0]: Q [seq_q, head_dim]
 * inputs[1]: K [seq_k, head_dim]
 * inputs[2]: V [seq_k, head_dim]
 * inputs[3]: mask [seq_q, seq_k] (optional)
 * params[0]: head_dim
 * output:    [seq_q, head_dim]
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
	int seq_q = Q->dims[0];
	int seq_k = K->dims[0];
	float scale = 1.0f / sqrtf((float)head_dim);

	const float *qdata = (const float *)Q->data;
	const float *kdata = (const float *)K->data;
	const float *vdata = (const float *)V->data;
	const float *mdata = mask ? (const float *)mask->data : NULL;
	float *odata = (float *)out->data;

	(void)pool;

	for (int i = 0; i < seq_q; i++) {
		const float *mask_row = mdata
					 ? mdata + i * seq_k : NULL;
		sdpa_row_f32(qdata + i * head_dim,
			     kdata, vdata, mask_row,
			     odata + i * head_dim,
			     seq_k, head_dim, scale);
	}

	return SAM3_OK;
}
