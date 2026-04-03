/*
 * src/backend/cpu/kernels/cpu_rope.c - Rotary position embedding kernel
 *
 * Applies rotary position embedding (RoPE) to pairs of elements in a
 * 4D tensor [batch, seq, n_heads, head_dim].  Each adjacent pair
 * (2*d, 2*d+1) is rotated by precomputed cos/sin frequencies indexed
 * by sequence position.  This is the standard interleaved-pair RoPE
 * used in transformer attention layers.
 *
 * Key types:  sam3_node, sam3_tensor
 * Depends on: cpu_kernels.h, core/tensor.h, util/log.h
 * Used by:    cpu_dispatch.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "cpu_kernels.h"
#include "core/tensor.h"
#include "util/log.h"

/*
 * cpu_kernel_rope - Apply rotary position embedding.
 *
 * @node:  Graph node with:
 *           inputs[0] = tensor [batch, seq, n_heads, head_dim] (F32)
 *           inputs[1] = cos frequencies [seq, head_dim/2] (F32)
 *           inputs[2] = sin frequencies [seq, head_dim/2] (F32)
 *           output    = rotated tensor, same shape as inputs[0]
 *           params[0] = head_dim (for validation)
 * @pool:  Thread pool (unused, not parallelised).
 *
 * Returns SAM3_OK on success, SAM3_EINVAL on shape mismatch.
 */
enum sam3_error cpu_kernel_rope(const struct sam3_node *node,
				struct sam3_threadpool *pool)
{
	(void)pool;  /* not parallelized for now */

	if (!node->inputs[0] || !node->inputs[1] ||
	    !node->inputs[2] || !node->output) {
		sam3_log_error("rope: NULL tensor");
		return SAM3_EINVAL;
	}

	struct sam3_tensor *inp = node->inputs[0];
	struct sam3_tensor *cos_f = node->inputs[1];
	struct sam3_tensor *sin_f = node->inputs[2];
	struct sam3_tensor *out = node->output;

	if (inp->n_dims != 4) {
		sam3_log_error("rope: input must be 4D, got %dD",
			       inp->n_dims);
		return SAM3_EINVAL;
	}

	/* inp: [batch, seq, n_heads, head_dim] */
	int batch = inp->dims[0];
	int seq = inp->dims[1];
	int n_heads = inp->dims[2];
	int head_dim = inp->dims[3];
	int half_dim = head_dim / 2;

	if (head_dim % 2 != 0) {
		sam3_log_error("rope: head_dim must be even, got %d",
			       head_dim);
		return SAM3_EINVAL;
	}

	/* Validate cos/sin shapes: [seq, half_dim] */
	if (cos_f->n_dims != 2 || cos_f->dims[0] != seq ||
	    cos_f->dims[1] != half_dim) {
		sam3_log_error("rope: cos shape mismatch, "
			       "expected [%d,%d]", seq, half_dim);
		return SAM3_EINVAL;
	}

	if (sin_f->n_dims != 2 || sin_f->dims[0] != seq ||
	    sin_f->dims[1] != half_dim) {
		sam3_log_error("rope: sin shape mismatch, "
			       "expected [%d,%d]", seq, half_dim);
		return SAM3_EINVAL;
	}

	const float *x = (const float *)inp->data;
	const float *cos_tab = (const float *)cos_f->data;
	const float *sin_tab = (const float *)sin_f->data;
	float *y = (float *)out->data;

	for (int b = 0; b < batch; b++) {
		for (int s = 0; s < seq; s++) {
			const float *cs = cos_tab + s * half_dim;
			const float *sn = sin_tab + s * half_dim;
			for (int h = 0; h < n_heads; h++) {
				int off = ((b * seq + s) * n_heads + h)
					  * head_dim;
				for (int d = 0; d < half_dim; d++) {
					float x0 = x[off + 2 * d];
					float x1 = x[off + 2 * d + 1];
					y[off + 2 * d] =
						x0 * cs[d] - x1 * sn[d];
					y[off + 2 * d + 1] =
						x0 * sn[d] + x1 * cs[d];
				}
			}
		}
	}

	return SAM3_OK;
}
