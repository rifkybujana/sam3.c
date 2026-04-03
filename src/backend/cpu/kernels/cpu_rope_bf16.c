/*
 * src/backend/cpu/kernels/cpu_rope_bf16.c - BF16 rotary position embedding kernel
 *
 * Applies rotary position embedding to bf16 input tensors with f32
 * cos/sin frequency tables (mixed-dtype).  Input is [batch, seq,
 * n_heads, head_dim] in BF16; cos/sin are [seq, head_dim/2] in F32.
 * Converts BF16 pairs to f32 for rotation, then back to BF16.
 *
 * Key types:  sam3_node, sam3_tensor
 * Depends on: cpu_kernels.h, core/half.h, core/tensor.h, util/log.h
 * Used by:    cpu_dispatch.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "cpu_kernels.h"
#include "core/half.h"
#include "core/tensor.h"
#include "util/log.h"

/*
 * cpu_kernel_rope_bf16 - Apply rotary position embedding to BF16 input.
 *
 * @node:  Graph node with:
 *           inputs[0] = tensor [batch, seq, n_heads, head_dim] (BF16)
 *           inputs[1] = cos frequencies [seq, head_dim/2] (F32)
 *           inputs[2] = sin frequencies [seq, head_dim/2] (F32)
 *           output    = rotated tensor, same shape as inputs[0] (BF16)
 *           params[0] = head_dim (for validation)
 * @pool:  Thread pool (unused, not parallelised).
 *
 * Returns SAM3_OK on success, SAM3_EINVAL on shape/dtype mismatch.
 */
enum sam3_error cpu_kernel_rope_bf16(const struct sam3_node *node,
				     struct sam3_threadpool *pool)
{
	(void)pool;  /* not parallelized for now */

	if (!node->inputs[0] || !node->inputs[1] ||
	    !node->inputs[2] || !node->output) {
		sam3_log_error("rope_bf16: NULL tensor");
		return SAM3_EINVAL;
	}

	struct sam3_tensor *inp = node->inputs[0];
	struct sam3_tensor *cos_f = node->inputs[1];
	struct sam3_tensor *sin_f = node->inputs[2];
	struct sam3_tensor *out = node->output;

	if (inp->dtype != SAM3_DTYPE_BF16) {
		sam3_log_error("rope_bf16: input must be BF16");
		return SAM3_EINVAL;
	}

	if (cos_f->dtype != SAM3_DTYPE_F32 ||
	    sin_f->dtype != SAM3_DTYPE_F32) {
		sam3_log_error("rope_bf16: cos/sin must be F32");
		return SAM3_EINVAL;
	}

	if (inp->n_dims != 4) {
		sam3_log_error("rope_bf16: input must be 4D, got %dD",
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
		sam3_log_error("rope_bf16: head_dim must be even, got %d",
			       head_dim);
		return SAM3_EINVAL;
	}

	/* Validate cos/sin shapes: [seq, half_dim] */
	if (cos_f->n_dims != 2 || cos_f->dims[0] != seq ||
	    cos_f->dims[1] != half_dim) {
		sam3_log_error("rope_bf16: cos shape mismatch, "
			       "expected [%d,%d]", seq, half_dim);
		return SAM3_EINVAL;
	}

	if (sin_f->n_dims != 2 || sin_f->dims[0] != seq ||
	    sin_f->dims[1] != half_dim) {
		sam3_log_error("rope_bf16: sin shape mismatch, "
			       "expected [%d,%d]", seq, half_dim);
		return SAM3_EINVAL;
	}

	const uint16_t *x = (const uint16_t *)inp->data;
	const float *cos_tab = (const float *)cos_f->data;
	const float *sin_tab = (const float *)sin_f->data;
	uint16_t *y = (uint16_t *)out->data;

	for (int b = 0; b < batch; b++) {
		for (int s = 0; s < seq; s++) {
			const float *cs = cos_tab + s * half_dim;
			const float *sn = sin_tab + s * half_dim;
			for (int h = 0; h < n_heads; h++) {
				int off = ((b * seq + s) * n_heads + h)
					  * head_dim;
				for (int d = 0; d < half_dim; d++) {
					float x0 = bf16_to_f32(
						x[off + 2 * d]);
					float x1 = bf16_to_f32(
						x[off + 2 * d + 1]);
					y[off + 2 * d] = f32_to_bf16(
						x0 * cs[d] - x1 * sn[d]);
					y[off + 2 * d + 1] = f32_to_bf16(
						x0 * sn[d] + x1 * cs[d]);
				}
			}
		}
	}

	return SAM3_OK;
}
