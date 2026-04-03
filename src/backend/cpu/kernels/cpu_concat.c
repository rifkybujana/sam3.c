/*
 * src/backend/cpu/kernels/cpu_concat.c - Concatenation kernel
 *
 * Concatenates N input tensors along a specified axis.  All inputs must
 * share the same shape on every dimension except the concat axis.  Uses
 * a block-copy approach: for axis 0 each input's entire data is copied
 * sequentially; for other axes the outer dimensions are iterated and
 * inner blocks are copied with memcpy.
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

#include <string.h>

enum sam3_error cpu_kernel_concat(const struct sam3_node *node,
				  struct sam3_threadpool *pool)
{
	(void)pool;

	if (node->n_inputs < 1 || !node->output) {
		sam3_log_error("concat: need >= 1 input and an output");
		return SAM3_EINVAL;
	}

	for (int i = 0; i < node->n_inputs; i++) {
		if (!node->inputs[i]) {
			sam3_log_error("concat: NULL input[%d]", i);
			return SAM3_EINVAL;
		}
	}

	const struct sam3_tensor *first = node->inputs[0];
	struct sam3_tensor *out = node->output;
	int axis = node->params[0];
	int n_dims = first->n_dims;

	if (axis < 0 || axis >= n_dims) {
		sam3_log_error("concat: axis %d out of range [0, %d)",
			       axis, n_dims);
		return SAM3_EINVAL;
	}

	/* Validate non-concat dims match across all inputs */
	for (int i = 1; i < node->n_inputs; i++) {
		const struct sam3_tensor *inp = node->inputs[i];
		if (inp->n_dims != n_dims) {
			sam3_log_error("concat: input[%d] n_dims=%d != %d",
				       i, inp->n_dims, n_dims);
			return SAM3_EINVAL;
		}
		for (int d = 0; d < n_dims; d++) {
			if (d == axis)
				continue;
			if (inp->dims[d] != first->dims[d]) {
				sam3_log_error("concat: input[%d] dim[%d]=%d "
					       "!= %d", i, d,
					       inp->dims[d], first->dims[d]);
				return SAM3_EINVAL;
			}
		}
	}

	/*
	 * Compute outer_iters = product of dims before axis.
	 * inner_size = product of dims after axis (not including axis).
	 */
	int outer_iters = 1;
	for (int d = 0; d < axis; d++)
		outer_iters *= first->dims[d];

	int inner_size = 1;
	for (int d = axis + 1; d < n_dims; d++)
		inner_size *= first->dims[d];

	size_t esz = sam3_dtype_size(first->dtype);
	char *dst = (char *)out->data;

	for (int o = 0; o < outer_iters; o++) {
		for (int i = 0; i < node->n_inputs; i++) {
			const struct sam3_tensor *inp = node->inputs[i];
			const char *src = (const char *)inp->data;
			int axis_len = inp->dims[axis];
			int block = axis_len * inner_size;

			/*
			 * Source offset: each outer iteration advances by
			 * (axis_len * inner_size) elements in that input.
			 */
			int src_off = o * block;
			memcpy(dst, src + (size_t)src_off * esz,
			       (size_t)block * esz);
			dst += (size_t)block * esz;
		}
	}

	return SAM3_OK;
}
