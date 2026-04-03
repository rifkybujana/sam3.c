/*
 * src/backend/cpu/kernels/cpu_slice.c - Slice extraction kernel
 *
 * Extracts a contiguous sub-tensor along a specified axis.  params[0]
 * is the axis, params[1] is the start index (inclusive), params[2] is
 * the end index (exclusive).  Uses the same outer/inner block-copy
 * approach as concat but copies from a single source into the output.
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

enum sam3_error cpu_kernel_slice(const struct sam3_node *node,
				 struct sam3_threadpool *pool)
{
	(void)pool;

	if (!node->inputs[0] || !node->output) {
		sam3_log_error("slice: NULL tensor");
		return SAM3_EINVAL;
	}

	const struct sam3_tensor *inp = node->inputs[0];
	struct sam3_tensor *out = node->output;
	int axis = node->params[0];
	int start = node->params[1];
	int end = node->params[2];
	int n_dims = inp->n_dims;

	if (axis < 0 || axis >= n_dims) {
		sam3_log_error("slice: axis %d out of range [0, %d)",
			       axis, n_dims);
		return SAM3_EINVAL;
	}

	if (start < 0 || end > inp->dims[axis] || start >= end) {
		sam3_log_error("slice: invalid range [%d, %d) for dim %d",
			       start, end, inp->dims[axis]);
		return SAM3_EINVAL;
	}

	int slice_len = end - start;

	/* Validate output shape */
	if (out->dims[axis] != slice_len) {
		sam3_log_error("slice: output axis dim %d != %d",
			       out->dims[axis], slice_len);
		return SAM3_EINVAL;
	}

	/*
	 * outer_iters = product of dims before axis.
	 * inner_size = product of dims after axis (not including axis).
	 */
	int outer_iters = 1;
	for (int d = 0; d < axis; d++)
		outer_iters *= inp->dims[d];

	int inner_size = 1;
	for (int d = axis + 1; d < n_dims; d++)
		inner_size *= inp->dims[d];

	int src_axis_stride = inp->dims[axis] * inner_size;
	int copy_block = slice_len * inner_size;

	const float *src = (const float *)inp->data;
	float *dst = (float *)out->data;

	for (int o = 0; o < outer_iters; o++) {
		int src_off = o * src_axis_stride + start * inner_size;
		memcpy(dst, src + src_off,
		       (size_t)copy_block * sizeof(float));
		dst += copy_block;
	}

	return SAM3_OK;
}
