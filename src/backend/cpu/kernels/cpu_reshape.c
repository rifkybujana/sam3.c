/*
 * src/backend/cpu/kernels/cpu_reshape.c - Zero-copy reshape kernel
 *
 * Reshape sets output->data = input->data and recomputes strides.
 * No data is copied — the output tensor aliases the input. This
 * requires that total element counts match.
 *
 * Key types:  sam3_node, sam3_tensor
 * Depends on: cpu_kernels.h, core/tensor.h
 * Used by:    cpu_backend.c (dispatch)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "cpu_kernels.h"
#include "core/tensor.h"
#include "util/log.h"

enum sam3_error cpu_kernel_reshape(const struct sam3_node *node)
{
	struct sam3_tensor *in = node->inputs[0];
	struct sam3_tensor *out = node->output;

	if (!in || !out) {
		sam3_log_error("reshape: NULL tensor");
		return SAM3_EINVAL;
	}

	if (sam3_tensor_nelems(in) != sam3_tensor_nelems(out)) {
		sam3_log_error("reshape: element count mismatch %d != %d",
			       sam3_tensor_nelems(in),
			       sam3_tensor_nelems(out));
		return SAM3_EINVAL;
	}

	out->data = in->data;
	out->nbytes = in->nbytes;
	sam3_tensor_compute_strides(out);

	return SAM3_OK;
}
