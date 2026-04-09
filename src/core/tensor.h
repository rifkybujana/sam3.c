/*
 * src/core/tensor.h - Multi-dimensional tensor type
 *
 * Defines the core tensor struct used throughout sam3. Tensors are
 * dense, contiguous, row-major arrays with up to SAM3_MAX_DIMS
 * dimensions. They do not own their memory — allocation is handled
 * by the arena allocator in alloc.h.
 *
 * Key types:  sam3_tensor
 * Depends on: sam3/sam3_types.h
 * Used by:    graph.h, all model/ files, all backend/ files
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_CORE_TENSOR_H
#define SAM3_CORE_TENSOR_H

#include "sam3/sam3_types.h"

struct sam3_tensor {
	enum sam3_dtype dtype;
	int             n_dims;
	int             dims[SAM3_MAX_DIMS];
	int             strides[SAM3_MAX_DIMS];
	void           *data;
	size_t          nbytes;
	/*
	 * Set to 1 for tensor headers that wrap external mutable data
	 * (created by gh_tensor_wrap). Backends must not cache these
	 * tensors across graph_eval calls: the header address may be
	 * reused by a subsequent arena allocation while the wrapped
	 * data contents have changed.
	 */
	int             ephemeral;
};

/* Return the total number of elements in the tensor. */
int sam3_tensor_nelems(const struct sam3_tensor *t);

/* Return the size in bytes of one element of the given dtype. */
size_t sam3_dtype_size(enum sam3_dtype dtype);

/* Compute strides from dims (row-major). Fills t->strides. */
void sam3_tensor_compute_strides(struct sam3_tensor *t);

/* Return a short string name for the dtype ("F32", "F16", etc). */
const char *sam3_dtype_str(enum sam3_dtype dtype);

#endif /* SAM3_CORE_TENSOR_H */
