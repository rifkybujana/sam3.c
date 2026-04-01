/*
 * src/core/tensor.c - Tensor operations
 *
 * Implements element counting, dtype sizing, and stride computation
 * for the sam3_tensor type. These are low-level utilities used by
 * the arena allocator and compute graph.
 *
 * Key types:  sam3_tensor
 * Depends on: tensor.h
 * Used by:    alloc.c, graph.c, model/ files
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "tensor.h"

int sam3_tensor_nelems(const struct sam3_tensor *t)
{
	int n = 1;

	for (int i = 0; i < t->n_dims; i++)
		n *= t->dims[i];

	return n;
}

size_t sam3_dtype_size(enum sam3_dtype dtype)
{
	switch (dtype) {
	case SAM3_DTYPE_F32: return 4;
	case SAM3_DTYPE_F16: return 2;
	case SAM3_DTYPE_BF16: return 2;
	case SAM3_DTYPE_I32: return 4;
	case SAM3_DTYPE_I8:  return 1;
	}
	return 0;
}

void sam3_tensor_compute_strides(struct sam3_tensor *t)
{
	t->strides[t->n_dims - 1] = 1;
	for (int i = t->n_dims - 2; i >= 0; i--)
		t->strides[i] = t->strides[i + 1] * t->dims[i + 1];
}

const char *sam3_dtype_str(enum sam3_dtype dtype)
{
	switch (dtype) {
	case SAM3_DTYPE_F32:  return "F32";
	case SAM3_DTYPE_F16:  return "F16";
	case SAM3_DTYPE_BF16: return "BF16";
	case SAM3_DTYPE_I32:  return "I32";
	case SAM3_DTYPE_I8:   return "I8";
	}
	return "UNKNOWN";
}
