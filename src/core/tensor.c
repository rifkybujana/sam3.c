/*
 * src/core/tensor.c - Tensor operations
 *
 * Implements element counting, dtype sizing, stride computation, and
 * deep-copy helpers for the sam3_tensor type. These are low-level
 * utilities used by the arena allocator, compute graph, and modules
 * that must retain tensor snapshots beyond a single graph evaluation.
 *
 * Key types:  sam3_tensor
 * Depends on: tensor.h, alloc.h
 * Used by:    alloc.c, graph.c, model/ files
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <string.h>

#include "tensor.h"
#include "alloc.h"

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
	case SAM3_DTYPE_F32:  return 4;
	case SAM3_DTYPE_F16:  return 2;
	case SAM3_DTYPE_BF16: return 2;
	case SAM3_DTYPE_I32:  return 4;
	case SAM3_DTYPE_I8:   return 1;
	case SAM3_DTYPE_Q8_0: return 0; /* block type */
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
	case SAM3_DTYPE_Q8_0: return "Q8_0";
	}
	return "UNKNOWN";
}

struct sam3_tensor *sam3_tensor_clone_persist(struct sam3_arena *arena,
					      const struct sam3_tensor *src)
{
	struct sam3_tensor *dst;

	if (!arena || !src)
		return NULL;

	dst = (struct sam3_tensor *)
		sam3_arena_alloc(arena, sizeof(struct sam3_tensor));
	if (!dst)
		return NULL;

	dst->dtype    = src->dtype;
	dst->n_dims   = src->n_dims;
	for (int i = 0; i < SAM3_MAX_DIMS; i++) {
		dst->dims[i]    = src->dims[i];
		dst->strides[i] = src->strides[i];
	}
	dst->nbytes    = src->nbytes;
	dst->ephemeral = 0;

	if (src->nbytes == 0 || !src->data) {
		dst->data = NULL;
		return dst;
	}

	dst->data = sam3_arena_alloc_raw(arena, src->nbytes);
	if (!dst->data)
		return NULL;

	memcpy(dst->data, src->data, src->nbytes);
	return dst;
}
