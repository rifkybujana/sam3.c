/*
 * src/backend/cpu/cpu_backend.h - CPU compute backend
 *
 * CPU fallback backend using scalar and SIMD operations. Implements
 * all operations in the sam3_backend_ops vtable. Tensor data is
 * allocated from an internal arena allocator for zero-fragmentation
 * inference. Arena capacity is configurable via arena_capacity.
 *
 * Key types:  sam3_cpu_backend
 * Depends on: backend/backend.h, core/alloc.h
 * Used by:    backend.h (via sam3_backend_init), tests
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_BACKEND_CPU_H
#define SAM3_BACKEND_CPU_H

#include "backend/backend.h"
#include "core/alloc.h"

/* Default arena capacity: 256 MiB — enough for SAM3 inference. */
#define SAM3_CPU_ARENA_DEFAULT_CAPACITY (256UL * 1024 * 1024)

/* Default scratch arena: 64 MiB for conv2d im2col temp buffers. */
#define SAM3_CPU_SCRATCH_DEFAULT_CAPACITY (64UL * 1024 * 1024)

struct sam3_cpu_backend {
	struct sam3_backend base;           /* Must be first member */
	struct sam3_arena   arena;          /* Tensor data arena */
	struct sam3_arena   scratch;        /* Scratch arena for temp buffers */
	size_t              arena_capacity; /* 0 = use default */
};

/* Get the CPU backend ops vtable. */
const struct sam3_backend_ops *sam3_cpu_backend_ops(void);

#endif /* SAM3_BACKEND_CPU_H */
