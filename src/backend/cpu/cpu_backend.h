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
 * Used by:    backend.h (via sam3_backend_init), tests, cpu_matmul.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_BACKEND_CPU_H
#define SAM3_BACKEND_CPU_H

#include "backend/backend.h"
#include "core/alloc.h"

/* Default arena capacity: 256 MiB — enough for SAM3 inference. */
#define SAM3_CPU_ARENA_DEFAULT_CAPACITY (256UL * 1024 * 1024)

/* Default scratch arena: 128 MiB for conv2d im2col temp buffers.
 * SAM3 neck conv2d on 72x72 patches needs ~85 MiB for im2col.
 * The FPN pixel decoder needs ~893 MiB for 3x3 conv on 288x288x256
 * (81 MiB NHWC→NCHW in + 81 MiB out + 729 MiB im2col); set
 * scratch_capacity >= 1 GiB for full inference on the CPU path. */
#define SAM3_CPU_SCRATCH_DEFAULT_CAPACITY (128UL * 1024 * 1024)

struct sam3_profiler;   /* Forward declaration */
struct sam3_threadpool; /* Forward declaration */

struct sam3_cpu_backend {
	struct sam3_backend     base;           /* Must be first member */
	struct sam3_arena       arena;          /* Tensor data arena */
	struct sam3_arena       scratch;        /* Scratch arena for temp buffers */
	size_t                  arena_capacity;   /* 0 = use default */
	size_t                  scratch_capacity; /* 0 = use default */
	struct sam3_threadpool *pool;       /* Thread pool for kernel parallelism */
#ifdef SAM3_HAS_PROFILE
	struct sam3_profiler   *profiler;     /* NULL when profiling disabled */
#endif
};

/* Get the CPU backend ops vtable. */
const struct sam3_backend_ops *sam3_cpu_backend_ops(void);

#endif /* SAM3_BACKEND_CPU_H */
