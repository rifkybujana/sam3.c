/*
 * src/backend/cpu/cpu_parallel.h - Range-based parallel_for helper
 *
 * Thin wrapper over sam3_threadpool_parallel_for that splits an iteration
 * range [0, total) into balanced chunks and invokes fn(begin, end, ctx)
 * on each worker. Falls back to a serial in-line call when the workload
 * is below grain*2, the pool is NULL, or the pool has only one thread.
 *
 * Key types:  sam3_range_fn
 * Depends on: util/threadpool.h
 * Used by:    cpu kernels needing simple data-parallel loops
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */
#ifndef SAM3_CPU_PARALLEL_H
#define SAM3_CPU_PARALLEL_H

#include <stddef.h>

struct sam3_threadpool;

typedef void (*sam3_range_fn)(size_t begin, size_t end, void *ctx);

/*
 * sam3_parallel_for_range - Run fn over [0, total) in parallel chunks.
 *
 * @pool:  Thread pool (NULL = serial).
 * @total: Total iteration count.
 * @grain: Minimum chunk size; if total < grain*2 the call runs serially.
 * @fn:    Worker function fn(begin, end, ctx).
 * @ctx:   Opaque context forwarded to fn.
 *
 * Iteration space is partitioned so each worker gets ceil(total/n) or
 * floor(total/n) elements; fn is invoked exactly once per worker that
 * receives a non-empty range. Total coverage is exact: every index in
 * [0, total) is visited exactly once.
 */
void sam3_parallel_for_range(struct sam3_threadpool *pool,
			     size_t total, size_t grain,
			     sam3_range_fn fn, void *ctx);

#endif /* SAM3_CPU_PARALLEL_H */
