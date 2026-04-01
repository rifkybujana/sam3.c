/*
 * src/util/threadpool.h - Fixed-size thread pool with fork-join parallel_for
 *
 * Provides a barrier-based fork-join thread pool for intra-kernel parallelism.
 * Workers sleep until parallel_for is called, then each executes a slice of
 * the work. The calling thread participates as task 0. Designed for CPU
 * backend matmul/conv2d row-splitting.
 *
 * Key types:  sam3_threadpool, sam3_parallel_fn
 * Depends on: (none)
 * Used by:    cpu_backend.c, cpu_matmul.c, cpu_conv2d.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_UTIL_THREADPOOL_H
#define SAM3_UTIL_THREADPOOL_H

struct sam3_threadpool;

/* Work function signature: called once per task_id in [0, n_tasks). */
typedef void (*sam3_parallel_fn)(void *ctx, int task_id, int n_tasks);

/*
 * sam3_threadpool_create - Create a thread pool.
 *
 * @n_threads: Total thread count including caller. 0 = auto-detect CPU cores.
 *             Minimum effective value is 1 (single-threaded, no workers spawned).
 *
 * Returns NULL on failure (allocation or pthread error).
 */
struct sam3_threadpool *sam3_threadpool_create(int n_threads);

/*
 * sam3_threadpool_free - Destroy pool and join all workers.
 *
 * Signals shutdown, wakes workers, joins them, frees memory.
 * Safe to call with NULL.
 */
void sam3_threadpool_free(struct sam3_threadpool *pool);

/*
 * sam3_threadpool_parallel_for - Execute fn across n_tasks in parallel.
 *
 * @pool:    Thread pool (if NULL, executes serially on caller)
 * @fn:      Work function called as fn(ctx, task_id, n_tasks)
 * @ctx:     Opaque context passed to fn
 * @n_tasks: Number of tasks to distribute.
 *
 * Blocks until all tasks complete. Caller thread executes task 0.
 * If pool is NULL or n_threads==1, runs all tasks serially.
 */
void sam3_threadpool_parallel_for(struct sam3_threadpool *pool,
				  sam3_parallel_fn fn, void *ctx,
				  int n_tasks);

/* Return the number of threads in the pool (including caller thread). */
int sam3_threadpool_n_threads(const struct sam3_threadpool *pool);

#endif /* SAM3_UTIL_THREADPOOL_H */
