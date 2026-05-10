/*
 * src/util/threadpool.c - Fork-join thread pool implementation
 *
 * Implements a fixed-size thread pool where workers sleep on a condition
 * variable until parallel_for dispatches work. Uses a generation counter
 * to wake workers, an atomic task counter for work-stealing, and a done
 * counter with condition variable for join. The calling thread always
 * executes task 0 to avoid unnecessary context switches.
 *
 * Key types:  sam3_threadpool
 * Depends on: util/threadpool.h, util/log.h, util/platform.h
 * Used by:    cpu_backend.c, cpu_matmul.c, cpu_conv2d.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>

#include "util/threadpool.h"
#include "util/log.h"
#include "util/platform.h"

struct sam3_threadpool {
	int              n_threads;     /* total threads including caller */
	sam3_thread     *workers;       /* n_threads - 1 worker threads */
	int              n_workers;     /* n_threads - 1 */

	sam3_mutex       mutex;
	sam3_cond        cond_work;     /* workers wait on this */
	sam3_cond        cond_done;     /* caller waits on this */

	/* Work descriptor (protected by mutex) */
	sam3_parallel_fn work_fn;
	void            *work_ctx;
	int              n_tasks;
	int              n_active;      /* min(n_tasks, n_threads) */

	/* Coordination counters (protected by mutex) */
	unsigned int     generation;    /* incremented each parallel_for */
	int              task_counter;  /* next task_id to claim */
	int              done_counter;  /* tasks completed by workers */
	int              shutdown;
};

static int detect_n_threads(void)
{
	int n = sam3_platform_cpu_count();
	if (n < 1)
		n = 1;
	return n;
}

static void *worker_main(void *arg)
{
	struct sam3_threadpool *pool = arg;
	unsigned int my_gen = 0;

	for (;;) {
		sam3_parallel_fn fn = NULL;
		void *ctx = NULL;
		int task_id = -1;
		int n_tasks = 0;

		sam3_mutex_lock(&pool->mutex);

		/* Wait until new work arrives or shutdown */
		while (pool->generation == my_gen && !pool->shutdown)
			sam3_cond_wait(&pool->cond_work, &pool->mutex);

		if (pool->shutdown) {
			sam3_mutex_unlock(&pool->mutex);
			break;
		}

		my_gen = pool->generation;

		/* Grab a task */
		task_id = pool->task_counter++;
		n_tasks = pool->n_tasks;
		fn = pool->work_fn;
		ctx = pool->work_ctx;

		sam3_mutex_unlock(&pool->mutex);

		/* Execute tasks while work remains */
		while (task_id < n_tasks) {
			fn(ctx, task_id, n_tasks);

			sam3_mutex_lock(&pool->mutex);
			task_id = pool->task_counter++;
			sam3_mutex_unlock(&pool->mutex);
		}

		/* Signal completion */
		sam3_mutex_lock(&pool->mutex);
		pool->done_counter++;
		if (pool->done_counter >= pool->n_active - 1)
			sam3_cond_signal(&pool->cond_done);
		sam3_mutex_unlock(&pool->mutex);
	}

	return NULL;
}

struct sam3_threadpool *sam3_threadpool_create(int n_threads)
{
	struct sam3_threadpool *pool = NULL;
	int i;

	if (n_threads == 0)
		n_threads = detect_n_threads();
	if (n_threads < 1)
		n_threads = 1;

	pool = calloc(1, sizeof(*pool));
	if (!pool) {
		sam3_log_error("threadpool: failed to allocate pool");
		return NULL;
	}

	pool->n_threads = n_threads;
	pool->n_workers = n_threads - 1;
	pool->generation = 0;
	pool->shutdown = 0;

	if (sam3_mutex_init(&pool->mutex) != 0) {
		sam3_log_error("threadpool: mutex init failed");
		free(pool);
		return NULL;
	}

	if (sam3_cond_init(&pool->cond_work) != 0) {
		sam3_log_error("threadpool: cond_work init failed");
		sam3_mutex_destroy(&pool->mutex);
		free(pool);
		return NULL;
	}

	if (sam3_cond_init(&pool->cond_done) != 0) {
		sam3_log_error("threadpool: cond_done init failed");
		sam3_cond_destroy(&pool->cond_work);
		sam3_mutex_destroy(&pool->mutex);
		free(pool);
		return NULL;
	}

	if (pool->n_workers > 0) {
		pool->workers = calloc(pool->n_workers, sizeof(*pool->workers));
		if (!pool->workers) {
			sam3_log_error("threadpool: failed to allocate workers");
			sam3_cond_destroy(&pool->cond_done);
			sam3_cond_destroy(&pool->cond_work);
			sam3_mutex_destroy(&pool->mutex);
			free(pool);
			return NULL;
		}

		for (i = 0; i < pool->n_workers; i++) {
			int rc = sam3_thread_create(&pool->workers[i],
						     worker_main, pool);
			if (rc != 0) {
				sam3_log_error("threadpool: thread_create "
					       "failed for worker %d", i);
				/* Shut down already-created workers */
				sam3_mutex_lock(&pool->mutex);
				pool->shutdown = 1;
				pool->n_workers = i;
				sam3_cond_broadcast(&pool->cond_work);
				sam3_mutex_unlock(&pool->mutex);

				for (int j = 0; j < i; j++)
					sam3_thread_join(&pool->workers[j]);

				free(pool->workers);
				sam3_cond_destroy(&pool->cond_done);
				sam3_cond_destroy(&pool->cond_work);
				sam3_mutex_destroy(&pool->mutex);
				free(pool);
				return NULL;
			}
		}
	}

	sam3_log_info("threadpool: created with %d threads (%d workers)",
		      n_threads, pool->n_workers);
	return pool;
}

void sam3_threadpool_free(struct sam3_threadpool *pool)
{
	int i;

	if (!pool)
		return;

	/* Signal shutdown to all workers */
	sam3_mutex_lock(&pool->mutex);
	pool->shutdown = 1;
	sam3_cond_broadcast(&pool->cond_work);
	sam3_mutex_unlock(&pool->mutex);

	/* Join all workers */
	for (i = 0; i < pool->n_workers; i++)
		sam3_thread_join(&pool->workers[i]);

	free(pool->workers);
	sam3_cond_destroy(&pool->cond_done);
	sam3_cond_destroy(&pool->cond_work);
	sam3_mutex_destroy(&pool->mutex);
	free(pool);
}

void sam3_threadpool_parallel_for(struct sam3_threadpool *pool,
				  sam3_parallel_fn fn, void *ctx,
				  int n_tasks)
{
	int n_active;

	if (n_tasks <= 0)
		return;

	/* Serial fallback: no pool or single-threaded */
	if (!pool || pool->n_threads <= 1) {
		for (int i = 0; i < n_tasks; i++)
			fn(ctx, i, n_tasks);
		return;
	}

	/* Single task: no need to wake workers */
	if (n_tasks == 1) {
		fn(ctx, 0, 1);
		return;
	}

	n_active = n_tasks < pool->n_threads ? n_tasks : pool->n_threads;

	sam3_mutex_lock(&pool->mutex);

	pool->work_fn = fn;
	pool->work_ctx = ctx;
	pool->n_tasks = n_tasks;
	pool->n_active = n_active;
	pool->task_counter = 1;   /* workers start from task 1 */
	pool->done_counter = 0;
	pool->generation++;

	sam3_cond_broadcast(&pool->cond_work);
	sam3_mutex_unlock(&pool->mutex);

	/* Caller executes task 0 */
	fn(ctx, 0, n_tasks);

	/* Wait for workers to finish */
	sam3_mutex_lock(&pool->mutex);
	while (pool->done_counter < n_active - 1)
		sam3_cond_wait(&pool->cond_done, &pool->mutex);
	sam3_mutex_unlock(&pool->mutex);
}

int sam3_threadpool_n_threads(const struct sam3_threadpool *pool)
{
	if (!pool)
		return 0;
	return pool->n_threads;
}
