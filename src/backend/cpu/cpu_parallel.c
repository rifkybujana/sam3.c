/*
 * src/backend/cpu/cpu_parallel.c - Range-based parallel_for helper
 *
 * Implementation: builds an internal context, dispatches via the
 * existing sam3_threadpool fork-join primitive, and computes balanced
 * chunk boundaries so leftover elements are distributed across the
 * first `rem` tasks rather than dumped on the last one.
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */
#include "cpu_parallel.h"
#include "util/threadpool.h"

struct sam3_range_ctx {
	size_t total;
	sam3_range_fn fn;
	void *user;
};

static void sam3_range_dispatch(void *arg, int task_id, int n_tasks)
{
	struct sam3_range_ctx *c = (struct sam3_range_ctx *)arg;
	size_t chunk = c->total / (size_t)n_tasks;
	size_t rem   = c->total % (size_t)n_tasks;
	size_t begin = (size_t)task_id * chunk +
		       ((size_t)task_id < rem ? (size_t)task_id : rem);
	size_t end   = begin + chunk + ((size_t)task_id < rem ? 1 : 0);
	if (begin < end)
		c->fn(begin, end, c->user);
}

void sam3_parallel_for_range(struct sam3_threadpool *pool,
			     size_t total, size_t grain,
			     sam3_range_fn fn, void *ctx)
{
	if (!fn || total == 0)
		return;

	int n = pool ? sam3_threadpool_n_threads(pool) : 1;
	if (n < 1)
		n = 1;

	if (!pool || n == 1 || grain == 0 || total < grain * 2) {
		fn(0, total, ctx);
		return;
	}

	size_t max_tasks = total / grain;
	if (max_tasks == 0)
		max_tasks = 1;
	if ((size_t)n > max_tasks)
		n = (int)max_tasks;
	if (n < 2) {
		fn(0, total, ctx);
		return;
	}

	struct sam3_range_ctx c = { .total = total, .fn = fn, .user = ctx };
	sam3_threadpool_parallel_for(pool, sam3_range_dispatch, &c, n);
}
