/*
 * tests/test_threadpool.c - Unit tests for the thread pool
 *
 * Tests creation, destruction, serial fallback, parallel execution, and
 * reuse of the fork-join thread pool. Verifies correctness under various
 * thread/task count combinations.
 *
 * Key types:  sam3_threadpool
 * Depends on: util/threadpool.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <string.h>

#include "test_helpers.h"
#include "util/threadpool.h"

/* --- Test 1: create and free a 4-thread pool --- */
static void test_threadpool_create_free(void)
{
	struct sam3_threadpool *pool = sam3_threadpool_create(4);
	ASSERT(pool != NULL);
	ASSERT_EQ(sam3_threadpool_n_threads(pool), 4);
	sam3_threadpool_free(pool);
}

/* --- Test 2: auto-detect thread count --- */
static void test_threadpool_create_auto(void)
{
	struct sam3_threadpool *pool = sam3_threadpool_create(0);
	ASSERT(pool != NULL);
	ASSERT(sam3_threadpool_n_threads(pool) >= 1);
	sam3_threadpool_free(pool);
}

/* --- Test 3: single-threaded pool --- */
static void test_threadpool_create_single(void)
{
	struct sam3_threadpool *pool = sam3_threadpool_create(1);
	ASSERT(pool != NULL);
	ASSERT_EQ(sam3_threadpool_n_threads(pool), 1);
	sam3_threadpool_free(pool);
}

/* --- Test 4: free(NULL) does not crash --- */
static void test_threadpool_free_null(void)
{
	sam3_threadpool_free(NULL);
	ASSERT(1); /* If we got here, it didn't crash */
}

/* --- Helper for serial/parallel tests --- */
struct task_result {
	int values[64];
};

static void write_task_id(void *ctx, int task_id, int n_tasks)
{
	struct task_result *r = ctx;
	r->values[task_id] = task_id * 10 + n_tasks;
}

/* --- Test 5: NULL pool runs serially --- */
static void test_threadpool_null_serial(void)
{
	struct task_result r;
	memset(&r, 0, sizeof(r));

	sam3_threadpool_parallel_for(NULL, write_task_id, &r, 4);

	for (int i = 0; i < 4; i++)
		ASSERT_EQ(r.values[i], i * 10 + 4);
}

/* --- Test 6: 4-thread pool, 4 tasks --- */
static void test_threadpool_parallel_for(void)
{
	struct sam3_threadpool *pool = sam3_threadpool_create(4);
	struct task_result r;
	memset(&r, 0, sizeof(r));

	sam3_threadpool_parallel_for(pool, write_task_id, &r, 4);

	for (int i = 0; i < 4; i++)
		ASSERT_EQ(r.values[i], i * 10 + 4);

	sam3_threadpool_free(pool);
}

/* --- Test 7: more tasks than threads --- */
static void test_threadpool_more_tasks_than_threads(void)
{
	struct sam3_threadpool *pool = sam3_threadpool_create(2);
	struct task_result r;
	memset(&r, 0, sizeof(r));

	sam3_threadpool_parallel_for(pool, write_task_id, &r, 8);

	for (int i = 0; i < 8; i++)
		ASSERT_EQ(r.values[i], i * 10 + 8);

	sam3_threadpool_free(pool);
}

/* --- Test 8: parallel sum of 1M floats --- */
#define SUM_N 1000000

struct sum_ctx {
	const float *data;
	float        partial_sums[16];
	int          n;
};

static void sum_chunk(void *arg, int task_id, int n_tasks)
{
	struct sum_ctx *s = arg;
	int chunk = s->n / n_tasks;
	int start = task_id * chunk;
	int end = (task_id == n_tasks - 1) ? s->n : start + chunk;
	float acc = 0.0f;

	for (int i = start; i < end; i++)
		acc += s->data[i];

	s->partial_sums[task_id] = acc;
}

static void test_threadpool_parallel_sum(void)
{
	struct sam3_threadpool *pool = sam3_threadpool_create(4);
	float *data = malloc(SUM_N * sizeof(float));
	ASSERT(data != NULL);

	for (int i = 0; i < SUM_N; i++)
		data[i] = 1.0f;

	struct sum_ctx s;
	memset(&s, 0, sizeof(s));
	s.data = data;
	s.n = SUM_N;

	sam3_threadpool_parallel_for(pool, sum_chunk, &s, 4);

	float total = 0.0f;
	for (int i = 0; i < 4; i++)
		total += s.partial_sums[i];

	ASSERT_NEAR(total, (float)SUM_N, 1.0f);

	free(data);
	sam3_threadpool_free(pool);
}

/* --- Test 9: reuse pool across multiple parallel_for calls --- */
static void test_threadpool_reuse(void)
{
	struct sam3_threadpool *pool = sam3_threadpool_create(4);
	struct task_result r;

	for (int round = 0; round < 10; round++) {
		memset(&r, 0, sizeof(r));
		sam3_threadpool_parallel_for(pool, write_task_id, &r, 4);

		for (int i = 0; i < 4; i++)
			ASSERT_EQ(r.values[i], i * 10 + 4);
	}

	sam3_threadpool_free(pool);
}

int main(void)
{
	test_threadpool_create_free();
	test_threadpool_create_auto();
	test_threadpool_create_single();
	test_threadpool_free_null();
	test_threadpool_null_serial();
	test_threadpool_parallel_for();
	test_threadpool_more_tasks_than_threads();
	test_threadpool_parallel_sum();
	test_threadpool_reuse();

	TEST_REPORT();
}
