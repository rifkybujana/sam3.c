/*
 * tests/test_bench.c - Unit tests for benchmark harness
 *
 * Tests config defaults, timing loop with statistics, environment
 * detection, and filter matching. Uses a trivial callback that
 * burns CPU cycles to produce measurable timings.
 *
 * Key types:  sam3_bench_config, sam3_bench_result, sam3_bench_env
 * Depends on: bench/bench.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "bench/bench.h"

#include <string.h>

/* Counter incremented by the benchmark callback. */
static int g_call_count;

/* Trivial benchmark function that burns some cycles. */
static void bench_fn(void *ctx)
{
	(void)ctx;
	volatile int x = 0;
	for (int i = 0; i < 1000; i++)
		x += i;
	(void)x;
	g_call_count++;
}

/* --- test_config_defaults --- */

static void test_config_defaults(void)
{
	struct sam3_bench_config cfg;
	memset(&cfg, 0xFF, sizeof(cfg)); /* poison */
	sam3_bench_config_defaults(&cfg);

	ASSERT_EQ(cfg.warmup_iters, 5);
	ASSERT_EQ(cfg.timed_iters, 50);
	ASSERT_NEAR(cfg.threshold_pct, 5.0, 0.001);
	ASSERT_EQ(cfg.statistical, false);
	ASSERT(cfg.output_path == NULL);
	ASSERT(cfg.baseline_path == NULL);
	ASSERT_EQ(cfg.backend, SAM3_BACKEND_CPU);
	ASSERT(cfg.filter == NULL);
	ASSERT_EQ(cfg.verbose, false);
}

/* --- test_bench_run_basic --- */

static void test_bench_run_basic(void)
{
	struct sam3_bench_config cfg;
	sam3_bench_config_defaults(&cfg);
	cfg.warmup_iters = 2;
	cfg.timed_iters  = 10;

	struct sam3_bench_result res;
	g_call_count = 0;

	int err = sam3_bench_run(&cfg, "test_basic", "kernel",
				 bench_fn, NULL, 0.0, 0.0, &res);

	ASSERT_EQ(err, 0);

	/* 2 warmup + 10 timed = 12 total calls. */
	ASSERT_EQ(g_call_count, 12);

	/* Timing sanity checks. */
	ASSERT(res.mean_ms >= 0.0);
	ASSERT(res.min_ms >= 0.0);
	ASSERT(res.max_ms >= res.min_ms);
	ASSERT(res.mean_ms >= res.min_ms);
	ASSERT(res.mean_ms <= res.max_ms);
	ASSERT_EQ(res.iterations, 10);

	/* Name and suite preserved. */
	ASSERT_EQ(strcmp(res.name, "test_basic"), 0);
	ASSERT_EQ(strcmp(res.suite, "kernel"), 0);

	/* No flops/bytes passed, so derived metrics should be 0. */
	ASSERT_NEAR(res.gflops, 0.0, 0.001);
	ASSERT_NEAR(res.throughput_mbs, 0.0, 0.001);

	/* Stddev should be 0 when statistical mode is off. */
	ASSERT_NEAR(res.stddev_ms, 0.0, 0.001);
}

/* --- test_bench_run_statistical --- */

static void test_bench_run_statistical(void)
{
	struct sam3_bench_config cfg;
	sam3_bench_config_defaults(&cfg);
	cfg.warmup_iters = 1;
	cfg.timed_iters  = 20;
	cfg.statistical  = true;

	struct sam3_bench_result res;
	g_call_count = 0;

	int err = sam3_bench_run(&cfg, "test_stat", "kernel",
				 bench_fn, NULL, 0.0, 0.0, &res);

	ASSERT_EQ(err, 0);
	ASSERT(res.stddev_ms >= 0.0);
	ASSERT(res.mean_ms > 0.0);
}

/* --- test_bench_run_with_flops --- */

static void test_bench_run_with_flops(void)
{
	struct sam3_bench_config cfg;
	sam3_bench_config_defaults(&cfg);
	cfg.warmup_iters = 1;
	cfg.timed_iters  = 10;

	struct sam3_bench_result res;

	/* 1e9 flops per call. */
	int err = sam3_bench_run(&cfg, "test_flops", "kernel",
				 bench_fn, NULL, 1e9, 0.0, &res);

	ASSERT_EQ(err, 0);
	ASSERT(res.gflops > 0.0);
	ASSERT_NEAR(res.throughput_mbs, 0.0, 0.001);
}

/* --- test_bench_run_with_bytes --- */

static void test_bench_run_with_bytes(void)
{
	struct sam3_bench_config cfg;
	sam3_bench_config_defaults(&cfg);
	cfg.warmup_iters = 1;
	cfg.timed_iters  = 10;

	struct sam3_bench_result res;

	/* 1e6 bytes per call. */
	int err = sam3_bench_run(&cfg, "test_bytes", "pipeline",
				 bench_fn, NULL, 0.0, 1e6, &res);

	ASSERT_EQ(err, 0);
	ASSERT(res.throughput_mbs > 0.0);
	ASSERT_NEAR(res.gflops, 0.0, 0.001);
}

/* --- test_env_detect --- */

static void test_env_detect(void)
{
	struct sam3_bench_env env;
	sam3_bench_env_detect(&env, SAM3_BACKEND_CPU);

	/* OS string should be non-empty. */
	ASSERT(strlen(env.os) > 0);

	/* CPU cores should be positive. */
#ifdef __APPLE__
	ASSERT(env.cpu_cores > 0);
#endif

	/* Backend string should be "cpu". */
	ASSERT_EQ(strcmp(env.backend, "cpu"), 0);

	/* Timestamp should be non-empty. */
	ASSERT(strlen(env.timestamp) > 0);

	/* Also test metal backend string. */
	struct sam3_bench_env env2;
	sam3_bench_env_detect(&env2, SAM3_BACKEND_METAL);
	ASSERT_EQ(strcmp(env2.backend, "metal"), 0);
}

/* --- test_filter_match --- */

static void test_filter_match(void)
{
	/* NULL filter matches everything. */
	ASSERT(sam3_bench_filter_match("anything", NULL) == true);
	ASSERT(sam3_bench_filter_match("", NULL) == true);

	/* Exact match. */
	ASSERT(sam3_bench_filter_match("matmul_f32", "matmul_f32") == true);
	ASSERT(sam3_bench_filter_match("matmul_f32", "matmul_f16") == false);

	/* Prefix glob: "matmul*" */
	ASSERT(sam3_bench_filter_match("matmul_f32", "matmul*") == true);
	ASSERT(sam3_bench_filter_match("matmul_f16", "matmul*") == true);
	ASSERT(sam3_bench_filter_match("conv2d_f32", "matmul*") == false);

	/* Suffix glob: "*f32" */
	ASSERT(sam3_bench_filter_match("matmul_f32", "*f32") == true);
	ASSERT(sam3_bench_filter_match("conv2d_f32", "*f32") == true);
	ASSERT(sam3_bench_filter_match("matmul_f16", "*f32") == false);

	/* Mismatch cases. */
	ASSERT(sam3_bench_filter_match("short", "very_long_filter") == false);
	ASSERT(sam3_bench_filter_match("abc", "*xyz") == false);
	ASSERT(sam3_bench_filter_match("abc", "xyz*") == false);

	/* Empty filter matches all. */
	ASSERT(sam3_bench_filter_match("anything", "") == true);
}

int main(void)
{
	test_config_defaults();
	test_bench_run_basic();
	test_bench_run_statistical();
	test_bench_run_with_flops();
	test_bench_run_with_bytes();
	test_env_detect();
	test_filter_match();

	TEST_REPORT();
}
