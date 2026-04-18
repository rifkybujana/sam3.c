/*
 * tests/test_bench.c - Unit tests for benchmark harness
 *
 * Tests config defaults, timing loop with statistics, environment
 * detection, filter matching, JSON round-trip serialization, and
 * baseline comparison with regression detection. Uses a trivial
 * callback that burns CPU cycles to produce measurable timings.
 *
 * Key types:  sam3_bench_config, sam3_bench_result, sam3_bench_env
 * Depends on: bench/bench.h, bench/bench_json.h, bench/bench_compare.h,
 *             test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "bench/bench.h"
#include "bench/bench_json.h"
#include "bench/bench_compare.h"

#include <string.h>
#include <stdio.h>

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

	/* Wrap glob: "*foo*" — substring contains. */
	ASSERT(sam3_bench_filter_match("video_per_frame_4obj", "*_4obj*") == true);
	ASSERT(sam3_bench_filter_match("matmul_f32_1024", "*1024*") == true);
	ASSERT(sam3_bench_filter_match("matmul_f32_1024", "*4obj*") == false);
	ASSERT(sam3_bench_filter_match("ab", "*abc*") == false);
	/* Trivial "**" matches anything. */
	ASSERT(sam3_bench_filter_match("anything", "**") == true);

	/* Empty filter matches all. */
	ASSERT(sam3_bench_filter_match("anything", "") == true);
}

/* --- JSON serialization tests --- */

#define TEST_JSON_PATH "/tmp/test_bench_results.json"

static void test_json_write_read_roundtrip(void)
{
	/* Build environment. */
	struct sam3_bench_env env;
	memset(&env, 0, sizeof(env));
	snprintf(env.chip, sizeof(env.chip), "Apple M2 Pro");
	snprintf(env.os, sizeof(env.os), "Darwin 24.6.0");
	env.cpu_cores = 12;
	env.gpu_cores = 19;
	snprintf(env.backend, sizeof(env.backend), "metal");
	snprintf(env.commit, sizeof(env.commit), "abc1234");
	snprintf(env.timestamp, sizeof(env.timestamp),
		 "2026-04-15T00:00:00Z");
	snprintf(env.model_variant, sizeof(env.model_variant), "hiera_large");

	/* Build config. */
	struct sam3_bench_config cfg;
	sam3_bench_config_defaults(&cfg);
	cfg.warmup_iters = 3;
	cfg.timed_iters = 25;
	cfg.statistical = true;
	cfg.threshold_pct = 7.5;

	/* Build two results. */
	struct sam3_bench_result results[2];
	memset(results, 0, sizeof(results));

	snprintf(results[0].name, sizeof(results[0].name),
		 "matmul_f32_1024x1024");
	snprintf(results[0].suite, sizeof(results[0].suite), "kernel");
	results[0].mean_ms = 1.234;
	results[0].min_ms = 1.100;
	results[0].max_ms = 1.500;
	results[0].stddev_ms = 0.045;
	results[0].gflops = 42.5;
	results[0].throughput_mbs = 1024.0;
	results[0].iterations = 25;

	snprintf(results[1].name, sizeof(results[1].name),
		 "softmax_f16_4096");
	snprintf(results[1].suite, sizeof(results[1].suite), "kernel");
	results[1].mean_ms = 0.567;
	results[1].min_ms = 0.510;
	results[1].max_ms = 0.620;
	results[1].stddev_ms = 0.012;
	results[1].gflops = 10.2;
	results[1].throughput_mbs = 512.5;
	results[1].iterations = 25;

	/* Write to file. */
	int err = sam3_bench_write_json(TEST_JSON_PATH, &env, &cfg,
					results, 2);
	ASSERT_EQ(err, 0);

	/* Read back. */
	struct sam3_bench_env env2;
	struct sam3_bench_result results2[16];
	int n_read = 0;

	err = sam3_bench_read_json(TEST_JSON_PATH, &env2,
				   results2, 16, &n_read);
	ASSERT_EQ(err, 0);
	ASSERT_EQ(n_read, 2);

	/* Verify environment fields. */
	ASSERT_EQ(strcmp(env2.chip, "Apple M2 Pro"), 0);
	ASSERT_EQ(strcmp(env2.os, "Darwin 24.6.0"), 0);
	ASSERT_EQ(env2.cpu_cores, 12);
	ASSERT_EQ(env2.gpu_cores, 19);
	ASSERT_EQ(strcmp(env2.backend, "metal"), 0);
	ASSERT_EQ(strcmp(env2.commit, "abc1234"), 0);
	ASSERT_EQ(strcmp(env2.timestamp, "2026-04-15T00:00:00Z"), 0);
	ASSERT_EQ(strcmp(env2.model_variant, "hiera_large"), 0);

	/* Verify first result. */
	ASSERT_EQ(strcmp(results2[0].name, "matmul_f32_1024x1024"), 0);
	ASSERT_EQ(strcmp(results2[0].suite, "kernel"), 0);
	ASSERT_NEAR(results2[0].mean_ms, 1.234, 0.001);
	ASSERT_NEAR(results2[0].min_ms, 1.100, 0.001);
	ASSERT_NEAR(results2[0].max_ms, 1.500, 0.001);
	ASSERT_NEAR(results2[0].stddev_ms, 0.045, 0.001);
	ASSERT_NEAR(results2[0].gflops, 42.5, 0.1);
	ASSERT_NEAR(results2[0].throughput_mbs, 1024.0, 0.1);
	ASSERT_EQ(results2[0].iterations, 25);

	/* Verify second result. */
	ASSERT_EQ(strcmp(results2[1].name, "softmax_f16_4096"), 0);
	ASSERT_EQ(strcmp(results2[1].suite, "kernel"), 0);
	ASSERT_NEAR(results2[1].mean_ms, 0.567, 0.001);
	ASSERT_NEAR(results2[1].min_ms, 0.510, 0.001);
	ASSERT_NEAR(results2[1].max_ms, 0.620, 0.001);
	ASSERT_NEAR(results2[1].stddev_ms, 0.012, 0.001);
	ASSERT_NEAR(results2[1].gflops, 10.2, 0.1);
	ASSERT_NEAR(results2[1].throughput_mbs, 512.5, 0.1);
	ASSERT_EQ(results2[1].iterations, 25);

	/* Clean up test file. */
	remove(TEST_JSON_PATH);
}

/* --- Comparison tests --- */

/* Helper to build a result with given timing values. */
static void fill_result(struct sam3_bench_result *r, const char *name,
			double mean, double stddev)
{
	memset(r, 0, sizeof(*r));
	snprintf(r->name, sizeof(r->name), "%s", name);
	snprintf(r->suite, sizeof(r->suite), "kernel");
	r->mean_ms = mean;
	r->min_ms = mean * 0.95;
	r->max_ms = mean * 1.05;
	r->stddev_ms = stddev;
	r->iterations = 50;
}

static void test_compare_no_regression(void)
{
	/* Baseline: 1.00ms, current: 1.03ms -> 3% delta, below 5% threshold. */
	struct sam3_bench_result base, cur;
	fill_result(&base, "test_op", 1.00, 0.0);
	fill_result(&cur, "test_op", 1.03, 0.0);

	int reg = sam3_bench_compare_results(&base, 1, &cur, 1, 5.0, false);
	ASSERT_EQ(reg, 0);
}

static void test_compare_with_regression(void)
{
	/* Baseline: 1.00ms, current: 1.10ms -> 10% delta, above 5% threshold. */
	struct sam3_bench_result base, cur;
	fill_result(&base, "test_op", 1.00, 0.0);
	fill_result(&cur, "test_op", 1.10, 0.0);

	int reg = sam3_bench_compare_results(&base, 1, &cur, 1, 5.0, false);
	ASSERT_EQ(reg, 1);
}

static void test_compare_improvement(void)
{
	/* Baseline: 1.00ms, current: 0.80ms -> -20% (faster), no regression. */
	struct sam3_bench_result base, cur;
	fill_result(&base, "test_op", 1.00, 0.0);
	fill_result(&cur, "test_op", 0.80, 0.0);

	int reg = sam3_bench_compare_results(&base, 1, &cur, 1, 5.0, false);
	ASSERT_EQ(reg, 0);
}

static void test_compare_statistical(void)
{
	/* Baseline: mean=1.00, stddev=0.05. Limit = 1.00 + 2*0.05 = 1.10. */
	struct sam3_bench_result base;
	fill_result(&base, "test_op", 1.00, 0.05);

	/* Current 1.08 is within limit (1.08 < 1.10) -> no regression. */
	struct sam3_bench_result cur1;
	fill_result(&cur1, "test_op", 1.08, 0.0);

	int reg1 = sam3_bench_compare_results(&base, 1, &cur1, 1,
					      5.0, true);
	ASSERT_EQ(reg1, 0);

	/* Current 1.15 exceeds limit (1.15 > 1.10) -> regression. */
	struct sam3_bench_result cur2;
	fill_result(&cur2, "test_op", 1.15, 0.0);

	int reg2 = sam3_bench_compare_results(&base, 1, &cur2, 1,
					      5.0, true);
	ASSERT_EQ(reg2, 1);
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

	/* JSON serialization tests. */
	test_json_write_read_roundtrip();

	/* Comparison tests. */
	test_compare_no_regression();
	test_compare_with_regression();
	test_compare_improvement();
	test_compare_statistical();

	TEST_REPORT();
}
