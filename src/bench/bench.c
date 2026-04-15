/*
 * src/bench/bench.c - Benchmark harness implementation
 *
 * Implements the timing loop, statistics computation, hardware environment
 * detection, and filter matching for the SAM3 benchmark framework. All
 * benchmark suites delegate to sam3_bench_run() for consistent timing
 * and reporting. Per-iteration timings are stored on the stack (cap 10000).
 *
 * Key types:  sam3_bench_config, sam3_bench_result, sam3_bench_env
 * Depends on: bench/bench.h, util/time.h, util/log.h
 * Used by:    bench_kernels.c, bench_pipeline.c, cli_bench.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#ifdef __APPLE__
#include <sys/utsname.h>
#include <sys/sysctl.h>
#endif

#ifdef __linux__
#include <sys/utsname.h>
#endif

#include "bench/bench.h"
#include "util/time.h"
#include "util/log.h"

/* Maximum per-iteration timing samples stored on the stack. */
#define BENCH_MAX_SAMPLES 10000

void sam3_bench_config_defaults(struct sam3_bench_config *cfg)
{
	if (!cfg)
		return;

	memset(cfg, 0, sizeof(*cfg));
	cfg->warmup_iters  = 5;
	cfg->timed_iters   = 50;
	cfg->threshold_pct = 5.0;
	cfg->statistical   = false;
	cfg->output_path   = NULL;
	cfg->baseline_path = NULL;
	cfg->backend       = SAM3_BACKEND_CPU;
	cfg->filter        = NULL;
	cfg->verbose       = false;
}

int sam3_bench_run(const struct sam3_bench_config *cfg,
		   const char *name, const char *suite,
		   void (*fn)(void *ctx), void *ctx,
		   double flops, double bytes,
		   struct sam3_bench_result *out)
{
	if (!cfg || !fn || !out || !name || !suite) {
		sam3_log_error("bench_run: NULL argument");
		return -1;
	}

	int iters = cfg->timed_iters;
	if (iters <= 0 || iters > BENCH_MAX_SAMPLES) {
		sam3_log_error("bench_run: timed_iters %d out of range [1, %d]",
			       iters, BENCH_MAX_SAMPLES);
		return -1;
	}

	/* Warmup phase: run without recording. */
	for (int i = 0; i < cfg->warmup_iters; i++)
		fn(ctx);

	/* Timed phase: record per-iteration nanosecond durations. */
	double samples_ms[BENCH_MAX_SAMPLES];
	for (int i = 0; i < iters; i++) {
		uint64_t t0 = sam3_time_ns();
		fn(ctx);
		uint64_t t1 = sam3_time_ns();
		samples_ms[i] = (double)(t1 - t0) / 1e6;

		if (cfg->verbose) {
			fprintf(stderr, "  [%s] iter %d: %.4f ms\n",
				name, i, samples_ms[i]);
		}
	}

	/* Compute statistics in a single pass for mean, min, max. */
	double sum = 0.0;
	double mn  = samples_ms[0];
	double mx  = samples_ms[0];

	for (int i = 0; i < iters; i++) {
		double v = samples_ms[i];
		sum += v;
		if (v < mn) mn = v;
		if (v > mx) mx = v;
	}

	double mean = sum / iters;

	/* Stddev (second pass, only if statistical mode). */
	double stddev = 0.0;
	if (cfg->statistical) {
		double sq_sum = 0.0;
		for (int i = 0; i < iters; i++) {
			double d = samples_ms[i] - mean;
			sq_sum += d * d;
		}
		stddev = sqrt(sq_sum / iters);
	}

	/* Fill result struct. */
	memset(out, 0, sizeof(*out));
	snprintf(out->name, sizeof(out->name), "%s", name);
	snprintf(out->suite, sizeof(out->suite), "%s", suite);
	out->mean_ms   = mean;
	out->min_ms    = mn;
	out->max_ms    = mx;
	out->stddev_ms = stddev;
	out->iterations = iters;

	/* Derived metrics. */
	if (flops > 0.0 && mean > 0.0)
		out->gflops = flops / (mean * 1e6);

	if (bytes > 0.0 && mean > 0.0)
		out->throughput_mbs = bytes / (mean * 1e3);

	return 0;
}

void sam3_bench_env_detect(struct sam3_bench_env *env,
			   enum sam3_backend_type backend)
{
	if (!env)
		return;

	memset(env, 0, sizeof(*env));

	/* OS info via uname. */
#if defined(__APPLE__) || defined(__linux__)
	{
		struct utsname u;
		if (uname(&u) == 0) {
			snprintf(env->os, sizeof(env->os), "%s %s",
				 u.sysname, u.release);
		} else {
			snprintf(env->os, sizeof(env->os), "unknown");
		}
	}
#else
	snprintf(env->os, sizeof(env->os), "unknown");
#endif

	/* Chip name and core counts (macOS via sysctl). */
#ifdef __APPLE__
	{
		size_t len = sizeof(env->chip);
		if (sysctlbyname("machdep.cpu.brand_string",
				 env->chip, &len, NULL, 0) != 0) {
			snprintf(env->chip, sizeof(env->chip), "unknown");
		}

		int ncpu = 0;
		len = sizeof(ncpu);
		if (sysctlbyname("hw.ncpu", &ncpu, &len, NULL, 0) == 0)
			env->cpu_cores = ncpu;
	}
#else
	snprintf(env->chip, sizeof(env->chip), "unknown");
#endif

	/* Backend string. */
	switch (backend) {
	case SAM3_BACKEND_CPU:
		snprintf(env->backend, sizeof(env->backend), "cpu");
		break;
	case SAM3_BACKEND_METAL:
		snprintf(env->backend, sizeof(env->backend), "metal");
		break;
	default:
		snprintf(env->backend, sizeof(env->backend), "unknown");
		break;
	}

	/* ISO 8601 timestamp. */
	{
		time_t now = time(NULL);
		struct tm *t = gmtime(&now);
		if (t) {
			strftime(env->timestamp, sizeof(env->timestamp),
				 "%Y-%m-%dT%H:%M:%SZ", t);
		}
	}
}

bool sam3_bench_filter_match(const char *name, const char *filter)
{
	/* NULL filter matches everything. */
	if (!filter)
		return true;

	if (!name)
		return false;

	size_t flen = strlen(filter);
	if (flen == 0)
		return true;

	/* Check for prefix glob: "foo*" */
	if (filter[flen - 1] == '*') {
		return strncmp(name, filter, flen - 1) == 0;
	}

	/* Check for suffix glob: "*foo" */
	if (filter[0] == '*') {
		const char *suffix = filter + 1;
		size_t slen = strlen(suffix);
		size_t nlen = strlen(name);
		if (nlen < slen)
			return false;
		return strcmp(name + nlen - slen, suffix) == 0;
	}

	/* Exact match. */
	return strcmp(name, filter) == 0;
}
