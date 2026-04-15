# Benchmarking System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a unified benchmarking system with a harness library, kernel/pipeline benchmark suites, JSON baselines, and regression detection, integrated into the `sam3` CLI.

**Architecture:** A `src/bench/` library providing timing loops, stats, JSON I/O, and comparison. Two suites (kernel microbenchmarks + pipeline benchmarks) register cases with the harness. The CLI dispatches to bench mode via `sam3 bench` subcommand. Results are JSON files; baselines live in `benchmarks/baselines/`.

**Tech Stack:** C11, vendored cJSON, existing `sam3_time_ns()` clock, existing backend vtable, existing arena allocator.

**Design doc:** `docs/plans/2026-04-15-benchmarking-system-design.md`

---

### Task 1: Bench Harness — Types and Config (`src/bench/bench.h`)

**Files:**
- Create: `src/bench/bench.h`

**Step 1: Create the header with types and API declarations**

```c
/*
 * src/bench/bench.h - Benchmark harness API
 *
 * Provides the execution framework for all SAM3 benchmarks: configurable
 * warmup/timed iterations, nanosecond timing with statistics, and result
 * collection. All benchmarks delegate to this harness — no benchmark
 * writes its own timing loop.
 *
 * Key types:  sam3_bench_config, sam3_bench_result, sam3_bench_env
 * Depends on: sam3/sam3_types.h, backend/backend.h
 * Used by:    bench_kernels.c, bench_pipeline.c, cli_bench.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_BENCH_H
#define SAM3_BENCH_H

#include "sam3/sam3_types.h"
#include "backend/backend.h"

#include <stdbool.h>

/* Maximum benchmark cases in a single suite run. */
#define SAM3_BENCH_MAX_RESULTS 512

/* Benchmark configuration. */
struct sam3_bench_config {
	int                    warmup_iters;   /* default: 5 */
	int                    timed_iters;    /* default: 50 */
	bool                   statistical;    /* compute stddev */
	double                 threshold_pct;  /* regression threshold, default 5.0 */
	const char            *output_path;    /* JSON output file, NULL = stdout */
	const char            *baseline_path;  /* JSON baseline to compare against */
	enum sam3_backend_type backend;        /* CPU or Metal */
	const char            *filter;         /* glob pattern to filter cases */
	bool                   verbose;        /* print per-iteration timings */
};

/* Result of a single benchmark case. */
struct sam3_bench_result {
	char    name[128];         /* e.g., "matmul_f32_1024x1024" */
	char    suite[16];         /* "kernel" or "pipeline" */
	double  mean_ms;
	double  min_ms;
	double  max_ms;
	double  stddev_ms;         /* 0 if !statistical */
	double  gflops;            /* 0 if not applicable */
	double  throughput_mbs;    /* MB/s, 0 if N/A */
	int     iterations;
};

/* Hardware/environment metadata. */
struct sam3_bench_env {
	char chip[64];             /* e.g., "Apple M2 Pro" */
	char os[64];               /* e.g., "Darwin 24.6.0" */
	int  cpu_cores;
	int  gpu_cores;            /* 0 if CPU-only */
	char backend[16];          /* "cpu" or "metal" */
	char commit[12];           /* short git SHA */
	char timestamp[32];        /* ISO 8601 */
	char model_variant[32];    /* e.g., "hiera_large" or "" */
};

/*
 * sam3_bench_config_defaults - Fill config with default values.
 *
 * @cfg: Config to initialize.
 */
void sam3_bench_config_defaults(struct sam3_bench_config *cfg);

/*
 * sam3_bench_run - Execute a single benchmark case.
 *
 * @cfg:   Benchmark configuration (warmup, iterations, etc.)
 * @name:  Human-readable name for this case
 * @suite: Suite name ("kernel" or "pipeline")
 * @fn:    Function to benchmark (called once per iteration)
 * @ctx:   Opaque context passed to fn
 * @flops: Total floating-point operations per call (0 to skip GFLOPS)
 * @bytes: Total bytes moved per call (0 to skip throughput)
 * @out:   Result struct to fill
 *
 * Returns 0 on success, -1 on error.
 */
int sam3_bench_run(const struct sam3_bench_config *cfg,
		   const char *name, const char *suite,
		   void (*fn)(void *ctx), void *ctx,
		   double flops, double bytes,
		   struct sam3_bench_result *out);

/*
 * sam3_bench_env_detect - Collect hardware/environment metadata.
 *
 * @env:     Struct to fill.
 * @backend: Which backend is being benchmarked.
 */
void sam3_bench_env_detect(struct sam3_bench_env *env,
			   enum sam3_backend_type backend);

/*
 * sam3_bench_filter_match - Check if a benchmark name matches the filter.
 *
 * @name:   Benchmark case name.
 * @filter: Glob pattern (NULL matches everything).
 *
 * Returns true if the name matches.
 */
bool sam3_bench_filter_match(const char *name, const char *filter);

#endif /* SAM3_BENCH_H */
```

**Step 2: Commit**

```bash
git add -f src/bench/bench.h
git commit -m "bench: add harness header with types and API"
```

---

### Task 2: Bench Harness — Implementation (`src/bench/bench.c`)

**Files:**
- Create: `src/bench/bench.c`

**Step 1: Write the test for the harness core**

Create `tests/test_bench.c`:

```c
/*
 * tests/test_bench.c - Bench harness unit tests
 *
 * Tests the benchmark harness timing loop, statistics computation,
 * config defaults, environment detection, and filter matching.
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
#include <math.h>

/* Dummy benchmark: just spin for a bit. */
static void dummy_bench(void *ctx)
{
	volatile int *counter = (volatile int *)ctx;
	(*counter)++;
}

static void test_config_defaults(void)
{
	struct sam3_bench_config cfg;
	sam3_bench_config_defaults(&cfg);
	ASSERT_EQ(cfg.warmup_iters, 5);
	ASSERT_EQ(cfg.timed_iters, 50);
	ASSERT(cfg.threshold_pct > 4.9 && cfg.threshold_pct < 5.1);
	ASSERT(!cfg.statistical);
	ASSERT(cfg.output_path == NULL);
	ASSERT(cfg.baseline_path == NULL);
	ASSERT(cfg.filter == NULL);
	ASSERT(!cfg.verbose);
}

static void test_bench_run_basic(void)
{
	struct sam3_bench_config cfg;
	sam3_bench_config_defaults(&cfg);
	cfg.warmup_iters = 2;
	cfg.timed_iters = 10;

	int counter = 0;
	struct sam3_bench_result result;
	int ret = sam3_bench_run(&cfg, "dummy", "test",
				 dummy_bench, &counter,
				 0.0, 0.0, &result);
	ASSERT_EQ(ret, 0);
	ASSERT_EQ(counter, 12);  /* 2 warmup + 10 timed */
	ASSERT_EQ(result.iterations, 10);
	ASSERT(result.mean_ms >= 0.0);
	ASSERT(result.min_ms >= 0.0);
	ASSERT(result.min_ms <= result.mean_ms);
	ASSERT(result.max_ms >= result.mean_ms);
	ASSERT(strcmp(result.name, "dummy") == 0);
	ASSERT(strcmp(result.suite, "test") == 0);
	ASSERT(result.gflops == 0.0);
	ASSERT(result.throughput_mbs == 0.0);
}

static void test_bench_run_statistical(void)
{
	struct sam3_bench_config cfg;
	sam3_bench_config_defaults(&cfg);
	cfg.warmup_iters = 1;
	cfg.timed_iters = 20;
	cfg.statistical = true;

	int counter = 0;
	struct sam3_bench_result result;
	int ret = sam3_bench_run(&cfg, "stat_dummy", "test",
				 dummy_bench, &counter,
				 0.0, 0.0, &result);
	ASSERT_EQ(ret, 0);
	ASSERT(result.stddev_ms >= 0.0);
}

static void test_bench_run_with_flops(void)
{
	struct sam3_bench_config cfg;
	sam3_bench_config_defaults(&cfg);
	cfg.warmup_iters = 1;
	cfg.timed_iters = 5;

	int counter = 0;
	struct sam3_bench_result result;
	/* Claim 1e9 flops per call */
	int ret = sam3_bench_run(&cfg, "flops_test", "test",
				 dummy_bench, &counter,
				 1e9, 0.0, &result);
	ASSERT_EQ(ret, 0);
	ASSERT(result.gflops > 0.0);
}

static void test_bench_run_with_bytes(void)
{
	struct sam3_bench_config cfg;
	sam3_bench_config_defaults(&cfg);
	cfg.warmup_iters = 1;
	cfg.timed_iters = 5;

	int counter = 0;
	struct sam3_bench_result result;
	/* Claim 1 GB per call */
	int ret = sam3_bench_run(&cfg, "bw_test", "test",
				 dummy_bench, &counter,
				 0.0, 1e9, &result);
	ASSERT_EQ(ret, 0);
	ASSERT(result.throughput_mbs > 0.0);
}

static void test_env_detect(void)
{
	struct sam3_bench_env env;
	memset(&env, 0, sizeof(env));
	sam3_bench_env_detect(&env, SAM3_BACKEND_CPU);
	ASSERT(strlen(env.os) > 0);
	ASSERT(env.cpu_cores > 0);
	ASSERT(strlen(env.backend) > 0);
	ASSERT(strlen(env.timestamp) > 0);
}

static void test_filter_match(void)
{
	/* NULL filter matches everything */
	ASSERT(sam3_bench_filter_match("anything", NULL));

	/* Exact match */
	ASSERT(sam3_bench_filter_match("matmul_f32", "matmul_f32"));

	/* Wildcard suffix */
	ASSERT(sam3_bench_filter_match("matmul_f32_1024", "matmul*"));
	ASSERT(!sam3_bench_filter_match("softmax_f32", "matmul*"));

	/* Wildcard prefix */
	ASSERT(sam3_bench_filter_match("matmul_f32_1024", "*1024"));
	ASSERT(!sam3_bench_filter_match("matmul_f32_512", "*1024"));
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
```

**Step 2: Run test to verify it fails**

```bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DSAM3_BENCH=ON && make test_bench 2>&1
```

Expected: Fails (bench.c doesn't exist yet).

**Step 3: Implement `src/bench/bench.c`**

```c
/*
 * src/bench/bench.c - Benchmark harness implementation
 *
 * Core timing loop, statistics computation, environment detection,
 * and glob-based filter matching. Uses sam3_time_ns() for nanosecond
 * resolution. All benchmarks delegate to sam3_bench_run() which handles
 * warmup, timed iterations, and result collection.
 *
 * Key types:  sam3_bench_config, sam3_bench_result, sam3_bench_env
 * Depends on: bench.h, util/time.h
 * Used by:    bench_kernels.c, bench_pipeline.c, cli_bench.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "bench.h"
#include "util/time.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <sys/utsname.h>
#else
#include <sys/utsname.h>
#include <unistd.h>
#endif

void sam3_bench_config_defaults(struct sam3_bench_config *cfg)
{
	memset(cfg, 0, sizeof(*cfg));
	cfg->warmup_iters = 5;
	cfg->timed_iters = 50;
	cfg->statistical = false;
	cfg->threshold_pct = 5.0;
	cfg->output_path = NULL;
	cfg->baseline_path = NULL;
	cfg->backend = SAM3_BACKEND_CPU;
	cfg->filter = NULL;
	cfg->verbose = false;
}

int sam3_bench_run(const struct sam3_bench_config *cfg,
		   const char *name, const char *suite,
		   void (*fn)(void *ctx), void *ctx,
		   double flops, double bytes,
		   struct sam3_bench_result *out)
{
	if (!cfg || !fn || !out)
		return -1;

	memset(out, 0, sizeof(*out));
	snprintf(out->name, sizeof(out->name), "%s", name);
	snprintf(out->suite, sizeof(out->suite), "%s", suite);

	/* Warmup */
	for (int i = 0; i < cfg->warmup_iters; i++)
		fn(ctx);

	int n = cfg->timed_iters;
	if (n < 1)
		n = 1;

	/* Allocate per-iteration timing on the stack (up to 10000) */
	double times_ms[10000];
	if (n > 10000)
		n = 10000;

	double total_ms = 0.0;
	double min_ms = 1e30;
	double max_ms = 0.0;

	for (int i = 0; i < n; i++) {
		uint64_t t0 = sam3_time_ns();
		fn(ctx);
		uint64_t t1 = sam3_time_ns();

		double dt = (double)(t1 - t0) / 1e6;
		times_ms[i] = dt;
		total_ms += dt;
		if (dt < min_ms)
			min_ms = dt;
		if (dt > max_ms)
			max_ms = dt;

		if (cfg->verbose)
			fprintf(stderr, "  [%s] iter %d: %.3f ms\n",
				name, i, dt);
	}

	double mean_ms = total_ms / n;

	/* Compute stddev if statistical mode */
	double stddev_ms = 0.0;
	if (cfg->statistical && n > 1) {
		double sum_sq = 0.0;
		for (int i = 0; i < n; i++) {
			double d = times_ms[i] - mean_ms;
			sum_sq += d * d;
		}
		stddev_ms = sqrt(sum_sq / (n - 1));
	}

	out->mean_ms = mean_ms;
	out->min_ms = min_ms;
	out->max_ms = max_ms;
	out->stddev_ms = stddev_ms;
	out->iterations = n;

	/* Compute derived metrics */
	if (flops > 0.0 && mean_ms > 0.0)
		out->gflops = flops / (mean_ms * 1e6);

	if (bytes > 0.0 && mean_ms > 0.0)
		out->throughput_mbs = bytes / (mean_ms * 1e3);

	return 0;
}

void sam3_bench_env_detect(struct sam3_bench_env *env,
			   enum sam3_backend_type backend)
{
	memset(env, 0, sizeof(*env));

	/* OS info */
	struct utsname uts;
	if (uname(&uts) == 0)
		snprintf(env->os, sizeof(env->os), "%s %s",
			 uts.sysname, uts.release);

#ifdef __APPLE__
	/* Chip name */
	size_t len = sizeof(env->chip);
	if (sysctlbyname("machdep.cpu.brand_string", env->chip, &len,
			  NULL, 0) != 0)
		snprintf(env->chip, sizeof(env->chip), "Unknown");

	/* CPU cores */
	int ncpu = 0;
	len = sizeof(ncpu);
	if (sysctlbyname("hw.ncpu", &ncpu, &len, NULL, 0) == 0)
		env->cpu_cores = ncpu;

	/* GPU cores (approximate from perflevel) */
	int gpu_cores = 0;
	len = sizeof(gpu_cores);
	if (sysctlbyname("machdep.cpu.core_count", &gpu_cores, &len,
			  NULL, 0) == 0)
		env->gpu_cores = 0;  /* Not easily available; leave 0 */
#else
	snprintf(env->chip, sizeof(env->chip), "Unknown");
	env->cpu_cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif

	/* Backend */
	snprintf(env->backend, sizeof(env->backend), "%s",
		 backend == SAM3_BACKEND_METAL ? "metal" : "cpu");

	/* Git commit (best-effort) */
	env->commit[0] = '\0';

	/* Timestamp (ISO 8601) */
	time_t now = time(NULL);
	struct tm *tm = gmtime(&now);
	if (tm)
		strftime(env->timestamp, sizeof(env->timestamp),
			 "%Y-%m-%dT%H:%M:%SZ", tm);
}

bool sam3_bench_filter_match(const char *name, const char *filter)
{
	if (!filter)
		return true;

	size_t flen = strlen(filter);
	size_t nlen = strlen(name);

	/* No wildcards: exact match */
	const char *star = strchr(filter, '*');
	if (!star)
		return strcmp(name, filter) == 0;

	/* Prefix wildcard: *suffix */
	if (star == filter) {
		const char *suffix = filter + 1;
		size_t slen = strlen(suffix);
		if (slen > nlen)
			return false;
		return strcmp(name + nlen - slen, suffix) == 0;
	}

	/* Suffix wildcard: prefix* */
	if (star == filter + flen - 1) {
		size_t plen = flen - 1;
		return strncmp(name, filter, plen) == 0;
	}

	/* Middle wildcard: prefix*suffix */
	size_t plen = (size_t)(star - filter);
	const char *suffix = star + 1;
	size_t slen = strlen(suffix);
	if (plen + slen > nlen)
		return false;
	return strncmp(name, filter, plen) == 0 &&
	       strcmp(name + nlen - slen, suffix) == 0;
}
```

**Step 4: Update CMakeLists.txt for bench library and test**

Add after the existing `option()` block (around line 16):

```cmake
option(SAM3_BENCH  "Build benchmarking harness" OFF)
```

Add after the `add_library(sam3 ...)` and its linking section (around line 144), before the CLI tool section:

```cmake
# Bench harness library
if(SAM3_BENCH)
	add_definitions(-DSAM3_HAS_BENCH)
	file(GLOB_RECURSE SAM3_BENCH_SOURCES "src/bench/*.c")
	add_library(sam3_bench STATIC ${SAM3_BENCH_SOURCES})
	target_link_libraries(sam3_bench sam3)
endif()
```

In the test section, add after the `foreach(test_src ...)` block:

```cmake
	# test_bench needs the bench library
	if(SAM3_BENCH AND TARGET test_bench)
		target_link_libraries(test_bench sam3_bench)
	endif()
```

Also update the CLI target to conditionally link bench:

```cmake
if(SAM3_BENCH)
	target_link_libraries(sam3_cli sam3_bench)
endif()
```

**Step 5: Build and run test**

```bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DSAM3_BENCH=ON && make -j$(nproc) test_bench && ./test_bench
```

Expected: All tests pass.

**Step 6: Commit**

```bash
git add -f src/bench/bench.c tests/test_bench.c CMakeLists.txt
git commit -m "bench: implement harness core with timing, stats, and filter"
```

---

### Task 3: JSON Serialization (`src/bench/bench_json.h`, `src/bench/bench_json.c`)

**Files:**
- Create: `src/bench/bench_json.h`
- Create: `src/bench/bench_json.c`

**Step 1: Write the test**

Add to `tests/test_bench.c` (new test functions, add calls in `main()`):

```c
#include "bench/bench_json.h"
#include <unistd.h>

static void test_json_write_read_roundtrip(void)
{
	struct sam3_bench_env env;
	memset(&env, 0, sizeof(env));
	snprintf(env.chip, sizeof(env.chip), "Test Chip");
	snprintf(env.os, sizeof(env.os), "TestOS 1.0");
	env.cpu_cores = 8;
	snprintf(env.backend, sizeof(env.backend), "cpu");
	snprintf(env.commit, sizeof(env.commit), "abc123");
	snprintf(env.timestamp, sizeof(env.timestamp), "2026-04-15T00:00:00Z");

	struct sam3_bench_result results[2];
	memset(results, 0, sizeof(results));
	snprintf(results[0].name, sizeof(results[0].name), "matmul_f32_1024");
	snprintf(results[0].suite, sizeof(results[0].suite), "kernel");
	results[0].mean_ms = 1.23;
	results[0].min_ms = 1.10;
	results[0].max_ms = 1.45;
	results[0].gflops = 1748.0;
	results[0].iterations = 50;

	snprintf(results[1].name, sizeof(results[1].name), "softmax_f32_4096");
	snprintf(results[1].suite, sizeof(results[1].suite), "kernel");
	results[1].mean_ms = 0.08;
	results[1].min_ms = 0.07;
	results[1].max_ms = 0.10;
	results[1].iterations = 50;

	struct sam3_bench_config cfg;
	sam3_bench_config_defaults(&cfg);

	/* Write to temp file */
	const char *path = "/tmp/test_bench_results.json";
	int ret = sam3_bench_write_json(path, &env, &cfg, results, 2);
	ASSERT_EQ(ret, 0);

	/* Read back */
	struct sam3_bench_env env2;
	struct sam3_bench_result results2[SAM3_BENCH_MAX_RESULTS];
	int n_results2 = 0;
	ret = sam3_bench_read_json(path, &env2, results2,
				   SAM3_BENCH_MAX_RESULTS, &n_results2);
	ASSERT_EQ(ret, 0);
	ASSERT_EQ(n_results2, 2);

	/* Verify roundtrip */
	ASSERT(strcmp(env2.chip, "Test Chip") == 0);
	ASSERT_EQ(env2.cpu_cores, 8);
	ASSERT(strcmp(results2[0].name, "matmul_f32_1024") == 0);
	ASSERT_NEAR(results2[0].mean_ms, 1.23, 0.01);
	ASSERT_NEAR(results2[0].gflops, 1748.0, 0.1);
	ASSERT(strcmp(results2[1].name, "softmax_f32_4096") == 0);

	unlink(path);
}
```

**Step 2: Run test — expected fail (no bench_json.h yet)**

**Step 3: Create `src/bench/bench_json.h`**

```c
/*
 * src/bench/bench_json.h - JSON serialization for benchmark results
 *
 * Writes and reads benchmark results in a versioned JSON format.
 * Uses the vendored cJSON library for parsing and generation.
 *
 * Key types:  (uses types from bench.h)
 * Depends on: bench.h
 * Used by:    bench_compare.c, cli_bench.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_BENCH_JSON_H
#define SAM3_BENCH_JSON_H

#include "bench.h"

/*
 * sam3_bench_write_json - Write benchmark results to a JSON file.
 *
 * @path:       Output file path
 * @env:        Hardware/environment metadata
 * @cfg:        Config used for this run
 * @results:    Array of benchmark results
 * @n_results:  Number of results
 *
 * Returns 0 on success, -1 on error.
 */
int sam3_bench_write_json(const char *path,
			  const struct sam3_bench_env *env,
			  const struct sam3_bench_config *cfg,
			  const struct sam3_bench_result *results,
			  int n_results);

/*
 * sam3_bench_read_json - Read benchmark results from a JSON file.
 *
 * @path:         Input file path
 * @env:          Environment metadata (filled on output)
 * @results:      Array to fill with results
 * @max_results:  Capacity of results array
 * @n_results:    Number of results read (output)
 *
 * Returns 0 on success, -1 on error.
 */
int sam3_bench_read_json(const char *path,
			 struct sam3_bench_env *env,
			 struct sam3_bench_result *results,
			 int max_results, int *n_results);

/*
 * sam3_bench_print_results - Print results table to stderr.
 *
 * @results:    Array of benchmark results
 * @n_results:  Number of results
 */
void sam3_bench_print_results(const struct sam3_bench_result *results,
			      int n_results);

#endif /* SAM3_BENCH_JSON_H */
```

**Step 4: Create `src/bench/bench_json.c`**

```c
/*
 * src/bench/bench_json.c - JSON serialization implementation
 *
 * Implements JSON write/read for benchmark results using vendored cJSON.
 * Format version 1: env object, config object, results array.
 *
 * Key types:  (uses types from bench.h)
 * Depends on: bench_json.h, core/json/cJSON.h
 * Used by:    bench_compare.c, cli_bench.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "bench_json.h"
#include "core/json/cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int sam3_bench_write_json(const char *path,
			  const struct sam3_bench_env *env,
			  const struct sam3_bench_config *cfg,
			  const struct sam3_bench_result *results,
			  int n_results)
{
	cJSON *root = cJSON_CreateObject();
	if (!root)
		return -1;

	cJSON_AddNumberToObject(root, "version", 1);

	/* Environment */
	cJSON *jenv = cJSON_AddObjectToObject(root, "env");
	cJSON_AddStringToObject(jenv, "chip", env->chip);
	cJSON_AddStringToObject(jenv, "os", env->os);
	cJSON_AddNumberToObject(jenv, "cpu_cores", env->cpu_cores);
	cJSON_AddNumberToObject(jenv, "gpu_cores", env->gpu_cores);
	cJSON_AddStringToObject(jenv, "backend", env->backend);
	cJSON_AddStringToObject(jenv, "commit", env->commit);
	cJSON_AddStringToObject(jenv, "timestamp", env->timestamp);
	cJSON_AddStringToObject(jenv, "model_variant", env->model_variant);

	/* Config */
	cJSON *jcfg = cJSON_AddObjectToObject(root, "config");
	cJSON_AddNumberToObject(jcfg, "warmup_iters", cfg->warmup_iters);
	cJSON_AddNumberToObject(jcfg, "timed_iters", cfg->timed_iters);
	cJSON_AddBoolToObject(jcfg, "statistical", cfg->statistical);
	cJSON_AddNumberToObject(jcfg, "threshold_pct", cfg->threshold_pct);

	/* Results */
	cJSON *jarr = cJSON_AddArrayToObject(root, "results");
	for (int i = 0; i < n_results; i++) {
		cJSON *jr = cJSON_CreateObject();
		cJSON_AddStringToObject(jr, "name", results[i].name);
		cJSON_AddStringToObject(jr, "suite", results[i].suite);
		cJSON_AddNumberToObject(jr, "mean_ms", results[i].mean_ms);
		cJSON_AddNumberToObject(jr, "min_ms", results[i].min_ms);
		cJSON_AddNumberToObject(jr, "max_ms", results[i].max_ms);
		cJSON_AddNumberToObject(jr, "stddev_ms", results[i].stddev_ms);
		cJSON_AddNumberToObject(jr, "gflops", results[i].gflops);
		cJSON_AddNumberToObject(jr, "throughput_mbs",
					results[i].throughput_mbs);
		cJSON_AddNumberToObject(jr, "iterations",
					results[i].iterations);
		cJSON_AddItemToArray(jarr, jr);
	}

	char *json_str = cJSON_Print(root);
	cJSON_Delete(root);
	if (!json_str)
		return -1;

	FILE *fp = fopen(path, "w");
	if (!fp) {
		free(json_str);
		return -1;
	}
	fputs(json_str, fp);
	fclose(fp);
	free(json_str);
	return 0;
}

int sam3_bench_read_json(const char *path,
			 struct sam3_bench_env *env,
			 struct sam3_bench_result *results,
			 int max_results, int *n_results)
{
	FILE *fp = fopen(path, "r");
	if (!fp)
		return -1;

	fseek(fp, 0, SEEK_END);
	long size = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	char *buf = malloc(size + 1);
	if (!buf) {
		fclose(fp);
		return -1;
	}
	fread(buf, 1, size, fp);
	buf[size] = '\0';
	fclose(fp);

	cJSON *root = cJSON_Parse(buf);
	free(buf);
	if (!root)
		return -1;

	/* Environment */
	if (env) {
		memset(env, 0, sizeof(*env));
		cJSON *jenv = cJSON_GetObjectItem(root, "env");
		if (jenv) {
			cJSON *j;
			j = cJSON_GetObjectItem(jenv, "chip");
			if (j && j->valuestring)
				snprintf(env->chip, sizeof(env->chip),
					 "%s", j->valuestring);
			j = cJSON_GetObjectItem(jenv, "os");
			if (j && j->valuestring)
				snprintf(env->os, sizeof(env->os),
					 "%s", j->valuestring);
			j = cJSON_GetObjectItem(jenv, "cpu_cores");
			if (j)
				env->cpu_cores = j->valueint;
			j = cJSON_GetObjectItem(jenv, "gpu_cores");
			if (j)
				env->gpu_cores = j->valueint;
			j = cJSON_GetObjectItem(jenv, "backend");
			if (j && j->valuestring)
				snprintf(env->backend, sizeof(env->backend),
					 "%s", j->valuestring);
			j = cJSON_GetObjectItem(jenv, "commit");
			if (j && j->valuestring)
				snprintf(env->commit, sizeof(env->commit),
					 "%s", j->valuestring);
			j = cJSON_GetObjectItem(jenv, "timestamp");
			if (j && j->valuestring)
				snprintf(env->timestamp,
					 sizeof(env->timestamp),
					 "%s", j->valuestring);
			j = cJSON_GetObjectItem(jenv, "model_variant");
			if (j && j->valuestring)
				snprintf(env->model_variant,
					 sizeof(env->model_variant),
					 "%s", j->valuestring);
		}
	}

	/* Results */
	*n_results = 0;
	cJSON *jarr = cJSON_GetObjectItem(root, "results");
	if (jarr && cJSON_IsArray(jarr)) {
		int n = cJSON_GetArraySize(jarr);
		if (n > max_results)
			n = max_results;
		for (int i = 0; i < n; i++) {
			cJSON *jr = cJSON_GetArrayItem(jarr, i);
			struct sam3_bench_result *r = &results[i];
			memset(r, 0, sizeof(*r));

			cJSON *j;
			j = cJSON_GetObjectItem(jr, "name");
			if (j && j->valuestring)
				snprintf(r->name, sizeof(r->name),
					 "%s", j->valuestring);
			j = cJSON_GetObjectItem(jr, "suite");
			if (j && j->valuestring)
				snprintf(r->suite, sizeof(r->suite),
					 "%s", j->valuestring);
			j = cJSON_GetObjectItem(jr, "mean_ms");
			if (j) r->mean_ms = j->valuedouble;
			j = cJSON_GetObjectItem(jr, "min_ms");
			if (j) r->min_ms = j->valuedouble;
			j = cJSON_GetObjectItem(jr, "max_ms");
			if (j) r->max_ms = j->valuedouble;
			j = cJSON_GetObjectItem(jr, "stddev_ms");
			if (j) r->stddev_ms = j->valuedouble;
			j = cJSON_GetObjectItem(jr, "gflops");
			if (j) r->gflops = j->valuedouble;
			j = cJSON_GetObjectItem(jr, "throughput_mbs");
			if (j) r->throughput_mbs = j->valuedouble;
			j = cJSON_GetObjectItem(jr, "iterations");
			if (j) r->iterations = j->valueint;
		}
		*n_results = n;
	}

	cJSON_Delete(root);
	return 0;
}

void sam3_bench_print_results(const struct sam3_bench_result *results,
			      int n_results)
{
	fprintf(stderr,
		"\n %-30s %9s %9s %9s %9s %9s\n",
		"Benchmark", "Mean(ms)", "Min(ms)", "Max(ms)",
		"GFLOPS", "MB/s");
	fprintf(stderr,
		"------------------------------------"
		"------------------------------------\n");

	for (int i = 0; i < n_results; i++) {
		const struct sam3_bench_result *r = &results[i];
		fprintf(stderr, " %-30s %9.3f %9.3f %9.3f",
			r->name, r->mean_ms, r->min_ms, r->max_ms);
		if (r->gflops > 0.0)
			fprintf(stderr, " %9.1f", r->gflops);
		else
			fprintf(stderr, " %9s", "-");
		if (r->throughput_mbs > 0.0)
			fprintf(stderr, " %9.1f", r->throughput_mbs);
		else
			fprintf(stderr, " %9s", "-");
		fprintf(stderr, "\n");
	}
	fprintf(stderr,
		"------------------------------------"
		"------------------------------------\n\n");
}
```

**Step 5: Build and run test**

```bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DSAM3_BENCH=ON && make -j$(nproc) test_bench && ./test_bench
```

Expected: All tests pass including the JSON roundtrip test.

**Step 6: Commit**

```bash
git add -f src/bench/bench_json.h src/bench/bench_json.c tests/test_bench.c
git commit -m "bench: add JSON serialization for results and environment"
```

---

### Task 4: Baseline Comparison (`src/bench/bench_compare.h`, `src/bench/bench_compare.c`)

**Files:**
- Create: `src/bench/bench_compare.h`
- Create: `src/bench/bench_compare.c`

**Step 1: Write the test**

Add to `tests/test_bench.c`:

```c
#include "bench/bench_compare.h"

static void test_compare_no_regression(void)
{
	struct sam3_bench_result baseline[1];
	memset(baseline, 0, sizeof(baseline));
	snprintf(baseline[0].name, sizeof(baseline[0].name), "matmul_f32");
	baseline[0].mean_ms = 1.00;

	struct sam3_bench_result current[1];
	memset(current, 0, sizeof(current));
	snprintf(current[0].name, sizeof(current[0].name), "matmul_f32");
	current[0].mean_ms = 1.03;  /* 3% slower — under 5% threshold */

	int regressions = sam3_bench_compare_results(
		baseline, 1, current, 1, 5.0, false);
	ASSERT_EQ(regressions, 0);
}

static void test_compare_with_regression(void)
{
	struct sam3_bench_result baseline[1];
	memset(baseline, 0, sizeof(baseline));
	snprintf(baseline[0].name, sizeof(baseline[0].name), "matmul_f32");
	baseline[0].mean_ms = 1.00;

	struct sam3_bench_result current[1];
	memset(current, 0, sizeof(current));
	snprintf(current[0].name, sizeof(current[0].name), "matmul_f32");
	current[0].mean_ms = 1.10;  /* 10% slower — over 5% threshold */

	int regressions = sam3_bench_compare_results(
		baseline, 1, current, 1, 5.0, false);
	ASSERT_EQ(regressions, 1);
}

static void test_compare_improvement(void)
{
	struct sam3_bench_result baseline[1];
	memset(baseline, 0, sizeof(baseline));
	snprintf(baseline[0].name, sizeof(baseline[0].name), "matmul_f32");
	baseline[0].mean_ms = 1.00;

	struct sam3_bench_result current[1];
	memset(current, 0, sizeof(current));
	snprintf(current[0].name, sizeof(current[0].name), "matmul_f32");
	current[0].mean_ms = 0.80;  /* 20% faster — no regression */

	int regressions = sam3_bench_compare_results(
		baseline, 1, current, 1, 5.0, false);
	ASSERT_EQ(regressions, 0);
}

static void test_compare_statistical(void)
{
	struct sam3_bench_result baseline[1];
	memset(baseline, 0, sizeof(baseline));
	snprintf(baseline[0].name, sizeof(baseline[0].name), "matmul_f32");
	baseline[0].mean_ms = 1.00;
	baseline[0].stddev_ms = 0.05;

	struct sam3_bench_result current[1];
	memset(current, 0, sizeof(current));
	snprintf(current[0].name, sizeof(current[0].name), "matmul_f32");
	current[0].mean_ms = 1.08;  /* Within 2*stddev=0.10 */

	int regressions = sam3_bench_compare_results(
		baseline, 1, current, 1, 5.0, true);
	ASSERT_EQ(regressions, 0);

	/* Now exceed 2*stddev */
	current[0].mean_ms = 1.15;  /* > 1.00 + 2*0.05 = 1.10 */
	regressions = sam3_bench_compare_results(
		baseline, 1, current, 1, 5.0, true);
	ASSERT_EQ(regressions, 1);
}
```

**Step 2: Run test — expected fail**

**Step 3: Create `src/bench/bench_compare.h`**

```c
/*
 * src/bench/bench_compare.h - Baseline comparison and regression detection
 *
 * Compares current benchmark results against a stored baseline.
 * Supports percentage-based and statistical comparison modes.
 *
 * Key types:  (uses types from bench.h)
 * Depends on: bench.h
 * Used by:    cli_bench.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_BENCH_COMPARE_H
#define SAM3_BENCH_COMPARE_H

#include "bench.h"

#include <stdbool.h>

/*
 * sam3_bench_compare_results - Compare current results against baseline.
 *
 * @baseline:      Array of baseline results
 * @n_baseline:    Number of baseline results
 * @current:       Array of current results
 * @n_current:     Number of current results
 * @threshold_pct: Regression threshold (percentage)
 * @statistical:   Use statistical comparison (2*stddev)
 *
 * Prints comparison report to stderr. Returns the number of regressions
 * detected (0 = all OK).
 */
int sam3_bench_compare_results(const struct sam3_bench_result *baseline,
			       int n_baseline,
			       const struct sam3_bench_result *current,
			       int n_current,
			       double threshold_pct, bool statistical);

/*
 * sam3_bench_compare_files - Load baseline JSON and compare with current.
 *
 * @baseline_path: Path to baseline JSON file
 * @current:       Array of current results
 * @n_current:     Number of current results
 * @threshold_pct: Regression threshold
 * @statistical:   Use statistical comparison
 *
 * Returns the number of regressions, or -1 on file error.
 */
int sam3_bench_compare_files(const char *baseline_path,
			     const struct sam3_bench_result *current,
			     int n_current,
			     double threshold_pct, bool statistical);

#endif /* SAM3_BENCH_COMPARE_H */
```

**Step 4: Create `src/bench/bench_compare.c`**

```c
/*
 * src/bench/bench_compare.c - Baseline comparison implementation
 *
 * Matches current results to baseline by name, computes deltas,
 * and flags regressions. Percentage mode: flag if delta > threshold.
 * Statistical mode: flag if current > baseline_mean + 2*baseline_stddev.
 *
 * Key types:  (uses types from bench.h)
 * Depends on: bench_compare.h, bench_json.h
 * Used by:    cli_bench.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "bench_compare.h"
#include "bench_json.h"

#include <stdio.h>
#include <string.h>

/* Find a result by name in an array. Returns NULL if not found. */
static const struct sam3_bench_result *
find_by_name(const struct sam3_bench_result *results, int n,
	     const char *name)
{
	for (int i = 0; i < n; i++) {
		if (strcmp(results[i].name, name) == 0)
			return &results[i];
	}
	return NULL;
}

int sam3_bench_compare_results(const struct sam3_bench_result *baseline,
			       int n_baseline,
			       const struct sam3_bench_result *current,
			       int n_current,
			       double threshold_pct, bool statistical)
{
	int regressions = 0;

	fprintf(stderr,
		"\nBenchmark Comparison (threshold: %.1f%%%s)\n",
		threshold_pct,
		statistical ? ", statistical" : "");
	fprintf(stderr,
		"------------------------------------"
		"------------------------------------\n");

	for (int i = 0; i < n_current; i++) {
		const struct sam3_bench_result *cur = &current[i];
		const struct sam3_bench_result *base =
			find_by_name(baseline, n_baseline, cur->name);

		if (!base) {
			fprintf(stderr, "  %-28s %8.2fms  (NEW)\n",
				cur->name, cur->mean_ms);
			continue;
		}

		double delta_ms = cur->mean_ms - base->mean_ms;
		double delta_pct = (base->mean_ms > 0.0)
			? 100.0 * delta_ms / base->mean_ms
			: 0.0;

		bool is_regression;
		if (statistical && base->stddev_ms > 0.0) {
			is_regression = cur->mean_ms >
				base->mean_ms + 2.0 * base->stddev_ms;
		} else {
			is_regression = delta_pct > threshold_pct;
		}

		if (is_regression)
			regressions++;

		fprintf(stderr, "  %-28s %8.2fms -> %8.2fms  (%+.1f%%)  %s\n",
			cur->name, base->mean_ms, cur->mean_ms,
			delta_pct,
			is_regression ? "REGRESSION" : "OK");
	}

	fprintf(stderr,
		"------------------------------------"
		"------------------------------------\n");

	if (regressions > 0)
		fprintf(stderr, "  %d regression(s) detected\n\n",
			regressions);
	else
		fprintf(stderr, "  No regressions detected\n\n");

	return regressions;
}

int sam3_bench_compare_files(const char *baseline_path,
			     const struct sam3_bench_result *current,
			     int n_current,
			     double threshold_pct, bool statistical)
{
	struct sam3_bench_env base_env;
	struct sam3_bench_result base_results[SAM3_BENCH_MAX_RESULTS];
	int n_base = 0;

	int ret = sam3_bench_read_json(baseline_path, &base_env,
				       base_results,
				       SAM3_BENCH_MAX_RESULTS, &n_base);
	if (ret != 0) {
		fprintf(stderr, "error: failed to read baseline '%s'\n",
			baseline_path);
		return -1;
	}

	return sam3_bench_compare_results(base_results, n_base,
					  current, n_current,
					  threshold_pct, statistical);
}
```

**Step 5: Build and run test**

```bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DSAM3_BENCH=ON && make -j$(nproc) test_bench && ./test_bench
```

Expected: All tests pass.

**Step 6: Commit**

```bash
git add -f src/bench/bench_compare.h src/bench/bench_compare.c tests/test_bench.c
git commit -m "bench: add baseline comparison with regression detection"
```

---

### Task 5: Kernel Microbenchmark Suite (`src/bench/bench_kernels.h`, `src/bench/bench_kernels.c`)

**Files:**
- Create: `src/bench/bench_kernels.h`
- Create: `src/bench/bench_kernels.c`

**Step 1: Create `src/bench/bench_kernels.h`**

```c
/*
 * src/bench/bench_kernels.h - Kernel microbenchmark suite
 *
 * Registers individual operation benchmarks at controlled sizes and
 * dtypes. Each benchmark builds a one-op graph and evaluates it
 * through the backend vtable.
 *
 * Key types:  (uses types from bench.h)
 * Depends on: bench.h, backend/backend.h
 * Used by:    cli_bench.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_BENCH_KERNELS_H
#define SAM3_BENCH_KERNELS_H

#include "bench.h"

/*
 * sam3_bench_run_kernels - Run all kernel microbenchmarks.
 *
 * @cfg:        Benchmark configuration
 * @be:         Backend to benchmark on
 * @results:    Array to fill with results
 * @max_results: Capacity of results array
 *
 * Returns number of results written, or -1 on error.
 */
int sam3_bench_run_kernels(const struct sam3_bench_config *cfg,
			   struct sam3_backend *be,
			   struct sam3_bench_result *results,
			   int max_results);

#endif /* SAM3_BENCH_KERNELS_H */
```

**Step 2: Create `src/bench/bench_kernels.c`**

This file is large. It registers benchmarks for matmul, add, mul, softmax, layernorm, gelu, silu, transpose, and conv2d across sizes and dtypes.

Each benchmark follows the same pattern:
1. Allocate tensors from a scratch arena
2. Fill with random data
3. Build a one-node graph
4. The harness calls `sam3_backend_graph_eval()` in a timing loop
5. Reset arena between cases

```c
/*
 * src/bench/bench_kernels.c - Kernel microbenchmark implementation
 *
 * Benchmarks individual compute operations at various sizes and dtypes.
 * Each case builds a minimal compute graph with one operation and
 * evaluates it through the backend vtable.
 *
 * Key types:  (uses types from bench.h)
 * Depends on: bench_kernels.h, bench.h, core/graph.h, core/tensor.h,
 *             core/alloc.h, backend/backend.h
 * Used by:    cli_bench.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "bench_kernels.h"
#include "core/graph.h"
#include "core/tensor.h"
#include "core/alloc.h"
#include "core/half.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Context for a single kernel benchmark case. */
struct kernel_bench_ctx {
	struct sam3_backend *be;
	struct sam3_graph    graph;
};

/* Callback: evaluate the pre-built graph. */
static void eval_graph_fn(void *opaque)
{
	struct kernel_bench_ctx *kctx = opaque;
	kctx->be->ops->graph_eval(kctx->be, &kctx->graph);
	if (kctx->be->ops->arena_reset)
		kctx->be->ops->arena_reset(kctx->be);
}

/* Fill tensor data with random values. */
static void fill_random(struct sam3_tensor *t)
{
	int n = sam3_tensor_nelems(t);
	if (t->dtype == SAM3_DTYPE_F32) {
		float *d = t->data;
		for (int i = 0; i < n; i++)
			d[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
	} else if (t->dtype == SAM3_DTYPE_F16) {
		uint16_t *d = t->data;
		for (int i = 0; i < n; i++) {
			float v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
			d[i] = f32_to_fp16(v);
		}
	} else if (t->dtype == SAM3_DTYPE_BF16) {
		uint16_t *d = t->data;
		for (int i = 0; i < n; i++) {
			float v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
			d[i] = f32_to_bf16(v);
		}
	}
}

/* Initialize a 2D tensor in the arena. */
static void init_tensor_2d(struct sam3_tensor *t, struct sam3_arena *arena,
			   enum sam3_dtype dtype, int rows, int cols)
{
	memset(t, 0, sizeof(*t));
	t->dtype = dtype;
	t->n_dims = 2;
	t->dims[0] = rows;
	t->dims[1] = cols;
	t->dims[2] = 1;
	t->dims[3] = 1;
	sam3_tensor_compute_strides(t);
	t->nbytes = (size_t)rows * cols * sam3_dtype_size(dtype);
	t->data = sam3_arena_alloc(arena, t->nbytes);
}

/* Initialize a 1D tensor in the arena. */
static void init_tensor_1d(struct sam3_tensor *t, struct sam3_arena *arena,
			   enum sam3_dtype dtype, int size)
{
	memset(t, 0, sizeof(*t));
	t->dtype = dtype;
	t->n_dims = 1;
	t->dims[0] = size;
	t->dims[1] = 1;
	t->dims[2] = 1;
	t->dims[3] = 1;
	sam3_tensor_compute_strides(t);
	t->nbytes = (size_t)size * sam3_dtype_size(dtype);
	t->data = sam3_arena_alloc(arena, t->nbytes);
}

/* Build a matmul graph: C = A @ B */
static int setup_matmul(struct kernel_bench_ctx *kctx,
			struct sam3_arena *arena,
			enum sam3_dtype dtype, int m, int n, int k)
{
	struct sam3_tensor a, b, c;
	init_tensor_2d(&a, arena, dtype, m, k);
	init_tensor_2d(&b, arena, dtype, k, n);
	init_tensor_2d(&c, arena, dtype, m, n);

	if (!a.data || !b.data || !c.data)
		return -1;

	fill_random(&a);
	fill_random(&b);

	sam3_graph_init(&kctx->graph);
	struct sam3_tensor *inputs[2] = {&a, &b};

	/* We need stable tensor pointers — copy to arena */
	struct sam3_tensor *pa = sam3_arena_alloc(arena, sizeof(*pa));
	struct sam3_tensor *pb = sam3_arena_alloc(arena, sizeof(*pb));
	struct sam3_tensor *pc = sam3_arena_alloc(arena, sizeof(*pc));
	*pa = a; *pb = b; *pc = c;

	struct sam3_tensor *in[2] = {pa, pb};
	sam3_graph_add_op(&kctx->graph, SAM3_OP_MATMUL, in, 2, pc);
	return 0;
}

/* Matmul FLOPS: 2*M*N*K */
static double matmul_flops(int m, int n, int k)
{
	return 2.0 * m * n * k;
}

/* Add/Mul/Gelu/Silu benchmarks: elementwise on 1D */
static int setup_elementwise(struct kernel_bench_ctx *kctx,
			     struct sam3_arena *arena,
			     enum sam3_op op, enum sam3_dtype dtype, int n)
{
	struct sam3_tensor *a = sam3_arena_alloc(arena, sizeof(*a));
	init_tensor_1d(a, arena, dtype, n);

	struct sam3_tensor *out = sam3_arena_alloc(arena, sizeof(*out));
	init_tensor_1d(out, arena, dtype, n);

	if (!a->data || !out->data)
		return -1;

	fill_random(a);

	sam3_graph_init(&kctx->graph);

	if (op == SAM3_OP_ADD || op == SAM3_OP_MUL) {
		struct sam3_tensor *b = sam3_arena_alloc(arena, sizeof(*b));
		init_tensor_1d(b, arena, dtype, n);
		if (!b->data)
			return -1;
		fill_random(b);
		struct sam3_tensor *in[2] = {a, b};
		sam3_graph_add_op(&kctx->graph, op, in, 2, out);
	} else {
		struct sam3_tensor *in[1] = {a};
		sam3_graph_add_op(&kctx->graph, op, in, 1, out);
	}
	return 0;
}

/* Softmax benchmark: 2D [1, seq_len] */
static int setup_softmax(struct kernel_bench_ctx *kctx,
			 struct sam3_arena *arena,
			 enum sam3_dtype dtype, int seq_len)
{
	struct sam3_tensor *a = sam3_arena_alloc(arena, sizeof(*a));
	init_tensor_2d(a, arena, dtype, 1, seq_len);

	struct sam3_tensor *out = sam3_arena_alloc(arena, sizeof(*out));
	init_tensor_2d(out, arena, dtype, 1, seq_len);

	if (!a->data || !out->data)
		return -1;

	fill_random(a);
	sam3_graph_init(&kctx->graph);
	struct sam3_tensor *in[1] = {a};
	sam3_graph_add_op(&kctx->graph, SAM3_OP_SOFTMAX, in, 1, out);
	return 0;
}

/* LayerNorm benchmark: 2D [batch, hidden] with gamma/beta */
static int setup_layernorm(struct kernel_bench_ctx *kctx,
			   struct sam3_arena *arena,
			   enum sam3_dtype dtype, int hidden)
{
	struct sam3_tensor *x = sam3_arena_alloc(arena, sizeof(*x));
	init_tensor_2d(x, arena, dtype, 1, hidden);

	struct sam3_tensor *gamma = sam3_arena_alloc(arena, sizeof(*gamma));
	init_tensor_1d(gamma, arena, dtype, hidden);

	struct sam3_tensor *beta = sam3_arena_alloc(arena, sizeof(*beta));
	init_tensor_1d(beta, arena, dtype, hidden);

	struct sam3_tensor *out = sam3_arena_alloc(arena, sizeof(*out));
	init_tensor_2d(out, arena, dtype, 1, hidden);

	if (!x->data || !gamma->data || !beta->data || !out->data)
		return -1;

	fill_random(x);
	fill_random(gamma);
	fill_random(beta);

	sam3_graph_init(&kctx->graph);
	struct sam3_tensor *in[3] = {x, gamma, beta};
	sam3_graph_add_op(&kctx->graph, SAM3_OP_LAYERNORM, in, 3, out);
	return 0;
}

/* Transpose benchmark: 2D [rows, cols] -> [cols, rows] */
static int setup_transpose(struct kernel_bench_ctx *kctx,
			   struct sam3_arena *arena,
			   enum sam3_dtype dtype, int rows, int cols)
{
	struct sam3_tensor *a = sam3_arena_alloc(arena, sizeof(*a));
	init_tensor_2d(a, arena, dtype, rows, cols);

	struct sam3_tensor *out = sam3_arena_alloc(arena, sizeof(*out));
	init_tensor_2d(out, arena, dtype, cols, rows);

	if (!a->data || !out->data)
		return -1;

	fill_random(a);

	sam3_graph_init(&kctx->graph);
	struct sam3_tensor *in[1] = {a};
	sam3_graph_add_op(&kctx->graph, SAM3_OP_TRANSPOSE, in, 1, out);
	return 0;
}

/* dtype name for benchmark naming */
static const char *dtype_tag(enum sam3_dtype d)
{
	switch (d) {
	case SAM3_DTYPE_F32:  return "f32";
	case SAM3_DTYPE_F16:  return "f16";
	case SAM3_DTYPE_BF16: return "bf16";
	default:              return "unk";
	}
}

int sam3_bench_run_kernels(const struct sam3_bench_config *cfg,
			   struct sam3_backend *be,
			   struct sam3_bench_result *results,
			   int max_results)
{
	int n = 0;
	struct sam3_arena arena;

	/* 256 MB scratch arena for benchmark tensors */
	if (sam3_arena_init(&arena, 256 * 1024 * 1024) != SAM3_OK) {
		fprintf(stderr, "error: failed to allocate bench arena\n");
		return -1;
	}

	/* Matmul sizes */
	static const int mm_sizes[][3] = {
		{ 256,  256,  256},
		{ 512,  512,  512},
		{1024, 1024, 1024},
		{2048, 2048, 2048},
		{4096, 4096, 4096},
	};
	static const enum sam3_dtype mm_dtypes[] = {
		SAM3_DTYPE_F32, SAM3_DTYPE_F16, SAM3_DTYPE_BF16
	};

	for (int di = 0; di < 3; di++) {
		for (int si = 0; si < 5; si++) {
			int m = mm_sizes[si][0];
			int nk = mm_sizes[si][1];
			int k = mm_sizes[si][2];

			char name[128];
			snprintf(name, sizeof(name), "matmul_%s_%dx%dx%d",
				 dtype_tag(mm_dtypes[di]), m, nk, k);

			if (!sam3_bench_filter_match(name, cfg->filter))
				continue;
			if (n >= max_results)
				goto done;

			sam3_arena_reset(&arena);
			struct kernel_bench_ctx kctx = { .be = be };
			if (setup_matmul(&kctx, &arena, mm_dtypes[di],
					 m, nk, k) != 0)
				continue;

			/* Allocate backend tensors */
			for (int j = 0; j < kctx.graph.n_nodes; j++) {
				struct sam3_node *node = &kctx.graph.nodes[j];
				be->ops->alloc_tensor(be, node->output);
				for (int q = 0; q < node->n_inputs; q++)
					be->ops->alloc_tensor(be,
							      node->inputs[q]);
			}

			sam3_bench_run(cfg, name, "kernel",
				       eval_graph_fn, &kctx,
				       matmul_flops(m, nk, k), 0.0,
				       &results[n]);
			n++;
		}
	}

	/* Elementwise ops: add, mul, gelu, silu */
	static const struct {
		enum sam3_op op;
		const char  *tag;
	} elem_ops[] = {
		{SAM3_OP_ADD,  "add"},
		{SAM3_OP_MUL,  "mul"},
		{SAM3_OP_GELU, "gelu"},
		{SAM3_OP_SILU, "silu"},
	};
	static const int elem_sizes[] = {
		1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024
	};
	static const enum sam3_dtype elem_dtypes[] = {
		SAM3_DTYPE_F32, SAM3_DTYPE_F16
	};

	for (int oi = 0; oi < 4; oi++) {
		for (int di = 0; di < 2; di++) {
			for (int si = 0; si < 3; si++) {
				int sz = elem_sizes[si];
				char name[128];
				snprintf(name, sizeof(name), "%s_%s_%dM",
					 elem_ops[oi].tag,
					 dtype_tag(elem_dtypes[di]),
					 sz / (1024 * 1024));

				if (!sam3_bench_filter_match(name, cfg->filter))
					continue;
				if (n >= max_results)
					goto done;

				sam3_arena_reset(&arena);
				struct kernel_bench_ctx kctx = { .be = be };
				if (setup_elementwise(&kctx, &arena,
						      elem_ops[oi].op,
						      elem_dtypes[di],
						      sz) != 0)
					continue;

				for (int j = 0; j < kctx.graph.n_nodes; j++) {
					struct sam3_node *node =
						&kctx.graph.nodes[j];
					be->ops->alloc_tensor(be,
							      node->output);
					for (int q = 0; q < node->n_inputs; q++)
						be->ops->alloc_tensor(
							be,
							node->inputs[q]);
				}

				double bw = (double)sz *
					sam3_dtype_size(elem_dtypes[di]) * 3;
				sam3_bench_run(cfg, name, "kernel",
					       eval_graph_fn, &kctx,
					       0.0, bw, &results[n]);
				n++;
			}
		}
	}

	/* Softmax */
	static const int sm_sizes[] = {1024, 4096, 16384, 65536};
	for (int di = 0; di < 2; di++) {
		for (int si = 0; si < 4; si++) {
			char name[128];
			snprintf(name, sizeof(name), "softmax_%s_%d",
				 dtype_tag(elem_dtypes[di]), sm_sizes[si]);

			if (!sam3_bench_filter_match(name, cfg->filter))
				continue;
			if (n >= max_results)
				goto done;

			sam3_arena_reset(&arena);
			struct kernel_bench_ctx kctx = { .be = be };
			if (setup_softmax(&kctx, &arena, elem_dtypes[di],
					  sm_sizes[si]) != 0)
				continue;

			for (int j = 0; j < kctx.graph.n_nodes; j++) {
				struct sam3_node *node = &kctx.graph.nodes[j];
				be->ops->alloc_tensor(be, node->output);
				for (int q = 0; q < node->n_inputs; q++)
					be->ops->alloc_tensor(be,
							      node->inputs[q]);
			}

			sam3_bench_run(cfg, name, "kernel",
				       eval_graph_fn, &kctx,
				       0.0, 0.0, &results[n]);
			n++;
		}
	}

	/* LayerNorm */
	static const int ln_sizes[] = {256, 512, 1024, 2048};
	for (int di = 0; di < 2; di++) {
		for (int si = 0; si < 4; si++) {
			char name[128];
			snprintf(name, sizeof(name), "layernorm_%s_%d",
				 dtype_tag(elem_dtypes[di]), ln_sizes[si]);

			if (!sam3_bench_filter_match(name, cfg->filter))
				continue;
			if (n >= max_results)
				goto done;

			sam3_arena_reset(&arena);
			struct kernel_bench_ctx kctx = { .be = be };
			if (setup_layernorm(&kctx, &arena, elem_dtypes[di],
					    ln_sizes[si]) != 0)
				continue;

			for (int j = 0; j < kctx.graph.n_nodes; j++) {
				struct sam3_node *node = &kctx.graph.nodes[j];
				be->ops->alloc_tensor(be, node->output);
				for (int q = 0; q < node->n_inputs; q++)
					be->ops->alloc_tensor(be,
							      node->inputs[q]);
			}

			sam3_bench_run(cfg, name, "kernel",
				       eval_graph_fn, &kctx,
				       0.0, 0.0, &results[n]);
			n++;
		}
	}

	/* Transpose */
	static const int tr_sizes[][2] = {{1024, 1024}, {2048, 2048}};
	for (int di = 0; di < 2; di++) {
		for (int si = 0; si < 2; si++) {
			int r = tr_sizes[si][0], c = tr_sizes[si][1];
			char name[128];
			snprintf(name, sizeof(name), "transpose_%s_%dx%d",
				 dtype_tag(elem_dtypes[di]), r, c);

			if (!sam3_bench_filter_match(name, cfg->filter))
				continue;
			if (n >= max_results)
				goto done;

			sam3_arena_reset(&arena);
			struct kernel_bench_ctx kctx = { .be = be };
			if (setup_transpose(&kctx, &arena, elem_dtypes[di],
					    r, c) != 0)
				continue;

			for (int j = 0; j < kctx.graph.n_nodes; j++) {
				struct sam3_node *node = &kctx.graph.nodes[j];
				be->ops->alloc_tensor(be, node->output);
				for (int q = 0; q < node->n_inputs; q++)
					be->ops->alloc_tensor(be,
							      node->inputs[q]);
			}

			double bw = (double)r * c *
				sam3_dtype_size(elem_dtypes[di]) * 2;
			sam3_bench_run(cfg, name, "kernel",
				       eval_graph_fn, &kctx,
				       0.0, bw, &results[n]);
			n++;
		}
	}

done:
	sam3_arena_free(&arena);
	return n;
}
```

**Step 3: Commit**

```bash
git add -f src/bench/bench_kernels.h src/bench/bench_kernels.c
git commit -m "bench: add kernel microbenchmark suite"
```

---

### Task 6: Pipeline Benchmark Suite (`src/bench/bench_pipeline.h`, `src/bench/bench_pipeline.c`)

**Files:**
- Create: `src/bench/bench_pipeline.h`
- Create: `src/bench/bench_pipeline.c`

**Step 1: Create `src/bench/bench_pipeline.h`**

```c
/*
 * src/bench/bench_pipeline.h - Pipeline benchmark suite
 *
 * End-to-end inference benchmarks using real model weights. Tests each
 * pipeline stage (image encode, text encode, prompt encode, mask decode)
 * and full inference with different prompt types.
 *
 * Key types:  (uses types from bench.h)
 * Depends on: bench.h, sam3/sam3.h
 * Used by:    cli_bench.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_BENCH_PIPELINE_H
#define SAM3_BENCH_PIPELINE_H

#include "bench.h"
#include "sam3/sam3.h"

/*
 * sam3_bench_run_pipeline - Run all pipeline benchmarks.
 *
 * @cfg:         Benchmark configuration
 * @ctx:         Initialized sam3 context with model loaded
 * @results:     Array to fill with results
 * @max_results: Capacity of results array
 *
 * Returns number of results written, or -1 on error.
 */
int sam3_bench_run_pipeline(const struct sam3_bench_config *cfg,
			    sam3_ctx *ctx,
			    struct sam3_bench_result *results,
			    int max_results);

#endif /* SAM3_BENCH_PIPELINE_H */
```

**Step 2: Create `src/bench/bench_pipeline.c`**

```c
/*
 * src/bench/bench_pipeline.c - Pipeline benchmark implementation
 *
 * Benchmarks end-to-end inference stages using real model weights.
 * Uses synthetic test images (solid gradient) for deterministic timing.
 * Automatically adapts to whichever backbone is loaded.
 *
 * Key types:  (uses types from bench.h)
 * Depends on: bench_pipeline.h, bench.h, sam3/sam3.h
 * Used by:    cli_bench.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "bench_pipeline.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Context for image_encode benchmark. */
struct image_encode_ctx {
	sam3_ctx      *ctx;
	uint8_t       *pixels;
	int            width;
	int            height;
};

static void image_encode_fn(void *opaque)
{
	struct image_encode_ctx *ic = opaque;
	sam3_set_image(ic->ctx, ic->pixels, ic->width, ic->height);
}

/* Context for full pipeline benchmark (point prompt). */
struct pipeline_point_ctx {
	sam3_ctx           *ctx;
	struct sam3_prompt  prompt;
};

static void pipeline_point_fn(void *opaque)
{
	struct pipeline_point_ctx *pc = opaque;
	struct sam3_result result;
	memset(&result, 0, sizeof(result));
	sam3_segment(pc->ctx, &pc->prompt, 1, &result);
	sam3_result_free(&result);
}

/* Context for full pipeline benchmark (box prompt). */
struct pipeline_box_ctx {
	sam3_ctx           *ctx;
	struct sam3_prompt  prompt;
};

static void pipeline_box_fn(void *opaque)
{
	struct pipeline_box_ctx *bc = opaque;
	struct sam3_result result;
	memset(&result, 0, sizeof(result));
	sam3_segment(bc->ctx, &bc->prompt, 1, &result);
	sam3_result_free(&result);
}

/* Context for text pipeline benchmark. */
struct pipeline_text_ctx {
	sam3_ctx           *ctx;
	struct sam3_prompt  prompt;
};

static void pipeline_text_fn(void *opaque)
{
	struct pipeline_text_ctx *tc = opaque;
	struct sam3_result result;
	memset(&result, 0, sizeof(result));
	sam3_segment(tc->ctx, &tc->prompt, 1, &result);
	sam3_result_free(&result);
}

/* Generate a synthetic gradient test image. */
static uint8_t *make_test_image(int size)
{
	uint8_t *pixels = malloc((size_t)size * size * 3);
	if (!pixels)
		return NULL;

	for (int y = 0; y < size; y++) {
		for (int x = 0; x < size; x++) {
			int idx = (y * size + x) * 3;
			pixels[idx + 0] = (uint8_t)(255 * x / size);
			pixels[idx + 1] = (uint8_t)(255 * y / size);
			pixels[idx + 2] = 128;
		}
	}
	return pixels;
}

int sam3_bench_run_pipeline(const struct sam3_bench_config *cfg,
			    sam3_ctx *ctx,
			    struct sam3_bench_result *results,
			    int max_results)
{
	int n = 0;
	int img_size = sam3_get_image_size(ctx);
	if (img_size <= 0) {
		fprintf(stderr, "error: no model loaded\n");
		return -1;
	}

	uint8_t *pixels = make_test_image(img_size);
	if (!pixels) {
		fprintf(stderr, "error: failed to allocate test image\n");
		return -1;
	}

	/* 1. Image encode benchmark */
	if (n < max_results &&
	    sam3_bench_filter_match("image_encode", cfg->filter)) {
		struct image_encode_ctx ic = {
			.ctx = ctx,
			.pixels = pixels,
			.width = img_size,
			.height = img_size,
		};
		sam3_bench_run(cfg, "image_encode", "pipeline",
			       image_encode_fn, &ic,
			       0.0, 0.0, &results[n]);
		n++;
	}

	/* Set image once for subsequent benchmarks */
	sam3_set_image(ctx, pixels, img_size, img_size);

	/* 2. Full pipeline with point prompt (center of image) */
	if (n < max_results &&
	    sam3_bench_filter_match("full_pipeline_point", cfg->filter)) {
		struct pipeline_point_ctx pc = {
			.ctx = ctx,
			.prompt = {
				.type = SAM3_PROMPT_POINT,
				.point = {
					.x = (float)img_size / 2,
					.y = (float)img_size / 2,
					.label = 1,
				},
			},
		};
		sam3_set_prompt_space(ctx, img_size, img_size);
		sam3_bench_run(cfg, "full_pipeline_point", "pipeline",
			       pipeline_point_fn, &pc,
			       0.0, 0.0, &results[n]);
		n++;
	}

	/* 3. Full pipeline with box prompt */
	if (n < max_results &&
	    sam3_bench_filter_match("full_pipeline_box", cfg->filter)) {
		float quarter = (float)img_size / 4;
		struct pipeline_box_ctx bc = {
			.ctx = ctx,
			.prompt = {
				.type = SAM3_PROMPT_BOX,
				.box = {
					.x1 = quarter,
					.y1 = quarter,
					.x2 = quarter * 3,
					.y2 = quarter * 3,
				},
			},
		};
		sam3_set_prompt_space(ctx, img_size, img_size);
		sam3_bench_run(cfg, "full_pipeline_box", "pipeline",
			       pipeline_box_fn, &bc,
			       0.0, 0.0, &results[n]);
		n++;
	}

	/* 4. Full pipeline with text prompt */
	if (n < max_results &&
	    sam3_bench_filter_match("full_pipeline_text", cfg->filter)) {
		struct pipeline_text_ctx tc = {
			.ctx = ctx,
			.prompt = {
				.type = SAM3_PROMPT_TEXT,
				.text = "cat",
			},
		};
		sam3_bench_run(cfg, "full_pipeline_text", "pipeline",
			       pipeline_text_fn, &tc,
			       0.0, 0.0, &results[n]);
		n++;
	}

	free(pixels);
	return n;
}
```

**Step 3: Commit**

```bash
git add -f src/bench/bench_pipeline.h src/bench/bench_pipeline.c
git commit -m "bench: add pipeline benchmark suite"
```

---

### Task 7: CLI Integration (`tools/cli_bench.h`, `tools/cli_bench.c`)

**Files:**
- Create: `tools/cli_bench.h`
- Create: `tools/cli_bench.c`
- Modify: `tools/sam3_cli.c` (add bench subcommand)
- Modify: `CMakeLists.txt` (add cli_bench.c to sam3_cli sources)

**Step 1: Create `tools/cli_bench.h`**

```c
/*
 * tools/cli_bench.h - Bench subcommand declaration
 *
 * Key types:  (none)
 * Depends on: (none)
 * Used by:    tools/sam3_cli.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_CLI_BENCH_H
#define SAM3_CLI_BENCH_H

int cli_bench(int argc, char **argv);

#endif
```

**Step 2: Create `tools/cli_bench.c`**

```c
/*
 * tools/cli_bench.c - Bench subcommand implementation
 *
 * Parses bench-specific CLI arguments and dispatches to kernel and/or
 * pipeline benchmark suites. Handles JSON output, baseline comparison,
 * and result printing.
 *
 * Key types:  (none)
 * Depends on: cli_bench.h, cli_common.h, bench/bench.h,
 *             bench/bench_json.h, bench/bench_compare.h,
 *             bench/bench_kernels.h, bench/bench_pipeline.h
 * Used by:    tools/sam3_cli.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "cli_bench.h"
#include "cli_common.h"
#include "sam3/sam3.h"
#include "bench/bench.h"
#include "bench/bench_json.h"
#include "bench/bench_compare.h"
#include "bench/bench_kernels.h"
#include "bench/bench_pipeline.h"
#include "backend/backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_bench_usage(const char *prog)
{
	fprintf(stderr, "Usage: %s bench [SUITE] [OPTIONS]\n\n", prog);
	fprintf(stderr, "Suites:\n");
	fprintf(stderr, "  kernels    Kernel microbenchmarks only\n");
	fprintf(stderr, "  pipeline   Pipeline benchmarks (requires --model)\n");
	fprintf(stderr, "  all        Both suites (default)\n\n");
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "  --model PATH      .sam3 model weights\n");
	fprintf(stderr, "  --backend cpu|metal  Backend (default: auto)\n");
	fprintf(stderr, "  --output PATH     Write JSON results\n");
	fprintf(stderr, "  --compare PATH    Compare against baseline\n");
	fprintf(stderr, "  --threshold PCT   Regression threshold "
		"(default: 5.0%%)\n");
	fprintf(stderr, "  --statistical     Statistical comparison\n");
	fprintf(stderr, "  --warmup N        Warmup iterations (default: 5)\n");
	fprintf(stderr, "  --iters N         Timed iterations (default: 50)\n");
	fprintf(stderr, "  --filter PATTERN  Filter benchmarks by name\n");
	fprintf(stderr, "  -v                Verbose output\n");
}

int cli_bench(int argc, char **argv)
{
	struct sam3_bench_config cfg;
	sam3_bench_config_defaults(&cfg);

	const char *suite = "all";
	const char *model_path = NULL;
	const char *backend_str = NULL;

	/* Parse arguments */
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--help") == 0 ||
		    strcmp(argv[i], "-h") == 0) {
			print_bench_usage("sam3");
			return SAM3_EXIT_OK;
		}
		if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
			model_path = argv[++i];
		} else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
			backend_str = argv[++i];
		} else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
			cfg.output_path = argv[++i];
		} else if (strcmp(argv[i], "--compare") == 0 && i + 1 < argc) {
			cfg.baseline_path = argv[++i];
		} else if (strcmp(argv[i], "--threshold") == 0 &&
			   i + 1 < argc) {
			cfg.threshold_pct = atof(argv[++i]);
		} else if (strcmp(argv[i], "--statistical") == 0) {
			cfg.statistical = true;
		} else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
			cfg.warmup_iters = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
			cfg.timed_iters = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
			cfg.filter = argv[++i];
		} else if (strcmp(argv[i], "-v") == 0) {
			cfg.verbose = true;
		} else if (argv[i][0] != '-') {
			suite = argv[i];
		}
	}

	/* Resolve backend */
	if (backend_str) {
		if (strcmp(backend_str, "metal") == 0)
			cfg.backend = SAM3_BACKEND_METAL;
		else if (strcmp(backend_str, "cpu") == 0)
			cfg.backend = SAM3_BACKEND_CPU;
		else {
			fprintf(stderr, "error: unknown backend '%s'\n",
				backend_str);
			return SAM3_EXIT_USAGE;
		}
	} else {
#ifdef SAM3_HAS_METAL
		cfg.backend = SAM3_BACKEND_METAL;
#else
		cfg.backend = SAM3_BACKEND_CPU;
#endif
	}

	bool run_kernels = strcmp(suite, "kernels") == 0 ||
			   strcmp(suite, "all") == 0;
	bool run_pipeline = strcmp(suite, "pipeline") == 0 ||
			    strcmp(suite, "all") == 0;

	if (run_pipeline && !model_path) {
		fprintf(stderr,
			"error: pipeline benchmarks require --model\n");
		return SAM3_EXIT_USAGE;
	}

	/* Collect results */
	struct sam3_bench_result results[SAM3_BENCH_MAX_RESULTS];
	int n_results = 0;

	/* Kernel benchmarks */
	if (run_kernels) {
		fprintf(stderr, "Running kernel benchmarks...\n");
		struct sam3_backend *be = sam3_backend_init(cfg.backend);
		if (!be) {
			fprintf(stderr, "error: failed to init backend\n");
			return SAM3_EXIT_INTERNAL;
		}

		int n = sam3_bench_run_kernels(&cfg, be,
					       results + n_results,
					       SAM3_BENCH_MAX_RESULTS -
					       n_results);
		if (n > 0)
			n_results += n;

		sam3_backend_free(be);
	}

	/* Pipeline benchmarks */
	if (run_pipeline) {
		fprintf(stderr, "Running pipeline benchmarks...\n");
		sam3_ctx *ctx = sam3_init();
		if (!ctx) {
			fprintf(stderr, "error: failed to init context\n");
			return SAM3_EXIT_INTERNAL;
		}

		enum sam3_error err = sam3_load_model(ctx, model_path);
		if (err != SAM3_OK) {
			fprintf(stderr, "error: failed to load model: %d\n",
				err);
			sam3_free(ctx);
			return sam3_error_to_exit(err);
		}

		int n = sam3_bench_run_pipeline(&cfg, ctx,
						results + n_results,
						SAM3_BENCH_MAX_RESULTS -
						n_results);
		if (n > 0)
			n_results += n;

		sam3_free(ctx);
	}

	if (n_results == 0) {
		fprintf(stderr, "No benchmarks matched.\n");
		return SAM3_EXIT_OK;
	}

	/* Print results table */
	sam3_bench_print_results(results, n_results);

	/* Environment metadata */
	struct sam3_bench_env env;
	sam3_bench_env_detect(&env, cfg.backend);

	/* Write JSON if requested */
	if (cfg.output_path) {
		if (sam3_bench_write_json(cfg.output_path, &env, &cfg,
					 results, n_results) != 0) {
			fprintf(stderr, "error: failed to write '%s'\n",
				cfg.output_path);
			return SAM3_EXIT_IO;
		}
		fprintf(stderr, "Results written to %s\n", cfg.output_path);
	}

	/* Compare against baseline if requested */
	if (cfg.baseline_path) {
		int regressions = sam3_bench_compare_files(
			cfg.baseline_path, results, n_results,
			cfg.threshold_pct, cfg.statistical);
		if (regressions < 0)
			return SAM3_EXIT_IO;
		if (regressions > 0)
			return 1;  /* CI: exit 1 on regression */
	}

	return SAM3_EXIT_OK;
}
```

**Step 3: Modify `tools/sam3_cli.c` — add bench subcommand**

Add after the `#include "cli_info.h"` line:

```c
#ifdef SAM3_HAS_BENCH
#include "cli_bench.h"
#endif
```

In `print_usage()`, add a line after `"  info       Print model file metadata\n"`:

```c
#ifdef SAM3_HAS_BENCH
	fprintf(stderr, "  bench      Run performance benchmarks\n");
#endif
```

In the subcommand dispatch block (after `cli_info`), add:

```c
#ifdef SAM3_HAS_BENCH
	if (strcmp(cmd, "bench") == 0)
		return cli_bench(argc - 1, argv + 1);
#endif
```

**Step 4: Modify `CMakeLists.txt` — add cli_bench.c**

In the `sam3_cli` target source list, wrap the bench source conditionally. After the existing `add_executable(sam3_cli ...)` block, add:

```cmake
if(SAM3_BENCH)
	target_sources(sam3_cli PRIVATE tools/cli_bench.c)
endif()
```

**Step 5: Create `benchmarks/baselines/.gitkeep`**

```bash
mkdir -p benchmarks/baselines
touch benchmarks/baselines/.gitkeep
```

**Step 6: Build and verify**

```bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DSAM3_BENCH=ON && make -j$(nproc) sam3_cli
./sam3_cli bench --help
```

Expected: Prints bench usage.

```bash
./sam3_cli bench kernels --backend cpu --warmup 2 --iters 5 --filter "matmul_f32_256*"
```

Expected: Runs a quick matmul benchmark and prints results.

**Step 7: Commit**

```bash
git add -f tools/cli_bench.h tools/cli_bench.c tools/sam3_cli.c CMakeLists.txt benchmarks/baselines/.gitkeep
git commit -m "bench: integrate benchmarking into sam3 CLI as bench subcommand"
```

---

### Task 8: Final Integration Test and Cleanup

**Step 1: Run full test suite to ensure nothing broke**

```bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DSAM3_BENCH=ON && make -j$(nproc) && ctest --output-on-failure
```

Expected: All existing tests pass plus `test_bench`.

**Step 2: Run kernel benchmarks with JSON output**

```bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DSAM3_BENCH=ON && make -j$(nproc) sam3_cli
./sam3_cli bench kernels --backend cpu --warmup 3 --iters 10 --output /tmp/bench_baseline.json
```

Expected: JSON file written with results.

**Step 3: Compare against self (baseline = current run)**

```bash
./sam3_cli bench kernels --backend cpu --warmup 3 --iters 10 --compare /tmp/bench_baseline.json
```

Expected: Comparison report with small deltas, exit code 0.

**Step 4: Commit any final fixes**

```bash
git add -A && git commit -m "bench: final integration test fixes"
```

---

## Dependency Graph

```
Task 1 (bench.h types)
  └── Task 2 (bench.c harness + tests)
        ├── Task 3 (bench_json.h/c + tests)
        │     └── Task 4 (bench_compare.h/c + tests)
        ├── Task 5 (bench_kernels.h/c)
        └── Task 6 (bench_pipeline.h/c)
              └── Task 7 (CLI integration)
                    └── Task 8 (integration test)
```

Tasks 3-4 and 5-6 can be parallelized (JSON/compare is independent of kernel/pipeline suites). Task 7 depends on all of 3-6. Task 8 is final verification.
