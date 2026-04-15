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
