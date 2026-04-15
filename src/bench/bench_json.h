/*
 * src/bench/bench_json.h - JSON serialization for benchmark results
 *
 * Provides read/write of benchmark results and environment metadata
 * in a versioned JSON format. Used by the CLI to persist results to
 * disk for later comparison and by bench_compare to load baselines.
 * Also provides a tabular stderr printer for human-readable output.
 *
 * Key types:  sam3_bench_result, sam3_bench_env
 * Depends on: bench/bench.h
 * Used by:    bench_compare.c, cli_bench.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_BENCH_JSON_H
#define SAM3_BENCH_JSON_H

#include "bench/bench.h"

/*
 * sam3_bench_write_json - Write benchmark results to a JSON file.
 *
 * @path:      Output file path.
 * @env:       Environment metadata to include.
 * @cfg:       Benchmark configuration to include.
 * @results:   Array of benchmark results.
 * @n_results: Number of results.
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
 * @path:        Input file path.
 * @env:         Environment struct to fill (may be NULL to skip).
 * @results:     Array to fill with parsed results.
 * @max_results: Capacity of results array.
 * @n_results:   Output: number of results actually read.
 *
 * Returns 0 on success, -1 on error.
 */
int sam3_bench_read_json(const char *path,
			 struct sam3_bench_env *env,
			 struct sam3_bench_result *results,
			 int max_results, int *n_results);

/*
 * sam3_bench_print_results - Print a results table to stderr.
 *
 * @results:   Array of benchmark results.
 * @n_results: Number of results.
 *
 * Columns: Name, Mean(ms), Min(ms), Max(ms), GFLOPS, MB/s.
 */
void sam3_bench_print_results(const struct sam3_bench_result *results,
			      int n_results);

#endif /* SAM3_BENCH_JSON_H */
