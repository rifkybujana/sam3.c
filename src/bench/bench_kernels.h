/*
 * src/bench/bench_kernels.h - Kernel microbenchmark suite
 *
 * Declares the entry point for running individual compute kernel
 * benchmarks (matmul, elementwise, softmax, layernorm, transpose).
 * Each kernel is benchmarked across a matrix of sizes and dtypes
 * using the bench harness. Results are written into a caller-provided
 * array for JSON serialization and comparison.
 *
 * Key types:  sam3_bench_config, sam3_bench_result, sam3_backend
 * Depends on: bench/bench.h, backend/backend.h
 * Used by:    cli_bench.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_BENCH_KERNELS_H
#define SAM3_BENCH_KERNELS_H

#include "bench/bench.h"
#include "backend/backend.h"

/*
 * sam3_bench_run_kernels - Run all kernel microbenchmarks.
 *
 * @cfg:         Benchmark configuration (iterations, filter, etc.)
 * @be:          Initialized backend to benchmark against.
 * @results:     Array to fill with per-case results.
 * @max_results: Capacity of the results array.
 *
 * Benchmarks matmul, add, mul, gelu, silu, softmax, layernorm, and
 * transpose kernels across multiple sizes and dtypes. Each case builds
 * a single-op compute graph and evaluates it through the backend vtable.
 *
 * Returns the number of results written, or -1 on error.
 */
int sam3_bench_run_kernels(const struct sam3_bench_config *cfg,
			   struct sam3_backend *be,
			   struct sam3_bench_result *results,
			   int max_results);

#endif /* SAM3_BENCH_KERNELS_H */
