/*
 * src/bench/bench_pipeline.h - Pipeline benchmark suite
 *
 * Declares the entry point for end-to-end pipeline benchmarks that
 * exercise the full SAM3 inference path: image encoding, point prompts,
 * box prompts, and text prompts. Requires a loaded model context.
 * Results are written into a caller-provided array for JSON output
 * and comparison.
 *
 * Key types:  sam3_bench_config, sam3_bench_result, sam3_ctx
 * Depends on: bench/bench.h, sam3/sam3.h
 * Used by:    cli_bench.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_BENCH_PIPELINE_H
#define SAM3_BENCH_PIPELINE_H

#include "bench/bench.h"
#include "sam3/sam3.h"

/*
 * sam3_bench_run_pipeline - Run all pipeline benchmarks.
 *
 * @cfg:         Benchmark configuration (iterations, filter, etc.)
 * @ctx:         Initialized sam3 context with a loaded model.
 * @results:     Array to fill with per-case results.
 * @max_results: Capacity of the results array.
 *
 * Benchmarks image encoding, full segmentation with point/box/text
 * prompts using a synthetic gradient test image. The model must be
 * loaded before calling this function.
 *
 * Returns the number of results written, or -1 on error.
 */
int sam3_bench_run_pipeline(const struct sam3_bench_config *cfg,
			    sam3_ctx *ctx,
			    struct sam3_bench_result *results,
			    int max_results);

#endif /* SAM3_BENCH_PIPELINE_H */
