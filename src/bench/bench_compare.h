/*
 * src/bench/bench_compare.h - Baseline comparison for benchmark results
 *
 * Compares current benchmark results against a stored baseline to detect
 * performance regressions. Supports both simple percentage-threshold mode
 * and statistical mode (mean + 2*stddev). Prints a comparison table to
 * stderr showing deltas and OK/REGRESSION status for each matched case.
 *
 * Key types:  sam3_bench_result
 * Depends on: bench/bench.h
 * Used by:    cli_bench.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_BENCH_COMPARE_H
#define SAM3_BENCH_COMPARE_H

#include "bench/bench.h"

/*
 * sam3_bench_compare_results - Compare current results against a baseline.
 *
 * @baseline:      Array of baseline results.
 * @n_baseline:    Number of baseline results.
 * @current:       Array of current results.
 * @n_current:     Number of current results.
 * @threshold_pct: Regression threshold in percent (e.g. 5.0 for 5%).
 * @statistical:   If true, use mean + 2*stddev instead of percentage.
 *
 * Matches cases by name. Prints a comparison table to stderr.
 * Returns the number of regressions detected (0 = all OK).
 */
int sam3_bench_compare_results(const struct sam3_bench_result *baseline,
			       int n_baseline,
			       const struct sam3_bench_result *current,
			       int n_current,
			       double threshold_pct,
			       bool statistical);

/*
 * sam3_bench_compare_files - Load baseline from JSON and compare.
 *
 * @baseline_path: Path to baseline JSON file.
 * @current:       Array of current results.
 * @n_current:     Number of current results.
 * @threshold_pct: Regression threshold in percent.
 * @statistical:   If true, use statistical mode.
 *
 * Returns the number of regressions, or -1 on file/parse error.
 */
int sam3_bench_compare_files(const char *baseline_path,
			     const struct sam3_bench_result *current,
			     int n_current,
			     double threshold_pct,
			     bool statistical);

#endif /* SAM3_BENCH_COMPARE_H */
