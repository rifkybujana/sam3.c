/*
 * src/bench/bench_compare.c - Baseline comparison for benchmark results
 *
 * Implements regression detection by comparing current benchmark timings
 * against a stored baseline. Cases are matched by name. In percentage mode,
 * a regression is flagged when the current mean exceeds the baseline mean
 * by more than the configured threshold. In statistical mode, the threshold
 * is baseline mean + 2 * baseline stddev. Prints a formatted comparison
 * table to stderr.
 *
 * Key types:  sam3_bench_result
 * Depends on: bench/bench_compare.h, bench/bench_json.h, util/log.h
 * Used by:    cli_bench.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>

#include "bench/bench_compare.h"
#include "bench/bench_json.h"
#include "util/log.h"

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
			       double threshold_pct,
			       bool statistical)
{
	if (!baseline || !current || n_baseline <= 0 || n_current <= 0)
		return 0;

	int regressions = 0;

	fprintf(stderr,
		"%-40s %10s %10s %8s  %s\n",
		"Name", "Old(ms)", "New(ms)", "Delta%", "Status");
	fprintf(stderr,
		"%-40s %10s %10s %8s  %s\n",
		"----------------------------------------",
		"----------", "----------", "--------",
		"-----------");

	for (int i = 0; i < n_current; i++) {
		const struct sam3_bench_result *cur = &current[i];
		const struct sam3_bench_result *base =
			find_by_name(baseline, n_baseline, cur->name);

		if (!base) {
			fprintf(stderr, "%-40s %10s %10.3f %8s  %s\n",
				cur->name, "N/A", cur->mean_ms, "N/A",
				"NEW");
			continue;
		}

		double delta_pct = 0.0;
		if (base->mean_ms > 0.0)
			delta_pct = ((cur->mean_ms - base->mean_ms)
				     / base->mean_ms) * 100.0;

		bool regressed = false;

		if (statistical) {
			/* Regression if current exceeds mean + 2*stddev. */
			double limit = base->mean_ms + 2.0 * base->stddev_ms;
			regressed = cur->mean_ms > limit;
		} else {
			/* Regression if delta exceeds threshold. */
			regressed = delta_pct > threshold_pct;
		}

		const char *status = regressed ? "REGRESSION" : "OK";
		if (regressed)
			regressions++;

		fprintf(stderr,
			"%-40s %10.3f %10.3f %+7.1f%%  %s\n",
			cur->name, base->mean_ms, cur->mean_ms,
			delta_pct, status);
	}

	if (regressions > 0) {
		fprintf(stderr, "\n%d regression(s) detected\n", regressions);
	} else {
		fprintf(stderr, "\nNo regressions detected\n");
	}

	return regressions;
}

int sam3_bench_compare_files(const char *baseline_path,
			     const struct sam3_bench_result *current,
			     int n_current,
			     double threshold_pct,
			     bool statistical)
{
	if (!baseline_path || !current) {
		sam3_log_error("bench_compare_files: invalid arguments");
		return -1;
	}

	struct sam3_bench_result baseline[SAM3_BENCH_MAX_RESULTS];
	int n_baseline = 0;

	int err = sam3_bench_read_json(baseline_path, NULL,
				       baseline, SAM3_BENCH_MAX_RESULTS,
				       &n_baseline);
	if (err != 0) {
		sam3_log_error("bench_compare_files: failed to read %s",
			       baseline_path);
		return -1;
	}

	return sam3_bench_compare_results(baseline, n_baseline,
					  current, n_current,
					  threshold_pct, statistical);
}
