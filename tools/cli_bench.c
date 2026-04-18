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
 *             bench/bench_kernels.h, bench/bench_pipeline.h,
 *             bench/bench_video.h
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
#include "bench/bench_video.h"
#include "backend/backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_bench_usage(const char *prog)
{
	fprintf(stderr, "Usage: %s bench [SUITE] [OPTIONS]\n\n", prog);
	fprintf(stderr, "Suites:\n");
	fprintf(stderr, "  kernels    Kernel microbenchmarks only\n");
	fprintf(stderr, "  pipeline   Pipeline benchmarks "
		"(requires --model)\n");
	fprintf(stderr, "  all        Both suites (default)\n\n");
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "  --model PATH      "
		".sam3 model weights\n");
	fprintf(stderr, "  --backend cpu|metal  "
		"Backend (default: auto)\n");
	fprintf(stderr, "  --output PATH     "
		"Write JSON results\n");
	fprintf(stderr, "  --compare PATH    "
		"Compare against baseline\n");
	fprintf(stderr, "  --threshold PCT   "
		"Regression threshold (default: 5.0%%)\n");
	fprintf(stderr, "  --statistical     "
		"Statistical comparison\n");
	fprintf(stderr, "  --warmup N        "
		"Warmup iterations (default: 5)\n");
	fprintf(stderr, "  --iters N         "
		"Timed iterations (default: 50)\n");
	fprintf(stderr, "  --filter PATTERN  "
		"Filter benchmarks by name\n");
	fprintf(stderr, "  -v                "
		"Verbose output\n");
}

int cli_bench(int argc, char **argv)
{
	struct sam3_bench_config cfg;
	sam3_bench_config_defaults(&cfg);

	const char *suite = "all";
	const char *model_path = NULL;
	const char *backend_str = NULL;

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--help") == 0 ||
		    strcmp(argv[i], "-h") == 0) {
			print_bench_usage("sam3");
			return SAM3_EXIT_OK;
		}
		if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
			model_path = argv[++i];
		} else if (strcmp(argv[i], "--backend") == 0 &&
			   i + 1 < argc) {
			backend_str = argv[++i];
		} else if (strcmp(argv[i], "--output") == 0 &&
			   i + 1 < argc) {
			cfg.output_path = argv[++i];
		} else if (strcmp(argv[i], "--compare") == 0 &&
			   i + 1 < argc) {
			cfg.baseline_path = argv[++i];
		} else if (strcmp(argv[i], "--threshold") == 0 &&
			   i + 1 < argc) {
			cfg.threshold_pct = atof(argv[++i]);
		} else if (strcmp(argv[i], "--statistical") == 0) {
			cfg.statistical = true;
		} else if (strcmp(argv[i], "--warmup") == 0 &&
			   i + 1 < argc) {
			cfg.warmup_iters = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--iters") == 0 &&
			   i + 1 < argc) {
			cfg.timed_iters = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--filter") == 0 &&
			   i + 1 < argc) {
			cfg.filter = argv[++i];
		} else if (strcmp(argv[i], "-v") == 0) {
			cfg.verbose = true;
		} else if (argv[i][0] != '-') {
			suite = argv[i];
		}
	}

	/* Resolve backend */
	if (backend_str) {
		if (strcmp(backend_str, "metal") == 0) {
			cfg.backend = SAM3_BACKEND_METAL;
		} else if (strcmp(backend_str, "cpu") == 0) {
			cfg.backend = SAM3_BACKEND_CPU;
		} else {
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

	if (cfg.verbose)
		sam3_log_set_level(SAM3_LOG_DEBUG);

	bool run_kernels = strcmp(suite, "kernels") == 0 ||
			   strcmp(suite, "all") == 0;
	bool run_pipeline = strcmp(suite, "pipeline") == 0 ||
			    strcmp(suite, "all") == 0;

	if (run_pipeline && !model_path) {
		fprintf(stderr,
			"error: pipeline benchmarks require --model\n");
		return SAM3_EXIT_USAGE;
	}

	struct sam3_bench_result results[SAM3_BENCH_MAX_RESULTS];
	int n_results = 0;

	/* Kernel benchmarks */
	if (run_kernels) {
		fprintf(stderr, "Running kernel benchmarks...\n");
		struct sam3_backend *be = sam3_backend_init(cfg.backend);
		if (!be) {
			fprintf(stderr,
				"error: failed to init backend\n");
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
			fprintf(stderr,
				"error: failed to init context\n");
			return SAM3_EXIT_INTERNAL;
		}

		enum sam3_error err = sam3_load_model(ctx, model_path);
		if (err != SAM3_OK) {
			fprintf(stderr,
				"error: failed to load model: %d\n",
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

		n = sam3_bench_run_video_frame(&cfg, ctx,
					       results + n_results,
					       SAM3_BENCH_MAX_RESULTS -
					       n_results);
		if (n > 0)
			n_results += n;

		n = sam3_bench_run_video_end_to_end(&cfg, ctx,
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
		fprintf(stderr, "Results written to %s\n",
			cfg.output_path);
	}

	/* Compare against baseline if requested */
	if (cfg.baseline_path) {
		int regressions = sam3_bench_compare_files(
			cfg.baseline_path, results, n_results,
			cfg.threshold_pct, cfg.statistical);
		if (regressions < 0)
			return SAM3_EXIT_IO;
		if (regressions > 0)
			return 1;
	}

	return SAM3_EXIT_OK;
}
