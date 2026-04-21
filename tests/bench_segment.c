/*
 * tests/bench_segment.c - End-to-end segmentation benchmark
 *
 * Loads a .sam3 model once, then runs sam3_set_image + sam3_set_text +
 * sam3_segment many times in a single process. Reports mean / median /
 * stdev / min / max for each stage (image_encode, text_encode, segment)
 * after warmup iterations to amortize first-run cost (Metal shader
 * compilation, lazy alloc, page faults). Designed to compare MobileCLIP
 * variants against each other and against standard CLIP without the
 * noise of fresh-process benchmarking via the CLI.
 *
 * Key types:  sam3_ctx (opaque)
 * Depends on: sam3/sam3.h, sam3/sam3_types.h
 * Used by:    manual benchmarking; not in CTest.
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <getopt.h>

#include "sam3/sam3.h"
#include "sam3/sam3_types.h"

/* --- Defaults --- */

#define DEFAULT_WARMUP   2
#define DEFAULT_ITERS   10
#define MAX_ITERS      256

/* --- Timing --- */

static double now_ms(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* --- Stats --- */

struct stats {
	double mean;
	double median;
	double stdev;
	double min;
	double max;
};

static int cmp_double(const void *a, const void *b)
{
	double x = *(const double *)a;
	double y = *(const double *)b;
	return (x > y) - (x < y);
}

static struct stats compute_stats(double *vals, int n)
{
	struct stats s = {0};
	if (n <= 0)
		return s;

	double *sorted = malloc((size_t)n * sizeof(double));
	memcpy(sorted, vals, (size_t)n * sizeof(double));
	qsort(sorted, (size_t)n, sizeof(double), cmp_double);

	s.min = sorted[0];
	s.max = sorted[n - 1];
	s.median = (n % 2 == 0)
		? 0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
		: sorted[n / 2];

	double sum = 0.0;
	for (int i = 0; i < n; i++)
		sum += vals[i];
	s.mean = sum / n;

	double var = 0.0;
	for (int i = 0; i < n; i++) {
		double d = vals[i] - s.mean;
		var += d * d;
	}
	s.stdev = (n > 1) ? sqrt(var / (n - 1)) : 0.0;

	free(sorted);
	return s;
}

static void print_stage(const char *name, double *vals, int n)
{
	struct stats s = compute_stats(vals, n);
	printf("  %-18s  mean=%7.1f  median=%7.1f  "
	       "stdev=%6.1f  min=%7.1f  max=%7.1f  ms (n=%d)\n",
	       name, s.mean, s.median, s.stdev, s.min, s.max, n);
}

/* --- Usage --- */

static void usage(const char *prog)
{
	printf("Usage: %s -m <model.sam3> -i <image> -t <text> "
	       "[--iters N] [--warmup N]\n\n", prog);
	printf("Loads the model once, then runs end-to-end segmentation\n"
	       "N iterations, reporting per-stage timing statistics.\n\n");
	printf("Options:\n");
	printf("  -m <path>      Model .sam3 file (required)\n");
	printf("  -i <path>      Input image (required)\n");
	printf("  -t <text>      Text prompt (required)\n");
	printf("  --iters N      Timed iterations (default: %d)\n",
	       DEFAULT_ITERS);
	printf("  --warmup N     Warmup iterations (default: %d)\n",
	       DEFAULT_WARMUP);
	printf("  -h, --help     Show this help\n");
}

/* --- Main --- */

int main(int argc, char **argv)
{
	const char *model_path = NULL;
	const char *image_path = NULL;
	const char *text       = NULL;
	int warmup = DEFAULT_WARMUP;
	int iters  = DEFAULT_ITERS;

	static struct option long_opts[] = {
		{"iters",  required_argument, 0, 'I'},
		{"warmup", required_argument, 0, 'W'},
		{"help",   no_argument,       0, 'h'},
		{0, 0, 0, 0}
	};

	int c;
	while ((c = getopt_long(argc, argv, "m:i:t:h", long_opts, NULL)) != -1) {
		switch (c) {
		case 'm': model_path = optarg; break;
		case 'i': image_path = optarg; break;
		case 't': text       = optarg; break;
		case 'I': iters  = atoi(optarg); break;
		case 'W': warmup = atoi(optarg); break;
		case 'h': usage(argv[0]); return 0;
		default:  usage(argv[0]); return 1;
		}
	}

	if (!model_path || !image_path || !text) {
		fprintf(stderr, "error: -m, -i, and -t are required\n");
		usage(argv[0]);
		return 1;
	}
	if (iters <= 0 || iters > MAX_ITERS) {
		fprintf(stderr, "error: --iters must be in [1, %d]\n",
			MAX_ITERS);
		return 1;
	}
	if (warmup < 0) warmup = 0;

	/* Suppress per-iteration log noise. */
	sam3_log_set_level(SAM3_LOG_WARN);

	printf("bench_segment\n");
	printf("  model:   %s\n", model_path);
	printf("  image:   %s\n", image_path);
	printf("  text:    \"%s\"\n", text);
	printf("  warmup:  %d iter(s)\n", warmup);
	printf("  timed:   %d iter(s)\n\n", iters);

	/* --- One-shot init --- */
	double t_init = now_ms();
	sam3_ctx *ctx = sam3_init();
	if (!ctx) {
		fprintf(stderr, "sam3_init failed\n");
		return 2;
	}
	enum sam3_error err = sam3_load_model(ctx, model_path);
	if (err != SAM3_OK) {
		fprintf(stderr, "sam3_load_model failed: %s\n",
			sam3_error_str(err));
		sam3_free(ctx);
		return 3;
	}
	double init_ms = now_ms() - t_init;
	printf("  init+load:        %7.1f ms (one-shot)\n\n", init_ms);

	/* --- Warmup --- */
	for (int w = 0; w < warmup; w++) {
		err = sam3_set_image_file(ctx, image_path);
		if (err != SAM3_OK) {
			fprintf(stderr, "warmup set_image failed\n");
			goto fail;
		}
		err = sam3_set_text(ctx, text);
		if (err != SAM3_OK) {
			fprintf(stderr, "warmup set_text failed\n");
			goto fail;
		}
		struct sam3_prompt p = { .type = SAM3_PROMPT_TEXT,
					 .text = text };
		struct sam3_result r = {0};
		err = sam3_segment(ctx, &p, 1, &r);
		if (err != SAM3_OK) {
			fprintf(stderr, "warmup segment failed\n");
			goto fail;
		}
		sam3_result_free(&r);
	}
	if (warmup > 0)
		printf("  (warmup done)\n\n");

	/* --- Timed iterations --- */
	double set_image_ms[MAX_ITERS];
	double set_text_ms [MAX_ITERS];
	double segment_ms  [MAX_ITERS];
	double total_ms    [MAX_ITERS];
	int n_masks_last   = 0;
	float best_iou_last = 0.0f;

	for (int i = 0; i < iters; i++) {
		double t0 = now_ms();
		err = sam3_set_image_file(ctx, image_path);
		double t1 = now_ms();
		if (err != SAM3_OK) {
			fprintf(stderr, "iter %d set_image failed\n", i);
			goto fail;
		}
		err = sam3_set_text(ctx, text);
		double t2 = now_ms();
		if (err != SAM3_OK) {
			fprintf(stderr, "iter %d set_text failed\n", i);
			goto fail;
		}
		struct sam3_prompt p = { .type = SAM3_PROMPT_TEXT,
					 .text = text };
		struct sam3_result r = {0};
		err = sam3_segment(ctx, &p, 1, &r);
		double t3 = now_ms();
		if (err != SAM3_OK) {
			fprintf(stderr, "iter %d segment failed\n", i);
			goto fail;
		}

		set_image_ms[i] = t1 - t0;
		set_text_ms [i] = t2 - t1;
		segment_ms  [i] = t3 - t2;
		total_ms    [i] = t3 - t0;

		n_masks_last  = r.n_masks;
		best_iou_last = (r.n_masks > 0 && r.iou_scores)
				? r.iou_scores[0] : 0.0f;

		sam3_result_free(&r);
	}

	/* --- Report --- */
	printf("Per-stage timing (n=%d):\n", iters);
	print_stage("set_image_file:",  set_image_ms, iters);
	print_stage("set_text:",        set_text_ms,  iters);
	print_stage("segment:",         segment_ms,   iters);
	print_stage("TOTAL/iter:",      total_ms,     iters);

	struct stats tot = compute_stats(total_ms, iters);
	printf("\nThroughput:    %.2f iter/s  (%.0f ms/iter median)\n",
	       1000.0 / tot.median, tot.median);
	printf("Last result:   %d mask(s), best IoU = %.4f\n",
	       n_masks_last, best_iou_last);

	sam3_free(ctx);
	return 0;

fail:
	sam3_free(ctx);
	return 4;
}
