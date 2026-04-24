/*
 * tests/bench_segment_batch.c - Per-batch-size segment latency.
 *
 * Measures wall-clock latency of sam3_segment_batch at batch sizes
 * 1, 2, 4, 8 with identical shape-compatible prompt sets, and
 * compares against N x serial sam3_segment calls. The batched path
 * is expected to be faster for B >= 2 because the DETR decoder and
 * seg head run as single-pass batched ops instead of B serial passes.
 *
 * Key types:  sam3_ctx, sam3_prompt_set
 * Depends on: sam3/sam3.h
 * Used by:    Manual benchmarking (gated by SAM3_BENCH=ON); not in CTest.
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "sam3/sam3.h"

#ifndef SAM3_SOURCE_DIR
#define SAM3_SOURCE_DIR "."
#endif
#define MODEL_PATH SAM3_SOURCE_DIR "/models/sam3.sam3"

static int model_available(void)
{
	return access(MODEL_PATH, F_OK) == 0;
}

static double time_ms(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

static void fill_deterministic(uint8_t *pix, int sz)
{
	for (int i = 0; i < sz * sz * 3; i++)
		pix[i] = (uint8_t)((i * 31 + 17) & 0xff);
}

/*
 * bench_batched - Run the batched path at batch size @B (identical
 * text prompts) @iters times and return average ms per full call.
 */
static double bench_batched(sam3_ctx *ctx, int B, int iters,
			    const char *text)
{
	struct sam3_prompt p = { .type = SAM3_PROMPT_TEXT, .text = text };
	struct sam3_prompt_set *sets = calloc((size_t)B, sizeof(*sets));
	struct sam3_result *results = calloc((size_t)B, sizeof(*results));
	for (int i = 0; i < B; i++) {
		sets[i].prompts = &p;
		sets[i].n_prompts = 1;
	}

	/* Warm-up */
	sam3_segment_batch(ctx, sets, B, results);
	for (int i = 0; i < B; i++)
		sam3_result_free(&results[i]);

	double t0 = time_ms();
	for (int it = 0; it < iters; it++) {
		sam3_segment_batch(ctx, sets, B, results);
		for (int i = 0; i < B; i++)
			sam3_result_free(&results[i]);
	}
	double t1 = time_ms();

	free(sets);
	free(results);
	return (t1 - t0) / iters;
}

/*
 * bench_serial - Run B serial single-shot sam3_segment calls,
 * return average ms total per outer iteration.
 */
static double bench_serial(sam3_ctx *ctx, int B, int iters, const char *text)
{
	struct sam3_prompt p = { .type = SAM3_PROMPT_TEXT, .text = text };

	/* Warm-up */
	struct sam3_result r = {0};
	sam3_segment(ctx, &p, 1, &r);
	sam3_result_free(&r);

	double t0 = time_ms();
	for (int it = 0; it < iters; it++) {
		for (int i = 0; i < B; i++) {
			struct sam3_result result = {0};
			sam3_segment(ctx, &p, 1, &result);
			sam3_result_free(&result);
		}
	}
	double t1 = time_ms();
	return (t1 - t0) / iters;
}

int main(void)
{
	if (!model_available()) {
		printf("skip: no model\n");
		return 0;
	}

	/* Suppress per-iteration log noise. */
	sam3_log_set_level(SAM3_LOG_WARN);

	sam3_ctx *ctx = sam3_init();
	if (!ctx)
		return 1;
	if (sam3_load_model(ctx, MODEL_PATH) != SAM3_OK) {
		sam3_free(ctx);
		return 1;
	}

	int sz = sam3_get_image_size(ctx);
	uint8_t *pix = malloc((size_t)sz * sz * 3);
	if (!pix) {
		sam3_free(ctx);
		return 1;
	}
	fill_deterministic(pix, sz);
	if (sam3_set_image(ctx, pix, sz, sz) != SAM3_OK) {
		free(pix);
		sam3_free(ctx);
		return 1;
	}

	const int iters = 5;
	const char *text = "cat";

	printf("B | serial (ms) | batched (ms) | speedup\n");
	printf("--+-------------+--------------+--------\n");
	for (int B = 1; B <= 8; B *= 2) {
		double ser = bench_serial(ctx, B, iters, text);
		double bat = bench_batched(ctx, B, iters, text);
		printf("%d | %11.2f | %12.2f | %.2fx\n",
		       B, ser, bat, ser / bat);
	}

	free(pix);
	sam3_free(ctx);
	return 0;
}
