/*
 * src/bench/bench_pipeline.c - Pipeline benchmark suite implementation
 *
 * Benchmarks end-to-end SAM3 inference pipelines: image encoding alone,
 * and full segmentation with point, box, and text prompts. Uses a
 * synthetic gradient test image (RGB) sized to the model's input. Each
 * case has its own context struct and callback for the bench harness.
 *
 * Key types:  sam3_bench_config, sam3_bench_result, sam3_ctx
 * Depends on: bench/bench.h, sam3/sam3.h, util/log.h
 * Used by:    cli_bench.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "bench/bench_pipeline.h"
#include "util/log.h"

/* --- Context structs for each benchmark case --- */

struct image_encode_ctx {
	sam3_ctx      *ctx;
	const uint8_t *pixels;
	int            width;
	int            height;
};

struct segment_ctx {
	sam3_ctx          *ctx;
	struct sam3_prompt prompt;
};

/* --- Benchmark callbacks ─ --- */

static void image_encode_fn(void *arg)
{
	struct image_encode_ctx *ec = arg;
	sam3_set_image(ec->ctx, ec->pixels, ec->width, ec->height);
}

static void segment_fn(void *arg)
{
	struct segment_ctx *sc = arg;
	struct sam3_result result;
	memset(&result, 0, sizeof(result));

	sam3_segment(sc->ctx, &sc->prompt, 1, &result);
	sam3_result_free(&result);
}

/* --- Synthetic test image generation ─────── --- */

/*
 * generate_gradient_image - Create an RGB gradient test pattern.
 *
 * @pixels: Output buffer (must hold size * size * 3 bytes).
 * @size:   Image width and height.
 *
 * R channel = 255 * x / size, G channel = 255 * y / size, B = 128.
 */
static void generate_gradient_image(uint8_t *pixels, int size)
{
	for (int y = 0; y < size; y++) {
		for (int x = 0; x < size; x++) {
			int idx = (y * size + x) * 3;
			pixels[idx + 0] = (uint8_t)(255 * x / size);
			pixels[idx + 1] = (uint8_t)(255 * y / size);
			pixels[idx + 2] = 128;
		}
	}
}

/* --- Public entry point ── --- */

int sam3_bench_run_pipeline(const struct sam3_bench_config *cfg,
			    sam3_ctx *ctx,
			    struct sam3_bench_result *results,
			    int max_results)
{
	if (!cfg || !ctx || !results || max_results <= 0) {
		sam3_log_error("bench_run_pipeline: invalid arguments");
		return -1;
	}

	int img_size = sam3_get_image_size(ctx);
	if (img_size <= 0) {
		sam3_log_error("bench_run_pipeline: no model loaded "
			       "(image size = %d)", img_size);
		return -1;
	}

	/* Allocate and fill synthetic test image. */
	size_t pixel_bytes = (size_t)img_size * img_size * 3;
	uint8_t *pixels = malloc(pixel_bytes);
	if (!pixels) {
		sam3_log_error("bench_run_pipeline: failed to allocate "
			       "test image (%zu bytes)", pixel_bytes);
		return -1;
	}
	generate_gradient_image(pixels, img_size);

	int count = 0;

	/* --- image_encode --- */
	if (count < max_results &&
	    sam3_bench_filter_match("image_encode", cfg->filter)) {
		struct image_encode_ctx ec;
		ec.ctx    = ctx;
		ec.pixels = pixels;
		ec.width  = img_size;
		ec.height = img_size;

		if (sam3_bench_run(cfg, "image_encode", "pipeline",
				   image_encode_fn, &ec,
				   0, 0,
				   &results[count]) != 0) {
			sam3_log_error("pipeline bench: image_encode failed");
			goto error;
		}
		count++;
	}

	/*
	 * Skip the set_image setup when no segment case will run, so a
	 * filter like "video_*" does not pay the image-encoder cost (or
	 * trigger unrelated encoder errors on a model that fails to
	 * encode).
	 */
	bool any_segment = sam3_bench_filter_match("full_pipeline_point",
						   cfg->filter) ||
			   sam3_bench_filter_match("full_pipeline_box",
						   cfg->filter) ||
			   sam3_bench_filter_match("full_pipeline_text",
						   cfg->filter);

	if (any_segment) {
		/*
		 * Set image once for segment benchmarks. All subsequent segment
		 * calls reuse the encoded image features.
		 */
		if (sam3_set_image(ctx, pixels, img_size, img_size) != SAM3_OK) {
			sam3_log_error("pipeline bench: sam3_set_image failed");
			goto error;
		}

		/* Set prompt space to match image dimensions. */
		sam3_set_prompt_space(ctx, img_size, img_size);
	}

	/* --- full_pipeline_point ─────────── --- */
	if (count < max_results &&
	    sam3_bench_filter_match("full_pipeline_point", cfg->filter)) {
		struct segment_ctx sc;
		sc.ctx = ctx;
		memset(&sc.prompt, 0, sizeof(sc.prompt));
		sc.prompt.type = SAM3_PROMPT_POINT;
		sc.prompt.point.x = (float)img_size / 2.0f;
		sc.prompt.point.y = (float)img_size / 2.0f;
		sc.prompt.point.label = 1;

		if (sam3_bench_run(cfg, "full_pipeline_point", "pipeline",
				   segment_fn, &sc,
				   0, 0,
				   &results[count]) != 0) {
			sam3_log_error("pipeline bench: point prompt failed");
			goto error;
		}
		count++;
	}

	/* --- full_pipeline_box ───────────── --- */
	if (count < max_results &&
	    sam3_bench_filter_match("full_pipeline_box", cfg->filter)) {
		struct segment_ctx sc;
		sc.ctx = ctx;
		memset(&sc.prompt, 0, sizeof(sc.prompt));
		sc.prompt.type = SAM3_PROMPT_BOX;
		sc.prompt.box.x1 = (float)img_size * 0.25f;
		sc.prompt.box.y1 = (float)img_size * 0.25f;
		sc.prompt.box.x2 = (float)img_size * 0.75f;
		sc.prompt.box.y2 = (float)img_size * 0.75f;

		if (sam3_bench_run(cfg, "full_pipeline_box", "pipeline",
				   segment_fn, &sc,
				   0, 0,
				   &results[count]) != 0) {
			sam3_log_error("pipeline bench: box prompt failed");
			goto error;
		}
		count++;
	}

	/* --- full_pipeline_text ──────────── --- */
	if (count < max_results &&
	    sam3_bench_filter_match("full_pipeline_text", cfg->filter)) {
		struct segment_ctx sc;
		sc.ctx = ctx;
		memset(&sc.prompt, 0, sizeof(sc.prompt));
		sc.prompt.type = SAM3_PROMPT_TEXT;
		sc.prompt.text = "cat";

		if (sam3_bench_run(cfg, "full_pipeline_text", "pipeline",
				   segment_fn, &sc,
				   0, 0,
				   &results[count]) != 0) {
			sam3_log_error("pipeline bench: text prompt failed");
			goto error;
		}
		count++;
	}

	sam3_log_info("pipeline benchmarks: %d cases completed", count);
	free(pixels);
	return count;

error:
	free(pixels);
	return -1;
}
