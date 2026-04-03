/*
 * src/sam3.c - Top-level SAM3 public API implementation
 *
 * Implements the functions declared in sam3/sam3.h: context lifecycle,
 * model loading, image encoding, segmentation, and version query.
 * Delegates inference to sam3_processor which manages backend, arenas,
 * and the full image model pipeline.
 *
 * Key types:  sam3_ctx
 * Depends on: sam3/sam3.h, model/sam3_processor.h
 * Used by:    tools/sam3_main.c, user applications
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>

#include "sam3/sam3.h"
#include "core/weight.h"
#include "util/image.h"
#include "model/sam3_processor.h"
#ifdef SAM3_HAS_PROFILE
#include "util/profile.h"
#endif

/* Internal context definition. */
struct sam3_ctx {
	struct sam3_model_config config;
	struct sam3_weight_file weights;
	int loaded;
	struct sam3_processor proc;
	int proc_ready;
#ifdef SAM3_HAS_PROFILE
	struct sam3_profiler *profiler;
#endif
};

const char *sam3_version(void)
{
	return "0.1.0";
}

sam3_ctx *sam3_init(void)
{
	sam3_ctx *ctx = calloc(1, sizeof(*ctx));
	return ctx;
}

void sam3_free(sam3_ctx *ctx)
{
	if (!ctx)
		return;
	if (ctx->proc_ready)
		sam3_processor_free(&ctx->proc);
	if (ctx->loaded)
		sam3_weight_close(&ctx->weights);
#ifdef SAM3_HAS_PROFILE
	sam3_profiler_free(ctx->profiler);
#endif
	free(ctx);
}

enum sam3_error sam3_load_model(sam3_ctx *ctx, const char *path)
{
	if (!ctx || !path)
		return SAM3_EINVAL;
	if (ctx->loaded) {
		sam3_weight_close(&ctx->weights);
		ctx->loaded = 0;
	}

	enum sam3_error err = sam3_weight_open(&ctx->weights, path);
	if (err)
		return err;

	/* Copy model config from weight file header */
	const struct sam3_weight_header *h = ctx->weights.header;
	ctx->config.image_size       = h->image_size;
	ctx->config.encoder_dim      = h->encoder_dim;
	ctx->config.decoder_dim      = h->decoder_dim;
	ctx->config.n_encoder_layers = h->n_encoder_layers;
	ctx->config.n_decoder_layers = h->n_decoder_layers;

	ctx->loaded = 1;

	/* Initialize processor and load model weights */
	err = sam3_processor_init(&ctx->proc);
	if (err != SAM3_OK)
		return err;

	err = sam3_processor_load(&ctx->proc, path, NULL);
	if (err != SAM3_OK) {
		sam3_processor_free(&ctx->proc);
		return err;
	}
	ctx->proc_ready = 1;

	return SAM3_OK;
}

enum sam3_error sam3_set_image(sam3_ctx *ctx, const uint8_t *pixels,
			       int width, int height)
{
	if (!ctx || !pixels || width <= 0 || height <= 0)
		return SAM3_EINVAL;
	if (!ctx->proc_ready)
		return SAM3_EINVAL;

	return sam3_processor_set_image(&ctx->proc, pixels, width, height);
}

enum sam3_error sam3_set_image_file(sam3_ctx *ctx, const char *path)
{
	if (!ctx || !path)
		return SAM3_EINVAL;

	struct sam3_image raw = {0};
	enum sam3_error err = sam3_image_load(path, &raw);
	if (err)
		return err;

	int target = ctx->config.image_size;
	if (target <= 0)
		target = 1024;

	struct sam3_image letterboxed = {0};
	err = sam3_image_letterbox(&raw, &letterboxed, target);
	sam3_image_free(&raw);
	if (err)
		return err;

	err = sam3_set_image(ctx, letterboxed.pixels,
			     letterboxed.width, letterboxed.height);
	sam3_image_free(&letterboxed);
	return err;
}

enum sam3_error sam3_segment(sam3_ctx *ctx, const struct sam3_prompt *prompts,
			     int n_prompts, struct sam3_result *result)
{
	if (!ctx || !result)
		return SAM3_EINVAL;
	if (!ctx->proc_ready)
		return SAM3_EINVAL;

	return sam3_processor_segment(&ctx->proc, prompts, n_prompts, result);
}

void sam3_result_free(struct sam3_result *result)
{
	if (!result)
		return;
	free(result->masks);
	free(result->iou_scores);
	result->masks      = NULL;
	result->iou_scores = NULL;
	result->n_masks    = 0;
}

enum sam3_error sam3_profile_enable(sam3_ctx *ctx)
{
#ifdef SAM3_HAS_PROFILE
	if (!ctx)
		return SAM3_EINVAL;
	if (!ctx->profiler) {
		ctx->profiler = sam3_profiler_create();
		if (!ctx->profiler)
			return SAM3_ENOMEM;
	}
	sam3_profiler_enable(ctx->profiler);
	return SAM3_OK;
#else
	(void)ctx;
	return SAM3_OK;
#endif
}

void sam3_profile_disable(sam3_ctx *ctx)
{
#ifdef SAM3_HAS_PROFILE
	if (ctx && ctx->profiler)
		sam3_profiler_disable(ctx->profiler);
#else
	(void)ctx;
#endif
}

void sam3_profile_report(sam3_ctx *ctx)
{
#ifdef SAM3_HAS_PROFILE
	if (ctx && ctx->profiler)
		sam3_profiler_report(ctx->profiler);
#else
	(void)ctx;
#endif
}

void sam3_profile_reset(sam3_ctx *ctx)
{
#ifdef SAM3_HAS_PROFILE
	if (ctx && ctx->profiler)
		sam3_profiler_reset(ctx->profiler);
#else
	(void)ctx;
#endif
}
