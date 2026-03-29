/*
 * src/sam3.c - Top-level SAM3 public API implementation
 *
 * Implements the functions declared in sam3/sam3.h: context lifecycle,
 * model loading, image encoding, segmentation, and version query.
 * Currently provides stubs that return error codes; real logic will
 * be wired up as each subsystem is completed.
 *
 * Key types:  sam3_ctx
 * Depends on: sam3/sam3.h
 * Used by:    tools/sam3_main.c, user applications
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>

#include "sam3/sam3.h"

/* Internal context definition. */
struct sam3_ctx {
	struct sam3_model_config config;
	int loaded;
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
	free(ctx);
}

enum sam3_error sam3_load_model(sam3_ctx *ctx, const char *path)
{
	(void)ctx;
	(void)path;
	return SAM3_EMODEL; /* Not yet implemented */
}

enum sam3_error sam3_set_image(sam3_ctx *ctx, const uint8_t *pixels,
			       int width, int height)
{
	(void)ctx;
	(void)pixels;
	(void)width;
	(void)height;
	return SAM3_EINVAL; /* Not yet implemented */
}

enum sam3_error sam3_segment(sam3_ctx *ctx, const struct sam3_prompt *prompts,
			     int n_prompts, struct sam3_result *result)
{
	(void)ctx;
	(void)prompts;
	(void)n_prompts;
	(void)result;
	return SAM3_EINVAL; /* Not yet implemented */
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
