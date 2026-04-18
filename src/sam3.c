/*
 * src/sam3.c - Top-level SAM3 public API implementation
 *
 * Implements the functions declared in sam3/sam3.h: context lifecycle,
 * model loading, image encoding, segmentation, and version query.
 * Delegates inference to sam3_processor which manages backend, arenas,
 * and the full image model pipeline.
 *
 * Key types:  sam3_ctx
 * Depends on: sam3/sam3.h, model/sam3_internal.h
 * Used by:    tools/sam3_main.c, user applications
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/mman.h>

#include "sam3/sam3.h"
#include "sam3/internal/tensor_dump.h"
#include "core/weight.h"
#include "core/tensor.h"
#include "core/alloc.h"
#include "util/image.h"
#include "util/log.h"
#include "model/sam3_internal.h"

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

/*
 * try_bpe_path - Check if a BPE vocab file exists at the given path.
 *
 * Returns 1 if the file exists, 0 otherwise.
 */
static int try_bpe_path(const char *path)
{
	FILE *f = fopen(path, "rb");
	if (f) {
		fclose(f);
		return 1;
	}
	return 0;
}

/*
 * find_bpe_next_to_model - Look for BPE vocab file in the model directory.
 *
 * Searches for "bpe_simple_vocab_16e6.txt.gz" in the same directory
 * as the model file. Writes the path to buf (at most bufsz bytes).
 * Returns 1 if found, 0 otherwise.
 */
static int find_bpe_next_to_model(const char *model_path,
				   char *buf, size_t bufsz)
{
	static const char bpe_name[] = "bpe_simple_vocab_16e6.txt.gz";

	const char *last_sep = strrchr(model_path, '/');
	size_t dir_len;

	if (last_sep) {
		dir_len = (size_t)(last_sep - model_path + 1);
	} else {
		/* No directory separator — look in current directory */
		dir_len = 0;
	}

	if (dir_len + sizeof(bpe_name) > bufsz)
		return 0;

	if (dir_len > 0)
		memcpy(buf, model_path, dir_len);
	memcpy(buf + dir_len, bpe_name, sizeof(bpe_name));

	return try_bpe_path(buf);
}

enum sam3_error sam3_load_model(sam3_ctx *ctx, const char *path)
{
	if (!ctx || !path)
		return SAM3_EINVAL;
	if (ctx->loaded) {
		sam3_weight_close(&ctx->weights);
		ctx->loaded = 0;
	}

#ifdef SAM3_HAS_PROFILE
	SAM3_PROF_BEGIN(ctx->profiler, "model_load");
#endif

	enum sam3_error err = sam3_weight_open(&ctx->weights, path);
	if (err) {
#ifdef SAM3_HAS_PROFILE
		SAM3_PROF_END(ctx->profiler, "model_load");
#endif
		return err;
	}

	/* Copy model config from weight file header */
	const struct sam3_weight_header *h = ctx->weights.header;
	ctx->config.image_size       = h->image_size;
	ctx->config.encoder_dim      = h->encoder_dim;
	ctx->config.decoder_dim      = h->decoder_dim;
	ctx->config.n_encoder_layers = h->n_encoder_layers;
	ctx->config.n_decoder_layers = h->n_decoder_layers;
	ctx->config.backbone_type    = (int)h->reserved[0];
	ctx->config.variant          = (int)h->reserved[1];
	ctx->config.n_fpn_scales     = (int)h->reserved[2];

	/* Legacy (pre-SAM3.1) .sam3 files have reserved[1..2] == 0.
	 * Treat that as SAM 3 with 4 FPN scales. */
	if (ctx->config.variant == 0 && ctx->config.n_fpn_scales == 0) {
		ctx->config.n_fpn_scales = 4;
	}
	if (ctx->config.variant != SAM3_VARIANT_SAM3 &&
	    ctx->config.variant != SAM3_VARIANT_SAM3_1) {
		sam3_log_error("unknown model variant %d in %s",
			       ctx->config.variant, path);
		sam3_weight_close(&ctx->weights);
		return SAM3_EMODEL;
	}
	if (ctx->config.n_fpn_scales < 1 || ctx->config.n_fpn_scales > 4) {
		sam3_log_error("invalid n_fpn_scales %d in %s (expect 3 or 4)",
			       ctx->config.n_fpn_scales, path);
		sam3_weight_close(&ctx->weights);
		return SAM3_EMODEL;
	}

	ctx->loaded = 1;

	/* Auto-discover BPE vocab file next to model */
	char bpe_path[1024];
	const char *vocab = NULL;
	if (find_bpe_next_to_model(path, bpe_path, sizeof(bpe_path))) {
		vocab = bpe_path;
		sam3_log_info("auto-discovered BPE vocab: %s", bpe_path);
	}

	/* Initialize processor and load model weights */
	err = sam3_processor_init(&ctx->proc, ctx->config.backbone_type);
	if (err != SAM3_OK) {
#ifdef SAM3_HAS_PROFILE
		SAM3_PROF_END(ctx->profiler, "model_load");
#endif
		return err;
	}

#ifdef SAM3_HAS_PROFILE
	ctx->proc.profiler = ctx->profiler;
#endif

	err = sam3_processor_load(&ctx->proc, &ctx->weights, vocab);
	if (err != SAM3_OK) {
		sam3_processor_free(&ctx->proc);
#ifdef SAM3_HAS_PROFILE
		SAM3_PROF_END(ctx->profiler, "model_load");
#endif
		return err;
	}
	ctx->proc_ready = 1;

	/*
	 * Override config.image_size with the backbone's actual
	 * img_size. The weight file header may carry a stale default
	 * (e.g. 1008) even for backbones that expect a different
	 * resolution (EfficientViT = 512). sam3_set_image_file reads
	 * config.image_size for the resize target, so this must match.
	 */
	ctx->config.image_size = sam3_processor_img_size(&ctx->proc);

	/* Ensure background prefetch is complete before returning */
	sam3_weight_prefetch_wait(&ctx->weights);

#ifdef SAM3_HAS_PROFILE
	SAM3_PROF_END(ctx->profiler, "model_load");
#endif

	return SAM3_OK;
}

enum sam3_error sam3_load_bpe(sam3_ctx *ctx, const char *path)
{
	if (!ctx || !path)
		return SAM3_EINVAL;
	if (!ctx->proc_ready)
		return SAM3_EINVAL;

	return sam3_tokenizer_load_bpe(
		&ctx->proc.model.backbone.tokenizer, path);
}

int sam3_get_image_size(const sam3_ctx *ctx)
{
	if (!ctx || !ctx->proc_ready)
		return 0;
	return ctx->config.image_size;
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
		target = 1008;

	/*
	 * Resize directly to target x target, matching the Python
	 * reference: v2.Resize(size=(resolution, resolution)).
	 * No letterboxing — squash to square like the reference.
	 */
	int orig_w = raw.width;
	int orig_h = raw.height;

	struct sam3_image resized = {0};
	err = sam3_image_resize(&raw, &resized, target, target);
	sam3_image_free(&raw);
	if (err)
		return err;

	err = sam3_set_image(ctx, resized.pixels,
			     resized.width, resized.height);
	sam3_image_free(&resized);
	if (err)
		return err;

	/*
	 * Override prompt coordinate space with original image dims.
	 * sam3_set_image stored the resized (square) dims, but users
	 * provide point/box coordinates in the original image space.
	 */
	ctx->proc.prompt_w = orig_w;
	ctx->proc.prompt_h = orig_h;

	return SAM3_OK;
}

void sam3_set_prompt_space(sam3_ctx *ctx, int width, int height)
{
	if (!ctx || !ctx->proc_ready)
		return;
	ctx->proc.prompt_w = width;
	ctx->proc.prompt_h = height;
}

enum sam3_error sam3_set_text(sam3_ctx *ctx, const char *text)
{
	if (!ctx || !text)
		return SAM3_EINVAL;
	if (!ctx->proc_ready)
		return SAM3_EINVAL;

	return sam3_processor_set_text(&ctx->proc, text);
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
	free(result->boxes);
	result->masks      = NULL;
	result->iou_scores = NULL;
	result->boxes      = NULL;
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
	sam3_arena_set_profiler(ctx->profiler);
	if (ctx->proc_ready)
		ctx->proc.profiler = ctx->profiler;
	return SAM3_OK;
#else
	(void)ctx;
	return SAM3_OK;
#endif
}

void sam3_profile_disable(sam3_ctx *ctx)
{
#ifdef SAM3_HAS_PROFILE
	if (ctx && ctx->profiler) {
		sam3_profiler_disable(ctx->profiler);
		sam3_arena_set_profiler(NULL);
	}
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

struct sam3_profiler *sam3_profile_get(sam3_ctx *ctx)
{
#ifdef SAM3_HAS_PROFILE
	return ctx ? ctx->profiler : NULL;
#else
	(void)ctx;
	return NULL;
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

enum sam3_error sam3_dump_tensors(sam3_ctx *ctx, const char *out_dir)
{
	char path[1024];
	struct sam3_image_model *m;

	if (!ctx || !out_dir || !ctx->proc_ready)
		return SAM3_EINVAL;

	m = &ctx->proc.model;
	if (!m->image_encoded)
		return SAM3_EINVAL;

	/* Dump neck features at each scale */
	struct { const char *name; struct sam3_tensor *t; } dumps[] = {
		{"neck_4x.bin",  m->cached_feat_4x_nhwc},
		{"neck_2x.bin",  m->cached_feat_s0_nhwc},
		{"neck_1x.bin",  m->cached_feat_s1_nhwc},
		{"neck_05x.bin", m->cached_image_features},
	};

	for (int i = 0; i < 4; i++) {
		if (!dumps[i].t)
			continue;
		snprintf(path, sizeof(path), "%s/%s",
			 out_dir, dumps[i].name);
		if (sam3_tensor_dump(path, dumps[i].t) != 0) {
			sam3_log_warn("dump_tensors: failed to write %s",
				      path);
		} else {
			sam3_log_info("dump_tensors: wrote %s "
				      "[%d,%d,%d,%d]",
				      dumps[i].name,
				      dumps[i].t->dims[0],
				      dumps[i].t->dims[1],
				      dumps[i].t->dims[2],
				      dumps[i].t->dims[3]);
		}
	}

	/* Dump text features if available */
	if (m->cached_text_features) {
		snprintf(path, sizeof(path),
			 "%s/text_features.bin", out_dir);
		if (sam3_tensor_dump(path, m->cached_text_features) != 0)
			sam3_log_warn("dump_tensors: failed to write "
				      "text_features.bin");
		else
			sam3_log_info("dump_tensors: wrote "
				      "text_features.bin");
	}

	return SAM3_OK;
}
