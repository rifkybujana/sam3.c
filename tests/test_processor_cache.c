/*
 * tests/test_processor_cache.c - End-to-end image/text cache tests
 *
 * Loads a real model and verifies that repeated set_image / set_text
 * calls hit the cache. Skips if model weights are not present.
 *
 * Key types:  sam3_processor
 * Depends on: test_helpers.h, sam3/sam3.h, model/sam3_processor.h,
 *             model/feature_cache.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"

#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include "sam3/sam3.h"
#include "core/weight.h"
#include "model/sam3_processor.h"
#include "model/feature_cache.h"
#include "model/sam3_internal.h"

#ifndef SAM3_SOURCE_DIR
#define SAM3_SOURCE_DIR "."
#endif

#define MODEL_PATH SAM3_SOURCE_DIR "/models/sam3.sam3"
#define VOCAB_PATH SAM3_SOURCE_DIR "/models/bpe_simple_vocab_16e6.txt.gz"

static int model_available(void)
{
	return access(MODEL_PATH, F_OK) == 0;
}

static void test_image_cache_hit_skips_encoder(void)
{
	if (!model_available()) {
		printf("  model weights missing, skipping\n");
		return;
	}

	struct sam3_processor proc;
	enum sam3_error err;
	err = sam3_processor_init(&proc, SAM3_BACKBONE_HIERA, 4);
	ASSERT_EQ(err, SAM3_OK);

	struct sam3_weight_file wf;
	err = sam3_weight_open(&wf, MODEL_PATH);
	ASSERT_EQ(err, SAM3_OK);
	err = sam3_processor_load(&proc, &wf, VOCAB_PATH);
	ASSERT_EQ(err, SAM3_OK);

	int sz = sam3_processor_img_size(&proc);
	uint8_t *pixels = calloc((size_t)sz * sz * 3, 1);
	for (int i = 0; i < sz * sz * 3; i++)
		pixels[i] = (uint8_t)(i & 0xff);

	err = sam3_processor_set_image(&proc, pixels, sz, sz);
	ASSERT_EQ(err, SAM3_OK);

	struct sam3_cache_stats st0 = {0};
	sam3_image_cache_stats(proc.img_cache, &st0);
	ASSERT_EQ(st0.image_hits, 0u);
	ASSERT_EQ(st0.image_misses, 1u);

	err = sam3_processor_set_image(&proc, pixels, sz, sz);
	ASSERT_EQ(err, SAM3_OK);

	struct sam3_cache_stats st1 = {0};
	sam3_image_cache_stats(proc.img_cache, &st1);
	ASSERT_EQ(st1.image_hits, 1u);
	ASSERT_EQ(st1.image_misses, 1u);

	free(pixels);
	sam3_processor_free(&proc);
	sam3_weight_close(&wf);
}

static void test_text_cache_hit_skips_worker(void)
{
	if (!model_available()) {
		printf("  model weights missing, skipping\n");
		return;
	}

	struct sam3_processor proc;
	enum sam3_error err;
	err = sam3_processor_init(&proc, SAM3_BACKBONE_HIERA, 4);
	ASSERT_EQ(err, SAM3_OK);

	struct sam3_weight_file wf;
	err = sam3_weight_open(&wf, MODEL_PATH);
	ASSERT_EQ(err, SAM3_OK);
	err = sam3_processor_load(&proc, &wf, VOCAB_PATH);
	ASSERT_EQ(err, SAM3_OK);

	/* First call → miss + worker spawned. */
	err = sam3_processor_set_text(&proc, "cat");
	ASSERT_EQ(err, SAM3_OK);

	/* Drive the segment path so the worker output gets registered. */
	int sz = sam3_processor_img_size(&proc);
	uint8_t *pixels = calloc((size_t)sz * sz * 3, 1);
	err = sam3_processor_set_image(&proc, pixels, sz, sz);
	ASSERT_EQ(err, SAM3_OK);

	struct sam3_prompt p = {.type = SAM3_PROMPT_TEXT,
				.text = "cat"};
	struct sam3_result r = {0};
	err = sam3_processor_segment(&proc, &p, 1, &r);
	ASSERT_EQ(err, SAM3_OK);
	sam3_result_free(&r);

	struct sam3_cache_stats s0 = {0};
	sam3_text_cache_stats(proc.txt_cache, &s0);
	ASSERT_EQ(s0.text_misses, 1u);
	ASSERT_EQ(s0.text_hits, 0u);

	/* Second set_text("cat") → hit, no worker. */
	err = sam3_processor_set_text(&proc, "cat");
	ASSERT_EQ(err, SAM3_OK);
	ASSERT_EQ(proc.text_thread_active, 0);

	struct sam3_cache_stats s1 = {0};
	sam3_text_cache_stats(proc.txt_cache, &s1);
	ASSERT_EQ(s1.text_hits, 1u);

	free(pixels);
	sam3_processor_free(&proc);
	sam3_weight_close(&wf);
}

static void test_top_level_cache_clear_and_stats(void)
{
	if (!model_available()) {
		printf("  model weights missing, skipping\n");
		return;
	}
	sam3_ctx *ctx = sam3_init();
	ASSERT_NOT_NULL(ctx);
	enum sam3_error err = sam3_load_model(ctx, MODEL_PATH);
	ASSERT_EQ(err, SAM3_OK);

	int sz = sam3_get_image_size(ctx);
	uint8_t *pixels = calloc((size_t)sz * sz * 3, 1);
	ASSERT_EQ(sam3_set_image(ctx, pixels, sz, sz), SAM3_OK);
	ASSERT_EQ(sam3_set_image(ctx, pixels, sz, sz), SAM3_OK);

	struct sam3_cache_stats st = {0};
	sam3_cache_stats(ctx, &st);
	ASSERT_EQ(st.image_hits, 1u);

	sam3_cache_clear(ctx, SAM3_CACHE_IMAGE);
	ASSERT_EQ(sam3_set_image(ctx, pixels, sz, sz), SAM3_OK);
	sam3_cache_stats(ctx, &st);
	/* clear resets counters, then a fresh miss bumps image_misses to 1. */
	ASSERT_EQ(st.image_hits, 0u);
	ASSERT(st.image_misses >= 1u);

	free(pixels);
	sam3_free(ctx);
}

/*
 * test_segment_does_not_grow_model_arena - Repeated segment() calls
 * must not grow model_arena.offset after weights are loaded. Since the
 * image-cache redesign, encoder outputs live in per-slot arenas and
 * text features live in the text cache arena; model_arena should hold
 * weights only. Per-segment tensors go into scratch and are rolled back.
 */
static void test_segment_does_not_grow_model_arena(void)
{
	if (!model_available()) {
		printf("  model weights missing, skipping\n");
		return;
	}

	struct sam3_processor proc;
	enum sam3_error err = sam3_processor_init(&proc,
						  SAM3_BACKBONE_HIERA, 4);
	ASSERT_EQ(err, SAM3_OK);
	struct sam3_weight_file wf;
	ASSERT_EQ(sam3_weight_open(&wf, MODEL_PATH), SAM3_OK);
	ASSERT_EQ(sam3_processor_load(&proc, &wf, VOCAB_PATH), SAM3_OK);

	int sz = sam3_processor_img_size(&proc);
	uint8_t *pixels = calloc((size_t)sz * sz * 3, 1);
	ASSERT_EQ(sam3_processor_set_image(&proc, pixels, sz, sz), SAM3_OK);

	size_t arena_after_load = proc.model_arena.offset;

	struct sam3_prompt p = {.type = SAM3_PROMPT_TEXT, .text = "cat"};
	struct sam3_result r = {0};

	for (int i = 0; i < 4; i++) {
		ASSERT_EQ(sam3_processor_segment(&proc, &p, 1, &r), SAM3_OK);
		sam3_result_free(&r);
		memset(&r, 0, sizeof(r));
	}

	ASSERT_EQ(proc.model_arena.offset, arena_after_load);

	free(pixels);
	sam3_processor_free(&proc);
	sam3_weight_close(&wf);
}

/*
 * test_image_cache_clear_drops_cached_pointers - Locks down the
 * invariant that sam3_cache_clear(SAM3_CACHE_IMAGE) must null the
 * model's cached_* pointers so a subsequent segment() without a fresh
 * set_image fails predictably with SAM3_EINVAL (rather than reading
 * poisoned arena memory).
 */
static void test_image_cache_clear_drops_cached_pointers(void)
{
	if (!model_available()) {
		printf("  model weights missing, skipping\n");
		return;
	}

	sam3_ctx *ctx = sam3_init();
	ASSERT_NOT_NULL(ctx);
	ASSERT_EQ(sam3_load_model(ctx, MODEL_PATH), SAM3_OK);

	int sz = sam3_get_image_size(ctx);
	uint8_t *pixels = calloc((size_t)sz * sz * 3, 1);
	ASSERT_EQ(sam3_set_image(ctx, pixels, sz, sz), SAM3_OK);

	sam3_cache_clear(ctx, SAM3_CACHE_IMAGE);

	struct sam3_prompt p = {.type = SAM3_PROMPT_TEXT, .text = "cat"};
	struct sam3_result r = {0};
	enum sam3_error err = sam3_segment(ctx, &p, 1, &r);
	ASSERT_EQ(err, SAM3_EINVAL);

	free(pixels);
	sam3_free(ctx);
}

/*
 * test_precache_image_preserves_state - sam3_precache_image must
 * populate the cache without touching the processor's current-image
 * state. A later set_image with the same pixels hits the cache.
 */
static void test_precache_image_preserves_state(void)
{
	if (!model_available()) {
		printf("  model weights missing, skipping\n");
		return;
	}

	sam3_ctx *ctx = sam3_init();
	ASSERT_NOT_NULL(ctx);
	ASSERT_EQ(sam3_load_model(ctx, MODEL_PATH), SAM3_OK);

	int sz = sam3_get_image_size(ctx);
	uint8_t *pix_a = calloc((size_t)sz * sz * 3, 1);
	uint8_t *pix_b = calloc((size_t)sz * sz * 3, 1);
	for (int i = 0; i < sz * sz * 3; i++)
		pix_b[i] = (uint8_t)(i & 0xff);

	/* Make A current, then precache B — A must stay current. */
	ASSERT_EQ(sam3_set_image(ctx, pix_a, sz, sz), SAM3_OK);
	int slot_a = ctx->proc.current_img_slot;
	ASSERT_EQ(sam3_precache_image(ctx, pix_b, sz, sz), SAM3_OK);
	ASSERT_EQ(ctx->proc.current_img_slot, slot_a);
	ASSERT_EQ(ctx->proc.image_loaded, 1);

	/* Now activate B via set_image — it should hit the precache. */
	struct sam3_cache_stats before = {0};
	sam3_cache_stats(ctx, &before);
	ASSERT_EQ(sam3_set_image(ctx, pix_b, sz, sz), SAM3_OK);
	struct sam3_cache_stats after = {0};
	sam3_cache_stats(ctx, &after);
	ASSERT_EQ(after.image_hits, before.image_hits + 1u);

	free(pix_a);
	free(pix_b);
	sam3_free(ctx);
}

/*
 * test_precache_text_preserves_state - sam3_precache_text must
 * populate the text cache synchronously without setting any
 * pending-prompt state on the processor.
 */
static void test_precache_text_preserves_state(void)
{
	if (!model_available()) {
		printf("  model weights missing, skipping\n");
		return;
	}

	sam3_ctx *ctx = sam3_init();
	ASSERT_NOT_NULL(ctx);
	ASSERT_EQ(sam3_load_model(ctx, MODEL_PATH), SAM3_OK);

	/* Precache "cat" — no worker should be active, no bundle pending. */
	ASSERT_EQ(sam3_precache_text(ctx, "cat"), SAM3_OK);
	ASSERT_EQ(ctx->proc.text_thread_active, 0);
	ASSERT(ctx->proc.text_cached_bundle == NULL);
	ASSERT_EQ(ctx->proc.text_worker_slot, -1);

	/* A later set_text("cat") must hit the cache. */
	struct sam3_cache_stats before = {0};
	sam3_cache_stats(ctx, &before);
	ASSERT_EQ(sam3_set_text(ctx, "cat"), SAM3_OK);
	struct sam3_cache_stats after = {0};
	sam3_cache_stats(ctx, &after);
	ASSERT_EQ(after.text_hits, before.text_hits + 1u);
	ASSERT_EQ(ctx->proc.text_thread_active, 0);

	sam3_free(ctx);
}

/*
 * test_cache_persist_image_roundtrip - Precache an image, save it to
 * a .sam3cache file, load it into a FRESH ctx, verify set_image on
 * the same pixels hits the cache.
 */
static void test_cache_persist_image_roundtrip(void)
{
	if (!model_available()) {
		printf("  model weights missing, skipping\n");
		return;
	}

	const char *tmpfile = "/tmp/sam3_test_image.sam3cache";
	int sz;
	uint8_t *pix;

	/* Session 1: precache + save. */
	{
		sam3_ctx *ctx = sam3_init();
		ASSERT_NOT_NULL(ctx);
		ASSERT_EQ(sam3_load_model(ctx, MODEL_PATH), SAM3_OK);
		sz = sam3_get_image_size(ctx);
		pix = malloc((size_t)sz * sz * 3);
		for (int i = 0; i < sz * sz * 3; i++)
			pix[i] = (uint8_t)((i * 17) & 0xff);
		ASSERT_EQ(sam3_precache_image(ctx, pix, sz, sz), SAM3_OK);
		ASSERT_EQ(sam3_cache_save_image(ctx, pix, sz, sz, tmpfile),
			  SAM3_OK);
		sam3_free(ctx);
	}

	/* Session 2: fresh ctx, load from disk, verify hit. */
	{
		sam3_ctx *ctx = sam3_init();
		ASSERT_NOT_NULL(ctx);
		ASSERT_EQ(sam3_load_model(ctx, MODEL_PATH), SAM3_OK);
		ASSERT_EQ(sam3_cache_load_image(ctx, tmpfile), SAM3_OK);

		struct sam3_cache_stats before = {0};
		sam3_cache_stats(ctx, &before);
		ASSERT_EQ(sam3_set_image(ctx, pix, sz, sz), SAM3_OK);
		struct sam3_cache_stats after = {0};
		sam3_cache_stats(ctx, &after);
		ASSERT_EQ(after.image_hits, before.image_hits + 1u);
		sam3_free(ctx);
	}

	free(pix);
	unlink(tmpfile);
}

/*
 * test_cache_persist_text_roundtrip - Same for text.
 */
static void test_cache_persist_text_roundtrip(void)
{
	if (!model_available()) {
		printf("  model weights missing, skipping\n");
		return;
	}

	const char *tmpfile = "/tmp/sam3_test_text.sam3cache";
	const char *text = "cat";

	{
		sam3_ctx *ctx = sam3_init();
		ASSERT_NOT_NULL(ctx);
		ASSERT_EQ(sam3_load_model(ctx, MODEL_PATH), SAM3_OK);
		ASSERT_EQ(sam3_precache_text(ctx, text), SAM3_OK);
		ASSERT_EQ(sam3_cache_save_text(ctx, text, tmpfile), SAM3_OK);
		sam3_free(ctx);
	}

	{
		sam3_ctx *ctx = sam3_init();
		ASSERT_NOT_NULL(ctx);
		ASSERT_EQ(sam3_load_model(ctx, MODEL_PATH), SAM3_OK);
		ASSERT_EQ(sam3_cache_load_text(ctx, tmpfile), SAM3_OK);

		struct sam3_cache_stats before = {0};
		sam3_cache_stats(ctx, &before);
		ASSERT_EQ(sam3_set_text(ctx, text), SAM3_OK);
		struct sam3_cache_stats after = {0};
		sam3_cache_stats(ctx, &after);
		ASSERT_EQ(after.text_hits, before.text_hits + 1u);
		sam3_free(ctx);
	}

	unlink(tmpfile);
}

int main(void)
{
	test_image_cache_hit_skips_encoder();
	test_text_cache_hit_skips_worker();
	test_top_level_cache_clear_and_stats();
	test_segment_does_not_grow_model_arena();
	test_image_cache_clear_drops_cached_pointers();
	test_precache_image_preserves_state();
	test_precache_text_preserves_state();
	test_cache_persist_image_roundtrip();
	test_cache_persist_text_roundtrip();
	TEST_REPORT();
}
