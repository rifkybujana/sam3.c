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

int main(void)
{
	test_image_cache_hit_skips_encoder();
	test_text_cache_hit_skips_worker();
	test_top_level_cache_clear_and_stats();
	test_segment_does_not_grow_model_arena();
	TEST_REPORT();
}
