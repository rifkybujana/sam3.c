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

int main(void)
{
	test_image_cache_hit_skips_encoder();
	TEST_REPORT();
}
