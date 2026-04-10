/*
 * tests/test_processor_async.c - Async text encoding API tests
 *
 * Tests sam3_processor_set_text() in both synchronous and asynchronous
 * modes, plus the integration with set_image() and segment(). Skips
 * gracefully if model weights are not present.
 *
 * Key types:  sam3_processor
 * Depends on: test_helpers.h, sam3/sam3.h, model/sam3_processor.h
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

#ifndef SAM3_SOURCE_DIR
#define SAM3_SOURCE_DIR "."
#endif

#define MODEL_PATH SAM3_SOURCE_DIR "/models/sam3.sam3"
#define VOCAB_PATH SAM3_SOURCE_DIR "/models/bpe_simple_vocab_16e6.txt.gz"

static int model_available(void)
{
	return access(MODEL_PATH, F_OK) == 0;
}

/*
 * test_set_text_sync - Verify set_text produces a non-null
 * text_features_async tensor of the right shape.
 */
static void test_set_text_sync(void)
{
	if (!model_available()) {
		printf("  model weights missing, skipping\n");
		return;
	}

	struct sam3_processor proc;
	enum sam3_error err = sam3_processor_init(&proc);
	ASSERT_EQ(err, SAM3_OK);

	struct sam3_weight_file wf;
	err = sam3_weight_open(&wf, MODEL_PATH);
	ASSERT_EQ(err, SAM3_OK);

	err = sam3_processor_load(&proc, &wf, VOCAB_PATH);
	ASSERT_EQ(err, SAM3_OK);

	err = sam3_processor_set_text(&proc, "cat");
	ASSERT_EQ(err, SAM3_OK);
	ASSERT(proc.text_features_async != NULL);
	ASSERT(proc.text_features_async->n_dims == 2);
	ASSERT(proc.text_features_async->dims[0] >= 1);
	ASSERT(proc.text_features_async->dims[1] == 256);
	ASSERT_EQ(proc.text_n_tokens, proc.text_features_async->dims[0]);

	sam3_weight_close(&wf);
	sam3_processor_free(&proc);
}

/*
 * test_set_text_then_segment - Verify the set_text → set_image →
 * segment flow produces a valid mask result.
 */
static void test_set_text_then_segment(void)
{
	if (!model_available()) {
		printf("  model weights missing, skipping\n");
		return;
	}

	struct sam3_processor proc;
	enum sam3_error err = sam3_processor_init(&proc);
	ASSERT_EQ(err, SAM3_OK);

	struct sam3_weight_file wf;
	err = sam3_weight_open(&wf, MODEL_PATH);
	ASSERT_EQ(err, SAM3_OK);

	err = sam3_processor_load(&proc, &wf, VOCAB_PATH);
	ASSERT_EQ(err, SAM3_OK);

	/* Pre-encode text */
	err = sam3_processor_set_text(&proc, "cat");
	ASSERT_EQ(err, SAM3_OK);
	ASSERT(proc.text_features_async != NULL);

	/*
	 * Build a tiny synthetic image — we just need set_image to run
	 * the encoder. The mask result will be garbage but that is OK
	 * for an API smoke test.
	 */
	const int W = 1008, H = 1008;
	uint8_t *pixels = calloc((size_t)W * H * 3, 1);
	ASSERT(pixels != NULL);
	for (int i = 0; i < W * H * 3; i++)
		pixels[i] = (uint8_t)(i % 256);

	err = sam3_processor_set_image(&proc, pixels, W, H);
	ASSERT_EQ(err, SAM3_OK);

	/* Segment with a single text prompt */
	struct sam3_prompt prompts[1];
	prompts[0].type = SAM3_PROMPT_TEXT;
	prompts[0].text = "cat";

	struct sam3_result result;
	err = sam3_processor_segment(&proc, prompts, 1, &result);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT(result.masks != NULL);
	ASSERT(result.n_masks > 0);

	/* After consumption, text_features_async should be NULL */
	ASSERT(proc.text_features_async == NULL);

	free(pixels);
	sam3_result_free(&result);
	sam3_weight_close(&wf);
	sam3_processor_free(&proc);
}

int main(void)
{
	test_set_text_sync();
	test_set_text_then_segment();
	TEST_REPORT();
}
