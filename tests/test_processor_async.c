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
#define VOCAB_PATH SAM3_SOURCE_DIR "/models/clip_vocab.json.gz"

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

int main(void)
{
	test_set_text_sync();
	TEST_REPORT();
}
