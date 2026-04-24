/*
 * tests/test_segment_batch.c - End-to-end batch segmentation tests
 *
 * Verifies sam3_segment_batch() is equivalent to running sam3_segment()
 * once per prompt set against the same cached image, and that the
 * input-validation error paths behave as documented. Skips if model
 * weights are not present.
 *
 * Key types:  sam3_ctx, sam3_prompt_set, sam3_result
 * Depends on: test_helpers.h, sam3/sam3.h
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

#ifndef SAM3_SOURCE_DIR
#define SAM3_SOURCE_DIR "."
#endif

#define MODEL_PATH SAM3_SOURCE_DIR "/models/sam3.sam3"

static int model_available(void)
{
	return access(MODEL_PATH, F_OK) == 0;
}

/* --- helpers --- */

static void fill_pixels(uint8_t *pix, int sz)
{
	for (int i = 0; i < sz * sz * 3; i++)
		pix[i] = (uint8_t)(i & 0xff);
}

/*
 * Compares two results byte-for-byte on mask and score arrays.
 * Batch must produce bit-identical output to the per-set single-shot
 * path because both dispatch the same compute graph with the same
 * inputs on the same backend.
 */
static void assert_results_identical(const struct sam3_result *a,
				     const struct sam3_result *b)
{
	ASSERT_EQ(a->n_masks,     b->n_masks);
	ASSERT_EQ(a->mask_height, b->mask_height);
	ASSERT_EQ(a->mask_width,  b->mask_width);
	ASSERT_EQ(a->iou_valid,   b->iou_valid);
	ASSERT_EQ(a->boxes_valid, b->boxes_valid);
	ASSERT_EQ(a->best_mask,   b->best_mask);

	size_t mask_n = (size_t)a->n_masks * a->mask_height * a->mask_width;
	ASSERT_EQ(memcmp(a->masks, b->masks, mask_n * sizeof(float)), 0);
	ASSERT_EQ(memcmp(a->iou_scores, b->iou_scores,
			 (size_t)a->n_masks * sizeof(float)),
		  0);
	if (a->boxes_valid)
		ASSERT_EQ(memcmp(a->boxes, b->boxes,
				 (size_t)a->n_masks * 4 * sizeof(float)),
			  0);
}

/* --- tests --- */

static void test_batch_matches_single_shot(void)
{
	if (!model_available()) {
		printf("  model weights missing, skipping\n");
		return;
	}

	sam3_ctx *ctx = sam3_init();
	ASSERT_NOT_NULL(ctx);
	ASSERT_EQ(sam3_load_model(ctx, MODEL_PATH), SAM3_OK);

	int sz = sam3_get_image_size(ctx);
	uint8_t *pixels = malloc((size_t)sz * sz * 3);
	ASSERT_NOT_NULL(pixels);
	fill_pixels(pixels, sz);

	ASSERT_EQ(sam3_set_image(ctx, pixels, sz, sz), SAM3_OK);

	/* Single-shot baseline: two independent segments, same image. */
	struct sam3_prompt pa = {.type = SAM3_PROMPT_TEXT, .text = "cat"};
	struct sam3_prompt pb = {.type = SAM3_PROMPT_TEXT, .text = "dog"};

	struct sam3_result ref_a = {0}, ref_b = {0};
	ASSERT_EQ(sam3_segment(ctx, &pa, 1, &ref_a), SAM3_OK);
	ASSERT_EQ(sam3_segment(ctx, &pb, 1, &ref_b), SAM3_OK);

	/* Batch: the same two prompts as two independent prompt sets. */
	struct sam3_prompt_set sets[2] = {
		{ .prompts = &pa, .n_prompts = 1 },
		{ .prompts = &pb, .n_prompts = 1 },
	};
	struct sam3_result batch[2] = {0};
	ASSERT_EQ(sam3_segment_batch(ctx, sets, 2, batch), SAM3_OK);

	assert_results_identical(&batch[0], &ref_a);
	assert_results_identical(&batch[1], &ref_b);

	sam3_result_free(&ref_a);
	sam3_result_free(&ref_b);
	sam3_result_free(&batch[0]);
	sam3_result_free(&batch[1]);
	free(pixels);
	sam3_free(ctx);
}

static void test_batch_invalid_args(void)
{
	sam3_ctx *ctx = sam3_init();
	ASSERT_NOT_NULL(ctx);

	/* No image loaded yet: should fail cleanly even without a model. */
	struct sam3_prompt p = {.type = SAM3_PROMPT_TEXT, .text = "cat"};
	struct sam3_prompt_set set = { .prompts = &p, .n_prompts = 1 };
	struct sam3_result r = {0};

	ASSERT_EQ(sam3_segment_batch(ctx, &set, 1, &r), SAM3_EINVAL);
	ASSERT_EQ(sam3_segment_batch(NULL, &set, 1, &r), SAM3_EINVAL);
	ASSERT_EQ(sam3_segment_batch(ctx, &set, 1, NULL), SAM3_EINVAL);
	ASSERT_EQ(sam3_segment_batch(ctx, NULL, 1, &r),   SAM3_EINVAL);
	ASSERT_EQ(sam3_segment_batch(ctx, &set, 0, &r),   SAM3_EINVAL);

	sam3_free(ctx);
}

static void test_batch_empty_prompt_set_rejected(void)
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

	/* Second set has n_prompts=0 — whole batch must be rejected
	 * without leaving the first result allocated. */
	struct sam3_prompt p = {.type = SAM3_PROMPT_TEXT, .text = "cat"};
	struct sam3_prompt_set sets[2] = {
		{ .prompts = &p, .n_prompts = 1 },
		{ .prompts = &p, .n_prompts = 0 },
	};
	struct sam3_result results[2] = {
		{ .masks = (float *)0xdeadbeef },  /* poison */
		{ .masks = (float *)0xdeadbeef },
	};
	ASSERT_EQ(sam3_segment_batch(ctx, sets, 2, results), SAM3_EINVAL);

	/* Validation zeros the result slots before attempting any segment. */
	ASSERT_EQ(results[0].masks, NULL);
	ASSERT_EQ(results[1].masks, NULL);

	free(pixels);
	sam3_free(ctx);
}

int main(void)
{
	test_batch_invalid_args();
	test_batch_empty_prompt_set_rejected();
	test_batch_matches_single_shot();
	TEST_REPORT();
}
