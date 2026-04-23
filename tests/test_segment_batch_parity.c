/*
 * tests/test_segment_batch_parity.c - Pin single-shot segment as the
 * parity reference for the batched-decoder refactor.
 *
 * Every subsequent task in the batched-decoder refactor must keep the
 * two tests in this file passing. Equivalence is byte-exact today
 * (sam3_segment_batch internally loops sam3_segment per set); when the
 * graph-level batched pipeline lands, equivalence stays byte-exact
 * because the same compute graph is dispatched, just with B>1.
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

static void fill_deterministic(uint8_t *pix, int sz)
{
	/* Deterministic, non-degenerate pattern. Index-based RGB so the
	 * encoded features are non-uniform and exercise every stage. */
	for (int i = 0; i < sz * sz * 3; i++)
		pix[i] = (uint8_t)((i * 31 + 17) & 0xff);
}

static void assert_results_byte_exact(const struct sam3_result *a,
				       const struct sam3_result *b)
{
	ASSERT_EQ(a->n_masks,      b->n_masks);
	ASSERT_EQ(a->mask_height,  b->mask_height);
	ASSERT_EQ(a->mask_width,   b->mask_width);
	ASSERT_EQ(a->iou_valid,    b->iou_valid);
	ASSERT_EQ(a->boxes_valid,  b->boxes_valid);
	ASSERT_EQ(a->best_mask,    b->best_mask);

	size_t mn = (size_t)a->n_masks * a->mask_height * a->mask_width;
	ASSERT_EQ(memcmp(a->masks, b->masks, mn * sizeof(float)), 0);
	ASSERT_EQ(memcmp(a->iou_scores, b->iou_scores,
			 (size_t)a->n_masks * sizeof(float)),
		  0);
	if (a->boxes_valid)
		ASSERT_EQ(memcmp(a->boxes, b->boxes,
				 (size_t)a->n_masks * 4 * sizeof(float)),
			  0);
}

/*
 * Batch with N identical prompt sets == N single-shot calls. Catches
 * any accidental state leak across batch slots.
 */
static void test_batch_vs_single_identical_sets(void)
{
	if (!model_available()) {
		printf("  model weights missing, skipping\n");
		return;
	}

	sam3_ctx *ctx = sam3_init();
	ASSERT_NOT_NULL(ctx);
	ASSERT_EQ(sam3_load_model(ctx, MODEL_PATH), SAM3_OK);

	int sz = sam3_get_image_size(ctx);
	uint8_t *pix = malloc((size_t)sz * sz * 3);
	ASSERT_NOT_NULL(pix);
	fill_deterministic(pix, sz);
	ASSERT_EQ(sam3_set_image(ctx, pix, sz, sz), SAM3_OK);

	struct sam3_prompt p = {.type = SAM3_PROMPT_TEXT, .text = "cat"};

	struct sam3_result refs[3] = {0};
	for (int i = 0; i < 3; i++)
		ASSERT_EQ(sam3_segment(ctx, &p, 1, &refs[i]), SAM3_OK);

	struct sam3_prompt_set sets[3] = {
		{.prompts = &p, .n_prompts = 1},
		{.prompts = &p, .n_prompts = 1},
		{.prompts = &p, .n_prompts = 1},
	};
	struct sam3_result batch[3] = {0};
	ASSERT_EQ(sam3_segment_batch(ctx, sets, 3, batch), SAM3_OK);

	for (int i = 0; i < 3; i++)
		assert_results_byte_exact(&refs[i], &batch[i]);

	for (int i = 0; i < 3; i++) {
		sam3_result_free(&refs[i]);
		sam3_result_free(&batch[i]);
	}
	free(pix);
	sam3_free(ctx);
}

/*
 * Batch with N different text prompts produces the correct per-set
 * masks. Catches any batch-slot aliasing in the decoder state.
 */
static void test_batch_vs_single_different_sets(void)
{
	if (!model_available()) {
		printf("  model weights missing, skipping\n");
		return;
	}

	sam3_ctx *ctx = sam3_init();
	ASSERT_NOT_NULL(ctx);
	ASSERT_EQ(sam3_load_model(ctx, MODEL_PATH), SAM3_OK);

	int sz = sam3_get_image_size(ctx);
	uint8_t *pix = malloc((size_t)sz * sz * 3);
	ASSERT_NOT_NULL(pix);
	fill_deterministic(pix, sz);
	ASSERT_EQ(sam3_set_image(ctx, pix, sz, sz), SAM3_OK);

	struct sam3_prompt pa = {.type = SAM3_PROMPT_TEXT, .text = "cat"};
	struct sam3_prompt pb = {.type = SAM3_PROMPT_TEXT, .text = "dog"};

	struct sam3_result ref_a = {0}, ref_b = {0};
	ASSERT_EQ(sam3_segment(ctx, &pa, 1, &ref_a), SAM3_OK);
	ASSERT_EQ(sam3_segment(ctx, &pb, 1, &ref_b), SAM3_OK);

	struct sam3_prompt_set sets[2] = {
		{.prompts = &pa, .n_prompts = 1},
		{.prompts = &pb, .n_prompts = 1},
	};
	struct sam3_result batch[2] = {0};
	ASSERT_EQ(sam3_segment_batch(ctx, sets, 2, batch), SAM3_OK);

	assert_results_byte_exact(&ref_a, &batch[0]);
	assert_results_byte_exact(&ref_b, &batch[1]);

	sam3_result_free(&ref_a);
	sam3_result_free(&ref_b);
	sam3_result_free(&batch[0]);
	sam3_result_free(&batch[1]);
	free(pix);
	sam3_free(ctx);
}

int main(void)
{
	test_batch_vs_single_identical_sets();
	test_batch_vs_single_different_sets();
	TEST_REPORT();
}
