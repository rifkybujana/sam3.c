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

#include <math.h>
#include <stdio.h>
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
 * Compares two results with per-tensor Metal tolerance. Once the
 * batched-driver path replaced the per-slot loop, the batch output
 * diverges from single-shot at ULP level due to Metal float
 * reassociation across the larger reduction tiles.
 *
 * Tolerance is split by internal dtype:
 *   - mask logits run F16 end-to-end; accumulated reassociation
 *     reaches ~1 F16 ULP (~0.0078 at magnitude 8). rtol=2e-3
 *     atol=5e-3 covers 1 ULP up to magnitude ~32 with margin.
 *   - iou_scores and boxes are F32 end-to-end; hold them to the
 *     per-op Metal tolerance rtol=1e-4, atol=1e-5.
 *
 * Reports one assert per tensor (early-exit on first divergence) so
 * the test count stays comparable to the previous memcmp helper.
 */
static void assert_results_close(const struct sam3_result *a,
				 const struct sam3_result *b)
{
	ASSERT_EQ(a->n_masks,     b->n_masks);
	ASSERT_EQ(a->mask_height, b->mask_height);
	ASSERT_EQ(a->mask_width,  b->mask_width);
	ASSERT_EQ(a->iou_valid,   b->iou_valid);
	ASSERT_EQ(a->boxes_valid, b->boxes_valid);
	ASSERT_EQ(a->best_mask,   b->best_mask);

	const float mask_rtol = 2e-3f;
	const float mask_atol = 5e-3f;
	const float fp32_rtol = 1e-4f;
	const float fp32_atol = 1e-5f;

	const float *am = (const float *)a->masks;
	const float *bm = (const float *)b->masks;
	size_t mask_n = (size_t)a->n_masks * a->mask_height * a->mask_width;
	int mask_fail = 0;
	size_t fail_idx = 0;
	float fail_a = 0.f, fail_b = 0.f;
	for (size_t i = 0; i < mask_n; i++) {
		float tol = mask_atol + mask_rtol * fabsf(bm[i]);
		if (fabsf(am[i] - bm[i]) > tol) {
			mask_fail = 1;
			fail_idx = i;
			fail_a = am[i];
			fail_b = bm[i];
			break;
		}
	}
	if (mask_fail) {
		fprintf(stderr,
			"FAIL %s:%d: masks tolerance at idx %zu: "
			"a=%.6f b=%.6f\n",
			__FILE__, __LINE__, fail_idx,
			(double)fail_a, (double)fail_b);
		tests_failed++;
	}
	tests_run++;

	int iou_fail = 0;
	int iou_fail_i = 0;
	float iou_a = 0.f, iou_b = 0.f;
	for (int i = 0; i < a->n_masks; i++) {
		float tol = fp32_atol + fp32_rtol * fabsf(b->iou_scores[i]);
		if (fabsf(a->iou_scores[i] - b->iou_scores[i]) > tol) {
			iou_fail = 1;
			iou_fail_i = i;
			iou_a = a->iou_scores[i];
			iou_b = b->iou_scores[i];
			break;
		}
	}
	if (iou_fail) {
		fprintf(stderr,
			"FAIL %s:%d: iou_scores tolerance at idx %d: "
			"a=%.6f b=%.6f\n",
			__FILE__, __LINE__, iou_fail_i,
			(double)iou_a, (double)iou_b);
		tests_failed++;
	}
	tests_run++;

	if (a->boxes_valid) {
		int box_fail = 0;
		int box_fail_i = 0;
		float box_a = 0.f, box_b = 0.f;
		for (int i = 0; i < a->n_masks * 4; i++) {
			float tol = fp32_atol + fp32_rtol * fabsf(b->boxes[i]);
			if (fabsf(a->boxes[i] - b->boxes[i]) > tol) {
				box_fail = 1;
				box_fail_i = i;
				box_a = a->boxes[i];
				box_b = b->boxes[i];
				break;
			}
		}
		if (box_fail) {
			fprintf(stderr,
				"FAIL %s:%d: boxes tolerance at idx %d: "
				"a=%.6f b=%.6f\n",
				__FILE__, __LINE__, box_fail_i,
				(double)box_a, (double)box_b);
			tests_failed++;
		}
		tests_run++;
	}
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

	assert_results_close(&batch[0], &ref_a);
	assert_results_close(&batch[1], &ref_b);

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
