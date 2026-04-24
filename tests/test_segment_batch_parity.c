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

static void fill_deterministic(uint8_t *pix, int sz)
{
	/* Deterministic, non-degenerate pattern. Index-based RGB so the
	 * encoded features are non-uniform and exercise every stage. */
	for (int i = 0; i < sz * sz * 3; i++)
		pix[i] = (uint8_t)((i * 31 + 17) & 0xff);
}

/*
 * assert_results_close - Compare two sam3_results with per-tensor
 * Metal tolerance. Metal float reassociation between the single-slot
 * segment path and the batched-driver path produces ULP-level
 * differences at the mask/score outputs, even when each batched op
 * is byte-exact at its own unit test.
 *
 * Tolerance is split by internal dtype:
 *   - mask logits run F16 end-to-end on Metal; accumulated
 *     reassociation reaches ~1 F16 ULP. F16 ULP at magnitude M is
 *     2^(floor(log2(M))-10); at M=8 that is 0.0078. rtol=2e-3
 *     atol=5e-3 covers 1 ULP up to M~32 with margin, and beyond
 *     that rtol*M dominates, still within noise.
 *   - iou_scores and boxes are F32 end-to-end; hold them to the
 *     per-op Metal tolerance used across the batched unit tests.
 *
 * Each tensor is checked element-wise but reported as one assert per
 * tensor (early-exit on first divergence) to keep the test count
 * comparable to the previous memcmp-based helper.
 */
static void assert_results_close(const struct sam3_result *a,
				 const struct sam3_result *b)
{
	ASSERT_EQ(a->n_masks,      b->n_masks);
	ASSERT_EQ(a->mask_height,  b->mask_height);
	ASSERT_EQ(a->mask_width,   b->mask_width);
	ASSERT_EQ(a->iou_valid,    b->iou_valid);
	ASSERT_EQ(a->boxes_valid,  b->boxes_valid);
	ASSERT_EQ(a->best_mask,    b->best_mask);

	const float mask_rtol = 2e-3f;
	const float mask_atol = 5e-3f;
	const float fp32_rtol = 1e-4f;
	const float fp32_atol = 1e-5f;

	const float *am = (const float *)a->masks;
	const float *bm = (const float *)b->masks;
	size_t mn = (size_t)a->n_masks * a->mask_height * a->mask_width;
	int mask_fail = 0;
	size_t fail_idx = 0;
	float fail_a = 0.f, fail_b = 0.f;
	for (size_t i = 0; i < mn; i++) {
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
		assert_results_close(&refs[i], &batch[i]);

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

	assert_results_close(&ref_a, &batch[0]);
	assert_results_close(&ref_b, &batch[1]);

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
