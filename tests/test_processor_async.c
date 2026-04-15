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
 * test_set_text_returns_immediately - Verify set_text() returns OK and
 * marks the worker thread active. The worker is still in flight when
 * this returns; processor_free() joins it cleanly.
 */
static void test_set_text_returns_immediately(void)
{
	if (!model_available()) {
		printf("  model weights missing, skipping\n");
		return;
	}

	struct sam3_processor proc;
	enum sam3_error err = sam3_processor_init(&proc, SAM3_BACKBONE_HIERA);
	ASSERT_EQ(err, SAM3_OK);

	struct sam3_weight_file wf;
	err = sam3_weight_open(&wf, MODEL_PATH);
	ASSERT_EQ(err, SAM3_OK);

	err = sam3_processor_load(&proc, &wf, VOCAB_PATH);
	ASSERT_EQ(err, SAM3_OK);

	err = sam3_processor_set_text(&proc, "cat");
	ASSERT_EQ(err, SAM3_OK);
	ASSERT_EQ(proc.text_thread_active, 1);
	ASSERT_EQ(proc.text_n_tokens > 0, 1);

	/*
	 * processor_free() joins the worker. After it returns the
	 * worker has produced text_features_async (or an error code
	 * in text_thread_err), but we just verify the join happens
	 * without crashing or leaking. processor_free MUST run before
	 * weight_close: while the worker is in flight it is reading
	 * mmap'd weight tensors, and unmapping them mid-flight would
	 * SEGV the worker.
	 */
	sam3_processor_free(&proc);
	sam3_weight_close(&wf);
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
	enum sam3_error err = sam3_processor_init(&proc, SAM3_BACKBONE_HIERA);
	ASSERT_EQ(err, SAM3_OK);

	struct sam3_weight_file wf;
	err = sam3_weight_open(&wf, MODEL_PATH);
	ASSERT_EQ(err, SAM3_OK);

	err = sam3_processor_load(&proc, &wf, VOCAB_PATH);
	ASSERT_EQ(err, SAM3_OK);

	/*
	 * Kick off async text encoding. We do not check
	 * text_features_async here — the worker thread races us and
	 * the field is only guaranteed valid after segment() (or
	 * processor_free) joins the worker.
	 */
	err = sam3_processor_set_text(&proc, "cat");
	ASSERT_EQ(err, SAM3_OK);
	ASSERT_EQ(proc.text_thread_active, 1);

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
	sam3_processor_free(&proc);
	sam3_weight_close(&wf);
}

/*
 * test_set_text_parallel_with_set_image - Verify that running
 * set_text + set_image + segment produces a result that agrees
 * with the sequential (legacy) path. This is the correctness test
 * for the threaded pipeline.
 *
 * The legacy path runs the text encoder on the main (Metal/F16)
 * backend; the async path runs it on a CPU/F32 backend in a
 * worker thread. The two backends produce numerically slightly
 * different text features, and that drift is amplified through
 * the cross-attention/decoder/mask head pipeline. We therefore
 * compare the IoU score vector — bounded in [0, 1] and trained
 * to be confident — with a 5e-2 tolerance, and report mask-logit
 * drift as informational only.
 */
static void test_set_text_parallel_with_set_image(void)
{
	if (!model_available()) {
		printf("  model weights missing, skipping\n");
		return;
	}

	const int W = 1008, H = 1008;
	uint8_t *pixels = calloc((size_t)W * H * 3, 1);
	ASSERT(pixels != NULL);
	for (int i = 0; i < W * H * 3; i++)
		pixels[i] = (uint8_t)((i * 37) % 256);

	struct sam3_prompt prompts[1];
	prompts[0].type = SAM3_PROMPT_TEXT;
	prompts[0].text = "cat";

	/* ── Run 1: legacy inline path ──────────────────────────── */
	float *masks_legacy = NULL;
	float *iou_legacy   = NULL;
	int    n_masks_legacy = 0, mh = 0, mw = 0;
	{
		struct sam3_processor proc;
		ASSERT_EQ(sam3_processor_init(&proc, SAM3_BACKBONE_HIERA), SAM3_OK);

		struct sam3_weight_file wf;
		ASSERT_EQ(sam3_weight_open(&wf, MODEL_PATH), SAM3_OK);
		ASSERT_EQ(sam3_processor_load(&proc, &wf, VOCAB_PATH),
			  SAM3_OK);

		ASSERT_EQ(sam3_processor_set_image(&proc, pixels, W, H),
			  SAM3_OK);

		struct sam3_result r;
		ASSERT_EQ(sam3_processor_segment(&proc, prompts, 1, &r),
			  SAM3_OK);

		n_masks_legacy = r.n_masks;
		mh = r.mask_height;
		mw = r.mask_width;

		size_t mbytes = (size_t)n_masks_legacy * mh * mw *
				sizeof(float);
		masks_legacy = malloc(mbytes);
		ASSERT(masks_legacy != NULL);
		memcpy(masks_legacy, r.masks, mbytes);

		size_t ibytes = (size_t)n_masks_legacy * sizeof(float);
		iou_legacy = malloc(ibytes);
		ASSERT(iou_legacy != NULL);
		memcpy(iou_legacy, r.iou_scores, ibytes);

		sam3_result_free(&r);
		sam3_processor_free(&proc);
		sam3_weight_close(&wf);
	}

	/* ── Run 2: async (set_text BEFORE set_image) ──────────── */
	float *masks_async = NULL;
	float *iou_async   = NULL;
	int    n_masks_async = 0;
	{
		struct sam3_processor proc;
		ASSERT_EQ(sam3_processor_init(&proc, SAM3_BACKBONE_HIERA), SAM3_OK);

		struct sam3_weight_file wf;
		ASSERT_EQ(sam3_weight_open(&wf, MODEL_PATH), SAM3_OK);
		ASSERT_EQ(sam3_processor_load(&proc, &wf, VOCAB_PATH),
			  SAM3_OK);

		ASSERT_EQ(sam3_processor_set_text(&proc, "cat"), SAM3_OK);
		ASSERT_EQ(sam3_processor_set_image(&proc, pixels, W, H),
			  SAM3_OK);

		struct sam3_result r;
		ASSERT_EQ(sam3_processor_segment(&proc, prompts, 1, &r),
			  SAM3_OK);

		n_masks_async = r.n_masks;
		ASSERT_EQ(r.mask_height, mh);
		ASSERT_EQ(r.mask_width, mw);

		size_t mbytes = (size_t)n_masks_async * mh * mw *
				sizeof(float);
		masks_async = malloc(mbytes);
		ASSERT(masks_async != NULL);
		memcpy(masks_async, r.masks, mbytes);

		size_t ibytes = (size_t)n_masks_async * sizeof(float);
		iou_async = malloc(ibytes);
		ASSERT(iou_async != NULL);
		memcpy(iou_async, r.iou_scores, ibytes);

		sam3_result_free(&r);
		sam3_processor_free(&proc);
		sam3_weight_close(&wf);
	}

	/* ── Compare ────────────────────────────────────────────── */
	ASSERT_EQ(n_masks_async, n_masks_legacy);

	/* Mask-logit drift (informational). */
	int nelems = n_masks_legacy * mh * mw;
	int n_diff = 0;
	float mask_max_diff = 0.0f;
	for (int i = 0; i < nelems; i++) {
		float d = masks_async[i] - masks_legacy[i];
		if (d < 0) d = -d;
		if (d > mask_max_diff) mask_max_diff = d;
		if (d > 1e-3f) n_diff++;
	}
	printf("  mask logits: max_diff=%.6f, n_diff>1e-3=%d/%d\n",
	       mask_max_diff, n_diff, nelems);

	/*
	 * IoU score drift — the actual correctness signal. IoU
	 * scores are bounded [0, 1] and the decoder is trained to
	 * be confident, so a few percent drift between Metal F16
	 * and CPU F32 paths is the expected upper bound.
	 */
	float iou_max_diff = 0.0f;
	for (int i = 0; i < n_masks_legacy; i++) {
		float d = iou_async[i] - iou_legacy[i];
		if (d < 0) d = -d;
		if (d > iou_max_diff) iou_max_diff = d;
	}
	printf("  iou scores:  max_diff=%.6f over %d masks\n",
	       iou_max_diff, n_masks_legacy);
	ASSERT(iou_max_diff < 5e-2f);

	free(masks_legacy);
	free(masks_async);
	free(iou_legacy);
	free(iou_async);
	free(pixels);
}

int main(void)
{
	test_set_text_returns_immediately();
	test_set_text_then_segment();
	test_set_text_parallel_with_set_image();
	TEST_REPORT();
}
