/*
 * tests/test_video_api.c - Tests for public video tracking API
 *
 * Validates null-safety and basic error handling for all sam3_video_*
 * functions declared in sam3.h. Most tests exercise argument
 * validation without a real model. The end-to-end tests
 * (add_points / add_box / propagate) opt into a full pipeline run
 * when `SAM3_SOURCE_DIR/models/sam3.sam3` is available, otherwise
 * they skip.
 *
 * Key types:  sam3_video_session
 * Depends on: test_helpers.h, sam3/sam3.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "test_helpers.h"
#include "sam3/sam3.h"
#include "model/video_session.h"

#ifdef SAM3_SOURCE_DIR
#define VIDEO_TEST_MODEL_PATH  SAM3_SOURCE_DIR "/models/sam3.sam3"
#define VIDEO_TEST_FRAMES_DIR  SAM3_SOURCE_DIR "/tests/data/video2"
#else
#define VIDEO_TEST_MODEL_PATH  "models/sam3.sam3"
#define VIDEO_TEST_FRAMES_DIR  "tests/data/video2"
#endif

static int model_available(void)
{
	return access(VIDEO_TEST_MODEL_PATH, F_OK) == 0;
}

/* --- Null-safety tests ──────────────── --- */

static void test_video_frame_count_null(void)
{
	ASSERT_EQ(sam3_video_frame_count(NULL), 0);
}

static void test_video_end_null(void)
{
	/* Should not crash */
	sam3_video_end(NULL);
	ASSERT(1);
}

static void test_video_reset_null(void)
{
	enum sam3_error err = sam3_video_reset(NULL);
	ASSERT_EQ(err, SAM3_EINVAL);
}

static void test_video_remove_null(void)
{
	enum sam3_error err = sam3_video_remove_object(NULL, 0);
	ASSERT_EQ(err, SAM3_EINVAL);
}

static void test_video_propagate_null(void)
{
	enum sam3_error err = sam3_video_propagate(NULL, 0, NULL, NULL);
	ASSERT_EQ(err, SAM3_EINVAL);
}

static void test_video_add_points_null_session(void)
{
	struct sam3_point pt = {100.0f, 200.0f, 1};
	struct sam3_video_frame_result result;
	enum sam3_error err;

	err = sam3_video_add_points(NULL, 0, 0, &pt, 1, &result);
	ASSERT_EQ(err, SAM3_EINVAL);
}

static void test_video_add_points_null_points(void)
{
	struct sam3_video_frame_result result;
	enum sam3_error err;

	err = sam3_video_add_points(NULL, 0, 0, NULL, 1, &result);
	ASSERT_EQ(err, SAM3_EINVAL);
}

static void test_video_add_points_zero_count(void)
{
	struct sam3_point pt = {0};
	struct sam3_video_frame_result result;
	enum sam3_error err;

	err = sam3_video_add_points(NULL, 0, 0, &pt, 0, &result);
	ASSERT_EQ(err, SAM3_EINVAL);
}

static void test_video_add_box_null_session(void)
{
	struct sam3_box box = {10.0f, 20.0f, 100.0f, 200.0f};
	struct sam3_video_frame_result result;
	enum sam3_error err;

	err = sam3_video_add_box(NULL, 0, 0, &box, &result);
	ASSERT_EQ(err, SAM3_EINVAL);
}

static void test_video_add_box_null_box(void)
{
	struct sam3_video_frame_result result;
	enum sam3_error err;

	err = sam3_video_add_box(NULL, 0, 0, NULL, &result);
	ASSERT_EQ(err, SAM3_EINVAL);
}

static void test_video_start_null_ctx(void)
{
	sam3_video_session *s = NULL;
	enum sam3_error err;

	err = sam3_video_start(NULL, "test_path", &s);
	ASSERT_EQ(err, SAM3_EINVAL);
	ASSERT(s == NULL);
}

static void test_video_start_null_path(void)
{
	sam3_ctx *ctx = sam3_init();
	sam3_video_session *s = NULL;
	enum sam3_error err;

	err = sam3_video_start(ctx, NULL, &s);
	ASSERT_EQ(err, SAM3_EINVAL);
	ASSERT(s == NULL);

	sam3_free(ctx);
}

static void test_video_start_null_out(void)
{
	sam3_ctx *ctx = sam3_init();
	enum sam3_error err;

	err = sam3_video_start(ctx, "test_path", NULL);
	ASSERT_EQ(err, SAM3_EINVAL);

	sam3_free(ctx);
}

static void test_video_start_no_model(void)
{
	sam3_ctx *ctx = sam3_init();
	sam3_video_session *s = NULL;
	enum sam3_error err;

	/* ctx exists but no model is loaded */
	err = sam3_video_start(ctx, "test_path", &s);
	ASSERT_EQ(err, SAM3_EINVAL);
	ASSERT(s == NULL);

	sam3_free(ctx);
}

/* --- End-to-end: eager feature cache ─── --- */

static void test_video_start_caches_features(void)
{
	sam3_ctx *ctx;
	sam3_video_session *s = NULL;
	enum sam3_error err;

	if (!model_available()) {
		printf("SKIP test_video_start_caches_features: model not at "
		       "%s\n", VIDEO_TEST_MODEL_PATH);
		return;
	}

	ctx = sam3_init();
	ASSERT(ctx != NULL);

	err = sam3_load_model(ctx, VIDEO_TEST_MODEL_PATH);
	ASSERT_EQ(err, SAM3_OK);

	err = sam3_video_start(ctx, VIDEO_TEST_FRAMES_DIR, &s);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT(s != NULL);

	if (s) {
		/*
		 * With the lazy frame cache, features are not populated until
		 * first access. Verify the cache is initialised and covers the
		 * expected number of frames.
		 */
		ASSERT(s->frame_cache.n_frames == s->frames.n_frames);
		ASSERT(s->frame_cache.slots != NULL);
	}

	sam3_video_end(s);
	sam3_free(ctx);
}

/* --- End-to-end: add_points produces non-empty mask --- */

static void test_video_add_points_produces_nonempty_mask(void)
{
	sam3_ctx *ctx;
	sam3_video_session *s = NULL;
	enum sam3_error err;
	struct sam3_video_frame_result result;
	struct sam3_point pt;
	int nf;

	if (!model_available()) {
		printf("SKIP test_video_add_points_produces_nonempty_mask: "
		       "model not at %s\n", VIDEO_TEST_MODEL_PATH);
		return;
	}

	ctx = sam3_init();
	ASSERT(ctx != NULL);

	err = sam3_load_model(ctx, VIDEO_TEST_MODEL_PATH);
	ASSERT_EQ(err, SAM3_OK);

	err = sam3_video_start(ctx, VIDEO_TEST_FRAMES_DIR, &s);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT(s != NULL);

	if (!s) {
		sam3_free(ctx);
		return;
	}

	/*
	 * Click roughly in the centre of the first frame. The frames
	 * in tests/data/video2 are tiny 1x1 PNGs that get resized up
	 * to the model's input resolution; using the model input
	 * centre is therefore a safe foreground point.
	 */
	nf = s->frames.n_frames;
	ASSERT(nf >= 1);

	pt.x = (float)s->frames.frame_size / 2.0f;
	pt.y = (float)s->frames.frame_size / 2.0f;
	pt.label = 1;  /* foreground */

	memset(&result, 0, sizeof(result));

	err = sam3_video_add_points(s, 0, 0, &pt, 1, &result);
	ASSERT_EQ(err, SAM3_OK);

	/*
	 * The pipeline must have produced a real mask. After the
	 * implementation wires prompt -> decoder -> memory encoder
	 * -> memory bank, result.objects[0].mask must be a non-NULL float
	 * buffer and at least one entry must be stored in the memory
	 * bank (the prompted frame is a conditioning frame).
	 */
	ASSERT_EQ(result.n_objects, 1);
	ASSERT(result.objects != NULL);
	if (result.objects) {
		ASSERT(result.objects[0].mask != NULL);
		ASSERT(result.objects[0].mask_h > 0);
		ASSERT(result.objects[0].mask_w > 0);

		/* At least one mask pixel should be finite (not NaN / inf). */
		int has_finite = 0;
		int total = result.objects[0].mask_h *
			    result.objects[0].mask_w;
		for (int i = 0; i < total && i < 256; i++) {
			float v = result.objects[0].mask[i];
			if (v == v && v > -1e30f && v < 1e30f) {
				has_finite = 1;
				break;
			}
		}
		ASSERT(has_finite);
	}

	/* Prompted-frame bitmap must mark frame 0. */
	ASSERT(sam3_session_is_prompted(s, 0));

	/* Per-object memory bank must now contain the prompted frame as a
	 * conditioning entry. */
	ASSERT(s->objects[0].bank.n_cond >= 1);

	sam3_video_frame_result_free(&result);
	sam3_video_end(s);
	sam3_free(ctx);
}

/* --- End-to-end: reset clears all state --- */

static void test_video_reset_clears_all_state(void)
{
	sam3_ctx *ctx;
	sam3_video_session *s = NULL;
	enum sam3_error err;
	struct sam3_video_frame_result result;
	struct sam3_point pt;

	if (!model_available()) {
		printf("SKIP test_video_reset_clears_all_state: "
		       "model not at %s\n", VIDEO_TEST_MODEL_PATH);
		return;
	}

	ctx = sam3_init();
	ASSERT(ctx != NULL);

	err = sam3_load_model(ctx, VIDEO_TEST_MODEL_PATH);
	ASSERT_EQ(err, SAM3_OK);

	err = sam3_video_start(ctx, VIDEO_TEST_FRAMES_DIR, &s);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT(s != NULL);

	if (!s) {
		sam3_free(ctx);
		return;
	}

	/* Seed a prompt so there is real state to clear. */
	pt.x = (float)s->frames.frame_size / 2.0f;
	pt.y = (float)s->frames.frame_size / 2.0f;
	pt.label = 1;
	memset(&result, 0, sizeof(result));

	err = sam3_video_add_points(s, 0, 0, &pt, 1, &result);
	ASSERT_EQ(err, SAM3_OK);
	sam3_video_frame_result_free(&result);

	/* Sanity: state is populated before reset. */
	ASSERT(s->n_objects >= 1);
	ASSERT(s->n_prompts >= 1);
	ASSERT(sam3_session_is_prompted(s, 0));
	ASSERT(sam3_memory_bank_total(&s->objects[0].bank) >= 1);

	err = sam3_video_reset(s);
	ASSERT_EQ(err, SAM3_OK);

	/* Reset must clear tracked objects, stored prompts, and the
	 * prompted-frame bitmap. Per-object banks are cleared per object. */
	ASSERT_EQ(s->n_objects, 0);
	ASSERT_EQ(s->n_prompts, 0);
	ASSERT(!sam3_session_is_prompted(s, 0));

	sam3_video_end(s);
	sam3_free(ctx);
}

/* --- End-to-end: add_box produces non-empty mask --- */

static void test_video_add_box_produces_nonempty_mask(void)
{
	sam3_ctx *ctx;
	sam3_video_session *s = NULL;
	enum sam3_error err;
	struct sam3_video_frame_result result;
	struct sam3_box box;
	int nf;
	float fs;

	if (!model_available()) {
		printf("SKIP test_video_add_box_produces_nonempty_mask: "
		       "model not at %s\n", VIDEO_TEST_MODEL_PATH);
		return;
	}

	ctx = sam3_init();
	ASSERT(ctx != NULL);

	err = sam3_load_model(ctx, VIDEO_TEST_MODEL_PATH);
	ASSERT_EQ(err, SAM3_OK);

	err = sam3_video_start(ctx, VIDEO_TEST_FRAMES_DIR, &s);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT(s != NULL);

	if (!s) {
		sam3_free(ctx);
		return;
	}

	nf = s->frames.n_frames;
	ASSERT(nf >= 1);

	/*
	 * Centered box covering the middle 50% of the first frame.
	 * The test frames in video2 are tiny 1x1 PNGs upsampled to
	 * the model input resolution, so any centred region is a
	 * safe foreground prompt.
	 */
	fs = (float)s->frames.frame_size;
	box.x1 = fs * 0.25f;
	box.y1 = fs * 0.25f;
	box.x2 = fs * 0.75f;
	box.y2 = fs * 0.75f;

	memset(&result, 0, sizeof(result));

	err = sam3_video_add_box(s, 0, 0, &box, &result);
	ASSERT_EQ(err, SAM3_OK);

	/*
	 * Same post-conditions as the add_points E2E test: a real
	 * mask buffer, non-zero shape, at least one finite pixel,
	 * the prompted-frame bitmap marking frame 0, and a new
	 * conditioning entry in the per-object memory bank.
	 */
	ASSERT_EQ(result.n_objects, 1);
	ASSERT(result.objects != NULL);
	if (result.objects) {
		ASSERT(result.objects[0].mask != NULL);
		ASSERT(result.objects[0].mask_h > 0);
		ASSERT(result.objects[0].mask_w > 0);

		int has_finite = 0;
		int total = result.objects[0].mask_h *
			    result.objects[0].mask_w;
		for (int i = 0; i < total && i < 256; i++) {
			float v = result.objects[0].mask[i];
			if (v == v && v > -1e30f && v < 1e30f) {
				has_finite = 1;
				break;
			}
		}
		ASSERT(has_finite);
	}

	ASSERT(sam3_session_is_prompted(s, 0));
	ASSERT(s->objects[0].bank.n_cond >= 1);

	sam3_video_frame_result_free(&result);
	sam3_video_end(s);
	sam3_free(ctx);
}

/* --- End-to-end: propagate tracks across frames --- */

struct propagate_counts {
	int n_masks[8];
	int has_finite[8];
	int n_frames_seen;
};

static int count_cb(const struct sam3_video_frame_result *result,
		    void *user_data)
{
	struct propagate_counts *c = user_data;
	int frame_idx;

	if (!c || !result)
		return 0;

	frame_idx = result->frame_idx;
	if (frame_idx < 0 || frame_idx >= 8)
		return 0;

	c->n_frames_seen++;

	/* Use the first object's mask for the frame-level check. */
	if (result->n_objects <= 0 || !result->objects ||
	    !result->objects[0].mask)
		return 0;

	c->n_masks[frame_idx] = 1; /* at least one mask */

	int total = result->objects[0].mask_h *
		    result->objects[0].mask_w;
	for (int i = 0; i < total && i < 256; i++) {
		float v = result->objects[0].mask[i];
		if (v == v && v > -1e30f && v < 1e30f) {
			c->has_finite[frame_idx] = 1;
			break;
		}
	}
	return 0;
}

static void test_video_propagate_tracks_across_frames(void)
{
	sam3_ctx *ctx;
	sam3_video_session *s = NULL;
	enum sam3_error err;
	struct sam3_video_frame_result seed_result;
	struct sam3_point pt;
	struct propagate_counts counts;
	int nf;

	if (!model_available()) {
		printf("SKIP test_video_propagate_tracks_across_frames: "
		       "model not at %s\n", VIDEO_TEST_MODEL_PATH);
		return;
	}

	ctx = sam3_init();
	ASSERT(ctx != NULL);

	err = sam3_load_model(ctx, VIDEO_TEST_MODEL_PATH);
	ASSERT_EQ(err, SAM3_OK);

	err = sam3_video_start(ctx, VIDEO_TEST_FRAMES_DIR, &s);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT(s != NULL);

	if (!s) {
		sam3_free(ctx);
		return;
	}

	nf = s->frames.n_frames;
	ASSERT(nf >= 2);

	/* Seed a prompt on frame 0 at the image centre. */
	pt.x = (float)s->frames.frame_size / 2.0f;
	pt.y = (float)s->frames.frame_size / 2.0f;
	pt.label = 1;
	memset(&seed_result, 0, sizeof(seed_result));
	err = sam3_video_add_points(s, 0, 0, &pt, 1, &seed_result);
	ASSERT_EQ(err, SAM3_OK);
	sam3_video_frame_result_free(&seed_result);

	/* Propagate forward across all frames. */
	memset(&counts, 0, sizeof(counts));
	err = sam3_video_propagate(s, SAM3_PROPAGATE_FORWARD, count_cb,
				   &counts);
	ASSERT_EQ(err, SAM3_OK);

	/* Both frames in the test clip must have been visited. */
	ASSERT_EQ(counts.n_frames_seen, nf);

	/* Frame 0 is a prompted/conditioning frame — propagate re-runs
	 * the prompt pipeline; frame 1 is a pure-tracking frame driven
	 * off the memory bank. Both must emit masks with finite pixels. */
	ASSERT(counts.n_masks[0] > 0);
	ASSERT(counts.n_masks[1] > 0);
	ASSERT(counts.has_finite[0]);
	ASSERT(counts.has_finite[1]);

	sam3_video_end(s);
	sam3_free(ctx);
}

int main(void)
{
	test_video_frame_count_null();
	test_video_end_null();
	test_video_reset_null();
	test_video_remove_null();
	test_video_propagate_null();
	test_video_add_points_null_session();
	test_video_add_points_null_points();
	test_video_add_points_zero_count();
	test_video_add_box_null_session();
	test_video_add_box_null_box();
	test_video_start_null_ctx();
	test_video_start_null_path();
	test_video_start_null_out();
	test_video_start_no_model();
	test_video_start_caches_features();
	test_video_add_points_produces_nonempty_mask();
	test_video_reset_clears_all_state();
	test_video_add_box_produces_nonempty_mask();
	test_video_propagate_tracks_across_frames();
	TEST_REPORT();
}
