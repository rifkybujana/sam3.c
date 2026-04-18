/*
 * tests/test_video_e2e.c - End-to-end video tracking integration test
 *
 * Generates a synthetic 8-frame "moving square" clip in a temp
 * directory, starts a real sam3 video session against the clip,
 * seeds a point prompt at the square's centre on frame 0, propagates
 * forward, and asserts that the per-frame mask centroid tracks the
 * ground-truth diagonal motion within a pixel tolerance. Gated
 * behind SAM3_E2E_TESTS: the translation unit requires SAM3_TEST_MODEL
 * to point at a real .sam3 checkpoint at build time.
 *
 * Key types:  sam3_video_session, sam3_result
 * Depends on: test_helpers.h, sam3/sam3.h,
 *             src/util/vendor/stb_image_write.h
 * Used by:    CTest (only when SAM3_E2E_TESTS=ON)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>

#include "test_helpers.h"
#include "sam3/sam3.h"
#include "util/log.h"

#ifndef SAM3_TEST_MODEL
#error "SAM3_TEST_MODEL must be defined by CMake when SAM3_E2E_TESTS=ON"
#endif

/* stb_image_write: declaration only; implementation lives in
 * src/util/image.c inside the sam3 library, which this test links. */
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wstrict-prototypes"
#pragma clang diagnostic ignored "-Wdouble-promotion"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wcast-align"
#pragma clang diagnostic ignored "-Wcast-qual"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wmissing-prototypes"
#pragma GCC diagnostic ignored "-Wstrict-prototypes"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif

#include "util/vendor/stb_image_write.h"

#ifdef __clang__
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

/* Synthetic clip parameters. Kept small (256x256) so the test runs
 * fast; the motion pattern mirrors scripts/gen_tracker_fixtures.py
 * (SQUARE_* constants) at a reduced image size. */
#define E2E_IMG_SIZE        256
#define E2E_N_FRAMES        8
#define E2E_SQUARE_SIZE     32
#define E2E_SQUARE_START    100
#define E2E_SQUARE_STEP     8
#define E2E_TOLERANCE_PX    8.0f

struct centroid {
	float x;
	float y;
	int   valid;
};

struct centroid_tracker {
	struct centroid centroids[E2E_N_FRAMES];
	int   mask_w;
	int   mask_h;
	int   n_frames_seen;
};

/*
 * Simple 32-bit LCG so the noise pattern is reproducible across
 * platforms without pulling in rand() locale quirks. The exact
 * values don't need to match Python — only determinism and coverage
 * of the [100, 156) range matter for the tracker.
 */
static uint32_t lcg_state = 1u;

static void lcg_seed(uint32_t seed)
{
	lcg_state = seed ? seed : 1u;
}

static uint32_t lcg_next(void)
{
	lcg_state = lcg_state * 1664525u + 1013904223u;
	return lcg_state;
}

/*
 * square_center - Ground-truth centroid of the square on frame i.
 *
 * Matches the synthesizer below: top-left corner at
 * (E2E_SQUARE_START + i * E2E_SQUARE_STEP, ...), size E2E_SQUARE_SIZE,
 * so the centre is at corner + size/2.
 */
static void square_center(int i, float *cx, float *cy)
{
	float c = (float)(E2E_SQUARE_START + i * E2E_SQUARE_STEP) +
		  (float)E2E_SQUARE_SIZE * 0.5f;
	*cx = c;
	*cy = c;
}

/*
 * generate_moving_square_clip - Write N PNG frames to @dir.
 *
 * Each frame is an E2E_IMG_SIZE x E2E_IMG_SIZE RGB image: uniform
 * noise in [100, 156) overlaid with a white 32x32 square that slides
 * diagonally by E2E_SQUARE_STEP px per frame starting at
 * (E2E_SQUARE_START, E2E_SQUARE_START).
 *
 * Returns 0 on success, non-zero on any I/O failure.
 */
static int generate_moving_square_clip(const char *dir, int n)
{
	size_t nbytes = (size_t)E2E_IMG_SIZE * E2E_IMG_SIZE * 3;
	uint8_t *buf = malloc(nbytes);
	char path[1024];
	int rc = 0;

	if (!buf) {
		sam3_log_error("out of memory for %dx%d frame buffer",
			       E2E_IMG_SIZE, E2E_IMG_SIZE);
		return 1;
	}

	lcg_seed(0xC0FFEEu);

	for (int i = 0; i < n; i++) {
		/* Gray noise background in [100, 156). */
		for (size_t k = 0; k < nbytes; k++)
			buf[k] = (uint8_t)(100u + (lcg_next() % 56u));

		int x0 = E2E_SQUARE_START + i * E2E_SQUARE_STEP;
		int y0 = E2E_SQUARE_START + i * E2E_SQUARE_STEP;
		int x1 = x0 + E2E_SQUARE_SIZE;
		int y1 = y0 + E2E_SQUARE_SIZE;

		for (int y = y0; y < y1; y++) {
			uint8_t *row = buf +
				       ((size_t)y * E2E_IMG_SIZE + x0) * 3;
			size_t run = (size_t)(x1 - x0) * 3;
			memset(row, 255, run);
		}

		int nb = snprintf(path, sizeof(path),
				  "%s/frame_%04d.png", dir, i);
		if (nb < 0 || (size_t)nb >= sizeof(path)) {
			sam3_log_error("temp frame path too long");
			rc = 1;
			break;
		}

		int ok = stbi_write_png(path, E2E_IMG_SIZE, E2E_IMG_SIZE,
					3, buf, E2E_IMG_SIZE * 3);
		if (!ok) {
			sam3_log_error("failed to write '%s'", path);
			rc = 1;
			break;
		}
	}

	free(buf);
	return rc;
}

/*
 * compute_mask_centroid - Binary centroid of a single mask plane.
 *
 * @mask: Pointer to w*h floats. A pixel is counted as "on" when
 *        its logit is >= 0 (the convention used elsewhere in sam3).
 * @w, @h: Mask dimensions.
 * @out_cx, @out_cy: On success, receive the centroid in mask-pixel
 *                   space. Untouched if the mask is all zero.
 *
 * Returns 1 if a centroid was computed, 0 if the mask had no
 * positive pixels.
 */
static int compute_mask_centroid(const float *mask, int w, int h,
				 float *out_cx, float *out_cy)
{
	double sx = 0.0, sy = 0.0;
	long   cnt = 0;

	for (int y = 0; y < h; y++) {
		const float *row = mask + (size_t)y * w;
		for (int x = 0; x < w; x++) {
			if (row[x] >= 0.0f) {
				sx += x;
				sy += y;
				cnt++;
			}
		}
	}

	if (cnt == 0)
		return 0;

	*out_cx = (float)(sx / (double)cnt);
	*out_cy = (float)(sy / (double)cnt);
	return 1;
}

static int e2e_frame_callback(const struct sam3_video_frame_result *result,
			      void *user_data)
{
	struct centroid_tracker *ct = user_data;
	int frame_idx;

	if (!ct || !result)
		return 0;

	frame_idx = result->frame_idx;
	if (frame_idx < 0 || frame_idx >= E2E_N_FRAMES)
		return 0;

	ct->n_frames_seen++;

	/* Use first object's mask for centroid computation. */
	if (result->n_objects <= 0 || !result->objects ||
	    !result->objects[0].mask ||
	    result->objects[0].mask_w <= 0 ||
	    result->objects[0].mask_h <= 0)
		return 0;

	const float *plane_ptr = result->objects[0].mask;

	float cx, cy;
	if (compute_mask_centroid(plane_ptr, result->objects[0].mask_w,
				  result->objects[0].mask_h, &cx, &cy)) {
		ct->centroids[frame_idx].x = cx;
		ct->centroids[frame_idx].y = cy;
		ct->centroids[frame_idx].valid = 1;
	}

	if (ct->mask_w == 0) {
		ct->mask_w = result->objects[0].mask_w;
		ct->mask_h = result->objects[0].mask_h;
	}

	return 0;
}

/*
 * rmtree - Best-effort recursive remove of @dir.
 *
 * Only recurses one level deep, which matches what this test creates
 * (flat directory of PNGs). Errors are logged but never abort the
 * test — cleanup failures should not mask real regressions.
 */
static void rmtree(const char *dir)
{
	DIR *d = opendir(dir);
	char path[1024];
	struct dirent *ent;

	if (!d) {
		if (rmdir(dir) != 0)
			sam3_log_warn("rmtree: cannot remove '%s'", dir);
		return;
	}

	while ((ent = readdir(d)) != NULL) {
		if (strcmp(ent->d_name, ".") == 0 ||
		    strcmp(ent->d_name, "..") == 0)
			continue;
		int n = snprintf(path, sizeof(path), "%s/%s",
				 dir, ent->d_name);
		if (n < 0 || (size_t)n >= sizeof(path)) {
			sam3_log_warn("rmtree: path too long, skipping '%s'",
				      ent->d_name);
			continue;
		}
		if (unlink(path) != 0)
			sam3_log_warn("rmtree: cannot unlink '%s'", path);
	}
	closedir(d);

	if (rmdir(dir) != 0)
		sam3_log_warn("rmtree: cannot rmdir '%s'", dir);
}

static void test_video_e2e_tracks_moving_square(void)
{
	char tmpdir[] = "/tmp/sam3_e2e_XXXXXX";
	sam3_ctx *ctx = NULL;
	sam3_video_session *session = NULL;
	struct sam3_video_frame_result seed_result;
	struct sam3_point seed_pt;
	struct centroid_tracker ct;
	enum sam3_error err;
	float gt_cx, gt_cy;
	float scale_x, scale_y;
	int model_img_size;

	if (access(SAM3_TEST_MODEL, F_OK) != 0) {
		printf("SKIP test_video_e2e_tracks_moving_square: "
		       "SAM3_TEST_MODEL not found at %s\n", SAM3_TEST_MODEL);
		return;
	}

	if (!mkdtemp(tmpdir)) {
		sam3_log_error("mkdtemp failed for E2E clip dir");
		ASSERT(0);
		return;
	}

	if (generate_moving_square_clip(tmpdir, E2E_N_FRAMES) != 0) {
		sam3_log_error("failed to synthesize moving-square clip");
		rmtree(tmpdir);
		ASSERT(0);
		return;
	}

	ctx = sam3_init();
	ASSERT(ctx != NULL);
	if (!ctx) {
		rmtree(tmpdir);
		return;
	}

	err = sam3_load_model(ctx, SAM3_TEST_MODEL);
	ASSERT_EQ(err, SAM3_OK);
	if (err != SAM3_OK) {
		sam3_free(ctx);
		rmtree(tmpdir);
		return;
	}

	model_img_size = sam3_get_image_size(ctx);
	ASSERT(model_img_size > 0);

	err = sam3_video_start(ctx, tmpdir, &session);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT(session != NULL);
	if (err != SAM3_OK || !session) {
		sam3_free(ctx);
		rmtree(tmpdir);
		return;
	}

	ASSERT_EQ(sam3_video_frame_count(session), E2E_N_FRAMES);

	/*
	 * Seed a foreground point at the ground-truth centre of the
	 * square on frame 0. Prompts are given in the model's input
	 * coordinate space (the PNG is resized from E2E_IMG_SIZE to
	 * model_img_size), so scale accordingly.
	 */
	square_center(0, &gt_cx, &gt_cy);
	scale_x = (float)model_img_size / (float)E2E_IMG_SIZE;
	scale_y = (float)model_img_size / (float)E2E_IMG_SIZE;

	seed_pt.x = gt_cx * scale_x;
	seed_pt.y = gt_cy * scale_y;
	seed_pt.label = 1;
	memset(&seed_result, 0, sizeof(seed_result));

	err = sam3_video_add_points(session, 0, 0, &seed_pt, 1,
				    &seed_result);
	ASSERT_EQ(err, SAM3_OK);
	sam3_video_frame_result_free(&seed_result);

	/* Propagate forward; the callback stashes per-frame centroids. */
	memset(&ct, 0, sizeof(ct));
	err = sam3_video_propagate(session, SAM3_PROPAGATE_FORWARD,
				   e2e_frame_callback, &ct);
	ASSERT_EQ(err, SAM3_OK);

	ASSERT_EQ(ct.n_frames_seen, E2E_N_FRAMES);
	ASSERT(ct.mask_w > 0);
	ASSERT(ct.mask_h > 0);

	/*
	 * Verify per-frame centroids track the ground truth within
	 * E2E_TOLERANCE_PX pixels, comparing in PNG pixel space.
	 * mask -> PNG scale is (E2E_IMG_SIZE / mask_dim) because the
	 * mask covers the resized frame, which itself was resized
	 * from the synthetic PNG.
	 */
	float px_per_mask_x = (float)E2E_IMG_SIZE / (float)ct.mask_w;
	float px_per_mask_y = (float)E2E_IMG_SIZE / (float)ct.mask_h;

	for (int i = 0; i < E2E_N_FRAMES; i++) {
		ASSERT(ct.centroids[i].valid);
		if (!ct.centroids[i].valid)
			continue;

		float cx_png = ct.centroids[i].x * px_per_mask_x;
		float cy_png = ct.centroids[i].y * px_per_mask_y;

		float expected_cx, expected_cy;
		square_center(i, &expected_cx, &expected_cy);

		ASSERT_NEAR(cx_png, expected_cx, E2E_TOLERANCE_PX);
		ASSERT_NEAR(cy_png, expected_cy, E2E_TOLERANCE_PX);
	}

	sam3_video_end(session);
	sam3_free(ctx);
	rmtree(tmpdir);
}

int main(void)
{
	test_video_e2e_tracks_moving_square();
	TEST_REPORT();
}
