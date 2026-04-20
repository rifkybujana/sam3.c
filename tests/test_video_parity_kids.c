/*
 * tests/test_video_parity_kids.c - End-to-end parity vs Python on kids.mp4
 *
 * Two variants selected at build time via
 * SAM3_PARITY_VARIANT_{sam3,sam3_1}:
 *
 *   sam3    - Legacy scaffold (Sam3VideoPredictor text prompt). Still
 *             a stub: PNG compare is not wired. Skips at runtime when
 *             fixtures absent.
 *   sam3_1  - C-seeded, Python-propagated parity:
 *               1. sam3_init + load sam3.1 + video_start(kids.mp4)
 *               2. add_points(center, frame=0) -> C frame-0 mask
 *               3. Load seed_mask.png; IoU(C_f0, seed) warn < 0.9,
 *                                      fail  < 0.5
 *               4. propagate(FORWARD, callback stops at 3 frames)
 *               5. For n in {1,2,3}: IoU(C_fn, frames/frame_000n_obj_1.png)
 *                  >= 0.75
 *
 * Gated on SAM3_BUILD_PARITY_TESTS=ON. Self-skips on missing fixtures or
 * model so the CI default profile stays clean.
 *
 * Key types: sam3_video_session
 * Depends on: sam3/sam3.h, test_helpers.h, test_helpers_png.h
 * Used by:    CTest (opt-in)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "test_helpers.h"
#include "test_helpers_png.h"
#include "sam3/sam3.h"

#ifndef SAM3_SOURCE_DIR
#error "SAM3_SOURCE_DIR must be defined (via CMake)"
#endif
#ifndef SAM3_TEST_MODEL
#error "SAM3_TEST_MODEL must be defined (via -DSAM3_TEST_MODEL=<path>)"
#endif

#if !defined(SAM3_PARITY_VARIANT_sam3) && !defined(SAM3_PARITY_VARIANT_sam3_1)
#error "SAM3_PARITY_VARIANT_{sam3,sam3_1} must be set (CMake cache)"
#endif

#ifdef SAM3_PARITY_VARIANT_sam3

/* --- SAM 3 variant: original scaffold, unchanged --- */

static int fixture_dir_exists(void)
{
	struct stat st;
	const char *path =
		SAM3_SOURCE_DIR "/tests/fixtures/video_kids/frames";
	return (stat(path, &st) == 0) && S_ISDIR(st.st_mode);
}

int main(void)
{
	if (!fixture_dir_exists()) {
		fprintf(stderr,
			"SKIP: fixtures absent. See "
			"tests/fixtures/video_kids/README.md\n");
		return 0;
	}
	if (SAM3_TEST_MODEL[0] == '\0') {
		fprintf(stderr, "SKIP: SAM3_TEST_MODEL is empty\n");
		return 0;
	}
	fprintf(stderr,
		"NOTE: test_video_parity_kids sam3 variant is a scaffold. "
		"See tests/fixtures/video_kids/README.md.\n");
	return 0;
}

#else /* SAM3_PARITY_VARIANT_sam3_1 */

/* --- SAM 3.1 variant: C-seeded parity run --- */

#define MODEL_PATH    SAM3_SOURCE_DIR "/models/sam3.1.sam3"
#define VIDEO_PATH    SAM3_SOURCE_DIR "/assets/kids.mp4"
#define FIXTURE_DIR   SAM3_SOURCE_DIR "/tests/fixtures/video_kids/sam3_1"
#define SEED_PATH     FIXTURE_DIR "/seed_mask.png"
#define FRAME_1_PATH  FIXTURE_DIR "/frames/frame_0001_obj_1.png"

#define N_PROP_FRAMES 3
#define IOU_FRAME_THRESH 0.75f
#define IOU_SEED_WARN    0.90f
#define IOU_SEED_FAIL    0.50f

static float
mask_iou_logits_vs_png(const float *logits, int h, int w,
		       const uint8_t *png, int png_h, int png_w)
{
	if (h != png_h || w != png_w) {
		fprintf(stderr,
			"iou: dim mismatch logits %dx%d vs png %dx%d\n",
			h, w, png_h, png_w);
		return -1.0f;
	}
	size_t inter = 0, uni = 0;
	for (int i = 0; i < h * w; i++) {
		int a = (logits[i] > 0.0f);
		int b = (png[i] > 127);
		inter += (size_t)(a & b);
		uni   += (size_t)(a | b);
	}
	if (uni == 0)
		return 1.0f;   /* both empty — treat as matching */
	return (float)inter / (float)uni;
}

struct cb_state {
	int        frames_seen;
	int        passed;      /* 0 on any per-frame IoU failure */
	const char *fixture_dir;
};

static int
frame_cb(const struct sam3_video_frame_result *r, void *ud)
{
	struct cb_state *s = (struct cb_state *)ud;

	/*
	 * Frame 0 is the cond frame (already sanity-checked vs seed in
	 * main() before propagation started). The Python generator
	 * skips frame 0 too, so there is no frame_0000_*.png to compare
	 * against. Don't count it toward frames_seen either.
	 */
	if (r->frame_idx == 0)
		return 0;

	s->frames_seen++;

	if (r->n_objects < 1 || !r->objects || !r->objects[0].mask) {
		fprintf(stderr,
			"parity: frame %d missing mask\n", r->frame_idx);
		s->passed = 0;
		return 1;
	}

	char path[1024];
	snprintf(path, sizeof(path), "%s/frames/frame_%04d_obj_1.png",
		 s->fixture_dir, r->frame_idx);

	int ph = 0, pw = 0;
	uint8_t *png = load_png_grayscale(path, &ph, &pw);
	if (!png) {
		fprintf(stderr,
			"parity: frame %d fixture %s missing / unreadable\n",
			r->frame_idx, path);
		s->passed = 0;
		return 1;
	}

	float iou = mask_iou_logits_vs_png(
		r->objects[0].mask, r->objects[0].mask_h,
		r->objects[0].mask_w, png, ph, pw);
	free(png);
	if (iou < 0.0f) {
		s->passed = 0;
		return 1;
	}
	fprintf(stderr, "parity: frame %d IoU=%.4f\n",
		r->frame_idx, (double)iou);
	if (iou < IOU_FRAME_THRESH) {
		fprintf(stderr,
			"parity: frame %d IoU %.4f < %.4f — FAIL\n",
			r->frame_idx, (double)iou, (double)IOU_FRAME_THRESH);
		s->passed = 0;
	}
	return (s->frames_seen >= N_PROP_FRAMES) ? 1 : 0;
}

int main(void)
{
	if (access(MODEL_PATH, F_OK) != 0 ||
	    access(VIDEO_PATH, F_OK) != 0 ||
	    access(SEED_PATH,  F_OK) != 0 ||
	    access(FRAME_1_PATH, F_OK) != 0) {
		fprintf(stderr,
			"SKIP: model/video/seed/frame-fixtures missing. "
			"See %s/README.md\n", FIXTURE_DIR);
		return 0;
	}

	sam3_ctx *ctx = sam3_init();
	ASSERT_NOT_NULL(ctx);
	ASSERT_EQ(sam3_load_model(ctx, MODEL_PATH), SAM3_OK);

	sam3_video_session *sess = NULL;
	ASSERT_EQ(sam3_video_start(ctx, VIDEO_PATH, &sess), SAM3_OK);
	ASSERT_NOT_NULL(sess);

	struct sam3_point pt;
	memset(&pt, 0, sizeof(pt));
	pt.x = 0.5f;
	pt.y = 0.5f;
	pt.label = 1;

	struct sam3_video_frame_result r0;
	memset(&r0, 0, sizeof(r0));
	ASSERT_EQ(sam3_video_add_points(sess, 0, 1, &pt, 1, &r0), SAM3_OK);
	ASSERT(r0.n_objects == 1);
	ASSERT_NOT_NULL(r0.objects);
	ASSERT_NOT_NULL(r0.objects[0].mask);

	/* Frame-0 sanity: compare C output to committed seed */
	int sh = 0, sw = 0;
	uint8_t *seed = load_png_grayscale(SEED_PATH, &sh, &sw);
	ASSERT_NOT_NULL(seed);
	float seed_iou = mask_iou_logits_vs_png(
		r0.objects[0].mask, r0.objects[0].mask_h,
		r0.objects[0].mask_w, seed, sh, sw);
	free(seed);
	ASSERT(seed_iou >= 0.0f);
	fprintf(stderr, "parity: frame-0 vs seed IoU=%.4f\n",
		(double)seed_iou);
	if (seed_iou < IOU_SEED_FAIL) {
		fprintf(stderr,
			"parity: frame-0 IoU %.4f < %.4f — fixture is stale, "
			"regenerate via sam3_1_dump_seed + "
			"gen_video_parity_fixtures.py\n",
			(double)seed_iou, (double)IOU_SEED_FAIL);
		ASSERT(0);
	} else if (seed_iou < IOU_SEED_WARN) {
		fprintf(stderr,
			"parity: WARN frame-0 IoU %.4f < %.4f "
			"(numerical drift tolerated)\n",
			(double)seed_iou, (double)IOU_SEED_WARN);
	}
	sam3_video_frame_result_free(&r0);

	struct cb_state cbs = {0};
	cbs.passed = 1;
	cbs.fixture_dir = FIXTURE_DIR;

	ASSERT_EQ(sam3_video_propagate(sess, SAM3_PROPAGATE_FORWARD,
				       frame_cb, &cbs), SAM3_OK);
	fprintf(stderr, "parity: frames_seen=%d passed=%d\n",
		cbs.frames_seen, cbs.passed);
	ASSERT(cbs.frames_seen >= N_PROP_FRAMES);
	ASSERT(cbs.passed == 1);

	sam3_video_end(sess);
	sam3_free(ctx);
	TEST_REPORT();
}

#endif /* SAM3_PARITY_VARIANT_* */
