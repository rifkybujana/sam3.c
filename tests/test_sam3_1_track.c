/*
 * tests/test_sam3_1_track.c - End-to-end SAM 3.1 video-tracker smoke test.
 *
 * Drives sam3_video_start -> add_points -> propagate(BOTH) against a
 * SAM 3.1 model + a small video clip. Verifies that the tracker_multiplex
 * pipeline wired in phase 2.5 runs without crashing and produces
 * finite, non-trivial masks per frame (not all zeros, not constant).
 *
 * The test is PARITY-NEGATIVE: it does not compare masks to any Python
 * reference. Point-prompt conditioning on SAM 3.1 requires the
 * interactive decoder from sub-project 3 to be correct; this test only
 * confirms that the plumbing (variant dispatch, memory attention,
 * multiplex mask decoder, maskmem backbone) does not blow up.
 *
 * Skips cleanly when models/sam3.1.sam3 or assets/kids.mp4 are absent
 * so a default CI profile without the big model still passes.
 *
 * Key types:  (none; uses the public sam3.h API)
 * Depends on: sam3/sam3.h, sam3/sam3_types.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sam3/sam3.h"
#include "sam3/sam3_types.h"
#include "test_helpers.h"

#ifndef SAM3_SOURCE_DIR
#error "SAM3_SOURCE_DIR must be defined by CMake"
#endif

#define MODEL_PATH SAM3_SOURCE_DIR "/models/sam3.1.sam3"
#define VIDEO_PATH SAM3_SOURCE_DIR "/assets/kids.mp4"

/*
 * The test is a smoke test — it runs the full plumbing but there's no
 * point in iterating all 100 video frames × 2 directions at ~5s/frame.
 * Stop after this many callback invocations.
 */
#define MAX_FRAMES_FOR_TEST 3

struct cb_state {
	int frames_seen;
	int finite_ok;       /* all mask values finite */
	int non_constant_ok; /* at least one frame had varying mask values */
};

static int
frame_cb(const struct sam3_video_frame_result *r, void *ud)
{
	struct cb_state *s = (struct cb_state *)ud;

	s->frames_seen++;

	if (r->n_objects < 1 || !r->objects || !r->objects[0].mask) {
		fprintf(stderr, "frame %d: no mask in result\n", r->frame_idx);
		s->finite_ok = 0;
		return 0;
	}

	int hw = r->objects[0].mask_h * r->objects[0].mask_w;
	if (hw <= 0) {
		fprintf(stderr, "frame %d: bad mask shape %dx%d\n",
			r->frame_idx, r->objects[0].mask_h,
			r->objects[0].mask_w);
		s->finite_ok = 0;
		return 0;
	}

	const float *m = r->objects[0].mask;
	float min_v = m[0];
	float max_v = m[0];
	for (int i = 0; i < hw; i++) {
		if (!isfinite(m[i])) {
			fprintf(stderr, "frame %d: non-finite at %d\n",
				r->frame_idx, i);
			s->finite_ok = 0;
			return 0;
		}
		if (m[i] < min_v) min_v = m[i];
		if (m[i] > max_v) max_v = m[i];
	}
	if (max_v > min_v)
		s->non_constant_ok = 1;

	/* Stop early once we've seen enough frames to validate plumbing. */
	return (s->frames_seen >= MAX_FRAMES_FOR_TEST) ? 1 : 0;
}

int main(void)
{
	if (access(MODEL_PATH, F_OK) != 0 ||
	    access(VIDEO_PATH, F_OK) != 0) {
		printf("SKIP: %s or %s not found (CI default)\n",
		       MODEL_PATH, VIDEO_PATH);
		return 0;
	}

	sam3_ctx *ctx = sam3_init();
	ASSERT_NOT_NULL(ctx);

	ASSERT_EQ(sam3_load_model(ctx, MODEL_PATH), SAM3_OK);

	sam3_video_session *sess = NULL;
	ASSERT_EQ(sam3_video_start(ctx, VIDEO_PATH, &sess), SAM3_OK);
	ASSERT_NOT_NULL(sess);

	/* Add a point prompt on frame 0. The point content is effectively
	 * ignored by the current multiplex pipeline (no interactive decoder yet)
	 * but it still registers the object and runs the tracker. */
	struct sam3_point pt;
	memset(&pt, 0, sizeof(pt));
	pt.x     = 0.5f;
	pt.y     = 0.5f;
	pt.label = 1;

	struct sam3_video_frame_result r0;
	memset(&r0, 0, sizeof(r0));
	ASSERT_EQ(sam3_video_add_points(sess, 0, 0, &pt, 1, &r0), SAM3_OK);
	ASSERT(r0.n_objects == 1);
	ASSERT_NOT_NULL(r0.objects);
	ASSERT_NOT_NULL(r0.objects[0].mask);
	sam3_video_frame_result_free(&r0);

	struct cb_state cbs;
	memset(&cbs, 0, sizeof(cbs));
	cbs.finite_ok = 1;

	ASSERT_EQ(sam3_video_propagate(sess, SAM3_PROPAGATE_BOTH,
				       frame_cb, &cbs), SAM3_OK);

	printf("frames_seen=%d finite_ok=%d non_constant_ok=%d\n",
	       cbs.frames_seen, cbs.finite_ok, cbs.non_constant_ok);

	ASSERT(cbs.frames_seen > 0);
	ASSERT(cbs.finite_ok == 1);
	/*
	 * non_constant_ok is only meaningful once the tracker has picked
	 * up at least one propagation frame: the conditioning frame uses
	 * the no_mem path with learned mask_tokens, which can plausibly
	 * produce a near-constant mask when memory is empty.
	 */
	if (cbs.frames_seen >= 2)
		ASSERT(cbs.non_constant_ok == 1);

	sam3_video_end(sess);
	sam3_free(ctx);

	TEST_REPORT();
}
