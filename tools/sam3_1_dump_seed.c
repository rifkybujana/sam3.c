/*
 * tools/sam3_1_dump_seed.c - Dump the C frame-0 mask as a grayscale PNG.
 *
 * Drives sam3_init -> sam3_load_model -> sam3_video_start ->
 * sam3_video_add_points on frame 0 of a video with a single
 * point prompt, binarizes the resulting logits (>0 -> 255, else 0),
 * and writes the mask as an 8-bit grayscale PNG.
 *
 * Used exclusively to seed tests/fixtures/video_kids/sam3_1/seed_mask.png
 * so the Python reference propagator (tools/gen_video_parity_fixtures.py
 * --variant sam3.1) can feed the same seed into
 * Sam3MultiplexTracking.add_new_mask. See
 * docs/superpowers/plans/2026-04-20-sam3-1-parity-fixture.md.
 *
 * Key types:  (uses public sam3.h API)
 * Depends on: sam3/sam3.h, src/util/vendor/stb_image_write.h (impl in sam3 lib)
 * Used by:    manual fixture regeneration only (not CI, not CTest)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sam3/sam3.h"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#endif
#include "util/vendor/stb_image_write.h"
#ifdef __clang__
#pragma clang diagnostic pop
#endif

static void usage(const char *argv0)
{
	fprintf(stderr,
		"Usage: %s --model PATH --video PATH "
		"--point X,Y,LABEL --out PATH\n\n"
		"  X, Y are normalized [0,1] click coordinates.\n"
		"  LABEL is 1 (positive) or 0 (negative).\n",
		argv0);
}

static int parse_point(const char *s, float *x, float *y, int *label)
{
	if (sscanf(s, "%f,%f,%d", x, y, label) != 3)
		return -1;
	if (*x < 0.0f || *x > 1.0f || *y < 0.0f || *y > 1.0f)
		return -1;
	if (*label != 0 && *label != 1)
		return -1;
	return 0;
}

int main(int argc, char **argv)
{
	const char *model_path = NULL;
	const char *video_path = NULL;
	const char *out_path   = NULL;
	float px = 0.5f, py = 0.5f;
	int   plabel = 1;
	int   have_point = 0;

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--model") && i + 1 < argc) {
			model_path = argv[++i];
		} else if (!strcmp(argv[i], "--video") && i + 1 < argc) {
			video_path = argv[++i];
		} else if (!strcmp(argv[i], "--out") && i + 1 < argc) {
			out_path = argv[++i];
		} else if (!strcmp(argv[i], "--point") && i + 1 < argc) {
			if (parse_point(argv[++i], &px, &py, &plabel) != 0) {
				usage(argv[0]);
				return 1;
			}
			have_point = 1;
		} else {
			usage(argv[0]);
			return 1;
		}
	}
	if (!model_path || !video_path || !out_path || !have_point) {
		usage(argv[0]);
		return 1;
	}

	sam3_ctx *ctx = sam3_init();
	if (!ctx) {
		fprintf(stderr, "sam3_init failed\n");
		return 2;
	}
	if (sam3_load_model(ctx, model_path) != SAM3_OK) {
		fprintf(stderr, "sam3_load_model(%s) failed\n", model_path);
		sam3_free(ctx);
		return 2;
	}

	sam3_video_session *sess = NULL;
	if (sam3_video_start(ctx, video_path, &sess) != SAM3_OK) {
		fprintf(stderr, "sam3_video_start(%s) failed\n", video_path);
		sam3_free(ctx);
		return 2;
	}

	struct sam3_point pt;
	memset(&pt, 0, sizeof(pt));
	pt.x = px;
	pt.y = py;
	pt.label = plabel;

	struct sam3_video_frame_result r;
	memset(&r, 0, sizeof(r));
	if (sam3_video_add_points(sess, 0, 1, &pt, 1, &r) != SAM3_OK) {
		fprintf(stderr, "sam3_video_add_points failed\n");
		sam3_video_end(sess);
		sam3_free(ctx);
		return 3;
	}
	if (r.n_objects < 1 || !r.objects || !r.objects[0].mask) {
		fprintf(stderr, "add_points returned no mask\n");
		sam3_video_frame_result_free(&r);
		sam3_video_end(sess);
		sam3_free(ctx);
		return 3;
	}

	int H = r.objects[0].mask_h;
	int W = r.objects[0].mask_w;
	const float *logits = r.objects[0].mask;
	uint8_t *bin = malloc((size_t)H * (size_t)W);
	if (!bin) {
		fprintf(stderr, "malloc failed\n");
		sam3_video_frame_result_free(&r);
		sam3_video_end(sess);
		sam3_free(ctx);
		return 4;
	}
	for (int i = 0; i < H * W; i++)
		bin[i] = (logits[i] > 0.0f) ? (uint8_t)255 : (uint8_t)0;

	int ok = stbi_write_png(out_path, W, H, 1, bin, W);
	free(bin);
	sam3_video_frame_result_free(&r);
	sam3_video_end(sess);
	sam3_free(ctx);

	if (!ok) {
		fprintf(stderr, "stbi_write_png(%s) failed\n", out_path);
		return 4;
	}
	fprintf(stderr, "wrote seed mask %dx%d to %s\n", W, H, out_path);
	return 0;
}
