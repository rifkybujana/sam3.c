/*
 * src/bench/bench_video_end_to_end.c - End-to-end video pipeline benchmark
 *
 * Drives a table of cases that each run a full video_start → seed →
 * propagate → end cycle on a synthetic moving-square clip. Cases vary
 * clip length and object count. One clip is synthesised per run and
 * sized to the largest n_frames in the case table; each case reuses
 * the directory. Times the total user-facing latency including
 * session init, feature caching, and teardown.
 *
 * Key types:  sam3_bench_config, sam3_bench_result, sam3_ctx,
 *             sam3_bench_video_case
 * Depends on: bench/bench_video.h, sam3/sam3.h, util/log.h
 * Used by:    cli_bench.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "bench/bench_video.h"
#include "util/log.h"

/* Case table — end-to-end uses four cases (see spec). */
static const struct sam3_bench_video_case e2e_cases[] = {
	{ 8,  1, 0, SAM3_PROPAGATE_FORWARD, "8f_1obj_fwd"  },
	{ 32, 1, 0, SAM3_PROPAGATE_FORWARD, "32f_1obj_fwd" },
	{ 64, 1, 0, SAM3_PROPAGATE_FORWARD, "64f_1obj_fwd" },
	{ 32, 4, 0, SAM3_PROPAGATE_FORWARD, "32f_4obj_fwd" },
};

struct video_e2e_ctx {
	sam3_ctx                             *ctx;
	const char                           *clip_dir;
	const struct sam3_bench_video_case   *c;
	float                                 scale;
};

/*
 * Place n objects in one add_points batch on the conditioning frame.
 *
 * Duplicates seed_n_objects in bench_video_frame.c because each .c
 * file wants a file-static helper that does not leak symbols. Kept
 * in sync with the per-frame version by spec.
 */
static enum sam3_error e2e_seed_n_objects(sam3_video_session *s, int n,
					  int seed, float scale)
{
	float square_x0 = (float)sam3_bench_bounce_pos(seed);
	float square_y0 = square_x0;
	float step      = (float)SAM3_BENCH_VIDEO_SQUARE_SIZE / (float)(n + 1);

	for (int k = 0; k < n; k++) {
		struct sam3_point pt;
		struct sam3_video_frame_result r;
		float px = square_x0 + step * (float)(k + 1);
		float py = square_y0 + step * (float)(k + 1);

		pt.x     = px * scale;
		pt.y     = py * scale;
		pt.label = 1;

		memset(&r, 0, sizeof(r));
		enum sam3_error err = sam3_video_add_points(
			s, seed, /* obj_id */ k, &pt, 1, &r);
		sam3_video_frame_result_free(&r);
		if (err != SAM3_OK)
			return err;
	}
	return SAM3_OK;
}

static void video_e2e_fn(void *arg)
{
	struct video_e2e_ctx *vc = arg;
	sam3_video_session *session = NULL;

	if (sam3_video_start(vc->ctx, vc->clip_dir, &session) != SAM3_OK)
		return;

	if (e2e_seed_n_objects(session, vc->c->n_objects,
			       vc->c->seed_frame, vc->scale) == SAM3_OK) {
		sam3_video_propagate(session, vc->c->direction, NULL, NULL);
	}

	sam3_video_end(session);
}

int sam3_bench_run_video_end_to_end(const struct sam3_bench_config *cfg,
				    sam3_ctx *ctx,
				    struct sam3_bench_result *results,
				    int max_results)
{
	char tmpdir[] = "/tmp/sam3_bench_ve_XXXXXX";
	int model_img_size;
	float scale;
	int count = 0;
	size_t n_cases = sizeof(e2e_cases) / sizeof(e2e_cases[0]);
	int max_frames = 0;

	if (!cfg || !ctx || !results || max_results <= 0) {
		sam3_log_error("bench_run_video_end_to_end: "
			       "invalid arguments");
		return -1;
	}

	for (size_t i = 0; i < n_cases; i++) {
		if (e2e_cases[i].n_frames > max_frames)
			max_frames = e2e_cases[i].n_frames;
	}
	if (max_frames > SAM3_BENCH_VIDEO_CLIP_MAX_FRAMES) {
		sam3_log_error("bench_run_video_end_to_end: n_frames %d "
			       "exceeds CLIP_MAX_FRAMES %d",
			       max_frames, SAM3_BENCH_VIDEO_CLIP_MAX_FRAMES);
		return -1;
	}

	model_img_size = sam3_get_image_size(ctx);
	if (model_img_size <= 0) {
		sam3_log_error("bench_run_video_end_to_end: no model "
			       "loaded (image size = %d)", model_img_size);
		return -1;
	}

	if (!mkdtemp(tmpdir)) {
		sam3_log_error("bench_run_video_end_to_end: mkdtemp "
			       "failed for clip dir");
		return -1;
	}

	if (sam3_bench_generate_clip(tmpdir, max_frames) != 0) {
		sam3_log_error("bench_run_video_end_to_end: failed to "
			       "synthesize clip in '%s'", tmpdir);
		sam3_bench_rmtree(tmpdir);
		return -1;
	}

	scale = (float)model_img_size / (float)SAM3_BENCH_VIDEO_IMG_SIZE;

	for (size_t i = 0; i < n_cases && count < max_results; i++) {
		char name[128];
		struct video_e2e_ctx vc;
		int rc;

		snprintf(name, sizeof(name), "video_e2e_%s",
			 e2e_cases[i].label);

		if (!sam3_bench_filter_match(name, cfg->filter))
			continue;

		vc.ctx      = ctx;
		vc.clip_dir = tmpdir;
		vc.c        = &e2e_cases[i];
		vc.scale    = scale;

		rc = sam3_bench_run(cfg, name, "pipeline",
				    video_e2e_fn, &vc,
				    0, 0, &results[count]);
		if (rc != 0) {
			sam3_log_error("video bench: %s failed", name);
			count = -1;
			goto cleanup;
		}
		count++;
	}

	sam3_log_info("video benchmarks: end-to-end driver completed "
		      "(%d cases)", count);

cleanup:
	sam3_bench_rmtree(tmpdir);
	return count;
}
