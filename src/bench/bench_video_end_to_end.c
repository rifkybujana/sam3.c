/*
 * src/bench/bench_video_end_to_end.c - End-to-end video pipeline benchmark
 *
 * Times a full video_start → add_points → propagate → end cycle on an
 * 8-frame synthetic moving-square clip. Represents the total
 * user-facing latency for a short video with a single point prompt.
 * Mirrors the pattern of the pipeline benchmarks and reuses the
 * clip-synthesis helper declared in bench_video.h.
 *
 * Key types:  sam3_bench_config, sam3_bench_result, sam3_ctx
 * Depends on: bench/bench_video.h, sam3/sam3.h, util/log.h
 * Used by:    cli_bench.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "bench/bench_video.h"
#include "util/log.h"

#define SAM3_BENCH_VIDEO_E2E_FRAMES 8

struct video_e2e_ctx {
	sam3_ctx           *ctx;
	const char         *clip_dir;
	struct sam3_point   seed_pt;
	int                 obj_id;
	int                 frame_idx;
};

static void video_e2e_fn(void *arg)
{
	struct video_e2e_ctx *vc = arg;
	sam3_video_session *session = NULL;
	struct sam3_video_frame_result r;

	if (sam3_video_start(vc->ctx, vc->clip_dir, &session) != SAM3_OK)
		return;

	memset(&r, 0, sizeof(r));
	if (sam3_video_add_points(session, vc->frame_idx, vc->obj_id,
				  &vc->seed_pt, 1, &r) == SAM3_OK) {
		sam3_video_frame_result_free(&r);
		sam3_video_propagate(session, SAM3_PROPAGATE_FORWARD,
				     NULL, NULL);
	} else {
		sam3_video_frame_result_free(&r);
	}

	sam3_video_end(session);
}

int sam3_bench_run_video_end_to_end(const struct sam3_bench_config *cfg,
				    sam3_ctx *ctx,
				    struct sam3_bench_result *results,
				    int max_results)
{
	char tmpdir[] = "/tmp/sam3_bench_ve_XXXXXX";
	struct video_e2e_ctx vc;
	int model_img_size;
	float scale;
	int count = 0;
	int rc;

	if (!cfg || !ctx || !results || max_results <= 0) {
		sam3_log_error("bench_run_video_end_to_end: "
			       "invalid arguments");
		return -1;
	}

	/* Filter gate — bail cheaply if excluded. */
	if (!sam3_bench_filter_match("video_end_to_end", cfg->filter))
		return 0;

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

	if (sam3_bench_generate_clip(tmpdir,
				     SAM3_BENCH_VIDEO_E2E_FRAMES) != 0) {
		sam3_log_error("bench_run_video_end_to_end: failed to "
			       "synthesize clip in '%s'", tmpdir);
		sam3_bench_rmtree(tmpdir);
		return -1;
	}

	/*
	 * Seed point: centre of the square on frame 0, scaled from the
	 * synthetic PNG space into the model's input space.
	 */
	scale = (float)model_img_size / (float)SAM3_BENCH_VIDEO_IMG_SIZE;
	vc.ctx = ctx;
	vc.clip_dir = tmpdir;
	{
		float cx = (float)sam3_bench_bounce_pos(0) +
			   (float)SAM3_BENCH_VIDEO_SQUARE_SIZE * 0.5f;
		vc.seed_pt.x = cx * scale;
		vc.seed_pt.y = cx * scale;
	}
	vc.seed_pt.label = 1;
	vc.obj_id = 0;
	vc.frame_idx = 0;

	if (count < max_results) {
		rc = sam3_bench_run(cfg, "video_end_to_end", "pipeline",
				    video_e2e_fn, &vc,
				    0, 0, &results[count]);
		if (rc != 0) {
			sam3_log_error("video bench: video_end_to_end "
				       "failed");
			count = -1;
			goto cleanup;
		}
		count++;
	}

	sam3_log_info("video benchmarks: video_end_to_end completed");

cleanup:
	sam3_bench_rmtree(tmpdir);
	return count;
}
