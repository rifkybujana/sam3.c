/*
 * src/bench/bench_video_frame.c - Per-frame video tracker benchmark
 *
 * Times a single sam3_tracker_track_frame invocation as exposed through
 * the public video API: sam3_video_reset → sam3_video_add_points →
 * sam3_video_propagate(FORWARD) over a 2-frame synthetic clip. The
 * clip is synthesised once before the timing loop and the session is
 * reused across iterations so that the benchmark isolates the
 * per-frame tracking cost (memory attention + mask decoder) from
 * video-start overhead. Also defines the shared clip-synthesis helpers
 * used by bench_video_end_to_end.c.
 *
 * Key types:  sam3_bench_config, sam3_bench_result, sam3_ctx,
 *             sam3_video_session
 * Depends on: bench/bench_video.h, sam3/sam3.h, util/log.h,
 *             src/util/vendor/stb_image_write.h
 * Used by:    cli_bench.c (through bench_video.h), bench_video_end_to_end.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>

#include "bench/bench_video.h"
#include "util/log.h"

/* stb_image_write: declaration only; implementation lives in
 * src/util/image.c inside the sam3 library, which this benchmark
 * links. */
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

/* ── Deterministic noise generator ─────────────────────────────────── */

int sam3_bench_bounce_pos(int i)
{
	int max_pos = SAM3_BENCH_VIDEO_IMG_SIZE -
		      SAM3_BENCH_VIDEO_SQUARE_SIZE;
	int period  = 2 * max_pos;
	int raw     = i * SAM3_BENCH_VIDEO_SQUARE_STEP;
	int t       = raw % period;
	if (t < 0)
		t += period;
	return (t <= max_pos) ? t : (period - t);
}

/*
 * Simple 32-bit LCG so the noise pattern is reproducible across
 * platforms. Only determinism and coverage of the [100, 156) range
 * matter for the tracker's memory attention.
 */
static uint32_t bench_lcg_state = 1u;

static void bench_lcg_seed(uint32_t seed)
{
	bench_lcg_state = seed ? seed : 1u;
}

static uint32_t bench_lcg_next(void)
{
	bench_lcg_state = bench_lcg_state * 1664525u + 1013904223u;
	return bench_lcg_state;
}

/* ── Shared clip helpers (declared in bench_video.h) ───────────────── */

int sam3_bench_generate_clip(const char *dir, int n)
{
	size_t nbytes;
	uint8_t *buf;
	char path[1024];
	int rc = 0;

	if (!dir || n <= 0) {
		sam3_log_error("sam3_bench_generate_clip: invalid arguments");
		return 1;
	}

	nbytes = (size_t)SAM3_BENCH_VIDEO_IMG_SIZE *
		 SAM3_BENCH_VIDEO_IMG_SIZE * 3;
	buf = malloc(nbytes);
	if (!buf) {
		sam3_log_error("sam3_bench_generate_clip: out of memory "
			       "for %dx%d frame buffer",
			       SAM3_BENCH_VIDEO_IMG_SIZE,
			       SAM3_BENCH_VIDEO_IMG_SIZE);
		return 1;
	}

	bench_lcg_seed(0xC0FFEEu);

	for (int i = 0; i < n; i++) {
		/* Gray noise background in [100, 156). */
		for (size_t k = 0; k < nbytes; k++)
			buf[k] = (uint8_t)(100u + (bench_lcg_next() % 56u));

		int x0 = sam3_bench_bounce_pos(i);
		int y0 = x0;
		int x1 = x0 + SAM3_BENCH_VIDEO_SQUARE_SIZE;
		int y1 = y0 + SAM3_BENCH_VIDEO_SQUARE_SIZE;

		for (int y = y0; y < y1; y++) {
			uint8_t *row = buf +
				       ((size_t)y *
					SAM3_BENCH_VIDEO_IMG_SIZE + x0) * 3;
			size_t run = (size_t)(x1 - x0) * 3;
			memset(row, 255, run);
		}

		int nb = snprintf(path, sizeof(path),
				  "%s/frame_%04d.png", dir, i);
		if (nb < 0 || (size_t)nb >= sizeof(path)) {
			sam3_log_error("bench clip: frame path too long");
			rc = 1;
			break;
		}

		int ok = stbi_write_png(path,
					SAM3_BENCH_VIDEO_IMG_SIZE,
					SAM3_BENCH_VIDEO_IMG_SIZE,
					3, buf,
					SAM3_BENCH_VIDEO_IMG_SIZE * 3);
		if (!ok) {
			sam3_log_error("bench clip: failed to write '%s'",
				       path);
			rc = 1;
			break;
		}
	}

	free(buf);
	return rc;
}

void sam3_bench_rmtree(const char *dir)
{
	DIR *d;
	char path[1024];
	struct dirent *ent;

	if (!dir)
		return;

	d = opendir(dir);
	if (!d) {
		if (rmdir(dir) != 0)
			sam3_log_warn("bench rmtree: cannot remove '%s'",
				      dir);
		return;
	}

	while ((ent = readdir(d)) != NULL) {
		if (strcmp(ent->d_name, ".") == 0 ||
		    strcmp(ent->d_name, "..") == 0)
			continue;
		int n = snprintf(path, sizeof(path), "%s/%s",
				 dir, ent->d_name);
		if (n < 0 || (size_t)n >= sizeof(path)) {
			sam3_log_warn("bench rmtree: path too long, "
				      "skipping '%s'", ent->d_name);
			continue;
		}
		if (unlink(path) != 0)
			sam3_log_warn("bench rmtree: cannot unlink '%s'",
				      path);
	}
	closedir(d);

	if (rmdir(dir) != 0)
		sam3_log_warn("bench rmtree: cannot rmdir '%s'", dir);
}

/* ── Per-frame benchmark ──────────────────────────────────────────── */

struct video_frame_ctx {
	sam3_video_session *session;
	struct sam3_point   seed_pt;
	int                 obj_id;    /* 0 */
	int                 frame_idx; /* 0 */
};

static void video_frame_fn(void *arg)
{
	struct video_frame_ctx *vc = arg;
	struct sam3_video_frame_result r;
	memset(&r, 0, sizeof(r));

	if (sam3_video_reset(vc->session) != SAM3_OK)
		return;
	if (sam3_video_add_points(vc->session, vc->frame_idx, vc->obj_id,
				  &vc->seed_pt, 1, &r) != SAM3_OK) {
		sam3_video_frame_result_free(&r);
		return;
	}
	sam3_video_frame_result_free(&r);
	sam3_video_propagate(vc->session, SAM3_PROPAGATE_FORWARD,
			     NULL, NULL);
}

int sam3_bench_run_video_frame(const struct sam3_bench_config *cfg,
			       sam3_ctx *ctx,
			       struct sam3_bench_result *results,
			       int max_results)
{
	char tmpdir[] = "/tmp/sam3_bench_vf_XXXXXX";
	sam3_video_session *session = NULL;
	struct video_frame_ctx vc;
	int model_img_size;
	float scale;
	int count = 0;
	int rc;

	if (!cfg || !ctx || !results || max_results <= 0) {
		sam3_log_error("bench_run_video_frame: invalid arguments");
		return -1;
	}

	/* Filter gate — bail cheaply if excluded. */
	if (!sam3_bench_filter_match("video_per_frame", cfg->filter))
		return 0;

	model_img_size = sam3_get_image_size(ctx);
	if (model_img_size <= 0) {
		sam3_log_error("bench_run_video_frame: no model loaded "
			       "(image size = %d)", model_img_size);
		return -1;
	}

	if (!mkdtemp(tmpdir)) {
		sam3_log_error("bench_run_video_frame: mkdtemp failed "
			       "for clip dir");
		return -1;
	}

	if (sam3_bench_generate_clip(tmpdir, 2) != 0) {
		sam3_log_error("bench_run_video_frame: failed to "
			       "synthesize clip in '%s'", tmpdir);
		sam3_bench_rmtree(tmpdir);
		return -1;
	}

	if (sam3_video_start(ctx, tmpdir, &session) != SAM3_OK || !session) {
		sam3_log_error("bench_run_video_frame: sam3_video_start "
			       "failed");
		sam3_bench_rmtree(tmpdir);
		return -1;
	}

	/*
	 * Seed point: centre of the square on frame 0, scaled from the
	 * synthetic PNG space (SAM3_BENCH_VIDEO_IMG_SIZE) into the
	 * model's input space. Matches the pattern used in
	 * tests/test_video_e2e.c.
	 */
	scale = (float)model_img_size / (float)SAM3_BENCH_VIDEO_IMG_SIZE;
	/*
	 * Square at frame 0 lives at (bounce_pos(0), bounce_pos(0)).
	 * Seed point is its centre in PNG pixel space, scaled to model space.
	 */
	{
		float cx = (float)sam3_bench_bounce_pos(0) +
			   (float)SAM3_BENCH_VIDEO_SQUARE_SIZE * 0.5f;
		vc.seed_pt.x = cx * scale;
		vc.seed_pt.y = cx * scale;
	}
	vc.seed_pt.label = 1;
	vc.session = session;
	vc.obj_id = 0;
	vc.frame_idx = 0;

	if (count < max_results) {
		rc = sam3_bench_run(cfg, "video_per_frame", "pipeline",
				    video_frame_fn, &vc,
				    0, 0, &results[count]);
		if (rc != 0) {
			sam3_log_error("video bench: video_per_frame "
				       "failed");
			count = -1;
			goto cleanup;
		}
		count++;
	}

	sam3_log_info("video benchmarks: video_per_frame completed");

cleanup:
	sam3_video_end(session);
	sam3_bench_rmtree(tmpdir);
	return count;
}
