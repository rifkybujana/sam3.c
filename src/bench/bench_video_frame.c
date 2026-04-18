/*
 * src/bench/bench_video_frame.c - Per-frame video tracker benchmark
 *
 * Drives a static case-table sweep of sam3_tracker_track_frame via the
 * public video API: sam3_video_reset → sam3_video_add_points →
 * sam3_video_propagate(direction). For each unique n_frames value in
 * the case table, one clip directory and one video session are
 * synthesised at setup time and reused across all matching cases, so
 * the timed function isolates per-frame tracking cost (memory
 * attention + mask decoder) from video-start overhead. Also defines
 * the shared clip-synthesis helpers used by bench_video_end_to_end.c.
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

/* ── Multi-object seeding helper ──────────────────────────────────── */

/*
 * seed_n_objects - Place @n distinct obj_id points inside the frame-@seed
 * square of a bench synthetic clip.
 *
 * @s:     Active video session.
 * @n:     Number of objects to seed (>= 1).
 * @seed:  Frame index passed through to sam3_video_add_points.
 * @scale: PNG-pixel → model-pixel scale factor for this session.
 *
 * Points are spread evenly inside the 32×32 white square at
 * (bounce_pos(seed), bounce_pos(seed)) so each distinct obj_id
 * gets a slightly different point. Returns SAM3_OK or the first
 * error from sam3_video_add_points.
 */
static enum sam3_error seed_n_objects(sam3_video_session *s, int n,
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

/* ── Per-frame case-table driver ──────────────────────────────────── */

/*
 * Case table — each row is one benchmark emitted by this function.
 * Order matters only for readability; bench result ordering mirrors it.
 */
static const struct sam3_bench_video_case per_frame_cases[] = {
	{ 8,  1, 0,  SAM3_PROPAGATE_FORWARD, "8f_1obj_fwd"   },
	{ 32, 1, 0,  SAM3_PROPAGATE_FORWARD, "32f_1obj_fwd"  },
	{ 64, 1, 0,  SAM3_PROPAGATE_FORWARD, "64f_1obj_fwd"  },
	{ 32, 2, 0,  SAM3_PROPAGATE_FORWARD, "32f_2obj_fwd"  },
	{ 32, 4, 0,  SAM3_PROPAGATE_FORWARD, "32f_4obj_fwd"  },
	{ 32, 8, 0,  SAM3_PROPAGATE_FORWARD, "32f_8obj_fwd"  },
	{ 32, 1, 16, SAM3_PROPAGATE_BOTH,    "32f_1obj_both" },
};

struct video_frame_ctx {
	sam3_video_session                  *session;
	const struct sam3_bench_video_case  *c;
	float                                scale;
};

static void video_frame_fn(void *arg)
{
	struct video_frame_ctx *vc = arg;

	if (sam3_video_reset(vc->session) != SAM3_OK)
		return;

	if (seed_n_objects(vc->session, vc->c->n_objects,
			   vc->c->seed_frame, vc->scale) != SAM3_OK)
		return;

	sam3_video_propagate(vc->session, vc->c->direction, NULL, NULL);
}

int sam3_bench_run_video_frame(const struct sam3_bench_config *cfg,
			       sam3_ctx *ctx,
			       struct sam3_bench_result *results,
			       int max_results)
{
	/*
	 * Iterate unique n_frames in the outer loop; for each value,
	 * open a single session and run all cases that match it, then
	 * tear the session down before moving to the next n_frames.
	 * Keeping one session alive at a time bounds peak memory to
	 * one frame-cache budget rather than N × budget — important
	 * for large backbones (e.g. Hiera) where three concurrent 64-
	 * frame feature caches exhaust the tensor arena and produce
	 * silent "clone frame X failed" errors mid-case.
	 */
	int model_img_size;
	float scale;
	int count = 0;
	size_t n_cases = sizeof(per_frame_cases) /
			 sizeof(per_frame_cases[0]);

	if (!cfg || !ctx || !results || max_results <= 0) {
		sam3_log_error("bench_run_video_frame: invalid arguments");
		return -1;
	}

	/* Collect unique n_frames values (keep small — N <= 4 in practice). */
	int unique_nf[8];
	int n_unique = 0;
	for (size_t i = 0; i < n_cases; i++) {
		int nf = per_frame_cases[i].n_frames;
		int seen = 0;
		for (int j = 0; j < n_unique; j++) {
			if (unique_nf[j] == nf) {
				seen = 1;
				break;
			}
		}
		if (!seen && n_unique < (int)(sizeof(unique_nf) /
					      sizeof(unique_nf[0]))) {
			unique_nf[n_unique++] = nf;
		}
	}

	for (int u = 0; u < n_unique; u++) {
		if (unique_nf[u] > SAM3_BENCH_VIDEO_CLIP_MAX_FRAMES) {
			sam3_log_error("bench_run_video_frame: n_frames %d "
				       "exceeds CLIP_MAX_FRAMES %d",
				       unique_nf[u],
				       SAM3_BENCH_VIDEO_CLIP_MAX_FRAMES);
			return -1;
		}
	}

	model_img_size = sam3_get_image_size(ctx);
	if (model_img_size <= 0) {
		sam3_log_error("bench_run_video_frame: no model loaded "
			       "(image size = %d)", model_img_size);
		return -1;
	}

	scale = (float)model_img_size / (float)SAM3_BENCH_VIDEO_IMG_SIZE;

	for (int u = 0; u < n_unique; u++) {
		char tmpdir[32];
		sam3_video_session *session = NULL;
		int nf_u = unique_nf[u];

		snprintf(tmpdir, sizeof(tmpdir),
			 "/tmp/sam3_bench_vf_XXXXXX");
		if (!mkdtemp(tmpdir)) {
			sam3_log_error("bench_run_video_frame: mkdtemp "
				       "failed for %d-frame clip", nf_u);
			return -1;
		}

		if (sam3_bench_generate_clip(tmpdir, nf_u) != 0) {
			sam3_log_error("bench_run_video_frame: generate_clip "
				       "failed for %d frames", nf_u);
			sam3_bench_rmtree(tmpdir);
			return -1;
		}

		if (sam3_video_start(ctx, tmpdir, &session) != SAM3_OK ||
		    !session) {
			sam3_log_error("bench_run_video_frame: video_start "
				       "failed for %d frames", nf_u);
			sam3_bench_rmtree(tmpdir);
			return -1;
		}

		/* Run every case whose n_frames matches this session. */
		for (size_t i = 0; i < n_cases && count < max_results; i++) {
			char name[128];
			struct video_frame_ctx vc;
			int rc;

			if (per_frame_cases[i].n_frames != nf_u)
				continue;

			snprintf(name, sizeof(name), "video_per_frame_%s",
				 per_frame_cases[i].label);

			if (!sam3_bench_filter_match(name, cfg->filter))
				continue;

			vc.session = session;
			vc.c       = &per_frame_cases[i];
			vc.scale   = scale;

			rc = sam3_bench_run(cfg, name, "pipeline",
					    video_frame_fn, &vc,
					    0, 0, &results[count]);
			if (rc != 0) {
				sam3_log_error("video bench: %s failed", name);
				sam3_video_end(session);
				sam3_bench_rmtree(tmpdir);
				return -1;
			}
			count++;
		}

		sam3_video_end(session);
		sam3_bench_rmtree(tmpdir);
	}

	sam3_log_info("video benchmarks: per-frame driver completed "
		      "(%d cases, %d clip sizes)", count, n_unique);

	return count;
}
