/*
 * src/bench/bench_video.h - Video tracker benchmark suite
 *
 * Declares entry points for video tracker benchmarks: per-frame tracking
 * cost (bench_video_frame) and full pipeline cost (bench_video_end_to_end).
 * Both require a loaded model context. Also declares a shared helper for
 * synthesising a moving-square clip on disk, used by both benchmarks.
 *
 * Key types:  sam3_bench_config, sam3_bench_result, sam3_ctx
 * Depends on: bench/bench.h, sam3/sam3.h
 * Used by:    cli_bench.c, bench_video_frame.c, bench_video_end_to_end.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_BENCH_VIDEO_H
#define SAM3_BENCH_VIDEO_H

#include "bench/bench.h"
#include "sam3/sam3.h"

/* Shared clip parameters. Kept small so the benchmark runs fast. */
#define SAM3_BENCH_VIDEO_IMG_SIZE    256
#define SAM3_BENCH_VIDEO_SQUARE_SIZE  32
#define SAM3_BENCH_VIDEO_SQUARE_START 100
#define SAM3_BENCH_VIDEO_SQUARE_STEP   8

/* Max frames in any case; sets the canvas size for the generator. */
#define SAM3_BENCH_VIDEO_CLIP_MAX_FRAMES 128

/*
 * sam3_bench_bounce_pos - Triangle-wave position for the moving square.
 *
 * @i: Frame index (>= 0).
 *
 * Returns a position in [0, SAM3_BENCH_VIDEO_IMG_SIZE -
 * SAM3_BENCH_VIDEO_SQUARE_SIZE]. The sequence reflects off each edge
 * every (SAM3_BENCH_VIDEO_IMG_SIZE - SAM3_BENCH_VIDEO_SQUARE_SIZE) /
 * SAM3_BENCH_VIDEO_SQUARE_STEP frames, so the square never writes OOB
 * regardless of how many frames are rendered.
 */
int sam3_bench_bounce_pos(int i);

/*
 * bench_video_case - One parameterised video benchmark case.
 *
 * @n_frames:   Total frames the clip directory contains (<=
 *              SAM3_BENCH_VIDEO_CLIP_MAX_FRAMES).
 * @n_objects:  Number of simultaneously-tracked objects seeded on
 *              @seed_frame.
 * @seed_frame: Frame index where add_points is called (0 for FORWARD,
 *              middle frame for BOTH).
 * @direction:  SAM3_PROPAGATE_FORWARD or SAM3_PROPAGATE_BOTH.
 * @label:      Suffix appended to the benchmark case name.
 */
struct bench_video_case {
	int         n_frames;
	int         n_objects;
	int         seed_frame;
	int         direction;
	const char *label;
};

/*
 * sam3_bench_generate_clip - Synthesise a moving-square clip on disk.
 *
 * @dir: Directory (must already exist and be writable).
 * @n:   Number of frames to write.
 *
 * Writes PNG files frame_0000.png .. frame_{n-1}.png into @dir. Each
 * frame is a SAM3_BENCH_VIDEO_IMG_SIZE^2 RGB image with gray noise
 * background and a white SAM3_BENCH_VIDEO_SQUARE_SIZE^2 square that
 * slides diagonally by SAM3_BENCH_VIDEO_SQUARE_STEP px per frame.
 *
 * Returns 0 on success, non-zero on any I/O failure.
 */
int sam3_bench_generate_clip(const char *dir, int n);

/*
 * sam3_bench_rmtree - Best-effort flat recursive remove of @dir.
 *
 * Only recurses one level deep, matching what sam3_bench_generate_clip
 * produces (a flat directory of PNGs). Failures are logged but never
 * fatal.
 */
void sam3_bench_rmtree(const char *dir);

/*
 * sam3_bench_run_video_frame - Run per-frame tracking benchmark.
 *
 * @cfg:         Benchmark configuration.
 * @ctx:         Initialised sam3 context with a loaded model.
 * @results:     Array to fill with per-case results.
 * @max_results: Capacity of the results array.
 *
 * Sets up a 2-frame synthetic clip once (outside the timing loop).
 * Each timed iteration: sam3_video_reset, sam3_video_add_points on
 * frame 0 (populates the memory bank with one conditioning entry),
 * sam3_video_propagate forward (tracks exactly one non-prompted
 * frame through sam3_tracker_track_frame). The encoded frame
 * features are cached in the session and reused across iterations.
 *
 * Returns the number of results written, or -1 on error.
 */
int sam3_bench_run_video_frame(const struct sam3_bench_config *cfg,
			       sam3_ctx *ctx,
			       struct sam3_bench_result *results,
			       int max_results);

/*
 * sam3_bench_run_video_end_to_end - Run end-to-end video pipeline benchmark.
 *
 * @cfg:         Benchmark configuration.
 * @ctx:         Initialised sam3 context with a loaded model.
 * @results:     Array to fill with per-case results.
 * @max_results: Capacity of the results array.
 *
 * Synthesises an 8-frame clip once (outside the timing loop). Each
 * timed iteration runs the full pipeline: sam3_video_start,
 * sam3_video_add_points on frame 0, sam3_video_propagate forward,
 * sam3_video_end. Measures total user-facing latency for a short
 * video.
 *
 * Returns the number of results written, or -1 on error.
 */
int sam3_bench_run_video_end_to_end(const struct sam3_bench_config *cfg,
				    sam3_ctx *ctx,
				    struct sam3_bench_result *results,
				    int max_results);

#endif /* SAM3_BENCH_VIDEO_H */
