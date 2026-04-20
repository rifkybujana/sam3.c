# Video Benchmark Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-case video benchmarks with a parameterised case-table (clip length × object count × direction), add committed baseline JSON files for regression detection, and document the new cases in `BENCHMARK.md`.

**Architecture:** Pure refactor of `src/bench/bench_video_frame.c` and `src/bench/bench_video_end_to_end.c` from single-case functions into table-driven loops. A small pure-function `sam3_bench_bounce_pos` encapsulates a triangle-wave reflection so the synthetic square stays in-bounds on clips of any length. A new `seed_n_objects` helper seeds N distinct `obj_id` points in one call. Baseline JSON files live under `benchmarks/baselines/` and are refreshed by `scripts/refresh_baselines.sh`. `BENCHMARK.md` gets two new sections with `_tbd_` placeholders that are populated by running the bench against each model.

**Tech Stack:** C11, CMake, CTest, stb_image_write (already vendored), existing `sam3_bench_run` harness, existing cJSON round-trip.

**Spec:** `docs/superpowers/specs/2026-04-18-video-benchmark-design.md`

---

## Task 1: Add bounce helper and case struct to `bench_video.h`

**Files:**
- Modify: `src/bench/bench_video.h`

- [ ] **Step 1: Write the failing test**

Create `tests/test_bench_video.c`:

```c
/*
 * tests/test_bench_video.c - Unit tests for video benchmark helpers
 *
 * Tests the pure helpers in bench/bench_video.h: the triangle-wave
 * bounce position function (regression test for the clip-generator
 * OOB bug) and the filter-glob matching on the case-table naming
 * scheme. Pure functions only; no model required.
 *
 * Key types:  (none)
 * Depends on: test_helpers.h, bench/bench.h, bench/bench_video.h
 * Used by:    CTest (when SAM3_BENCH is ON)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "bench/bench.h"
#include "bench/bench_video.h"

#include <string.h>

/* --- test_bounce_stays_in_bounds --- */

static void test_bounce_stays_in_bounds(void)
{
	int max_pos = SAM3_BENCH_VIDEO_IMG_SIZE - SAM3_BENCH_VIDEO_SQUARE_SIZE;

	for (int i = 0; i < 256; i++) {
		int p = sam3_bench_bounce_pos(i);
		ASSERT(p >= 0);
		ASSERT(p <= max_pos);
	}
}

/* --- test_bounce_starts_at_zero --- */

static void test_bounce_starts_at_zero(void)
{
	ASSERT_EQ(sam3_bench_bounce_pos(0), 0);
}

int main(void)
{
	test_bounce_stays_in_bounds();
	test_bounce_starts_at_zero();
	TEST_REPORT();
}
```

- [ ] **Step 2: Run test to verify it fails to compile**

Run: `cd build-release && cmake .. -DCMAKE_BUILD_TYPE=Release -DSAM3_BENCH=ON && make test_bench_video 2>&1 | tail -20`

Expected: compile error — `sam3_bench_bounce_pos` not declared, `SAM3_BENCH_VIDEO_IMG_SIZE` already defined (OK).

- [ ] **Step 3: Extend `bench_video.h`**

Add these additions to `src/bench/bench_video.h` (insert after the existing `#define SAM3_BENCH_VIDEO_SQUARE_STEP` line and before the function decls):

```c
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
```

- [ ] **Step 4: Add `sam3_bench_bounce_pos` implementation**

Add to the top of `src/bench/bench_video_frame.c`, after the `stb_image_write` pragma block, before `bench_lcg_seed`:

```c
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
```

- [ ] **Step 5: Add CMake wiring so the test compiles**

In `CMakeLists.txt`, inside the existing tests block (around the `foreach(test_src ${TEST_SOURCES})` loop), add this after the existing `if(SAM3_BENCH AND TARGET test_bench)` block:

```cmake
# test_bench_video needs the bench harness library
if(SAM3_BENCH AND TARGET test_bench_video)
	target_link_libraries(test_bench_video sam3_bench)
endif()

# Skip bench-dependent tests when SAM3_BENCH is OFF
if(NOT SAM3_BENCH)
	list(REMOVE_ITEM TEST_SOURCES
		${CMAKE_SOURCE_DIR}/tests/test_bench.c
		${CMAKE_SOURCE_DIR}/tests/test_bench_video.c)
endif()
```

Note the `REMOVE_ITEM` goes **before** the `foreach` — move it up. Place it right after the other `list(REMOVE_ITEM TEST_SOURCES ...)` calls (the existing SAM3_FIXTURE_TESTS / SAM3_E2E_TESTS / SAM3_BUILD_PARITY_TESTS guards).

Final form of that section:

```cmake
	file(GLOB TEST_SOURCES "tests/test_*.c")
	if(NOT SAM3_FIXTURE_TESTS)
		list(REMOVE_ITEM TEST_SOURCES
			${CMAKE_SOURCE_DIR}/tests/test_tracker_fixtures.c)
	endif()
	if(NOT SAM3_E2E_TESTS)
		list(REMOVE_ITEM TEST_SOURCES
			${CMAKE_SOURCE_DIR}/tests/test_video_e2e.c)
	endif()
	if(NOT SAM3_BUILD_PARITY_TESTS)
		list(REMOVE_ITEM TEST_SOURCES
			${CMAKE_SOURCE_DIR}/tests/test_video_parity_kids.c)
	endif()
	if(NOT SAM3_BENCH)
		list(REMOVE_ITEM TEST_SOURCES
			${CMAKE_SOURCE_DIR}/tests/test_bench.c
			${CMAKE_SOURCE_DIR}/tests/test_bench_video.c)
	endif()
	foreach(test_src ${TEST_SOURCES})
```

And add the target-link block next to the existing `test_bench` one:

```cmake
	if(SAM3_BENCH AND TARGET test_bench_video)
		target_link_libraries(test_bench_video sam3_bench)
	endif()
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd build-release && cmake .. -DCMAKE_BUILD_TYPE=Release -DSAM3_BENCH=ON && make test_bench_video && ./test_bench_video`

Expected: `2 tests, 0 failures`.

- [ ] **Step 7: Commit**

```bash
git add src/bench/bench_video.h src/bench/bench_video_frame.c \
        tests/test_bench_video.c CMakeLists.txt
git commit -m "bench/video: add bounce-position helper and case struct

Pure triangle-wave helper sam3_bench_bounce_pos keeps the moving
square inside the 256² canvas for any frame count, removing the
OOB write that limited the old generator to <20 frames. Adds a
bench_video_case struct and CLIP_MAX_FRAMES constant in preparation
for the case-table driver. test_bench_video.c covers the bounce
invariant (position stays in [0, max_pos] for 256 frames).
"
```

---

## Task 2: Replace `sam3_bench_generate_clip` with bounce-safe version

**Files:**
- Modify: `src/bench/bench_video_frame.c:85-153` (the existing `sam3_bench_generate_clip`)

- [ ] **Step 1: Add a failing file-existence test**

Append to `tests/test_bench_video.c` (before `main`):

```c
/* --- test_generate_clip_128_frames --- */

#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

static void test_generate_clip_128_frames(void)
{
	char dir[] = "/tmp/sam3_tbvb_XXXXXX";
	ASSERT(mkdtemp(dir) != NULL);

	int rc = sam3_bench_generate_clip(dir, 128);
	ASSERT_EQ(rc, 0);

	/* Spot-check: first, middle, last frames exist and are nonempty. */
	int idxs[] = {0, 64, 127};
	for (int k = 0; k < 3; k++) {
		char path[1024];
		snprintf(path, sizeof(path), "%s/frame_%04d.png",
			 dir, idxs[k]);
		struct stat st;
		ASSERT_EQ(stat(path, &st), 0);
		ASSERT(st.st_size >= 100);
	}

	sam3_bench_rmtree(dir);
}
```

And call it in `main`:

```c
int main(void)
{
	test_bounce_stays_in_bounds();
	test_bounce_starts_at_zero();
	test_generate_clip_128_frames();
	TEST_REPORT();
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd build-release && make test_bench_video && ./test_bench_video`

Expected: passes the first two tests, fails or crashes on `test_generate_clip_128_frames` because the current generator writes OOB at i≈19 (ASan will report a stack/heap buffer overflow, or the write loop will segfault).

If running without `-fsanitize=address`, the test may "pass" because OOB writes over random memory can silently succeed. Force-check by running a Debug build with ASan:

```bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DSAM3_BENCH=ON && \
    make test_bench_video && ./test_bench_video
```

Expected: ASan error or test failure.

- [ ] **Step 3: Rewrite the generator loop body**

In `src/bench/bench_video_frame.c`, replace the loop body inside `sam3_bench_generate_clip` (the `for (int i = 0; i < n; i++)` block):

Old:

```c
	for (int i = 0; i < n; i++) {
		/* Gray noise background in [100, 156). */
		for (size_t k = 0; k < nbytes; k++)
			buf[k] = (uint8_t)(100u + (bench_lcg_next() % 56u));

		int x0 = SAM3_BENCH_VIDEO_SQUARE_START +
			 i * SAM3_BENCH_VIDEO_SQUARE_STEP;
		int y0 = SAM3_BENCH_VIDEO_SQUARE_START +
			 i * SAM3_BENCH_VIDEO_SQUARE_STEP;
		int x1 = x0 + SAM3_BENCH_VIDEO_SQUARE_SIZE;
		int y1 = y0 + SAM3_BENCH_VIDEO_SQUARE_SIZE;

		for (int y = y0; y < y1; y++) {
			uint8_t *row = buf +
				       ((size_t)y *
					SAM3_BENCH_VIDEO_IMG_SIZE + x0) * 3;
			size_t run = (size_t)(x1 - x0) * 3;
			memset(row, 255, run);
		}
		/* ... (snprintf and stbi_write_png unchanged) ... */
	}
```

New:

```c
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
		/* snprintf + stbi_write_png block unchanged below */

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
```

Key change: `x0 = sam3_bench_bounce_pos(i)` replaces `SAM3_BENCH_VIDEO_SQUARE_START + i * SAM3_BENCH_VIDEO_SQUARE_STEP`. Note: `SAM3_BENCH_VIDEO_SQUARE_START` is no longer used inside this function but remains in the header because the consumers still compute seed-point coordinates from the frame-0 position (which is now `sam3_bench_bounce_pos(0) = 0`, not 100). This changes seed-point semantics — see Task 3 for the fix-up.

- [ ] **Step 4: Run test to verify it passes**

```bash
cd build && make test_bench_video && ./test_bench_video
cd ../build-release && make test_bench_video && ./test_bench_video
```

Expected both: `3 tests, 0 failures`.

- [ ] **Step 5: Commit**

```bash
git add src/bench/bench_video_frame.c tests/test_bench_video.c
git commit -m "bench/video: replace diagonal walk with bounce-safe clip

Generator now uses sam3_bench_bounce_pos so the moving square stays
inside the 256² canvas for clip lengths up to
SAM3_BENCH_VIDEO_CLIP_MAX_FRAMES (128). Fixes the OOB write that
limited the old generator to ~19 frames. test_generate_clip_128_frames
covers the new upper bound.
"
```

---

## Task 3: Update existing seed-point coordinates for new frame-0 position

**Files:**
- Modify: `src/bench/bench_video_frame.c:275-278`
- Modify: `src/bench/bench_video_end_to_end.c:104-110`

The bounce fix moves the frame-0 square from `(100, 100)` to `(0, 0)`. The existing seed-point calculation hard-codes the old position and will now land on background noise instead of the square. Update both files.

- [ ] **Step 1: Fix `bench_video_frame.c` seed-point calculation**

In `src/bench/bench_video_frame.c`, replace:

```c
	scale = (float)model_img_size / (float)SAM3_BENCH_VIDEO_IMG_SIZE;
	vc.seed_pt.x = ((float)SAM3_BENCH_VIDEO_SQUARE_START +
			(float)SAM3_BENCH_VIDEO_SQUARE_SIZE * 0.5f) * scale;
	vc.seed_pt.y = vc.seed_pt.x;
```

With:

```c
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
```

- [ ] **Step 2: Fix `bench_video_end_to_end.c` seed-point calculation**

In `src/bench/bench_video_end_to_end.c`, replace:

```c
	scale = (float)model_img_size / (float)SAM3_BENCH_VIDEO_IMG_SIZE;
	vc.ctx = ctx;
	vc.clip_dir = tmpdir;
	vc.seed_pt.x = ((float)SAM3_BENCH_VIDEO_SQUARE_START +
			(float)SAM3_BENCH_VIDEO_SQUARE_SIZE * 0.5f) * scale;
	vc.seed_pt.y = vc.seed_pt.x;
	vc.seed_pt.label = 1;
```

With:

```c
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
```

- [ ] **Step 3: Build and run the full bench test suite**

```bash
cd build-release && make test_bench_video && ./test_bench_video
```

Expected: `3 tests, 0 failures`.

- [ ] **Step 4: Commit**

```bash
git add src/bench/bench_video_frame.c src/bench/bench_video_end_to_end.c
git commit -m "bench/video: update seed-point coords for bounced clip

Frame 0 now places the square at bounce_pos(0) = 0, not (100, 100).
Both per-frame and end-to-end benchmarks must seed at the new
coordinate or they land on background noise. Uses a single call to
sam3_bench_bounce_pos so future changes to the bounce formula stay
in one place.
"
```

---

## Task 4: Add `seed_n_objects` helper

**Files:**
- Modify: `src/bench/bench_video_frame.c`

- [ ] **Step 1: Add helper function**

Append this static helper to `src/bench/bench_video_frame.c` after `sam3_bench_rmtree`, before the per-frame benchmark section:

```c
/* --- Multi-object seeding helper ─────── --- */

/*
 * seed_n_objects - Place @n distinct obj_id points inside the frame-@seed
 * square of a bench synthetic clip.
 *
 * @s:    Active video session.
 * @n:    Number of objects to seed (>= 1).
 * @seed: Frame index passed through to sam3_video_add_points.
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
```

- [ ] **Step 2: Verify it compiles**

```bash
cd build-release && make sam3_bench 2>&1 | tail -5
```

Expected: no errors. (Unused-function warning is acceptable until Task 5 wires it into the driver.)

- [ ] **Step 3: Commit**

```bash
git add src/bench/bench_video_frame.c
git commit -m "bench/video: add seed_n_objects multi-object helper

Places N distinct obj_id points inside the frame-seed white square
at spread positions (centre + k*offset, k=0..n-1). Used by the
upcoming case-table driver to seed multi-object tracking workloads
without per-case boilerplate.
"
```

---

## Task 5: Replace `sam3_bench_run_video_frame` body with case-table driver

**Files:**
- Modify: `src/bench/bench_video_frame.c:220-303` (the existing `sam3_bench_run_video_frame`)

- [ ] **Step 1: Define case table and driver**

Replace the existing `struct video_frame_ctx`, `video_frame_fn`, and `sam3_bench_run_video_frame` with:

```c
/* --- Per-frame case-table driver ─────── --- */

/*
 * Case table — each row is one benchmark emitted by this function.
 * Order matters only for readability; bench result ordering mirrors it.
 */
static const struct bench_video_case per_frame_cases[] = {
	{ 8,  1, 0,  SAM3_PROPAGATE_FORWARD, "8f_1obj_fwd"   },
	{ 32, 1, 0,  SAM3_PROPAGATE_FORWARD, "32f_1obj_fwd"  },
	{ 64, 1, 0,  SAM3_PROPAGATE_FORWARD, "64f_1obj_fwd"  },
	{ 32, 2, 0,  SAM3_PROPAGATE_FORWARD, "32f_2obj_fwd"  },
	{ 32, 4, 0,  SAM3_PROPAGATE_FORWARD, "32f_4obj_fwd"  },
	{ 32, 8, 0,  SAM3_PROPAGATE_FORWARD, "32f_8obj_fwd"  },
	{ 32, 1, 16, SAM3_PROPAGATE_BOTH,    "32f_1obj_both" },
};

struct video_frame_ctx {
	sam3_video_session             *session;
	const struct bench_video_case  *c;
	float                           scale;
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
	char tmpdir[] = "/tmp/sam3_bench_vf_XXXXXX";
	sam3_video_session *session = NULL;
	int model_img_size;
	float scale;
	int count = 0;
	size_t n_cases = sizeof(per_frame_cases) /
			 sizeof(per_frame_cases[0]);
	/* Max frames across all cases — only the largest clip is synthesised. */
	int max_frames = 0;

	if (!cfg || !ctx || !results || max_results <= 0) {
		sam3_log_error("bench_run_video_frame: invalid arguments");
		return -1;
	}

	for (size_t i = 0; i < n_cases; i++) {
		if (per_frame_cases[i].n_frames > max_frames)
			max_frames = per_frame_cases[i].n_frames;
	}
	if (max_frames > SAM3_BENCH_VIDEO_CLIP_MAX_FRAMES) {
		sam3_log_error("bench_run_video_frame: n_frames %d exceeds "
			       "CLIP_MAX_FRAMES %d",
			       max_frames, SAM3_BENCH_VIDEO_CLIP_MAX_FRAMES);
		return -1;
	}

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

	if (sam3_bench_generate_clip(tmpdir, max_frames) != 0) {
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

	scale = (float)model_img_size / (float)SAM3_BENCH_VIDEO_IMG_SIZE;

	for (size_t i = 0; i < n_cases && count < max_results; i++) {
		char name[128];
		struct video_frame_ctx vc;
		int rc;

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
			count = -1;
			goto cleanup;
		}
		count++;
	}

	sam3_log_info("video benchmarks: per-frame driver completed "
		      "(%d cases)", count);

cleanup:
	sam3_video_end(session);
	sam3_bench_rmtree(tmpdir);
	return count;
}
```

- [ ] **Step 2: Build to verify no compile errors**

```bash
cd build-release && make sam3_cli 2>&1 | tail -5
```

Expected: clean build.

- [ ] **Step 3: Add a filter test to `test_bench_video.c`**

Append before `main`:

```c
/* --- test_filter_glob_matches_case_names --- */

static void test_filter_glob_matches_case_names(void)
{
	/* 4obj filter matches only the 4obj cases. */
	ASSERT(sam3_bench_filter_match(
		"video_per_frame_32f_4obj_fwd", "*_4obj_*"));
	ASSERT(sam3_bench_filter_match(
		"video_e2e_32f_4obj_fwd", "*_4obj_*"));
	ASSERT(!sam3_bench_filter_match(
		"video_per_frame_32f_1obj_fwd", "*_4obj_*"));

	/* video_* matches every new case. */
	ASSERT(sam3_bench_filter_match(
		"video_per_frame_8f_1obj_fwd", "video_*"));
	ASSERT(sam3_bench_filter_match(
		"video_e2e_64f_1obj_fwd", "video_*"));

	/* Unrelated kernel cases do not match. */
	ASSERT(!sam3_bench_filter_match(
		"matmul_f32_1024x1024", "video_*"));
}
```

And add to `main`:

```c
	test_filter_glob_matches_case_names();
```

- [ ] **Step 4: Run tests**

```bash
cd build-release && make test_bench_video && ./test_bench_video
```

Expected: `4 tests, 0 failures`.

- [ ] **Step 5: Commit**

```bash
git add src/bench/bench_video_frame.c tests/test_bench_video.c
git commit -m "bench/video: drive per-frame bench from a case table

Replaces the single-case hardcoded flow with a table of seven cases
covering clip length (8/32/64), object count (1/2/4/8), and
propagation direction (FORWARD/BOTH). One clip is synthesised for
the largest n_frames and reused across all cases; each case resets
the session, seeds n_objects distinct tracks, and propagates.
Filter-glob matching is applied per case so --filter video_* or
*_4obj_* select subsets as expected.
"
```

---

## Task 6: Replace `sam3_bench_run_video_end_to_end` body with case-table driver

**Files:**
- Modify: `src/bench/bench_video_end_to_end.c` (replace the whole body)

- [ ] **Step 1: Rewrite the file**

Full replacement for `src/bench/bench_video_end_to_end.c` (keep the existing copyright/doc header; replace from the first `#define` down):

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "bench/bench_video.h"
#include "util/log.h"

/* Case table — end-to-end uses four cases (see spec). */
static const struct bench_video_case e2e_cases[] = {
	{ 8,  1, 0, SAM3_PROPAGATE_FORWARD, "8f_1obj_fwd"  },
	{ 32, 1, 0, SAM3_PROPAGATE_FORWARD, "32f_1obj_fwd" },
	{ 64, 1, 0, SAM3_PROPAGATE_FORWARD, "64f_1obj_fwd" },
	{ 32, 4, 0, SAM3_PROPAGATE_FORWARD, "32f_4obj_fwd" },
};

struct video_e2e_ctx {
	sam3_ctx                      *ctx;
	const char                    *clip_dir;
	const struct bench_video_case *c;
	float                          scale;
};

/*
 * Place n objects in one add_points batch on the conditioning frame.
 *
 * Note: this helper duplicates seed_n_objects in bench_video_frame.c
 * because each .c file wants a static that does not leak symbols.
 * Kept in sync with the per-frame version by spec.
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
```

- [ ] **Step 2: Update the file doc header**

At the top of `src/bench/bench_video_end_to_end.c`, replace the description with:

```c
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
 *             bench_video_case
 * Depends on: bench/bench_video.h, sam3/sam3.h, util/log.h
 * Used by:    cli_bench.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */
```

- [ ] **Step 3: Build and verify**

```bash
cd build-release && make sam3_cli test_bench_video 2>&1 | tail -5
./test_bench_video
```

Expected: clean build, `4 tests, 0 failures`.

- [ ] **Step 4: Commit**

```bash
git add src/bench/bench_video_end_to_end.c
git commit -m "bench/video: drive end-to-end bench from a case table

Replaces the single-case 8-frame run with a table of four cases
covering clip length (8/32/64) and a multi-object datapoint
(32 frames, 4 objects). Each case runs the full video_start →
seed → propagate → end cycle on a shared synthetic clip.
"
```

---

## Task 7: Add baseline-refresh script

**Files:**
- Create: `scripts/refresh_baselines.sh`

- [ ] **Step 1: Create the script**

Write `scripts/refresh_baselines.sh`:

```bash
#!/usr/bin/env bash
#
# scripts/refresh_baselines.sh - Regenerate per-model baseline JSON files.
#
# Builds sam3_cli in Release mode and runs `bench all` against each
# available model variant under models/, writing the result JSON to
# benchmarks/baselines/<variant>.json. Missing models are skipped with
# a warning so partial regenerations are OK.
#
# Usage: scripts/refresh_baselines.sh [BUILD_DIR]
#   BUILD_DIR defaults to build-release.
#
# Copyright (c) 2026 Rifky Bujana Bisri
# SPDX-License-Identifier: MIT

set -euo pipefail

BUILD="${1:-${BUILD:-build-release}}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Configuring $BUILD (Release, SAM3_BENCH=ON)"
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DSAM3_BENCH=ON

echo "==> Building sam3_cli"
cmake --build "$BUILD" --target sam3_cli -j

mkdir -p benchmarks/baselines

for variant in efficient tinyvit hiera; do
    case "$variant" in
        efficient) model="models/efficient.sam3" ;;
        tinyvit)   model="models/tinyvit_l.sam3" ;;
        hiera)     model="models/sam3.sam3" ;;
    esac

    if [ ! -f "$model" ]; then
        echo "skip: $model not found"
        continue
    fi

    out="benchmarks/baselines/${variant}.json"
    echo "==> Running bench all for $variant -> $out"
    "$BUILD/sam3_cli" bench all \
        --model "$model" --backend metal \
        --output "$out"
done

echo "==> Done"
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x scripts/refresh_baselines.sh
```

- [ ] **Step 3: Lint-check with ShellCheck if available, otherwise smoke-test the header**

Run either:

```bash
which shellcheck && shellcheck scripts/refresh_baselines.sh || \
    bash -n scripts/refresh_baselines.sh
```

Expected: no errors (or ShellCheck clean).

- [ ] **Step 4: Verify dry behaviour when no models present**

```bash
mkdir -p /tmp/sam3_bench_dry && cd /tmp/sam3_bench_dry
cp -r "$OLDPWD"/{scripts,benchmarks,src,include,tools,tests,CMakeLists.txt} . 2>/dev/null || true
# Run the script, expecting it to build then print "skip:" for each variant.
# (Requires cmake, may take a couple of minutes for the build step.)
```

Optional — if short on time, skip this smoke step. The build step alone is a ~2-minute investment.

- [ ] **Step 5: Commit**

```bash
cd /Users/rbisri/Documents/sam3
git add scripts/refresh_baselines.sh
git commit -m "scripts: add refresh_baselines.sh

One-command regeneration of benchmarks/baselines/<variant>.json for
efficient / tinyvit / hiera model variants. Builds sam3_cli in
Release, runs 'bench all' per available model, skips any model whose
.sam3 file is not on disk.
"
```

---

## Task 8: Add placeholder/validated baseline JSON files

**Files:**
- Create: `benchmarks/baselines/efficient.json`
- Create: `benchmarks/baselines/tinyvit.json`
- Create: `benchmarks/baselines/hiera.json`

These start as *real* baselines populated by Task 11 (which runs on hardware with model weights). To unblock the regression-flow wiring before that, commit minimal-but-valid JSON skeletons that `sam3_bench_read_json` accepts.

- [ ] **Step 1: Write skeleton JSON files**

Create `benchmarks/baselines/efficient.json`:

```json
{
  "version": 1,
  "env": {
    "chip": "placeholder",
    "os": "placeholder",
    "cpu_cores": 0,
    "gpu_cores": 0,
    "backend": "metal",
    "commit": "placeholder",
    "timestamp": "1970-01-01T00:00:00Z",
    "model_variant": "efficientvit"
  },
  "config": {
    "warmup_iters": 5,
    "timed_iters": 50,
    "statistical": false,
    "threshold_pct": 5.0
  },
  "results": []
}
```

Copy the same content to `benchmarks/baselines/tinyvit.json` (change `model_variant` to `"tinyvit_l"`) and `benchmarks/baselines/hiera.json` (change `model_variant` to `"hiera_large"`).

- [ ] **Step 2: Verify `sam3_bench_read_json` accepts the skeletons**

Build `sam3_cli` (if not already), then run `bench all --compare` against a skeleton and expect it to succeed with "No regressions detected" plus three "NEW" rows per case (since `results` is empty, every current case is new).

If no model is available, just confirm the JSON parses via:

```bash
python3 -c "import json; json.load(open('benchmarks/baselines/efficient.json'))"
python3 -c "import json; json.load(open('benchmarks/baselines/tinyvit.json'))"
python3 -c "import json; json.load(open('benchmarks/baselines/hiera.json'))"
```

Expected: no output, exit code 0.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/baselines/efficient.json \
        benchmarks/baselines/tinyvit.json \
        benchmarks/baselines/hiera.json
git commit -m "bench/baselines: add skeleton JSON files for 3 variants

Empty but valid baseline files for efficient / tinyvit / hiera so
'sam3_cli bench all --compare' has a file to read. Populated with
real numbers by scripts/refresh_baselines.sh once hardware runs are
available. All current cases will register as NEW until then.
"
```

---

## Task 9: Add Video sections to `BENCHMARK.md`

**Files:**
- Modify: `BENCHMARK.md` (insert after the "End-to-End Segmentation" section, before "Peak Throughput Summary")

- [ ] **Step 1: Insert new sections**

Open `BENCHMARK.md`, locate the line `## Peak Throughput Summary`, and insert *before* it:

```markdown
## Video: Per-Frame Tracking Cost

Measures one tracker step (memory attention + mask decoder + minor
IO) inside a `reset → seed → propagate` loop on a synthetic moving-
square clip. `mean_ms` in the result table is for the full loop;
per-frame cost below is `mean_ms / (n_frames - 1)` to amortise the
reset/seed overhead. The memory bank saturates after ~6–8 tracked
frames, so the 32f and 64f columns measure steady-state cost.

### Clip-length scaling (1 object, FORWARD)

| Model | 8f (ms/frame) | 32f (ms/frame) | 64f (ms/frame) |
|-------|--------------:|---------------:|---------------:|
| EfficientViT (efficient.sam3) | _tbd_ | _tbd_ | _tbd_ |
| TinyViT-L (tinyvit_l.sam3)    | _tbd_ | _tbd_ | _tbd_ |
| Hiera-Large (sam3.sam3)       | _tbd_ | _tbd_ | _tbd_ |

Per-frame cost should stay roughly flat between 32f and 64f once the
memory bank saturates — a linear climb would indicate the bank is
growing unbounded.

### Multi-object scaling (32 frames, FORWARD)

| Model | 1 obj | 2 obj | 4 obj | 8 obj |
|-------|------:|------:|------:|------:|
| EfficientViT | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| TinyViT-L    | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| Hiera-Large  | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

Image encoder cost is fixed per frame; mask decoder and memory
attention run once per tracked object. Linear scaling in N objects
with a shared encoder term is the expected pattern.

### Propagation direction (32 frames, 1 object)

| Model | FORWARD | BOTH (fwd+bwd) |
|-------|--------:|---------------:|
| EfficientViT | _tbd_ | _tbd_ |
| TinyViT-L    | _tbd_ | _tbd_ |
| Hiera-Large  | _tbd_ | _tbd_ |

The BOTH case seeds at the middle frame so each pass does equal work.
Expected ratio is ~1.5–2× forward — encoded features are reused
across both passes, so the reverse direction mostly pays for its own
memory attention and mask decode.

## Video: End-to-End Clip Latency

Full user-facing latency for `sam3_video_start → add_points →
propagate → sam3_video_end` on the synthetic clip. Includes session
init, feature caching, propagation, and teardown.

### End-to-end latency (1 object, FORWARD)

| Model | 8f (ms) | 32f (ms) | 64f (ms) | Effective FPS (64f) |
|-------|--------:|---------:|---------:|--------------------:|
| EfficientViT | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| TinyViT-L    | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| Hiera-Large  | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

### Multi-object end-to-end (32 frames, FORWARD)

| Model | 1 obj | 4 obj |
|-------|------:|------:|
| EfficientViT | _tbd_ | _tbd_ |
| TinyViT-L    | _tbd_ | _tbd_ |
| Hiera-Large  | _tbd_ | _tbd_ |

Effective FPS is the clip length divided by the total wall-clock
time — the interactive-annotation ceiling for that model.

```

- [ ] **Step 2: Add video-filter example to "Running the Benchmark"**

In the same file, locate the "Running the Benchmark" section and add this block after the last example:

```markdown
# Only the video cases, with results written to JSON
./sam3_cli bench all --model ../models/efficient.sam3 --backend metal \
    --filter "video_*" --output video.json
```

- [ ] **Step 3: Verify markdown renders cleanly**

```bash
wc -l BENCHMARK.md
head -n 5 BENCHMARK.md
grep -c "^## " BENCHMARK.md
```

Expected: file length grew, `##` header count increased by 2 (Video: Per-Frame Tracking Cost, Video: End-to-End Clip Latency).

- [ ] **Step 4: Commit**

```bash
git add BENCHMARK.md
git commit -m "docs: add Video Tracking sections to BENCHMARK.md

Two new sections with five tables covering per-frame cost (clip-
length, multi-object, direction) and end-to-end clip latency.
Numbers are placeholder (_tbd_) until the bench is run against each
model variant and scripts/refresh_baselines.sh produces real data.
"
```

---

## Task 10: Generate real baseline numbers and populate `BENCHMARK.md`

This task requires the actual `.sam3` model files under `models/` and a Metal-capable Mac. If any of those are unavailable, skip to Task 11 and note in the commit that baselines remain placeholders.

- [ ] **Step 1: Run the refresh script**

```bash
./scripts/refresh_baselines.sh
```

Expected runtime: ~10 min per model (EfficientViT), ~30 min per model (Hiera-Large). Total ~1 hr on an M4.

- [ ] **Step 2: Verify each baseline file has the new cases**

```bash
for v in efficient tinyvit hiera; do
    echo "=== $v ==="
    python3 -c "
import json, sys
d = json.load(open(f'benchmarks/baselines/$v.json'))
names = [r['name'] for r in d['results']]
print(f'total: {len(names)}')
print('video_per_frame:', sum(1 for n in names if n.startswith('video_per_frame_')))
print('video_e2e:', sum(1 for n in names if n.startswith('video_e2e_')))
"
done
```

Expected: each variant shows 7 per-frame + 4 e2e = 11 video cases (plus the existing non-video cases).

- [ ] **Step 3: Fill in `BENCHMARK.md` `_tbd_` entries**

For each model × (clip length, objects, direction) triple, read `mean_ms` from the baseline JSON and compute:

- **Per-frame tables:** ms/frame = `mean_ms / (n_frames - 1)`.
- **End-to-end table:** use `mean_ms` directly.
- **Effective FPS:** `n_frames * 1000.0 / mean_ms` (for `video_e2e_64f_1obj_fwd`).

Replace each `_tbd_` with the computed number, rounded to 1 decimal for ms, integer for FPS.

A helper command to extract the numbers:

```bash
python3 <<'EOF'
import json
for v in ["efficient", "tinyvit", "hiera"]:
    d = json.load(open(f"benchmarks/baselines/{v}.json"))
    r = {x["name"]: x["mean_ms"] for x in d["results"]}
    print(f"\n=== {v} ===")
    for k in sorted(n for n in r if n.startswith("video_")):
        print(f"  {k:40s} {r[k]:.1f} ms")
EOF
```

Copy numbers into the tables.

- [ ] **Step 4: Verify `--compare` runs clean against freshly-generated baselines**

```bash
./build-release/sam3_cli bench all \
    --model models/efficient.sam3 --backend metal \
    --filter "video_*" \
    --compare benchmarks/baselines/efficient.json --threshold 5.0
```

Expected: `No regressions detected`, exit code 0 (may flag 1–2 cases due to run-to-run variance; if >5% regression consistently, investigate).

- [ ] **Step 5: Commit the baselines and doc numbers separately**

```bash
git add benchmarks/baselines/
git commit -m "bench/baselines: populate real numbers for efficient/tinyvit/hiera"

git add BENCHMARK.md
git commit -m "docs: fill in Video Tracking numbers for all three models"
```

---

## Self-Review

**1. Spec coverage:**

| Spec requirement | Implementing task |
|---|---|
| Bounce-safe clip generator | Task 1 (helper) + Task 2 (generator) |
| `bench_video_case` struct | Task 1 |
| `CLIP_MAX_FRAMES` constant | Task 1 |
| Seed-point coordinate fix | Task 3 |
| `seed_n_objects` helper | Task 4 |
| Per-frame case-table driver | Task 5 |
| End-to-end case-table driver | Task 6 |
| `scripts/refresh_baselines.sh` | Task 7 |
| Committed baseline JSON files | Task 8 (skeletons) + Task 10 (real data) |
| `BENCHMARK.md` — Per-Frame Tracking Cost section | Task 9 |
| `BENCHMARK.md` — End-to-End Clip Latency section | Task 9 |
| `--filter "video_*"` example in Running section | Task 9 |
| Filter-glob case-name test | Task 5 |
| Clip-in-bounds generator test | Task 1 (bounce invariant) + Task 2 (file-existence) |
| CMake wiring for `test_bench_video.c` + SAM3_BENCH guard | Task 1 |

Every spec requirement lands in a task.

**2. Placeholder scan:** no "TBD", "TODO", "implement later" inside step text. The `_tbd_` tokens inside `BENCHMARK.md` are intentional placeholders for runtime data (populated in Task 10).

**3. Type consistency:**

- `sam3_bench_bounce_pos` — declared in header (Task 1), implemented in `bench_video_frame.c` (Task 1 Step 4), used in `bench_video_frame.c` (Task 2, 3, 4, 5) and `bench_video_end_to_end.c` (Task 3, 6). Same signature `int sam3_bench_bounce_pos(int i)` everywhere.
- `struct bench_video_case` — declared in header (Task 1) with fields `n_frames, n_objects, seed_frame, direction, label`. Used in the same form in Task 5 (`per_frame_cases[]`) and Task 6 (`e2e_cases[]`).
- `seed_n_objects` (per-frame) and `e2e_seed_n_objects` (end-to-end) — deliberate duplicate statics (one per TU), both with signature `enum sam3_error f(sam3_video_session*, int n, int seed, float scale)`.
- Case names: `video_per_frame_<label>` (Task 5) and `video_e2e_<label>` (Task 6). Filter test (Task 5 Step 3) asserts both prefixes.
- Baseline JSON schema (Task 8) matches the `BENCH_JSON_VERSION = 1` contract in `bench_json.c`.

No mismatches found.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-18-video-benchmark.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
