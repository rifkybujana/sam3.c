# Video Tracker Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add SAM3 video object tracking (Sam3TrackerPredictor equivalent) with memory-based frame propagation, multi-object support, and SAM2Long memory selection.

**Architecture:** Bottom-up component build matching existing codebase patterns (init/load/build per module). Each component is a separate .c/.h pair, tested independently against Python reference fixtures, then composed into the tracker. Video I/O via bundled pl_mpeg + frame directory loading.

**Tech Stack:** C11, Metal/CPU backends, arena allocators, compute graph (existing infra). pl_mpeg for MPEG decoding. No new external dependencies beyond the single-header decoder.

**Design Doc:** `docs/plans/2026-04-16-video-tracker-design.md`

---

## Task 1: Public API Types and Error Codes

Add video-related types to the public headers. No implementation yet — just the interface contract.

**Files:**
- Modify: `include/sam3/sam3_types.h`
- Modify: `include/sam3/sam3.h`
- Test: `tests/test_video_types.c`

**Step 1: Add video types to sam3_types.h**

Add after the existing `sam3_model_config` struct, before `#endif`:

```c
/* Propagation direction for video tracking. */
enum sam3_propagate_dir {
	SAM3_PROPAGATE_BOTH    = 0,
	SAM3_PROPAGATE_FORWARD = 1,
	SAM3_PROPAGATE_BACKWARD = 2,
};

/* Maximum simultaneous tracked objects. */
#define SAM3_MAX_OBJECTS 64

/* Maximum memory frames in tracker bank. */
#define SAM3_MAX_MEMORY_FRAMES 16

/* Opaque video session handle. */
typedef struct sam3_video_session sam3_video_session;
```

Also add `SAM3_EVIDEO = -7` to the `sam3_error` enum (after `SAM3_EDTYPE`).

**Step 2: Add video API declarations to sam3.h**

Add before `#endif /* SAM3_H */`:

```c
/* --- Video Tracking API --- */

/*
 * sam3_video_start - Start a video tracking session.
 *
 * @ctx:           Context with loaded model
 * @resource_path: Path to video file (MPEG) or directory of frames
 * @out_session:   Receives session handle (caller must call sam3_video_end)
 *
 * Loads all frames, initializes tracker with model weights.
 * Returns SAM3_OK on success.
 */
enum sam3_error sam3_video_start(sam3_ctx *ctx,
				 const char *resource_path,
				 sam3_video_session **out_session);

/*
 * sam3_video_add_points - Add point prompts on a frame for an object.
 *
 * @session:   Active video session
 * @frame_idx: Frame index (0-based)
 * @obj_id:    User-defined object ID (creates new if unseen)
 * @points:    Array of point prompts
 * @n_points:  Number of points
 * @result:    Output mask for this frame (caller frees via sam3_result_free)
 */
enum sam3_error sam3_video_add_points(sam3_video_session *session,
				      int frame_idx, int obj_id,
				      const struct sam3_point *points,
				      int n_points,
				      struct sam3_result *result);

/*
 * sam3_video_add_box - Add a box prompt on a frame for an object.
 */
enum sam3_error sam3_video_add_box(sam3_video_session *session,
				   int frame_idx, int obj_id,
				   const struct sam3_box *box,
				   struct sam3_result *result);

/*
 * sam3_video_frame_cb - Callback invoked per frame during propagation.
 *
 * Return non-zero to stop propagation early.
 */
typedef int (*sam3_video_frame_cb)(int frame_idx,
				   const struct sam3_result *result,
				   int n_objects,
				   const int *obj_ids,
				   void *user_data);

/*
 * sam3_video_propagate - Propagate tracked objects across all frames.
 *
 * @session:   Active session with at least one prompted object
 * @direction: SAM3_PROPAGATE_BOTH, _FORWARD, or _BACKWARD
 * @callback:  Per-frame result callback
 * @user_data: Passed through to callback
 */
enum sam3_error sam3_video_propagate(sam3_video_session *session,
				     int direction,
				     sam3_video_frame_cb callback,
				     void *user_data);

/*
 * sam3_video_remove_object - Remove an object from tracking.
 */
enum sam3_error sam3_video_remove_object(sam3_video_session *session,
					 int obj_id);

/*
 * sam3_video_reset - Reset session to initial state (keep frames loaded).
 */
enum sam3_error sam3_video_reset(sam3_video_session *session);

/*
 * sam3_video_end - End session and free all resources.
 */
void sam3_video_end(sam3_video_session *session);

/*
 * sam3_video_frame_count - Get number of frames in session.
 */
int sam3_video_frame_count(const sam3_video_session *session);
```

**Step 3: Write test for types**

Create `tests/test_video_types.c`:

```c
/*
 * tests/test_video_types.c - Verify video API types compile and have correct values
 */

#include "test_helpers.h"
#include "sam3/sam3.h"

static void test_propagate_dir_values(void)
{
	ASSERT_EQ(SAM3_PROPAGATE_BOTH, 0);
	ASSERT_EQ(SAM3_PROPAGATE_FORWARD, 1);
	ASSERT_EQ(SAM3_PROPAGATE_BACKWARD, 2);
}

static void test_error_code(void)
{
	ASSERT_EQ(SAM3_EVIDEO, -7);
}

static void test_max_objects(void)
{
	ASSERT(SAM3_MAX_OBJECTS >= 64);
}

int main(void)
{
	test_propagate_dir_values();
	test_error_code();
	test_max_objects();
	TEST_REPORT();
}
```

**Step 4: Add test to CMakeLists.txt**

Add to the test section:

```cmake
if(SAM3_TESTS)
	add_executable(test_video_types tests/test_video_types.c)
	target_link_libraries(test_video_types sam3)
	add_test(NAME test_video_types COMMAND test_video_types)
endif()
```

**Step 5: Build and run test**

Run: `cd build && cmake .. && make test_video_types && ./test_video_types`
Expected: `3 tests, 0 failures`

**Step 6: Commit**

```
git add include/sam3/sam3_types.h include/sam3/sam3.h tests/test_video_types.c CMakeLists.txt
git commit -m "video: add public API types and declarations for video tracking"
```

---

## Task 2: Memory Bank (Pure Data Structure)

Ring buffer for storing past frame memories. No ML ops — just insert, evict, and query. Easiest to test in isolation.

**Files:**
- Create: `src/model/memory_bank.h`
- Create: `src/model/memory_bank.c`
- Test: `tests/test_memory_bank.c`

**Step 1: Write test**

Create `tests/test_memory_bank.c`:

```c
/*
 * tests/test_memory_bank.c - Memory bank ring buffer tests
 */

#include "test_helpers.h"
#include "model/memory_bank.h"
#include "core/alloc.h"

static void test_bank_init(void)
{
	struct sam3_memory_bank bank;
	sam3_memory_bank_init(&bank, 7, 4, 1, 0.01f);
	ASSERT_EQ(bank.n_non_cond, 0);
	ASSERT_EQ(bank.n_cond, 0);
	ASSERT_EQ(bank.capacity, 7);
	ASSERT_EQ(bank.max_cond_frames, 4);
}

static void test_bank_add_cond(void)
{
	struct sam3_memory_bank bank;
	sam3_memory_bank_init(&bank, 7, 4, 1, 0.01f);

	struct sam3_memory_entry e = {
		.spatial_features = NULL,
		.obj_pointers = NULL,
		.frame_idx = 0,
		.is_conditioning = 1,
		.obj_score = 1.0f,
	};
	sam3_memory_bank_add(&bank, &e);
	ASSERT_EQ(bank.n_cond, 1);
	ASSERT_EQ(bank.n_non_cond, 0);
}

static void test_bank_add_non_cond(void)
{
	struct sam3_memory_bank bank;
	sam3_memory_bank_init(&bank, 7, 4, 1, 0.01f);

	struct sam3_memory_entry e = {
		.frame_idx = 5,
		.is_conditioning = 0,
		.obj_score = 0.5f,
	};
	sam3_memory_bank_add(&bank, &e);
	ASSERT_EQ(bank.n_non_cond, 1);
}

static void test_bank_evict_oldest_non_cond(void)
{
	struct sam3_memory_bank bank;
	sam3_memory_bank_init(&bank, 3, 1, 1, 0.01f);
	/* capacity=3 means 1 cond + 2 non-cond max */

	struct sam3_memory_entry cond = { .frame_idx = 0, .is_conditioning = 1, .obj_score = 1.0f };
	struct sam3_memory_entry nc1 = { .frame_idx = 1, .is_conditioning = 0, .obj_score = 0.5f };
	struct sam3_memory_entry nc2 = { .frame_idx = 2, .is_conditioning = 0, .obj_score = 0.5f };
	struct sam3_memory_entry nc3 = { .frame_idx = 3, .is_conditioning = 0, .obj_score = 0.5f };

	sam3_memory_bank_add(&bank, &cond);
	sam3_memory_bank_add(&bank, &nc1);
	sam3_memory_bank_add(&bank, &nc2);
	ASSERT_EQ(bank.n_non_cond, 2);

	/* Adding nc3 should evict nc1 (oldest) */
	sam3_memory_bank_add(&bank, &nc3);
	ASSERT_EQ(bank.n_non_cond, 2);
	ASSERT_EQ(bank.non_cond[0].frame_idx, 2);
	ASSERT_EQ(bank.non_cond[1].frame_idx, 3);
}

static void test_bank_sam2long_selection(void)
{
	struct sam3_memory_bank bank;
	sam3_memory_bank_init(&bank, 7, 4, 1, 0.01f);

	/* Frame with score below threshold should be rejected */
	struct sam3_memory_entry low = { .frame_idx = 5, .is_conditioning = 0, .obj_score = 0.005f };
	sam3_memory_bank_add(&bank, &low);
	ASSERT_EQ(bank.n_non_cond, 0); /* rejected */

	/* Frame with score above threshold should be accepted */
	struct sam3_memory_entry high = { .frame_idx = 6, .is_conditioning = 0, .obj_score = 0.5f };
	sam3_memory_bank_add(&bank, &high);
	ASSERT_EQ(bank.n_non_cond, 1);
}

static void test_bank_select_closest(void)
{
	struct sam3_memory_bank bank;
	sam3_memory_bank_init(&bank, 7, 4, 1, 0.01f);

	struct sam3_memory_entry e0 = { .frame_idx = 0, .is_conditioning = 1, .obj_score = 1.0f };
	struct sam3_memory_entry e5 = { .frame_idx = 5, .is_conditioning = 1, .obj_score = 1.0f };
	struct sam3_memory_entry e10 = { .frame_idx = 10, .is_conditioning = 1, .obj_score = 1.0f };
	sam3_memory_bank_add(&bank, &e0);
	sam3_memory_bank_add(&bank, &e5);
	sam3_memory_bank_add(&bank, &e10);

	int indices[2];
	int n = sam3_memory_bank_select_closest_cond(&bank, 7, indices, 2);
	ASSERT_EQ(n, 2);
	/* frame 5 is closest to 7, then frame 10 */
	ASSERT_EQ(bank.cond[indices[0]].frame_idx, 5);
	ASSERT_EQ(bank.cond[indices[1]].frame_idx, 10);
}

static void test_bank_clear(void)
{
	struct sam3_memory_bank bank;
	sam3_memory_bank_init(&bank, 7, 4, 1, 0.01f);

	struct sam3_memory_entry e = { .frame_idx = 0, .is_conditioning = 1, .obj_score = 1.0f };
	sam3_memory_bank_add(&bank, &e);
	ASSERT_EQ(bank.n_cond, 1);

	sam3_memory_bank_clear(&bank);
	ASSERT_EQ(bank.n_cond, 0);
	ASSERT_EQ(bank.n_non_cond, 0);
}

int main(void)
{
	test_bank_init();
	test_bank_add_cond();
	test_bank_add_non_cond();
	test_bank_evict_oldest_non_cond();
	test_bank_sam2long_selection();
	test_bank_select_closest();
	test_bank_clear();
	TEST_REPORT();
}
```

**Step 2: Run test to verify it fails**

Run: `cd build && cmake .. && make test_memory_bank 2>&1`
Expected: FAIL (header not found)

**Step 3: Write memory_bank.h**

Create `src/model/memory_bank.h`:

```c
/*
 * src/model/memory_bank.h - Frame memory ring buffer for video tracking
 *
 * Stores spatial features and object pointers from past frames.
 * Maintains separate storage for conditioning frames (user-annotated)
 * and non-conditioning frames (propagated). Implements SAM2Long-style
 * memory selection that filters low-confidence frames.
 *
 * Key types:  sam3_memory_entry, sam3_memory_bank
 * Depends on: core/tensor.h
 * Used by:    tracker.h
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_MEMORY_BANK_H
#define SAM3_MODEL_MEMORY_BANK_H

#include "core/tensor.h"
#include "sam3/sam3_types.h"

struct sam3_memory_entry {
	struct sam3_tensor *spatial_features; /* [HW, mem_dim] */
	struct sam3_tensor *obj_pointers;     /* [n_obj, hidden_dim] */
	int    frame_idx;
	int    is_conditioning;
	float  obj_score; /* max object score for SAM2Long selection */
};

struct sam3_memory_bank {
	/* Conditioning frames (user-annotated): separate array, not evicted */
	struct sam3_memory_entry cond[SAM3_MAX_MEMORY_FRAMES];
	int    n_cond;
	int    max_cond_frames;

	/* Non-conditioning frames: ring buffer, oldest evicted when full */
	struct sam3_memory_entry non_cond[SAM3_MAX_MEMORY_FRAMES];
	int    n_non_cond;

	int    capacity;         /* total frames allowed (cond + non-cond) */
	int    temporal_stride;  /* stride for non-cond frame selection */
	float  mf_threshold;    /* SAM2Long: reject frames below this score */
};

/*
 * sam3_memory_bank_init - Initialize memory bank.
 *
 * @bank:            Bank struct (caller-allocated)
 * @capacity:        Max total memory frames (default 7)
 * @max_cond_frames: Max conditioning frames (default 4)
 * @temporal_stride: Stride for non-cond selection (default 1)
 * @mf_threshold:    SAM2Long selection threshold (default 0.01)
 */
void sam3_memory_bank_init(struct sam3_memory_bank *bank,
			   int capacity, int max_cond_frames,
			   int temporal_stride, float mf_threshold);

/*
 * sam3_memory_bank_add - Add a memory entry to the bank.
 *
 * Conditioning entries go to cond array. Non-conditioning entries
 * are subject to SAM2Long selection (rejected if obj_score < threshold)
 * and ring buffer eviction (oldest removed when full).
 */
void sam3_memory_bank_add(struct sam3_memory_bank *bank,
			  const struct sam3_memory_entry *entry);

/*
 * sam3_memory_bank_select_closest_cond - Find conditioning frames
 * closest to a target frame index.
 *
 * @bank:       Memory bank
 * @frame_idx:  Target frame
 * @out_indices: Output array of indices into bank->cond[]
 * @max_n:      Maximum entries to return
 *
 * Returns number of entries written to out_indices.
 */
int sam3_memory_bank_select_closest_cond(
	const struct sam3_memory_bank *bank,
	int frame_idx, int *out_indices, int max_n);

/*
 * sam3_memory_bank_clear - Reset bank to empty state.
 */
void sam3_memory_bank_clear(struct sam3_memory_bank *bank);

/*
 * sam3_memory_bank_total - Get total number of entries (cond + non-cond).
 */
int sam3_memory_bank_total(const struct sam3_memory_bank *bank);

#endif /* SAM3_MODEL_MEMORY_BANK_H */
```

**Step 4: Write memory_bank.c**

Create `src/model/memory_bank.c`:

```c
/*
 * src/model/memory_bank.c - Frame memory ring buffer implementation
 *
 * Ring buffer for caching past frame features during video tracking.
 * Conditioning frames are stored separately and never evicted.
 * Non-conditioning frames use a simple FIFO ring buffer with
 * SAM2Long-style filtering by object confidence score.
 *
 * Key types:  sam3_memory_bank
 * Depends on: memory_bank.h
 * Used by:    tracker.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "memory_bank.h"
#include <string.h>
#include <stdlib.h>

void sam3_memory_bank_init(struct sam3_memory_bank *bank,
			   int capacity, int max_cond_frames,
			   int temporal_stride, float mf_threshold)
{
	memset(bank, 0, sizeof(*bank));
	bank->capacity = capacity;
	bank->max_cond_frames = max_cond_frames;
	bank->temporal_stride = temporal_stride;
	bank->mf_threshold = mf_threshold;
}

void sam3_memory_bank_add(struct sam3_memory_bank *bank,
			  const struct sam3_memory_entry *entry)
{
	if (entry->is_conditioning) {
		if (bank->n_cond < bank->max_cond_frames &&
		    bank->n_cond < SAM3_MAX_MEMORY_FRAMES) {
			bank->cond[bank->n_cond++] = *entry;
		}
		return;
	}

	/* SAM2Long selection: reject low-confidence frames */
	if (entry->obj_score < bank->mf_threshold)
		return;

	int max_non_cond = bank->capacity - 1; /* reserve 1 for cond */
	if (max_non_cond > SAM3_MAX_MEMORY_FRAMES)
		max_non_cond = SAM3_MAX_MEMORY_FRAMES;

	if (bank->n_non_cond >= max_non_cond) {
		/* Evict oldest: shift array left */
		memmove(&bank->non_cond[0], &bank->non_cond[1],
			(bank->n_non_cond - 1) * sizeof(bank->non_cond[0]));
		bank->n_non_cond--;
	}
	bank->non_cond[bank->n_non_cond++] = *entry;
}

/* Compare function for sorting by distance to target frame */
struct dist_pair { int index; int distance; };

static int cmp_dist(const void *a, const void *b)
{
	const struct dist_pair *da = a, *db = b;
	return da->distance - db->distance;
}

int sam3_memory_bank_select_closest_cond(
	const struct sam3_memory_bank *bank,
	int frame_idx, int *out_indices, int max_n)
{
	if (bank->n_cond == 0 || max_n <= 0)
		return 0;

	struct dist_pair pairs[SAM3_MAX_MEMORY_FRAMES];
	for (int i = 0; i < bank->n_cond; i++) {
		pairs[i].index = i;
		int d = bank->cond[i].frame_idx - frame_idx;
		pairs[i].distance = d < 0 ? -d : d;
	}
	qsort(pairs, bank->n_cond, sizeof(pairs[0]), cmp_dist);

	int n = bank->n_cond < max_n ? bank->n_cond : max_n;
	for (int i = 0; i < n; i++)
		out_indices[i] = pairs[i].index;
	return n;
}

void sam3_memory_bank_clear(struct sam3_memory_bank *bank)
{
	bank->n_cond = 0;
	bank->n_non_cond = 0;
}

int sam3_memory_bank_total(const struct sam3_memory_bank *bank)
{
	return bank->n_cond + bank->n_non_cond;
}
```

**Step 5: Build and run test**

Run: `cd build && cmake .. && make test_memory_bank && ./test_memory_bank`
Expected: `7 tests, 0 failures`

**Step 6: Commit**

```
git add src/model/memory_bank.h src/model/memory_bank.c tests/test_memory_bank.c
git commit -m "video/memory_bank: add ring buffer for frame memory storage"
```

---

## Task 3: Video I/O (Frame Loading)

Bundle pl_mpeg and implement frame loading from video files and frame directories.

**Files:**
- Create: `src/util/vendor/pl_mpeg.h` (download MIT-licensed single header)
- Create: `src/util/video.h`
- Create: `src/util/video.c`
- Test: `tests/test_video_io.c`
- Test data: `tests/data/frames/` (3 small JPEG files for testing)

**Step 1: Download pl_mpeg.h**

```bash
curl -L -o src/util/vendor/pl_mpeg.h \
  https://raw.githubusercontent.com/phoboslab/pl_mpeg/master/pl_mpeg.h
```

Verify it's MIT licensed (check header comment).

**Step 2: Write test**

Create `tests/test_video_io.c`:

```c
/*
 * tests/test_video_io.c - Video frame loading tests
 */

#include "test_helpers.h"
#include "util/video.h"
#include "core/alloc.h"
#include <string.h>

static void test_detect_frame_dir(void)
{
	ASSERT_EQ(sam3_video_detect_type("tests/data/frames"), SAM3_VIDEO_FRAME_DIR);
}

static void test_detect_video_file(void)
{
	ASSERT_EQ(sam3_video_detect_type("video.mpg"), SAM3_VIDEO_MPEG);
	ASSERT_EQ(sam3_video_detect_type("video.mpeg"), SAM3_VIDEO_MPEG);
}

static void test_detect_unknown(void)
{
	ASSERT_EQ(sam3_video_detect_type("photo.jpg"), SAM3_VIDEO_UNKNOWN);
}

static void test_load_frame_dir(void)
{
	struct sam3_arena arena;
	sam3_arena_init(&arena, 64 * 1024 * 1024); /* 64 MB */

	struct sam3_video_frames frames;
	int err = sam3_video_load("tests/data/frames", 512, &frames, &arena);
	if (err == SAM3_EIO) {
		/* test data not present — skip gracefully */
		sam3_arena_free(&arena);
		return;
	}
	ASSERT_EQ(err, SAM3_OK);
	ASSERT(frames.n_frames > 0);
	ASSERT_EQ(frames.frame_size, 512);
	ASSERT(frames.orig_width > 0);
	ASSERT(frames.orig_height > 0);

	/* Each frame should be [3, 512, 512] F32 */
	for (int i = 0; i < frames.n_frames; i++) {
		ASSERT(frames.pixels[i] != NULL);
		ASSERT_EQ(frames.pixels[i]->n_dims, 3);
		ASSERT_EQ(frames.pixels[i]->dims[0], 3);
		ASSERT_EQ(frames.pixels[i]->dims[1], 512);
		ASSERT_EQ(frames.pixels[i]->dims[2], 512);
	}

	sam3_arena_free(&arena);
}

int main(void)
{
	test_detect_frame_dir();
	test_detect_video_file();
	test_detect_unknown();
	test_load_frame_dir();
	TEST_REPORT();
}
```

**Step 3: Write video.h**

Create `src/util/video.h`:

```c
/*
 * src/util/video.h - Video frame loading for video tracking
 *
 * Loads video frames from MPEG files (via bundled pl_mpeg) or from
 * directories of JPEG/PNG images (via stb_image). Frames are decoded,
 * resized, normalized, and stored as F32 tensors.
 *
 * Key types:  sam3_video_frames, sam3_video_type
 * Depends on: core/tensor.h, core/alloc.h, sam3/sam3_types.h
 * Used by:    model/video_session.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_UTIL_VIDEO_H
#define SAM3_UTIL_VIDEO_H

#include "core/tensor.h"
#include "core/alloc.h"
#include "sam3/sam3_types.h"

enum sam3_video_type {
	SAM3_VIDEO_UNKNOWN   = 0,
	SAM3_VIDEO_FRAME_DIR = 1,
	SAM3_VIDEO_MPEG      = 2,
};

struct sam3_video_frames {
	struct sam3_tensor **pixels;  /* Array of [3, H, W] F32 tensors */
	int    n_frames;
	int    frame_size;           /* Model input size (e.g. 1008) */
	int    orig_width;
	int    orig_height;
};

/*
 * sam3_video_detect_type - Detect whether path is a video file or frame dir.
 */
enum sam3_video_type sam3_video_detect_type(const char *path);

/*
 * sam3_video_load - Load all frames from a video resource.
 *
 * @path:       Path to MPEG file or directory of images
 * @image_size: Target frame size (e.g. 1008)
 * @out:        Output frame storage
 * @arena:      Arena for all allocations
 *
 * Frames are resized to image_size x image_size and normalized to
 * [-1, 1] with mean=0.5, std=0.5.
 *
 * Returns SAM3_OK on success, SAM3_EIO on load failure.
 */
enum sam3_error sam3_video_load(const char *path, int image_size,
				struct sam3_video_frames *out,
				struct sam3_arena *arena);

#endif /* SAM3_UTIL_VIDEO_H */
```

**Step 4: Write video.c**

Create `src/util/video.c`. Implementation should:
- Use `sam3_video_detect_type()` to check if path is directory or `.mpg`/`.mpeg` file
- For frame directories: list `.jpg`/`.jpeg`/`.png` files sorted alphabetically, load each via `stb_image`, resize via `stb_image_resize2`, normalize
- For MPEG files: use `pl_mpeg.h` to decode each frame, resize, normalize
- Store each frame as `[3, image_size, image_size]` F32 tensor
- Mean/std normalization: `(pixel/255.0 - 0.5) / 0.5`

Reference: see `src/util/image.h` for existing image loading patterns, and `reference/sam3/sam3/model/io_utils.py:load_resource_as_video_frames()`.

**Step 5: Create test frame data**

Create 3 small test JPEG files in `tests/data/frames/` (64x64 solid color images). Can be generated with a small C program or copied from existing test assets.

**Step 6: Build and run test**

Run: `cd build && cmake .. && make test_video_io && ./test_video_io`
Expected: All detection tests pass. Frame loading test passes if test data exists.

**Step 7: Commit**

```
git add src/util/vendor/pl_mpeg.h src/util/video.h src/util/video.c tests/test_video_io.c
git commit -m "video/io: add frame loading from directories and MPEG files"
```

---

## Task 4: Memory Encoder

Implements `SimpleMaskEncoder` from `reference/sam3/sam3/model/memory.py`.

**Files:**
- Create: `src/model/memory_encoder.h`
- Create: `src/model/memory_encoder.c`
- Test: `tests/test_memory_encoder.c`

**Step 1: Write test**

Test verifies shapes and basic functionality. Full numerical matching against Python fixtures is a separate integration test.

```c
/*
 * tests/test_memory_encoder.c - Memory encoder shape and smoke tests
 */

#include "test_helpers.h"
#include "model/memory_encoder.h"
#include "core/alloc.h"
#include "core/graph.h"
#include "backend/backend.h"

static void test_mem_encoder_init(void)
{
	struct sam3_memory_encoder enc;
	int err = sam3_memory_encoder_init(&enc, 256, 64);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT_EQ(enc.in_dim, 256);
	ASSERT_EQ(enc.out_dim, 64);
}

static void test_mem_encoder_build_shapes(void)
{
	struct sam3_arena scratch, persist;
	sam3_arena_init(&scratch, 128 * 1024 * 1024);
	sam3_arena_init(&persist, 128 * 1024 * 1024);

	struct sam3_memory_encoder enc;
	sam3_memory_encoder_init(&enc, 256, 64);
	sam3_memory_encoder_load(&enc, NULL, &persist); /* zero-init */

	struct sam3_graph g;
	sam3_graph_init(&g);

	/* Create dummy inputs */
	int pix_dims[] = {1, 256, 72, 72};
	struct sam3_tensor *pix_feat = sam3_arena_alloc_tensor(
		&scratch, SAM3_DTYPE_F32, 4, pix_dims);

	int mask_dims[] = {1, 1, 288, 288}; /* low_res_mask_size * 4 */
	struct sam3_tensor *masks = sam3_arena_alloc_tensor(
		&scratch, SAM3_DTYPE_F32, 4, mask_dims);

	struct sam3_tensor *out_feat = NULL;
	struct sam3_tensor *out_pos = NULL;
	int err = sam3_memory_encoder_build(&enc, &g, pix_feat, masks,
					     &scratch, &out_feat, &out_pos);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT(out_feat != NULL);
	ASSERT(out_pos != NULL);

	/* Output should be [1, 64, 72, 72] */
	ASSERT_EQ(out_feat->n_dims, 4);
	ASSERT_EQ(out_feat->dims[0], 1);
	ASSERT_EQ(out_feat->dims[1], 64);

	sam3_arena_free(&scratch);
	sam3_arena_free(&persist);
}

int main(void)
{
	test_mem_encoder_init();
	test_mem_encoder_build_shapes();
	TEST_REPORT();
}
```

**Step 2: Write memory_encoder.h**

Struct fields matching `SimpleMaskEncoder`:
- `mask_downsampler`: 2 conv layers (kernel=3, stride=2, pad=1) + LayerNorm2d + GELU
- `pix_feat_proj`: 1x1 conv
- `fuser`: 2 CXBlock layers (depthwise conv-7x7 + LayerNorm + pointwise GELU MLP + layer_scale)
- `out_proj`: 1x1 conv (256->64)
- Position encoding: reuse existing `sam3_position_encoding` from `position_encoding.h`

Weight prefix: `tracker_model.maskmem_backbone.`

Reference: `reference/sam3/sam3/model/memory.py` lines 21-209 and `reference/sam3/sam3/model_builder.py:_create_tracker_maskmem_backbone()` lines 344-377.

**Step 3: Write memory_encoder.c**

Follow the init/load/build pattern from existing modules (see `src/model/segmentation.c` for a comparable example). The build function should:
1. Sigmoid the input mask
2. Interpolate mask to `interpol_size` if configured
3. Run mask through downsampler conv cascade
4. Project pixel features with 1x1 conv
5. Add downsampled mask to projected features
6. Run through 2 CXBlock fuser layers
7. Project to output dimension
8. Compute sinusoidal position encoding

Reference: `SimpleMaskEncoder.forward()` in `reference/sam3/sam3/model/memory.py:186-209`.

**Step 4: Build and run**

Run: `cd build && cmake .. && make test_memory_encoder && ./test_memory_encoder`
Expected: `2 tests, 0 failures`

**Step 5: Commit**

```
git add src/model/memory_encoder.h src/model/memory_encoder.c tests/test_memory_encoder.c
git commit -m "video/memory_encoder: add mask-to-memory feature encoder"
```

---

## Task 5: Memory Attention (RoPE Cross-Attention)

Fill the existing `memory_attn.h/.c` stub with 4-layer RoPE cross-attention transformer.

**Files:**
- Modify: `src/model/memory_attn.h` (expand struct, add load/build functions)
- Modify: `src/model/memory_attn.c` (full implementation)
- Test: `tests/test_memory_attn.c`

**Step 1: Write test**

```c
/*
 * tests/test_memory_attn.c - Memory attention shape and smoke tests
 */
#include "test_helpers.h"
#include "model/memory_attn.h"
#include "core/alloc.h"
#include "core/graph.h"

static void test_mem_attn_init(void)
{
	struct sam3_memory_attn attn;
	int err = sam3_memory_attn_init(&attn, 256, 64, 4, 1, 72, 72);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT_EQ(attn.d_model, 256);
	ASSERT_EQ(attn.mem_dim, 64);
	ASSERT_EQ(attn.n_layers, 4);
}

static void test_mem_attn_build_shapes(void)
{
	struct sam3_arena scratch, persist;
	sam3_arena_init(&scratch, 256 * 1024 * 1024);
	sam3_arena_init(&persist, 256 * 1024 * 1024);

	struct sam3_memory_attn attn;
	sam3_memory_attn_init(&attn, 256, 64, 4, 1, 72, 72);
	sam3_memory_attn_load(&attn, NULL, &persist); /* zero-init */

	struct sam3_graph g;
	sam3_graph_init(&g);

	/* current_features: [5184, 256] (72*72 tokens) */
	int cur_dims[] = {5184, 256};
	struct sam3_tensor *cur = sam3_arena_alloc_tensor(
		&scratch, SAM3_DTYPE_F32, 2, cur_dims);

	/* memory: 2 frames of [5184, 64] stacked = [10368, 64] */
	int mem_dims[] = {10368, 64};
	struct sam3_tensor *mem = sam3_arena_alloc_tensor(
		&scratch, SAM3_DTYPE_F32, 2, mem_dims);

	int mem_pos_dims[] = {10368, 64};
	struct sam3_tensor *mem_pos = sam3_arena_alloc_tensor(
		&scratch, SAM3_DTYPE_F32, 2, mem_pos_dims);

	struct sam3_tensor *output = NULL;
	int err = sam3_memory_attn_build_full(&attn, &g, cur, mem,
					       mem_pos, &scratch, &output);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT(output != NULL);

	/* Output should be [5184, 256] */
	ASSERT_EQ(output->n_dims, 2);
	ASSERT_EQ(output->dims[0], 5184);
	ASSERT_EQ(output->dims[1], 256);

	sam3_arena_free(&scratch);
	sam3_arena_free(&persist);
}

int main(void)
{
	test_mem_attn_init();
	test_mem_attn_build_shapes();
	TEST_REPORT();
}
```

**Step 2: Redesign memory_attn.h**

Replace the existing stub with a full struct matching `_create_tracker_transformer()` from `reference/sam3/sam3/model_builder.py:380-443`. Each of the 4 layers has:

- Self-attention: RoPE attention (d=256, 1 head)
- Cross-attention: RoPE attention (d=256, 1 head, kv_dim=64)
- LayerNorm x4 (pre-norm for each attention + FFN)
- FFN: Linear(256, 2048) + ReLU + Linear(2048, 256)

Weight prefix: `tracker_model.transformer.encoder.layers.{0-3}.*`

Reference: `reference/sam3/sam3/model/sam3_tracker_base.py` for the module structure, `reference/sam3/sam3/model_builder.py:380-443` for the exact configuration.

**Step 3: Implement memory_attn.c**

The build function must:
1. For each of 4 layers:
   a. Self-attention with RoPE on queries and keys (2D sinusoidal frequencies with theta=10000, feat_sizes=[72,72])
   b. Cross-attention: queries from current features (256-dim), keys/values from memory (64-dim) with RoPE (rope_k_repeat=True for the 64-dim KV)
   c. FFN with pre-norm

The existing codebase has RoPE patterns in the encoder (check `src/model/encoder.c` for RoPE usage).

**Step 4: Build and run**

Run: `cd build && cmake .. && make test_memory_attn && ./test_memory_attn`
Expected: `2 tests, 0 failures`

**Step 5: Commit**

```
git add src/model/memory_attn.h src/model/memory_attn.c tests/test_memory_attn.c
git commit -m "video/memory_attn: implement 4-layer RoPE cross-attention transformer"
```

---

## Task 6: Tracker Core

The main tracker module that wires memory encoder, memory attention, memory bank, and SAM mask decoder together.

**Files:**
- Create: `src/model/tracker.h`
- Create: `src/model/tracker.c`
- Test: `tests/test_tracker.c`

**Step 1: Write test**

Test init/load lifecycle and single-frame tracking with zero-initialized weights (smoke test for graph construction).

```c
/*
 * tests/test_tracker.c - Tracker lifecycle and graph construction tests
 */

#include "test_helpers.h"
#include "model/tracker.h"
#include "core/alloc.h"
#include "backend/backend.h"

static void test_tracker_init(void)
{
	struct sam3_tracker trk;
	int err = sam3_tracker_init(&trk);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT_EQ(trk.num_maskmem, 7);
	ASSERT_EQ(trk.image_size, 1008);
}

static void test_tracker_load_zero(void)
{
	struct sam3_arena arena;
	sam3_arena_init(&arena, 512 * 1024 * 1024);

	struct sam3_tracker trk;
	sam3_tracker_init(&trk);
	int err = sam3_tracker_load(&trk, NULL, &arena);
	ASSERT_EQ(err, SAM3_OK);

	sam3_arena_free(&arena);
}

int main(void)
{
	test_tracker_init();
	test_tracker_load_zero();
	TEST_REPORT();
}
```

**Step 2: Write tracker.h**

```c
/*
 * src/model/tracker.h - SAM3 video tracker core
 *
 * Implements the Sam3TrackerPredictor equivalent: memory-based
 * tracking across video frames using memory attention, SAM mask
 * decoder, and a memory bank.
 *
 * Key types:  sam3_tracker
 * Depends on: memory_encoder.h, memory_attn.h, memory_bank.h,
 *             mask_decoder.h, core/weight.h
 * Used by:    video_session.c
 */
```

Struct contains:
- `sam3_mask_decoder sam_decoder` (reused type from mask_decoder.h)
- `sam3_memory_encoder mem_encoder`
- `sam3_memory_attn mem_attention`
- `sam3_memory_bank mem_bank`
- Learned parameters: `maskmem_tpos_enc`, `no_mem_embed`, `no_mem_pos_enc`, `no_obj_ptr`, `no_obj_embed_spatial`, `mask_downsample_w/b`
- Config fields: `num_maskmem`, `max_cond_frames`, `image_size`, `backbone_stride`, `max_obj_ptrs`, `sigmoid_scale`, `sigmoid_bias`, `mf_threshold`, multimask settings

API:
- `sam3_tracker_init()` — set defaults
- `sam3_tracker_load()` — load weights with prefix `tracker_model.*`
- `sam3_tracker_track_frame()` — process one frame given backbone features + prompts, return masks + update memory bank
- `sam3_tracker_reset()` — clear memory bank

Reference: `reference/sam3/sam3/model/sam3_tracker_base.py:26-150` for the tracker structure, `reference/sam3/sam3/model/sam3_tracking_predictor.py:15-55` for the predictor wrapper.

**Step 3: Write tracker.c**

The `track_frame()` function implements the per-frame workflow:
1. If memory bank is empty, use `no_mem_embed` as placeholder memory
2. Otherwise, collect memory tokens from bank (spatial features + temporal pos enc)
3. Collect object pointers from bank entries (up to `max_obj_ptrs`)
4. Run memory attention: cross-attend current backbone features to memory tokens
5. Run SAM mask decoder with memory-conditioned features + point/box prompts
6. Extract object pointer from mask decoder output tokens
7. Apply sigmoid scaling to mask logits for memory encoding
8. Run memory encoder on mask logits + pixel features
9. Store result in memory bank

Weight loading: iterate tensor names with prefix `tracker_model.` and delegate to sub-module loaders:
- `tracker_model.sam_mask_decoder.*` → `sam3_mask_decoder_load()`
- `tracker_model.maskmem_backbone.*` → `sam3_memory_encoder_load()`
- `tracker_model.transformer.*` → `sam3_memory_attn_load()`
- Direct tensors: `maskmem_tpos_enc`, `no_mem_embed`, etc.

Reference: `Sam3TrackerBase._track_step()` method from `reference/sam3/sam3/model/sam3_tracker_base.py` (read the full method to understand the exact flow).

**Step 4: Build and run**

Run: `cd build && cmake .. && make test_tracker && ./test_tracker`
Expected: `2 tests, 0 failures`

**Step 5: Commit**

```
git add src/model/tracker.h src/model/tracker.c tests/test_tracker.c
git commit -m "video/tracker: add core tracker module wiring memory components"
```

---

## Task 7: Video Session

Session management that ties the tracker to video frames and provides the implementation behind the public API.

**Files:**
- Create: `src/model/video_session.h`
- Create: `src/model/video_session.c`
- Test: `tests/test_video_session.c`

**Step 1: Write test**

```c
/*
 * tests/test_video_session.c - Video session lifecycle tests
 */

#include "test_helpers.h"
#include "model/video_session.h"

static void test_session_object_management(void)
{
	struct sam3_video_session session;
	memset(&session, 0, sizeof(session));
	session.n_objects = 0;

	/* Add first object */
	int idx = sam3_session_get_or_add_obj(&session, 42);
	ASSERT_EQ(idx, 0);
	ASSERT_EQ(session.n_objects, 1);
	ASSERT_EQ(session.obj_ids[0], 42);

	/* Same obj_id returns same index */
	int idx2 = sam3_session_get_or_add_obj(&session, 42);
	ASSERT_EQ(idx2, 0);
	ASSERT_EQ(session.n_objects, 1);

	/* Different obj_id gets new index */
	int idx3 = sam3_session_get_or_add_obj(&session, 99);
	ASSERT_EQ(idx3, 1);
	ASSERT_EQ(session.n_objects, 2);
}

static void test_session_remove_object(void)
{
	struct sam3_video_session session;
	memset(&session, 0, sizeof(session));

	sam3_session_get_or_add_obj(&session, 10);
	sam3_session_get_or_add_obj(&session, 20);
	ASSERT_EQ(session.n_objects, 2);

	int err = sam3_session_remove_obj(&session, 10);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT_EQ(session.n_objects, 1);
	ASSERT_EQ(session.obj_ids[0], 20);
}

int main(void)
{
	test_session_object_management();
	test_session_remove_object();
	TEST_REPORT();
}
```

**Step 2: Write video_session.h**

Define `struct sam3_video_session` with:
- Pointer to `sam3_ctx`
- `sam3_tracker` instance
- Frame storage (`sam3_video_frames`)
- Object tracking arrays (IDs, indices, count)
- Per-frame feature cache
- Per-object per-frame prompt storage
- Output dictionaries (cond/non-cond frame outputs)
- Arenas (persist + scratch)

API functions:
- `sam3_session_get_or_add_obj()` — object ID mapping
- `sam3_session_remove_obj()` — remove tracked object
- Internal helpers for propagation

Reference: `Sam3TrackerPredictor.init_state()` from `reference/sam3/sam3/model/sam3_tracking_predictor.py:57-136`.

**Step 3: Write video_session.c**

Implement session state management matching the Python reference's inference_state dictionary.

**Step 4: Build and run**

Run: `cd build && cmake .. && make test_video_session && ./test_video_session`
Expected: `2 tests, 0 failures`

**Step 5: Commit**

```
git add src/model/video_session.h src/model/video_session.c tests/test_video_session.c
git commit -m "video/session: add session management for video tracking"
```

---

## Task 8: Public API Implementation

Wire the public `sam3_video_*` functions to the session and tracker internals.

**Files:**
- Create: `src/model/sam3_video.c` (implements sam3.h video functions)
- Modify: `CMakeLists.txt` (ensure new sources are compiled)
- Test: `tests/test_video_api.c`

**Step 1: Write test**

Test the full API lifecycle with zero-init weights (no model file needed).

```c
/*
 * tests/test_video_api.c - Public video API lifecycle test
 */

#include "test_helpers.h"
#include "sam3/sam3.h"

static void test_api_lifecycle(void)
{
	/* This test validates the API compiles and the lifecycle works.
	 * Without real model weights, we test structure only. */
	sam3_ctx *ctx = sam3_init();
	ASSERT(ctx != NULL);

	/* sam3_video_start without a loaded model should return an error */
	sam3_video_session *session = NULL;
	int err = sam3_video_start(ctx, "tests/data/frames", &session);
	/* Expect SAM3_EMODEL since no model is loaded */
	ASSERT(err != SAM3_OK);
	ASSERT(session == NULL);

	sam3_free(ctx);
}

static void test_frame_count_null(void)
{
	ASSERT_EQ(sam3_video_frame_count(NULL), 0);
}

int main(void)
{
	test_api_lifecycle();
	test_frame_count_null();
	TEST_REPORT();
}
```

**Step 2: Implement sam3_video.c**

Wire each public function:
- `sam3_video_start()`: validate ctx has model, load frames, init tracker with model weights, create session
- `sam3_video_add_points()`: map obj_id, encode frame if needed, run tracker single-frame inference
- `sam3_video_add_box()`: same as add_points but with box prompt
- `sam3_video_propagate()`: iterate frames (forward/backward), encode each, run tracker, invoke callback
- `sam3_video_remove_object()`: delegate to session
- `sam3_video_reset()`: clear memory bank and per-frame state
- `sam3_video_end()`: free session arenas, tracker, frames

Reference: `Sam3BasePredictor.propagate_in_video()` from `reference/sam3/sam3/model/sam3_base_predictor.py:237-286` for the propagation loop structure.

**Step 3: Update CMakeLists.txt**

The new `.c` files in `src/model/` and `src/util/` are already picked up by the GLOB_RECURSE. Verify by checking that `memory_bank.c`, `memory_encoder.c`, `tracker.c`, `video_session.c`, `sam3_video.c`, and `video.c` are all compiled. Add test executables:

```cmake
if(SAM3_TESTS)
	# ... existing tests ...
	add_executable(test_video_types tests/test_video_types.c)
	add_executable(test_memory_bank tests/test_memory_bank.c)
	add_executable(test_video_io tests/test_video_io.c)
	add_executable(test_memory_encoder tests/test_memory_encoder.c)
	add_executable(test_memory_attn tests/test_memory_attn.c)
	add_executable(test_tracker tests/test_tracker.c)
	add_executable(test_video_session tests/test_video_session.c)
	add_executable(test_video_api tests/test_video_api.c)

	foreach(t test_video_types test_memory_bank test_video_io
		test_memory_encoder test_memory_attn test_tracker
		test_video_session test_video_api)
		target_link_libraries(${t} sam3)
		add_test(NAME ${t} COMMAND ${t})
	endforeach()
endif()
```

**Step 4: Build and run all tests**

Run: `cd build && cmake .. && make -j$(nproc) && ctest --output-on-failure`
Expected: All video tests pass.

**Step 5: Commit**

```
git add src/model/sam3_video.c CMakeLists.txt tests/test_video_api.c
git commit -m "video: wire public API to tracker and session internals"
```

---

## Task 9: Weight Converter Update

Update `sam3_convert` to include tracker weights when converting from PyTorch safetensors.

**Files:**
- Modify: `tools/sam3_convert.c` (add tracker weight prefixes)
- Test: Manual verification with reference checkpoint

**Step 1: Identify tracker weight names**

From the reference model, tracker weights use these prefixes:
- `tracker_model.maskmem_backbone.*`
- `tracker_model.transformer.*`
- `tracker_model.sam_mask_decoder.*`
- `tracker_model.sam_prompt_encoder.*`
- `tracker_model.maskmem_tpos_enc`
- `tracker_model.no_mem_embed`
- `tracker_model.no_mem_pos_enc`
- `tracker_model.no_obj_ptr`
- `tracker_model.no_obj_embed_spatial`
- `tracker_model.mask_downsample.*`

**Step 2: Update weight converter rename table**

Add tracker weight name mappings to the converter's rename table. The converter already handles `detector_model.*` prefixes; add `tracker_model.*` with the same pattern.

Reference: look at the existing rename table in `tools/sam3_convert.c` to understand the mapping format.

**Step 3: Test with checkpoint**

Run: `./sam3_convert <checkpoint.safetensors> output.sam3`
Verify tracker weights are present: check output log shows tracker_model tensors.

**Step 4: Commit**

```
git add tools/sam3_convert.c
git commit -m "convert: include tracker model weights in .sam3 format"
```

---

## Task 10: Reference Fixture Generator

Python script that generates binary tensor fixtures from the reference implementation for numerical validation.

**Files:**
- Create: `scripts/gen_tracker_fixtures.py`

**Step 1: Write fixture generator**

Python script that:
1. Loads the SAM3 model with `build_sam3_video_model()`
2. Creates a 3-frame test video (solid color frames)
3. Runs tracker on frame 0 with a point prompt, captures intermediate tensors
4. Propagates to frame 1, captures intermediate tensors
5. Saves all tensors as `.bin` files with a simple header (shape as int32s, then raw float32 data)

Fixture list (from design doc):
- Memory encoder: input mask, pixel features, downsampled, fused, output, pos_enc
- Memory attention: current features, memory tokens, memory pos, output
- SAM decoder: conditioned embed, point coords/labels, mask logits, obj pointer
- Integration: frame 0/1 backbone features, frame 0/1 masks
- Memory selection: obj scores, filtered indices

**Step 2: Commit**

```
git add scripts/gen_tracker_fixtures.py
git commit -m "scripts: add Python fixture generator for tracker numerical tests"
```

---

## Task 11: Numerical Validation Tests

Tests that compare C implementation outputs against Python reference fixtures.

**Files:**
- Create: `tests/test_tracker_fixtures.c`

**Step 1: Write fixture comparison tests**

Use the existing fixture comparison infrastructure (see `tests/test_fixture_compare.c`) to load `.bin` fixture files and compare against C outputs at each component boundary.

Key comparisons:
- Memory encoder output vs `mem_enc_vision_features.bin` (tolerance ~1e-4)
- Memory attention output vs `mem_attn_output.bin` (tolerance ~1e-3 due to fp16 accumulation)
- Full tracker frame 1 masks vs `tracker_frame1_masks.bin` (tolerance ~1e-2 for final masks)

These tests require fixture files to be present in `tests/data/fixtures/tracker/`. They skip gracefully if fixtures are not generated.

**Step 2: Commit**

```
git add tests/test_tracker_fixtures.c
git commit -m "test: add numerical validation against Python reference fixtures"
```

---

## Task 12: CLI Integration

Add a `--video` mode to the existing `sam3_main` CLI tool.

**Files:**
- Modify: `tools/sam3_main.c` (add --video flag and video tracking mode)

**Step 1: Add CLI flags**

Add to argument parsing:
- `--video <path>`: Path to video file or frame directory
- `--frame <idx>`: Frame index for point/box prompts (default 0)
- `--propagate <dir>`: "forward", "backward", or "both" (default "both")
- `--output-dir <path>`: Directory to write per-frame mask PNGs

**Step 2: Implement video mode**

When `--video` is provided:
1. Call `sam3_video_start()`
2. Parse point/box prompts from existing `--point`/`--box` flags, applied to `--frame`
3. Call `sam3_video_add_points()` or `sam3_video_add_box()`
4. Call `sam3_video_propagate()` with a callback that writes masks to `--output-dir`
5. Call `sam3_video_end()`

**Step 3: Test manually**

Run with test frames:
```
./sam3_main -m model.sam3 --video tests/data/frames --point 256,256,1 --frame 0 --output-dir /tmp/masks
```

**Step 4: Commit**

```
git add tools/sam3_main.c
git commit -m "cli: add --video mode for video tracking"
```

---

## Dependency Graph

```
Task 1 (types)
    |
    +---> Task 2 (memory bank) ----+
    |                              |
    +---> Task 3 (video I/O) ------+
    |                              |
    +---> Task 4 (mem encoder) ----+---> Task 6 (tracker core) ---> Task 8 (public API)
    |                              |         |                          |
    +---> Task 5 (mem attention) --+         v                          v
                                        Task 7 (session)          Task 12 (CLI)
                                                                       |
                                    Task 9 (converter) -----------+    |
                                    Task 10 (fixtures) --> Task 11 (validation)
```

Tasks 2, 3, 4, 5 can be done in parallel after Task 1.
Task 6 depends on 2, 4, 5.
Task 7 depends on 6 and 3.
Task 8 depends on 7.
Tasks 9 and 10 can be done any time after Task 6.
Task 11 depends on 10 and 8.
Task 12 depends on 8.
