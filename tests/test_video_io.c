/*
 * tests/test_video_io.c - Video frame loading tests
 *
 * Tests video source detection (directory vs file vs unknown) and
 * frame loading from image directories with resize and normalization.
 *
 * Key types:  sam3_video_frames, sam3_video_type
 * Depends on: util/video.h, core/alloc.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "util/video.h"
#include "core/alloc.h"
#include <string.h>

#ifndef SAM3_SOURCE_DIR
#define SAM3_SOURCE_DIR "."
#endif

static void test_detect_frame_dir(void)
{
	/* tests/data/frames should be detected as frame directory */
	char path[512];
	snprintf(path, sizeof(path), "%s/tests/data/frames",
		 SAM3_SOURCE_DIR);
	ASSERT_EQ(sam3_video_detect_type(path), SAM3_VIDEO_FRAME_DIR);
}

static void test_detect_video_file(void)
{
	ASSERT_EQ(sam3_video_detect_type("video.mpg"), SAM3_VIDEO_MPEG);
	ASSERT_EQ(sam3_video_detect_type("video.mpeg"), SAM3_VIDEO_MPEG);
	ASSERT_EQ(sam3_video_detect_type("VIDEO.MPG"), SAM3_VIDEO_MPEG);
}

static void test_detect_unknown(void)
{
	ASSERT_EQ(sam3_video_detect_type("photo.jpg"), SAM3_VIDEO_UNKNOWN);
	ASSERT_EQ(sam3_video_detect_type("data.bin"), SAM3_VIDEO_UNKNOWN);
	ASSERT_EQ(sam3_video_detect_type(NULL), SAM3_VIDEO_UNKNOWN);
}

static void test_load_frame_dir(void)
{
	char path[512];
	snprintf(path, sizeof(path), "%s/tests/data/frames",
		 SAM3_SOURCE_DIR);

	struct sam3_arena arena;
	enum sam3_error err = sam3_arena_init(&arena, 64 * 1024 * 1024);
	ASSERT_EQ(err, SAM3_OK);

	struct sam3_video_frames frames;
	err = sam3_video_load(path, 512, &frames, &arena);
	if (err == SAM3_EIO) {
		/* test data not present -- skip */
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

	/* Verify normalization: all values should be in [-1, 1] */
	float *data = (float *)frames.pixels[0]->data;
	int n_elem = 3 * 512 * 512;
	for (int i = 0; i < n_elem; i++) {
		ASSERT(data[i] >= -1.0f && data[i] <= 1.0f);
	}

	sam3_arena_free(&arena);
}

static void test_load_invalid_args(void)
{
	struct sam3_arena arena;
	enum sam3_error err = sam3_arena_init(&arena, 1024 * 1024);
	ASSERT_EQ(err, SAM3_OK);

	struct sam3_video_frames frames;

	/* NULL path */
	err = sam3_video_load(NULL, 512, &frames, &arena);
	ASSERT_EQ(err, SAM3_EINVAL);

	/* zero image_size */
	err = sam3_video_load("some/path", 0, &frames, &arena);
	ASSERT_EQ(err, SAM3_EINVAL);

	sam3_arena_free(&arena);
}

int main(void)
{
	test_detect_frame_dir();
	test_detect_video_file();
	test_detect_unknown();
	test_load_frame_dir();
	test_load_invalid_args();
	TEST_REPORT();
}
