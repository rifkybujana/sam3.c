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
 *
 * @path: Path to check
 *
 * Returns SAM3_VIDEO_FRAME_DIR if path is a directory, SAM3_VIDEO_MPEG if
 * file extension is .mpg or .mpeg, SAM3_VIDEO_UNKNOWN otherwise.
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
 * [-1, 1] with mean=0.5, std=0.5: (pixel/255.0 - 0.5) / 0.5
 *
 * Returns SAM3_OK on success, SAM3_EIO on load failure.
 */
enum sam3_error sam3_video_load(const char *path, int image_size,
				struct sam3_video_frames *out,
				struct sam3_arena *arena);

#endif /* SAM3_UTIL_VIDEO_H */
