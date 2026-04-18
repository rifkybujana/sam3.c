/*
 * src/util/video_internal.h - Internal helpers shared by video decode paths
 *
 * Exposes two pieces of implementation currently private to util/video.c
 * so that tools/cli_track.c can reuse them during the overlay-video pass 2:
 *   (1) Sorted frame-directory listing, for iterating PNG/JPG sources.
 *   (2) A raw RGB24 libav iterator that yields native-resolution frames
 *       without the resize/normalize that sam3_video_load performs.
 *
 * Private API — not part of the public sam3 ABI. Include path
 * "util/video_internal.h".
 *
 * Key types:  sam3_frame_dir_list, sam3_rgb_iter
 * Depends on: util/video.h
 * Used by:    src/util/video.c, tools/cli_track.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_UTIL_VIDEO_INTERNAL_H
#define SAM3_UTIL_VIDEO_INTERNAL_H

#include <stdint.h>

#include "sam3/sam3_types.h"

/*
 * sam3_frame_dir_list - Sorted list of image filenames in a directory.
 *
 * Filenames are heap-allocated (one malloc per name). Free with
 * sam3_frame_dir_list_free. dir_path is an unowned borrow of the caller's
 * string.
 */
struct sam3_frame_dir_list {
	const char  *dir_path;  /* borrowed */
	char       **names;     /* heap-allocated, sam3_frame_dir_list_free */
	int          n;
};

/*
 * sam3_frame_dir_list_open - List image files in a directory, sorted.
 *
 * Sorts numerically by leading integer in each filename, then
 * alphabetically. Accepts .jpg/.jpeg/.png/.ppm/.bmp. Caps at 4096 entries.
 *
 * Returns SAM3_OK on success, SAM3_EIO if the directory can't be opened
 * or is empty of image files, SAM3_ENOMEM on allocation failure.
 */
enum sam3_error sam3_frame_dir_list_open(const char *dir_path,
					 struct sam3_frame_dir_list *out);

void sam3_frame_dir_list_free(struct sam3_frame_dir_list *list);

/*
 * sam3_rgb_iter - Iterator producing native-resolution RGB24 frames.
 *
 * Opaque. Backed by libav when the input is a video file, by stb_image
 * when the input is a frame directory. Frame-directory iteration uses
 * the same sort order as sam3_frame_dir_list_open.
 */
struct sam3_rgb_iter;

/*
 * sam3_rgb_iter_open - Open a raw-RGB iterator over a video or frame dir.
 *
 * @path: Same path accepted by sam3_video_load (file or directory).
 * @out:  Receives a heap-allocated iterator on success.
 *
 * Returns SAM3_OK on success, SAM3_EIO on open failure, SAM3_ENOMEM on
 * allocation failure.
 */
enum sam3_error sam3_rgb_iter_open(const char *path,
				   struct sam3_rgb_iter **out);

/*
 * sam3_rgb_iter_next - Fetch the next frame.
 *
 * @it:      Iterator opened via sam3_rgb_iter_open.
 * @out_rgb: On success, receives a pointer to a frame-owned RGB24 buffer
 *           of size (*out_w) * (*out_h) * 3. Valid until the next call
 *           to sam3_rgb_iter_next or sam3_rgb_iter_close on this
 *           iterator. Caller must NOT free.
 * @out_w:   On success, receives frame width in pixels.
 * @out_h:   On success, receives frame height in pixels.
 * @out_eof: On success, set to 1 if end-of-stream was reached and
 *           *out_rgb is NULL; 0 otherwise.
 *
 * Returns SAM3_OK on success (including the EOF case), a negative
 * sam3_error on decode/read failure.
 */
enum sam3_error sam3_rgb_iter_next(struct sam3_rgb_iter *it,
				   const uint8_t **out_rgb,
				   int *out_w, int *out_h,
				   int *out_eof);

void sam3_rgb_iter_close(struct sam3_rgb_iter *it);

/*
 * sam3_rgb_iter_fps - Report the source frame rate.
 *
 * @it:      Iterator opened via sam3_rgb_iter_open.
 * @out_num: On return, frame-rate numerator (0 if unknown).
 * @out_den: On return, frame-rate denominator (1 if unknown).
 *
 * For frame-directory sources, always returns 0/1 (no native rate).
 * For video files, returns the stream's avg_frame_rate from libav.
 */
void sam3_rgb_iter_fps(const struct sam3_rgb_iter *it,
		       int *out_num, int *out_den);

#endif /* SAM3_UTIL_VIDEO_INTERNAL_H */
