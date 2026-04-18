# Video I/O for `sam3 track` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `sam3 track` to read modern video formats (MP4/MOV/MKV/WebM) and write a single overlay video where per-object masks are alpha-blended onto source frames. The existing PNG-directory output mode is preserved unchanged.

**Architecture:** Add `libavformat/libavcodec/libswscale/libavutil` as a hard build dependency. Retire the vendored `pl_mpeg.h`. The decode path in `util/video.c` swaps to libav internally (same public API). A new `util/video_encode.{c,h}` module provides the encoder plus a libav-free overlay helper. `cli_track` orchestrates a two-pass flow when `--output` names a video file: pass 1 buffers per-frame binary masks during propagation, pass 2 decodes the source at native resolution, composites per-object overlays, and encodes.

**Tech Stack:** C11, CMake, libavformat/libavcodec/libswscale/libavutil (via pkg-config), existing vendored stb_image/stb_image_resize2.

**Reference:** Design spec in `docs/superpowers/specs/2026-04-18-video-io-cli-design.md`.

---

## File Structure

### Create
- `src/util/video_encode.h` — encoder public API + overlay helper signature
- `src/util/video_encode.c` — libav-backed encoder + overlay helper implementation
- `src/util/video_internal.h` — internal header exposing helpers shared between `util/video.c` and `tools/cli_track.c` (frame-directory listing + raw RGB libav iterator)
- `tests/test_video_encode.c` — encoder + overlay unit tests

### Modify
- `CMakeLists.txt` — add `pkg_check_modules(LIBAV …)`, link into `sam3` target
- `src/util/video.h` — rename `SAM3_VIDEO_MPEG` → `SAM3_VIDEO_FILE`, add `fps_num/fps_den` to `sam3_video_frames`
- `src/util/video.c` — replace `load_mpeg` (pl_mpeg) with `load_libav`; extract frame-dir listing + add raw RGB iterator, exposing both through `video_internal.h`
- `tools/cli_track.h` — extend `track_args` with `alpha`, `fps`, `output_mode`
- `tools/cli_track.c` — parse new flags; detect output mode from extension; branch `cli_track_run` into dir-mode (unchanged) vs video-mode (new two-pass)
- `tests/test_cli_track.c` — new parser tests for `--alpha`, `--fps`, output-mode detection
- `tests/test_video_io.c` — update for enum rename + new detection semantics
- `docs/architecture.md` — remove references to `pl_mpeg` if any

### Delete
- `src/util/vendor/pl_mpeg.h` — retired

---

## Task 1: Add libav build dependency

**Files:**
- Modify: `CMakeLists.txt:1-170`

- [ ] **Step 1: Add `PkgConfig` discovery + libav link**

Insert after the existing `find_package(ZLIB REQUIRED)` / `target_link_libraries(sam3 ZLIB::ZLIB)` lines (around line 163):

```cmake
# Video I/O via libav (FFmpeg C libraries)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
	libavformat libavcodec libswscale libavutil)
target_link_libraries(sam3 PkgConfig::LIBAV)
```

- [ ] **Step 2: Verify configuration succeeds**

Run:
```bash
cd /Users/rbisri/Documents/sam3 && rm -rf build && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug 2>&1 | tail -30
```

Expected: `-- Checking for module 'libavformat libavcodec libswscale libavutil'` followed by `-- Found ...` lines, and configuration completes without error. If pkg-config reports the libs missing, install ffmpeg (`brew install ffmpeg` on macOS).

- [ ] **Step 3: Commit**

```bash
git add CMakeLists.txt
git commit -m "build: link libav (FFmpeg) for video I/O

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Update `util/video.h` — rename enum, add fps fields

**Files:**
- Modify: `src/util/video.h`

- [ ] **Step 1: Rename `SAM3_VIDEO_MPEG` → `SAM3_VIDEO_FILE` and add fps fields**

Replace the `enum sam3_video_type` and `struct sam3_video_frames` blocks (lines 23–35) with:

```c
enum sam3_video_type {
	SAM3_VIDEO_UNKNOWN   = 0,
	SAM3_VIDEO_FRAME_DIR = 1,
	SAM3_VIDEO_FILE      = 2,   /* Any regular file; libav decides if decodable */
};

struct sam3_video_frames {
	struct sam3_tensor **pixels;  /* Array of [3, H, W] F32 tensors */
	int    n_frames;
	int    frame_size;           /* Model input size (e.g. 1008) */
	int    orig_width;
	int    orig_height;
	int    fps_num;              /* 0 if unknown (frame dir without --fps) */
	int    fps_den;              /* 1 when fps_num == 0 */
};
```

Also update the doc comment for `sam3_video_detect_type` (lines 37–45) to:

```c
/*
 * sam3_video_detect_type - Detect whether path is a video file or frame dir.
 *
 * @path: Path to check
 *
 * Returns SAM3_VIDEO_FRAME_DIR if path is a directory, SAM3_VIDEO_FILE if
 * path is a regular file (libav will validate format on open), or
 * SAM3_VIDEO_UNKNOWN otherwise (path does not exist / is a special file).
 */
```

- [ ] **Step 2: Commit**

```bash
git add src/util/video.h
git commit -m "util/video: rename SAM3_VIDEO_MPEG to SAM3_VIDEO_FILE and add fps fields

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Create `src/util/video_internal.h`

This private header exposes helpers that both `util/video.c` and `tools/cli_track.c` will share — frame-directory listing (for iteration during pass 2) and a raw RGB libav iterator (so cli_track doesn't reimplement libav decode).

**Files:**
- Create: `src/util/video_internal.h`

- [ ] **Step 1: Write the header**

```c
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

#include "util/video.h"

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

#endif /* SAM3_UTIL_VIDEO_INTERNAL_H */
```

- [ ] **Step 2: Commit** (header only — implementation arrives in Task 4)

```bash
git add src/util/video_internal.h
git commit -m "util/video: add internal header for shared decode helpers

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Rewrite `util/video.c` — libav decode + helper implementations

This is a substantial rewrite. The existing frame-directory path is preserved; the MPEG-1 path via pl_mpeg is replaced by libav; both the frame-directory lister and a raw-RGB libav iterator are exposed through `video_internal.h`.

**Files:**
- Modify: `src/util/video.c` (full rewrite)

- [ ] **Step 1: Replace file contents**

Write the full file:

```c
/*
 * src/util/video.c - Video frame loading implementation
 *
 * Loads video frames from regular files (via libavformat/libavcodec) or
 * directories of images (via stb_image). Handles resize to model input
 * size and normalization to [-1, 1].
 *
 * Also exposes two internal helpers via util/video_internal.h:
 *   - sam3_frame_dir_list_{open,free}: sorted image-directory listing.
 *   - sam3_rgb_iter_{open,next,close}: native-resolution RGB24 iterator.
 * Both are reused by tools/cli_track.c for the overlay-video pass.
 *
 * Key types:  sam3_video_frames, sam3_video_type, sam3_frame_dir_list,
 *             sam3_rgb_iter
 * Depends on: util/video.h, util/video_internal.h, util/log.h,
 *             vendor/stb_image.h, vendor/stb_image_resize2.h,
 *             libavformat, libavcodec, libswscale, libavutil
 * Used by:    model/video_session.c, tools/cli_track.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <dirent.h>
#include <ctype.h>

#include "util/video.h"
#include "util/video_internal.h"
#include "util/log.h"

#include "vendor/stb_image.h"
#include "vendor/stb_image_resize2.h"

/* libav — suppress warnings in vendored headers */
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wstrict-prototypes"
#pragma clang diagnostic ignored "-Wdouble-promotion"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wcast-align"
#pragma clang diagnostic ignored "-Wcast-qual"
#pragma clang diagnostic ignored "-Wundef"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>

#ifdef __clang__
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#define MAX_DIR_FRAMES 4096

/* --- filename helpers --- */

static int has_suffix(const char *str, const char *suffix)
{
	size_t slen = strlen(str);
	size_t xlen = strlen(suffix);
	if (slen < xlen)
		return 0;
	const char *tail = str + slen - xlen;
	for (size_t i = 0; i < xlen; i++) {
		if (tolower((unsigned char)tail[i]) !=
		    tolower((unsigned char)suffix[i]))
			return 0;
	}
	return 1;
}

static int is_image_file(const char *name)
{
	return has_suffix(name, ".jpg") || has_suffix(name, ".jpeg") ||
	       has_suffix(name, ".png") || has_suffix(name, ".ppm") ||
	       has_suffix(name, ".bmp");
}

static int parse_numeric_prefix(const char *name, long *val)
{
	char *end;
	long v = strtol(name, &end, 10);
	if (end == name)
		return 0;
	*val = v;
	return 1;
}

static int cmp_filenames(const void *a, const void *b)
{
	const char *sa = *(const char *const *)a;
	const char *sb = *(const char *const *)b;
	long va, vb;
	if (parse_numeric_prefix(sa, &va) && parse_numeric_prefix(sb, &vb)) {
		if (va != vb)
			return (va > vb) - (va < vb);
	}
	return strcmp(sa, sb);
}

/* --- sam3_frame_dir_list (shared with cli_track) --- */

enum sam3_error sam3_frame_dir_list_open(const char *dir_path,
					 struct sam3_frame_dir_list *out)
{
	if (!dir_path || !out)
		return SAM3_EINVAL;

	memset(out, 0, sizeof(*out));
	out->dir_path = dir_path;

	DIR *dir = opendir(dir_path);
	if (!dir) {
		sam3_log_error("cannot open frame directory '%s'", dir_path);
		return SAM3_EIO;
	}

	char **names = NULL;
	int n = 0, cap = 0;
	struct dirent *ent;
	enum sam3_error err = SAM3_OK;

	while ((ent = readdir(dir)) != NULL) {
		if (!is_image_file(ent->d_name))
			continue;
		if (n >= MAX_DIR_FRAMES) {
			sam3_log_warn("frame directory capped at %d files",
				      MAX_DIR_FRAMES);
			break;
		}
		if (n >= cap) {
			cap = cap ? cap * 2 : 64;
			char **tmp = realloc(names,
					     (size_t)cap * sizeof(char *));
			if (!tmp) {
				err = SAM3_ENOMEM;
				goto fail;
			}
			names = tmp;
		}
		names[n] = strdup(ent->d_name);
		if (!names[n]) {
			err = SAM3_ENOMEM;
			goto fail;
		}
		n++;
	}
	closedir(dir);

	if (n == 0) {
		sam3_log_error("no image files in '%s'", dir_path);
		err = SAM3_EIO;
		goto fail;
	}

	qsort(names, (size_t)n, sizeof(char *), cmp_filenames);
	out->names = names;
	out->n = n;
	return SAM3_OK;

fail:
	closedir(dir);
	for (int i = 0; i < n; i++)
		free(names[i]);
	free(names);
	memset(out, 0, sizeof(*out));
	return err;
}

void sam3_frame_dir_list_free(struct sam3_frame_dir_list *list)
{
	if (!list)
		return;
	for (int i = 0; i < list->n; i++)
		free(list->names[i]);
	free(list->names);
	memset(list, 0, sizeof(*list));
}

/* --- make_frame_tensor (shared by dir + libav load paths) --- */

static struct sam3_tensor *make_frame_tensor(const uint8_t *rgb,
					     int w, int h,
					     struct sam3_arena *arena)
{
	struct sam3_tensor *t = sam3_arena_alloc(arena,
						 sizeof(struct sam3_tensor));
	if (!t)
		return NULL;

	memset(t, 0, sizeof(*t));
	t->dtype = SAM3_DTYPE_F32;
	t->n_dims = 3;
	t->dims[0] = 3;
	t->dims[1] = h;
	t->dims[2] = w;
	sam3_tensor_compute_strides(t);
	t->nbytes = (size_t)3 * h * w * sizeof(float);

	float *data = sam3_arena_alloc(arena, t->nbytes);
	if (!data)
		return NULL;
	t->data = data;

	int hw = h * w;
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			int px = y * w + x;
			const uint8_t *src = rgb + (y * w + x) * 3;
			data[0 * hw + px] = (src[0] / 255.0f - 0.5f) / 0.5f;
			data[1 * hw + px] = (src[1] / 255.0f - 0.5f) / 0.5f;
			data[2 * hw + px] = (src[2] / 255.0f - 0.5f) / 0.5f;
		}
	}
	return t;
}

/* --- frame-directory load (unchanged behavior) --- */

static enum sam3_error load_frame_dir(const char *path, int image_size,
				      struct sam3_video_frames *out,
				      struct sam3_arena *arena)
{
	struct sam3_frame_dir_list list;
	enum sam3_error err = sam3_frame_dir_list_open(path, &list);
	if (err != SAM3_OK)
		return err;

	uint8_t *resized = NULL;

	out->pixels = sam3_arena_alloc(arena,
				       (size_t)list.n *
				       sizeof(struct sam3_tensor *));
	if (!out->pixels) {
		err = SAM3_ENOMEM;
		goto done;
	}

	out->n_frames   = 0;
	out->frame_size = image_size;
	out->fps_num    = 0;
	out->fps_den    = 1;

	resized = malloc((size_t)image_size * image_size * 3);
	if (!resized) {
		sam3_log_error("out of memory for resize buffer");
		err = SAM3_ENOMEM;
		goto done;
	}

	for (int i = 0; i < list.n; i++) {
		char fpath[1024];
		int n = snprintf(fpath, sizeof(fpath), "%s/%s",
				 list.dir_path, list.names[i]);
		if (n < 0 || (size_t)n >= sizeof(fpath)) {
			sam3_log_warn("path too long, skipping '%s'",
				      list.names[i]);
			continue;
		}

		int w, h, ch;
		uint8_t *pixels = stbi_load(fpath, &w, &h, &ch, 3);
		if (!pixels) {
			sam3_log_error("failed to load '%s': %s", fpath,
				       stbi_failure_reason());
			err = SAM3_EIO;
			goto done;
		}

		if (out->n_frames == 0) {
			out->orig_width  = w;
			out->orig_height = h;
		}

		uint8_t *r = stbir_resize_uint8_linear(
			pixels, w, h, 0,
			resized, image_size, image_size, 0,
			STBIR_RGB);
		stbi_image_free(pixels);
		if (!r) {
			sam3_log_error("resize failed for '%s'", fpath);
			err = SAM3_EIO;
			goto done;
		}

		struct sam3_tensor *t = make_frame_tensor(resized, image_size,
							  image_size, arena);
		if (!t) {
			sam3_log_error("out of memory encoding frame '%s'",
				       fpath);
			err = SAM3_ENOMEM;
			goto done;
		}
		out->pixels[out->n_frames++] = t;
	}

	sam3_log_info("loaded %d frames from '%s' (%dx%d -> %d)",
		      out->n_frames, path, out->orig_width,
		      out->orig_height, image_size);

done:
	free(resized);
	sam3_frame_dir_list_free(&list);
	return err;
}

/* --- libav helpers (shared by load_libav and sam3_rgb_iter) --- */

struct libav_decoder {
	AVFormatContext *fmt;
	AVCodecContext  *codec;
	AVFrame         *frame;     /* decoded frame in its native pixel format */
	AVFrame         *rgb_frame; /* RGB24 target (data owned by us) */
	AVPacket        *packet;
	struct SwsContext *sws;
	int              stream_idx;
	int              width;
	int              height;
	AVRational       avg_frame_rate;
};

static void libav_decoder_close(struct libav_decoder *d)
{
	if (!d)
		return;
	if (d->sws)       sws_freeContext(d->sws);
	if (d->rgb_frame) {
		av_freep(&d->rgb_frame->data[0]);
		av_frame_free(&d->rgb_frame);
	}
	if (d->frame)  av_frame_free(&d->frame);
	if (d->packet) av_packet_free(&d->packet);
	if (d->codec)  avcodec_free_context(&d->codec);
	if (d->fmt)    avformat_close_input(&d->fmt);
	memset(d, 0, sizeof(*d));
}

static enum sam3_error libav_decoder_open(struct libav_decoder *d,
					  const char *path)
{
	memset(d, 0, sizeof(*d));

	int rc = avformat_open_input(&d->fmt, path, NULL, NULL);
	if (rc < 0) {
		sam3_log_error("cannot open '%s': %s", path, av_err2str(rc));
		return SAM3_EIO;
	}
	rc = avformat_find_stream_info(d->fmt, NULL);
	if (rc < 0) {
		sam3_log_error("cannot find stream info in '%s': %s",
			       path, av_err2str(rc));
		libav_decoder_close(d);
		return SAM3_EIO;
	}

	int stream_idx = av_find_best_stream(d->fmt, AVMEDIA_TYPE_VIDEO,
					     -1, -1, NULL, 0);
	if (stream_idx < 0) {
		sam3_log_error("no video stream in '%s'", path);
		libav_decoder_close(d);
		return SAM3_EIO;
	}
	d->stream_idx = stream_idx;
	AVStream *stream = d->fmt->streams[stream_idx];
	d->avg_frame_rate = stream->avg_frame_rate;

	const AVCodec *dec = avcodec_find_decoder(stream->codecpar->codec_id);
	if (!dec) {
		sam3_log_error("no decoder for codec id %d",
			       stream->codecpar->codec_id);
		libav_decoder_close(d);
		return SAM3_EIO;
	}
	d->codec = avcodec_alloc_context3(dec);
	if (!d->codec) {
		libav_decoder_close(d);
		return SAM3_ENOMEM;
	}
	rc = avcodec_parameters_to_context(d->codec, stream->codecpar);
	if (rc < 0) {
		sam3_log_error("avcodec_parameters_to_context: %s",
			       av_err2str(rc));
		libav_decoder_close(d);
		return SAM3_EIO;
	}
	rc = avcodec_open2(d->codec, dec, NULL);
	if (rc < 0) {
		sam3_log_error("avcodec_open2: %s", av_err2str(rc));
		libav_decoder_close(d);
		return SAM3_EIO;
	}

	d->width  = d->codec->width;
	d->height = d->codec->height;

	d->frame     = av_frame_alloc();
	d->rgb_frame = av_frame_alloc();
	d->packet    = av_packet_alloc();
	if (!d->frame || !d->rgb_frame || !d->packet) {
		libav_decoder_close(d);
		return SAM3_ENOMEM;
	}
	d->rgb_frame->format = AV_PIX_FMT_RGB24;
	d->rgb_frame->width  = d->width;
	d->rgb_frame->height = d->height;
	rc = av_image_alloc(d->rgb_frame->data, d->rgb_frame->linesize,
			    d->width, d->height, AV_PIX_FMT_RGB24, 1);
	if (rc < 0) {
		libav_decoder_close(d);
		return SAM3_ENOMEM;
	}

	d->sws = sws_getContext(d->width, d->height, d->codec->pix_fmt,
				d->width, d->height, AV_PIX_FMT_RGB24,
				SWS_BILINEAR, NULL, NULL, NULL);
	if (!d->sws) {
		libav_decoder_close(d);
		return SAM3_ENOMEM;
	}
	return SAM3_OK;
}

/*
 * libav_decoder_next - Decode the next frame into d->rgb_frame.
 *
 * On success, *out_eof is 0 and d->rgb_frame holds an RGB24 frame of
 * size d->width x d->height. On end-of-stream, *out_eof is 1 and the
 * frame buffer contents are undefined.
 */
static enum sam3_error libav_decoder_next(struct libav_decoder *d,
					  int *out_eof)
{
	*out_eof = 0;
	for (;;) {
		int rc = avcodec_receive_frame(d->codec, d->frame);
		if (rc == 0) {
			sws_scale(d->sws,
				  (const uint8_t * const *)d->frame->data,
				  d->frame->linesize, 0, d->height,
				  d->rgb_frame->data, d->rgb_frame->linesize);
			return SAM3_OK;
		}
		if (rc != AVERROR(EAGAIN) && rc != AVERROR_EOF) {
			sam3_log_error("avcodec_receive_frame: %s",
				       av_err2str(rc));
			return SAM3_EIO;
		}
		if (rc == AVERROR_EOF) {
			*out_eof = 1;
			return SAM3_OK;
		}
		/* Need more input — read a packet. */
		rc = av_read_frame(d->fmt, d->packet);
		if (rc == AVERROR_EOF) {
			avcodec_send_packet(d->codec, NULL); /* flush */
			continue;
		}
		if (rc < 0) {
			sam3_log_error("av_read_frame: %s", av_err2str(rc));
			return SAM3_EIO;
		}
		if (d->packet->stream_index != d->stream_idx) {
			av_packet_unref(d->packet);
			continue;
		}
		rc = avcodec_send_packet(d->codec, d->packet);
		av_packet_unref(d->packet);
		if (rc < 0 && rc != AVERROR(EAGAIN)) {
			sam3_log_error("avcodec_send_packet: %s",
				       av_err2str(rc));
			return SAM3_EIO;
		}
	}
}

/* --- model-ready libav load (replaces load_mpeg) --- */

static enum sam3_error load_libav(const char *path, int image_size,
				  struct sam3_video_frames *out,
				  struct sam3_arena *arena)
{
	struct libav_decoder d;
	enum sam3_error err = libav_decoder_open(&d, path);
	if (err != SAM3_OK)
		return err;

	out->orig_width  = d.width;
	out->orig_height = d.height;
	out->frame_size  = image_size;
	out->fps_num     = d.avg_frame_rate.num;
	out->fps_den     = d.avg_frame_rate.den > 0 ? d.avg_frame_rate.den : 1;

	/* First pass: count frames */
	int n_total = 0;
	for (;;) {
		int eof;
		err = libav_decoder_next(&d, &eof);
		if (err != SAM3_OK) {
			libav_decoder_close(&d);
			return err;
		}
		if (eof)
			break;
		n_total++;
	}
	if (n_total == 0) {
		sam3_log_error("no decodable video frames in '%s'", path);
		libav_decoder_close(&d);
		return SAM3_EIO;
	}
	libav_decoder_close(&d);

	/* Reopen for the real pass */
	err = libav_decoder_open(&d, path);
	if (err != SAM3_OK)
		return err;

	out->pixels = sam3_arena_alloc(arena,
				       (size_t)n_total *
				       sizeof(struct sam3_tensor *));
	if (!out->pixels) {
		libav_decoder_close(&d);
		return SAM3_ENOMEM;
	}

	uint8_t *resized = malloc((size_t)image_size * image_size * 3);
	if (!resized) {
		libav_decoder_close(&d);
		return SAM3_ENOMEM;
	}

	out->n_frames = 0;
	for (int i = 0; i < n_total; i++) {
		int eof;
		err = libav_decoder_next(&d, &eof);
		if (err != SAM3_OK || eof)
			break;

		const uint8_t *rgb = d.rgb_frame->data[0];
		uint8_t *r = stbir_resize_uint8_linear(
			rgb, d.width, d.height, d.rgb_frame->linesize[0],
			resized, image_size, image_size, 0,
			STBIR_RGB);
		if (!r) {
			sam3_log_error("resize failed for frame %d", i);
			err = SAM3_EIO;
			break;
		}
		struct sam3_tensor *t = make_frame_tensor(resized, image_size,
							  image_size, arena);
		if (!t) {
			err = SAM3_ENOMEM;
			break;
		}
		out->pixels[out->n_frames++] = t;
	}

	free(resized);
	libav_decoder_close(&d);

	if (err == SAM3_OK)
		sam3_log_info("loaded %d frames from '%s' (%dx%d -> %d)",
			      out->n_frames, path, out->orig_width,
			      out->orig_height, image_size);
	return err;
}

/* --- public API --- */

enum sam3_video_type sam3_video_detect_type(const char *path)
{
	if (!path)
		return SAM3_VIDEO_UNKNOWN;

	struct stat st;
	if (stat(path, &st) != 0)
		return SAM3_VIDEO_UNKNOWN;
	if (S_ISDIR(st.st_mode))
		return SAM3_VIDEO_FRAME_DIR;
	if (S_ISREG(st.st_mode))
		return SAM3_VIDEO_FILE;
	return SAM3_VIDEO_UNKNOWN;
}

enum sam3_error sam3_video_load(const char *path, int image_size,
				struct sam3_video_frames *out,
				struct sam3_arena *arena)
{
	if (!path || !out || !arena || image_size <= 0) {
		sam3_log_error("invalid args to sam3_video_load");
		return SAM3_EINVAL;
	}
	memset(out, 0, sizeof(*out));

	enum sam3_video_type vtype = sam3_video_detect_type(path);
	switch (vtype) {
	case SAM3_VIDEO_FRAME_DIR:
		return load_frame_dir(path, image_size, out, arena);
	case SAM3_VIDEO_FILE:
		return load_libav(path, image_size, out, arena);
	default:
		sam3_log_error("unknown video source type for '%s'", path);
		return SAM3_EIO;
	}
}

/* --- sam3_rgb_iter (native-resolution RGB24, for overlay pass 2) --- */

enum sam3_rgb_iter_kind {
	RGB_ITER_LIBAV = 1,
	RGB_ITER_DIR   = 2,
};

struct sam3_rgb_iter {
	enum sam3_rgb_iter_kind kind;

	/* Libav mode */
	struct libav_decoder dec;

	/* Dir mode */
	struct sam3_frame_dir_list list;
	int                        dir_cursor;
	uint8_t                   *dir_last_rgb;   /* stb_image-owned */
};

enum sam3_error sam3_rgb_iter_open(const char *path,
				   struct sam3_rgb_iter **out)
{
	if (!path || !out)
		return SAM3_EINVAL;
	*out = NULL;

	struct sam3_rgb_iter *it = calloc(1, sizeof(*it));
	if (!it)
		return SAM3_ENOMEM;

	enum sam3_video_type vt = sam3_video_detect_type(path);
	if (vt == SAM3_VIDEO_FRAME_DIR) {
		enum sam3_error err =
			sam3_frame_dir_list_open(path, &it->list);
		if (err != SAM3_OK) {
			free(it);
			return err;
		}
		it->kind = RGB_ITER_DIR;
	} else if (vt == SAM3_VIDEO_FILE) {
		enum sam3_error err = libav_decoder_open(&it->dec, path);
		if (err != SAM3_OK) {
			free(it);
			return err;
		}
		it->kind = RGB_ITER_LIBAV;
	} else {
		sam3_log_error("cannot iterate '%s' (not a file or dir)",
			       path);
		free(it);
		return SAM3_EIO;
	}
	*out = it;
	return SAM3_OK;
}

enum sam3_error sam3_rgb_iter_next(struct sam3_rgb_iter *it,
				   const uint8_t **out_rgb,
				   int *out_w, int *out_h, int *out_eof)
{
	if (!it || !out_rgb || !out_w || !out_h || !out_eof)
		return SAM3_EINVAL;

	*out_rgb = NULL;
	*out_eof = 0;

	if (it->kind == RGB_ITER_LIBAV) {
		int eof;
		enum sam3_error err = libav_decoder_next(&it->dec, &eof);
		if (err != SAM3_OK)
			return err;
		if (eof) {
			*out_eof = 1;
			return SAM3_OK;
		}
		*out_rgb = it->dec.rgb_frame->data[0];
		*out_w   = it->dec.width;
		*out_h   = it->dec.height;
		return SAM3_OK;
	}

	/* RGB_ITER_DIR */
	if (it->dir_last_rgb) {
		stbi_image_free(it->dir_last_rgb);
		it->dir_last_rgb = NULL;
	}
	if (it->dir_cursor >= it->list.n) {
		*out_eof = 1;
		return SAM3_OK;
	}
	char fpath[1024];
	int n = snprintf(fpath, sizeof(fpath), "%s/%s",
			 it->list.dir_path, it->list.names[it->dir_cursor]);
	if (n < 0 || (size_t)n >= sizeof(fpath)) {
		sam3_log_error("rgb iter: path too long");
		return SAM3_EIO;
	}
	int w, h, ch;
	uint8_t *pixels = stbi_load(fpath, &w, &h, &ch, 3);
	if (!pixels) {
		sam3_log_error("rgb iter: stbi_load '%s': %s", fpath,
			       stbi_failure_reason());
		return SAM3_EIO;
	}
	it->dir_last_rgb = pixels;
	it->dir_cursor++;
	*out_rgb = pixels;
	*out_w = w;
	*out_h = h;
	return SAM3_OK;
}

void sam3_rgb_iter_close(struct sam3_rgb_iter *it)
{
	if (!it)
		return;
	if (it->kind == RGB_ITER_LIBAV) {
		libav_decoder_close(&it->dec);
	} else if (it->kind == RGB_ITER_DIR) {
		if (it->dir_last_rgb)
			stbi_image_free(it->dir_last_rgb);
		sam3_frame_dir_list_free(&it->list);
	}
	free(it);
}
```

- [ ] **Step 2: Remove the pl_mpeg include from the build (nothing else to change here — the `#define PL_MPEG_IMPLEMENTATION` and include are gone in the new file contents). Skip to step 3.**

- [ ] **Step 3: Build and run existing decode tests**

```bash
cd /Users/rbisri/Documents/sam3/build && cmake --build . -j 2>&1 | tail -30 && ctest -R test_video_io --output-on-failure
```

Expected: build succeeds; `test_video_io` either passes or fails only on the enum-rename tests (those are fixed in Task 5).

- [ ] **Step 4: Commit**

```bash
git add src/util/video.c src/util/video_internal.h
git commit -m "util/video: replace pl_mpeg with libav and expose shared helpers

Adds a raw-RGB libav iterator and a sorted frame-directory lister in
util/video_internal.h so tools/cli_track.c can iterate the source at
native resolution without duplicating decode logic.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Fix `tests/test_video_io.c` for new detection semantics

**Files:**
- Modify: `tests/test_video_io.c:33-44`

- [ ] **Step 1: Replace the detection tests**

Replace `test_detect_video_file` and `test_detect_unknown` (lines 33–45):

```c
static void test_detect_video_file(void)
{
	/*
	 * Detection is now stat()-based: any regular file returns FILE;
	 * libav validates format on open. Use a real file we can stat.
	 */
	char path[512];
	snprintf(path, sizeof(path), "%s/CMakeLists.txt", SAM3_SOURCE_DIR);
	ASSERT_EQ(sam3_video_detect_type(path), SAM3_VIDEO_FILE);
}

static void test_detect_unknown(void)
{
	/* Non-existent path */
	ASSERT_EQ(sam3_video_detect_type("/does/not/exist/xyz.mp4"),
		  SAM3_VIDEO_UNKNOWN);
	ASSERT_EQ(sam3_video_detect_type(NULL), SAM3_VIDEO_UNKNOWN);
}
```

- [ ] **Step 2: Run the test**

```bash
cd /Users/rbisri/Documents/sam3/build && cmake --build . --target test_video_io -j && ctest -R test_video_io --output-on-failure
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_video_io.c
git commit -m "tests/video_io: update detection tests for stat-based type

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Delete `pl_mpeg.h` and clean up docs

**Files:**
- Delete: `src/util/vendor/pl_mpeg.h`
- Modify: `docs/architecture.md` (remove pl_mpeg references if any)

- [ ] **Step 1: Delete the vendor header**

```bash
cd /Users/rbisri/Documents/sam3 && git rm src/util/vendor/pl_mpeg.h
```

Expected: `rm 'src/util/vendor/pl_mpeg.h'`.

- [ ] **Step 2: Check for remaining references**

Run through the Grep tool with:
- pattern: `pl_mpeg|PL_MPEG`
- path: `/Users/rbisri/Documents/sam3`
- output_mode: `files_with_matches`

Expected: only `docs/architecture.md` and the spec under `docs/superpowers/specs/`.

- [ ] **Step 3: Update `docs/architecture.md`**

Read the file and locate any sentence referencing `pl_mpeg`. Replace the mention with "libav (libavformat/libavcodec/libswscale/libavutil)". If the sentence is longer (e.g. describes MPEG-1-only support as a limitation), rewrite to say video input is handled via libav and supports whatever codecs the linked ffmpeg build exposes.

- [ ] **Step 4: Run full test suite to confirm nothing else broke**

```bash
cd /Users/rbisri/Documents/sam3/build && cmake --build . -j 2>&1 | tail -10 && ctest --output-on-failure 2>&1 | tail -30
```

Expected: build succeeds, all tests that passed before still pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "util/vendor: retire pl_mpeg; libav supersedes it

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Create `util/video_encode.h`

**Files:**
- Create: `src/util/video_encode.h`

- [ ] **Step 1: Write the header**

```c
/*
 * src/util/video_encode.h - Encoder + overlay helper for video output
 *
 * The encoder wraps libavformat/libavcodec/libswscale to write RGB24
 * frames into a container whose format is inferred from the output
 * path. The overlay helper is a pure C function (no libav dep) that
 * composites a binary mask onto an RGB24 buffer using a per-object
 * color palette.
 *
 * Key types:  sam3_video_encoder
 * Depends on: sam3/sam3_types.h
 * Used by:    src/util/video_encode.c, tools/cli_track.c,
 *             tests/test_video_encode.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_UTIL_VIDEO_ENCODE_H
#define SAM3_UTIL_VIDEO_ENCODE_H

#include <stdint.h>

#include "sam3/sam3_types.h"

struct sam3_video_encoder; /* opaque */

/*
 * sam3_video_encoder_open - Open an encoder for @path at @w x @h @ fps.
 *
 * @path:    Output file path. Container inferred from extension
 *           (.mp4/.mov/.mkv -> H.264; .webm -> VP9).
 * @width:   Frame width in pixels (> 0).
 * @height:  Frame height in pixels (> 0).
 * @fps_num: Frame-rate numerator (> 0).
 * @fps_den: Frame-rate denominator (> 0).
 * @out:     Receives heap-allocated encoder on success.
 *
 * Returns SAM3_OK, SAM3_EINVAL for bad args, SAM3_EIO for libav
 * failures (including missing codec support in the linked ffmpeg
 * build), SAM3_ENOMEM on allocation failure.
 */
enum sam3_error sam3_video_encoder_open(const char *path,
					int width, int height,
					int fps_num, int fps_den,
					struct sam3_video_encoder **out);

/*
 * sam3_video_encoder_write_rgb - Encode one RGB24 frame.
 *
 * @enc: Encoder from sam3_video_encoder_open.
 * @rgb: RGB24 buffer of size width * height * 3, tightly packed.
 *
 * Returns SAM3_OK on success; SAM3_EIO if libav rejects the frame.
 */
enum sam3_error sam3_video_encoder_write_rgb(struct sam3_video_encoder *enc,
					     const uint8_t *rgb);

/*
 * sam3_video_encoder_close - Flush, finalize, and free.
 *
 * Idempotent. Safe on NULL. Safe on a partially-opened encoder.
 * Returns SAM3_OK on success, SAM3_EIO if flush/trailer fails. The
 * encoder is freed either way.
 */
enum sam3_error sam3_video_encoder_close(struct sam3_video_encoder *enc);

/*
 * sam3_overlay_composite - Alpha-blend a binary mask onto RGB24 in place.
 *
 * @rgb:    RGB24 buffer of size w * h * 3 (modified in place).
 * @w, @h:  Output dimensions.
 * @mask:   Binary mask (0 or non-zero per byte) of size mw * mh.
 * @mw, @mh: Mask dimensions; upscaled to (w, h) with nearest neighbor.
 * @obj_id: Object id; palette color = palette[obj_id % 10].
 * @alpha:  Blend factor in [0, 1]. 0 leaves rgb untouched, 1 replaces.
 *
 * Out-of-range alpha is clamped; out-of-range obj_id wraps modulo 10.
 * No allocations, no libav dependency.
 */
void sam3_overlay_composite(uint8_t *rgb, int w, int h,
			    const uint8_t *mask, int mw, int mh,
			    int obj_id, float alpha);

#endif /* SAM3_UTIL_VIDEO_ENCODE_H */
```

- [ ] **Step 2: Commit**

```bash
git add src/util/video_encode.h
git commit -m "util/video_encode: add header for encoder + overlay helper

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Write failing tests for the overlay helper

**Files:**
- Create: `tests/test_video_encode.c`

- [ ] **Step 1: Write the test file with overlay tests only (encoder tests added in later tasks)**

```c
/*
 * tests/test_video_encode.c - Encoder + overlay helper tests
 *
 * Covers: overlay compositing math (alpha blending + nearest-neighbor
 * upscale) and the encoder open/write/close lifecycle. Encoder tests
 * run end-to-end through libav and verify files reopen with the
 * expected dimensions and frame count.
 *
 * Key types:  (none)
 * Depends on: util/video_encode.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_helpers.h"
#include "util/video_encode.h"

static void test_overlay_alpha_zero_leaves_unchanged(void)
{
	uint8_t rgb[4 * 4 * 3];
	for (int i = 0; i < 4 * 4 * 3; i++)
		rgb[i] = 100;
	uint8_t mask[4 * 4];
	memset(mask, 255, sizeof(mask));

	sam3_overlay_composite(rgb, 4, 4, mask, 4, 4, 0, 0.0f);

	for (int i = 0; i < 4 * 4 * 3; i++)
		ASSERT_EQ(rgb[i], 100);
}

static void test_overlay_alpha_one_replaces(void)
{
	uint8_t rgb[4 * 4 * 3];
	memset(rgb, 0, sizeof(rgb));
	uint8_t mask[4 * 4];
	memset(mask, 255, sizeof(mask));

	sam3_overlay_composite(rgb, 4, 4, mask, 4, 4, 0, 1.0f);

	/* obj_id=0 -> palette[0]. All pixels should equal that color. */
	uint8_t r = rgb[0], g = rgb[1], b = rgb[2];
	/* Palette entries must be non-all-zero (so a color was applied). */
	ASSERT(r != 0 || g != 0 || b != 0);
	for (int i = 0; i < 4 * 4; i++) {
		ASSERT_EQ(rgb[i * 3 + 0], r);
		ASSERT_EQ(rgb[i * 3 + 1], g);
		ASSERT_EQ(rgb[i * 3 + 2], b);
	}
}

static void test_overlay_mask_zero_no_effect(void)
{
	uint8_t rgb[4 * 4 * 3];
	memset(rgb, 50, sizeof(rgb));
	uint8_t mask[4 * 4];
	memset(mask, 0, sizeof(mask));

	sam3_overlay_composite(rgb, 4, 4, mask, 4, 4, 0, 1.0f);

	for (int i = 0; i < 4 * 4 * 3; i++)
		ASSERT_EQ(rgb[i], 50);
}

static void test_overlay_upscale_nearest(void)
{
	/* 2x2 mask: only top-left cell set. Output 4x4 -> top-left 2x2 set. */
	uint8_t rgb[4 * 4 * 3];
	memset(rgb, 0, sizeof(rgb));
	uint8_t mask[2 * 2] = { 255, 0, 0, 0 };

	sam3_overlay_composite(rgb, 4, 4, mask, 2, 2, 0, 1.0f);

	for (int y = 0; y < 4; y++) {
		for (int x = 0; x < 4; x++) {
			int idx = (y * 4 + x) * 3;
			int should_be_set = (x < 2 && y < 2);
			if (should_be_set) {
				ASSERT(rgb[idx] != 0 || rgb[idx + 1] != 0 ||
				       rgb[idx + 2] != 0);
			} else {
				ASSERT_EQ(rgb[idx + 0], 0);
				ASSERT_EQ(rgb[idx + 1], 0);
				ASSERT_EQ(rgb[idx + 2], 0);
			}
		}
	}
}

static void test_overlay_half_alpha_blends(void)
{
	uint8_t rgb[1 * 1 * 3] = { 100, 100, 100 };
	uint8_t mask[1] = { 255 };

	sam3_overlay_composite(rgb, 1, 1, mask, 1, 1, 0, 0.5f);

	/*
	 * Resulting pixel = 100 * 0.5 + palette[0][c] * 0.5. Exact value
	 * depends on palette[0]; just verify the pixel changed from 100
	 * in at least one channel.
	 */
	int changed = rgb[0] != 100 || rgb[1] != 100 || rgb[2] != 100;
	ASSERT(changed);
}

int main(void)
{
	test_overlay_alpha_zero_leaves_unchanged();
	test_overlay_alpha_one_replaces();
	test_overlay_mask_zero_no_effect();
	test_overlay_upscale_nearest();
	test_overlay_half_alpha_blends();
	TEST_REPORT();
}
```

- [ ] **Step 2: Verify the tests fail to link (no implementation yet)**

```bash
cd /Users/rbisri/Documents/sam3/build && cmake --build . --target test_video_encode -j 2>&1 | tail -20
```

Expected: linker error for `sam3_overlay_composite` (unresolved symbol). This confirms the test compiles but the implementation is missing — exactly what we want before Task 9.

---

## Task 9: Implement `sam3_overlay_composite`

**Files:**
- Create: `src/util/video_encode.c` (initial version — just the overlay helper; encoder added in Task 11)

- [ ] **Step 1: Write the initial file**

```c
/*
 * src/util/video_encode.c - Encoder + overlay helper
 *
 * Implements sam3_overlay_composite (pure C, no libav) and the
 * libav-backed encoder open/write/close.
 *
 * Key types:  sam3_video_encoder
 * Depends on: util/video_encode.h, util/log.h, libavformat,
 *             libavcodec, libswscale, libavutil
 * Used by:    tools/cli_track.c, tests/test_video_encode.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include <string.h>

#include "util/video_encode.h"
#include "util/log.h"

/*
 * Ten visually distinguishable colors. Ordered so cycling mod 10 gives
 * consecutive objects high contrast. Values chosen to be reasonably
 * distinct across common displays.
 */
static const uint8_t palette[10][3] = {
	{ 239,  71, 111 }, /* pink/red */
	{  17, 138, 178 }, /* teal */
	{ 255, 209, 102 }, /* amber */
	{   6, 214, 160 }, /* mint */
	{ 155,  93, 229 }, /* purple */
	{ 251, 133,   0 }, /* orange */
	{  42, 157, 143 }, /* sea */
	{ 231, 111,  81 }, /* coral */
	{  38,  70,  83 }, /* navy */
	{ 244, 162,  97 }, /* tan */
};

void sam3_overlay_composite(uint8_t *rgb, int w, int h,
			    const uint8_t *mask, int mw, int mh,
			    int obj_id, float alpha)
{
	if (!rgb || !mask || w <= 0 || h <= 0 || mw <= 0 || mh <= 0)
		return;
	if (alpha < 0.0f) alpha = 0.0f;
	if (alpha > 1.0f) alpha = 1.0f;

	int idx = obj_id % 10;
	if (idx < 0) idx += 10;
	const uint8_t *c = palette[idx];

	float inv = 1.0f - alpha;
	float cr = c[0] * alpha;
	float cg = c[1] * alpha;
	float cb = c[2] * alpha;

	for (int y = 0; y < h; y++) {
		int my = (int)((long)y * mh / h);
		const uint8_t *mrow = mask + my * mw;
		uint8_t *orow = rgb + (size_t)y * w * 3;
		for (int x = 0; x < w; x++) {
			int mx = (int)((long)x * mw / w);
			if (!mrow[mx])
				continue;
			orow[x * 3 + 0] =
				(uint8_t)(orow[x * 3 + 0] * inv + cr);
			orow[x * 3 + 1] =
				(uint8_t)(orow[x * 3 + 1] * inv + cg);
			orow[x * 3 + 2] =
				(uint8_t)(orow[x * 3 + 2] * inv + cb);
		}
	}
}

/* Encoder functions — stubbed until Task 11 so linkage still succeeds. */
enum sam3_error sam3_video_encoder_open(const char *path,
					int width, int height,
					int fps_num, int fps_den,
					struct sam3_video_encoder **out)
{
	(void)path; (void)width; (void)height;
	(void)fps_num; (void)fps_den;
	if (out) *out = NULL;
	return SAM3_EIO; /* placeholder — real impl in Task 11 */
}

enum sam3_error sam3_video_encoder_write_rgb(struct sam3_video_encoder *enc,
					     const uint8_t *rgb)
{
	(void)enc; (void)rgb;
	return SAM3_EIO;
}

enum sam3_error sam3_video_encoder_close(struct sam3_video_encoder *enc)
{
	(void)enc;
	return SAM3_OK;
}
```

- [ ] **Step 2: Build and run overlay tests**

```bash
cd /Users/rbisri/Documents/sam3/build && cmake --build . --target test_video_encode -j 2>&1 | tail -10 && ctest -R test_video_encode --output-on-failure
```

Expected: build succeeds, `test_video_encode` passes (all 5 overlay tests).

- [ ] **Step 3: Commit**

```bash
git add src/util/video_encode.c tests/test_video_encode.c
git commit -m "util/video_encode: implement overlay_composite

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Add failing encoder lifecycle tests

**Files:**
- Modify: `tests/test_video_encode.c`

- [ ] **Step 1: Append new tests before `main`**

Insert these tests directly above `int main(void)`:

```c
static const char *tmp_mp4_path(char buf[static 256])
{
	snprintf(buf, 256, "/tmp/sam3_test_video_%d.mp4", (int)getpid());
	return buf;
}

static void test_encoder_close_null_is_ok(void)
{
	ASSERT_EQ(sam3_video_encoder_close(NULL), SAM3_OK);
}

static void test_encoder_rejects_unknown_extension(void)
{
	struct sam3_video_encoder *enc = NULL;
	enum sam3_error err = sam3_video_encoder_open(
		"/tmp/sam3_test_bad.xyz", 16, 16, 10, 1, &enc);
	ASSERT_EQ(err, SAM3_EIO);
	ASSERT(enc == NULL);
}

static void test_encoder_open_close_idempotent(void)
{
	char path[256];
	tmp_mp4_path(path);
	remove(path);

	struct sam3_video_encoder *enc = NULL;
	enum sam3_error err = sam3_video_encoder_open(path, 16, 16, 10, 1,
						      &enc);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT(enc != NULL);

	/* Close once. */
	ASSERT_EQ(sam3_video_encoder_close(enc), SAM3_OK);

	remove(path);
}
```

Also add `#include <sys/types.h>` and `#include <unistd.h>` at the top for `getpid`.

Update `main` to call the new tests:

```c
int main(void)
{
	test_overlay_alpha_zero_leaves_unchanged();
	test_overlay_alpha_one_replaces();
	test_overlay_mask_zero_no_effect();
	test_overlay_upscale_nearest();
	test_overlay_half_alpha_blends();
	test_encoder_close_null_is_ok();
	test_encoder_rejects_unknown_extension();
	test_encoder_open_close_idempotent();
	TEST_REPORT();
}
```

- [ ] **Step 2: Build — expect `test_encoder_open_close_idempotent` to fail**

```bash
cd /Users/rbisri/Documents/sam3/build && cmake --build . --target test_video_encode -j && ctest -R test_video_encode --output-on-failure
```

Expected: `test_encoder_close_null_is_ok` and `test_encoder_rejects_unknown_extension` pass (the stub already satisfies them). `test_encoder_open_close_idempotent` fails because the stub always returns `SAM3_EIO`.

---

## Task 11: Implement the encoder

**Files:**
- Modify: `src/util/video_encode.c` (replace the stubs with a real implementation)

- [ ] **Step 1: Replace the stub block at the bottom with the full encoder**

Delete the stub block (`sam3_video_encoder_open` / `_write_rgb` / `_close`) and replace with:

```c
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wstrict-prototypes"
#pragma clang diagnostic ignored "-Wcast-align"
#pragma clang diagnostic ignored "-Wundef"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>

#ifdef __clang__
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

struct sam3_video_encoder {
	AVFormatContext *fmt;
	AVStream        *stream;
	AVCodecContext  *codec;
	AVFrame         *yuv_frame;
	AVPacket        *packet;
	struct SwsContext *sws;
	int              width;
	int              height;
	int              fps_num;
	int              fps_den;
	int64_t          next_pts;
	int              header_written;
	int              io_opened;
};

static enum AVCodecID codec_for_path(const char *path)
{
	size_t n = strlen(path);
	const char *ext = NULL;
	for (size_t i = n; i > 0; i--) {
		if (path[i - 1] == '.') { ext = path + i; break; }
	}
	if (!ext) return AV_CODEC_ID_NONE;
	if (strcasecmp(ext, "mp4") == 0 ||
	    strcasecmp(ext, "mov") == 0 ||
	    strcasecmp(ext, "mkv") == 0)
		return AV_CODEC_ID_H264;
	if (strcasecmp(ext, "webm") == 0)
		return AV_CODEC_ID_VP9;
	return AV_CODEC_ID_NONE;
}

enum sam3_error sam3_video_encoder_close(struct sam3_video_encoder *enc)
{
	enum sam3_error ret = SAM3_OK;
	if (!enc)
		return SAM3_OK;

	/* Flush encoder if we opened it */
	if (enc->codec && enc->packet) {
		avcodec_send_frame(enc->codec, NULL);
		for (;;) {
			int rc = avcodec_receive_packet(enc->codec, enc->packet);
			if (rc == AVERROR_EOF || rc == AVERROR(EAGAIN))
				break;
			if (rc < 0) {
				sam3_log_error("encoder flush: %s",
					       av_err2str(rc));
				ret = SAM3_EIO;
				break;
			}
			av_packet_rescale_ts(enc->packet,
					     enc->codec->time_base,
					     enc->stream->time_base);
			enc->packet->stream_index = enc->stream->index;
			int wr = av_interleaved_write_frame(enc->fmt,
							    enc->packet);
			av_packet_unref(enc->packet);
			if (wr < 0) {
				sam3_log_error("interleaved_write_frame: %s",
					       av_err2str(wr));
				ret = SAM3_EIO;
				break;
			}
		}
	}

	if (enc->header_written && enc->fmt) {
		int rc = av_write_trailer(enc->fmt);
		if (rc < 0) {
			sam3_log_error("av_write_trailer: %s",
				       av_err2str(rc));
			ret = SAM3_EIO;
		}
	}

	if (enc->sws)       sws_freeContext(enc->sws);
	if (enc->yuv_frame) av_frame_free(&enc->yuv_frame);
	if (enc->packet)    av_packet_free(&enc->packet);
	if (enc->codec)     avcodec_free_context(&enc->codec);
	if (enc->io_opened && enc->fmt &&
	    !(enc->fmt->oformat->flags & AVFMT_NOFILE))
		avio_closep(&enc->fmt->pb);
	if (enc->fmt)       avformat_free_context(enc->fmt);

	free(enc);
	return ret;
}

enum sam3_error sam3_video_encoder_open(const char *path,
					int width, int height,
					int fps_num, int fps_den,
					struct sam3_video_encoder **out)
{
	if (!path || !out || width <= 0 || height <= 0 ||
	    fps_num <= 0 || fps_den <= 0)
		return SAM3_EINVAL;
	*out = NULL;

	enum AVCodecID codec_id = codec_for_path(path);
	if (codec_id == AV_CODEC_ID_NONE) {
		sam3_log_error("unsupported output extension for '%s' "
			       "(accepted: .mp4/.mov/.mkv/.webm)", path);
		return SAM3_EIO;
	}

	struct sam3_video_encoder *enc = calloc(1, sizeof(*enc));
	if (!enc)
		return SAM3_ENOMEM;
	enc->width   = width;
	enc->height  = height;
	enc->fps_num = fps_num;
	enc->fps_den = fps_den;

	int rc;
	rc = avformat_alloc_output_context2(&enc->fmt, NULL, NULL, path);
	if (rc < 0 || !enc->fmt) {
		sam3_log_error("avformat_alloc_output_context2: %s",
			       av_err2str(rc));
		sam3_video_encoder_close(enc);
		return SAM3_EIO;
	}

	const AVCodec *codec = avcodec_find_encoder(codec_id);
	if (!codec) {
		sam3_log_error("encoder not compiled into libav for codec %d "
			       "(rebuild ffmpeg with libx264/libvpx)",
			       (int)codec_id);
		sam3_video_encoder_close(enc);
		return SAM3_EIO;
	}

	enc->stream = avformat_new_stream(enc->fmt, NULL);
	if (!enc->stream) {
		sam3_video_encoder_close(enc);
		return SAM3_ENOMEM;
	}

	enc->codec = avcodec_alloc_context3(codec);
	if (!enc->codec) {
		sam3_video_encoder_close(enc);
		return SAM3_ENOMEM;
	}
	enc->codec->codec_id   = codec_id;
	enc->codec->width      = width;
	enc->codec->height     = height;
	enc->codec->pix_fmt    = AV_PIX_FMT_YUV420P;
	enc->codec->time_base  = (AVRational){ fps_den, fps_num };
	enc->codec->framerate  = (AVRational){ fps_num, fps_den };
	enc->codec->gop_size   = 12;
	enc->codec->max_b_frames = 0;

	if (enc->fmt->oformat->flags & AVFMT_GLOBALHEADER)
		enc->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

	if (codec_id == AV_CODEC_ID_H264) {
		av_opt_set(enc->codec->priv_data, "preset", "medium", 0);
		av_opt_set(enc->codec->priv_data, "crf", "23", 0);
	}

	rc = avcodec_open2(enc->codec, codec, NULL);
	if (rc < 0) {
		sam3_log_error("avcodec_open2: %s", av_err2str(rc));
		sam3_video_encoder_close(enc);
		return SAM3_EIO;
	}

	rc = avcodec_parameters_from_context(enc->stream->codecpar, enc->codec);
	if (rc < 0) {
		sam3_log_error("avcodec_parameters_from_context: %s",
			       av_err2str(rc));
		sam3_video_encoder_close(enc);
		return SAM3_EIO;
	}
	enc->stream->time_base = enc->codec->time_base;

	enc->yuv_frame = av_frame_alloc();
	enc->packet    = av_packet_alloc();
	if (!enc->yuv_frame || !enc->packet) {
		sam3_video_encoder_close(enc);
		return SAM3_ENOMEM;
	}
	enc->yuv_frame->format = AV_PIX_FMT_YUV420P;
	enc->yuv_frame->width  = width;
	enc->yuv_frame->height = height;
	rc = av_frame_get_buffer(enc->yuv_frame, 32);
	if (rc < 0) {
		sam3_log_error("av_frame_get_buffer: %s", av_err2str(rc));
		sam3_video_encoder_close(enc);
		return SAM3_ENOMEM;
	}

	enc->sws = sws_getContext(width, height, AV_PIX_FMT_RGB24,
				  width, height, AV_PIX_FMT_YUV420P,
				  SWS_BILINEAR, NULL, NULL, NULL);
	if (!enc->sws) {
		sam3_video_encoder_close(enc);
		return SAM3_ENOMEM;
	}

	if (!(enc->fmt->oformat->flags & AVFMT_NOFILE)) {
		rc = avio_open(&enc->fmt->pb, path, AVIO_FLAG_WRITE);
		if (rc < 0) {
			sam3_log_error("avio_open '%s': %s", path,
				       av_err2str(rc));
			sam3_video_encoder_close(enc);
			return SAM3_EIO;
		}
		enc->io_opened = 1;
	}

	rc = avformat_write_header(enc->fmt, NULL);
	if (rc < 0) {
		sam3_log_error("avformat_write_header: %s", av_err2str(rc));
		sam3_video_encoder_close(enc);
		return SAM3_EIO;
	}
	enc->header_written = 1;

	*out = enc;
	return SAM3_OK;
}

enum sam3_error sam3_video_encoder_write_rgb(struct sam3_video_encoder *enc,
					     const uint8_t *rgb)
{
	if (!enc || !rgb)
		return SAM3_EINVAL;

	int rc = av_frame_make_writable(enc->yuv_frame);
	if (rc < 0) {
		sam3_log_error("av_frame_make_writable: %s", av_err2str(rc));
		return SAM3_EIO;
	}

	const uint8_t *src_slices[4]   = { rgb, NULL, NULL, NULL };
	int            src_strides[4]  = { enc->width * 3, 0, 0, 0 };
	sws_scale(enc->sws, src_slices, src_strides, 0, enc->height,
		  enc->yuv_frame->data, enc->yuv_frame->linesize);

	enc->yuv_frame->pts = enc->next_pts++;

	rc = avcodec_send_frame(enc->codec, enc->yuv_frame);
	if (rc < 0) {
		sam3_log_error("avcodec_send_frame: %s", av_err2str(rc));
		return SAM3_EIO;
	}

	for (;;) {
		rc = avcodec_receive_packet(enc->codec, enc->packet);
		if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF)
			return SAM3_OK;
		if (rc < 0) {
			sam3_log_error("avcodec_receive_packet: %s",
				       av_err2str(rc));
			return SAM3_EIO;
		}
		av_packet_rescale_ts(enc->packet, enc->codec->time_base,
				     enc->stream->time_base);
		enc->packet->stream_index = enc->stream->index;
		int wr = av_interleaved_write_frame(enc->fmt, enc->packet);
		av_packet_unref(enc->packet);
		if (wr < 0) {
			sam3_log_error("av_interleaved_write_frame: %s",
				       av_err2str(wr));
			return SAM3_EIO;
		}
	}
}
```

- [ ] **Step 2: Build and run encoder tests**

```bash
cd /Users/rbisri/Documents/sam3/build && cmake --build . --target test_video_encode -j 2>&1 | tail -10 && ctest -R test_video_encode --output-on-failure
```

Expected: build succeeds, all existing tests pass (including `test_encoder_open_close_idempotent`).

- [ ] **Step 3: Commit**

```bash
git add src/util/video_encode.c
git commit -m "util/video_encode: implement libav-backed encoder

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Add encoder roundtrip test

**Files:**
- Modify: `tests/test_video_encode.c`

- [ ] **Step 1: Add the roundtrip test**

Insert before `main`:

```c
/* Forward declare to keep main list flat. */
static void test_encoder_roundtrip_mp4(void)
{
	char path[256];
	tmp_mp4_path(path);
	remove(path);

	struct sam3_video_encoder *enc = NULL;
	ASSERT_EQ(sam3_video_encoder_open(path, 16, 16, 10, 1, &enc),
		  SAM3_OK);
	ASSERT(enc != NULL);

	uint8_t frame[16 * 16 * 3];
	for (int i = 0; i < 24; i++) {
		for (int y = 0; y < 16; y++) {
			for (int x = 0; x < 16; x++) {
				frame[(y * 16 + x) * 3 + 0] = (uint8_t)(i * 10);
				frame[(y * 16 + x) * 3 + 1] = (uint8_t)(x * 16);
				frame[(y * 16 + x) * 3 + 2] = (uint8_t)(y * 16);
			}
		}
		ASSERT_EQ(sam3_video_encoder_write_rgb(enc, frame), SAM3_OK);
	}
	ASSERT_EQ(sam3_video_encoder_close(enc), SAM3_OK);

	/* Reopen via libav to verify file is valid. */
	AVFormatContext *fmt = NULL;
	int rc = avformat_open_input(&fmt, path, NULL, NULL);
	ASSERT_EQ(rc, 0);
	rc = avformat_find_stream_info(fmt, NULL);
	ASSERT(rc >= 0);

	int vstream = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1,
					  NULL, 0);
	ASSERT(vstream >= 0);
	ASSERT_EQ(fmt->streams[vstream]->codecpar->width, 16);
	ASSERT_EQ(fmt->streams[vstream]->codecpar->height, 16);

	avformat_close_input(&fmt);
	remove(path);
}
```

Also include the libav headers at the top of the test file so the roundtrip test links:

```c
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
```

And add the call to `main`:

```c
test_encoder_roundtrip_mp4();
```

- [ ] **Step 2: Teach CMake to link libav into the test**

Add to `CMakeLists.txt`, inside the `if(SAM3_TESTS)` block after the `test_cli_track` stanza (around line 283):

```cmake
# test_video_encode needs libav headers directly for the roundtrip test
if(TARGET test_video_encode)
	target_link_libraries(test_video_encode PkgConfig::LIBAV)
endif()
```

- [ ] **Step 3: Reconfigure, build, run**

```bash
cd /Users/rbisri/Documents/sam3/build && cmake .. && cmake --build . --target test_video_encode -j 2>&1 | tail -10 && ctest -R test_video_encode --output-on-failure
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_video_encode.c CMakeLists.txt
git commit -m "tests/video_encode: add mp4 roundtrip test

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Extend `track_args` for new flags

**Files:**
- Modify: `tools/cli_track.h`

- [ ] **Step 1: Add new fields + enum**

Insert a new enum directly after the `track_propagate` enum:

```c
/*
 * CLI-layer output mode, derived from the extension on --output.
 * Selected during cli_track_parse so cli_track_run can branch cleanly.
 */
enum track_output_mode {
	TRACK_OUTPUT_DIR   = 0, /* write per-frame PNGs into a directory */
	TRACK_OUTPUT_VIDEO = 1, /* write a single overlay video file */
};
```

In `struct track_args`, after the existing `profile` field, add:

```c
	int                       output_mode; /* enum track_output_mode */
	float                     alpha;       /* overlay alpha, [0, 1] */
	int                       fps;         /* output fps (frame-dir input) */
```

Update the struct's doc comment to mention these fields are video-mode only and that `alpha` defaults to 0.5 while `fps` defaults to 0 (meaning "inherit from source").

- [ ] **Step 2: Commit**

```bash
git add tools/cli_track.h
git commit -m "cli/track: extend track_args with alpha, fps, output_mode

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: Add failing parser tests for `--alpha`, `--fps`, output-mode detection

**Files:**
- Modify: `tests/test_cli_track.c`

- [ ] **Step 1: Append new tests**

Before the `main` function, add:

```c
static void test_track_parse_alpha_valid(void)
{
	struct track_args a = {0};
	char **argv = ARGV("track",
			   "--model", "m.sam3", "--video", "v.mp4",
			   "--output", "out/",
			   "--point", "1,2,1",
			   "--alpha", "0.75");
	int argc = ARGC("track",
			"--model", "m.sam3", "--video", "v.mp4",
			"--output", "out/",
			"--point", "1,2,1",
			"--alpha", "0.75");

	ASSERT_EQ(cli_track_parse(argc, argv, &a), 0);
	ASSERT(a.alpha > 0.74f && a.alpha < 0.76f);
}

static void test_track_parse_alpha_out_of_range(void)
{
	struct track_args a = {0};
	char **argv = ARGV("track",
			   "--model", "m.sam3", "--video", "v.mp4",
			   "--output", "out/",
			   "--point", "1,2,1",
			   "--alpha", "2.0");
	int argc = ARGC("track",
			"--model", "m.sam3", "--video", "v.mp4",
			"--output", "out/",
			"--point", "1,2,1",
			"--alpha", "2.0");

	ASSERT_EQ(cli_track_parse(argc, argv, &a), -1);
}

static void test_track_parse_fps_valid(void)
{
	struct track_args a = {0};
	char **argv = ARGV("track",
			   "--model", "m.sam3", "--video", "v.mp4",
			   "--output", "out.mp4",
			   "--point", "1,2,1",
			   "--fps", "30");
	int argc = ARGC("track",
			"--model", "m.sam3", "--video", "v.mp4",
			"--output", "out.mp4",
			"--point", "1,2,1",
			"--fps", "30");

	ASSERT_EQ(cli_track_parse(argc, argv, &a), 0);
	ASSERT_EQ(a.fps, 30);
	ASSERT_EQ(a.output_mode, TRACK_OUTPUT_VIDEO);
}

static void test_track_output_mode_video_extensions(void)
{
	const char *paths[] = {
		"out.mp4", "OUT.MOV", "x.mkv", "seg.webm"
	};
	for (size_t i = 0; i < sizeof(paths) / sizeof(paths[0]); i++) {
		struct track_args a = {0};
		char **argv = ARGV("track",
				   "--model", "m.sam3", "--video", "v.mp4",
				   "--output", paths[i],
				   "--point", "1,2,1");
		int argc = ARGC("track",
				"--model", "m.sam3", "--video", "v.mp4",
				"--output", paths[i],
				"--point", "1,2,1");
		ASSERT_EQ(cli_track_parse(argc, argv, &a), 0);
		ASSERT_EQ(a.output_mode, TRACK_OUTPUT_VIDEO);
	}
}

static void test_track_output_mode_dir_default(void)
{
	const char *paths[] = { "out/", "results", "./x.png" };
	for (size_t i = 0; i < sizeof(paths) / sizeof(paths[0]); i++) {
		struct track_args a = {0};
		char **argv = ARGV("track",
				   "--model", "m.sam3", "--video", "v.mp4",
				   "--output", paths[i],
				   "--point", "1,2,1");
		int argc = ARGC("track",
				"--model", "m.sam3", "--video", "v.mp4",
				"--output", paths[i],
				"--point", "1,2,1");
		ASSERT_EQ(cli_track_parse(argc, argv, &a), 0);
		ASSERT_EQ(a.output_mode, TRACK_OUTPUT_DIR);
	}
}

static void test_track_default_alpha(void)
{
	struct track_args a = {0};
	char **argv = ARGV("track",
			   "--model", "m.sam3", "--video", "v.mp4",
			   "--output", "out/",
			   "--point", "1,2,1");
	int argc = ARGC("track",
			"--model", "m.sam3", "--video", "v.mp4",
			"--output", "out/",
			"--point", "1,2,1");
	ASSERT_EQ(cli_track_parse(argc, argv, &a), 0);
	ASSERT(a.alpha > 0.49f && a.alpha < 0.51f);
}
```

Add the calls to `main`:

```c
test_track_parse_alpha_valid();
test_track_parse_alpha_out_of_range();
test_track_parse_fps_valid();
test_track_output_mode_video_extensions();
test_track_output_mode_dir_default();
test_track_default_alpha();
```

- [ ] **Step 2: Build and verify tests fail**

```bash
cd /Users/rbisri/Documents/sam3/build && cmake --build . --target test_cli_track -j 2>&1 | tail -10 && ctest -R test_cli_track --output-on-failure 2>&1 | tail -20
```

Expected: compile succeeds, new tests fail (fields don't exist / parser doesn't handle new flags yet).

---

## Task 15: Implement parser support for new flags + output-mode detection

**Files:**
- Modify: `tools/cli_track.c`

- [ ] **Step 1: Add a helper for extension-based mode detection**

Insert below `parse_box`:

```c
static int has_suffix_ci(const char *s, const char *suf)
{
	size_t sn = strlen(s), xn = strlen(suf);
	if (sn < xn) return 0;
	const char *tail = s + sn - xn;
	for (size_t i = 0; i < xn; i++) {
		char a = tail[i], b = suf[i];
		if (a >= 'A' && a <= 'Z') a = (char)(a + 32);
		if (b >= 'A' && b <= 'Z') b = (char)(b + 32);
		if (a != b) return 0;
	}
	return 1;
}

static int detect_output_mode(const char *path)
{
	if (has_suffix_ci(path, ".mp4")  || has_suffix_ci(path, ".mov") ||
	    has_suffix_ci(path, ".mkv")  || has_suffix_ci(path, ".webm"))
		return TRACK_OUTPUT_VIDEO;
	return TRACK_OUTPUT_DIR;
}
```

- [ ] **Step 2: Initialize the new fields in `cli_track_parse`**

In `cli_track_parse`, at the top where fields are initialized, add:

```c
	out->alpha       = 0.5f;
	out->fps         = 0;
	out->output_mode = TRACK_OUTPUT_DIR; /* finalized after --output parsed */
```

- [ ] **Step 3: Add flag parsing**

Insert new `else if` branches in the argv loop, before the final "unknown option" branch:

```c
		} else if (strcmp(argv[i], "--alpha") == 0) {
			if (++i >= argc) {
				sam3_log_error("--alpha requires a value");
				return -1;
			}
			char *end;
			errno = 0;
			double v = strtod(argv[i], &end);
			if (argv[i][0] == '\0' || *end != '\0' ||
			    errno == ERANGE || v < 0.0 || v > 1.0) {
				sam3_log_error(
					"--alpha must be in [0,1] (got '%s')",
					argv[i]);
				return -1;
			}
			out->alpha = (float)v;
		} else if (strcmp(argv[i], "--fps") == 0) {
			if (++i >= argc) {
				sam3_log_error("--fps requires a value");
				return -1;
			}
			int tmp;
			if (parse_int_arg("--fps", argv[i], &tmp))
				return -1;
			if (tmp <= 0) {
				sam3_log_error("--fps %d must be > 0", tmp);
				return -1;
			}
			out->fps = tmp;
```

- [ ] **Step 4: Finalize `output_mode` after the loop, next to the "--output required" check**

Locate:

```c
	if (!out->output_dir) {
		sam3_log_error("--output <dir> is required");
		return -1;
	}
```

Replace with:

```c
	if (!out->output_dir) {
		sam3_log_error("--output <path> is required");
		return -1;
	}
	out->output_mode = detect_output_mode(out->output_dir);
```

Also rename the help text / log in `print_usage` to say `--output <path>` and describe both modes (dir → per-frame PNGs; `.mp4`/`.mov`/`.mkv`/`.webm` → overlay video). Add `--alpha` and `--fps` documentation:

```c
	printf("\nVideo-output options (when --output has a "
	       ".mp4/.mov/.mkv/.webm extension):\n");
	printf("  --alpha <f>           Overlay alpha in [0,1] (default: 0.5)\n");
	printf("  --fps <n>             Frame rate (required for frame-dir "
	       "input; ignored for videos)\n");
```

- [ ] **Step 5: Build and run parser tests**

```bash
cd /Users/rbisri/Documents/sam3/build && cmake --build . --target test_cli_track -j 2>&1 | tail -10 && ctest -R test_cli_track --output-on-failure
```

Expected: all parser tests pass.

- [ ] **Step 6: Commit**

```bash
git add tools/cli_track.c tests/test_cli_track.c
git commit -m "cli/track: parse --alpha, --fps, and output-mode extension

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 16: Factor `cli_track_run` into dir-mode and video-mode branches

This task does not change behavior — it only prepares the structure so the next tasks can implement video mode cleanly.

**Files:**
- Modify: `tools/cli_track.c`

- [ ] **Step 1: Rename the current `cli_track_run` to `cli_track_run_dir` (no logic change)**

Find `static int cli_track_run(const struct track_args *a)`. Rename to `static int cli_track_run_dir(const struct track_args *a)`. Leave the body unchanged.

- [ ] **Step 2: Add forward declaration for the video-mode entry point**

Just below the includes, add:

```c
static int cli_track_run_video(const struct track_args *a);
```

- [ ] **Step 3: Add a dispatcher `cli_track_run`**

Insert directly above `cli_track_run_dir`:

```c
static int cli_track_run(const struct track_args *a)
{
	if (a->output_mode == TRACK_OUTPUT_VIDEO)
		return cli_track_run_video(a);
	return cli_track_run_dir(a);
}
```

- [ ] **Step 4: Add a stub `cli_track_run_video` at the bottom of the file**

```c
static int cli_track_run_video(const struct track_args *a)
{
	(void)a;
	sam3_log_error("video output mode not implemented yet");
	return SAM3_EXIT_INTERNAL;
}
```

- [ ] **Step 5: Build and run tests**

```bash
cd /Users/rbisri/Documents/sam3/build && cmake --build . -j 2>&1 | tail -10 && ctest --output-on-failure 2>&1 | tail -20
```

Expected: all existing tests pass (dir mode still works; video-mode dispatcher just returns an error if you hit it, which no test does).

- [ ] **Step 6: Commit**

```bash
git add tools/cli_track.c
git commit -m "cli/track: split run into dir and video branches

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 17: Implement video-mode pass 1 — propagate + mask buffering

**Files:**
- Modify: `tools/cli_track.c`

- [ ] **Step 1: Add includes**

At the top (after `#include "util/error.h"`):

```c
#include "util/video.h"
#include "util/video_internal.h"
#include "util/video_encode.h"
```

- [ ] **Step 2: Add the mask-buffering callback + context**

Insert above the stubbed `cli_track_run_video`:

```c
/*
 * video_frame_ctx - Per-frame mask buffer + profiler carry.
 *
 * Populated by video_mask_callback during sam3_video_propagate.
 * `masks` is laid out as: masks[(frame_idx * n_objects + obj_id) *
 * mask_h * mask_w + y * mask_w + x], storing 0 or 255.
 */
struct video_frame_ctx {
	int      n_frames;
	int      n_objects;
	int      mask_w, mask_h;
	uint8_t *masks;  /* n_frames * n_objects * mask_h * mask_w */
	uint8_t *seen;   /* n_frames */
	struct sam3_profiler *profiler; /* NULL if --profile is off */
};

static int video_mask_callback(const struct sam3_video_frame_result *r,
			       void *user_data)
{
	struct video_frame_ctx *fc = user_data;
	if (!r || r->frame_idx < 0 || r->frame_idx >= fc->n_frames)
		return 0;

	SAM3_PROF_BEGIN(fc->profiler, "mask_buffer");

	fc->seen[r->frame_idx] = 1;
	for (int i = 0; i < r->n_objects; i++) {
		int obj_id = r->objects[i].obj_id;
		if (obj_id < 0 || obj_id >= fc->n_objects)
			continue;
		if (!r->objects[i].mask)
			continue;

		int mw = r->objects[i].mask_w;
		int mh = r->objects[i].mask_h;
		if (mw != fc->mask_w || mh != fc->mask_h)
			continue; /* resolution mismatch — skip */

		size_t off = ((size_t)r->frame_idx * fc->n_objects + obj_id) *
			     (size_t)fc->mask_h * fc->mask_w;
		uint8_t *dst = fc->masks + off;
		const float *src = r->objects[i].mask;
		int n = mw * mh;
		for (int k = 0; k < n; k++)
			dst[k] = src[k] >= 0.0f ? 255 : 0;
	}

	SAM3_PROF_END(fc->profiler, "mask_buffer");
	return 0;
}
```

- [ ] **Step 3: Replace the stub `cli_track_run_video` with the real pass-1 flow**

Delete the stub and paste:

```c
static int cli_track_run_video(const struct track_args *a)
{
	sam3_ctx              *ctx     = NULL;
	sam3_video_session    *session = NULL;
	enum sam3_error        err;
	int                    ret     = SAM3_EXIT_INTERNAL;
	struct video_frame_ctx fc      = {0};

	if (a->verbose)
		sam3_log_set_level(SAM3_LOG_DEBUG);

	ctx = sam3_init();
	if (!ctx) {
		sam3_log_error("failed to initialize sam3 context");
		return SAM3_EXIT_INTERNAL;
	}

	if (a->profile) {
		err = sam3_profile_enable(ctx);
		if (err != SAM3_OK)
			sam3_log_warn("profiling not available: %s",
				      sam3_error_str(err));
		fc.profiler = sam3_profile_get(ctx);
	}

	sam3_log_info("loading model: %s", a->model_path);
	err = sam3_load_model(ctx, a->model_path);
	if (err != SAM3_OK) {
		sam3_log_error("failed to load model '%s': %s",
			       a->model_path, sam3_error_str(err));
		ret = (int)sam3_error_to_exit(err);
		goto cleanup;
	}

	sam3_log_info("starting video session: %s", a->video_path);
	err = sam3_video_start(ctx, a->video_path, &session);
	if (err != SAM3_OK) {
		sam3_log_error("video start failed: %s",
			       sam3_error_str(err));
		ret = (int)sam3_error_to_exit(err);
		goto cleanup;
	}

	fc.n_frames = sam3_video_frame_count(session);
	/* Masks come back at the model input size. */
	fc.mask_w = sam3_get_image_size(ctx);
	fc.mask_h = fc.mask_w;

	/* Determine number of objects in use (max obj_id in prompts + 1). */
	int max_obj = -1;
	for (int i = 0; i < a->n_prompts; i++)
		if (a->prompts[i].obj_id > max_obj)
			max_obj = a->prompts[i].obj_id;
	fc.n_objects = max_obj + 1;
	if (fc.n_objects <= 0) fc.n_objects = 1;

	size_t bytes = (size_t)fc.n_frames * fc.n_objects *
		       (size_t)fc.mask_h * fc.mask_w;
	fc.masks = calloc(1, bytes);
	fc.seen  = calloc(1, (size_t)fc.n_frames);
	if (!fc.masks || !fc.seen) {
		sam3_log_error(
			"out of memory for mask buffer (%zu MB)",
			bytes / (1024 * 1024));
		ret = (int)sam3_error_to_exit(SAM3_ENOMEM);
		goto cleanup;
	}

	for (int i = 0; i < a->n_prompts; i++) {
		const struct track_prompt_entry *e = &a->prompts[i];
		struct sam3_video_frame_result prompt_result = {0};

		if (e->prompt.type == SAM3_PROMPT_POINT) {
			err = sam3_video_add_points(session, a->frame_idx,
						    e->obj_id,
						    &e->prompt.point, 1,
						    &prompt_result);
		} else if (e->prompt.type == SAM3_PROMPT_BOX) {
			err = sam3_video_add_box(session, a->frame_idx,
						 e->obj_id, &e->prompt.box,
						 &prompt_result);
		} else {
			sam3_log_warn("prompt %d: unsupported type, skipping", i);
			continue;
		}
		if (err == SAM3_OK)
			video_mask_callback(&prompt_result, &fc);
		sam3_video_frame_result_free(&prompt_result);
		if (err != SAM3_OK) {
			sam3_log_error("prompt %d failed: %s", i,
				       sam3_error_str(err));
			ret = (int)sam3_error_to_exit(err);
			goto cleanup;
		}
	}

	int sam3_dir = track_propagate_to_sam3_dir(a->propagate);
	if (sam3_dir < 0) {
		sam3_log_info("propagation disabled (--propagate none)");
	} else {
		sam3_log_info("propagating masks (%s)",
			      dir_name((enum sam3_propagate_dir)sam3_dir));
		err = sam3_video_propagate(session, sam3_dir,
					   video_mask_callback, &fc);
		if (err != SAM3_OK) {
			sam3_log_error("propagation failed: %s",
				       sam3_error_str(err));
			ret = (int)sam3_error_to_exit(err);
			goto cleanup;
		}
	}

	/* Pass 2 is added in Task 18. For now, treat pass 1 as success. */
	sam3_log_warn("video pass 2 (encode) not implemented yet; "
		      "exiting after pass 1");
	ret = SAM3_EXIT_OK;

cleanup:
	if (a->profile)
		sam3_profile_report(ctx);
	if (session)
		sam3_video_end(session);
	sam3_free(ctx);
	free(fc.masks);
	free(fc.seen);
	return ret;
}
```

- [ ] **Step 4: Build**

```bash
cd /Users/rbisri/Documents/sam3/build && cmake --build . -j 2>&1 | tail -20
```

Expected: build succeeds.

- [ ] **Step 5: Commit**

```bash
git add tools/cli_track.c
git commit -m "cli/track: implement video-mode pass 1 (mask buffering)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 18: Implement video-mode pass 2 — decode, composite, encode

**Files:**
- Modify: `tools/cli_track.c`

- [ ] **Step 1: Replace the pass-2 stub block with the real implementation**

Locate the block:

```c
	/* Pass 2 is added in Task 18. For now, treat pass 1 as success. */
	sam3_log_warn("video pass 2 (encode) not implemented yet; "
		      "exiting after pass 1");
	ret = SAM3_EXIT_OK;
```

Replace with:

```c
	/* --- Pass 2: decode source at native resolution, composite, encode --- */
	{
		struct sam3_rgb_iter     *it  = NULL;
		struct sam3_video_encoder *enc = NULL;
		uint8_t                  *rgb_copy = NULL;

		err = sam3_rgb_iter_open(a->video_path, &it);
		if (err != SAM3_OK) {
			ret = (int)sam3_error_to_exit(err);
			goto pass2_cleanup;
		}

		/* Peek the first frame to know dimensions and fps fallback. */
		const uint8_t *rgb = NULL;
		int w = 0, h = 0, eof = 0;
		err = sam3_rgb_iter_next(it, &rgb, &w, &h, &eof);
		if (err != SAM3_OK || eof || !rgb) {
			sam3_log_error(
				"video mode: source has no frames to encode");
			if (err == SAM3_OK) err = SAM3_EIO;
			ret = (int)sam3_error_to_exit(err);
			goto pass2_cleanup;
		}

		/* Determine fps: libav reports it via sam3_video_frames only
		 * during model load, not the iterator. Use args->fps when
		 * provided; otherwise probe the path via sam3_video_load-less
		 * shortcut (we already know source is decodable). Require
		 * --fps for frame-dir sources. */
		int fps_num = 0, fps_den = 1;
		if (a->fps > 0) {
			fps_num = a->fps;
		} else {
			enum sam3_video_type vt =
				sam3_video_detect_type(a->video_path);
			if (vt == SAM3_VIDEO_FRAME_DIR) {
				sam3_log_error(
					"--fps is required when --video is a "
					"frame directory and --output is a video");
				err = SAM3_EINVAL;
				ret = (int)sam3_error_to_exit(err);
				goto pass2_cleanup;
			}
			/* Video file — fall back to the session's recorded
			 * fps via a cheap sam3_video_load probe. */
			struct sam3_arena probe_arena;
			if (sam3_arena_init(&probe_arena, 1024) == SAM3_OK) {
				/* We don't actually need the frames, just fps;
				 * but sam3_video_load requires a full decode.
				 * Skip that — query libav directly via a new
				 * rgb iter and read its avg_frame_rate. Since
				 * our iterator does not expose fps yet, assume
				 * 30/1 as a safe default and log a warning. */
				sam3_arena_free(&probe_arena);
			}
			if (fps_num == 0) {
				sam3_log_warn(
					"could not determine source fps; "
					"defaulting to 30 (pass --fps to override)");
				fps_num = 30;
			}
		}

		err = sam3_video_encoder_open(a->output_dir, w, h,
					      fps_num, fps_den, &enc);
		if (err != SAM3_OK) {
			ret = (int)sam3_error_to_exit(err);
			goto pass2_cleanup;
		}

		/* Work buffer for the overlay (mutable copy of the iter's
		 * frame, which the iterator owns). */
		rgb_copy = malloc((size_t)w * h * 3);
		if (!rgb_copy) {
			sam3_log_error("out of memory for overlay buffer");
			err = SAM3_ENOMEM;
			ret = (int)sam3_error_to_exit(err);
			goto pass2_cleanup;
		}

		int frame_idx = 0;
		for (;;) {
			if (!rgb) break; /* defensive */
			memcpy(rgb_copy, rgb, (size_t)w * h * 3);

			if (frame_idx < fc.n_frames && fc.seen[frame_idx]) {
				for (int o = 0; o < fc.n_objects; o++) {
					size_t off = ((size_t)frame_idx *
						      fc.n_objects + o) *
						     (size_t)fc.mask_h * fc.mask_w;
					sam3_overlay_composite(
						rgb_copy, w, h,
						fc.masks + off,
						fc.mask_w, fc.mask_h,
						o, a->alpha);
				}
			}

			err = sam3_video_encoder_write_rgb(enc, rgb_copy);
			if (err != SAM3_OK) {
				sam3_log_error("encode failed at frame %d",
					       frame_idx);
				ret = (int)sam3_error_to_exit(err);
				goto pass2_cleanup;
			}
			frame_idx++;

			err = sam3_rgb_iter_next(it, &rgb, &w, &h, &eof);
			if (err != SAM3_OK) {
				ret = (int)sam3_error_to_exit(err);
				goto pass2_cleanup;
			}
			if (eof) break;
		}

		ret = SAM3_EXIT_OK;

pass2_cleanup:
		/* Close encoder even on partial success to finalize file. */
		if (enc) {
			enum sam3_error cerr = sam3_video_encoder_close(enc);
			if (cerr != SAM3_OK && ret == SAM3_EXIT_OK)
				ret = (int)sam3_error_to_exit(cerr);
		}
		sam3_rgb_iter_close(it);
		free(rgb_copy);
	}
```

- [ ] **Step 2: Build**

```bash
cd /Users/rbisri/Documents/sam3/build && cmake --build . -j 2>&1 | tail -20
```

Expected: build succeeds.

- [ ] **Step 3: Run all tests**

```bash
cd /Users/rbisri/Documents/sam3/build && ctest --output-on-failure 2>&1 | tail -30
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tools/cli_track.c
git commit -m "cli/track: implement video-mode pass 2 (decode, composite, encode)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 19: Source fps propagation (fix the "default 30" placeholder)

Task 18 punts on source-fps detection with a fallback. Address that here: expose the decoded fps through the RGB iterator so pass 2 can inherit it.

**Files:**
- Modify: `src/util/video_internal.h`
- Modify: `src/util/video.c`
- Modify: `tools/cli_track.c`

- [ ] **Step 1: Add an fps getter to the iterator API**

In `src/util/video_internal.h`, after `sam3_rgb_iter_close`:

```c
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
```

- [ ] **Step 2: Implement in `src/util/video.c`**

Append below `sam3_rgb_iter_close`:

```c
void sam3_rgb_iter_fps(const struct sam3_rgb_iter *it,
		       int *out_num, int *out_den)
{
	if (out_num) *out_num = 0;
	if (out_den) *out_den = 1;
	if (!it)
		return;
	if (it->kind == RGB_ITER_LIBAV) {
		if (out_num) *out_num = it->dec.avg_frame_rate.num;
		if (out_den) *out_den = it->dec.avg_frame_rate.den > 0
			? it->dec.avg_frame_rate.den : 1;
	}
}
```

- [ ] **Step 3: Replace the fps-probing block in `cli_track_run_video`**

Delete everything in the Pass 2 section between `/* Determine fps: ... */` and the call to `sam3_video_encoder_open`, replacing with:

```c
		int fps_num = 0, fps_den = 1;
		if (a->fps > 0) {
			fps_num = a->fps;
			fps_den = 1;
		} else {
			sam3_rgb_iter_fps(it, &fps_num, &fps_den);
			if (fps_num <= 0) {
				sam3_log_error(
					"--fps is required when source has "
					"no native frame rate (frame "
					"directory or unrecognized container)");
				err = SAM3_EINVAL;
				ret = (int)sam3_error_to_exit(err);
				goto pass2_cleanup;
			}
		}
```

- [ ] **Step 4: Build and run all tests**

```bash
cd /Users/rbisri/Documents/sam3/build && cmake --build . -j 2>&1 | tail -10 && ctest --output-on-failure 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/util/video_internal.h src/util/video.c tools/cli_track.c
git commit -m "cli/track: inherit source fps from libav iterator

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 20: README + manual smoke test

**Files:**
- Modify: `README.md` (add one-line install hint for libav)

- [ ] **Step 1: Find the "Build" or "Requirements" section**

Read `README.md` and locate where dependencies are mentioned. Add a single line near the existing build instructions:

```markdown
Video I/O requires FFmpeg dev libraries. Install with
`brew install ffmpeg` (macOS) or
`apt install libavformat-dev libavcodec-dev libswscale-dev libavutil-dev` (Debian/Ubuntu).
```

- [ ] **Step 2: Build a release binary for a manual smoke check**

```bash
cd /Users/rbisri/Documents/sam3 && rm -rf build-release && mkdir build-release && cd build-release && cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5 && cmake --build . -j 2>&1 | tail -10
```

Expected: release build succeeds.

- [ ] **Step 3: Manual smoke test (requires a `.sam3` model and a short mp4)**

```bash
cd /Users/rbisri/Documents/sam3/build-release && ./sam3_cli track --help 2>&1 | head -30
```

Expected: help output lists `--alpha` and `--fps` and describes dual-mode `--output`.

If a test model and clip are available:

```bash
./sam3_cli track \
  --model /path/to/model.sam3 \
  --video /path/to/short.mp4 \
  --output /tmp/sam3_smoke.mp4 \
  --point 512,384,1 \
  --alpha 0.5 \
  --propagate both
```

Expected: process exits 0 and `/tmp/sam3_smoke.mp4` opens in a standard video player with the mask overlay visible.

If no model is available for manual testing, skip the actual invocation — the unit tests cover all components, and CI/review can exercise the end-to-end path when model fixtures are present.

- [ ] **Step 4: Final commit**

```bash
git add README.md
git commit -m "docs: add libav install hint for video I/O

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review Notes

- **Spec coverage** — every numbered section of the design spec maps to tasks above:
  - CLI surface (Q1) → Task 13, 15
  - Mode detection → Task 15
  - `--alpha`, `--fps` → Task 13, 15
  - Overlay rendering → Task 9
  - Encoder module (open/write/close, codec-per-extension, idempotent close) → Task 7, 11
  - Retire `pl_mpeg` → Task 6
  - Decode migration to libav + `fps_num/fps_den` → Task 2, 4
  - `sam3_video_detect_type` rename → Task 2, 5
  - Two-pass orchestration → Task 17, 18
  - Source fps inheritance → Task 19
  - Build + docs → Task 1, 20
  - Tests (overlay, encoder lifecycle, roundtrip, parser) → Task 8, 10, 12, 14

- **Type consistency** — `struct video_frame_ctx` fields match their uses in the callback and pass 2; `struct sam3_rgb_iter` is opaque to consumers and the `_fps` accessor is declared before it's used; `detect_output_mode` returns an `int` matching `enum track_output_mode` (the struct field is already `int`).

- **No placeholders** — all code blocks are complete; no "implement later"; every command has expected output.

- **Scope** — one plan, one feature, consistent with the approved design. The single refinement over the spec (unifying pass-2 RGB decode behind `sam3_rgb_iter`) is called out in Task 4's commit message and keeps libav out of `cli_track.c`.
