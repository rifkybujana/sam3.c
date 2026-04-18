/*
 * src/util/video.c - Video frame loading implementation
 *
 * Loads video frames from MPEG files (via pl_mpeg) or directories of
 * images (via stb_image). Handles resize to model input size and
 * normalization to [-1, 1]. The pl_mpeg implementation is compiled here
 * via #define PL_MPEG_IMPLEMENTATION. stb_image is included as
 * declaration-only (implementation lives in image.c).
 *
 * Key types:  sam3_video_frames, sam3_video_type
 * Depends on: util/video.h, util/log.h, vendor/pl_mpeg.h,
 *             vendor/stb_image.h, vendor/stb_image_resize2.h
 * Used by:    model/video_session.c
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
#include "util/log.h"

/* stb_image: declaration only (implementation in image.c) */
#include "vendor/stb_image.h"
#include "vendor/stb_image_resize2.h"

/* Suppress warnings in vendored pl_mpeg header */
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wstrict-prototypes"
#pragma clang diagnostic ignored "-Wdouble-promotion"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wimplicit-fallthrough"
#pragma clang diagnostic ignored "-Wcomma"
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
#pragma clang diagnostic ignored "-Wconditional-uninitialized"
#pragma clang diagnostic ignored "-Wcast-align"
#pragma clang diagnostic ignored "-Wcast-qual"
#pragma clang diagnostic ignored "-Wextra-semi-stmt"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"
#pragma clang diagnostic ignored "-Wundef"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wmissing-prototypes"
#pragma GCC diagnostic ignored "-Wstrict-prototypes"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif

#define PL_MPEG_IMPLEMENTATION
#include "vendor/pl_mpeg.h"

#ifdef __clang__
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

/* Maximum frames we support loading at once */
#define MAX_DIR_FRAMES 4096

/* Case-insensitive string suffix check */
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

/* Check if filename has an image extension we support */
static int is_image_file(const char *name)
{
	return has_suffix(name, ".jpg") || has_suffix(name, ".jpeg") ||
	       has_suffix(name, ".png") || has_suffix(name, ".ppm") ||
	       has_suffix(name, ".bmp");
}

/*
 * Try to extract a leading integer from a filename for numeric sorting.
 * Returns 1 if successful, 0 otherwise. Fills *val on success.
 */
static int parse_numeric_prefix(const char *name, long *val)
{
	char *end;
	long v = strtol(name, &end, 10);
	/* Must have consumed at least one digit and hit '.' or end */
	if (end == name)
		return 0;
	*val = v;
	return 1;
}

/* qsort comparator: try numeric first, fall back to alphabetical */
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

/*
 * Allocate a [3, H, W] F32 tensor from the arena and fill it with
 * normalized pixel data from an RGB uint8 image. Pixels are stored
 * in CHW order: (pixel/255.0f - 0.5f) / 0.5f.
 */
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

	/* Convert HWC uint8 -> CHW float, normalized to [-1, 1] */
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

/*
 * Load frames from a directory of image files.
 * Lists, sorts, loads, resizes, and normalizes each image.
 */
static enum sam3_error load_frame_dir(const char *path, int image_size,
				      struct sam3_video_frames *out,
				      struct sam3_arena *arena)
{
	DIR *dir = opendir(path);
	if (!dir) {
		sam3_log_error("cannot open frame directory '%s'", path);
		return SAM3_EIO;
	}

	/* Collect image filenames (heap-allocated, freed in cleanup) */
	char **names = NULL;
	uint8_t *resized = NULL;
	int n_files = 0;
	int cap = 0;
	struct dirent *ent;
	enum sam3_error err = SAM3_OK;

	while ((ent = readdir(dir)) != NULL) {
		if (!is_image_file(ent->d_name))
			continue;
		if (n_files >= MAX_DIR_FRAMES) {
			sam3_log_warn("frame directory capped at %d files",
				      MAX_DIR_FRAMES);
			break;
		}
		if (n_files >= cap) {
			cap = cap ? cap * 2 : 64;
			char **tmp = realloc(names,
					     (size_t)cap * sizeof(char *));
			if (!tmp) {
				err = SAM3_ENOMEM;
				goto cleanup;
			}
			names = tmp;
		}
		names[n_files] = strdup(ent->d_name);
		if (!names[n_files]) {
			err = SAM3_ENOMEM;
			goto cleanup;
		}
		n_files++;
	}
	closedir(dir);
	dir = NULL;

	if (n_files == 0) {
		sam3_log_error("no image files in '%s'", path);
		err = SAM3_EIO;
		goto cleanup;
	}

	/* Sort: numeric first, alphabetical fallback */
	qsort(names, (size_t)n_files, sizeof(char *), cmp_filenames);

	/* Allocate pointer array for frame tensors */
	out->pixels = sam3_arena_alloc(arena,
				       (size_t)n_files *
				       sizeof(struct sam3_tensor *));
	if (!out->pixels) {
		err = SAM3_ENOMEM;
		goto cleanup;
	}

	out->n_frames = 0;
	out->frame_size = image_size;

	/* Temporary resize buffer (heap, freed in cleanup) */
	resized = malloc((size_t)image_size * image_size * 3);
	if (!resized) {
		sam3_log_error("out of memory for resize buffer");
		err = SAM3_ENOMEM;
		goto cleanup;
	}

	for (int i = 0; i < n_files; i++) {
		/* Build full path */
		char fpath[1024];
		int n = snprintf(fpath, sizeof(fpath), "%s/%s", path,
				 names[i]);
		if (n < 0 || (size_t)n >= sizeof(fpath)) {
			sam3_log_warn("path too long, skipping '%s'",
				      names[i]);
			continue;
		}

		/* Load image */
		int w, h, channels;
		uint8_t *pixels = stbi_load(fpath, &w, &h, &channels, 3);
		if (!pixels) {
			sam3_log_error("failed to load '%s': %s", fpath,
				       stbi_failure_reason());
			err = SAM3_EIO;
			goto cleanup;
		}

		/* Record original dimensions from first frame */
		if (out->n_frames == 0) {
			out->orig_width = w;
			out->orig_height = h;
		}

		/* Resize to image_size x image_size */
		uint8_t *result = stbir_resize_uint8_linear(
			pixels, w, h, 0,
			resized, image_size, image_size, 0,
			STBIR_RGB);
		stbi_image_free(pixels);

		if (!result) {
			sam3_log_error("resize failed for '%s'", fpath);
			err = SAM3_EIO;
			goto cleanup;
		}

		/* Convert to F32 CHW tensor */
		struct sam3_tensor *t = make_frame_tensor(resized,
							  image_size,
							  image_size,
							  arena);
		if (!t) {
			sam3_log_error("out of memory encoding frame '%s'",
				       fpath);
			err = SAM3_ENOMEM;
			goto cleanup;
		}

		out->pixels[out->n_frames++] = t;
	}

	sam3_log_info("loaded %d frames from '%s' (%dx%d -> %d)",
		      out->n_frames, path, out->orig_width,
		      out->orig_height, image_size);

cleanup:
	free(resized);
	if (dir)
		closedir(dir);
	for (int i = 0; i < n_files; i++)
		free(names[i]);
	free(names);
	return err;
}

/*
 * Load frames from an MPEG file via pl_mpeg.
 * Decodes all video frames, converts YCbCr to RGB, resizes, normalizes.
 */
static enum sam3_error load_mpeg(const char *path, int image_size,
				 struct sam3_video_frames *out,
				 struct sam3_arena *arena)
{
	plm_t *plm = plm_create_with_filename(path);
	if (!plm) {
		sam3_log_error("cannot open MPEG file '%s'", path);
		return SAM3_EIO;
	}

	plm_set_audio_enabled(plm, 0);

	int vw = plm_get_width(plm);
	int vh = plm_get_height(plm);
	if (vw <= 0 || vh <= 0) {
		sam3_log_error("invalid MPEG dimensions %dx%d", vw, vh);
		plm_destroy(plm);
		return SAM3_EIO;
	}

	out->orig_width = vw;
	out->orig_height = vh;
	out->frame_size = image_size;

	/* First pass: count frames */
	int n_total = 0;
	while (plm_decode_video(plm))
		n_total++;

	if (n_total == 0) {
		sam3_log_error("no video frames in '%s'", path);
		plm_destroy(plm);
		return SAM3_EIO;
	}

	/* Reopen to decode again */
	plm_destroy(plm);
	plm = plm_create_with_filename(path);
	if (!plm) {
		sam3_log_error("cannot reopen MPEG file '%s'", path);
		return SAM3_EIO;
	}
	plm_set_audio_enabled(plm, 0);

	out->pixels = sam3_arena_alloc(arena,
				       (size_t)n_total *
				       sizeof(struct sam3_tensor *));
	if (!out->pixels) {
		plm_destroy(plm);
		return SAM3_ENOMEM;
	}

	/* Temporary RGB buffer for frame conversion */
	uint8_t *rgb = malloc((size_t)vw * vh * 3);
	uint8_t *resized = malloc((size_t)image_size * image_size * 3);
	if (!rgb || !resized) {
		sam3_log_error("out of memory for MPEG decode buffers");
		free(rgb);
		free(resized);
		plm_destroy(plm);
		return SAM3_ENOMEM;
	}

	out->n_frames = 0;
	plm_frame_t *frame;
	while ((frame = plm_decode_video(plm)) != NULL &&
	       out->n_frames < n_total) {
		/* Convert YCbCr to RGB */
		plm_frame_to_rgb(frame, rgb, vw * 3);

		/* Resize to target */
		uint8_t *result = stbir_resize_uint8_linear(
			rgb, vw, vh, 0,
			resized, image_size, image_size, 0,
			STBIR_RGB);
		if (!result) {
			sam3_log_error("resize failed for MPEG frame %d",
				       out->n_frames);
			free(rgb);
			free(resized);
			plm_destroy(plm);
			return SAM3_EIO;
		}

		struct sam3_tensor *t = make_frame_tensor(resized,
							  image_size,
							  image_size,
							  arena);
		if (!t) {
			sam3_log_error("out of memory encoding MPEG frame %d",
				       out->n_frames);
			free(rgb);
			free(resized);
			plm_destroy(plm);
			return SAM3_ENOMEM;
		}

		out->pixels[out->n_frames++] = t;
	}

	free(rgb);
	free(resized);
	plm_destroy(plm);

	sam3_log_info("loaded %d MPEG frames from '%s' (%dx%d -> %d)",
		      out->n_frames, path, vw, vh, image_size);
	return SAM3_OK;
}

enum sam3_video_type sam3_video_detect_type(const char *path)
{
	if (!path)
		return SAM3_VIDEO_UNKNOWN;

	struct stat st;
	if (stat(path, &st) == 0 && S_ISDIR(st.st_mode))
		return SAM3_VIDEO_FRAME_DIR;

	if (has_suffix(path, ".mpg") || has_suffix(path, ".mpeg"))
		return SAM3_VIDEO_MPEG;

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
	case SAM3_VIDEO_MPEG:
		return load_mpeg(path, image_size, out, arena);
	default:
		sam3_log_error("unknown video source type for '%s'", path);
		return SAM3_EIO;
	}
}
