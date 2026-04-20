/*
 * tests/test_helpers_png.c - PNG helpers for parity fixture tests.
 *
 * stb_image / stb_image_write implementations are defined once in
 * src/util/image.c (compiled into the sam3 library). This TU only
 * calls the extern symbols.
 *
 * Key types:  (none)
 * Depends on: test_helpers_png.h, src/util/vendor/stb_image.h
 * Used by:    tests/test_video_parity_kids.c (SAM 3.1 variant)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_helpers_png.h"
#include "util/log.h"

/* stb_image / stb_image_write declarations without re-defining the
 * implementation (implementation lives in src/util/image.c). */
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#endif
#include "util/vendor/stb_image.h"
#include "util/vendor/stb_image_write.h"
#ifdef __clang__
#pragma clang diagnostic pop
#endif

uint8_t *load_png_grayscale(const char *path, int *out_h, int *out_w)
{
	int w, h, channels_in_file;

	if (!path || !out_h || !out_w) {
		sam3_log_error("load_png_grayscale: NULL arg");
		return NULL;
	}

	/* Request 1 channel; stb down-converts automatically. */
	uint8_t *data = stbi_load(path, &w, &h, &channels_in_file, 1);
	if (!data) {
		sam3_log_error("load_png_grayscale: %s: %s",
			       path, stbi_failure_reason());
		return NULL;
	}
	*out_h = h;
	*out_w = w;
	return data;
}

int save_png_grayscale(const char *path, const uint8_t *data,
		       int h, int w)
{
	if (!path || !data || h <= 0 || w <= 0) {
		sam3_log_error("save_png_grayscale: bad arg");
		return -1;
	}
	int stride = w;
	if (!stbi_write_png(path, w, h, 1, data, stride)) {
		sam3_log_error("save_png_grayscale: stbi_write_png failed "
			       "for %s", path);
		return -1;
	}
	return 0;
}
