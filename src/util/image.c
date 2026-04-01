/*
 * src/util/image.c - Image loading and resizing implementation
 *
 * Implements image I/O using vendored stb_image.h for decoding,
 * stb_image_resize2.h for bilinear resize, and stb_image_write.h for
 * PNG/BMP/TGA/JPEG output. The stb implementations are compiled here
 * via #define STB_*_IMPLEMENTATION includes.
 *
 * Key types:  sam3_image
 * Depends on: util/image.h, vendor/stb_image.h, vendor/stb_image_resize2.h,
 *             vendor/stb_image_write.h
 * Used by:    src/sam3.c, tools/sam3_main.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include <string.h>
#include "util/image.h"
#include "util/log.h"

/* Suppress warnings in vendored stb headers */
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
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wmissing-prototypes"
#pragma GCC diagnostic ignored "-Wstrict-prototypes"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "vendor/stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "vendor/stb_image_resize2.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vendor/stb_image_write.h"

#ifdef __clang__
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

enum sam3_error sam3_image_load(const char *path, struct sam3_image *img)
{
	if (!path || !img)
		return SAM3_EINVAL;

	int w, h, channels;
	uint8_t *data = stbi_load(path, &w, &h, &channels, 3);
	if (!data) {
		sam3_log_error("failed to load image '%s': %s",
			       path, stbi_failure_reason());
		return SAM3_EIO;
	}

	img->pixels = data;
	img->width  = w;
	img->height = h;
	return SAM3_OK;
}

enum sam3_error sam3_image_resize(const struct sam3_image *src,
				  struct sam3_image *dst,
				  int target_w, int target_h)
{
	if (!src || !src->pixels || !dst || target_w <= 0 || target_h <= 0)
		return SAM3_EINVAL;

	uint8_t *out = malloc((size_t)target_w * target_h * 3);
	if (!out)
		return SAM3_ENOMEM;

	uint8_t *result = stbir_resize_uint8_linear(
		src->pixels, src->width, src->height, 0,
		out, target_w, target_h, 0, STBIR_RGB);
	if (!result) {
		free(out);
		return SAM3_EIO;
	}

	dst->pixels = out;
	dst->width  = target_w;
	dst->height = target_h;
	return SAM3_OK;
}

enum sam3_error sam3_image_letterbox(const struct sam3_image *src,
				     struct sam3_image *dst,
				     int target_size)
{
	if (!src || !src->pixels || !dst || target_size <= 0)
		return SAM3_EINVAL;

	/* Compute scaled dimensions preserving aspect ratio */
	int sw, sh;
	if (src->width >= src->height) {
		sw = target_size;
		sh = (int)(((size_t)src->height * target_size + src->width / 2) / src->width);
	} else {
		sh = target_size;
		sw = (int)(((size_t)src->width * target_size + src->height / 2) / src->height);
	}

	/* Resize to scaled dimensions */
	struct sam3_image scaled = {0};
	enum sam3_error err = sam3_image_resize(src, &scaled, sw, sh);
	if (err)
		return err;

	/* Allocate square output, zero-filled (black padding) */
	size_t out_size = (size_t)target_size * target_size * 3;
	uint8_t *out = calloc(1, out_size);
	if (!out) {
		sam3_image_free(&scaled);
		return SAM3_ENOMEM;
	}

	/* Copy scaled image into center of output */
	int pad_x = (target_size - sw) / 2;
	int pad_y = (target_size - sh) / 2;
	for (int y = 0; y < sh; y++) {
		const uint8_t *src_row = scaled.pixels + y * sw * 3;
		uint8_t *dst_row = out + ((pad_y + y) * target_size + pad_x) * 3;
		memcpy(dst_row, src_row, (size_t)sw * 3);
	}

	sam3_image_free(&scaled);

	dst->pixels = out;
	dst->width  = target_size;
	dst->height = target_size;
	return SAM3_OK;
}

void sam3_image_free(struct sam3_image *img)
{
	if (!img)
		return;
	/*
	 * stbi_load uses malloc internally, and our resize/letterbox use
	 * malloc/calloc, so free() is correct for all paths.
	 */
	free(img->pixels);
	img->pixels = NULL;
	img->width  = 0;
	img->height = 0;
}
