/*
 * src/util/image.h - Image loading and resizing utilities
 *
 * Provides functions to load images from disk (PNG/JPEG/BMP) into RGB
 * uint8 buffers, resize them with bilinear interpolation, and letterbox
 * to a square target size. Backed by vendored stb_image.h and
 * stb_image_resize2.h.
 *
 * Key types:  sam3_image
 * Depends on: sam3/sam3_types.h
 * Used by:    src/sam3.c, tools/sam3_main.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_UTIL_IMAGE_H
#define SAM3_UTIL_IMAGE_H

#include "sam3/sam3_types.h"

struct sam3_image {
	uint8_t *pixels;    /* RGB interleaved, 3 bytes per pixel */
	int      width;
	int      height;
};

/*
 * sam3_image_load - Load an image file into an RGB buffer.
 *
 * @path: Path to PNG, JPEG, or BMP file.
 * @img:  Output image struct (caller must call sam3_image_free).
 *
 * Returns SAM3_OK on success, SAM3_EINVAL if args are NULL,
 * SAM3_EIO if file cannot be read or decoded.
 */
enum sam3_error sam3_image_load(const char *path, struct sam3_image *img);

/*
 * sam3_image_resize - Resize an image to target dimensions.
 *
 * @src:      Source image (must have valid pixels).
 * @dst:      Output image (caller must call sam3_image_free).
 * @target_w: Target width in pixels.
 * @target_h: Target height in pixels.
 *
 * Uses bilinear interpolation. Allocates dst->pixels via malloc.
 * Returns SAM3_OK on success, SAM3_ENOMEM on allocation failure.
 */
enum sam3_error sam3_image_resize(const struct sam3_image *src,
				  struct sam3_image *dst,
				  int target_w, int target_h);

/*
 * sam3_image_letterbox - Resize preserving aspect ratio, pad to square.
 *
 * @src:         Source image.
 * @dst:         Output image (square, caller must call sam3_image_free).
 * @target_size: Target side length (e.g. 1024).
 *
 * Fits longest edge to target_size, pads shorter dimension with zeros.
 * Returns SAM3_OK on success.
 */
enum sam3_error sam3_image_letterbox(const struct sam3_image *src,
				     struct sam3_image *dst,
				     int target_size);

/*
 * sam3_image_free - Free pixel data allocated by load/resize/letterbox.
 *
 * @img: Image to free (pixels set to NULL after).
 */
void sam3_image_free(struct sam3_image *img);

#endif /* SAM3_UTIL_IMAGE_H */
