/*
 * tests/test_helpers_png.h - 8-bit grayscale PNG load/save for tests.
 *
 * Thin wrappers around the stb_image.h / stb_image_write.h APIs whose
 * implementations are already defined once in src/util/image.c (linked
 * into the sam3 library). Test translation units only need to #include
 * this header; they do not re-declare the stb implementations.
 *
 * Key types:  uint8_t buffers + int dimensions
 * Depends on: <stdint.h>, src/util/vendor/stb_image.h (impl in sam3 lib)
 * Used by:    tests/test_video_parity_kids.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_TEST_HELPERS_PNG_H
#define SAM3_TEST_HELPERS_PNG_H

#include <stdint.h>

/*
 * load_png_grayscale - Load an 8-bit single-channel PNG.
 *
 * Returns a malloc'd buffer of size (*out_h) * (*out_w) uint8_t, or
 * NULL on error. Caller frees via free(). PNGs with >1 channel are
 * forced to grayscale by the decoder.
 */
uint8_t *load_png_grayscale(const char *path, int *out_h, int *out_w);

/*
 * save_png_grayscale - Write an 8-bit single-channel PNG.
 *
 * Returns 0 on success, -1 on error (errors are logged via
 * sam3_log_error). @data is row-major, size h*w.
 */
int save_png_grayscale(const char *path, const uint8_t *data,
		       int h, int w);

#endif /* SAM3_TEST_HELPERS_PNG_H */
