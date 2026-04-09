/*
 * include/sam3/internal/processor_normalize.h - Pixel normalization helper.
 *
 * Internal header exposing the uint8 HWC to float CHW normalization used
 * by the image processor. Exposed for unit testing. Not part of the
 * public API. The normalization matches the Python reference:
 * (x/255 - 0.5) / 0.5, equivalent to x/127.5 - 1.0, producing values
 * in [-1, 1].
 *
 * Key types:  none
 * Depends on: <stdint.h>
 * Used by:    src/model/sam3_processor.c, tests/test_processor_normalize.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */
#ifndef SAM3_INTERNAL_PROCESSOR_NORMALIZE_H
#define SAM3_INTERNAL_PROCESSOR_NORMALIZE_H

#include <stdint.h>

/*
 * sam3_normalize_rgb_chw - Normalize uint8 HWC RGB pixels into float CHW.
 *
 * @src:    Source buffer, uint8 RGB interleaved, layout HWC (width*height*3)
 * @dst:    Destination buffer, float CHW planar (3*height*width)
 * @width:  Image width in pixels
 * @height: Image height in pixels
 *
 * Applies the SAM3 reference normalization: each channel value is mapped
 * from [0, 255] to [-1, 1] via x/127.5 - 1.0.
 */
void sam3_normalize_rgb_chw(const uint8_t *src, float *dst,
			    int width, int height);

#endif /* SAM3_INTERNAL_PROCESSOR_NORMALIZE_H */
