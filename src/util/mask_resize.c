/*
 * src/util/mask_resize.c - Bilinear mask resize.
 *
 * Implements bilinear interpolation matching PyTorch's
 * F.interpolate(mode="bilinear", align_corners=False). Used to resize
 * mask logits from decoder resolution to original image resolution.
 * No allocations — caller provides the output buffer.
 *
 * Key types:  none
 * Depends on: sam3/internal/mask_resize.h
 * Used by:    src/model/sam3_processor.c, tools/sam3_main.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "sam3/internal/mask_resize.h"

int sam3_mask_resize_bilinear(const float *src, int src_w, int src_h,
			      float *dst, int dst_w, int dst_h)
{
	if (!src || !dst || src_w <= 0 || src_h <= 0 ||
	    dst_w <= 0 || dst_h <= 0)
		return -1;

	float sx = (float)src_w / (float)dst_w;
	float sy = (float)src_h / (float)dst_h;

	for (int dy = 0; dy < dst_h; dy++) {
		float fy = ((float)dy + 0.5f) * sy - 0.5f;

		/* Clamp to valid source range */
		if (fy < 0.0f) fy = 0.0f;
		if (fy > (float)(src_h - 1)) fy = (float)(src_h - 1);

		int y0 = (int)fy;
		int y1 = y0 + 1;
		if (y1 >= src_h) y1 = src_h - 1;
		float wy = fy - (float)y0;

		for (int dx = 0; dx < dst_w; dx++) {
			float fx = ((float)dx + 0.5f) * sx - 0.5f;

			if (fx < 0.0f) fx = 0.0f;
			if (fx > (float)(src_w - 1))
				fx = (float)(src_w - 1);

			int x0 = (int)fx;
			int x1 = x0 + 1;
			if (x1 >= src_w) x1 = src_w - 1;
			float wx = fx - (float)x0;

			float v00 = src[y0 * src_w + x0];
			float v01 = src[y0 * src_w + x1];
			float v10 = src[y1 * src_w + x0];
			float v11 = src[y1 * src_w + x1];

			float top = v00 + (v01 - v00) * wx;
			float bot = v10 + (v11 - v10) * wx;

			dst[dy * dst_w + dx] = top + (bot - top) * wy;
		}
	}

	return 0;
}
