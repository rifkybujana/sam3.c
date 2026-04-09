/*
 * src/util/mask_boxes.c - Bounding box extraction from masks.
 *
 * Derives axis-aligned bounding boxes from mask logits. Scans each
 * mask once to find min/max x and y coordinates of positive pixels.
 * No allocations — caller provides the output buffer.
 *
 * Key types:  none
 * Depends on: sam3/internal/mask_boxes.h
 * Used by:    src/model/sam3_processor.c, tools/sam3_main.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stddef.h>

#include "sam3/internal/mask_boxes.h"

int sam3_masks_to_boxes(const float *masks, int n_masks,
			int h, int w, float *boxes_out)
{
	if (!masks || !boxes_out || n_masks <= 0 || h <= 0 || w <= 0)
		return -1;

	int n_pix = h * w;

	for (int m = 0; m < n_masks; m++) {
		const float *mask = masks + (size_t)m * n_pix;
		int x_min = w, y_min = h, x_max = -1, y_max = -1;

		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				if (mask[y * w + x] > 0.0f) {
					if (x < x_min) x_min = x;
					if (x > x_max) x_max = x;
					if (y < y_min) y_min = y;
					if (y > y_max) y_max = y;
				}
			}
		}

		float *box = boxes_out + m * 4;
		if (x_max < 0) {
			box[0] = box[1] = box[2] = box[3] = 0.0f;
		} else {
			box[0] = (float)x_min;
			box[1] = (float)y_min;
			box[2] = (float)(x_max + 1);
			box[3] = (float)(y_max + 1);
		}
	}

	return 0;
}
