/*
 * src/util/mask_postprocess.c - Mask post-processing utilities.
 *
 * Implements sigmoid, morphological open, and connected-component
 * filtering for mask logit arrays. All functions operate on flat
 * arrays with caller-provided buffers — no heap allocations.
 *
 * Key types:  none
 * Depends on: sam3/internal/mask_postprocess.h, <math.h>
 * Used by:    tools/sam3_main.c, tests/test_mask_postprocess.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include <string.h>
#include "sam3/internal/mask_postprocess.h"

int sam3_mask_sigmoid(const float *src, float *dst, int n)
{
	if (!src || !dst || n <= 0)
		return -1;
	for (int i = 0; i < n; i++)
		dst[i] = 1.0f / (1.0f + expf(-src[i]));
	return 0;
}

/* 3x3 box erode: pixel is 1 only if all 8-neighbors + self are 1 */
static void erode3x3(const unsigned char *in, unsigned char *out,
		     int w, int h)
{
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			unsigned char val = 1;
			for (int dy = -1; dy <= 1 && val; dy++) {
				int ny = y + dy;
				if (ny < 0 || ny >= h) { val = 0; break; }
				for (int dx = -1; dx <= 1 && val; dx++) {
					int nx = x + dx;
					if (nx < 0 || nx >= w) { val = 0; break; }
					if (!in[ny * w + nx]) val = 0;
				}
			}
			out[y * w + x] = val;
		}
	}
}

/* 3x3 box dilate: pixel is 1 if any 8-neighbor or self is 1 */
static void dilate3x3(const unsigned char *in, unsigned char *out,
		      int w, int h)
{
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			unsigned char val = 0;
			for (int dy = -1; dy <= 1 && !val; dy++) {
				int ny = y + dy;
				if (ny < 0 || ny >= h) continue;
				for (int dx = -1; dx <= 1 && !val; dx++) {
					int nx = x + dx;
					if (nx < 0 || nx >= w) continue;
					if (in[ny * w + nx]) val = 1;
				}
			}
			out[y * w + x] = val;
		}
	}
}

int sam3_mask_morpho_open(const unsigned char *mask, unsigned char *out,
			  int w, int h, unsigned char *work)
{
	if (!mask || !out || !work || w <= 0 || h <= 0)
		return -1;
	erode3x3(mask, work, w, h);
	dilate3x3(work, out, w, h);
	return 0;
}

int sam3_mask_remove_small(unsigned char *mask, int w, int h,
			   int min_pixels, int *labels, int *stack)
{
	int n_pix, label_id, sp, size;
	int x, y, nx, ny;
	static const int dirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};

	if (!mask || !labels || !stack || w <= 0 || h <= 0)
		return -1;

	n_pix = w * h;
	memset(labels, 0, (size_t)n_pix * sizeof(int));
	label_id = 0;

	for (y = 0; y < h; y++) {
		for (x = 0; x < w; x++) {
			int idx = y * w + x;
			if (!mask[idx] || labels[idx])
				continue;

			/* BFS flood fill */
			label_id++;
			sp = 0;
			stack[sp++] = idx;
			labels[idx] = label_id;
			size = 0;

			while (sp > 0) {
				int cur = stack[--sp];
				size++;
				int cy = cur / w;
				int cx = cur % w;

				for (int d = 0; d < 4; d++) {
					ny = cy + dirs[d][0];
					nx = cx + dirs[d][1];
					if (ny < 0 || ny >= h ||
					    nx < 0 || nx >= w)
						continue;
					int ni = ny * w + nx;
					if (mask[ni] && !labels[ni]) {
						labels[ni] = label_id;
						stack[sp++] = ni;
					}
				}
			}

			/* If component too small, zero it out */
			if (size < min_pixels) {
				for (int i = 0; i < n_pix; i++) {
					if (labels[i] == label_id)
						mask[i] = 0;
				}
			}
		}
	}

	return 0;
}
