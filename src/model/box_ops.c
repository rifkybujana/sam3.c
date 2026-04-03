/*
 * src/model/box_ops.c - Bounding box operations for SAM3 mask decoder
 *
 * Implements box format conversions (xyxy, cxcywh, xywh), Intersection
 * over Union, Generalized IoU, and greedy Non-Maximum Suppression.
 * All functions operate on raw float arrays and have no project
 * dependencies beyond standard C math.
 *
 * Key types:  (none, operates on raw float arrays)
 * Depends on: box_ops.h
 * Used by:    decoder.c, tests/test_box_ops.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "box_ops.h"

void box_xyxy_to_cxcywh(const float *xyxy, float *cxcywh, int n)
{
	for (int i = 0; i < n; i++) {
		float x1 = xyxy[i * 4 + 0];
		float y1 = xyxy[i * 4 + 1];
		float x2 = xyxy[i * 4 + 2];
		float y2 = xyxy[i * 4 + 3];

		cxcywh[i * 4 + 0] = (x1 + x2) * 0.5f;
		cxcywh[i * 4 + 1] = (y1 + y2) * 0.5f;
		cxcywh[i * 4 + 2] = x2 - x1;
		cxcywh[i * 4 + 3] = y2 - y1;
	}
}

void box_cxcywh_to_xyxy(const float *cxcywh, float *xyxy, int n)
{
	for (int i = 0; i < n; i++) {
		float cx = cxcywh[i * 4 + 0];
		float cy = cxcywh[i * 4 + 1];
		float w  = cxcywh[i * 4 + 2];
		float h  = cxcywh[i * 4 + 3];

		xyxy[i * 4 + 0] = cx - w * 0.5f;
		xyxy[i * 4 + 1] = cy - h * 0.5f;
		xyxy[i * 4 + 2] = cx + w * 0.5f;
		xyxy[i * 4 + 3] = cy + h * 0.5f;
	}
}

void box_xyxy_to_xywh(const float *xyxy, float *xywh, int n)
{
	for (int i = 0; i < n; i++) {
		float x1 = xyxy[i * 4 + 0];
		float y1 = xyxy[i * 4 + 1];
		float x2 = xyxy[i * 4 + 2];
		float y2 = xyxy[i * 4 + 3];

		xywh[i * 4 + 0] = x1;
		xywh[i * 4 + 1] = y1;
		xywh[i * 4 + 2] = x2 - x1;
		xywh[i * 4 + 3] = y2 - y1;
	}
}

static float maxf(float a, float b)
{
	return a > b ? a : b;
}

static float minf(float a, float b)
{
	return a < b ? a : b;
}

float box_iou(const float *a, const float *b)
{
	float ix1 = maxf(a[0], b[0]);
	float iy1 = maxf(a[1], b[1]);
	float ix2 = minf(a[2], b[2]);
	float iy2 = minf(a[3], b[3]);

	float iw = maxf(0.0f, ix2 - ix1);
	float ih = maxf(0.0f, iy2 - iy1);
	float inter = iw * ih;

	float area_a = (a[2] - a[0]) * (a[3] - a[1]);
	float area_b = (b[2] - b[0]) * (b[3] - b[1]);
	float uni = area_a + area_b - inter;

	if (uni <= 0.0f)
		return 0.0f;

	return inter / uni;
}

float box_giou(const float *a, const float *b)
{
	float ix1 = maxf(a[0], b[0]);
	float iy1 = maxf(a[1], b[1]);
	float ix2 = minf(a[2], b[2]);
	float iy2 = minf(a[3], b[3]);

	float iw = maxf(0.0f, ix2 - ix1);
	float ih = maxf(0.0f, iy2 - iy1);
	float inter = iw * ih;

	float area_a = (a[2] - a[0]) * (a[3] - a[1]);
	float area_b = (b[2] - b[0]) * (b[3] - b[1]);
	float uni = area_a + area_b - inter;

	/* Smallest enclosing box */
	float cx1 = minf(a[0], b[0]);
	float cy1 = minf(a[1], b[1]);
	float cx2 = maxf(a[2], b[2]);
	float cy2 = maxf(a[3], b[3]);
	float c_area = (cx2 - cx1) * (cy2 - cy1);

	if (c_area <= 0.0f)
		return 0.0f;

	float iou = (uni > 0.0f) ? inter / uni : 0.0f;

	return iou - (c_area - uni) / c_area;
}

int box_nms(const float *boxes, const float *scores, int n,
	    float iou_threshold, int *keep)
{
	if (n <= 0)
		return 0;

	/* Build index array and sort by score descending (selection sort) */
	int order[n];
	for (int i = 0; i < n; i++)
		order[i] = i;

	for (int i = 0; i < n - 1; i++) {
		int best = i;
		for (int j = i + 1; j < n; j++) {
			if (scores[order[j]] > scores[order[best]])
				best = j;
		}
		if (best != i) {
			int tmp = order[i];
			order[i] = order[best];
			order[best] = tmp;
		}
	}

	/* Greedy suppression */
	int suppressed[n];
	for (int i = 0; i < n; i++)
		suppressed[i] = 0;

	int n_keep = 0;
	for (int i = 0; i < n; i++) {
		int idx = order[i];
		if (suppressed[idx])
			continue;

		keep[n_keep++] = idx;

		for (int j = i + 1; j < n; j++) {
			int jdx = order[j];
			if (suppressed[jdx])
				continue;

			float iou = box_iou(&boxes[idx * 4],
					    &boxes[jdx * 4]);
			if (iou > iou_threshold)
				suppressed[jdx] = 1;
		}
	}

	return n_keep;
}
