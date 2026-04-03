/*
 * tests/test_box_ops.c - Unit tests for bounding box operations
 *
 * Tests box format conversions (xyxy, cxcywh, xywh), roundtrip
 * consistency, IoU edge cases (perfect overlap, no overlap, partial),
 * GIoU with enclosed boxes, and greedy Non-Maximum Suppression.
 *
 * Key types:  (none)
 * Depends on: test_helpers.h, model/box_ops.h
 * Used by:    CTest
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "model/box_ops.h"
#include <math.h>

#define TOL 1e-5f

/* --- test_box_xyxy_to_cxcywh --- */

static void test_box_xyxy_to_cxcywh(void)
{
	float xyxy[4] = {10.0f, 20.0f, 30.0f, 40.0f};
	float cxcywh[4];

	box_xyxy_to_cxcywh(xyxy, cxcywh, 1);

	ASSERT_NEAR(cxcywh[0], 20.0f, TOL);  /* cx = (10+30)/2 */
	ASSERT_NEAR(cxcywh[1], 30.0f, TOL);  /* cy = (20+40)/2 */
	ASSERT_NEAR(cxcywh[2], 20.0f, TOL);  /* w  = 30-10 */
	ASSERT_NEAR(cxcywh[3], 20.0f, TOL);  /* h  = 40-20 */
}

/* --- test_box_cxcywh_to_xyxy --- */

static void test_box_cxcywh_to_xyxy(void)
{
	float cxcywh[4] = {20.0f, 30.0f, 20.0f, 20.0f};
	float xyxy[4];

	box_cxcywh_to_xyxy(cxcywh, xyxy, 1);

	ASSERT_NEAR(xyxy[0], 10.0f, TOL);  /* x1 = 20-10 */
	ASSERT_NEAR(xyxy[1], 20.0f, TOL);  /* y1 = 30-10 */
	ASSERT_NEAR(xyxy[2], 30.0f, TOL);  /* x2 = 20+10 */
	ASSERT_NEAR(xyxy[3], 40.0f, TOL);  /* y2 = 30+10 */
}

/* --- test_box_roundtrip --- */

static void test_box_roundtrip(void)
{
	float orig[4] = {5.0f, 15.0f, 25.0f, 45.0f};
	float cxcywh[4];
	float recovered[4];

	box_xyxy_to_cxcywh(orig, cxcywh, 1);
	box_cxcywh_to_xyxy(cxcywh, recovered, 1);

	ASSERT_NEAR(recovered[0], orig[0], TOL);
	ASSERT_NEAR(recovered[1], orig[1], TOL);
	ASSERT_NEAR(recovered[2], orig[2], TOL);
	ASSERT_NEAR(recovered[3], orig[3], TOL);
}

/* --- test_box_iou_perfect --- */

static void test_box_iou_perfect(void)
{
	float box[4] = {10.0f, 20.0f, 30.0f, 40.0f};

	float iou = box_iou(box, box);
	ASSERT_NEAR(iou, 1.0f, TOL);
}

/* --- test_box_iou_no_overlap --- */

static void test_box_iou_no_overlap(void)
{
	float a[4] = {0.0f, 0.0f, 10.0f, 10.0f};
	float b[4] = {20.0f, 20.0f, 30.0f, 30.0f};

	float iou = box_iou(a, b);
	ASSERT_NEAR(iou, 0.0f, TOL);
}

/* --- test_box_iou_partial --- */

static void test_box_iou_partial(void)
{
	/*
	 * Box A: [0, 0, 10, 10]  area = 100
	 * Box B: [5, 5, 15, 15]  area = 100
	 * Intersection: [5, 5, 10, 10]  area = 25
	 * Union: 100 + 100 - 25 = 175
	 * IoU = 25 / 175 = 1/7 ~= 0.142857
	 */
	float a[4] = {0.0f, 0.0f, 10.0f, 10.0f};
	float b[4] = {5.0f, 5.0f, 15.0f, 15.0f};

	float iou = box_iou(a, b);
	ASSERT_NEAR(iou, 1.0f / 7.0f, TOL);
}

/* --- test_box_giou_enclosed --- */

static void test_box_giou_enclosed(void)
{
	/*
	 * Box A (outer): [0, 0, 20, 20]  area = 400
	 * Box B (inner): [5, 5, 15, 15]  area = 100
	 * Intersection = 100 (B is inside A)
	 * Union = 400 + 100 - 100 = 400
	 * IoU = 100 / 400 = 0.25
	 *
	 * Enclosing box C = A = [0, 0, 20, 20], area = 400
	 * GIoU = IoU - (C - union) / C = 0.25 - (400 - 400) / 400 = 0.25
	 */
	float a[4] = {0.0f, 0.0f, 20.0f, 20.0f};
	float b[4] = {5.0f, 5.0f, 15.0f, 15.0f};

	float giou = box_giou(a, b);
	ASSERT_NEAR(giou, 0.25f, TOL);

	/* GIoU should be <= IoU */
	float iou = box_iou(a, b);
	ASSERT(giou <= iou + TOL);
}

/* --- test_box_nms_basic --- */

static void test_box_nms_basic(void)
{
	/*
	 * 3 boxes: box 0 and box 1 overlap heavily, box 2 is separate.
	 * Box 1 has highest score, so it should suppress box 0.
	 * Box 2 has no overlap, so it should be kept.
	 * Expected: keep boxes 1 and 2.
	 */
	float boxes[] = {
		0.0f,  0.0f, 10.0f, 10.0f,  /* box 0 */
		1.0f,  1.0f, 11.0f, 11.0f,  /* box 1 (overlaps box 0) */
		50.0f, 50.0f, 60.0f, 60.0f, /* box 2 (separate) */
	};
	float scores[] = {0.8f, 0.9f, 0.7f};
	int keep[3];

	int n_keep = box_nms(boxes, scores, 3, 0.5f, keep);

	ASSERT_EQ(n_keep, 2);
	/* Box 1 should be first (highest score) */
	ASSERT_EQ(keep[0], 1);
	/* Box 2 should be second */
	ASSERT_EQ(keep[1], 2);
}

/* --- Main --- */

int main(void)
{
	test_box_xyxy_to_cxcywh();
	test_box_cxcywh_to_xyxy();
	test_box_roundtrip();
	test_box_iou_perfect();
	test_box_iou_no_overlap();
	test_box_iou_partial();
	test_box_giou_enclosed();
	test_box_nms_basic();

	TEST_REPORT();
}
