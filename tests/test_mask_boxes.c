/*
 * tests/test_mask_boxes.c - Unit tests for bounding box extraction.
 *
 * Verifies box extraction from mask logits using hand-constructed
 * masks with known bounding boxes.
 *
 * Key types:  none
 * Depends on: sam3/internal/mask_boxes.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <string.h>
#include "test_helpers.h"
#include "sam3/internal/mask_boxes.h"

static void fill_rect(float *m, int x0, int y0, int x1, int y1)
{
	for (int y = 0; y < 4; y++)
		for (int x = 0; x < 4; x++)
			m[y * 4 + x] = (x >= x0 && x < x1 &&
					 y >= y0 && y < y1)
					? 1.0f : -1.0f;
}

static void test_boxes_full_mask(void)
{
	float masks[16];
	float boxes[4];
	fill_rect(masks, 0, 0, 4, 4);

	int rc = sam3_masks_to_boxes(masks, 1, 4, 4, boxes);
	ASSERT_EQ(rc, 0);
	ASSERT_NEAR(boxes[0], 0.0f, 1e-5f);
	ASSERT_NEAR(boxes[1], 0.0f, 1e-5f);
	ASSERT_NEAR(boxes[2], 4.0f, 1e-5f);
	ASSERT_NEAR(boxes[3], 4.0f, 1e-5f);
}

static void test_boxes_corner_mask(void)
{
	float masks[16];
	float boxes[4];
	fill_rect(masks, 0, 0, 2, 2);

	int rc = sam3_masks_to_boxes(masks, 1, 4, 4, boxes);
	ASSERT_EQ(rc, 0);
	ASSERT_NEAR(boxes[0], 0.0f, 1e-5f);
	ASSERT_NEAR(boxes[1], 0.0f, 1e-5f);
	ASSERT_NEAR(boxes[2], 2.0f, 1e-5f);
	ASSERT_NEAR(boxes[3], 2.0f, 1e-5f);
}

static void test_boxes_center_mask(void)
{
	float masks[16];
	float boxes[4];
	fill_rect(masks, 1, 1, 3, 3);

	int rc = sam3_masks_to_boxes(masks, 1, 4, 4, boxes);
	ASSERT_EQ(rc, 0);
	ASSERT_NEAR(boxes[0], 1.0f, 1e-5f);
	ASSERT_NEAR(boxes[1], 1.0f, 1e-5f);
	ASSERT_NEAR(boxes[2], 3.0f, 1e-5f);
	ASSERT_NEAR(boxes[3], 3.0f, 1e-5f);
}

static void test_boxes_empty_mask(void)
{
	float masks[16];
	float boxes[4];
	for (int i = 0; i < 16; i++)
		masks[i] = -1.0f;

	int rc = sam3_masks_to_boxes(masks, 1, 4, 4, boxes);
	ASSERT_EQ(rc, 0);
	ASSERT_NEAR(boxes[0], 0.0f, 1e-5f);
	ASSERT_NEAR(boxes[1], 0.0f, 1e-5f);
	ASSERT_NEAR(boxes[2], 0.0f, 1e-5f);
	ASSERT_NEAR(boxes[3], 0.0f, 1e-5f);
}

static void test_boxes_single_pixel(void)
{
	float masks[16];
	for (int i = 0; i < 16; i++)
		masks[i] = -1.0f;
	masks[1 * 4 + 2] = 1.0f;

	float boxes[4];
	int rc = sam3_masks_to_boxes(masks, 1, 4, 4, boxes);
	ASSERT_EQ(rc, 0);
	ASSERT_NEAR(boxes[0], 2.0f, 1e-5f);
	ASSERT_NEAR(boxes[1], 1.0f, 1e-5f);
	ASSERT_NEAR(boxes[2], 3.0f, 1e-5f);
	ASSERT_NEAR(boxes[3], 2.0f, 1e-5f);
}

static void test_boxes_multiple_masks(void)
{
	float masks[2 * 16];
	float boxes[2 * 4];

	fill_rect(&masks[0 * 16], 0, 0, 4, 4);
	fill_rect(&masks[1 * 16], 0, 0, 2, 2);

	int rc = sam3_masks_to_boxes(masks, 2, 4, 4, boxes);
	ASSERT_EQ(rc, 0);

	ASSERT_NEAR(boxes[0], 0.0f, 1e-5f);
	ASSERT_NEAR(boxes[1], 0.0f, 1e-5f);
	ASSERT_NEAR(boxes[2], 4.0f, 1e-5f);
	ASSERT_NEAR(boxes[3], 4.0f, 1e-5f);

	ASSERT_NEAR(boxes[4], 0.0f, 1e-5f);
	ASSERT_NEAR(boxes[5], 0.0f, 1e-5f);
	ASSERT_NEAR(boxes[6], 2.0f, 1e-5f);
	ASSERT_NEAR(boxes[7], 2.0f, 1e-5f);
}

static void test_boxes_invalid_args(void)
{
	float masks[16], boxes[4];
	ASSERT_EQ(sam3_masks_to_boxes(NULL, 1, 4, 4, boxes), -1);
	ASSERT_EQ(sam3_masks_to_boxes(masks, 1, 4, 4, NULL), -1);
	ASSERT_EQ(sam3_masks_to_boxes(masks, 0, 4, 4, boxes), -1);
	ASSERT_EQ(sam3_masks_to_boxes(masks, 1, 0, 4, boxes), -1);
	ASSERT_EQ(sam3_masks_to_boxes(masks, 1, 4, 0, boxes), -1);
}

int main(void)
{
	test_boxes_full_mask();
	test_boxes_corner_mask();
	test_boxes_center_mask();
	test_boxes_empty_mask();
	test_boxes_single_pixel();
	test_boxes_multiple_masks();
	test_boxes_invalid_args();

	TEST_REPORT();
}
