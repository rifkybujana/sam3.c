/*
 * src/model/box_ops.h - Bounding box operations for SAM3 mask decoder
 *
 * Box format conversions (xyxy, cxcywh, xywh), Intersection over Union,
 * Generalized IoU, and Non-Maximum Suppression. These are pure math
 * functions with no project dependencies, used by the mask decoder and
 * any post-processing stage that operates on bounding boxes.
 *
 * Key types:  (none, operates on raw float arrays)
 * Depends on: (standard headers only)
 * Used by:    decoder.c, tests/test_box_ops.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_BOX_OPS_H
#define SAM3_MODEL_BOX_OPS_H

/*
 * box_xyxy_to_cxcywh - Convert boxes from [x1,y1,x2,y2] to [cx,cy,w,h].
 *
 * @xyxy:  Input array of n*4 floats in [x1, y1, x2, y2] format
 * @cxcywh: Output array of n*4 floats in [cx, cy, w, h] format
 * @n:     Number of boxes
 */
void box_xyxy_to_cxcywh(const float *xyxy, float *cxcywh, int n);

/*
 * box_cxcywh_to_xyxy - Convert boxes from [cx,cy,w,h] to [x1,y1,x2,y2].
 *
 * @cxcywh: Input array of n*4 floats in [cx, cy, w, h] format
 * @xyxy:   Output array of n*4 floats in [x1, y1, x2, y2] format
 * @n:      Number of boxes
 */
void box_cxcywh_to_xyxy(const float *cxcywh, float *xyxy, int n);

/*
 * box_xyxy_to_xywh - Convert boxes from [x1,y1,x2,y2] to [x,y,w,h].
 *
 * @xyxy: Input array of n*4 floats in [x1, y1, x2, y2] format
 * @xywh: Output array of n*4 floats in [x, y, w, h] format
 * @n:    Number of boxes
 */
void box_xyxy_to_xywh(const float *xyxy, float *xywh, int n);

/*
 * box_iou - Intersection over Union for a single pair of boxes.
 *
 * @a: First box, 4 floats in [x1, y1, x2, y2] format
 * @b: Second box, 4 floats in [x1, y1, x2, y2] format
 *
 * Returns IoU in [0, 1]. Returns 0 for zero-area boxes.
 */
float box_iou(const float *a, const float *b);

/*
 * box_giou - Generalized Intersection over Union for a single pair.
 *
 * GIoU = IoU - (C - union) / C, where C is the area of the smallest
 * enclosing box. Returns a value in [-1, 1].
 *
 * @a: First box, 4 floats in [x1, y1, x2, y2] format
 * @b: Second box, 4 floats in [x1, y1, x2, y2] format
 *
 * Returns GIoU in [-1, 1]. Returns 0 for zero-area enclosing box.
 */
float box_giou(const float *a, const float *b);

/*
 * box_nms - Greedy Non-Maximum Suppression.
 *
 * Sorts boxes by score descending, then iteratively keeps the highest
 * scoring box and suppresses all remaining boxes with IoU > iou_threshold.
 *
 * @boxes:         Array of n*4 floats in [x1, y1, x2, y2] format
 * @scores:        Array of n scores
 * @n:             Number of boxes
 * @iou_threshold: Suppress boxes with IoU above this threshold
 * @keep:          Output array of kept indices (caller-allocated, size n)
 *
 * Returns the number of kept boxes.
 */
int box_nms(const float *boxes, const float *scores, int n,
	    float iou_threshold, int *keep);

#endif /* SAM3_MODEL_BOX_OPS_H */
