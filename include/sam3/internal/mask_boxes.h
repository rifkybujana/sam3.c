/*
 * include/sam3/internal/mask_boxes.h - Bounding box extraction from masks.
 *
 * Derives axis-aligned bounding boxes from mask logits. Matches the
 * Python reference box_ops.py:masks_to_boxes() — output is xyxy format
 * with exclusive upper bounds (x_max+1, y_max+1).
 *
 * Key types:  none
 * Depends on: (none)
 * Used by:    src/model/sam3_processor.c, tools/sam3_main.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */
#ifndef SAM3_INTERNAL_MASK_BOXES_H
#define SAM3_INTERNAL_MASK_BOXES_H

/*
 * sam3_masks_to_boxes - Extract bounding boxes from mask logits.
 *
 * For each mask, finds the bounding box of pixels where logit > 0.
 * Output format: [x_min, y_min, x_max+1, y_max+1] (xyxy, exclusive
 * upper bound, matching PyTorch convention).
 * Empty masks produce [0, 0, 0, 0].
 *
 * @masks:     [n_masks * h * w] flat mask logits (float)
 * @n_masks:   Number of masks
 * @h, @w:     Mask spatial dimensions
 * @boxes_out: Output buffer [n_masks * 4] floats (caller-allocated)
 *
 * Returns 0 on success, -1 on invalid arguments.
 */
int sam3_masks_to_boxes(const float *masks, int n_masks,
			int h, int w, float *boxes_out);

#endif /* SAM3_INTERNAL_MASK_BOXES_H */
