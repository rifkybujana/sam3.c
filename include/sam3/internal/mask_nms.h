/*
 * include/sam3/internal/mask_nms.h - Greedy mask NMS helper.
 *
 * Internal header exposing a greedy non-maximum-suppression routine over
 * mask logits. Prefilters by per-mask score, binarizes masks at logit > 0,
 * computes pairwise IoU, and greedily keeps highest-score masks that do
 * not overlap previously-kept masks above an IoU threshold. Matches the
 * Python reference `nms_masks_impl` / `generic_nms_cpu`.
 *
 * Key types:  none
 * Depends on: (none)
 * Used by:    tools/sam3_main.c, tests/test_mask_nms.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */
#ifndef SAM3_INTERNAL_MASK_NMS_H
#define SAM3_INTERNAL_MASK_NMS_H

/*
 * sam3_mask_nms - Greedy non-maximum suppression on mask logits.
 *
 * @masks:       [n_masks * h * w] flat mask logits (float)
 * @scores:      [n_masks] per-mask score (e.g. sigmoid(dot_score))
 * @n_masks:     Number of input masks (must be in [1, 512])
 * @h, @w:       Mask spatial dimensions
 * @prob_thresh: Drop any mask with score <= prob_thresh before NMS
 * @iou_thresh:  Drop any mask whose IoU with a kept mask > iou_thresh
 * @min_quality: Minimum fraction of confident pixels (|logit| > 2.0) per mask.
 *               Masks below this threshold are rejected after NMS.
 *               Set to 0.0 to disable quality filtering (backward compatible).
 * @kept_out:    Caller-allocated int array of size >= n_masks; filled
 *               with kept mask indices in descending-score order.
 *
 * A mask is binarized with logit > 0 (matches Python
 * `pred_masks > 0`). IoU is over binary masks. Returns the number of
 * kept indices written to @kept_out, or -1 on invalid arguments
 * (including n_masks > 512).
 */
int sam3_mask_nms(const float *masks, const float *scores,
		  int n_masks, int h, int w,
		  float prob_thresh, float iou_thresh,
		  float min_quality,
		  int *kept_out);

#endif /* SAM3_INTERNAL_MASK_NMS_H */
