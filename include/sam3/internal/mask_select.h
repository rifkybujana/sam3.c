/*
 * include/sam3/internal/mask_select.h - Stability-based mask selection.
 *
 * Selects the best mask from the SAM mask decoder's 4-mask output using
 * the stability score algorithm from the Python reference. Token 0 is
 * the single-mask output; tokens 1-3 are multimask outputs. If token 0
 * is stable, it is selected; otherwise the multimask with the highest
 * IoU prediction is chosen.
 *
 * Key types:  none
 * Depends on: (none)
 * Used by:    src/model/sam3_processor.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */
#ifndef SAM3_INTERNAL_MASK_SELECT_H
#define SAM3_INTERNAL_MASK_SELECT_H

/* Default stability parameters from Python reference */
#define SAM3_STABILITY_DELTA  0.05f
#define SAM3_STABILITY_THRESH 0.98f

/*
 * sam3_mask_select_best - Select best mask via stability score.
 *
 * Algorithm (from multiplex_mask_decoder.py):
 *   1. Compute stability of token 0: count(logit > delta) / count(logit > -delta)
 *   2. If stability >= thresh, return 0
 *   3. Else return index 1-3 with highest IoU score
 *
 * @masks:      [n_masks * h * w] flat mask logits (float)
 * @iou_scores: [n_masks] per-mask IoU predictions (post-sigmoid)
 * @n_masks:    Number of masks (expected 4 for SAM decoder)
 * @h, @w:      Mask spatial dimensions
 * @delta:      Stability threshold delta (use SAM3_STABILITY_DELTA)
 * @thresh:     Stability threshold (use SAM3_STABILITY_THRESH)
 *
 * Returns the index of the best mask [0, n_masks), or -1 on error.
 */
int sam3_mask_select_best(const float *masks, const float *iou_scores,
			  int n_masks, int h, int w,
			  float delta, float thresh);

#endif /* SAM3_INTERNAL_MASK_SELECT_H */
