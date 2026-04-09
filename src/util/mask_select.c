/*
 * src/util/mask_select.c - Stability-based mask selection.
 *
 * Implements the dynamic multimask selection algorithm from the
 * Python reference (multiplex_mask_decoder.py). Computes a stability
 * score for the single-mask token (index 0) and falls back to the
 * best multimask (indices 1-3) when the score is below threshold.
 *
 * Key types:  none
 * Depends on: sam3/internal/mask_select.h
 * Used by:    src/model/sam3_processor.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "sam3/internal/mask_select.h"

int sam3_mask_select_best(const float *masks, const float *iou_scores,
			  int n_masks, int h, int w,
			  float delta, float thresh)
{
	if (!masks || !iou_scores || n_masks <= 0 || h <= 0 || w <= 0)
		return -1;

	/* Single mask: only one choice */
	if (n_masks == 1)
		return 0;

	int n_pix = h * w;
	const float *m0 = masks; /* token 0 */

	/* Compute stability score for token 0:
	 * area_i = count(logit > delta)
	 * area_u = count(logit > -delta)
	 * stability = area_u > 0 ? area_i / area_u : 1.0 */
	int area_i = 0, area_u = 0;
	for (int i = 0; i < n_pix; i++) {
		area_i += (m0[i] > delta);
		area_u += (m0[i] > -delta);
	}

	float stability = (area_u > 0)
		? (float)area_i / (float)area_u
		: 1.0f;

	if (stability >= thresh)
		return 0;

	/* Unstable: select multimask (indices 1..n_masks-1) with
	 * highest IoU score. */
	int best = 1;
	float best_score = iou_scores[1];
	for (int i = 2; i < n_masks; i++) {
		if (iou_scores[i] > best_score) {
			best_score = iou_scores[i];
			best = i;
		}
	}

	return best;
}
