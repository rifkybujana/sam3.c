/*
 * src/util/mask_nms.c - Greedy mask NMS post-processing.
 *
 * Implements sam3_mask_nms: prefilter by score, binarize masks at
 * logit > 0, pairwise IoU, greedy keep-highest-that-does-not-overlap.
 * No allocations — caller provides output buffer. Uses a stack-allocated
 * work array for candidate sort order. Intended for n_masks <= 512
 * (SAM3 uses 200 queries).
 *
 * Key types:  none
 * Depends on: sam3/internal/mask_nms.h, <stdlib.h>
 * Used by:    tools/sam3_main.c (post-processing), tests/test_mask_nms.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include "sam3/internal/mask_nms.h"

/* Compute IoU between two binarized masks (logit > 0). */
static float mask_pair_iou(const float *a, const float *b, int n_pix)
{
	int inter = 0, uni = 0;
	int i;

	for (i = 0; i < n_pix; i++) {
		int ai = (a[i] > 0.0f);
		int bi = (b[i] > 0.0f);
		inter += ai & bi;
		uni   += ai | bi;
	}
	if (uni == 0)
		return 0.0f;
	return (float)inter / (float)uni;
}

/* Comparator for descending-score sort on (score, index) pairs. */
struct idx_score { int idx; float score; };
static int cmp_idx_score_desc(const void *pa, const void *pb)
{
	const struct idx_score *a = (const struct idx_score *)pa;
	const struct idx_score *b = (const struct idx_score *)pb;
	if (a->score < b->score) return  1;
	if (a->score > b->score) return -1;
	return a->idx - b->idx;
}

int sam3_mask_nms(const float *masks, const float *scores,
		  int n_masks, int h, int w,
		  float prob_thresh, float iou_thresh,
		  int *kept_out)
{
	struct idx_score cand[512];
	int n_cand = 0;
	int n_kept = 0;
	int i, j;
	int n_pix;

	if (!masks || !scores || !kept_out || n_masks <= 0 ||
	    n_masks > 512 || h <= 0 || w <= 0)
		return -1;

	n_pix = h * w;

	/* Prefilter by score, build candidate list. */
	for (i = 0; i < n_masks; i++) {
		if (scores[i] > prob_thresh) {
			cand[n_cand].idx = i;
			cand[n_cand].score = scores[i];
			n_cand++;
		}
	}

	/* Sort candidates by score descending. */
	qsort(cand, (size_t)n_cand, sizeof(cand[0]),
	      cmp_idx_score_desc);

	/* Greedy: for each candidate in score order, keep it if
	 * its IoU with every already-kept mask is <= iou_thresh. */
	for (i = 0; i < n_cand; i++) {
		int ci = cand[i].idx;
		const float *mi = masks + (size_t)ci * n_pix;
		int drop = 0;

		for (j = 0; j < n_kept; j++) {
			int kj = kept_out[j];
			const float *mj = masks + (size_t)kj * n_pix;
			float iou = mask_pair_iou(mi, mj, n_pix);
			if (iou > iou_thresh) {
				drop = 1;
				break;
			}
		}
		if (!drop) {
			kept_out[n_kept] = ci;
			n_kept++;
		}
	}

	return n_kept;
}
