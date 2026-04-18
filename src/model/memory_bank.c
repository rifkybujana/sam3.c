/*
 * src/model/memory_bank.c - Frame memory ring buffer for video tracking
 *
 * Implements the memory bank data structure that stores past frame
 * features for video object tracking. Conditioning frames (user-annotated)
 * are stored in a separate array and never evicted. Non-conditioning
 * frames (propagated) use FIFO eviction. SAM3-Long memory selection
 * rejects low-confidence non-conditioning frames.
 *
 * Key types:  sam3_memory_entry, sam3_memory_bank
 * Depends on: model/memory_bank.h, util/log.h
 * Used by:    model/tracker.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <string.h>
#include <stdlib.h>

#include "model/memory_bank.h"
#include "util/log.h"

void sam3_memory_bank_init(struct sam3_memory_bank *bank,
			   int capacity, int max_cond_frames_in_attn,
			   int temporal_stride, float mf_threshold)
{
	memset(bank, 0, sizeof(*bank));
	bank->capacity = capacity;
	bank->max_cond_frames_in_attn = max_cond_frames_in_attn;
	bank->temporal_stride = temporal_stride;
	bank->mf_threshold = mf_threshold;
}

void sam3_memory_bank_add(struct sam3_memory_bank *bank,
			  const struct sam3_memory_entry *entry)
{
	if (entry->is_conditioning) {
		if (bank->n_cond >= SAM3_MAX_MEMORY_FRAMES) {
			sam3_log_warn("conditioning bank at array limit (%d), "
				      "dropping frame %d",
				      SAM3_MAX_MEMORY_FRAMES,
				      entry->frame_idx);
			return;
		}
		bank->cond[bank->n_cond++] = *entry;
		return;
	}

	/* Non-conditioning: SAM3-Long selection filter */
	if (entry->obj_score < bank->mf_threshold) {
		sam3_log_debug("rejecting frame %d: obj_score %.4f < threshold %.4f",
			       entry->frame_idx, entry->obj_score,
			       bank->mf_threshold);
		return;
	}

	/* Max non-cond slots: capacity - 1 (reserve at least 1 for cond),
	 * capped at SAM3_MAX_MEMORY_FRAMES */
	int max_non_cond = bank->capacity - 1;
	if (max_non_cond > SAM3_MAX_MEMORY_FRAMES)
		max_non_cond = SAM3_MAX_MEMORY_FRAMES;
	if (max_non_cond < 1)
		max_non_cond = 1;

	/* Evict oldest if full */
	if (bank->n_non_cond >= max_non_cond) {
		memmove(&bank->non_cond[0], &bank->non_cond[1],
			(size_t)(bank->n_non_cond - 1) * sizeof(bank->non_cond[0]));
		bank->n_non_cond--;
	}

	bank->non_cond[bank->n_non_cond++] = *entry;
}

struct dist_pair {
	int idx;
	int dist;
};

static int dist_pair_cmp(const void *a, const void *b)
{
	const struct dist_pair *pa = a;
	const struct dist_pair *pb = b;
	return (pa->dist > pb->dist) - (pa->dist < pb->dist);
}

int sam3_memory_bank_select_closest_cond(
	const struct sam3_memory_bank *bank,
	int frame_idx, int *out_indices, int max_n)
{
	if (bank->n_cond == 0 || max_n <= 0)
		return 0;

	struct dist_pair pairs[SAM3_MAX_MEMORY_FRAMES];
	int n = bank->n_cond;

	for (int i = 0; i < n; i++) {
		pairs[i].idx = i;
		int d = bank->cond[i].frame_idx - frame_idx;
		pairs[i].dist = d < 0 ? -d : d;
	}

	qsort(pairs, (size_t)n, sizeof(pairs[0]), dist_pair_cmp);

	int count = n < max_n ? n : max_n;
	for (int i = 0; i < count; i++)
		out_indices[i] = pairs[i].idx;

	return count;
}

void sam3_memory_bank_clear(struct sam3_memory_bank *bank)
{
	bank->n_cond = 0;
	bank->n_non_cond = 0;
}

int sam3_memory_bank_total(const struct sam3_memory_bank *bank)
{
	return bank->n_cond + bank->n_non_cond;
}

static int int_cmp(const void *a, const void *b)
{
	int ia = *(const int *)a;
	int ib = *(const int *)b;
	return (ia > ib) - (ia < ib);
}

void sam3_memory_bank_build_view(const struct sam3_memory_bank *bank,
				 int frame_idx,
				 struct sam3_memory_bank_view *view)
{
	memset(view, 0, sizeof(*view));
	view->bank = bank;

	if (!bank || bank->n_cond == 0)
		return;

	int max_n = bank->max_cond_frames_in_attn;
	if (max_n <= 0 || max_n > SAM3_MAX_MEMORY_FRAMES)
		max_n = SAM3_MAX_MEMORY_FRAMES;

	int selected[SAM3_MAX_MEMORY_FRAMES];
	int n = sam3_memory_bank_select_closest_cond(
		bank, frame_idx, selected, max_n);

	/*
	 * Sort the selected cond indices ascending so downstream
	 * iteration over view->cond_idx[] yields a deterministic
	 * oldest-first row order, matching the pre-view concat layout.
	 */
	qsort(selected, (size_t)n, sizeof(selected[0]), int_cmp);

	for (int i = 0; i < n; i++)
		view->cond_idx[i] = selected[i];
	view->n_cond = n;
}

void sam3_memory_bank_clear_around_frame(struct sam3_memory_bank *bank,
					 int frame, int window)
{
	if (!bank || window < 0)
		return;

	int write = 0;
	for (int read = 0; read < bank->n_non_cond; read++) {
		int d = bank->non_cond[read].frame_idx - frame;
		int abs_d = d < 0 ? -d : d;
		if (abs_d <= window)
			continue;
		if (write != read)
			bank->non_cond[write] = bank->non_cond[read];
		write++;
	}
	bank->n_non_cond = write;
}

int sam3_memory_bank_select_non_cond_for_frame(
	const struct sam3_memory_bank *bank,
	int current_frame, int num_maskmem,
	int *out_indices, int max_n)
{
	if (!bank || bank->n_non_cond == 0 || num_maskmem <= 1 || max_n <= 0)
		return 0;

	int stride = bank->temporal_stride > 0 ? bank->temporal_stride : 1;
	int n_pick = num_maskmem - 1;
	if (n_pick > max_n)
		n_pick = max_n;

	int written = 0;
	for (int t = 1; t <= n_pick; t++) {
		int target = current_frame - stride * t;

		/* Find non-cond entry with frame_idx closest to target.
		 * Only entries with frame_idx < current_frame are eligible
		 * (future frames not yet tracked). */
		int best_idx = -1;
		int best_dist = 0;
		for (int i = 0; i < bank->n_non_cond; i++) {
			int f = bank->non_cond[i].frame_idx;
			if (f >= current_frame)
				continue;
			int d = f - target;
			int abs_d = d < 0 ? -d : d;
			if (best_idx < 0 || abs_d < best_dist) {
				best_idx = i;
				best_dist = abs_d;
			}
		}
		if (best_idx < 0)
			break; /* no eligible past entries */

		/* Skip if this entry is already selected for an earlier t.
		 * Matches Python: when the same frame is the closest match
		 * for multiple targets, it appears in memory once. */
		int dup = 0;
		for (int j = 0; j < written; j++) {
			if (out_indices[j] == best_idx) {
				dup = 1;
				break;
			}
		}
		if (!dup)
			out_indices[written++] = best_idx;
	}
	return written;
}
