/*
 * src/model/memory_bank.h - Frame memory ring buffer for video tracking
 *
 * Stores spatial features and object pointers from past frames.
 * Maintains separate storage for conditioning frames (user-annotated)
 * and non-conditioning frames (propagated). Implements SAM3-Long-style
 * memory selection that filters low-confidence frames.
 *
 * Key types:  sam3_memory_entry, sam3_memory_bank
 * Depends on: core/tensor.h, sam3/sam3_types.h
 * Used by:    model/tracker.h
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_MEMORY_BANK_H
#define SAM3_MODEL_MEMORY_BANK_H

#include "core/tensor.h"
#include "sam3/sam3_types.h"

struct sam3_memory_entry {
	struct sam3_tensor *spatial_features; /* [HW, mem_dim] maskmem output */
	/*
	 * obj_pointer: SAM 3 stores [1, 256]; SAM 3.1 multiplex stores
	 * [multiplex_count=16, 256] (one row per multiplex slot, per-slot
	 * argmax over 3 multimask heads and then obj_ptr_proj applied).
	 * Consumers read dims[0] as the row count.
	 */
	struct sam3_tensor *obj_pointer;
	/*
	 * image_features: raw 1x backbone features for this frame, flattened
	 * to [HW, 256] NHWC. Only populated in SAM 3.1 multiplex paths (the
	 * decoupled memory attention consumes image-side memory separately
	 * from maskmem). NULL on SAM 3 (SAM 3 memory attention is single-
	 * source, doesn't need this).
	 *
	 * Python analog: the `image_features` entry saved per frame when
	 * `save_image_features=True` (SAM 3.1 config).
	 */
	struct sam3_tensor *image_features;
	int    frame_idx;
	int    is_conditioning;
	float  obj_score; /* max object score for SAM3-Long selection */
};

struct sam3_memory_bank {
	/*
	 * Conditioning frames (user-annotated): stored in a separate array
	 * and never evicted. Up to SAM3_MAX_MEMORY_FRAMES entries may be
	 * kept; the `max_cond_frames_in_attn` cap applies at attention time
	 * (see sam3_memory_bank_build_view), matching Python's
	 * max_cond_frames_in_attn semantics.
	 */
	struct sam3_memory_entry cond[SAM3_MAX_MEMORY_FRAMES];
	int    n_cond;
	int    max_cond_frames_in_attn;

	/* Non-conditioning frames: ring buffer, oldest evicted when full */
	struct sam3_memory_entry non_cond[SAM3_MAX_MEMORY_FRAMES];
	int    n_non_cond;

	int    capacity;         /* non-cond ring buffer size (= num_maskmem) */
	int    temporal_stride;  /* stride for non-cond frame selection */
	float  mf_threshold;    /* SAM3-Long: reject frames below this score */
};

/*
 * sam3_memory_bank_view - Per-frame selection of which cond entries are
 *                         active in the memory attention.
 *
 * Non-conditioning frames are always iterated in full storage order
 * (oldest -> newest). Conditioning frames are filtered down to the
 * `max_cond_frames_in_attn` closest to the current frame index and the
 * resulting indices are sorted ascending so the concat/tpos row order
 * stays deterministic.
 */
struct sam3_memory_bank_view {
	const struct sam3_memory_bank *bank;
	int    cond_idx[SAM3_MAX_MEMORY_FRAMES];
	int    n_cond;
};

/*
 * sam3_memory_bank_init - Initialize a memory bank with given parameters.
 *
 * @bank:                   Memory bank to initialize (zeroed and configured)
 * @capacity:               Non-cond ring buffer size (= num_maskmem)
 * @max_cond_frames_in_attn: Cap on cond entries admitted per frame into
 *                           the memory attention (Python's
 *                           max_cond_frames_in_attn; typically 4).
 *                           Cond storage itself is only capped at
 *                           SAM3_MAX_MEMORY_FRAMES.
 * @temporal_stride:        Stride for non-conditioning frame selection
 * @mf_threshold:           Minimum obj_score to accept a non-cond frame
 *
 * Zeroes the bank and sets configuration. Must be called before any
 * other memory bank operation.
 */
void sam3_memory_bank_init(struct sam3_memory_bank *bank,
			   int capacity, int max_cond_frames_in_attn,
			   int temporal_stride, float mf_threshold);

/*
 * sam3_memory_bank_add - Insert a frame entry into the memory bank.
 *
 * @bank:  Initialized memory bank
 * @entry: Entry to insert (copied into the bank)
 *
 * If entry->is_conditioning is set, appends to the conditioning array
 * up to SAM3_MAX_MEMORY_FRAMES total cond entries (silently dropped if
 * that array is full). Matches Python semantics: all cond frames are
 * stored; the `max_cond_frames_in_attn` selection happens at attention
 * time via sam3_memory_bank_build_view.
 *
 * Non-conditioning entries apply SAM3-Long selection: rejects entries
 * with obj_score < mf_threshold. When the ring buffer is full, evicts
 * the oldest entry.
 */
void sam3_memory_bank_add(struct sam3_memory_bank *bank,
			  const struct sam3_memory_entry *entry);

/*
 * sam3_memory_bank_build_view - Build a per-frame selection of which
 *                               cond entries participate in attention.
 *
 * @bank:      Populated memory bank
 * @frame_idx: Frame being tracked (used to score cond-frame distance)
 * @view:      Output view (caller-allocated). view->bank is set to
 *             bank; view->cond_idx[] lists indices into bank->cond for
 *             up to bank->max_cond_frames_in_attn closest cond frames,
 *             sorted ascending; view->n_cond is the count.
 *
 * Non-cond entries are not filtered here; downstream helpers iterate
 * bank->non_cond in storage order.
 */
void sam3_memory_bank_build_view(const struct sam3_memory_bank *bank,
				 int frame_idx,
				 struct sam3_memory_bank_view *view);

/*
 * sam3_memory_bank_select_closest_cond - Find conditioning frames nearest
 *                                        to a given frame index.
 *
 * @bank:        Memory bank with conditioning frames
 * @frame_idx:   Target frame index to measure distance from
 * @out_indices: Output array of conditioning-array indices (not frame_idx)
 * @max_n:       Maximum number of indices to return
 *
 * Sorts conditioning entries by absolute distance to @frame_idx and
 * writes up to @max_n closest indices into @out_indices.
 * Returns the number of indices written (min of n_cond and max_n).
 */
int sam3_memory_bank_select_closest_cond(
	const struct sam3_memory_bank *bank,
	int frame_idx, int *out_indices, int max_n);

/*
 * sam3_memory_bank_clear - Reset the memory bank to empty state.
 *
 * @bank: Memory bank to clear.
 *
 * Sets both conditioning and non-conditioning counts to zero.
 * Configuration (capacity, thresholds) is preserved.
 */
void sam3_memory_bank_clear(struct sam3_memory_bank *bank);

/*
 * sam3_memory_bank_total - Return the total number of stored entries.
 *
 * @bank: Memory bank to query.
 *
 * Returns n_cond + n_non_cond.
 */
int sam3_memory_bank_total(const struct sam3_memory_bank *bank);

/*
 * sam3_memory_bank_clear_around_frame - Drop non-cond entries near a frame.
 *
 * @bank:   Memory bank to modify.
 * @frame:  Center frame index of the window to clear.
 * @window: Inclusive radius. Non-cond entries with
 *          |entry.frame_idx - frame| <= window are removed.
 *
 * Conditioning entries are not affected. Removal preserves the order
 * of surviving non-cond entries (stable compaction).
 *
 * Mirrors Python clear_non_cond_mem_around_input semantics: when a new
 * conditioning prompt arrives on a previously-tracked frame, the
 * propagated non-cond entries within the memory window become stale
 * and must be discarded so they do not pollute the re-decode.
 */
void sam3_memory_bank_clear_around_frame(struct sam3_memory_bank *bank,
					 int frame, int window);

/*
 * sam3_memory_bank_select_non_cond_for_frame - Pick non-cond entries
 *                                              participating in attn.
 *
 * @bank:           Memory bank.
 * @current_frame:  Frame currently being tracked.
 * @num_maskmem:    Total memory bank slots (Python's num_maskmem=7).
 * @out_indices:    Output: indices into bank->non_cond[].
 * @max_n:          Capacity of out_indices.
 *
 * Returns the number of indices written (≤ num_maskmem-1, ≤ max_n).
 *
 * Mirrors Python: target_frames = {current_frame - temporal_stride * t
 * for t in 1..num_maskmem-1}. For each target, selects the closest
 * available non-cond entry with frame_idx < current_frame. De-duplicates
 * if the same entry is selected for multiple targets (e.g., when
 * temporal_stride > number of stored frames).
 *
 * Indices are returned in newest-to-oldest order, matching Python's
 * tpos enumeration.
 */
int sam3_memory_bank_select_non_cond_for_frame(
	const struct sam3_memory_bank *bank,
	int current_frame, int num_maskmem,
	int *out_indices, int max_n);

#endif /* SAM3_MODEL_MEMORY_BANK_H */
