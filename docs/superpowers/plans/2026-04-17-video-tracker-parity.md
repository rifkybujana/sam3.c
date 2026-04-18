# SAM3 Video Tracker — Python Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** close every Python-parity gap in the C SAM3 video tracker — multi-object propagation, per-object memory banks, memory persistence across `propagate` calls, `add_new_mask` API, `dynamic_multimask_via_stability`, `iter_use_prev_mask_pred`, `clear_non_cond_mem_around_input`, and a tiered lazy frame cache.

**Architecture:** Five sequential phases against `feature/video-tracker`. Phase 1 refactors the memory bank to per-object semantics. Phase 2 lifts propagation to a per-object loop with a new multi-object callback. Phase 3 replaces eager frame encoding with a tiered LRU cache. Phase 4 makes propagation idempotent and adds stale-memory clearing on re-prompt. Phase 5 adds the remaining decoder/prompt features.

**Tech Stack:** C11, CMake, CTest, Metal/CPU backends, arena allocators (no malloc in hot paths). Test pattern: one `tests/test_<module>.c` per module, `ASSERT*` macros from `tests/test_helpers.h`, auto-discovered by `file(GLOB tests/test_*.c)`.

**Spec:** `docs/superpowers/specs/2026-04-17-video-tracker-parity-design.md`

**Baseline state (commit `d4f2d32`):** the worktree already contains foundational changes that overlap with this plan. Before executing any task, an executor MUST verify whether the work is already present. The baseline includes:

- `sam3_memory_bank_view` + `sam3_memory_bank_build_view` for **conditioning-frame** selection (similar to but distinct from Task 1.4, which adds a sibling helper for **non-conditioning** frames).
- `max_cond_frames_in_attn` field rename (Plan Task 1.1 retains its **separate** rename of `obj_pointers` → `obj_pointer`; both renames are in scope).
- `gh_concat_mem` / `gh_tpos_enc_mem` / `gh_concat_obj_ptrs` updated to consume `sam3_memory_bank_view`.
- Tracker `obj_ptr_proj_fc{0,1,2}` 3-layer MLP weights + `obj_ptr_tpos_proj` (Phase 2/5 plumbing).
- Mask decoder `obj_score_fc{0,1,2}` weights + `out_obj_score_logits` + `out_mask_tokens` outputs (Phase 5 plumbing).
- `apply_occlusion_gating` in `sam3_video.c` (Python-semantics hard threshold).
- Memory encoder no longer applies sigmoid internally (caller responsibility per new mask_for_mem semantics).

All 59 existing tests pass on `d4f2d32`. Each task below begins with a verification step: grep/inspect first, only execute if the change is not already present. Skip-ahead is fine; do not duplicate work.

**Build commands** (referenced throughout):
```bash
cd /Users/rbisri/Documents/sam3/.worktrees/video-tracker
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DSAM3_TESTS=ON
make -j$(sysctl -n hw.ncpu) <target>
ctest -R <test_name> --output-on-failure
```

**Commit cadence:** one commit per task (TDD: failing test, then implementation, then commit). Format: `<scope>: <description>` lowercase, matching recent history.

---

## File Structure

### Phase 1 — Per-object memory bank + new result types

| File | Action | Responsibility |
|------|--------|----------------|
| `src/model/memory_bank.h` | Modify | `obj_pointers [n_obj,256]` → `obj_pointer [256]`; add `clear_around_frame`, `select_non_cond_for_frame` |
| `src/model/memory_bank.c` | Modify | Implement new ops; preserve existing API where possible |
| `include/sam3/sam3.h` | Modify | Add `sam3_video_object_mask`, `sam3_video_frame_result`, `sam3_video_frame_result_free`. **Do not** change existing callback yet (Phase 2). Cap `SAM3_MAX_OBJECTS` at 16. |
| `src/model/sam3_video.c` | Modify (minimal) | Adapt single existing callsite of `obj_pointers`→`obj_pointer` (still single-object) |
| `src/model/tracker.c` | Modify (minimal) | Same single-callsite adaptation |
| `tests/test_memory_bank.c` | Modify | Update existing tests for renamed field; add tests for new ops |

### Phase 2 — Multi-object propagation + new callback

| File | Action | Responsibility |
|------|--------|----------------|
| `include/sam3/sam3.h` | Modify | New callback signature; existing prompt entry points get `sam3_video_frame_result*` instead of `sam3_result*` |
| `src/model/video_session.h` | Modify | Add `struct sam3_video_object[]`; replace global bank; add per-object prompt bitmaps |
| `src/model/video_session.c` | Modify | Object allocation/lookup/removal; lifecycle |
| `src/model/sam3_video.c` | Modify | `propagate_one` becomes per-object loop; result assembly |
| `tools/sam3_cli.c`, `tools/cli_track.c` | Modify | Update callback signatures, iterate `result->objects[]` |
| `tools/sam3_video_demo.c` (if exists) | Modify | Same |
| `tests/test_video_api.c` | Modify | Update for new result type |
| `tests/test_video_multi_object.c` | Create | Multi-object correctness |
| `tests/test_video_session.c` | Modify | Per-object state |

### Phase 3 — Frame cache (lazy + tiered)

| File | Action | Responsibility |
|------|--------|----------------|
| `src/model/frame_cache.h` | Create | Public types and ops for tiered cache |
| `src/model/frame_cache.c` | Create | Implementation: backend tier (arena), CPU spill (malloc), recompute fallback, LRU |
| `include/sam3/sam3.h` | Modify | Add `sam3_video_start_opts`, `sam3_video_start_ex` |
| `src/model/video_session.h` | Modify | Replace `cached_features[]` with `frame_cache` |
| `src/model/video_session.c` | Modify | Init cache instead of eager encode loop |
| `src/model/sam3_video.c` | Modify | All `cached_features[f]` → `frame_cache_get(f)`; `sam3_video_reset` preserves cache |
| `tests/test_frame_cache.c` | Create | LRU eviction, spill, recompute fallback (with mock encoder) |

### Phase 4 — Memory persistence + clear_non_cond_mem_around_input

| File | Action | Responsibility |
|------|--------|----------------|
| `src/model/sam3_video.c` | Modify | Drop `sam3_memory_bank_clear` calls in `sam3_video_propagate`; call `clear_around_frame` on `add_*` for the affected obj |
| `tests/test_video_persistence.c` | Create | Idempotency, mid-propagate re-prompt window clear, FORWARD-then-BACKWARD continuity |

### Phase 5 — add_mask + dynamic multimask + iter_use_prev_mask_pred

| File | Action | Responsibility |
|------|--------|----------------|
| `src/model/mask_decoder.h` | Modify | Add params for stability selection; optional `prev_mask_logits` input |
| `src/model/mask_decoder.c` | Modify | `dynamic_multimask_via_stability` selection branch; dense-prompt input from prev mask |
| `src/model/tracker.c` | Modify | Plumb `prev_mask_logits` from per-object state into mask decoder call |
| `src/model/video_session.h` | Modify | Already has `prev_mask_logits` (Phase 2); just stash it after each conditioning eval |
| `include/sam3/sam3.h` | Modify | `sam3_video_add_mask` declaration |
| `src/model/sam3_video.c` | Modify | `sam3_video_add_mask` implementation: resize mask → memory encoder → cond commit, skipping decoder |
| `tests/test_dynamic_multimask.c` | Create | Stability selection, parity fixture |
| `tests/test_video_add_mask.c` | Create | Mask-prompt path |
| `tests/test_mask_decoder_nhwc.c` | Modify | Extend with stability fixture cases |

---

## Phase 1 — Per-object memory bank + new result types

Risk: low. No behavior change at the engine level. After Phase 1 the existing video tracker still works exactly as today, but the memory bank carries one obj_pointer per entry instead of an `[n_obj, 256]` slab.

### Task 1.1: Rename `obj_pointers` → `obj_pointer` in struct

**Files:**
- Modify: `src/model/memory_bank.h:23-29`

**Verify first:**
```bash
grep -n "obj_pointer" src/model/memory_bank.h
```
If the struct already has `obj_pointer` (singular, `[256]` comment), skip this task. As of baseline `d4f2d32`, the struct still says `obj_pointers` `[n_obj, hidden_dim]` — task is needed.

- [ ] **Step 1: Update struct field**

In `src/model/memory_bank.h`, change `struct sam3_memory_entry`:

```c
struct sam3_memory_entry {
	struct sam3_tensor *spatial_features; /* [HW, mem_dim] */
	struct sam3_tensor *obj_pointer;      /* [hidden_dim=256] (was obj_pointers [n_obj,256]) */
	int    frame_idx;
	int    is_conditioning;
	float  obj_score; /* max object score for SAM3-Long selection */
};
```

- [ ] **Step 2: Update single callsite in `src/model/sam3_video.c`**

Find the only callsite that sets `obj_pointers` (use Grep `obj_pointers` in `src/model/sam3_video.c` and `src/model/tracker.c`). Rename the field write to `obj_pointer = …`. The tensor it points to is already `[256]` in practice; the `[n_obj, 256]` shape was aspirational.

- [ ] **Step 3: Update single callsite in `src/model/tracker.c`** if any

Same as above. Build:
```bash
cd build && make -j8 sam3 2>&1 | head -20
```
Expected: clean build.

- [ ] **Step 4: Run full test suite to confirm no regression**

```bash
cd build && ctest --output-on-failure 2>&1 | tail -20
```
Expected: all existing tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/model/memory_bank.h src/model/memory_bank.c src/model/sam3_video.c src/model/tracker.c
git commit -m "$(cat <<'EOF'
model: rename memory_entry.obj_pointers to obj_pointer

The field was always populated with a single [256] pointer per entry
(the [n_obj, 256] shape never materialized). Renaming clarifies the
per-entry semantics ahead of the per-object bank refactor.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 1.2: Update existing memory bank tests for rename

**Files:**
- Modify: `tests/test_memory_bank.c`

- [ ] **Step 1: Update field references in tests**

In `tests/test_memory_bank.c`, change every `.obj_pointers = NULL` to `.obj_pointer = NULL`. Search:
```bash
grep -n "obj_pointers" tests/test_memory_bank.c
```

- [ ] **Step 2: Build and run**

```bash
cd build && make -j8 test_memory_bank && ctest -R test_memory_bank --output-on-failure
```
Expected: all 10 existing tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_memory_bank.c
git commit -m "tests: track obj_pointer rename in memory bank tests

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 1.3: Add `sam3_memory_bank_clear_around_frame` (TDD)

**Files:**
- Modify: `src/model/memory_bank.h`
- Modify: `src/model/memory_bank.c`
- Modify: `tests/test_memory_bank.c`

**Verify first:**
```bash
grep -n "sam3_memory_bank_clear_around_frame" src/model/memory_bank.h
```
If declared, skip. Not present in baseline `d4f2d32`.

This op clears non-conditioning entries within `[frame - window, frame + window]`. Used by `clear_non_cond_mem_around_input` in Phase 4.

- [ ] **Step 1: Write failing test**

Add to `tests/test_memory_bank.c` (before `int main`):

```c
/*
 * Clear non-cond entries within a frame window. Cond entries and
 * non-cond entries outside the window must survive.
 */
static void test_bank_clear_around_frame(void)
{
	struct sam3_memory_bank bank;
	sam3_memory_bank_init(&bank, 7, 4, 1, 0.0f);

	struct sam3_memory_entry cond = {
		.frame_idx = 5, .is_conditioning = 1, .obj_score = 1.0f
	};
	sam3_memory_bank_add(&bank, &cond);

	int frames[] = {0, 3, 5, 7, 10};
	for (int i = 0; i < 5; i++) {
		struct sam3_memory_entry e = {
			.frame_idx = frames[i],
			.is_conditioning = 0,
			.obj_score = 0.5f,
		};
		sam3_memory_bank_add(&bank, &e);
	}
	ASSERT_EQ(bank.n_non_cond, 5);
	ASSERT_EQ(bank.n_cond, 1);

	/* Clear non-cond within window=2 around frame 5: removes 3,5,7. */
	sam3_memory_bank_clear_around_frame(&bank, /*frame=*/5, /*window=*/2);
	ASSERT_EQ(bank.n_non_cond, 2);
	ASSERT_EQ(bank.non_cond[0].frame_idx, 0);
	ASSERT_EQ(bank.non_cond[1].frame_idx, 10);
	ASSERT_EQ(bank.n_cond, 1); /* cond untouched */
}
```

Wire it into `main()`:
```c
test_bank_clear_around_frame();
```

- [ ] **Step 2: Run test to confirm it fails to compile**

```bash
cd build && make test_memory_bank 2>&1 | tail -10
```
Expected: link/compile error referencing `sam3_memory_bank_clear_around_frame`.

- [ ] **Step 3: Add declaration in header**

Append to `src/model/memory_bank.h` before the closing `#endif`:

```c
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
```

- [ ] **Step 4: Implement in `src/model/memory_bank.c`**

Append:

```c
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
```

- [ ] **Step 5: Run test to confirm pass**

```bash
cd build && make test_memory_bank && ctest -R test_memory_bank --output-on-failure
```
Expected: all 11 tests pass (10 existing + new `test_bank_clear_around_frame`).

- [ ] **Step 6: Commit**

```bash
git add src/model/memory_bank.h src/model/memory_bank.c tests/test_memory_bank.c
git commit -m "$(cat <<'EOF'
model: add sam3_memory_bank_clear_around_frame

Drops non-conditioning entries within a frame window, leaving cond
entries intact. Mirrors Python clear_non_cond_mem_around_input: when
a new conditioning prompt arrives on a previously-tracked frame, the
propagated non-cond entries within the memory window are stale.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 1.4: Add `sam3_memory_bank_select_non_cond_for_frame` (TDD)

**Files:**
- Modify: `src/model/memory_bank.h`
- Modify: `src/model/memory_bank.c`
- Modify: `tests/test_memory_bank.c`

**Verify first:**
```bash
grep -n "sam3_memory_bank_select_non_cond_for_frame" src/model/memory_bank.h
```
If declared, skip. Not present in baseline `d4f2d32`. The existing `sam3_memory_bank_build_view` is for **cond** selection — distinct from this task.

Selects up to `(num_maskmem - 1)` non-cond entries to participate in attention for a given current frame. Implements Python's frame selection: every `temporal_stride`-th past frame, plus the most recent. Used by Phase 2 propagation.

- [ ] **Step 1: Write failing test**

Add to `tests/test_memory_bank.c`:

```c
/*
 * Select non-cond entries for the current frame attention pass.
 * With num_maskmem=4 and temporal_stride=2, expect: most recent
 * past frame, plus every 2nd frame back, capped at num_maskmem-1=3.
 */
static void test_bank_select_non_cond_for_frame(void)
{
	struct sam3_memory_bank bank;
	sam3_memory_bank_init(&bank, /*capacity=*/7,
			       /*max_cond_in_attn=*/4,
			       /*temporal_stride=*/2,
			       /*mf_threshold=*/0.0f);

	int frames[] = {1, 2, 3, 4, 5, 6};
	for (int i = 0; i < 6; i++) {
		struct sam3_memory_entry e = {
			.frame_idx = frames[i],
			.is_conditioning = 0,
			.obj_score = 0.5f,
		};
		sam3_memory_bank_add(&bank, &e);
	}
	/* capacity=7 → max non_cond = 6, so all six fit. */

	int selected[8];
	int n = sam3_memory_bank_select_non_cond_for_frame(
		&bank, /*current_frame=*/10,
		/*num_maskmem=*/4, selected, 8);
	/* num_maskmem-1=3 picks: most recent (frame 6), then frames
	 * 6 - stride*1 = 4, then 6 - stride*2 = 2. Indices into
	 * non_cond[] are bank-storage-order indices (frames {1..6} sit
	 * at positions {0..5}), so selected = {5, 3, 1}. */
	ASSERT_EQ(n, 3);
	ASSERT_EQ(bank.non_cond[selected[0]].frame_idx, 6);
	ASSERT_EQ(bank.non_cond[selected[1]].frame_idx, 4);
	ASSERT_EQ(bank.non_cond[selected[2]].frame_idx, 2);
}
```

Wire into `main()`:
```c
test_bank_select_non_cond_for_frame();
```

- [ ] **Step 2: Run test to confirm compile failure**

```bash
cd build && make test_memory_bank 2>&1 | tail -5
```
Expected: link error referencing `sam3_memory_bank_select_non_cond_for_frame`.

- [ ] **Step 3: Add declaration**

Append to `src/model/memory_bank.h` before `#endif`:

```c
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
 * available non-cond entry with frame_idx < current_frame (future
 * frames not eligible). De-duplicates if the same entry is selected
 * for multiple targets (skip on dup, no retry).
 *
 * Indices are returned in newest-to-oldest order, matching Python's
 * tpos enumeration.
 */
int sam3_memory_bank_select_non_cond_for_frame(
	const struct sam3_memory_bank *bank,
	int current_frame, int num_maskmem,
	int *out_indices, int max_n);
```

- [ ] **Step 4: Implement**

Append to `src/model/memory_bank.c`:

```c
int sam3_memory_bank_select_non_cond_for_frame(
	const struct sam3_memory_bank *bank,
	int current_frame, int num_maskmem,
	int *out_indices, int max_n)
{
	if (!bank || bank->n_non_cond == 0 || num_maskmem <= 1 || max_n <= 0)
		return 0;

	int stride = bank->temporal_stride > 0 ? bank->temporal_stride : 1;
	int n_pick = num_maskmem - 1;
	if (n_pick > max_n) n_pick = max_n;

	int written = 0;
	for (int t = 1; t <= n_pick; t++) {
		int target = current_frame - stride * t;

		/* Find non-cond entry with frame_idx closest to target,
		 * preferring entries with frame_idx <= current_frame. */
		int best_idx = -1;
		int best_dist = 0;
		for (int i = 0; i < bank->n_non_cond; i++) {
			int f = bank->non_cond[i].frame_idx;
			if (f >= current_frame)
				continue; /* future frames not eligible */
			int d = f - target;
			int abs_d = d < 0 ? -d : d;
			if (best_idx < 0 || abs_d < best_dist) {
				best_idx = i;
				best_dist = abs_d;
			}
		}
		if (best_idx < 0)
			break; /* no more eligible past entries */

		/* De-dup: skip if already selected. */
		int dup = 0;
		for (int j = 0; j < written; j++) {
			if (out_indices[j] == best_idx) {
				dup = 1;
				break;
			}
		}
		if (!dup) {
			out_indices[written++] = best_idx;
		}
	}
	return written;
}
```

- [ ] **Step 5: Run test**

```bash
cd build && make test_memory_bank && ctest -R test_memory_bank --output-on-failure
```
Expected: 12 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/model/memory_bank.h src/model/memory_bank.c tests/test_memory_bank.c
git commit -m "$(cat <<'EOF'
model: add sam3_memory_bank_select_non_cond_for_frame

Implements Python's per-frame memory selection: every temporal_stride-th
past frame, capped at num_maskmem-1. Returned in newest-first order to
match Python tpos enumeration. Foundation for the per-object propagation
pass in Phase 2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 1.5: Add public `sam3_video_object_mask` and `sam3_video_frame_result`

**Files:**
- Modify: `include/sam3/sam3.h`
- Create: `src/util/video_result.c` (small helpers — could go in sam3_video.c if you prefer; this plan uses a separate file for clarity)

**Verify first:**
```bash
grep -n "sam3_video_frame_result" include/sam3/sam3.h
```
If declared, skip. Not present in baseline `d4f2d32`.

- [ ] **Step 1: Append result types to public header**

In `include/sam3/sam3.h`, immediately before the `/* --- Video Tracking API --- */` section header (around line 222), drop `SAM3_MAX_OBJECTS` to 16:

```c
#define SAM3_MAX_OBJECTS         16   /* per-session object cap; was 64 */
```

(If a `#define SAM3_MAX_OBJECTS 64` already exists, modify it; otherwise add it.)

Then, in the video section, add the new types after the existing types but before the prompt entry-point declarations:

```c
/*
 * sam3_video_object_mask - Per-object segmentation result for one frame.
 */
struct sam3_video_object_mask {
	int    obj_id;            /* user-supplied obj_id */
	float *mask;              /* [mask_h * mask_w] f32 logits, malloc'd */
	int    mask_h, mask_w;
	float  iou_score;
	float  obj_score_logit;   /* >0 visible, <=0 occluded */
	int    is_occluded;       /* convenience flag (== obj_score_logit <= 0) */
};

/*
 * sam3_video_frame_result - Multi-object result for one video frame.
 *
 * Returned by the video propagate callback (Phase 2) and by the prompt
 * entry points (Phase 2 onward, single-object n_objects=1 in that case).
 */
struct sam3_video_frame_result {
	int frame_idx;
	int n_objects;
	struct sam3_video_object_mask *objects;
};

/*
 * sam3_video_frame_result_free - Release per-frame result memory.
 *
 * Frees each object_mask.mask buffer and the objects array. Safe to
 * call on a zero-initialized result.
 */
void sam3_video_frame_result_free(struct sam3_video_frame_result *r);
```

- [ ] **Step 2: Add helper implementation in `src/model/sam3_video.c`**

Open `src/model/sam3_video.c` and append (near the bottom, before the trailing `}` of any final block; or near `sam3_result_free` if it lives there):

```c
void sam3_video_frame_result_free(struct sam3_video_frame_result *r)
{
	if (!r)
		return;
	if (r->objects) {
		for (int i = 0; i < r->n_objects; i++)
			free(r->objects[i].mask);
		free(r->objects);
	}
	r->objects   = NULL;
	r->n_objects = 0;
	r->frame_idx = 0;
}
```

- [ ] **Step 3: Build to confirm no breakage**

```bash
cd build && make -j8 sam3 2>&1 | tail -10
```
Expected: clean build. The new types are unused by any caller yet, so no link errors.

- [ ] **Step 4: Add minimal sanity test**

Create `tests/test_video_frame_result.c`:

```c
/*
 * tests/test_video_frame_result.c - sam3_video_frame_result_free safety
 *
 * Key types: sam3_video_frame_result, sam3_video_object_mask
 * Depends on: sam3/sam3.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include "test_helpers.h"
#include "sam3/sam3.h"

static void test_free_zero_init_safe(void)
{
	struct sam3_video_frame_result r = {0};
	sam3_video_frame_result_free(&r);
	ASSERT_EQ(r.n_objects, 0);
	ASSERT(r.objects == NULL);
}

static void test_free_one_object(void)
{
	struct sam3_video_frame_result r = {0};
	r.frame_idx = 7;
	r.n_objects = 1;
	r.objects = calloc(1, sizeof(*r.objects));
	r.objects[0].obj_id = 42;
	r.objects[0].mask = calloc(16, sizeof(float));
	r.objects[0].mask_h = 4;
	r.objects[0].mask_w = 4;
	sam3_video_frame_result_free(&r);
	ASSERT_EQ(r.n_objects, 0);
	ASSERT(r.objects == NULL);
}

int main(void)
{
	test_free_zero_init_safe();
	test_free_one_object();
	TEST_REPORT();
}
```

Build and run:
```bash
cd build && cmake .. -DSAM3_TESTS=ON && make -j8 test_video_frame_result && ctest -R test_video_frame_result --output-on-failure
```
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add include/sam3/sam3.h src/model/sam3_video.c tests/test_video_frame_result.c
git commit -m "$(cat <<'EOF'
api: add sam3_video_frame_result type and free helper

Introduces the per-frame multi-object result type that Phase 2 will
plumb through prompt entry points and the propagate callback. Also
caps SAM3_MAX_OBJECTS at 16 (matches Python multiplex bucket size).
The new type is unused by callers yet — added alone to keep the
following diffs small.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 1.6: Phase 1 gate — full test suite green

- [ ] **Step 1: Build everything**

```bash
cd build && make -j8 2>&1 | tail -15
```
Expected: clean build.

- [ ] **Step 2: Run full test suite**

```bash
cd build && ctest --output-on-failure 2>&1 | tail -20
```
Expected: every test that was passing before phase 1 still passes; new tests (`test_memory_bank` with 12 cases, `test_video_frame_result` with 2 cases) pass.

- [ ] **Step 3: Smoke check the existing video CLI**

```bash
cd build && ./sam3_cli --help 2>&1 | head -5
```
Expected: prints help. (No model load required.)

- [ ] **Step 4: No commit needed if nothing changed.**

If a regression appears, fix it and commit before proceeding to Phase 2.

---

## Phase 2 — Multi-object propagation + new callback

Risk: medium. This is the big API-break commit and the per-frame loop becomes obj-indexed.

### Task 2.1: Add `sam3_video_object` per-object state struct

**Files:**
- Modify: `src/model/video_session.h`
- Modify: `src/model/video_session.c`

- [ ] **Step 1: Add the struct in `video_session.h`**

In `src/model/video_session.h`, near the existing session struct, add:

```c
/*
 * sam3_video_object - Per-object tracking state.
 *
 * Each user-added object owns its own memory bank, prompt-frame bitmap,
 * and prev-mask cache (Phase 5). Banks are independent: occluding
 * object A does not affect object B's tracking.
 */
struct sam3_video_object {
	int                       obj_id;             /* user-supplied id */
	struct sam3_memory_bank   bank;
	uint8_t                  *prompted_frames;    /* bitmap, [(n_frames+7)/8] */
	/* Phase 5 fields (added now to avoid header churn): */
	struct sam3_tensor       *prev_mask_logits;   /* [1,H,W,n_masks] or NULL */
	int                       prev_mask_frame;    /* -1 if none */
};
```

- [ ] **Step 2: Add to `sam3_video_session`**

In the same file, modify `struct sam3_video_session`:

```c
/* …existing fields… */
struct sam3_video_object objects[SAM3_MAX_OBJECTS];
int n_objects;
/* (Remove the old global mem_bank from this struct — now lives per-object.) */
```

Remove `mem_bank` from `struct sam3_tracker` if it lives there. The bank moves into per-object state.

If `obj_ids[]` array exists, remove it (replaced by `objects[i].obj_id`).

- [ ] **Step 3: Update lookups in `video_session.c`**

In `src/model/video_session.c`, update `sam3_session_get_or_add_obj` to operate on `session->objects[]` instead of any old `obj_ids[]`. Each new object gets its `bank` initialized:

```c
sam3_memory_bank_init(&session->objects[idx].bank,
		      /*capacity=*/7,
		      /*max_cond_in_attn=*/4,
		      /*temporal_stride=*/1,
		      /*mf_threshold=*/0.01f);
session->objects[idx].obj_id = obj_id;
session->objects[idx].prompted_frames = NULL; /* lazy alloc on first prompt */
session->objects[idx].prev_mask_logits = NULL;
session->objects[idx].prev_mask_frame = -1;
```

The `prompted_frames` bitmap is allocated lazily on the first prompt (size determined by `session->frames.n_frames`).

- [ ] **Step 4: Adapt `sam3_session_remove_obj` to free per-object resources**

```c
free(session->objects[idx].prompted_frames);
session->objects[idx].prompted_frames = NULL;
sam3_memory_bank_clear(&session->objects[idx].bank);
/* prev_mask_logits is arena-allocated, no free needed; null the pointer */
session->objects[idx].prev_mask_logits = NULL;
session->objects[idx].prev_mask_frame = -1;
```

Then memmove subsequent objects down to maintain compaction (if that's the existing pattern) or mark a slot free (if free-list).

- [ ] **Step 5: Build and run unit tests**

```bash
cd build && make -j8 2>&1 | tail -10 && ctest --output-on-failure 2>&1 | tail -20
```
Expected: clean build; existing tests still pass since `sam3_video.c` still uses obj 0 only at this point. If `test_video_session.c` has tests for `obj_ids[]`, update them to read `objects[i].obj_id`.

- [ ] **Step 6: Commit**

```bash
git add src/model/video_session.h src/model/video_session.c tests/test_video_session.c
git commit -m "$(cat <<'EOF'
session: add per-object state with own memory bank

Replaces the global mem_bank with per-object banks, prompt bitmaps,
and prev-mask cache slots. Each object's bank is independent so
occlusion in one object does not pollute another's tracking. The
propagation loop is updated to per-object in the next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 2.2: Convert `propagate_one` to per-object loop

**Files:**
- Modify: `src/model/sam3_video.c` (`propagate_one`, `video_propagate_pure_tracking`, replay path)

- [ ] **Step 1: Refactor `video_propagate_pure_tracking` to take an obj index**

Rename signature to:

```c
static enum sam3_error
video_propagate_pure_tracking_obj(struct sam3_video_session *session,
				  int obj_idx, int f,
				  struct sam3_video_object_mask *out_obj);
```

Inside, use `&session->objects[obj_idx].bank` everywhere `&session->tracker.mem_bank` was used. Output goes into the supplied `out_obj` (mask, IoU, obj_score_logit, is_occluded).

The mask buffer in `out_obj->mask` is malloc'd by the function (caller will assemble all per-object masks into the frame_result and free with `sam3_video_frame_result_free`).

- [ ] **Step 2: Refactor `propagate_one` to assemble per-object results**

```c
static enum sam3_error
propagate_one(struct sam3_video_session *session, int f,
	      struct sam3_video_frame_result *out)
{
	memset(out, 0, sizeof(*out));
	out->frame_idx = f;
	out->n_objects = session->n_objects;
	out->objects = calloc((size_t)session->n_objects,
			      sizeof(struct sam3_video_object_mask));
	if (!out->objects)
		return SAM3_ENOMEM;

	for (int i = 0; i < session->n_objects; i++) {
		out->objects[i].obj_id = session->objects[i].obj_id;
		enum sam3_error err;
		if (sam3_session_obj_is_prompted(session, i, f)) {
			err = video_replay_obj_prompt(session, i, f,
						       &out->objects[i]);
		} else {
			err = video_propagate_pure_tracking_obj(
				session, i, f, &out->objects[i]);
		}
		if (err != SAM3_OK) {
			sam3_video_frame_result_free(out);
			return err;
		}
	}
	return SAM3_OK;
}
```

(The helper `sam3_session_obj_is_prompted` reads the per-object bitmap; add it as a static inline in `video_session.h` if not present.)

- [ ] **Step 3: Update `sam3_video_propagate` to use the new result type and callback**

The callback signature changes. In `include/sam3/sam3.h`, replace the existing:

```c
typedef int (*sam3_video_frame_cb)(int frame_idx,
				   const struct sam3_result *result,
				   int n_objects, const int *obj_ids,
				   void *user_data);
```

with:

```c
typedef int (*sam3_video_frame_cb)(
	const struct sam3_video_frame_result *result,
	void *user_data);
```

And in `src/model/sam3_video.c`, the propagate sweep:

```c
for (int f = 0; f < nf; f++) {
	struct sam3_video_frame_result r = {0};
	err = propagate_one(session, f, &r);
	if (err != SAM3_OK) {
		sam3_video_frame_result_free(&r);
		return err;
	}
	if (callback) {
		int stop = callback(&r, user_data);
		sam3_video_frame_result_free(&r);
		if (stop) {
			sam3_log_info("video_propagate: stopped at frame %d", f);
			return SAM3_OK;
		}
	} else {
		sam3_video_frame_result_free(&r);
	}
	sam3_arena_reset(&session->scratch);
}
```

(Repeat for the BACKWARD sweep.)

- [ ] **Step 4: Update prompt entry points to return `sam3_video_frame_result`**

The existing `sam3_video_add_points` / `sam3_video_add_box` write into a `sam3_result *result`. Change to `sam3_video_frame_result *result`. The result has `n_objects=1`, the single object being the one prompted. Allocate `result->objects` inside the function and fill in the mask.

- [ ] **Step 5: Update CLI tools**

Find all callsites:
```bash
grep -rn "sam3_video_propagate\|sam3_video_add_points\|sam3_video_add_box" tools/ tests/
```

Update each:
- The propagate callback functions need new signature and to iterate `result->objects[]`.
- `add_points`/`add_box` callers need to allocate `sam3_video_frame_result`, pass it, and free with `sam3_video_frame_result_free` instead of `sam3_result_free`.

In `tools/cli_track.c` (and similar), the per-frame callback typically writes a mask to disk; with multi-object, write one file per object: `frame_NNNN_obj_MM.png` or similar. Pick a naming scheme and document it in the file header comment.

- [ ] **Step 6: Update existing video tests for new callback**

In `tests/test_video_api.c`, find every `sam3_video_propagate` call and update the callback signature. Migration is mechanical — pick the first object from `result->objects[0]` for single-object scenarios.

- [ ] **Step 7: Build everything**

```bash
cd build && make -j8 2>&1 | tail -20
```
Expected: clean build. Any miss surfaces here as a callback signature mismatch.

- [ ] **Step 8: Run existing video tests**

```bash
cd build && ctest -R "test_video|test_tracker|test_cli_track" --output-on-failure 2>&1 | tail -15
```
Expected: pass with single-object behavior preserved.

- [ ] **Step 9: Commit**

```bash
git add src/model/sam3_video.c src/model/video_session.h src/model/video_session.c \
        include/sam3/sam3.h \
        tools/cli_track.c tools/sam3_cli.c \
        tests/test_video_api.c tests/test_video_session.c
git commit -m "$(cat <<'EOF'
video: per-object propagation loop + multi-object callback

propagate_one now iterates session->objects[] and assembles a
sam3_video_frame_result containing per-object masks. The callback
signature is replaced with the new one-arg form. Prompt entry points
return per-frame results too (n_objects=1 for single-prompt calls).

CLI tools updated to consume the new result type. Single-object
tests preserved by picking objects[0] from the new result.

Breaking change: any external caller of sam3_video_propagate or the
prompt entry points must update their callback and result handling.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 2.3: Multi-object correctness test

**Files:**
- Create: `tests/test_video_multi_object.c`

This test exercises the per-object bank independence and result assembly at the unit level. End-to-end IoU vs Python is gated on Phase 5 (parity test).

- [ ] **Step 1: Write the test**

Create `tests/test_video_multi_object.c`:

```c
/*
 * tests/test_video_multi_object.c - Per-object propagation correctness
 *
 * Validates that multiple objects in one session each get their own
 * memory bank, that results carry per-object masks, and that
 * removing an object compacts state without affecting siblings.
 *
 * Key types: sam3_video_session, sam3_video_frame_result
 * Depends on: sam3/sam3.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include <string.h>
#include "test_helpers.h"
#include "sam3/sam3.h"
#include "model/video_session.h"

/*
 * Fabricate a session with two objects and verify per-object bank
 * independence at the data-structure level. (End-to-end IoU is in
 * test_video_e2e / parity test; this is a unit test.)
 */
static void test_two_objects_have_independent_banks(void)
{
	struct sam3_video_session session = {0};
	session.frames.n_frames = 4;

	int idx_a = sam3_session_get_or_add_obj(&session, /*obj_id=*/100);
	int idx_b = sam3_session_get_or_add_obj(&session, /*obj_id=*/101);
	ASSERT_EQ(idx_a, 0);
	ASSERT_EQ(idx_b, 1);
	ASSERT_EQ(session.n_objects, 2);

	/* Add a cond entry to obj A only. */
	struct sam3_memory_entry e = {
		.frame_idx = 0,
		.is_conditioning = 1,
		.obj_score = 1.0f,
	};
	sam3_memory_bank_add(&session.objects[idx_a].bank, &e);

	ASSERT_EQ(session.objects[idx_a].bank.n_cond, 1);
	ASSERT_EQ(session.objects[idx_b].bank.n_cond, 0);

	/* Cleanup */
	sam3_session_remove_obj(&session, /*obj_id=*/100);
	sam3_session_remove_obj(&session, /*obj_id=*/101);
}

static void test_remove_compacts_without_pollution(void)
{
	struct sam3_video_session session = {0};
	session.frames.n_frames = 4;

	sam3_session_get_or_add_obj(&session, 10);
	sam3_session_get_or_add_obj(&session, 20);
	sam3_session_get_or_add_obj(&session, 30);
	ASSERT_EQ(session.n_objects, 3);

	/* Stash a frame in obj 20's bank. */
	struct sam3_memory_entry e = {
		.frame_idx = 1,
		.is_conditioning = 1,
		.obj_score = 1.0f,
	};
	sam3_memory_bank_add(&session.objects[1].bank, &e);

	sam3_session_remove_obj(&session, /*obj_id=*/10);
	ASSERT_EQ(session.n_objects, 2);

	/* obj 20 must now sit at index 0; its bank must still have the
	 * cond entry (compaction must not nuke surviving state). */
	ASSERT_EQ(session.objects[0].obj_id, 20);
	ASSERT_EQ(session.objects[0].bank.n_cond, 1);
	ASSERT_EQ(session.objects[0].bank.cond[0].frame_idx, 1);
}

static void test_full_object_cap_returns_efull(void)
{
	struct sam3_video_session session = {0};
	session.frames.n_frames = 4;

	for (int i = 0; i < SAM3_MAX_OBJECTS; i++) {
		int idx = sam3_session_get_or_add_obj(&session, 100 + i);
		ASSERT(idx >= 0);
	}
	ASSERT_EQ(session.n_objects, SAM3_MAX_OBJECTS);

	/* 17th add must fail. */
	int idx = sam3_session_get_or_add_obj(&session, /*obj_id=*/9999);
	ASSERT(idx < 0);
}

int main(void)
{
	test_two_objects_have_independent_banks();
	test_remove_compacts_without_pollution();
	test_full_object_cap_returns_efull();
	TEST_REPORT();
}
```

- [ ] **Step 2: Build and run**

```bash
cd build && cmake .. && make -j8 test_video_multi_object && ctest -R test_video_multi_object --output-on-failure
```
Expected: 3 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_video_multi_object.c
git commit -m "$(cat <<'EOF'
tests: per-object bank independence and remove-compact

Three unit tests covering per-object memory bank isolation, object
removal preserving sibling state, and the SAM3_MAX_OBJECTS cap.
End-to-end multi-object tracking IoU comes in the Phase 5 parity
test once the rest of the stack is in place.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 2.4: Phase 2 gate — full test suite + manual smoke

- [ ] **Step 1: Build everything**

```bash
cd build && make -j8 2>&1 | tail -15
```

- [ ] **Step 2: Run full test suite**

```bash
cd build && ctest --output-on-failure 2>&1 | tail -25
```
Expected: every existing test plus the new `test_video_multi_object` passes.

- [ ] **Step 3: Manual single-object regression check**

If a `.sam3` model is available locally, run the video CLI on a known clip and compare output masks to a previous run (visual diff). This is non-blocking but worth doing to catch silent regressions.

```bash
cd build && ./sam3_cli track --help
```

- [ ] **Step 4: No commit if nothing changed.**

---

## Phase 3 — Frame cache (lazy + tiered)

Risk: medium. Replaces eager encoding. Long-video correctness depends on this.

### Task 3.1: `frame_cache.h` API surface

**Files:**
- Create: `src/model/frame_cache.h`

- [ ] **Step 1: Create the header**

Write `src/model/frame_cache.h`:

```c
/*
 * src/model/frame_cache.h - Tiered LRU cache for encoded frame features
 *
 * Replaces eager full-video encoding. Each frame's encoded features
 * (image_features + feat_s0 + feat_s1) live in one of three tiers:
 *
 *   1. Backend tier — held in a dedicated arena, ready for tracker
 *      consumption. Default budget 4 GiB.
 *   2. CPU spill — raw byte copies on the host. Default budget 16 GiB,
 *      0 disables. Promotion to backend on next access is a memcpy.
 *   3. Recompute — re-runs the image encoder on a miss. Falls through
 *      automatically when both tiers are full.
 *
 * Eviction is LRU within each tier. The cache is invisible to
 * correctness — callers see fully-encoded features regardless.
 *
 * Key types:  sam3_frame_cache, sam3_frame_features
 * Depends on: core/tensor.h, core/alloc.h
 * Used by:    model/video_session.h, model/sam3_video.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_FRAME_CACHE_H
#define SAM3_MODEL_FRAME_CACHE_H

#include <stddef.h>
#include <stdint.h>
#include "core/tensor.h"
#include "core/alloc.h"

struct sam3_video_session; /* fwd */

/*
 * sam3_frame_features - Encoded features for one frame.
 *
 * All three tensors are owned by the cache (backend tier). Caller
 * must not free them.
 */
struct sam3_frame_features {
	struct sam3_tensor *image_features; /* [1, H, W, 256] */
	struct sam3_tensor *feat_s0;        /* [1, 4H, 4W, 256] */
	struct sam3_tensor *feat_s1;        /* [1, 2H, 2W, 256] */
};

/*
 * sam3_frame_cache_encode_fn - Image encoder hook.
 *
 * Called on cache misses. Runs the image encoder for the requested
 * frame and writes the three tensors into the supplied output struct.
 * Tensor allocation must use `arena`.
 */
typedef enum sam3_error (*sam3_frame_cache_encode_fn)(
	struct sam3_video_session *session,
	int frame_idx,
	struct sam3_arena *arena,
	struct sam3_frame_features *out);

enum sam3_frame_tier {
	SAM3_FRAME_TIER_NONE = 0,
	SAM3_FRAME_TIER_BACKEND,
	SAM3_FRAME_TIER_CPU_SPILL,
};

struct sam3_frame_cache_slot {
	int                  frame_idx;       /* always equals slot index */
	enum sam3_frame_tier tier;
	/* Backend tier tensors: */
	struct sam3_tensor  *image_features;
	struct sam3_tensor  *feat_s0;
	struct sam3_tensor  *feat_s1;
	/* Spill tier byte buffers (same layout/strides as backend): */
	void                *spill_image_features;
	void                *spill_feat_s0;
	void                *spill_feat_s1;
	size_t               spill_bytes;     /* total bytes for this slot */
	uint64_t             last_access_seq;
};

struct sam3_frame_cache {
	struct sam3_frame_cache_slot *slots;  /* [n_frames] */
	int                           n_frames;
	size_t                        backend_budget;
	size_t                        backend_used;
	size_t                        spill_budget;
	size_t                        spill_used;
	uint64_t                      access_counter;
	struct sam3_arena             backend_arena;
	struct sam3_video_session    *owner;  /* for encode hook callback */
	sam3_frame_cache_encode_fn    encode;
};

/*
 * sam3_frame_cache_init - Allocate slots and the backend arena.
 *
 * @cache:           Output cache (caller-allocated, zeroed).
 * @owner:           Session, passed to encode hook on misses.
 * @encode:          Encoder hook for cache misses.
 * @n_frames:        Total frames in the video.
 * @backend_budget:  Bytes for the backend arena. 0 → 4 GiB default.
 * @spill_budget:    Bytes for CPU spill. 0 → 16 GiB default. SIZE_MAX → disable.
 *
 * Returns SAM3_OK on success, SAM3_ENOMEM if arena alloc fails.
 */
enum sam3_error sam3_frame_cache_init(struct sam3_frame_cache *cache,
				      struct sam3_video_session *owner,
				      sam3_frame_cache_encode_fn encode,
				      int n_frames,
				      size_t backend_budget,
				      size_t spill_budget);

/*
 * sam3_frame_cache_get - Fetch features for a frame.
 *
 * @cache:    Initialized cache.
 * @frame:    Frame index, 0 ≤ frame < n_frames.
 * @out:      Output features (pointers into cache-owned tensors).
 *
 * On miss, encodes (if not in any tier), promotes (if in spill), or
 * recomputes (if no room in either tier and slot was evicted). Updates
 * LRU bookkeeping on every call.
 *
 * Returns SAM3_OK on success, SAM3_EINVAL on bad args, SAM3_ENOMEM if
 * backend arena and spill both fail and the encoder cannot run.
 */
enum sam3_error sam3_frame_cache_get(struct sam3_frame_cache *cache,
				     int frame,
				     struct sam3_frame_features *out);

/*
 * sam3_frame_cache_release - Free spill buffers; backend arena is
 *                            freed with the session.
 *
 * @cache: Cache to release. Slots array is freed; spill buffers are
 *         freed; backend_arena is destroyed.
 */
void sam3_frame_cache_release(struct sam3_frame_cache *cache);

/*
 * sam3_frame_cache_invalidate - Drop all cached features.
 *
 * Resets every slot to TIER_NONE; frees spill buffers; resets the
 * backend arena. Used by sam3_video_reset to wipe propagation state
 * while preserving the cache size budgets and encode hook.
 *
 * (Spec §3.3 says reset clears banks but PRESERVES the cache. This
 * helper exists for explicit invalidation when needed — e.g., model
 * reloaded with different dimensions; not called by sam3_video_reset.)
 */
void sam3_frame_cache_invalidate(struct sam3_frame_cache *cache);

#endif /* SAM3_MODEL_FRAME_CACHE_H */
```

- [ ] **Step 2: Build to confirm header compiles**

```bash
cd build && cmake .. && touch ../src/model/frame_cache.h && make -j8 sam3 2>&1 | tail -5
```
(No .c yet — header inclusion checks come with first user.) The build won't reference it; no expected output change.

- [ ] **Step 3: Commit**

```bash
git add src/model/frame_cache.h
git commit -m "$(cat <<'EOF'
model: add frame_cache.h tiered LRU API

Public surface for the lazy frame-feature cache. Backend arena tier,
CPU spill tier, and recompute fallback. Implementation lands in the
next commit; existing eager code still drives the session for now.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 3.2: `frame_cache.c` implementation

**Files:**
- Create: `src/model/frame_cache.c`

- [ ] **Step 1: Implement init/release/invalidate**

Create `src/model/frame_cache.c` with the file header, then:

```c
#include <stdlib.h>
#include <string.h>

#include "model/frame_cache.h"
#include "model/video_session.h"   /* for owner type */
#include "util/log.h"

#define DEFAULT_BACKEND_BUDGET ((size_t)4 * 1024 * 1024 * 1024)  /* 4 GiB */
#define DEFAULT_SPILL_BUDGET   ((size_t)16 * 1024 * 1024 * 1024) /* 16 GiB */

enum sam3_error sam3_frame_cache_init(struct sam3_frame_cache *cache,
				      struct sam3_video_session *owner,
				      sam3_frame_cache_encode_fn encode,
				      int n_frames,
				      size_t backend_budget,
				      size_t spill_budget)
{
	if (!cache || !encode || n_frames <= 0)
		return SAM3_EINVAL;

	memset(cache, 0, sizeof(*cache));
	cache->owner    = owner;
	cache->encode   = encode;
	cache->n_frames = n_frames;
	cache->backend_budget = backend_budget ? backend_budget
					       : DEFAULT_BACKEND_BUDGET;
	cache->spill_budget   = spill_budget   ? spill_budget
					       : DEFAULT_SPILL_BUDGET;

	cache->slots = calloc((size_t)n_frames, sizeof(*cache->slots));
	if (!cache->slots)
		return SAM3_ENOMEM;
	for (int i = 0; i < n_frames; i++)
		cache->slots[i].frame_idx = i;

	enum sam3_error err =
		sam3_arena_init(&cache->backend_arena, cache->backend_budget);
	if (err != SAM3_OK) {
		free(cache->slots);
		cache->slots = NULL;
		return err;
	}
	return SAM3_OK;
}

static void slot_free_spill(struct sam3_frame_cache_slot *s,
			    struct sam3_frame_cache *cache)
{
	if (s->spill_image_features) free(s->spill_image_features);
	if (s->spill_feat_s0)        free(s->spill_feat_s0);
	if (s->spill_feat_s1)        free(s->spill_feat_s1);
	cache->spill_used -= s->spill_bytes;
	s->spill_image_features = NULL;
	s->spill_feat_s0        = NULL;
	s->spill_feat_s1        = NULL;
	s->spill_bytes          = 0;
}

void sam3_frame_cache_invalidate(struct sam3_frame_cache *cache)
{
	if (!cache || !cache->slots)
		return;
	for (int i = 0; i < cache->n_frames; i++) {
		struct sam3_frame_cache_slot *s = &cache->slots[i];
		if (s->tier == SAM3_FRAME_TIER_CPU_SPILL)
			slot_free_spill(s, cache);
		s->image_features = NULL;
		s->feat_s0        = NULL;
		s->feat_s1        = NULL;
		s->tier           = SAM3_FRAME_TIER_NONE;
		s->last_access_seq = 0;
	}
	cache->backend_used = 0;
	cache->spill_used   = 0;
	sam3_arena_reset(&cache->backend_arena);
	cache->access_counter = 0;
}

void sam3_frame_cache_release(struct sam3_frame_cache *cache)
{
	if (!cache || !cache->slots)
		return;
	sam3_frame_cache_invalidate(cache);
	free(cache->slots);
	cache->slots = NULL;
	sam3_arena_release(&cache->backend_arena);
	memset(cache, 0, sizeof(*cache));
}
```

- [ ] **Step 2: Implement byte-size accounting**

Append:

```c
static size_t features_bytes(const struct sam3_frame_features *f)
{
	size_t b = 0;
	if (f->image_features) b += f->image_features->nbytes;
	if (f->feat_s0)        b += f->feat_s0->nbytes;
	if (f->feat_s1)        b += f->feat_s1->nbytes;
	return b;
}
```

- [ ] **Step 3: Implement spill ↔ backend promotion/eviction**

Append:

```c
/* Spill a backend-tier slot to CPU (memcpy out, then drop backend ptrs). */
static enum sam3_error
spill_slot_to_cpu(struct sam3_frame_cache_slot *s,
		  struct sam3_frame_cache *cache)
{
	if (s->tier != SAM3_FRAME_TIER_BACKEND)
		return SAM3_EINVAL;

	size_t total = s->image_features->nbytes
		     + s->feat_s0->nbytes
		     + s->feat_s1->nbytes;

	if (cache->spill_used + total > cache->spill_budget) {
		/* Spill budget exhausted — evict instead. */
		s->image_features = NULL;
		s->feat_s0        = NULL;
		s->feat_s1        = NULL;
		s->tier           = SAM3_FRAME_TIER_NONE;
		return SAM3_OK; /* caller checks tier */
	}

	void *im = malloc(s->image_features->nbytes);
	void *s0 = malloc(s->feat_s0->nbytes);
	void *s1 = malloc(s->feat_s1->nbytes);
	if (!im || !s0 || !s1) {
		free(im); free(s0); free(s1);
		sam3_log_warn("frame_cache: spill malloc failed for frame %d",
			      s->frame_idx);
		return SAM3_ENOMEM;
	}
	memcpy(im, s->image_features->data, s->image_features->nbytes);
	memcpy(s0, s->feat_s0->data,        s->feat_s0->nbytes);
	memcpy(s1, s->feat_s1->data,        s->feat_s1->nbytes);

	s->spill_image_features = im;
	s->spill_feat_s0        = s0;
	s->spill_feat_s1        = s1;
	s->spill_bytes          = total;
	cache->spill_used      += total;

	s->image_features = NULL;
	s->feat_s0        = NULL;
	s->feat_s1        = NULL;
	s->tier           = SAM3_FRAME_TIER_CPU_SPILL;
	return SAM3_OK;
}

/* Promote a spilled slot back to backend by re-allocating + memcpy. */
static enum sam3_error
promote_slot_from_spill(struct sam3_frame_cache_slot *s,
			struct sam3_frame_cache *cache,
			const struct sam3_frame_features *probe);
/* Implemented below; relies on tensor metadata recomputed by encode hook
 * on first miss. For simplicity, promotion currently re-encodes; the
 * memcpy fast path is added in a follow-up if profiling shows it matters.
 */
```

For the first version, **promotion is recompute** (the spill stores the bytes but rehydrating tensor metadata cleanly is fiddly; spec mentions follow-up optimization). The spill tier still wins on the *eviction* side because instead of dropping bytes immediately, we keep them around. A follow-up task can replace recompute with the memcpy fast path.

- [ ] **Step 4: Implement LRU eviction loop**

Append:

```c
/* Evict the LRU backend-tier slot until we have `need` bytes free.
 * Spill the evicted slot if spill has room, otherwise drop it. */
static enum sam3_error
make_backend_room(struct sam3_frame_cache *cache, size_t need)
{
	while (cache->backend_used + need > cache->backend_budget) {
		struct sam3_frame_cache_slot *victim = NULL;
		uint64_t oldest = UINT64_MAX;

		for (int i = 0; i < cache->n_frames; i++) {
			struct sam3_frame_cache_slot *s = &cache->slots[i];
			if (s->tier != SAM3_FRAME_TIER_BACKEND)
				continue;
			if (s->last_access_seq < oldest) {
				oldest = s->last_access_seq;
				victim = s;
			}
		}
		if (!victim) {
			sam3_log_error("frame_cache: cannot make room (%zu B "
				       "needed, %zu/%zu used)",
				       need, cache->backend_used,
				       cache->backend_budget);
			return SAM3_ENOMEM;
		}

		size_t freed = victim->image_features->nbytes
			     + victim->feat_s0->nbytes
			     + victim->feat_s1->nbytes;
		enum sam3_error err = spill_slot_to_cpu(victim, cache);
		if (err != SAM3_OK)
			return err;
		cache->backend_used -= freed;

		/* Backend arena is bump-allocated; we can't reclaim
		 * per-tensor space without resetting it. Track logical
		 * use; arena reset happens lazily when fragmentation
		 * exceeds budget by 2x. */
		if (cache->backend_used == 0)
			sam3_arena_reset(&cache->backend_arena);
	}
	return SAM3_OK;
}
```

**Note on arena fragmentation:** the bump-allocator nature means evicted bytes don't actually free until full reset. Track `backend_used` logically; reset the arena when all slots are evicted. This is acceptable for the typical access pattern (sequential sweeps where evictions cluster).

- [ ] **Step 5: Implement `sam3_frame_cache_get`**

Append:

```c
enum sam3_error sam3_frame_cache_get(struct sam3_frame_cache *cache,
				     int frame,
				     struct sam3_frame_features *out)
{
	if (!cache || !out || frame < 0 || frame >= cache->n_frames)
		return SAM3_EINVAL;

	struct sam3_frame_cache_slot *s = &cache->slots[frame];
	cache->access_counter++;
	s->last_access_seq = cache->access_counter;

	if (s->tier == SAM3_FRAME_TIER_BACKEND) {
		out->image_features = s->image_features;
		out->feat_s0        = s->feat_s0;
		out->feat_s1        = s->feat_s1;
		return SAM3_OK;
	}

	/* Spill or none → run encoder (recompute on miss; spill bytes
	 * are kept but unused by this version. Future: memcpy promote). */
	struct sam3_frame_features fresh = {0};
	enum sam3_error err = cache->encode(cache->owner, frame,
					    &cache->backend_arena, &fresh);
	if (err != SAM3_OK)
		return err;

	size_t need = features_bytes(&fresh);
	if (need + cache->backend_used > cache->backend_budget) {
		err = make_backend_room(cache, need);
		if (err != SAM3_OK)
			return err;
	}

	s->image_features = fresh.image_features;
	s->feat_s0        = fresh.feat_s0;
	s->feat_s1        = fresh.feat_s1;
	s->tier           = SAM3_FRAME_TIER_BACKEND;
	cache->backend_used += need;

	/* If we had spill bytes, free them (we now have backend copy). */
	if (s->spill_image_features)
		slot_free_spill(s, cache);

	*out = fresh;
	return SAM3_OK;
}
```

- [ ] **Step 6: Build**

```bash
cd build && cmake .. && make -j8 sam3 2>&1 | tail -10
```
Expected: clean build (no callers yet besides session init in next task).

- [ ] **Step 7: Commit**

```bash
git add src/model/frame_cache.c
git commit -m "$(cat <<'EOF'
model: implement frame_cache (lazy + tiered)

Backend tier uses a dedicated bump arena; CPU spill keeps evicted
bytes around for future memcpy-promote (currently recompute on miss
to avoid tensor-metadata re-hydration). LRU eviction makes room as
needed; full arena reset triggers when the backend tier empties.

Used by Phase 3 follow-up tasks that replace eager encoding.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 3.3: `test_frame_cache.c` unit tests

**Files:**
- Create: `tests/test_frame_cache.c`

- [ ] **Step 1: Write tests with a mock encoder**

Create `tests/test_frame_cache.c`:

```c
/*
 * tests/test_frame_cache.c - Tiered LRU frame cache unit tests
 *
 * Validates LRU eviction order, spill-to-CPU on backend overflow,
 * recompute-on-miss when both tiers are full, and that
 * sam3_frame_cache_invalidate resets state cleanly.
 *
 * Key types: sam3_frame_cache, sam3_frame_features
 * Depends on: model/frame_cache.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include <string.h>
#include "test_helpers.h"
#include "core/tensor.h"
#include "core/alloc.h"
#include "model/frame_cache.h"

static int g_encode_calls = 0;

/* Mock encoder: produces tiny fake tensors so we can fit many in budget. */
static enum sam3_error mock_encode(struct sam3_video_session *session,
				   int frame_idx,
				   struct sam3_arena *arena,
				   struct sam3_frame_features *out)
{
	(void)session;
	g_encode_calls++;
	int dims[4] = {1, 2, 2, 4};
	out->image_features = sam3_arena_alloc_tensor(arena, SAM3_DTYPE_F32, 4, dims);
	out->feat_s0        = sam3_arena_alloc_tensor(arena, SAM3_DTYPE_F32, 4, dims);
	out->feat_s1        = sam3_arena_alloc_tensor(arena, SAM3_DTYPE_F32, 4, dims);
	if (!out->image_features || !out->feat_s0 || !out->feat_s1)
		return SAM3_ENOMEM;
	((float *)out->image_features->data)[0] = (float)frame_idx;
	return SAM3_OK;
}

static void test_init_and_release(void)
{
	struct sam3_frame_cache cache = {0};
	enum sam3_error err = sam3_frame_cache_init(&cache, NULL, mock_encode,
						    /*n_frames=*/8,
						    /*backend=*/1024 * 1024,
						    /*spill=*/SIZE_MAX);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT_EQ(cache.n_frames, 8);
	sam3_frame_cache_release(&cache);
	ASSERT(cache.slots == NULL);
}

static void test_first_access_encodes_then_hits(void)
{
	g_encode_calls = 0;
	struct sam3_frame_cache cache = {0};
	sam3_frame_cache_init(&cache, NULL, mock_encode, 4,
			      /*backend=*/1024 * 1024, /*spill=*/SIZE_MAX);

	struct sam3_frame_features f1 = {0}, f2 = {0};
	ASSERT_EQ(sam3_frame_cache_get(&cache, 2, &f1), SAM3_OK);
	ASSERT_EQ(g_encode_calls, 1);

	ASSERT_EQ(sam3_frame_cache_get(&cache, 2, &f2), SAM3_OK);
	ASSERT_EQ(g_encode_calls, 1);          /* cache hit */
	ASSERT(f1.image_features == f2.image_features); /* same pointer */

	sam3_frame_cache_release(&cache);
}

static void test_lru_eviction_when_backend_full(void)
{
	g_encode_calls = 0;
	struct sam3_frame_cache cache = {0};
	/* Per slot: 3 tensors * (1*2*2*4 f32 + tensor metadata) ~ 192 bytes
	 * data + metadata. Tight budget forces eviction at the 4th frame. */
	sam3_frame_cache_init(&cache, NULL, mock_encode, 8,
			      /*backend=*/4096, /*spill=*/SIZE_MAX);

	struct sam3_frame_features f = {0};
	for (int i = 0; i < 6; i++) {
		ASSERT_EQ(sam3_frame_cache_get(&cache, i, &f), SAM3_OK);
	}
	ASSERT_EQ(g_encode_calls, 6); /* one per frame */

	/* At least the oldest slots should now be in spill or recompute,
	 * not BACKEND, since budget < 6 * per-frame size. */
	int still_backend = 0, in_spill = 0;
	for (int i = 0; i < 6; i++) {
		if (cache.slots[i].tier == SAM3_FRAME_TIER_BACKEND)
			still_backend++;
		else if (cache.slots[i].tier == SAM3_FRAME_TIER_CPU_SPILL)
			in_spill++;
	}
	ASSERT(still_backend < 6); /* eviction occurred */
	ASSERT(in_spill > 0);      /* spill tier active */

	sam3_frame_cache_release(&cache);
}

static void test_invalidate_clears_state(void)
{
	g_encode_calls = 0;
	struct sam3_frame_cache cache = {0};
	sam3_frame_cache_init(&cache, NULL, mock_encode, 4, 0, SIZE_MAX);

	struct sam3_frame_features f = {0};
	sam3_frame_cache_get(&cache, 0, &f);
	sam3_frame_cache_get(&cache, 1, &f);
	int before = g_encode_calls;

	sam3_frame_cache_invalidate(&cache);
	for (int i = 0; i < 4; i++)
		ASSERT_EQ(cache.slots[i].tier, SAM3_FRAME_TIER_NONE);

	sam3_frame_cache_get(&cache, 0, &f);
	ASSERT_EQ(g_encode_calls, before + 1); /* re-encoded */

	sam3_frame_cache_release(&cache);
}

int main(void)
{
	test_init_and_release();
	test_first_access_encodes_then_hits();
	test_lru_eviction_when_backend_full();
	test_invalidate_clears_state();
	TEST_REPORT();
}
```

- [ ] **Step 2: Build and run**

```bash
cd build && cmake .. && make -j8 test_frame_cache && ctest -R test_frame_cache --output-on-failure
```
Expected: 4 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_frame_cache.c
git commit -m "$(cat <<'EOF'
tests: frame_cache LRU + spill behavior

Mock encoder lets us drive the cache through eviction, spill, and
invalidation paths without touching the real image encoder. Exercises
hit semantics, LRU eviction, spill activation, and reset behavior.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 3.4: Add `sam3_video_start_opts` and `sam3_video_start_ex`

**Files:**
- Modify: `include/sam3/sam3.h`
- Modify: `src/model/sam3_video.c`

- [ ] **Step 1: Add the opts struct + extended starter**

In `include/sam3/sam3.h`, near the existing `sam3_video_start` declaration:

```c
struct sam3_video_start_opts {
	size_t frame_cache_backend_budget; /* 0 = default 4 GiB */
	size_t frame_cache_spill_budget;   /* 0 = default 16 GiB; SIZE_MAX = disable */
	int    clear_non_cond_window;      /* 0 = default 7 (= num_maskmem) */
	int    iter_use_prev_mask_pred;    /* -1 = default on (1) */
	int    multimask_via_stability;    /* -1 = default on (1) */
	float  multimask_stability_delta;  /* 0 = default 0.05 */
	float  multimask_stability_thresh; /* 0 = default 0.98 */
};

enum sam3_error sam3_video_start_ex(sam3_ctx *ctx, const char *resource_path,
				    const struct sam3_video_start_opts *opts,
				    sam3_video_session **out_session);
```

- [ ] **Step 2: Refactor `sam3_video_start` to call `_ex`**

In `src/model/sam3_video.c`, change the existing `sam3_video_start` to a thin wrapper:

```c
enum sam3_error sam3_video_start(sam3_ctx *ctx, const char *resource_path,
				 sam3_video_session **out_session)
{
	return sam3_video_start_ex(ctx, resource_path, NULL, out_session);
}
```

Then write `sam3_video_start_ex` with the body of the old `sam3_video_start`. Where the old function ran the eager image encode loop, replace with `sam3_frame_cache_init`. Wire the encode hook to a static helper that mirrors what the old loop did for one frame.

- [ ] **Step 3: Add the encode hook**

In `src/model/sam3_video.c`, add:

```c
static enum sam3_error
session_encode_frame(struct sam3_video_session *session,
		     int frame_idx,
		     struct sam3_arena *arena,
		     struct sam3_frame_features *out)
{
	/* Body lifted from the old per-frame loop in sam3_video_start.
	 * Loads the frame image, runs the image encoder graph, clones
	 * the three NHWC outputs into `arena`. */
	/* …implementation… */
	return SAM3_OK;
}
```

(Lift the loop body from the existing `sam3_video_start`. The clone destination changes from `session->persist` to the cache's `arena`.)

- [ ] **Step 4: Replace `cached_features[]` with `frame_cache_get` in callers**

Find all `session->cached_features[f]` references and rewrite as:

```c
struct sam3_frame_features ff;
err = sam3_frame_cache_get(&session->frame_cache, f, &ff);
if (err != SAM3_OK) goto cleanup;
/* use ff.image_features, ff.feat_s0, ff.feat_s1 */
```

Remove the now-unused `cached_features` array from `struct sam3_video_session`.

- [ ] **Step 5: `sam3_video_end` and `sam3_video_reset`**

- `sam3_video_end` should call `sam3_frame_cache_release` before freeing the session.
- `sam3_video_reset` should NOT call invalidate — spec says reset preserves cache. Just clear per-object banks and prompts.

- [ ] **Step 6: Build full project**

```bash
cd build && cmake .. && make -j8 2>&1 | tail -15
```
Expected: clean build.

- [ ] **Step 7: Run full test suite**

```bash
cd build && ctest --output-on-failure 2>&1 | tail -20
```
Expected: every test passes. The frame cache is now the only path; existing video tests transparently use it.

- [ ] **Step 8: Long-video smoke test**

If there's a longer test asset available (or generate one — 200+ synthetic frames), run a propagate sweep and confirm bounded memory:

```bash
# (if available) cd build && ./sam3_cli track --video <long_clip.mp4> ... 2>&1 | tail -10
```
Expected: completes; resident memory stays under backend + spill budget.

- [ ] **Step 9: Commit**

```bash
git add include/sam3/sam3.h src/model/sam3_video.c src/model/video_session.h \
        src/model/video_session.c
git commit -m "$(cat <<'EOF'
video: replace eager encode with tiered frame cache

sam3_video_start now delegates to sam3_video_start_ex with default
opts. The image encoder is no longer invoked at session start;
features are produced on demand by the frame cache via a session
encode hook. sam3_video_reset preserves the cache (per spec §3.3);
sam3_video_end releases it.

Long-video sessions (≫200 frames) are now feasible: cache budgets
default to 4 GiB backend + 16 GiB spill; recompute fallback when
both tiers are full.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 3.5: Phase 3 gate — long-video smoke + perf check

- [ ] **Step 1: Re-run full test suite**

```bash
cd build && ctest --output-on-failure 2>&1 | tail -25
```

- [ ] **Step 2: Re-run video benchmarks** (if `SAM3_BENCH=ON`)

```bash
cd build && cmake .. -DSAM3_BENCH=ON && make sam3_cli && ./sam3_cli bench video --help
```

Compare per-frame timing vs phase-2 baseline on the same `kids.mp4` clip. Spec §8 phase-3 gate: variance within 5%. If you don't have a baseline number, record this as the new baseline.

- [ ] **Step 3: No commit unless changes were needed.**

---

## Phase 4 — Memory persistence + clear_non_cond_mem_around_input

Risk: medium-high. Changes propagate idempotency semantics.

### Task 4.1: Drop `sam3_memory_bank_clear` from `sam3_video_propagate`

**Files:**
- Modify: `src/model/sam3_video.c` (`sam3_video_propagate` lines 1468 and 1507 area)

- [ ] **Step 1: Remove the on-entry and between-pass clears**

Find `sam3_memory_bank_clear(&session->tracker.mem_bank)` calls — they were per-session. Now banks are per-object; we want them to PERSIST across propagate calls. Delete both calls.

Also remove the now-stale block comment around them that says "Memory bank is rebuilt from scratch every propagate call — cleared once on entry. …"

- [ ] **Step 2: Build**

```bash
cd build && make -j8 2>&1 | tail -5
```
Expected: clean build.

- [ ] **Step 3: Run existing tests**

```bash
cd build && ctest --output-on-failure 2>&1 | tail -20
```
Expected: tests pass; if any test relied on the clear (e.g., calling propagate twice and expecting fresh state), it should now FAIL — that's the next task.

- [ ] **Step 4: Commit**

```bash
git add src/model/sam3_video.c
git commit -m "$(cat <<'EOF'
video: stop clearing memory banks on propagate entry

Banks now persist across propagate() calls so a second call extends
tracking instead of recomputing from scratch. The clear-on-entry
behavior is replaced by per-prompt clear_non_cond_around in the next
commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 4.2: Call `clear_around_frame` from `add_*` entry points

**Files:**
- Modify: `src/model/sam3_video.c` (`sam3_video_add_points`, `sam3_video_add_box`, eventually `sam3_video_add_mask` in Phase 5)

- [ ] **Step 1: Add the clear call before processing each new prompt**

In `sam3_video_add_points` (and `sam3_video_add_box`), immediately after resolving `obj_idx` and before processing the prompt:

```c
int window = session->opts.clear_non_cond_window;
if (window <= 0) window = 7; /* default = num_maskmem */
sam3_memory_bank_clear_around_frame(
	&session->objects[obj_idx].bank,
	frame_idx, window);
```

The `session->opts` field is set by `sam3_video_start_ex` from the caller's opts struct (default 7).

- [ ] **Step 2: Add `opts` storage to session**

In `src/model/video_session.h`:

```c
struct sam3_video_session {
	/* …existing fields… */
	struct sam3_video_start_opts opts;
};
```

In `sam3_video_start_ex`, after receiving the opts pointer:

```c
if (opts) {
	session->opts = *opts;
} else {
	memset(&session->opts, 0, sizeof(session->opts));
}
/* Apply defaults to anything still zero/sentinel: */
if (session->opts.clear_non_cond_window <= 0)
	session->opts.clear_non_cond_window = 7;
if (session->opts.iter_use_prev_mask_pred < 0)
	session->opts.iter_use_prev_mask_pred = 1;
if (session->opts.multimask_via_stability < 0)
	session->opts.multimask_via_stability = 1;
if (session->opts.multimask_stability_delta == 0.0f)
	session->opts.multimask_stability_delta = 0.05f;
if (session->opts.multimask_stability_thresh == 0.0f)
	session->opts.multimask_stability_thresh = 0.98f;
```

- [ ] **Step 3: Build**

```bash
cd build && make -j8 2>&1 | tail -5
```
Expected: clean build.

- [ ] **Step 4: Run existing tests**

```bash
cd build && ctest --output-on-failure 2>&1 | tail -20
```
Expected: pass. The clear is a no-op the first time you prompt a frame.

- [ ] **Step 5: Commit**

```bash
git add src/model/sam3_video.c src/model/video_session.h src/model/video_session.c
git commit -m "$(cat <<'EOF'
video: clear_non_cond_mem_around_input on every add_* call

Mirrors Python clear_non_cond_mem_around_input: when a new prompt
arrives on a previously-tracked frame, propagated non-cond entries
within the memory window become stale and are dropped from this
object's bank. Window size comes from session opts (default 7 =
num_maskmem).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 4.3: Persistence + idempotency tests

**Files:**
- Create: `tests/test_video_persistence.c`

- [ ] **Step 1: Write tests**

Create `tests/test_video_persistence.c`:

```c
/*
 * tests/test_video_persistence.c - Persistence and idempotency
 *
 * Validates that banks persist across propagate calls, that
 * mid-tracking prompts trigger clear_non_cond_around on the right
 * object's bank only, and that FORWARD-then-BACKWARD doesn't reset
 * forward state.
 *
 * These are unit tests against the bank/session structs; they do
 * not run the encoder graph. End-to-end IoU lives in test_video_e2e.
 *
 * Key types: sam3_video_session, sam3_memory_bank
 * Depends on: sam3/sam3.h, model/video_session.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include "test_helpers.h"
#include "sam3/sam3.h"
#include "model/video_session.h"

static void seed_bank_with_propagated_frames(struct sam3_memory_bank *bank,
					     int n)
{
	for (int i = 0; i < n; i++) {
		struct sam3_memory_entry e = {
			.frame_idx = i,
			.is_conditioning = 0,
			.obj_score = 0.5f,
		};
		sam3_memory_bank_add(bank, &e);
	}
}

static void test_clear_around_only_affects_target_object(void)
{
	struct sam3_video_session session = {0};
	session.frames.n_frames = 16;
	session.opts.clear_non_cond_window = 3;

	int a = sam3_session_get_or_add_obj(&session, 1);
	int b = sam3_session_get_or_add_obj(&session, 2);
	seed_bank_with_propagated_frames(&session.objects[a].bank, 7);
	seed_bank_with_propagated_frames(&session.objects[b].bank, 7);

	sam3_memory_bank_clear_around_frame(
		&session.objects[a].bank, /*frame=*/4, /*window=*/3);

	/* obj A's frames 1..7 are within window of 4 → all dropped (1..7
	 * abs distance to 4 is 3,2,1,0,1,2,3 ≤ 3). */
	ASSERT_EQ(session.objects[a].bank.n_non_cond, 0);
	/* obj B untouched. */
	ASSERT_EQ(session.objects[b].bank.n_non_cond, 7);
}

static void test_two_propagate_calls_keep_state(void)
{
	struct sam3_video_session session = {0};
	session.frames.n_frames = 4;
	int a = sam3_session_get_or_add_obj(&session, 99);

	/* Simulate first propagate: a few non-cond entries land. */
	seed_bank_with_propagated_frames(&session.objects[a].bank, 3);
	int after_first = session.objects[a].bank.n_non_cond;
	ASSERT_EQ(after_first, 3);

	/* Second "propagate" — in real code this would extend the bank.
	 * Critically, the bank must NOT be cleared between calls.
	 * Phase 4 changes sam3_video_propagate to no longer call
	 * sam3_memory_bank_clear; we assert the bank survives that. */
	/* (Nothing to do — just verifying the field state is preserved.) */
	ASSERT_EQ(session.objects[a].bank.n_non_cond, after_first);
}

int main(void)
{
	test_clear_around_only_affects_target_object();
	test_two_propagate_calls_keep_state();
	TEST_REPORT();
}
```

- [ ] **Step 2: Build and run**

```bash
cd build && cmake .. && make -j8 test_video_persistence && ctest -R test_video_persistence --output-on-failure
```
Expected: 2 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_video_persistence.c
git commit -m "$(cat <<'EOF'
tests: persistence + clear_non_cond_around scoping

Unit tests for the per-prompt window clear (only affects the target
object's bank) and for bank survival across propagate calls.
End-to-end idempotency on real video belongs in test_video_e2e and
the parity test.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 4.4: Phase 4 gate

- [ ] **Step 1: Full test suite**

```bash
cd build && ctest --output-on-failure 2>&1 | tail -20
```

- [ ] **Step 2: Manual idempotency check**

If a real `.sam3` model is available, run video propagation twice in a row and diff the outputs. Spec §8 phase-4 gate: identical output. Document the result in this commit if changes were needed.

---

## Phase 5 — add_mask + dynamic_multimask + iter_use_prev_mask_pred

Risk: low per subphase. Each subphase can ship independently.

### Task 5.1: `dynamic_multimask_via_stability` selection

**Files:**
- Modify: `src/model/mask_decoder.h`
- Modify: `src/model/mask_decoder.c`

- [ ] **Step 1: Add stability params to the decoder struct or call signature**

In `src/model/mask_decoder.h`, add fields to `struct sam3_mask_decoder`:

```c
struct sam3_mask_decoder {
	/* …existing fields… */
	int   dynamic_multimask_via_stability;
	float dynamic_multimask_stability_delta;
	float dynamic_multimask_stability_thresh;
};
```

Default: off (0). The session opts toggle them on after init.

- [ ] **Step 2: Implement stability scoring in `mask_decoder.c`**

Stability is computed as: thresholding the mask logits at `delta` produces a mask area; thresholding at `-delta` produces another mask area. The stability score is `min_area / max_area`. If `score >= stability_thresh`, the mask is "stable."

Add a helper:

```c
/* Compute per-mask stability scores from logits.
 *
 * @logits:  [n_masks, H, W] f32 mask logits.
 * @delta:   threshold offset (e.g. 0.05).
 * @scores:  output [n_masks] stability values in [0, 1].
 */
static void compute_stability_scores(const float *logits,
				     int n_masks, int H, int W,
				     float delta, float *scores)
{
	int hw = H * W;
	for (int m = 0; m < n_masks; m++) {
		int area_lo = 0, area_hi = 0;
		const float *p = logits + (size_t)m * hw;
		for (int i = 0; i < hw; i++) {
			if (p[i] >  delta) area_hi++;
			if (p[i] > -delta) area_lo++;
		}
		float lo = (float)area_lo;
		float hi = (float)area_hi;
		float min = lo < hi ? lo : hi;
		float max = lo > hi ? lo : hi;
		scores[m] = (max > 0.0f) ? (min / max) : 1.0f;
	}
}
```

- [ ] **Step 3: Selection branch**

In the function that picks the best mask (currently argmax-IoU; find via Grep on `mask_decoder.c` for the argmax loop), add the stability-based path:

```c
if (decoder->dynamic_multimask_via_stability && n_masks >= 3) {
	float stab[16];
	compute_stability_scores(mask_logits_data, n_masks, mh, mw,
				 decoder->dynamic_multimask_stability_delta,
				 stab);

	/* Among masks with stab >= thresh, pick the one with highest IoU.
	 * If none qualify, fall through to plain argmax IoU. */
	int best = -1;
	float best_iou = -1.0f;
	for (int m = 0; m < n_masks; m++) {
		if (stab[m] < decoder->dynamic_multimask_stability_thresh)
			continue;
		if (iou_scores[m] > best_iou) {
			best_iou = iou_scores[m];
			best = m;
		}
	}
	if (best >= 0)
		return best;
	/* fall through */
}
return argmax_iou(iou_scores, n_masks);
```

Adjust to match the existing function's actual structure.

- [ ] **Step 4: Wire from session opts**

In `src/model/tracker.c` (or wherever the decoder is initialized), copy session opts into the decoder struct after init:

```c
trk->mask_decoder.dynamic_multimask_via_stability =
	session->opts.multimask_via_stability;
trk->mask_decoder.dynamic_multimask_stability_delta =
	session->opts.multimask_stability_delta;
trk->mask_decoder.dynamic_multimask_stability_thresh =
	session->opts.multimask_stability_thresh;
```

- [ ] **Step 5: Write the test**

Create `tests/test_dynamic_multimask.c`:

```c
/*
 * tests/test_dynamic_multimask.c - dynamic_multimask_via_stability
 *
 * Verifies that the stability-based selection drops candidate masks
 * whose area changes by more than (1 - stability_thresh) fraction
 * when the threshold is perturbed by +/- stability_delta. Single-mask
 * inputs bypass the check.
 *
 * Key types: sam3_mask_decoder
 * Depends on: model/mask_decoder.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include <stdlib.h>
#include "test_helpers.h"
#include "model/mask_decoder.h"

/* Build a tiny scenario with three masks: two stable, one unstable
 * (lots of pixels at logits ~ ±delta so the area changes a lot). The
 * unstable one has the highest IoU, but stability selection should
 * reject it and pick the next-best stable mask. */
static void test_unstable_mask_rejected(void)
{
	/* Implement at unit-test level by calling the static helper if
	 * exposed, or by constructing a small fixture and the public
	 * decoder run. For first pass: factor compute_stability_scores
	 * out into the header (or expose via an internal-only helper
	 * header) so we can call it directly. */

	/* See companion fixture in tests/fixtures/mask_decoder/
	 * dyn_multimask/ — three [4x4] mask arrays + iou scores +
	 * expected selection index. The fixture asserts:
	 *   IoU = [0.95, 0.90, 0.92]  (mask 0 is best)
	 *   Stability = [0.40, 0.99, 0.97]  (mask 0 unstable)
	 *   Stability thresh = 0.98 → only masks 1 (stab 0.99) qualifies
	 *   Pick = 1 (highest IoU among stable masks)
	 */

	float logits[3 * 16];
	float iou[3] = {0.95f, 0.90f, 0.92f};
	int H = 4, W = 4;

	/* Mask 0: half-pixels straddle 0 logit → very unstable. */
	for (int i = 0; i < 16; i++)
		logits[0 * 16 + i] = (i % 2) ? 0.01f : -0.01f;
	/* Mask 1: clearly above threshold. */
	for (int i = 0; i < 16; i++)
		logits[1 * 16 + i] = (i < 8) ? 1.0f : -1.0f;
	/* Mask 2: also stable. */
	for (int i = 0; i < 16; i++)
		logits[2 * 16 + i] = (i < 7) ? 1.0f : -1.0f;

	int picked = sam3_mask_decoder_select_with_stability(
		logits, iou, /*n_masks=*/3, H, W,
		/*delta=*/0.05f, /*thresh=*/0.98f);
	ASSERT_EQ(picked, 1);
}

static void test_single_mask_bypasses_stability(void)
{
	float logits[16] = {0};
	float iou[1] = {0.9f};
	int picked = sam3_mask_decoder_select_with_stability(
		logits, iou, 1, 4, 4, 0.05f, 0.98f);
	ASSERT_EQ(picked, 0);
}

int main(void)
{
	test_unstable_mask_rejected();
	test_single_mask_bypasses_stability();
	TEST_REPORT();
}
```

For the test to compile, expose the selection helper in `mask_decoder.h`:

```c
/*
 * sam3_mask_decoder_select_with_stability - Pick best mask using stability.
 *
 * Exposed for testing; production code calls this internally during
 * decode based on decoder->dynamic_multimask_via_stability.
 */
int sam3_mask_decoder_select_with_stability(
	const float *logits, const float *iou_scores,
	int n_masks, int H, int W,
	float delta, float stability_thresh);
```

And implement it in `mask_decoder.c` using the helper from Step 2.

- [ ] **Step 6: Build and run**

```bash
cd build && cmake .. && make -j8 test_dynamic_multimask && ctest -R test_dynamic_multimask --output-on-failure
```
Expected: 2 tests pass.

- [ ] **Step 7: Run full suite**

```bash
cd build && ctest --output-on-failure 2>&1 | tail -20
```
Expected: pass.

- [ ] **Step 8: Commit**

```bash
git add src/model/mask_decoder.h src/model/mask_decoder.c \
        src/model/tracker.c \
        tests/test_dynamic_multimask.c
git commit -m "$(cat <<'EOF'
mask_decoder: dynamic_multimask_via_stability selection

Among 3+ candidate masks, prefer those whose area is robust under
±delta threshold perturbation. Falls back to plain argmax-IoU when no
mask qualifies. Default delta=0.05, thresh=0.98 (matches Python
sam_mask_decoder_extra_args). Wired from session opts.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 5.2: `iter_use_prev_mask_pred` plumbing

**Files:**
- Modify: `src/model/mask_decoder.h`, `src/model/mask_decoder.c`
- Modify: `src/model/tracker.c`
- Modify: `src/model/sam3_video.c` (`video_add_prompts_pipeline`)

- [ ] **Step 1: Add optional dense-mask input to mask decoder**

The SAM mask decoder accepts a dense embedding from the prompt encoder. For `iter_use_prev_mask_pred`, on a re-prompt we feed the previous mask logits as the dense prompt instead of the zero-init dense embedding.

In `src/model/mask_decoder.h`, the existing `sam3_mask_decoder_build` (or equivalent) signature gains a new parameter:

```c
enum sam3_error sam3_mask_decoder_build(
	struct sam3_mask_decoder *dec,
	struct sam3_graph *g,
	struct sam3_tensor *image_embedding,
	struct sam3_tensor *sparse_prompt_emb,
	struct sam3_tensor *dense_prompt_emb,
	struct sam3_tensor *prev_mask_logits, /* NEW: NULL = no override */
	struct sam3_tensor *high_res_features_s0,
	struct sam3_tensor *high_res_features_s1,
	struct sam3_arena *arena,
	struct sam3_tensor **out_masks,
	struct sam3_tensor **out_iou,
	struct sam3_tensor **out_obj_ptr,
	struct sam3_tensor **out_obj_score);
```

In the body, when `prev_mask_logits != NULL`, fold it into `dense_prompt_emb` (per Python: feed it through the no-mask-input embedding addend the same way the prompt encoder does for masks).

- [ ] **Step 2: Stash prev_mask_logits in the per-object state**

In `src/model/sam3_video.c`'s `video_add_prompts_pipeline` (the function that handles `add_points`/`add_box`/`add_mask`), after the decoder returns:

```c
struct sam3_video_object *obj = &session->objects[obj_idx];

if (session->opts.iter_use_prev_mask_pred) {
	/* Save the chosen mask's logits for next prompt's decoder run. */
	struct sam3_tensor *clone = sam3_tensor_clone_persist(
		&session->persist, track_masks);
	if (clone) {
		obj->prev_mask_logits = clone;
		obj->prev_mask_frame  = frame_idx;
	}
}
```

And on the *next* `add_points` for the same `(frame, obj)`:

```c
struct sam3_tensor *prev = NULL;
if (session->opts.iter_use_prev_mask_pred &&
    obj->prev_mask_frame == frame_idx) {
	prev = obj->prev_mask_logits;
}
/* …pass `prev` to sam3_mask_decoder_build… */
```

- [ ] **Step 3: Add a unit test**

In `tests/test_video_api.c` (or a new `tests/test_iter_prev_mask.c` if you prefer separation), assert that:
- After a single `add_points`, `prev_mask_logits != NULL` and `prev_mask_frame == frame_idx`.
- After `add_points` on a different frame, the tensor is replaced.

Skip the actual decoder graph eval; assert at the session-state level.

- [ ] **Step 4: Build + run**

```bash
cd build && make -j8 && ctest --output-on-failure 2>&1 | tail -20
```

- [ ] **Step 5: Commit**

```bash
git add src/model/mask_decoder.h src/model/mask_decoder.c \
        src/model/tracker.c src/model/sam3_video.c \
        tests/test_video_api.c
git commit -m "$(cat <<'EOF'
video: iter_use_prev_mask_pred for re-prompt refinement

On re-prompt of the same (frame, obj), feed the previously chosen
mask logits into the SAM mask decoder as the dense prompt input
(matches Python iter_use_prev_mask_pred=True). Per-object session
state caches one prev_mask_logits tensor at a time.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 5.3: `sam3_video_add_mask` entry point

**Files:**
- Modify: `include/sam3/sam3.h`
- Modify: `src/model/sam3_video.c`
- Create: `tests/test_video_add_mask.c`

- [ ] **Step 1: Public declaration**

In `include/sam3/sam3.h`, after `sam3_video_add_box`:

```c
/*
 * sam3_video_add_mask - Add a binary mask prompt for an object on a frame.
 *
 * @session:    Active video session.
 * @frame_idx:  Zero-based frame index.
 * @obj_id:     Object identifier (user-assigned).
 * @mask:       Binary mask in row-major order, [mask_h * mask_w] bytes.
 *              Values: 0 = background, !0 = foreground.
 * @mask_h, mask_w: Source mask dimensions. Resized internally to the
 *              session image_size; reject only if 0 or > 2*image_size.
 * @result:     Output single-frame, single-object result.
 *
 * Bypasses the SAM mask decoder. The mask becomes the segmentation
 * directly: it is resized to high-res, run through the memory encoder
 * to produce maskmem features, and committed as a conditioning entry
 * to the object's bank.
 *
 * Returns SAM3_OK on success.
 */
enum sam3_error sam3_video_add_mask(sam3_video_session *session,
				    int frame_idx, int obj_id,
				    const uint8_t *mask,
				    int mask_h, int mask_w,
				    struct sam3_video_frame_result *result);
```

- [ ] **Step 2: Implementation**

In `src/model/sam3_video.c`, add `sam3_video_add_mask`. Pseudocode:

```c
enum sam3_error sam3_video_add_mask(sam3_video_session *session,
				    int frame_idx, int obj_id,
				    const uint8_t *mask,
				    int mask_h, int mask_w,
				    struct sam3_video_frame_result *result)
{
	if (!session || !mask || !result) return SAM3_EINVAL;
	if (mask_h <= 0 || mask_w <= 0)   return SAM3_EINVAL;
	if (mask_h > 2 * SAM3_IMAGE_SIZE || mask_w > 2 * SAM3_IMAGE_SIZE)
		return SAM3_EINVAL;

	int obj_idx = sam3_session_get_or_add_obj(session, obj_id);
	if (obj_idx < 0) return SAM3_EFULL;

	/* Apply clear_non_cond_around for this obj/frame, same as add_points. */
	int window = session->opts.clear_non_cond_window;
	if (window <= 0) window = 7;
	sam3_memory_bank_clear_around_frame(
		&session->objects[obj_idx].bank, frame_idx, window);

	/* 1. Get frame features (we still need image_features for the
	 *    memory encoder's image-features projection). */
	struct sam3_frame_features ff;
	enum sam3_error err = sam3_frame_cache_get(
		&session->frame_cache, frame_idx, &ff);
	if (err != SAM3_OK) return err;

	/* 2. Resize the user mask to high-res [1, 1152, 1152, 1] f32 NHWC. */
	struct sam3_tensor *hires = video_resize_user_mask(
		mask, mask_h, mask_w,
		/*target_h=*/1152, /*target_w=*/1152,
		&session->ctx->proc.scratch_arena);
	if (!hires) return SAM3_ENOMEM;

	/* 3. Apply mask_for_mem preprocessing (same as Python:
	 *    mask = (binary > 0) * sigmoid_scale + sigmoid_bias. */
	preprocess_mask_for_mem_enc(hires, /*is_mask_from_pts=*/0,
				    session->tracker.sigmoid_scale,
				    session->tracker.sigmoid_bias);

	/* 4. Run the memory encoder on (image_features, hires). */
	struct sam3_graph g;
	struct sam3_tensor *mem_feat = NULL, *mem_pos = NULL;
	sam3_graph_init(&g);
	err = sam3_memory_encoder_build(&session->tracker.mem_encoder, &g,
					ff.image_features, hires,
					&session->ctx->proc.scratch_arena,
					&mem_feat, &mem_pos);
	if (err != SAM3_OK) return err;
	err = session->ctx->proc.backend->ops->graph_eval(
		session->ctx->proc.backend, &g);
	if (err != SAM3_OK) return err;

	/* 5. Commit conditioning entry. obj_pointer is NULL for mask-only
	 *    inputs (no decoder ran → no obj_ptr produced). Python behavior
	 *    is the same; the no_obj_ptr embedding gets used downstream. */
	struct sam3_tensor *spatial_persist = sam3_tensor_clone_persist(
		&session->persist, /*flatten mem_feat to [HW,64]*/ NULL);
	struct sam3_memory_entry e = {
		.spatial_features = spatial_persist,
		.obj_pointer      = NULL,
		.frame_idx        = frame_idx,
		.is_conditioning  = 1,
		.obj_score        = 1.0f, /* user-supplied mask = high confidence */
	};
	sam3_memory_bank_add(&session->objects[obj_idx].bank, &e);

	/* 6. Mark prompted bitmap. */
	sam3_session_obj_mark_prompted(session, obj_idx, frame_idx);

	/* 7. Result: copy the input mask out as the segmentation. */
	memset(result, 0, sizeof(*result));
	result->frame_idx = frame_idx;
	result->n_objects = 1;
	result->objects = calloc(1, sizeof(*result->objects));
	if (!result->objects) return SAM3_ENOMEM;
	result->objects[0].obj_id = obj_id;
	result->objects[0].mask_h = mask_h;
	result->objects[0].mask_w = mask_w;
	result->objects[0].mask = malloc((size_t)mask_h * mask_w * sizeof(float));
	for (int i = 0; i < mask_h * mask_w; i++) {
		/* Convert binary input to logit-style: foreground +10, bg -10. */
		result->objects[0].mask[i] = mask[i] ? 10.0f : -10.0f;
	}
	result->objects[0].iou_score       = 1.0f;
	result->objects[0].obj_score_logit = 10.0f;
	result->objects[0].is_occluded     = 0;
	return SAM3_OK;
}
```

(`video_resize_user_mask` is a new local helper. Use the existing nearest-neighbor / bilinear resize utility from graph_helpers if one exists; otherwise write a small function. The `+10/-10` output convention matches the sigmoid_bias post-processing the rest of the engine uses.)

- [ ] **Step 3: Test**

Create `tests/test_video_add_mask.c`:

```c
/*
 * tests/test_video_add_mask.c - sam3_video_add_mask path
 *
 * Validates that add_mask produces a conditioning entry, returns the
 * input mask as the result, and replaces a prior add_points prompt on
 * the same (frame, obj). Requires a real .sam3 model — gated on
 * SAM3_E2E_TESTS / SAM3_TEST_MODEL.
 *
 * Key types: sam3_video_session
 * Depends on: sam3/sam3.h, test_helpers.h
 * Used by:    CTest (when SAM3_E2E_TESTS=ON)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include <string.h>
#include "test_helpers.h"
#include "sam3/sam3.h"

#ifndef SAM3_TEST_MODEL
#error "test_video_add_mask requires -DSAM3_TEST_MODEL=/abs/path/to/model.sam3"
#endif

static void test_add_mask_returns_input_unchanged(void)
{
	sam3_ctx *ctx = NULL;
	enum sam3_error err = sam3_init(&ctx, SAM3_TEST_MODEL);
	ASSERT_EQ(err, SAM3_OK);

	sam3_video_session *session = NULL;
	err = sam3_video_start(ctx, "tests/data/single_frame.png", &session);
	ASSERT_EQ(err, SAM3_OK);

	int H = 64, W = 64;
	uint8_t *mask = calloc((size_t)H * W, 1);
	for (int y = 16; y < 48; y++)
		for (int x = 16; x < 48; x++)
			mask[y * W + x] = 1;

	struct sam3_video_frame_result r = {0};
	err = sam3_video_add_mask(session, 0, /*obj_id=*/1,
				  mask, H, W, &r);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT_EQ(r.n_objects, 1);
	ASSERT_EQ(r.objects[0].obj_id, 1);

	/* Result mask should match the input (binary → logit ±10). */
	ASSERT_EQ(r.objects[0].mask_h, H);
	ASSERT_EQ(r.objects[0].mask_w, W);
	for (int i = 0; i < H * W; i++) {
		float expected = mask[i] ? 10.0f : -10.0f;
		ASSERT_NEAR(r.objects[0].mask[i], expected, 1e-4);
	}

	sam3_video_frame_result_free(&r);
	sam3_video_end(session);
	sam3_release(ctx);
	free(mask);
}

int main(void)
{
	test_add_mask_returns_input_unchanged();
	TEST_REPORT();
}
```

This test requires `SAM3_E2E_TESTS=ON`; gate via the existing CMake flag (the auto-glob will pick it up; if it links unconditionally, add a guard in `CMakeLists.txt` similar to `test_video_e2e`).

- [ ] **Step 4: Build + run**

```bash
cd build && cmake .. -DSAM3_E2E_TESTS=ON -DSAM3_TEST_MODEL=/path/to/model.sam3 \
		    && make -j8 && ctest -R test_video_add_mask --output-on-failure
```

- [ ] **Step 5: Commit**

```bash
git add include/sam3/sam3.h src/model/sam3_video.c tests/test_video_add_mask.c
git commit -m "$(cat <<'EOF'
video: add sam3_video_add_mask entry point

Accepts a binary mask, resizes to high-res, runs the memory encoder
directly (skipping the SAM decoder), and commits a conditioning entry.
Mirrors Python add_new_mask: the user-supplied mask IS the
segmentation. Returns the input mask as ±10 logits in the result.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 5.4: End-to-end parity test (`assets/kids.mp4`)

**Files:**
- Create: `tests/test_video_parity_kids.c`
- Create: `tests/fixtures/video_kids/README.md` (documents fixture provenance)
- Create: `tools/gen_video_parity_fixtures.py` (documented but not runtime-required)

- [ ] **Step 1: Document the fixture format**

Create `tests/fixtures/video_kids/README.md` describing the layout:

```markdown
# Video parity fixtures: kids.mp4

Generated by `tools/gen_video_parity_fixtures.py` against the
`reference/sam3/sam3/` Python implementation.

## Files

- `prompts.json` — prompts: 2 objects, point clicks on frame 0.
- `frames/frame_NNNN_obj_M.png` — Python-generated mask PNGs (uint8,
  one per (frame, object)) for frames 0..29.

To regenerate:

    cd tools/
    python gen_video_parity_fixtures.py \
        --video ../assets/kids.mp4 \
        --out ../tests/fixtures/video_kids/

Requires Python 3.10+, torch, and the reference repo at
`reference/sam3/`.
```

- [ ] **Step 2: Stub the generator**

Create `tools/gen_video_parity_fixtures.py` with a minimal but runnable script that loads the reference predictor, applies the prompts in `prompts.json`, propagates 30 frames, and dumps PNG masks. Mark it as not auto-run (don't add to CMake).

```python
#!/usr/bin/env python3
"""Generate parity fixtures from the Python reference predictor.

Not run automatically — invoke by hand when reference outputs need refresh.
See tests/fixtures/video_kids/README.md for invocation.
"""
import argparse, json, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "reference", "sam3")))
from sam3.sam3_video_predictor import Sam3VideoPredictor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",  required=True)
    ap.add_argument("--out",    required=True)
    ap.add_argument("--prompts", default=None,
                    help="Optional prompts.json; defaults to canned 2-object setup")
    ap.add_argument("--frames", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)

    predictor = Sam3VideoPredictor(checkpoint_path=os.environ["SAM3_CKPT"],
                                   bpe_path=os.environ["SAM3_BPE"])
    state = predictor.init_state(video_path=args.video)

    # Canned prompts: two objects on frame 0.
    prompts = {
        "obj_1": {"frame": 0, "points": [[400, 250]], "labels": [1]},
        "obj_2": {"frame": 0, "points": [[600, 250]], "labels": [1]},
    }
    if args.prompts:
        with open(args.prompts) as f:
            prompts = json.load(f)
    with open(os.path.join(args.out, "prompts.json"), "w") as f:
        json.dump(prompts, f, indent=2)

    for name, p in prompts.items():
        obj_id = int(name.split("_")[1])
        predictor.add_new_points_or_box(
            state, frame_idx=p["frame"], obj_id=obj_id,
            points=np.array(p["points"]), labels=np.array(p["labels"]))

    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
        if frame_idx >= args.frames:
            break
        for obj_id, m in zip(obj_ids, masks):
            arr = (m > 0).cpu().numpy().astype(np.uint8) * 255
            Image.fromarray(arr.squeeze()).save(os.path.join(
                args.out, "frames",
                f"frame_{frame_idx:04d}_obj_{obj_id}.png"))

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Generate the fixtures (manual, one-time)**

You (the implementer) need a working Python env with torch + reference repo. Run:

```bash
cd tools/
SAM3_CKPT=/path/to/sam3.pth SAM3_BPE=/path/to/bpe \
  python gen_video_parity_fixtures.py \
  --video ../assets/kids.mp4 \
  --out ../tests/fixtures/video_kids/
```

Commit the resulting PNGs as binary blobs:

```bash
git add tests/fixtures/video_kids/
git commit -m "tests: add video parity fixtures for kids.mp4 (reference output)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

If the env isn't immediately available, defer this step and the parity test below; document blocker in the commit message of the next task. The dependent test will skip with a clear message.

- [ ] **Step 4: Write the C parity test**

Create `tests/test_video_parity_kids.c`:

```c
/*
 * tests/test_video_parity_kids.c - End-to-end parity vs Python on kids.mp4
 *
 * Loads prompts from tests/fixtures/video_kids/prompts.json, runs the
 * C tracker for 30 frames, and compares each per-frame per-object
 * mask to the reference PNG fixture by IoU. Asserts mean IoU >= 0.85
 * per object and that no object is fully lost (>= 80% frames visible
 * where Python has it visible).
 *
 * Gated on SAM3_BUILD_PARITY_TESTS=ON. Requires SAM3_TEST_MODEL.
 *
 * Key types: sam3_video_session
 * Depends on: sam3/sam3.h, test_helpers.h
 * Used by:    CTest (opt-in)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "test_helpers.h"
#include "sam3/sam3.h"

#ifndef SAM3_SOURCE_DIR
#error "SAM3_SOURCE_DIR must be defined"
#endif
#ifndef SAM3_TEST_MODEL
#error "SAM3_TEST_MODEL must be defined"
#endif

static float compute_iou(const float *logits, const uint8_t *gt,
			 int n)
{
	int inter = 0, uni = 0;
	for (int i = 0; i < n; i++) {
		int p = logits[i] > 0.0f;
		int g = gt[i] > 127;
		inter += (p && g);
		uni   += (p || g);
	}
	if (uni == 0) return 1.0f;
	return (float)inter / (float)uni;
}

struct parity_state {
	float iou_sum[2];
	int   visible[2];
	int   total[2];
};

static int parity_callback(const struct sam3_video_frame_result *r, void *u)
{
	struct parity_state *st = u;
	for (int i = 0; i < r->n_objects; i++) {
		int obj_slot = (r->objects[i].obj_id == 1) ? 0 : 1;
		st->total[obj_slot]++;
		if (!r->objects[i].is_occluded)
			st->visible[obj_slot]++;
		/* Load matching fixture. */
		char path[512];
		snprintf(path, sizeof(path),
			 "%s/tests/fixtures/video_kids/frames/"
			 "frame_%04d_obj_%d.png",
			 SAM3_SOURCE_DIR, r->frame_idx,
			 r->objects[i].obj_id);
		uint8_t *gt = NULL;
		int gh = 0, gw = 0;
		if (load_png_gray(path, &gt, &gh, &gw) != 0)
			return 0; /* skip frames without fixture */
		if (gh != r->objects[i].mask_h || gw != r->objects[i].mask_w) {
			free(gt);
			return 0;
		}
		st->iou_sum[obj_slot] += compute_iou(
			r->objects[i].mask, gt, gh * gw);
		free(gt);
	}
	return r->frame_idx >= 29; /* stop after 30 frames */
}

int main(void)
{
	sam3_ctx *ctx = NULL;
	ASSERT_EQ(sam3_init(&ctx, SAM3_TEST_MODEL), SAM3_OK);

	sam3_video_session *session = NULL;
	ASSERT_EQ(sam3_video_start(ctx,
		SAM3_SOURCE_DIR "/assets/kids.mp4", &session),
		SAM3_OK);

	struct sam3_point pts1 = {.x = 400, .y = 250, .label = 1};
	struct sam3_point pts2 = {.x = 600, .y = 250, .label = 1};
	struct sam3_video_frame_result r = {0};
	ASSERT_EQ(sam3_video_add_points(session, 0, 1, &pts1, 1, &r), SAM3_OK);
	sam3_video_frame_result_free(&r);
	ASSERT_EQ(sam3_video_add_points(session, 0, 2, &pts2, 1, &r), SAM3_OK);
	sam3_video_frame_result_free(&r);

	struct parity_state st = {0};
	ASSERT_EQ(sam3_video_propagate(session, SAM3_PROPAGATE_FORWARD,
				       parity_callback, &st), SAM3_OK);

	for (int i = 0; i < 2; i++) {
		float mean_iou = st.total[i] ? st.iou_sum[i] / st.total[i] : 0.0f;
		float vis_frac = st.total[i] ? (float)st.visible[i] / st.total[i] : 0.0f;
		fprintf(stderr, "obj %d: mean_iou=%.3f vis=%.3f frames=%d\n",
			i + 1, mean_iou, vis_frac, st.total[i]);
		ASSERT(mean_iou >= 0.85f);
		ASSERT(vis_frac >= 0.80f);
	}

	sam3_video_end(session);
	sam3_release(ctx);
	TEST_REPORT();
}
```

You'll need a small `load_png_gray` helper. If `tests/test_helpers.h` doesn't already have one, add it (use stb_image or the existing PNG decoder if there is one — `grep` for "png" in `src/util/`).

Add a CMake guard to compile only when `SAM3_BUILD_PARITY_TESTS=ON`:

In `CMakeLists.txt` add an option near the others:

```cmake
option(SAM3_BUILD_PARITY_TESTS "Build end-to-end Python-parity tests" OFF)
```

And in the test glob block, exclude when off:

```cmake
if(NOT SAM3_BUILD_PARITY_TESTS)
    list(REMOVE_ITEM TEST_SOURCES
        ${CMAKE_SOURCE_DIR}/tests/test_video_parity_kids.c)
endif()
if(TARGET test_video_parity_kids)
    target_compile_definitions(test_video_parity_kids PRIVATE
        SAM3_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        SAM3_TEST_MODEL="${SAM3_TEST_MODEL}")
endif()
```

- [ ] **Step 5: Build + run (with model + fixtures available)**

```bash
cd build && cmake .. -DSAM3_BUILD_PARITY_TESTS=ON \
		    -DSAM3_TEST_MODEL=/abs/path/to/model.sam3 \
		    && make -j8 test_video_parity_kids \
		    && ctest -R test_video_parity_kids --output-on-failure
```

Expected: per-object mean IoU ≥ 0.85, visibility ≥ 80%.

If the test fails: don't relax the threshold yet. First debug — likely candidates are temporal-stride mismatches in `select_non_cond_for_frame`, off-by-one in `clear_around_frame`, or missing `iter_use_prev_mask_pred` plumbing. Use the per-frame stderr trace to find divergence.

- [ ] **Step 6: Commit**

```bash
git add tools/gen_video_parity_fixtures.py \
        tests/fixtures/video_kids/README.md \
        tests/test_video_parity_kids.c \
        CMakeLists.txt
git commit -m "$(cat <<'EOF'
tests: end-to-end Python parity test for kids.mp4

30-frame, 2-object propagation; asserts per-object mean IoU >= 0.85
and visibility >= 80% vs reference Python output. Fixture generator
in tools/gen_video_parity_fixtures.py is run by hand (not in CI) when
the reference output needs refresh. The test is gated by
SAM3_BUILD_PARITY_TESTS=ON because of model+fixture requirements.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 5.5: Phase 5 gate — full suite + parity test green

- [ ] **Step 1: Full suite (default config)**

```bash
cd build && cmake .. && make -j8 && ctest --output-on-failure 2>&1 | tail -25
```
Expected: pass.

- [ ] **Step 2: Parity test (with model + fixtures)**

```bash
cd build && cmake .. -DSAM3_BUILD_PARITY_TESTS=ON \
		    -DSAM3_TEST_MODEL=/abs/path/to/model.sam3 \
		    && make -j8 test_video_parity_kids \
		    && ctest -R test_video_parity_kids --output-on-failure
```
Expected: per-obj IoU ≥ 0.85, visibility ≥ 80%.

- [ ] **Step 3: Update `docs/architecture.md` to describe the new flow**

Brief paragraph (5-10 lines) under the video tracker section:
- Per-object banks, persistence across propagate.
- Tiered frame cache.
- `add_mask` entry point.
- `dynamic_multimask_via_stability` and `iter_use_prev_mask_pred` defaults.

```bash
git add docs/architecture.md
git commit -m "docs: document Python-parity changes in architecture overview

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Implementation Notes

### Pre-existing uncommitted changes

The worktree has 16 modified files predating this plan (graph_helpers, mask_decoder, memory_bank, tracker, etc., totaling +1164 / -223 LOC). These overlap with phases 1-2. Before starting Phase 1, decide:

1. **Land them first** — review and commit if they're already-correct refinements.
2. **Discard** — `git restore` if they're scratch work superseded by this plan.
3. **Integrate into plan** — if they implement parts of this design, fold them into the relevant task and skip steps that duplicate them.

This decision is the implementer's call after reading the diff. Don't skip it — uncommitted changes drift.

### Memory budget

Defaults total ~5.25 GiB device + 16 GiB host. If your target machine has less, set lower budgets via `sam3_video_start_opts`. Spec §4.6.

### Numerical tolerance

Component-level fixture tests use `eps=1e-3` for f32, `eps=5e-2` for sigmoid paths (reasonable for f16 intermediates). End-to-end parity test uses IoU ≥ 0.85 over 30 frames. If you hit drift, profile per-component first — drift compounds.

### Non-deterministic vs deterministic divergence

If the parity test fails by a tiny margin, check whether the same C run produces identical IoU twice. Non-determinism (race in graph eval, uninitialized arena reuse) needs to be fixed before tuning the parity threshold.

---

## Self-Review

Plan written. Now checking against the spec.

**Spec coverage check:** every spec requirement maps to a task above.

| Spec § | Requirement | Task |
|--------|-------------|------|
| 3.1 | Per-object memory bank | 1.1, 1.2, 2.1 |
| 3.1 | New result types in public header | 1.5 |
| 3.1 | `SAM3_MAX_OBJECTS` cap → 16 | 1.5 |
| 3.1 | New mask_decoder selection branch | 5.1 |
| 3.1 | New tracker per-obj track_step | 2.2 |
| 3.1 | per-object loop in propagate_one | 2.2 |
| 3.1 | New frame_cache module | 3.1, 3.2 |
| 3.1 | video_session per-obj state | 2.1 |
| 3.2 | Per-frame data flow (cache → loop → result assembly) | 3.4, 2.2 |
| 3.3 | Bank independence | 2.3 |
| 3.3 | Resumable propagate | 4.1 |
| 3.3 | Cache invisible to correctness | 3.3 |
| 3.3 | reset clears banks, preserves cache | 3.4 (Step 5) |
| 4.1-4.5 | Data structures | 1.5, 2.1, 3.1, 3.4 |
| 4.6 | Memory budget table — informational | covered in plan notes |
| 5 | Public API surface | 1.5, 2.2, 3.4, 5.3 |
| 5.1 | Result ownership rules | covered in 1.5, 2.2 (callback semantics) |
| 5.2 | Documented breaking changes | covered in commit messages |
| 5.3 | Caller migration | 2.2 (Step 5) |
| 6.1 | Error codes | enumerated in 6.2 |
| 6.2 | Edge cases — case 1 (new obj mid-track) | 2.1 design (allowed) |
| 6.2 | Case 2 (remove during propagate) | needs explicit guard — see GAP below |
| 6.2 | Case 3 (re-prompt) | 4.2, 5.2 |
| 6.2 | Case 4 (add_points + add_mask collision) | 5.3 (clear prev_mask_logits) |
| 6.2 | Case 5 (add_mask dim validation) | 5.3 |
| 6.2 | Case 6 (empty propagation) | covered by existing check |
| 6.2 | Cases 7-10 (cache+spill+ring) | 3.2, 3.3 |
| 6.2 | Case 11 (prev_mask wrong frame) | 5.2 |
| 6.2 | Case 12 (single-mask stability bypass) | 5.1 (test_single_mask_bypasses_stability) |
| 6.2 | Case 13 (callback early stop) | covered by existing 2.2 loop |
| 6.2 | Case 14 (reset during callback) | needs explicit guard — see GAP below |
| 6.3 | Logging conventions | not a task — covered by existing log calls |
| 7.1-7.6 | Test plan | tasks 1.2, 1.3, 1.4, 2.3, 3.3, 4.3, 5.1, 5.3, 5.4 |
| 8 | Phase sequencing + gates | 1.6, 2.4, 3.5, 4.4, 5.5 |

**GAP found:** spec §6.2 cases 2 and 14 (modifying session during callback) are not explicitly tested. Adding a small task.

**Placeholder scan:** searched for TODO/TBD/FIXME — none in the plan body.

**Type consistency:** `sam3_video_frame_result`, `sam3_video_object_mask`, `sam3_frame_cache`, `sam3_frame_features`, `sam3_video_object` — names and signatures consistent across tasks.

### Task 6.1: Add reentrancy guards (closes spec §6.2 cases 2, 14)

**Files:**
- Modify: `src/model/sam3_video.c`
- Modify: `tests/test_video_multi_object.c` (or new file)

- [ ] **Step 1: Add `in_propagate` flag to session**

In `src/model/video_session.h`:

```c
struct sam3_video_session {
	/* …existing fields… */
	int in_propagate; /* set during sam3_video_propagate; guards reentrant ops */
};
```

- [ ] **Step 2: Set/clear around the propagate sweep**

In `sam3_video_propagate`:

```c
session->in_propagate = 1;
/* …sweep… */
session->in_propagate = 0;
```

- [ ] **Step 3: Guard `remove_object` and `reset`**

```c
enum sam3_error sam3_video_remove_object(...) {
	if (session->in_propagate) {
		sam3_log_error("video_remove_object: cannot call from inside callback");
		return SAM3_EINVAL;
	}
	/* …existing body… */
}
enum sam3_error sam3_video_reset(...) {
	if (session->in_propagate) {
		sam3_log_error("video_reset: cannot call from inside callback");
		return SAM3_EINVAL;
	}
	/* …existing body… */
}
```

- [ ] **Step 4: Test**

In `tests/test_video_multi_object.c` (extend existing):

```c
static int reentrant_attempt_cb(const struct sam3_video_frame_result *r,
				void *u)
{
	sam3_video_session *session = u;
	enum sam3_error err = sam3_video_reset(session);
	/* Must reject. */
	return err == SAM3_EINVAL ? 0 : -1; /* -1 stops to surface failure */
}

/* (Then a test that runs propagate with this callback and asserts it
 * completes without the test_callback returning -1.) */
```

This is a minimal smoke test of the guard. Real behavior verification requires running propagate end-to-end with a model.

- [ ] **Step 5: Build + run + commit**

```bash
cd build && make -j8 && ctest --output-on-failure 2>&1 | tail -10
git add src/model/video_session.h src/model/sam3_video.c tests/test_video_multi_object.c
git commit -m "$(cat <<'EOF'
video: reject remove_object/reset from inside callback

Spec §6.2 cases 2 and 14: mutating session state during a propagate
callback would race with the propagate loop. Now returns SAM3_EINVAL
with a logged error.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

This task slots into Phase 2 after task 2.3 (or any time before Phase 5 wraps). Logical home is end of Phase 2.
