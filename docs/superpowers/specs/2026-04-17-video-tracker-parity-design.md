# SAM3 Video Tracker — Python Parity Design

**Status:** Approved
**Date:** 2026-04-17
**Author:** Rifky Bujana Bisri
**Branch:** `feature/video-tracker`
**Scope:** Full parity with `Sam3TrackerPredictor` from `reference/sam3/sam3/`

---

## 1. Goal

Make `sam3_video_*` produce the same outputs as Python
`Sam3TrackerPredictor.propagate_in_video` for arbitrary multi-object videos,
within numerical tolerance. Close every behavioral and architectural gap
identified in the audit against `reference/sam3/sam3/`.

## 2. Background

A prior verification compared the C video tracker in
`src/model/{sam3_video.c, tracker.c, memory_bank.c, memory_encoder.c, memory_attn.c}`
against the Python reference in
`reference/sam3/sam3/{sam3_tracking_predictor.py, sam3_tracker_base.py, model_builder.py}`.

**Per-component math is correct:** memory attention (4 layers, d=256,
kv=64, heads=1), mask decoder (4 mask tokens), memory encoder (4-layer
downsampler + 2 CXBlock + 256→64 projection), `num_maskmem=7`,
`max_cond_frames_in_attn=4`, sigmoid_scale=20, sigmoid_bias=-10,
`NO_OBJ_SCORE=-1024.0`, image_size=1008, stride=14, eff_iou_score
selection — all match.

**Gaps identified:**

1. Multi-object propagation is stubbed
   (`sam3_video.c:1352-1354`: "a proper per-object mask merge is out of
   scope for this task"). `propagate_one` calls the tracker once per
   frame regardless of `n_objects`; only the last replayed mask is
   surfaced.
2. Memory bank is a single global ring shared across objects
   (`memory_bank.h:39-50`).
3. Memory cleared every `propagate` call
   (`sam3_video.c:1468,1507`); Python persists `output_dict_per_obj`
   across calls.
4. No `add_new_mask` API entry point.
5. No `dynamic_multimask_via_stability` (Python sets
   `delta=0.05, thresh=0.98` in `model_builder.py:487-491`).
6. No `iter_use_prev_mask_pred` refinement loop.
7. No `clear_non_cond_mem_around_input` (Python default `True`).
8. Eager full-video encoding at session start; no lazy/CPU-offload
   path. `~36 MiB` per frame stored, infeasible past a few hundred
   frames.

## 3. High-Level Architecture

### 3.1 Module changes

| File | Change |
|------|--------|
| `include/sam3/sam3.h` | New result types, new entry points, callback signature change, `SAM3_MAX_OBJECTS` cap drops from 64 to 16 |
| `src/model/memory_bank.{h,c}` | Per-object semantics; `obj_pointer` becomes `[256]` not `[n_obj, 256]`; new ops: `clear_around_frame`, `select_non_cond_for_frame` |
| `src/model/mask_decoder.{h,c}` | Add `dynamic_multimask_via_stability` selection; accept optional dense mask input for `iter_use_prev_mask_pred` |
| `src/model/tracker.{h,c}` | `track_step` becomes obj-indexed; mask-prompt path bypasses decoder |
| `src/model/sam3_video.c` | `propagate_one` becomes per-frame loop over objects; per-object result merge; no bank clear on entry |
| `src/model/frame_cache.{h,c}` | **New.** Tiered LRU: backend tier (default 4 GiB) + CPU spill (default 16 GiB) + recompute on miss. Replaces eager `cached_features[]` |
| `src/model/video_session.{h,c}` | Owns per-object banks, frame cache, persistence model |

### 3.2 Per-frame data flow during propagation

```
get_frame_features(f)             // tiered cache: backend → CPU spill → recompute
  └─→ for each obj_idx in 0..n_obj:
       ├─ if obj has prompt on frame f:    // conditioning frame for this obj
       │    ├─ clear_non_cond_around(obj.bank, f, window=num_maskmem)
       │    ├─ run prompt encoder + tracker (with this obj's bank as memory)
       │    ├─ apply dynamic-multimask-via-stability
       │    ├─ if iter_use_prev_mask_pred and re-prompt: feed prev mask
       │    ├─ apply occlusion gating
       │    ├─ run memory encoder
       │    └─ commit conditioning entry to obj.bank
       │
       ├─ elif obj has add_mask on frame f:
       │    ├─ resize user mask to high-res
       │    ├─ run memory encoder on mask directly
       │    └─ commit conditioning entry to obj.bank (skip decoder)
       │
       └─ else:                              // pure tracking
            ├─ run tracker (with obj.bank as memory)
            ├─ apply dynamic-multimask-via-stability
            ├─ apply occlusion gating
            ├─ run memory encoder
            └─ commit non-conditioning entry to obj.bank (filter by mf_threshold)

assemble per-object results into sam3_video_frame_result, invoke callback
```

### 3.3 Invariants

- Per-object banks are independent. No cross-object pollution.
- `propagate` is **resumable**: repeat calls (or `add_points` mid-tracking)
  extend rather than recompute.
- Frame cache LRU eviction is invisible to correctness: cache misses
  re-encode silently.
- `sam3_video_reset` clears banks but **preserves** the frame cache.

## 4. Data Structures

### 4.1 `sam3_video_frame_result` (new public type)

```c
struct sam3_video_object_mask {
    int    obj_id;            /* user-supplied obj_id */
    float *mask;              /* [H * W] f32 logits, malloc'd */
    int    mask_h, mask_w;    /* typically 1008x1008 (image_size) */
    float  iou_score;         /* selected mask's predicted IoU */
    float  obj_score_logit;   /* raw object-presence logit (>0 visible) */
    int    is_occluded;       /* convenience: obj_score_logit <= 0 */
};

struct sam3_video_frame_result {
    int frame_idx;
    int n_objects;
    struct sam3_video_object_mask *objects;  /* [n_objects], malloc'd */
};
void sam3_video_frame_result_free(struct sam3_video_frame_result *r);
```

### 4.2 Memory bank (per-object)

```c
struct sam3_memory_entry {
    struct sam3_tensor *spatial_features;   /* [HW, mem_dim=64] */
    struct sam3_tensor *obj_pointer;        /* [hidden_dim=256] — single, not [n_obj,256] */
    int    frame_idx;
    int    is_conditioning;
    float  obj_score;                       /* eff_iou_score for SAM2-Long filter */
};

struct sam3_memory_bank {
    struct sam3_memory_entry cond[SAM3_MAX_MEMORY_FRAMES];
    int n_cond;
    struct sam3_memory_entry non_cond[SAM3_MAX_MEMORY_FRAMES];  /* ring */
    int n_non_cond;
    int non_cond_head;                       /* ring write index */
    int capacity, max_cond_frames_in_attn, temporal_stride;
    float mf_threshold;
};
/* One bank per object, owned by sam3_video_session */
```

### 4.3 Per-object session state

```c
struct sam3_video_object {
    int obj_id;
    struct sam3_memory_bank bank;
    uint8_t *prompted_frames;                /* bitmap, [ceil(n_frames/8)] */
    struct sam3_tensor *prev_mask_logits;    /* [1, H, W, n_masks] or NULL */
    int                 prev_mask_frame;     /* -1 if none */
};

struct sam3_video_session {
    /* …existing fields… */
    struct sam3_video_object objects[SAM3_MAX_OBJECTS];   /* 16 */
    int n_objects;
    struct sam3_frame_cache frame_cache;     /* replaces cached_features[] */
};
```

### 4.4 Frame cache

```c
enum sam3_frame_tier { TIER_NONE, TIER_BACKEND, TIER_CPU_SPILL };

struct sam3_frame_cache_slot {
    int                 frame_idx;
    enum sam3_frame_tier tier;
    /* Backend tier: persist-arena tensors */
    struct sam3_tensor *image_features;      /* [1,H,W,256] */
    struct sam3_tensor *feat_s0, *feat_s1;
    /* CPU spill: raw bytes (same layout as backend tensors) */
    void   *spill_image_features, *spill_feat_s0, *spill_feat_s1;
    /* LRU bookkeeping */
    uint64_t last_access_seq;
};

struct sam3_frame_cache {
    struct sam3_frame_cache_slot *slots;     /* [n_frames] */
    int n_frames;
    size_t backend_budget, backend_used;
    size_t spill_budget,   spill_used;
    uint64_t access_counter;
    struct sam3_arena backend_arena;
};
```

### 4.5 Session opts

```c
struct sam3_video_start_opts {
    size_t frame_cache_backend_budget;       /* 0 = default 4 GiB */
    size_t frame_cache_spill_budget;         /* 0 = default 16 GiB; SIZE_MAX disable */
    int    clear_non_cond_window;            /* 0 = default 7 */
    int    iter_use_prev_mask_pred;          /* -1 = default 1 */
    int    multimask_via_stability;          /* -1 = default 1 */
    float  multimask_stability_delta;        /* 0 = default 0.05 */
    float  multimask_stability_thresh;       /* 0 = default 0.98 */
};
```

### 4.6 Memory budget (worst case)

Per-object bank (16 frames × 5184 HW × 64 dim f32 + 256 obj_ptr):
`16 × (5184·64 + 256) × 4 ≈ 21.3 MiB`

16 objects: `~340 MiB` — fits comfortably in current 1 GiB persist arena.

The frame cache lives in its **own** arena (`backend_arena` inside
`sam3_frame_cache`), separate from session persist (banks) and scratch
(per-frame). Session footprint with defaults:

| Arena | Default | Holds |
|-------|---------|-------|
| `persist` | 1 GiB | per-object banks |
| `scratch` | 256 MiB | per-frame graph eval |
| `frame_cache.backend_arena` | 4 GiB | encoded frame features |
| host malloc (spill) | 16 GiB | overflow features |

## 5. Public API

```c
/* --- Caps --- */
#define SAM3_MAX_OBJECTS 16        /* was 64 */
/* SAM3_MAX_MEMORY_FRAMES = 16 unchanged */
/* SAM3_MAX_POINTS_PER_OBJ = 16 unchanged */

/* --- Result types: see §4.1 --- */

/* --- Session lifecycle --- */
enum sam3_error sam3_video_start(sam3_ctx *ctx, const char *resource_path,
                                 sam3_video_session **out_session);
enum sam3_error sam3_video_start_ex(sam3_ctx *ctx, const char *resource_path,
                                    const struct sam3_video_start_opts *opts,
                                    sam3_video_session **out_session);
void sam3_video_end(sam3_video_session *session);
int  sam3_video_frame_count(const sam3_video_session *session);

/* --- Prompts --- */
enum sam3_error sam3_video_add_points(sam3_video_session *session,
                                      int frame_idx, int obj_id,
                                      const struct sam3_point *points,
                                      int n_points,
                                      struct sam3_video_frame_result *result);

enum sam3_error sam3_video_add_box(sam3_video_session *session,
                                   int frame_idx, int obj_id,
                                   const struct sam3_box *box,
                                   struct sam3_video_frame_result *result);

/* NEW */
enum sam3_error sam3_video_add_mask(sam3_video_session *session,
                                    int frame_idx, int obj_id,
                                    const uint8_t *mask,         /* binary, row-major */
                                    int mask_h, int mask_w,
                                    struct sam3_video_frame_result *result);

/* --- Object management --- */
enum sam3_error sam3_video_remove_object(sam3_video_session *session, int obj_id);
enum sam3_error sam3_video_reset(sam3_video_session *session);

/* --- Propagation --- */
typedef int (*sam3_video_frame_cb)(const struct sam3_video_frame_result *result,
                                   void *user_data);
enum sam3_error sam3_video_propagate(sam3_video_session *session,
                                     int direction,
                                     sam3_video_frame_cb callback,
                                     void *user_data);
```

### 5.1 Result ownership

- `add_points` / `add_box` / `add_mask` return a **single-frame, single-object**
  result. Caller frees with `sam3_video_frame_result_free`. `n_objects` is 1.
- `propagate`'s callback gets a **single-frame, multi-object** result, owned by
  the engine, valid only for the duration of the callback. Caller must not free.

### 5.2 Documented breaking changes vs. current code

1. Callback signature changes — current passes `(frame_idx, sam3_result*, n_objects, obj_ids[], user_data)`; new is `(sam3_video_frame_result*, user_data)`. The result already carries `frame_idx`, `n_objects`, and per-object `obj_id`.
2. `SAM3_MAX_OBJECTS` drops from 64 to 16.
3. `propagate` no longer clears banks on entry — repeat calls extend tracking. `sam3_video_reset` is the explicit clear.
4. Multiple `add_points` on the same `(frame, obj)` apply `clear_non_cond_mem_around_input` to wipe stale propagated entries near the prompt.

### 5.3 Caller migration

Existing callers (`tools/sam3_video_demo.c`, `tools/sam3_bench_video.c`)
must update their callback signatures and iterate `result->objects[]`. ~10-30
LOC each.

## 6. Error Handling & Edge Cases

### 6.1 Error codes used

- `SAM3_EINVAL` — bad args (NULL session, bad frame_idx, bad obj_id, mask dims mismatch)
- `SAM3_ENOMEM` — arena/alloc failure (cache+spill+recompute all fail; per-object bank full)
- `SAM3_EVIDEO` — video decode/load failure (existing)
- `SAM3_EMODEL` — graph eval failure (existing)
- `SAM3_EFULL` (new) — `add_points` rejected because `n_objects == SAM3_MAX_OBJECTS` and obj_id is new

### 6.2 Edge cases

| # | Case | Disposition |
|---|------|-------------|
| 1 | New object added after some frames propagated | Allowed; new bank starts empty. Documented: "objects added mid-tracking only have memory for frames touched after their first prompt." Stricter than Python (Python errors after `tracking_has_started`), more useful. |
| 2 | Object removal mid-propagate (from inside callback) | `SAM3_EINVAL`, logged. Caller must wait for `propagate` to return. |
| 3 | `add_points` on same `(frame, obj)` twice | First: registers + commits cond, **stores resulting mask logits in `obj.prev_mask_logits` with `obj.prev_mask_frame=frame`**. Second: `clear_around_frame(window)` wipes nearby non-cond on this obj's bank, re-runs decoder with `iter_use_prev_mask_pred` feeding `prev_mask_logits` as dense prompt, replaces cond entry, refreshes `prev_mask_logits`. |
| 4 | `add_points` and `add_mask` on same `(frame, obj)` | Last call wins; previous discarded; `prev_mask_logits` cleared. INFO log. |
| 5 | `add_mask` with mismatched dims | Resized to high-res internally (Python does the same). Reject only if dims are 0 or > `2 * image_size`. |
| 6 | Empty propagation (no prompts) | `n_objects <= 0 → SAM3_EINVAL`. Also: `n_objects > 0` but no obj has cond entry → `SAM3_EINVAL` "no conditioning prompts; add_points/box/mask first." |
| 7 | Frame cache fully full + spill disabled + recompute fails | Should not happen. If image encoder OOMs, propagate `SAM3_ENOMEM` with frame index. |
| 8 | CPU spill OOM (host malloc fails) | Fall through to recompute. WARN logged once per session. |
| 9 | `propagate(BOTH)` after partial `propagate(FORWARD)` | No bank clear between. Forward leaves cond+non_cond; backward extends from same state. Most-recent-wins for non-cond if both sweeps populated same frame. |
| 10 | Memory bank ring overflow during long propagation | Standard FIFO eviction with `mf_threshold` filter (matches Python `frame_filter()`). Cond entries cap at `SAM3_MAX_MEMORY_FRAMES=16` per object; 17th cond drops with WARN (C-specific cap, documented). |
| 11 | `iter_use_prev_mask_pred` when prev_mask is from different frame | Only used if `prev_mask_frame == current_frame`; else NULL → no dense prompt. (Same as Python.) |
| 12 | `dynamic_multimask_via_stability` with single-mask output | Bypassed — only runs when ≥3 candidate masks. |
| 13 | Callback returns non-zero (early stop) | `propagate` returns `SAM3_OK` immediately. Bank state stays consistent. |
| 14 | `sam3_video_reset` from inside callback | `SAM3_EINVAL`, logged. Use callback return to stop, then call reset. |

### 6.3 Logging conventions

- `INFO`: prompt-replacement events; `propagate` start/end with object/frame counts
- `WARN`: cap-hit events (cond bank full, spill OOM); unusual mask dims
- `ERROR`: arg validation failures; graph eval failures; cache+recompute failures
- `DEBUG`: per-frame per-object completion (gated; useful for parity debugging)

## 7. Test Plan

### 7.1 Existing tests to update

- `tests/test_memory_bank.c` — rewrite for per-object semantics; add
  `clear_around_frame` tests, `select_non_cond_for_frame` stride/threshold
  tests, ring overflow on per-object basis.
- `tests/test_tracker.c` — drop shared-bank assumption; add 2-object case
  asserting per-object bank independence.
- `tests/test_mask_decoder_nhwc.c` — extend with dynamic-multimask-via-stability
  fixtures (3 borderline-IoU candidates, verify stability-based selection
  differs from naive argmax).
- `tools/gen_nhwc_fixtures.c` — add fixture generators for new paths.

### 7.2 New tests

1. **`tests/test_frame_cache.c`** — sequential access (all hits within budget,
   LRU eviction order); backend full → CPU spill activation, promotion on
   next access; spill full → recompute path (mock encoder); reset semantics
   (cache survives `sam3_video_reset`).
2. **`tests/test_video_persistence.c`** — `propagate(FORWARD)` then `add_points`
   on tracked frame: verify `clear_non_cond_mem_around_input` wiped right
   window; `propagate(FORWARD)` twice: second exits without re-encoding (cache
   hits); FORWARD then BACKWARD: forward state survives; `sam3_video_reset`
   clears banks, preserves cache.
3. **`tests/test_video_multi_object.c`** — 2 objects on same frame: each gets
   own mask; object added mid-tracking: bank starts empty; `remove_object`
   mid-session: subsequent results omit it; per-object bank independence:
   occluding obj 1 does not perturb obj 2's mask within tolerance.
4. **`tests/test_video_add_mask.c`** — binary mask in → memory encoder runs →
   cond committed; result mask matches input (decoder skipped); resize handles
   224×224 input mask; `add_mask` after `add_points` on same frame: replaces.
5. **`tests/test_dynamic_multimask.c`** — 3 candidates at ~equal IoU, one with
   low stability (mask area changes >5% under threshold perturbation): unstable
   one dropped; single-mask case: stability bypassed; numerical parity fixture
   on 5 borderline cases vs Python.

### 7.3 End-to-end parity

6. **`tests/test_video_parity_kids.c`** — uses in-tree `assets/kids.mp4`.
   Fixture: Python-generated per-frame per-object masks for 2-object tracking
   (point prompts on frame 0, propagate forward 30 frames). Stored as PNG
   sequences under `tests/fixtures/video_kids/`.
   - C runs same scenario, computes per-object mean IoU vs Python masks,
     asserts `>= 0.85` per object (allows for accumulated numerical drift over
     30 frames).
   - Asserts no object fully lost (`obj_score_logit > 0` for ≥80% of frames
     where Python had it visible).

### 7.4 Fixture generation

`tools/gen_video_parity_fixtures.py` (new, documented but not committed-as-runtime).
Requires Python+PyTorch+reference repo. Generates binary fixtures for tests 5
and 6. Re-run when reference changes. Tests don't need the script at runtime —
fixtures are checked-in binary blobs.

### 7.5 CTest integration

All new tests added to `tests/CMakeLists.txt` per existing pattern. End-to-end
parity test opt-in via `-DSAM3_BUILD_PARITY_TESTS=ON` (~50 MiB fixtures, 30s
runtime). Default OFF in CI.

### 7.6 Out of scope for tests

- Per-object batched decoder path (deferred; see §8).
- CUDA/Vulkan backends (Metal+CPU only, same as current).
- Stress test for >16 objects (hard cap; 17th returns `SAM3_EFULL`, tested as
  edge case in `test_video_multi_object`).

## 8. Implementation Sequencing

Five phases. Each independently buildable + testable. Each phase ends with all
tests green before the next starts.

### Phase 1: Per-object memory bank + new result types — *~600 LOC, low risk*

- Refactor `memory_bank.{h,c}` per §4.2.
- Add `clear_around_frame()`, `select_non_cond_for_frame()`.
- Update `tests/test_memory_bank.c`.
- Define `sam3_video_frame_result` + free function in public header.
- No behavior change yet. Wired up in phase 2.

**Gate:** `test_memory_bank` green; `test_tracker` green; existing video
benchmarks pass with old behavior.

### Phase 2: Multi-object propagation + new callback — *~400 LOC, medium risk*

- `sam3_video_session` gains `objects[16]` array, each with own bank.
- `propagate_one` becomes per-object loop; assembles `sam3_video_frame_result`.
- New callback signature.
- Update `tools/sam3_video_demo.c`, `tools/sam3_bench_video.c`.
- Add `tests/test_video_multi_object.c`.

**Gate:** multi-object tests green; single-object regression vs phase-1
single-obj output mean IoU > 0.95 on `kids.mp4`.

### Phase 3: Frame cache (lazy + tiered) — *~500 LOC, medium risk*

- New `frame_cache.{h,c}`.
- `sam3_video_start` no longer encodes upfront; `sam3_video_start_ex` accepts opts.
- All `cached_features[f]` callers → `frame_cache_get(f)` (encoding on miss).
- `sam3_video_reset` preserves cache.
- `tests/test_frame_cache.c`.

**Gate:** `test_frame_cache` green; long-video smoke (1000-frame synthetic)
completes with bounded memory; benchmark variance vs phase-2 within 5% on
short videos.

### Phase 4: Memory persistence + clear_non_cond_mem_around_input — *~300 LOC, medium-high risk*

- Drop `sam3_memory_bank_clear` calls in `sam3_video_propagate`.
- On every `add_points`/`add_box`/`add_mask`, call `clear_around_frame(window=opts.clear_non_cond_window)` on that obj's bank.
- Add `tests/test_video_persistence.c`.

**Gate:** persistence tests green; idempotency assertion (two sequential
`propagate` on same prompts yield identical output); mid-tracking `add_points`
re-encodes one frame, not all.

### Phase 5: add_mask + dynamic multimask + iter_use_prev_mask_pred — *~400 LOC, low risk per subphase*

- 5a: `dynamic_multimask_via_stability` in `mask_decoder.c`. `tests/test_dynamic_multimask.c`.
- 5b: `iter_use_prev_mask_pred` — stash prev mask logits per (obj, frame); feed as dense prompt on re-prompt.
- 5c: `sam3_video_add_mask` + decoder bypass. `tests/test_video_add_mask.c`.

**Gate:** all unit tests green; **end-to-end parity test**
(`test_video_parity_kids.c`) green with mean IoU ≥ 0.85 per object over 30
frames.

### Risk summary

| Phase | Risk | Mitigation |
|-------|------|------------|
| 1 | Low | Pure refactor; old tests still pass |
| 2 | Medium | Unit + integration both required gates |
| 3 | Medium | Long-video smoke test mandatory before merge |
| 4 | Medium-High | Idempotency assertion in tests |
| 5 | Low (per subphase) | Each subphase independently revertable |

### Estimate

~2200 LOC total. ~5-7 working days. Each phase independently mergeable to
`feature/video-tracker` as separate commits.

### Out of scope (deferred follow-ups)

- Batched per-object decoder (~600 LOC, separate PR).
- CUDA/Vulkan backend support for new code paths.
- Multiplex tracking code path (separate Python module).

## 9. Open questions (none at spec time)

All design questions resolved during brainstorming. Implementation may surface
new ones; those go in plan-level docs.
