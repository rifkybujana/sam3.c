# SAM 3.1 Tracker Layer-Parity Debugging — Design

**Status:** Approved
**Date:** 2026-04-20
**Author:** Rifky Bujana Bisri
**Branch:** `feature/sam3.1-image-path`
**Scope:** Root-cause and fix the C-vs-Python divergence on SAM 3.1 multiplex tracker propagation frames until `test_video_parity_kids` (SAM 3.1 variant) passes at per-frame IoU ≥ 0.75.

---

## 1. Goal

Drive `tests/test_video_parity_kids.c` (SAM 3.1 variant) from its current
failing baseline (per-frame IoU = 0.0 on frames 1–3) to green by
layer-by-layer bisection: dump intermediate tensors from both C and
Python on the same inputs, diff them, and fix the first divergent op in
C. Repeat until IoU passes.

## 2. Background

### 2.1 The symptom

Committed fixture comparison (see
`tests/fixtures/video_kids/sam3_1/README.md`):

```
frame-0 vs seed IoU=1.0000   # C dumper is deterministic
frame 1 IoU=0.0000           # complete divergence
frame 2 IoU=0.0000
frame 3 IoU=0.0000
```

C produces all-`-1024.0` masks on propagation frames because
`apply_occlusion_gating` at `src/model/sam3_video.c:113-122` zeroes the
mask whenever `obj_score_logit <= 0`. Python uses the *same* threshold
(`video_tracking_multiplex.py:835`) but gets `obj_score_logit > 0`, so
Python's masks stay ~6% foreground. The divergence is upstream of
`obj_score_head(sam_tokens[0])` in the multiplex mask decoder.

### 2.2 The path to instrument

```
frame N image → ViT → neck → image_features
                                  │
          ┌───────────────────────┤
          │        (frame 0 uses interactivity_no_mem_embed; no mem path)
          │
memory_attn(image, memory_bank)       ← from frame 1 onward
          │
          ▼
enriched features
          │
multiplex_mask_decoder(enriched, image_pe, sparse_embed, extra_per_object)
          │
          ├─→ masks [16, 3, 288, 288]
          ├─→ iou   [16, 3]
          ├─→ obj_score_logit [16, 1]     ← SYMPTOM
          └─→ sam_tokens [16, 3, 256]
          │
apply_occlusion_gating(obj_score_logit) → NO_OBJ_SCORE fill
```

### 2.3 Existing infrastructure to reuse

- `SAM3_DEBUG_DUMP` CMake option (`CMakeLists.txt:15,64-65`). Already
  wired through; flipping it on sets `-DSAM3_DEBUG_DUMP`.
- `auto_dump_tensor()` helper (`src/model/sam3_video.c:73-86`) that
  writes NHWC f32 raw binary to any `/tmp/*.bin`.
- `sam3_dbg_*` extern globals pattern (`src/model/sam3_video.c:61-68`)
  already used by `src/model/mask_decoder.c:855-965` to capture
  intermediate tensors from the SAM 3 image path's two-way transformer.
- `scripts/compare_layer_dumps.py` — existing PAIRS-based comparator
  that prints cosine similarity + abs/rel error per pair. Covers ViT +
  4 FPN scales today; pattern extends to tracker paths trivially.
- `scripts/dump_reference_layers.py` — existing Python-reference dumper
  for the SAM 3 image path. Provides the CPU-patch template.
- `tools/sam3_1_dump_seed` — already accepts `--propagate-frames N` and
  runs the full C tracker.
- `tools/gen_video_parity_fixtures.py` — already drives the Python
  reference through `Sam3VideoTrackingMultiplexDemo`.

The instrumentation work is **extensions of existing patterns**, not
new infrastructure.

## 3. Approach

### 3.1 Bisection strategy

Start at a coarse level — dump only the top-level outputs of the two
suspect components. Based on which output first diverges, drill into
its internals. This minimizes instrumentation work: the user's own data
tells us where to look next, rather than front-loading ~30 dumps of
every transformer layer.

### 3.2 Level 0 — coarse outputs (6 tensors)

On frame 0 (cond — known-good sanity) and frame 1 (propagation —
known-broken), dump:

| Slot | Tensor | Shape | Purpose |
|---|---|---|---|
| `memory_attn_out` | memory-attn output | `[1, 5184, 256]` | Enriched image feats before mask decoder (frame 1 only) |
| `mask_dec_masks` | mask decoder masks | `[16, 3, 288, 288]` | Final mask logits |
| `mask_dec_iou` | mask decoder IoU | `[16, 3]` | Best-mask selection input |
| `mask_dec_obj_score` | obj_score_logit | `[16, 1]` | **The symptom** |
| `mask_dec_sam_tokens` | sam_tokens | `[16, 3, 256]` | Obj-ptr source |
| `memory_bank_state` | memory tensor at frame-1 read | `[1, Nm, 256]` | Sanity-check what was built |

That's 6 C-side slots, 6 matching Python dumps, 1 comparison script
invocation per iteration.

### 3.3 Level 1 — drill-downs (one path per divergent level-0 output)

**Path α — memory_attn output diverged first:**

Four decoupled transformer layers. Dump `layer[0..3].out` (`[1, 5184,
256]` each) and the three per-layer inputs (self-attn Q/K/V tied,
cross-attn image and memory sides). Narrows to a specific layer + op.

**Path β — mask_decoder outputs diverged, memory_attn matched:**

Two-way transformer layers (2 × Q/K/V/cross-attn). Dump per-layer
outputs + the `extra_per_object` input (confirms B4 slot assignment)
and the pre-head `sam_tokens[0]` (direct obj_score input).

**Path γ — memory_bank_state diverged:**

Dump the memory bank builder's intermediates: maskmem tokens,
obj_ptr rows, maskmem_tpos_enc application, obj_ptr_tpos_proj output,
memory_image_pos per-entry. This localizes B2/B3/B5/B6 regressions.

Each level-1 path adds ~5-10 dumps. Only one path's dumps are added
per iteration; premature instrumentation wastes effort.

### 3.4 Iteration loop

```
1. cmake -DSAM3_DEBUG_DUMP=ON .. && cmake --build . -j8

2. ./build/sam3_1_dump_seed \
     --model models/sam3.1.sam3 --video assets/kids.mp4 \
     --point 0.5,0.5,1 --out /tmp/seed.png \
     --propagate-frames 2 --frames-dir /tmp/c_frames
     # → /tmp/dbg_trk_<slot>_f<0|1>.bin

3. SAM3_CKPT=models/sam3.1_multiplex.pt \
     python scripts/dump_tracker_layers.py \
       --video assets/kids.mp4 --seed-mask /tmp/seed.png --frames 2
     # → /tmp/py_trk_<slot>_f<0|1>.bin

4. python scripts/compare_tracker_layers.py
     # Prints per-pair cosine + abs_err; first row with cos < 0.99
     # identifies where to drill

5. Narrow:
     - level-0 mismatch → add level-1 dumps on matching path; goto 1
     - level-1 mismatch → hypothesize + patch that C op; goto 1

6. Re-run ctest -R test_video_parity_kids after each patch
```

### 3.5 Thresholds

- **Per-layer diff threshold:** cosine similarity ≥ 0.99. Justification
  — SAM 3 image-path layer diffs regularly hit 0.995+; NHWC kernel
  variation accounts for the gap from 1.0. 0.99 is tight enough to
  catch real bugs without false-positive-ing on bf16/NHWC drift.
- **Terminal success:** `test_video_parity_kids.c` SAM 3.1 variant
  passes all three frames at per-frame IoU ≥ 0.75 (existing
  threshold — unchanged by this work).

## 4. Components

### 4.1 New files

| Path | Purpose |
|---|---|
| `scripts/dump_tracker_layers.py` | Python driver. Reuses `tools/_cpu_patches.py` + the `gen_video_parity_fixtures.py` model-build path; registers `forward_hook`s on target sub-modules to dump NHWC f32 binaries matching the C side's format. Takes `--video`, `--seed-mask`, `--frames`. |
| `scripts/compare_tracker_layers.py` | Extends the `compare_layer_dumps.py` PAIRS-table pattern with a tracker-scoped list. Reports per-pair cosine, abs_err max/mean, first divergent index. Emits one line per layer; first-divergent-row marker makes the bisection decision obvious. |

### 4.2 Modified files (level 0)

| Path | Change |
|---|---|
| `src/model/tracker_multiplex.c` | Add `#ifdef SAM3_DEBUG_DUMP` extern declarations + writes into `sam3_dbg_trk_*` globals at: (a) `sam3_multiplex_memory_attn_forward` exit, (b) `sam3_multiplex_mask_decoder_forward` exit (masks, iou, obj_score, sam_tokens). |
| `src/model/sam3_video.c` | Add level-0 `sam3_dbg_trk_*` extern globals + `auto_dump_tensor` calls after the frame's `graph_eval`. Per-frame suffix: `_f0` (cond), `_f1` (first propagation). |
| `tools/sam3_1_dump_seed.c` | No change. Existing `--propagate-frames N` flag already drives the full path. |

### 4.3 Modified files (level 1 paths — added as the data demands)

- **Path α (memory-attn)** — extends the hooks at
  `memory_attn_layer` in `src/model/tracker_multiplex.c` for per-layer
  outputs.
- **Path β (mask decoder)** — extends hooks inside the two-way
  transformer block in `src/model/tracker_multiplex.c`.
- **Path γ (memory bank)** — extends hooks in
  `multiplex_build_memory_from_bank` + `multiplex_apply_linear_256`.

Each path's additions follow the same `sam3_dbg_trk_*` + `extern` +
`auto_dump_tensor` triad. No new patterns.

### 4.4 Files NOT modified

- `tools/gen_video_parity_fixtures.py` — stays focused on fixture regen.
  Layer-dumping lives in the parallel `dump_tracker_layers.py`.
- `tests/test_video_parity_kids.c` — unchanged. It is the terminal
  success gate, not part of the debugging loop.
- Python reference under `reference/sam3/` — never modified. It's the
  source of truth. All fixes land in C.

## 5. Data flow

### 5.1 Per-iteration (offline, human-run)

```
C side:
  /tmp/seed.png               ← sam3_1_dump_seed
  /tmp/c_frames/frame_0001.png
  /tmp/dbg_trk_*.bin          ← auto_dump_tensor (SAM3_DEBUG_DUMP=ON)

Python side:
  /tmp/py_trk_*.bin           ← dump_tracker_layers.py (hooks on same model)

Comparison:
  table printed to stdout      ← compare_tracker_layers.py
```

### 5.2 Terminal validation

```
ctest -R test_video_parity_kids
  → 100% tests passed (IoU ≥ 0.75 on frames 1..3)
```

## 6. Error handling & edge cases

- **Dump slot not populated** (e.g. frame 0 has no memory-attn output):
  `auto_dump_tensor` already NULL-checks; files are simply absent.
  `compare_tracker_layers.py` prints "MISSING" for that row rather
  than failing.
- **Shape mismatch between C NHWC and Python NCHW:** the existing
  comparator reshapes via explicit PAIRS shape tuples. New PAIRS rows
  declare the C-side NHWC shape; the Python dumper transposes to NHWC
  before writing (mirrors `dump_reference_layers.py`).
- **bfloat16 / CPU drift on Python side:** reuse `tools/_cpu_patches.py`
  force-float32 path that `gen_video_parity_fixtures.py` already works
  against.
- **Hook placement bugs (e.g. hook fires before projection, not after):**
  caught cheaply — the level-0 diff will show a 0.0 cosine on that
  specific slot while others match. Re-inspect the hook site.
- **Too-tight threshold (0.99) flagging benign drift:** relax case-by-
  case with a justifying note in the commit message. Don't drop it
  below 0.95 without a write-up — at that point the "parity" framing
  is compromised.

## 7. Testing

- **Each iteration self-validates** via `compare_tracker_layers.py`.
  No separate unit test for the dumper scripts themselves — they're
  wrappers around existing helpers.
- **Final gate:** existing `tests/test_video_parity_kids.c`
  (SAM3_PARITY_VARIANT=sam3_1). No new test code needed.

## 8. Scope — explicitly NOT doing

- Fixing frame-0 divergence from the missing interactive decoder
  (sub-project 3 territory). The C-seeded approach sidesteps this.
- Multi-object or multiplex joint-forward parity (sub-project 4).
- Numerical parity *below* 0.99 cosine — kernel drift from NHWC / bf16
  is acceptable and documented.
- Modifications to the Python reference under `reference/sam3/`.
- A generic layer-diffing framework. The two new scripts are
  purpose-built for the SAM 3.1 tracker.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Python-side forward hooks have device / autocast interactions that produce different numerics than `gen_video_parity_fixtures.py` | Reuse the exact same CPU patches from that file; dump_tracker_layers.py wraps the same builder. Cross-check: fixtures regenerated from the dump path must IoU-match the committed `frames/frame_000{1,2,3}_obj_1.png`. |
| Bisection terminates in a part of the Python reference where the upstream has its own numerical slop (e.g. bfloat16 on CUDA in original weights) | Already mitigated by forcing fp32 via `install_addmm_act_fp32`. If further drift shows up, document and relax per-slot threshold to match observed baseline. |
| Divergence is spread across multiple ops rather than one | Fix the first; re-run; repeat. The IoU test is the termination condition, not a single-op fix. |
| A fix for one op introduces a regression elsewhere (breaking the C-seed produced by `sam3_1_dump_seed`) | `test_sam3_1_track` smoke test's strengthened invariants (mixed-sign, fg-frac) catch gross frame-0 regressions; CI's pre-existing SAM 3 image path parity tests catch image-encoder regressions. Run the existing full test suite after each fix. |

## 10. Deliverables checklist

- [ ] Level-0 C dump hooks in `tracker_multiplex.c` + `sam3_video.c`.
- [ ] `scripts/dump_tracker_layers.py` (Python hook-based dumper).
- [ ] `scripts/compare_tracker_layers.py` (diff table).
- [ ] First iteration: run, inspect diffs, record findings in a session
      note.
- [ ] Level-1 dumps (added to whichever path level-0 identified).
- [ ] C fix(es) for first divergent op.
- [ ] Each fix: smoke test (`test_sam3_1_track`) still passes.
- [ ] Terminal: `test_video_parity_kids` (SAM 3.1 variant) green at
      per-frame IoU ≥ 0.75.
- [ ] TODO.md updated to mark the sub-project 2 parity task closed
      (local-only file; not committed).

---

## Appendix A — Why bisection rather than exhaustive or hypothesis-driven?

- **Exhaustive** (dump all ~30 transformer intermediates up front):
  front-loads instrumentation work that may not be needed. If the bug
  lives at the top level (e.g. `extra_per_object` slot assignment
  fed into the mask decoder), we never look at the 4 memory-attn
  layers' guts.
- **Hypothesis-driven** (guess B4 then B3 then B6, re-testing after
  each): fast if the first guess is right, but we've already closed
  B1–B6 per Phase 2.5b and the divergence persisted. That suggests
  the bug is *not* a whole-item regression but something more subtle
  (e.g. a wrong index, a transposed tensor, a missing scalar). Data
  beats guessing in that regime.
- **Bisection** cost scales as O(log N) dumps, and each dump directly
  points at where to look next.

## Appendix B — Reference pointers

- Symptom: `src/model/sam3_video.c:113-122` — occlusion gate.
- Parity test: `tests/test_video_parity_kids.c` (SAM3_PARITY_VARIANT=sam3_1).
- C driver: `tools/sam3_1_dump_seed.c` with `--propagate-frames`.
- Python driver baseline: `tools/gen_video_parity_fixtures.py`.
- Python upstream: `reference/sam3/sam3/model/video_tracking_multiplex.py`,
  `video_tracking_multiplex_demo.py`.
- Existing SAM 3 layer parity: `scripts/{dump_reference,compare}_layer_dumps.py`.
- Fixture baseline: `tests/fixtures/video_kids/sam3_1/` (README notes the
  current 0.0 IoU baseline).
