# SAM 3.1 Tracker Parity Fixture — Design

**Status:** Approved
**Date:** 2026-04-20
**Author:** Rifky Bujana Bisri
**Branch:** `feature/sam3.1-image-path`
**Scope:** Close the "Strengthen `tests/test_sam3_1_track.c`" TODO from
Phase 2.5b of the SAM 3.1 multiplex tracker sub-project, plus delete one
dead `tgt_pos` branch.

---

## 1. Goal

Give the SAM 3.1 multiplex tracker a **per-frame IoU parity test** against
the Python reference on `assets/kids.mp4`, so that regressions of the six
B1–B6 memory-path fixes (image-memory plumbing, maskmem tpos, obj-ptr
tpos, `extra_per_object`, mask scale/bias, conditioning channels) surface
as a concrete failing assert rather than passing the looser
mixed-sign/fg-fraction invariants in `test_sam3_1_track.c`.

A small cleanup ships alongside: delete the dead `tgt_pos` branch in
`src/model/tracker_multiplex.c:1085-1099` — its only caller passes NULL.

## 2. Background

### 2.1 Why the obvious approach doesn't work

The straightforward design — "dump Python masks from the same point
prompt, assert IoU" — fails because sub-project 3 (the interactive
single-image decoder) isn't ported yet. On the C side, frame-0's
`obj_ptr` is a placeholder derived from a zero-seeded mask decoder, not
from the user's click. The Python reference routes point prompts through
the interactive decoder and produces a correct frame-0 mask. Naive
parity would diverge dramatically on frame 0 for reasons unrelated to
B1–B6, invalidating the test's signal on later frames.

### 2.2 The C-seeded approach

Instead, seed both sides identically:

1. **C runs first.** The current C frame-0 path (even with its
   placeholder decoder) produces a deterministic mask.
2. **Python is fed that mask** via `Sam3MultiplexTracking.add_new_mask`
   (`sam3_multiplex_tracking.py:2909`), bypassing the interactive
   decoder.
3. **Both sides propagate** from the same seed.
4. **Parity is measured on frames 1..N**, which is exactly what B1–B6
   control (memory attention + maskmem backbone).

This is self-referential (C produces its own seed), but that's the
point: we are parity-testing the memory stream, not the frame-0 decoder.
Frame 0 gets a looser sanity check against a committed `seed_mask.png`
so fixture staleness is visible.

### 2.3 Existing scaffolding we build on

- `tools/gen_video_parity_fixtures.py` — 69-line scaffold for SAM 3
  video parity. Uses `Sam3VideoPredictor` and dumps per-frame PNGs.
  Never run in CI; fixtures checked in as binary blobs.
- `tests/test_video_parity_kids.c` — 85-line stub gated on
  `SAM3_BUILD_PARITY_TESTS=ON`. Skips when fixtures absent; currently
  has no PNG loader or IoU compute.
- `tests/fixtures/video_kids/README.md` — documents the regen procedure.
- `tests/test_sam3_1_track.c` — the existing smoke test with the B1–B6
  mixed-sign + fg-frac invariants. Unchanged by this work; continues to
  run without the fixture.

## 3. High-Level Architecture

### 3.1 File changes

| File | Change |
|------|--------|
| `tools/sam3_1_dump_seed.c` (new) | ~50-line CLI: init → load → video_start → add_points(frame=0) → write grayscale PNG of frame-0 mask. |
| `tools/gen_video_parity_fixtures.py` | Add `--variant {sam3,sam3.1}` and `--seed-mask` flags. SAM 3.1 branch builds `Sam3MultiplexTracking`, calls `add_new_mask` with the seed, propagates forward 3 frames, writes PNGs. Existing SAM 3 path unchanged by default. |
| `tools/_cpu_patches.py` (new) | Extract the triton stub + CUDA→CPU redirect + `addmm_act` fp32 patch from `dump_reference.py` into a reusable module. Imported by both `dump_reference.py` and `gen_video_parity_fixtures.py`. |
| `tests/test_video_parity_kids.c` | Variant dispatch via `SAM3_PARITY_VARIANT` CMake define. SAM 3 branch stays as-is (still scaffold). SAM 3.1 branch: PNG-load seed, sanity-check C frame-0, propagate 3 frames forward, per-frame IoU ≥ 0.75 vs committed PNGs. |
| `tests/test_helpers_png.{h,c}` (new) | `load_png_grayscale(path, *h, *w) -> uint8_t*` and `save_png_grayscale(path, data, h, w) -> int`. Thin wrappers around `stb_image.h` / `stb_image_write.h` already in `third_party/`. |
| `tests/fixtures/video_kids/sam3_1/` (new) | `prompts.json`, `seed_mask.png`, `frames/frame_0001_obj_1.png`..`frame_0003_obj_1.png`, `README.md`. |
| `tests/fixtures/video_kids/README.md` | Updated to cover both variants. |
| `CMakeLists.txt` | `SAM3_PARITY_VARIANT` option (default `sam3_1`). Compile `tools/sam3_1_dump_seed.c` as an optional target under the existing `SAM3_BUILD_TOOLS` gate. |
| `src/model/tracker_multiplex.c:1085-1099` | Delete dead `tgt_pos` branch. |

### 3.2 Fixture layout

```
tests/fixtures/video_kids/
├── README.md                      # regen steps, both variants
├── frames/                        # SAM 3 scaffold (untouched)
└── sam3_1/
    ├── prompts.json               # single obj, center point (0.5, 0.5)
    ├── seed_mask.png              # C frame-0 mask (grayscale)
    ├── README.md                  # SAM 3.1-specific regen steps
    └── frames/
        ├── frame_0001_obj_1.png   # Python propagation output
        ├── frame_0002_obj_1.png
        └── frame_0003_obj_1.png
```

## 4. Data Flow

### 4.1 Fixture regeneration (offline, human-run)

```bash
# Step 1: C dumps its own frame-0 mask as the seed
./build/tools/sam3_1_dump_seed \
    --model models/sam3.1.sam3 \
    --video assets/kids.mp4 \
    --point 0.5,0.5,1 \
    --out tests/fixtures/video_kids/sam3_1/seed_mask.png

# Step 2: Python propagates from that seed, dumps frames 1..3
SAM3_CKPT=... SAM3_BPE=... python tools/gen_video_parity_fixtures.py \
    --variant sam3.1 \
    --video assets/kids.mp4 \
    --frames 3 \
    --seed-mask tests/fixtures/video_kids/sam3_1/seed_mask.png \
    --out tests/fixtures/video_kids/sam3_1/
```

### 4.2 CI test run (SAM 3.1 variant)

```
1. If fixtures/sam3_1/seed_mask.png absent  → SKIP (default profile)
2. sam3_init → sam3_load_model(sam3.1.sam3) → sam3_video_start(kids.mp4)
3. sam3_video_add_points(obj=1, frame=0, point=(0.5,0.5,1))
4. Load seed_mask.png.
5. Compute IoU(C_frame0, seed_mask):
     < 0.50  → hard fail   (fixture is fundamentally stale)
     < 0.90  → warn only   (numerical drift; re-bless if repeated)
     ≥ 0.90  → proceed
6. sam3_video_propagate(FORWARD, callback)
     callback stops after 3 frames
7. For n in {1, 2, 3}:
     Load frames/frame_000n_obj_1.png
     IoU(C_fn, python_fn) ≥ 0.75  → assert
```

## 5. Interfaces

### 5.1 `tools/sam3_1_dump_seed.c`

```c
/*
 * Usage:
 *   sam3_1_dump_seed --model PATH --video PATH \
 *                    --point X,Y,LABEL --out PATH.png
 *
 * Exit codes:
 *   0  success (PNG written)
 *   1  argparse error
 *   2  model/video load error
 *   3  add_points error
 *   4  PNG write error
 */
```

Minimal CLI — no --help-text polish, no multi-object support. The
fixture uses a single object per the test's prompt.

### 5.2 `gen_video_parity_fixtures.py` additions

```
--variant {sam3,sam3.1}     required (no default — explicit)
--seed-mask PATH             required iff --variant=sam3.1
--frames N                   default 30 for sam3, 3 for sam3.1
```

The SAM 3.1 branch instantiates `Sam3MultiplexTracking` directly (not
the user-facing `Sam3MultiplexVideoPredictor`, which hardcodes CUDA
autocast). Reuses the CPU-patches module for CUDA redirection.

### 5.3 `tests/test_helpers_png.{h,c}`

```c
/* Load 8-bit grayscale PNG. Returns malloc'd buffer; caller frees.
   Returns NULL on error (log via sam3_log_error). */
uint8_t *load_png_grayscale(const char *path, int *out_h, int *out_w);

/* Save 8-bit grayscale PNG. Returns 0 on success, -1 on error. */
int save_png_grayscale(const char *path, const uint8_t *data, int h, int w);
```

Both use `src/util/vendor/stb_image.h` / `stb_image_write.h` already in
tree. Image helpers already link these for other tools; add the
`tests/test_helpers_png.c` TU to the shared test-helpers CMake target.

### 5.4 `test_video_parity_kids.c` variant dispatch

Uses a compile-time `SAM3_PARITY_VARIANT` macro (from CMake):
- `SAM3_PARITY_VARIANT=sam3` → original scaffold path (unchanged).
- `SAM3_PARITY_VARIANT=sam3_1` → new SAM 3.1 parity path.

No runtime dispatch — the two variants share nothing except the skeleton
`main()`. Inline-split with `#if`/`#else`.

## 6. Error handling & edge cases

- **Absent fixtures.** `access(seed_mask.png, F_OK) != 0` → SKIP with
  message. Default CI profile stays clean. Same pattern as
  `test_sam3_1_track.c` and the existing `test_video_parity_kids.c`
  stub.
- **Seed-mask stale.** `IoU(C_frame0, seed_mask)` two-tier threshold:
  hard fail below 0.50 (regenerate), warn between 0.50 and 0.90
  (tolerate numerical drift across backends), pass ≥ 0.90. Warning is
  emitted via `fprintf(stderr, ...)`; doesn't fail CI but is visible in
  test output.
- **Python CPU-only reference.** `Sam3MultiplexTracking.init_state`
  hardcodes `torch.device("cuda")` (line 235). The shared
  `_cpu_patches.py` redirects that. Confirm at implementation time that
  `add_new_mask` accepts CPU tensors; if not, manually device-transfer
  in the Python script.
- **`add_new_mask` API drift.** The reference API signature is
  verified at implementation time; if the upstream shape differs
  (e.g. requires box/logits in addition to mask), fall back to the
  plan's equivalent path in `Sam3MultiplexTracking`.
- **Mask dimensions.** The C `sam3_video_frame_result.mask_{h,w}`
  reports the native output resolution (1008×1008 for SAM 3.1 at
  image_size=1008). Both `sam3_1_dump_seed` and the PNG fixtures write
  at that native size. The test hard-fails if PNG dims don't match the
  C frame output — message points to the regen procedure.
- **`SAM3_MAX_MEMORY_FRAMES = 16` bank cap.** 3 frames is well under
  the cap — not a concern for this fixture. If the frame count is ever
  raised past 16, the test exercises bank rollover too; note this in
  the fixture README.

## 7. Testing

The parity test itself is the deliverable. Validation of this work:

1. Generate the fixture once, by hand, from current-branch `HEAD`
   (the B1–B6 fixes are already committed).
2. Run `./build/tests/test_video_parity_kids` with
   `SAM3_PARITY_VARIANT=sam3_1` — should pass.
3. Locally revert one of the B1–B6 commits; re-run the parity test;
   expect IoU drop and failing assert. Restore.
4. Sanity: `test_sam3_1_track.c` still passes (its invariants are
   strictly looser).

Not testing in CI: fixture regeneration, Python reference execution.
Both are manual/offline.

## 8. Testing scope — what this does NOT catch

- **Frame-0 correctness.** Blocked on sub-project 3 (interactive
  decoder). This test accepts the current placeholder and measures only
  propagation.
- **Absolute parity with Python from a point prompt.** The seed is the
  C output, not a Python-computed "truth". An interactive-decoder
  regression on the C side would change the seed and, transitively, the
  Python fixtures on regeneration — but not the per-frame IoU assertion
  (both sides get the same seed).
- **Multi-object regressions.** Single object only; multiplex joint
  forward (sub-project 4) needs its own fixture.
- **Memory-bank rollover past frame 16.** Fast/3-frame scope. Separate
  follow-up if rollover behavior needs coverage.

## 9. Open work deliberately scoped out

- **Metal-backend parity.** The C test runs on whichever backend CI is
  configured for. If Metal/CPU drift exceeds 0.75 IoU, raise that
  concern in a follow-up spec rather than loosening the threshold
  pre-emptively.
- **Box-prompt parity.** Point prompt only; boxes can be added later.
- **Extending to SAM 3.** The SAM 3 branch of the scaffold remains
  stubbed. Out of scope here.

## 10. Deliverables checklist

- [ ] `tools/_cpu_patches.py` (extracted from `dump_reference.py`)
- [ ] `tools/sam3_1_dump_seed.c` + CMake wire-in
- [ ] `tools/gen_video_parity_fixtures.py` variant support
- [ ] `tests/test_helpers_png.{h,c}`
- [ ] `tests/test_video_parity_kids.c` SAM 3.1 branch
- [ ] `tests/fixtures/video_kids/sam3_1/` populated
- [ ] `CMakeLists.txt` `SAM3_PARITY_VARIANT` option
- [ ] `src/model/tracker_multiplex.c` dead-branch deletion
- [ ] Local pass/fail validation (Section 7)
- [ ] Update `TODO.md`: mark 2.5b IoU item done, mark tgt_pos item done

---

## Appendix A — Why not Option B (synthetic Python seed)?

A synthetic Gaussian blob at the click point would make the Python
propagation independent of C. But then we're comparing
"C-propagates-from-C-garbage" to "Python-propagates-from-blob", which
are different distributions, and the IoU baseline becomes arbitrary.
The C-seeded approach keeps both sides on the same input distribution
so the IoU threshold is a meaningful correctness signal.

## Appendix B — Why not Option C (regression hash)?

A per-frame output hash locks in current C behavior exactly. It would
catch *any* byte change — including fixes. That's a regression-detection
tool, not a correctness tool. The B1–B6 gaps were real bugs that a
hash-baseline would have frozen in.

## Appendix C — Reference pointers

- `reference/sam3/sam3/model/sam3_multiplex_tracking.py:207` — `init_state`
- `reference/sam3/sam3/model/sam3_multiplex_tracking.py:301` — `propagate_in_video`
- `reference/sam3/sam3/model/sam3_multiplex_tracking.py:2909` — `add_new_mask`
- `tools/dump_reference.py:40-74` — CPU patch snippets to extract
- `src/model/tracker_multiplex.c:1085-1099` — dead `tgt_pos` branch
- `src/model/sam3_video.c:1322` — `sam3_video_add_points` entry point
- `tests/test_sam3_1_track.c` — existing smoke test (untouched)
