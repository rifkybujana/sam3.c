# SAM 3.1 parity fixtures: kids.mp4

C-seeded, Python-propagated parity fixtures for
`test_video_parity_kids` (SAM 3.1 variant).

## Files

- `prompts.json` — single object, center point (0.5, 0.5) on frame 0.
  Matches the C test's fixed prompt.
- `seed_mask.png` — C frame-0 mask (grayscale, >0 → 255). Produced by
  `./build/sam3_1_dump_seed`. Committed as-is.
- `frames/frame_NNNN_obj_1.png` — Python reference propagation for
  frames 1..3, seeded via
  `Sam3VideoTrackingMultiplexDemo.add_new_masks(seed_mask)`.

## Regenerating

    # Step 1: Build the C helper and dump the frame-0 seed
    cmake -DSAM3_BUILD_PARITY_TESTS=ON .. && \
      cmake --build . --target sam3_1_dump_seed -j8
    ./sam3_1_dump_seed \
        --model ../models/sam3.1.sam3 \
        --video ../assets/kids.mp4 \
        --point 0.5,0.5,1 \
        --out ../tests/fixtures/video_kids/sam3_1/seed_mask.png

    # Step 2: Propagate Python from that seed
    cd ../tools
    SAM3_CKPT=/path/to/sam3.1_multiplex.pt \
      python gen_video_parity_fixtures.py \
        --variant sam3.1 \
        --video ../assets/kids.mp4 \
        --frames 3 \
        --seed-mask ../tests/fixtures/video_kids/sam3_1/seed_mask.png \
        --out ../tests/fixtures/video_kids/sam3_1/

Python outputs are downsampled to 288×288 (the C decoder's native mask
resolution — grid 72 × 4) so both sides compare at the same size.

Requires Python 3.10+, PyTorch (CPU-only is supported via
`tools/_cpu_patches.py`), the upstream reference at `reference/sam3/`,
and the `decord` video decoder (`pip install decord` — on Apple Silicon
use `pip install eva-decord` as a drop-in). On CPU-only machines a full
regen has not yet been validated end-to-end: the upstream demo hits
shim-able issues (e.g. `load_video_frames` kwargs, bf16 autocast device)
and possibly more as decoder bits materialize. If
`gen_video_parity_fixtures.py` fails, the current committed
`seed_mask.png` still lets the C parity test remain in its "fixtures
incomplete" skip path. Run on a CUDA machine for highest confidence.

## Gating

The SAM 3.1 parity test is the `SAM3_PARITY_VARIANT=sam3_1` variant
of `tests/test_video_parity_kids.c`, compiled only under
`SAM3_BUILD_PARITY_TESTS=ON`. Without that option the test is
excluded from the build. At runtime it skips cleanly when any of
`models/sam3.1.sam3`, `assets/kids.mp4`, or `seed_mask.png` are
absent.

## Assertions (C side)

- Frame-0 IoU (C vs seed_mask): warn if < 0.90, fail if < 0.50.
- Propagation frames 1..3: per-frame IoU ≥ 0.75 vs committed PNGs.

A failure on frame 1..3 indicates a regression in the memory
stream (Phase 2.5b items B1–B6 or later changes to the multiplex
mask decoder / memory-attn). Frame-0 drift indicates either a
change in the placeholder obj_ptr path or that the interactive
decoder (sub-project 3) has landed, in which case the fixture
needs regeneration.

## Current baseline (2026-04-20)

On the branch where this fixture was last regenerated:

    parity: frame-0 vs seed IoU=1.0000  # C dumper is deterministic
    parity: frame 1 IoU=0.0000          # complete divergence
    parity: frame 2 IoU=0.0000
    parity: frame 3 IoU=0.0000

### Root cause of the 0.0 IoU

The 0.0 IoU on propagation frames is *not* numerical drift — it's a
categorical divergence. Diagnostic data (`test_sam3_1_track` logs
the raw logits; `tools/sam3_1_dump_seed --propagate-frames 3
--frames-dir /tmp/c_frames` dumps C's propagation masks):

- **C frame-0 mask**: ~2.4% foreground (matches seed_mask.png
  exactly, IoU=1.0000).
- **C propagation frames 1..N**: every pixel is `-1024.0`
  (`SAM3_NO_OBJ_SCORE`) — i.e. the occlusion gate zeroed the entire
  mask because the multiplex mask decoder returned
  `obj_score_logit <= 0`.
- **Python frames 1..3**: ~6% foreground, localized to a plausible
  object region.

Python's gating uses the same `score > threshold (=0)` rule
(`video_tracking_multiplex.py:835`), so the gate itself isn't the
bug — the bug is that the *C* multiplex mask decoder predicts
`obj_score_logit <= 0` on every propagation frame, while Python
predicts `> 0`. Something upstream of the decoder's
`obj_score_head(sam_tokens[0])` is asymmetric between the two
implementations, possibly:

- the `extra_per_object` suppression vector (B4), esp. which slot
  is marked valid/invalid on propagation vs conditioning frames;
- the memory-attn output's impact on the `obj_score_token`
  position in the two-way transformer;
- the obj_ptr temporal positional encoding applied to memory
  tail rows (B3).

All three land in Phase 2.5b and the smoke test's mixed-sign
invariant only catches the degenerate frame-0 case. Root-causing
this is the next sub-project-2 task.

Default CI profile excludes this test
(`SAM3_BUILD_PARITY_TESTS=OFF`); opt-in users get the real signal
without breaking CI.

The per-frame IoU ≥ 0.75 threshold is intentionally kept at the
"should pass when correct" level rather than lowered to current
baseline — the goal is to track progress toward parity, not to
pin the failing state.
