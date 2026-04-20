# SAM 3.1 Tracker Layer-Parity Debugging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive `test_video_parity_kids.c` (SAM3_PARITY_VARIANT=sam3_1) from its current failing baseline (per-frame IoU = 0.0 on frames 1–3) to green by layer-by-layer bisection between C and Python.

**Architecture:** Dump a small set of intermediate tensors from both the C tracker and Python reference on the same frame 0 + frame 1 inputs; diff via cosine similarity + abs error; drill into whichever top-level output first diverges; patch the first divergent C op; repeat until `test_video_parity_kids` passes.

**Tech Stack:** C11 (`SAM3_DEBUG_DUMP` + `sam3_dbg_*` + `auto_dump_tensor` in `src/model/sam3_video.c`), Python (`torch` forward hooks on `Sam3VideoTrackingMultiplexDemo`), CMake, existing `tools/sam3_1_dump_seed` + `tools/gen_video_parity_fixtures.py`.

**Spec:** `docs/superpowers/specs/2026-04-20-sam3-1-tracker-layer-parity-design.md`

---

## File Structure

- **Modify** `src/model/tracker_multiplex.c` — add `sam3_dbg_trk_*` extern declarations and assignments at the level-0 dump points (memory_attn output, mask decoder four outputs, memory bank tensor).
- **Modify** `src/model/sam3_video.c` — add `sam3_dbg_trk_*` extern globals in the existing `#ifdef SAM3_DEBUG_DUMP` block, plus per-frame `auto_dump_tensor` calls after `graph_eval`.
- **Create** `scripts/dump_tracker_layers.py` — Python driver: reuses the `gen_video_parity_fixtures.py` model-build + patches, registers forward hooks on the memory-attn and mask-decoder submodules, writes NHWC f32 binaries to `/tmp/py_trk_*.bin`.
- **Create** `scripts/compare_tracker_layers.py` — diff table script: per-pair cosine + abs_err max/mean + rel_err; pretty-prints first row with cosine < 0.99.
- **Create** `docs/superpowers/notes/2026-04-20-sam3-1-tracker-parity-log.md` (after Task 4) — running log of each bisection iteration's findings. One commit per iteration.

---

## Task 1 — Level-0 C dump slots

**Files:**
- Modify: `src/model/sam3_video.c:58-87` (extend existing `SAM3_DEBUG_DUMP` block)
- Modify: `src/model/tracker_multiplex.c` (around lines 2069-2140 — add hooks)
- Modify: `src/model/sam3_video.c` (call `auto_dump_tensor` after each frame's `graph_eval`)

- [ ] **Step 1: Add level-0 extern globals to `sam3_video.c`**

In `src/model/sam3_video.c`, locate the existing `#ifdef SAM3_DEBUG_DUMP` block at line 58. After the existing `sam3_dbg_xformer_layer0_q` declaration (around line 68), add:

```c
/* Tracker-path dump slots (level 0). Populated by tracker_multiplex.c
 * during each frame; flushed to /tmp/dbg_trk_<slot>_f<frame>.bin after
 * graph_eval. All tensors are F32 NHWC or flat 2-D / 3-D. */
struct sam3_tensor *sam3_dbg_trk_memattn_out       = NULL;
struct sam3_tensor *sam3_dbg_trk_mask_dec_masks    = NULL;
struct sam3_tensor *sam3_dbg_trk_mask_dec_iou      = NULL;
struct sam3_tensor *sam3_dbg_trk_mask_dec_score    = NULL;
struct sam3_tensor *sam3_dbg_trk_mask_dec_sam      = NULL;
struct sam3_tensor *sam3_dbg_trk_memory            = NULL;
struct sam3_tensor *sam3_dbg_trk_memory_image      = NULL;
struct sam3_tensor *sam3_dbg_trk_memory_image_pos  = NULL;
```

- [ ] **Step 2: Add hook assignments in `tracker_multiplex.c`**

In `src/model/tracker_multiplex.c`, inside `sam3_tracker_multiplex_track_frame`, find the block around lines 2069-2110 that builds `memory`, `memory_image`, `memory_image_pos` and calls `sam3_multiplex_memory_attn_forward`.

After line 2075 (the `multiplex_build_memory_from_bank` call) — but only on the success branch (after the fallback check at line 2076) — insert:

```c
#ifdef SAM3_DEBUG_DUMP
		{
			extern struct sam3_tensor *sam3_dbg_trk_memory;
			extern struct sam3_tensor *sam3_dbg_trk_memory_image;
			extern struct sam3_tensor *sam3_dbg_trk_memory_image_pos;
			sam3_dbg_trk_memory           = memory;
			sam3_dbg_trk_memory_image     = memory_image;
			sam3_dbg_trk_memory_image_pos = memory_image_pos;
		}
#endif
```

Then, just after the `cond_2d = sam3_multiplex_memory_attn_forward(...)` call (around line 2099), but before the `gh_reshape` at 2103, insert:

```c
#ifdef SAM3_DEBUG_DUMP
			{
				extern struct sam3_tensor *sam3_dbg_trk_memattn_out;
				sam3_dbg_trk_memattn_out = cond_2d;
			}
#endif
```

Finally, after the `sam3_multiplex_mask_decoder_forward` call at line 2136-2140 (after the err check at 2141), insert:

```c
#ifdef SAM3_DEBUG_DUMP
	{
		extern struct sam3_tensor *sam3_dbg_trk_mask_dec_masks;
		extern struct sam3_tensor *sam3_dbg_trk_mask_dec_iou;
		extern struct sam3_tensor *sam3_dbg_trk_mask_dec_score;
		extern struct sam3_tensor *sam3_dbg_trk_mask_dec_sam;
		sam3_dbg_trk_mask_dec_masks = all_masks;
		sam3_dbg_trk_mask_dec_iou   = all_iou;
		sam3_dbg_trk_mask_dec_score = all_score;
		sam3_dbg_trk_mask_dec_sam   = all_sam;
	}
#endif
```

- [ ] **Step 3: Add per-frame flush in `sam3_video.c`**

In `src/model/sam3_video.c`, find `video_track_one_obj` — the call to the backend's `graph_eval` lives there for the tracker pipeline. Locate the point immediately after `graph_eval` returns `SAM3_OK` for the tracker-multiplex path (grep for `is_multiplex` in that function; the frame-eval block is there). Add the flush block right after the successful `graph_eval`:

```c
#ifdef SAM3_DEBUG_DUMP
	if (is_multiplex) {
		char pbuf[256];
		#define DUMP_TRK(slot) do { \
			extern struct sam3_tensor *sam3_dbg_trk_##slot; \
			if (sam3_dbg_trk_##slot) { \
				snprintf(pbuf, sizeof(pbuf), \
					 "/tmp/dbg_trk_" #slot "_f%d.bin", \
					 frame_idx); \
				auto_dump_tensor(pbuf, sam3_dbg_trk_##slot); \
				sam3_dbg_trk_##slot = NULL; \
			} \
		} while (0)
		DUMP_TRK(memattn_out);
		DUMP_TRK(mask_dec_masks);
		DUMP_TRK(mask_dec_iou);
		DUMP_TRK(mask_dec_score);
		DUMP_TRK(mask_dec_sam);
		DUMP_TRK(memory);
		DUMP_TRK(memory_image);
		DUMP_TRK(memory_image_pos);
		#undef DUMP_TRK
	}
#endif
```

Locate the exact line: search for `session->variant == SAM3_VARIANT_SAM3_1` or similar inside `video_track_one_obj`, or grep for the `sam3_tracker_multiplex_track_frame` call there. The flush goes right after that eval, inside the same scope where `frame_idx` and `is_multiplex` are in scope.

- [ ] **Step 4: Build with SAM3_DEBUG_DUMP=ON**

Run:

```bash
cd /Users/rbisri/Documents/sam3/build && \
  cmake -DSAM3_DEBUG_DUMP=ON .. && \
  cmake --build . --target sam3_1_dump_seed -j8 2>&1 | tail -10
```

Expected: clean build. If the build fails with "use of undeclared identifier" on any `sam3_dbg_trk_*`, re-check that the matching `extern struct sam3_tensor *sam3_dbg_trk_<slot>;` is inside the `#ifdef SAM3_DEBUG_DUMP` block at each call site.

- [ ] **Step 5: Smoke-run to generate dump files**

```bash
cd /Users/rbisri/Documents/sam3 && \
  ulimit -s unlimited && \
  ./build/sam3_1_dump_seed \
    --model models/sam3.1.sam3 --video assets/kids.mp4 \
    --point 0.5,0.5,1 --out /tmp/seed_lvl0.png \
    --propagate-frames 2 --frames-dir /tmp/c_frames_lvl0 && \
  ls -l /tmp/dbg_trk_*.bin 2>&1
```

Expected output: `/tmp/dbg_trk_mask_dec_*_f0.bin` and `_f1.bin` files present; `/tmp/dbg_trk_memattn_out_f1.bin` present (NOT `_f0.bin` since frame 0 uses the no-mem path); `/tmp/dbg_trk_memory*_f1.bin` present.

- [ ] **Step 6: Commit**

```bash
git add src/model/sam3_video.c src/model/tracker_multiplex.c
git commit -m "$(cat <<'EOF'
debug: add level-0 tracker dump hooks under SAM3_DEBUG_DUMP

Captures memory-attn output, multiplex mask decoder's four outputs
(masks, iou, obj_score, sam_tokens), and the memory bank state
(memory, memory_image, memory_image_pos) on each frame for layer-
parity diffing against the Python reference. Per-frame suffix
(_f0, _f1, ...) lets us compare cond vs propagation state side by
side.

See docs/superpowers/specs/2026-04-20-sam3-1-tracker-layer-parity-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2 — Python-side dumper

**Files:**
- Create: `scripts/dump_tracker_layers.py`

- [ ] **Step 1: Write the dumper script**

Create `scripts/dump_tracker_layers.py`:

```python
#!/usr/bin/env python3
"""
Dump Python-reference intermediate tensors for frames 0-2 of
kids.mp4 (seeded with a C-produced seed_mask) so we can compare
layer-by-layer against our C engine's /tmp/dbg_trk_*.bin files.

Writes NHWC f32 raw-binary files matching the C dump format:
  /tmp/py_trk_memattn_out_f1.bin   [1, 5184, 256]
  /tmp/py_trk_mask_dec_masks_fN.bin
  /tmp/py_trk_mask_dec_iou_fN.bin
  /tmp/py_trk_mask_dec_score_fN.bin
  /tmp/py_trk_mask_dec_sam_fN.bin
  /tmp/py_trk_memory_f1.bin (+ memory_image, memory_image_pos)

Usage:
  SAM3_CKPT=models/sam3.1_multiplex.pt \\
    python scripts/dump_tracker_layers.py \\
      --video assets/kids.mp4 \\
      --seed-mask /tmp/seed_lvl0.png \\
      --frames 2
"""
import argparse
import os
import sys
import numpy as np

# Upstream sam3 package lives under reference/
sys.path.insert(
    0,
    os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "reference", "sam3")),
)
# Share the CPU patches with gen_video_parity_fixtures.py.
sys.path.insert(
    0,
    os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "tools")),
)
from _cpu_patches import (  # noqa: E402
    install_triton_stub, install_cuda_redirect, install_addmm_act_fp32,
)
install_triton_stub()
install_cuda_redirect()

import torch  # noqa: E402
from PIL import Image  # noqa: E402


# ---------------------------------------------------------------------
# Dump helpers
# ---------------------------------------------------------------------

def _dump(path, t):
    """Write a torch.Tensor (any shape) as f32 raw binary.

    NHWC transpose: 4-D tensors in NCHW layout are transposed to NHWC
    before writing so the C engine's NHWC dumps align pixel-for-pixel.
    """
    x = t.detach().cpu().float().contiguous()
    if x.dim() == 4:
        # Python reference is NCHW; C dumps are NHWC.
        x = x.permute(0, 2, 3, 1).contiguous()
    arr = x.numpy().astype(np.float32)
    arr.tofile(path)
    print(f"dump: {path} shape={tuple(arr.shape)} dtype={arr.dtype}",
          file=sys.stderr)


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

def _build_model(checkpoint):
    from sam3.model_builder import build_sam3_multiplex_video_model
    model = build_sam3_multiplex_video_model(
        checkpoint_path=None,
        load_from_HF=False,
        multiplex_count=16,
        use_fa3=False,
        use_rope_real=False,
        strict_state_dict_loading=False,
        device="cpu",
        compile=False,
    )
    install_addmm_act_fp32()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    merged = {}
    for k, v in ckpt.items():
        if k.startswith("tracker.model."):
            merged[k[len("tracker.model."):]] = v
    for k, v in ckpt.items():
        if k.startswith("detector.backbone."):
            merged[k[len("detector."):]] = v
    model.load_state_dict(merged, strict=False)
    return model.float().eval()


def _patches():
    """Same shims gen_video_parity_fixtures.py applies."""
    import tools.gen_video_parity_fixtures as _gvp  # type: ignore
    _gvp._patch_load_video_frames()
    _gvp._patch_forward_image_clone_loop()


def _init_state(model, video_path):
    from sam3.model.video_tracking_multiplex_demo import (
        VideoTrackingMultiplexDemo,
    )
    state = VideoTrackingMultiplexDemo.init_state(
        model,
        video_path=video_path,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=False,
        use_cv2=True,
    )
    if not torch.cuda.is_available():
        state["device"] = torch.device("cpu")
        state["storage_device"] = torch.device("cpu")
    return state


def _register_hooks(model, captures):
    """Register forward_hooks on memory-attention + mask decoder.

    `captures` is a dict keyed by slot name; each value is a list the
    hook appends to. After each frame we pop the last appended and
    write it to /tmp/py_trk_<slot>_f<frame>.bin.
    """
    hooks = []

    def _cap(name):
        def _h(_m, _inp, out):
            # Memory-attn output is a single tensor [B, N, D].
            # Mask decoder returns a tuple; hook receives the tuple out.
            captures.setdefault(name, []).append(out)
        return _h

    hooks.append(model.memory_attention.register_forward_hook(
        _cap("memattn_out")))
    hooks.append(model.sam_mask_decoder.register_forward_hook(
        _cap("mask_decoder_tuple")))
    return hooks


def _flush_captures(captures, frame_idx):
    """Dump whatever was captured this frame to /tmp/py_trk_*_f<frame>.bin."""
    # Memory-attn output
    if captures.get("memattn_out"):
        t = captures["memattn_out"].pop(0)
        _dump(f"/tmp/py_trk_memattn_out_f{frame_idx}.bin", t)

    # Mask decoder tuple: (low_res_multimasks, iou_pred, sam_tokens_out,
    #                      object_score_logits, ...).
    # Exact order varies by upstream version; check and adapt.
    if captures.get("mask_decoder_tuple"):
        tup = captures["mask_decoder_tuple"].pop(0)
        if isinstance(tup, (tuple, list)):
            # Guess by shape: masks are 4-D, iou is 2-D, sam is 3-D,
            # score is 2-D with last-dim=1.
            for t in tup:
                if not isinstance(t, torch.Tensor):
                    continue
                if t.dim() == 4:
                    _dump(f"/tmp/py_trk_mask_dec_masks_f{frame_idx}.bin", t)
                elif t.dim() == 3:
                    _dump(f"/tmp/py_trk_mask_dec_sam_f{frame_idx}.bin", t)
                elif t.dim() == 2 and t.size(-1) == 1:
                    _dump(f"/tmp/py_trk_mask_dec_score_f{frame_idx}.bin", t)
                elif t.dim() == 2:
                    _dump(f"/tmp/py_trk_mask_dec_iou_f{frame_idx}.bin", t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--seed-mask", required=True,
                    help="C-produced seed PNG (frame-0 mask)")
    ap.add_argument("--frames", type=int, default=2,
                    help="number of propagation frames to capture")
    args = ap.parse_args()

    _patches()
    model = _build_model(os.environ["SAM3_CKPT"])
    state = _init_state(model, args.video)

    seed = np.array(Image.open(args.seed_mask).convert("L"))
    seed_bool = (seed > 127).astype(np.float32)
    seed_t = torch.from_numpy(seed_bool)[None, ...]

    captures = {}
    hooks = _register_hooks(model, captures)
    try:
        model.add_new_masks(
            inference_state=state,
            frame_idx=0,
            obj_ids=[1],
            masks=seed_t,
        )
        _flush_captures(captures, 0)

        count = 0
        for frame_idx, obj_ids, _low, _vid, _score in \
                model.propagate_in_video(
                    inference_state=state,
                    start_frame_idx=0,
                    max_frame_num_to_track=args.frames + 1,
                    reverse=False):
            if frame_idx == 0:
                continue
            _flush_captures(captures, frame_idx)
            count += 1
            if count >= args.frames:
                break
    finally:
        for h in hooks:
            h.remove()

    print("done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke test with --help**

Run:

```bash
cd /Users/rbisri/Documents/sam3 && \
  python3 scripts/dump_tracker_layers.py --help
```

Expected: help text printed, exit 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/dump_tracker_layers.py
git commit -m "$(cat <<'EOF'
scripts: add dump_tracker_layers.py for Python-side layer dumps

Registers forward hooks on Sam3VideoTrackingMultiplexDemo's
memory_attention and sam_mask_decoder submodules, writes NHWC f32
binaries to /tmp/py_trk_*.bin for bisection diffs against the C
tracker's SAM3_DEBUG_DUMP output. Reuses gen_video_parity_fixtures
model-build + CPU patches.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3 — Python-side comparator

**Files:**
- Create: `scripts/compare_tracker_layers.py`

- [ ] **Step 1: Write the comparator**

Create `scripts/compare_tracker_layers.py`:

```python
#!/usr/bin/env python3
"""Compare C-dump vs Python-reference dump for the SAM 3.1 tracker
per-layer binaries. First row with cosine < 0.99 is the bisection
divergence point.
"""
import os
import sys
import numpy as np

# (py_path, c_path, shape)
LEVEL0 = [
    # Memory-attn output (frame 1 only; no cond-frame memory)
    ("/tmp/py_trk_memattn_out_f1.bin",
     "/tmp/dbg_trk_memattn_out_f1.bin", (1, 5184, 256)),
    # Mask decoder outputs — cond frame
    ("/tmp/py_trk_mask_dec_masks_f0.bin",
     "/tmp/dbg_trk_mask_dec_masks_f0.bin", (16, 3, 288, 288)),
    ("/tmp/py_trk_mask_dec_iou_f0.bin",
     "/tmp/dbg_trk_mask_dec_iou_f0.bin", (16, 3)),
    ("/tmp/py_trk_mask_dec_score_f0.bin",
     "/tmp/dbg_trk_mask_dec_score_f0.bin", (16, 1)),
    ("/tmp/py_trk_mask_dec_sam_f0.bin",
     "/tmp/dbg_trk_mask_dec_sam_f0.bin", (16, 3, 256)),
    # Mask decoder outputs — propagation frame (the known-broken case)
    ("/tmp/py_trk_mask_dec_masks_f1.bin",
     "/tmp/dbg_trk_mask_dec_masks_f1.bin", (16, 3, 288, 288)),
    ("/tmp/py_trk_mask_dec_iou_f1.bin",
     "/tmp/dbg_trk_mask_dec_iou_f1.bin", (16, 3)),
    ("/tmp/py_trk_mask_dec_score_f1.bin",
     "/tmp/dbg_trk_mask_dec_score_f1.bin", (16, 1)),
    ("/tmp/py_trk_mask_dec_sam_f1.bin",
     "/tmp/dbg_trk_mask_dec_sam_f1.bin", (16, 3, 256)),
]

# Level-1 drill-down pairs are appended by the caller after the
# matching extern-dump slots land in C.

COS_THRESHOLD = 0.99


def _load(path, shape):
    if not os.path.exists(path):
        return None
    a = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(shape))
    if a.size != expected:
        print(f"  WARN size mismatch: {a.size} vs {expected} for {path}",
              file=sys.stderr)
        return a
    return a.reshape(shape)


def _cosine(a, b):
    af, bf = a.flatten(), b.flatten()
    denom = np.linalg.norm(af) * np.linalg.norm(bf) + 1e-9
    return float(af @ bf / denom)


def _report(name, py, c):
    if py is None or c is None:
        print(f"{name:40s} MISSING (py={py is not None}, c={c is not None})")
        return False  # not compared — treat as "no signal"
    if py.shape != c.shape:
        print(f"{name:40s} shape mismatch py={py.shape} c={c.shape}")
        return False
    diff = py - c
    abs_diff = np.abs(diff)
    py_abs = np.abs(py)
    cos = _cosine(py, c)
    marker = " <--- FIRST DIVERGENCE" if cos < COS_THRESHOLD else ""
    print(f"{name:40s} cos={cos:.5f} "
          f"abs_max={abs_diff.max():.4g} "
          f"abs_mean={abs_diff.mean():.4g} "
          f"rel={abs_diff.mean()/(py_abs.mean()+1e-9):.3%}"
          f"{marker}")
    return cos >= COS_THRESHOLD


def main():
    first_diverged = None
    for py_path, c_path, shape in LEVEL0:
        name = (os.path.basename(py_path)
                .replace("py_trk_", "")
                .replace(".bin", ""))
        py = _load(py_path, shape)
        c = _load(c_path, shape)
        ok = _report(name, py, c)
        if not ok and first_diverged is None and py is not None and c is not None:
            first_diverged = name

    print()
    if first_diverged:
        print(f"*** First divergent slot: {first_diverged}")
        print("    → drill down with the level-1 dumps for that path")
    else:
        print("All level-0 slots match within threshold. If IoU still "
              "fails, expand LEVEL0 with finer-grained slots.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test (no dumps present)**

Run:

```bash
cd /Users/rbisri/Documents/sam3 && python3 scripts/compare_tracker_layers.py
```

Expected: every row prints "MISSING" (because no dumps yet), and the footer is "All level-0 slots match within threshold. If IoU still fails…". That's fine — nothing to compare yet.

- [ ] **Step 3: Commit**

```bash
git add scripts/compare_tracker_layers.py
git commit -m "$(cat <<'EOF'
scripts: add compare_tracker_layers.py

Diff table for the level-0 SAM 3.1 tracker dump pairs: cosine,
abs_err, rel. First row below the 0.99 cosine threshold is
highlighted as the bisection starting point.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4 — First bisection iteration

**Files:**
- Create: `docs/superpowers/notes/2026-04-20-sam3-1-tracker-parity-log.md`

- [ ] **Step 1: Regenerate C dumps + fixture seed**

```bash
cd /Users/rbisri/Documents/sam3/build && \
  cmake -DSAM3_DEBUG_DUMP=ON -DSAM3_BUILD_PARITY_TESTS=ON \
        -DSAM3_TEST_MODEL="$PWD/../models/sam3.1.sam3" \
        -DSAM3_PARITY_VARIANT=sam3_1 .. && \
  cmake --build . -j8 2>&1 | tail -5
```

Expected: clean build.

```bash
cd /Users/rbisri/Documents/sam3 && \
  ulimit -s unlimited && \
  ./build/sam3_1_dump_seed \
    --model models/sam3.1.sam3 --video assets/kids.mp4 \
    --point 0.5,0.5,1 --out /tmp/seed_lvl0.png \
    --propagate-frames 2 --frames-dir /tmp/c_frames_lvl0 && \
  ls /tmp/dbg_trk_*.bin
```

Expected: `/tmp/dbg_trk_*_f0.bin` and `_f1.bin` files present. If any slot is missing check the corresponding hook placement from Task 1 Step 2.

- [ ] **Step 2: Run the Python dumper with the same seed**

```bash
cd /Users/rbisri/Documents/sam3 && \
  SAM3_CKPT=models/sam3.1_multiplex.pt \
    python3 scripts/dump_tracker_layers.py \
      --video assets/kids.mp4 \
      --seed-mask /tmp/seed_lvl0.png \
      --frames 2 2>&1 | tail -15 && \
  ls /tmp/py_trk_*.bin
```

Expected: `/tmp/py_trk_*.bin` files present matching the C dump slot names.

- [ ] **Step 3: Run the comparator**

```bash
cd /Users/rbisri/Documents/sam3 && \
  python3 scripts/compare_tracker_layers.py 2>&1 | tee /tmp/compare_lvl0.log
```

Expected: a table with per-slot cosine. At least one row on a frame-1 slot is expected to be under 0.99. The footer identifies the first-divergent slot.

- [ ] **Step 4: Record findings in the parity log**

Create `docs/superpowers/notes/2026-04-20-sam3-1-tracker-parity-log.md`:

```markdown
# SAM 3.1 Tracker Parity Debugging Log

Running log of bisection iterations. One section per round.

## Iteration 1 — level-0 coarse dumps (2026-04-20)

### Setup

- Branch: `feature/sam3.1-image-path`
- Seed: `/tmp/seed_lvl0.png` (sam3_1_dump_seed frame-0 output,
  288x288, ~2.4% foreground)
- Frames dumped: 0 (cond) and 1 (first propagation)

### Results

Paste the compare_tracker_layers.py output here.

### Analysis

- Identify the first row with cos < 0.99.
- Which of the three drill-down paths does that point to?
  - memattn_out diverged → Path α (memory-attn per-layer)
  - mask_dec_* diverged but memattn matched → Path β (two-way transformer)
  - memory/memory_image/memory_image_pos diverged → Path γ (bank builder)

### Next step

Proceed to Task 5 with Path { α | β | γ }.
```

Fill in the Results and Analysis sections from the `/tmp/compare_lvl0.log` output.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/notes/2026-04-20-sam3-1-tracker-parity-log.md
git commit -m "$(cat <<'EOF'
docs/notes: SAM 3.1 tracker parity iteration 1 (level-0 dumps)

Level-0 coarse tensor diff between C (SAM3_DEBUG_DUMP) and Python
(dump_tracker_layers.py) on frame 0 (cond) and frame 1 (propagation).
Identifies which of the three drill paths to pursue in iteration 2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5 — Level-1 drill (one of paths α, β, γ — determined by Task 4)

**The path selected by Task 4's output determines which sub-task to execute.** Each path below is independent; execute only the one the level-0 diff pointed at. If a later iteration changes the first-divergent slot, revisit the corresponding path.

### Task 5α — Memory-attention per-layer dumps

**Files:**
- Modify: `src/model/sam3_video.c` (extend SAM3_DEBUG_DUMP block)
- Modify: `src/model/tracker_multiplex.c` (hook inside memory_attn_layer loop)
- Modify: `scripts/dump_tracker_layers.py` (add hooks on each decoder layer)
- Modify: `scripts/compare_tracker_layers.py` (extend LEVEL0 list)

- [ ] **Step 1: Extend C dump slots**

In `src/model/sam3_video.c`, inside the same `#ifdef SAM3_DEBUG_DUMP` block, add four per-layer slots:

```c
struct sam3_tensor *sam3_dbg_trk_memattn_layer0 = NULL;
struct sam3_tensor *sam3_dbg_trk_memattn_layer1 = NULL;
struct sam3_tensor *sam3_dbg_trk_memattn_layer2 = NULL;
struct sam3_tensor *sam3_dbg_trk_memattn_layer3 = NULL;
```

- [ ] **Step 2: Hook each layer in tracker_multiplex.c**

In `src/model/tracker_multiplex.c`, find the 4-layer memory-attn loop
at `sam3_multiplex_memory_attn_forward` around lines 1102-1110:

```c
for (int i = 0; i < 4; i++) {
    output = memory_attn_layer(g, arena, &ma->layers[i], ...);
    if (!output)
        return NULL;
}
```

Replace the loop body with:

```c
	for (int i = 0; i < 4; i++) {
		output = memory_attn_layer(g, arena, &ma->layers[i],
			output, image, memory, memory_image,
			memory_image_pos, cos_q, sin_q, cos_k, sin_k,
			num_k_exclude_rope);
		if (!output)
			return NULL;
#ifdef SAM3_DEBUG_DUMP
		{
			extern struct sam3_tensor *sam3_dbg_trk_memattn_layer0;
			extern struct sam3_tensor *sam3_dbg_trk_memattn_layer1;
			extern struct sam3_tensor *sam3_dbg_trk_memattn_layer2;
			extern struct sam3_tensor *sam3_dbg_trk_memattn_layer3;
			struct sam3_tensor **slots[] = {
				&sam3_dbg_trk_memattn_layer0,
				&sam3_dbg_trk_memattn_layer1,
				&sam3_dbg_trk_memattn_layer2,
				&sam3_dbg_trk_memattn_layer3,
			};
			*slots[i] = output;
		}
#endif
	}
```

- [ ] **Step 3: Extend the per-frame flush in sam3_video.c**

Add four more `DUMP_TRK()` invocations inside the existing `#ifdef SAM3_DEBUG_DUMP` block in `video_track_one_obj`:

```c
		DUMP_TRK(memattn_layer0);
		DUMP_TRK(memattn_layer1);
		DUMP_TRK(memattn_layer2);
		DUMP_TRK(memattn_layer3);
```

- [ ] **Step 4: Extend Python-side hooks**

In `scripts/dump_tracker_layers.py`, inside `_register_hooks`, add per-layer hooks on the 4-layer decoupled transformer (upstream module path: `model.memory_attention.encoder.layers[i]`):

```python
    for i, layer in enumerate(model.memory_attention.encoder.layers):
        hooks.append(layer.register_forward_hook(
            _cap(f"memattn_layer{i}")))
```

Also extend `_flush_captures` to emit the layer dumps:

```python
    for i in range(4):
        key = f"memattn_layer{i}"
        if captures.get(key):
            t = captures[key].pop(0)
            _dump(f"/tmp/py_trk_{key}_f{frame_idx}.bin", t)
```

- [ ] **Step 5: Extend the comparator LEVEL0 list**

Append to the LEVEL0 list in `scripts/compare_tracker_layers.py`:

```python
    # Memory-attn per-layer (level-1 path α)
    ("/tmp/py_trk_memattn_layer0_f1.bin",
     "/tmp/dbg_trk_memattn_layer0_f1.bin", (1, 5184, 256)),
    ("/tmp/py_trk_memattn_layer1_f1.bin",
     "/tmp/dbg_trk_memattn_layer1_f1.bin", (1, 5184, 256)),
    ("/tmp/py_trk_memattn_layer2_f1.bin",
     "/tmp/dbg_trk_memattn_layer2_f1.bin", (1, 5184, 256)),
    ("/tmp/py_trk_memattn_layer3_f1.bin",
     "/tmp/dbg_trk_memattn_layer3_f1.bin", (1, 5184, 256)),
```

- [ ] **Step 6: Rebuild, re-run, update the log**

```bash
cd /Users/rbisri/Documents/sam3/build && \
  cmake --build . --target sam3_1_dump_seed -j8 2>&1 | tail -5 && \
  cd .. && \
  ulimit -s unlimited && \
  ./build/sam3_1_dump_seed \
    --model models/sam3.1.sam3 --video assets/kids.mp4 \
    --point 0.5,0.5,1 --out /tmp/seed_lvl0.png \
    --propagate-frames 2 --frames-dir /tmp/c_frames_lvl0 && \
  SAM3_CKPT=models/sam3.1_multiplex.pt \
    python3 scripts/dump_tracker_layers.py \
      --video assets/kids.mp4 \
      --seed-mask /tmp/seed_lvl0.png --frames 2 && \
  python3 scripts/compare_tracker_layers.py 2>&1 | tee /tmp/compare_lvl1_alpha.log
```

- [ ] **Step 7: Append iteration 2 to the parity log**

In `docs/superpowers/notes/2026-04-20-sam3-1-tracker-parity-log.md`, add:

```markdown
## Iteration 2 — path α: memory-attn per-layer (YYYY-MM-DD)

### Results

Paste the compare_tracker_layers.py output here.

### Analysis

- First divergent layer: `memattn_layer{N}`.
- Hypothesis: examine `memory_attn_layer()` in
  `src/model/tracker_multiplex.c` — focus on self-attn vs cross-attn
  projections, the decoupled RoPE application to image+memory K, and
  the num_k_exclude_rope slice.
```

- [ ] **Step 8: Commit**

```bash
git add src/model/sam3_video.c src/model/tracker_multiplex.c \
        scripts/dump_tracker_layers.py scripts/compare_tracker_layers.py \
        docs/superpowers/notes/2026-04-20-sam3-1-tracker-parity-log.md
git commit -m "$(cat <<'EOF'
debug: level-1 path alpha - memory-attn per-layer dumps

Adds 4 per-layer output slots (memattn_layer0..3) on both C and
Python sides. Iteration 2 of bisection.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 5β — Mask decoder two-way transformer dumps

**Files:**
- Modify: `src/model/sam3_video.c`
- Modify: `src/model/tracker_multiplex.c` (inside the two-way transformer block)
- Modify: `scripts/dump_tracker_layers.py`
- Modify: `scripts/compare_tracker_layers.py`

- [ ] **Step 1: Extend C dump slots**

In `src/model/sam3_video.c`, add:

```c
struct sam3_tensor *sam3_dbg_trk_mask_dec_layer0_queries   = NULL;
struct sam3_tensor *sam3_dbg_trk_mask_dec_layer0_keys      = NULL;
struct sam3_tensor *sam3_dbg_trk_mask_dec_layer1_queries   = NULL;
struct sam3_tensor *sam3_dbg_trk_mask_dec_layer1_keys      = NULL;
struct sam3_tensor *sam3_dbg_trk_mask_dec_extra_per_object = NULL;
struct sam3_tensor *sam3_dbg_trk_mask_dec_sam_token0       = NULL;
```

- [ ] **Step 2: Hook the two-way transformer block in tracker_multiplex.c**

Find the two-way transformer in `sam3_multiplex_mask_decoder_forward`
(grep for "two_way" / "TwoWayAttentionBlock" starting at line 1368).
Inside the per-layer block, add `sam3_dbg_trk_mask_dec_layer{0,1}_{queries,keys}` assignments after the layer computes its updated queries and keys.

Capture the sam_token[0] slice specifically before it's fed to `obj_score_head`. Search for `obj_score_head` or `obj_score_mlp` in `sam3_multiplex_mask_decoder_forward` and add the hook on the tensor passed in.

- [ ] **Step 3: Hook extra_per_object input**

In `tracker_multiplex.c`, at the point where `extra_per_object` is built (around line 2115-2129), add:

```c
#ifdef SAM3_DEBUG_DUMP
	{
		extern struct sam3_tensor *sam3_dbg_trk_mask_dec_extra_per_object;
		sam3_dbg_trk_mask_dec_extra_per_object = extra_per_object;
	}
#endif
```

- [ ] **Step 4: Extend the per-frame flush in sam3_video.c**

Add the matching `DUMP_TRK()` invocations:

```c
		DUMP_TRK(mask_dec_layer0_queries);
		DUMP_TRK(mask_dec_layer0_keys);
		DUMP_TRK(mask_dec_layer1_queries);
		DUMP_TRK(mask_dec_layer1_keys);
		DUMP_TRK(mask_dec_extra_per_object);
		DUMP_TRK(mask_dec_sam_token0);
```

- [ ] **Step 5: Extend the Python hooks**

In `scripts/dump_tracker_layers.py`, register hooks on
`model.sam_mask_decoder.transformer.layers[{0,1}]` to capture their
(queries, keys) tuple return. Adapt `_flush_captures` to unpack and
dump each tensor separately. Also register a hook on the module
that emits the sam_token[0] before the obj_score head — upstream
path is typically
`model.sam_mask_decoder.transformer` output[0] (queries), which
after the attention block is the token sequence where tokens[0] is
the obj-score token.

```python
    hooks.append(model.sam_mask_decoder.transformer.register_forward_hook(
        _cap("mask_dec_transformer_out")))
```

In `_flush_captures`, after the tuple capture:

```python
    if captures.get("mask_dec_transformer_out"):
        out = captures["mask_dec_transformer_out"].pop(0)
        # Upstream returns (queries, keys); tokens[0,:,0,:] is the
        # obj_score_token.
        if isinstance(out, (tuple, list)) and len(out) >= 1:
            queries = out[0]
            _dump(f"/tmp/py_trk_mask_dec_sam_token0_f{frame_idx}.bin",
                  queries[:, 0:1, :])
```

- [ ] **Step 6: Extend the comparator LEVEL0 list**

Append:

```python
    # Mask decoder internals (level-1 path β)
    ("/tmp/py_trk_mask_dec_extra_per_object_f1.bin",
     "/tmp/dbg_trk_mask_dec_extra_per_object_f1.bin", (16, 256)),
    ("/tmp/py_trk_mask_dec_sam_token0_f1.bin",
     "/tmp/dbg_trk_mask_dec_sam_token0_f1.bin", (16, 1, 256)),
```

- [ ] **Step 7: Rebuild, re-run, log, commit**

Follow the same rebuild/re-run/log/commit sequence as Task 5α steps 6-8, substituting path β for path α and appending "Iteration 2 — path β" to the log.

### Task 5γ — Memory bank content dumps

**Files:**
- Modify: `src/model/sam3_video.c`
- Modify: `src/model/tracker_multiplex.c` (inside `multiplex_build_memory_from_bank`)
- Modify: `scripts/dump_tracker_layers.py`
- Modify: `scripts/compare_tracker_layers.py`

- [ ] **Step 1: Extend C dump slots**

In `src/model/sam3_video.c`, add:

```c
struct sam3_tensor *sam3_dbg_trk_bank_maskmem_entry0 = NULL;
struct sam3_tensor *sam3_dbg_trk_bank_objptr_row0    = NULL;
struct sam3_tensor *sam3_dbg_trk_bank_tpos_entry0    = NULL;  /* maskmem_tpos_enc[0] applied */
```

- [ ] **Step 2: Hook inside `multiplex_build_memory_from_bank`**

In `src/model/tracker_multiplex.c:1779`+, add hooks inside the loop that writes each entry's block into `memory` / `memory_image` / `memory_image_pos`. The first entry's outputs are the most informative (they correspond to frame 0's cond output which we've already verified matches).

Pick specific slices — e.g. after writing the first maskmem block:

```c
#ifdef SAM3_DEBUG_DUMP
	if (entry_idx == 0) {
		/* Capture the first maskmem entry as written into `memory`.
		 * The block is contiguous [entry_rows, 256] starting at
		 * `memory` offset entry_idx * entry_rows * 256. */
		extern struct sam3_tensor *sam3_dbg_trk_bank_maskmem_entry0;
		int dims[2] = {entry_rows, 256};
		struct sam3_tensor *view = gh_alloc_tensor(arena,
				SAM3_DTYPE_F32, 2, dims);
		if (view) {
			memcpy(view->data,
			       (float *)memory->data
			         + (size_t)entry_idx * entry_rows * 256,
			       (size_t)entry_rows * 256 * sizeof(float));
			sam3_dbg_trk_bank_maskmem_entry0 = view;
		}
	}
#endif
```

(Adapt variable names — `entry_rows`, `entry_idx` — to match the actual loop in that function; grep for the loop body to align.)

- [ ] **Step 3: Extend the per-frame flush**

Add the matching `DUMP_TRK()` invocations:

```c
		DUMP_TRK(bank_maskmem_entry0);
		DUMP_TRK(bank_objptr_row0);
		DUMP_TRK(bank_tpos_entry0);
```

- [ ] **Step 4: Extend the Python dumper**

In `scripts/dump_tracker_layers.py`, after `propagate_in_video` yields frame 1, inspect `state["output_dict_per_obj"]` to pull the frame-0 maskmem tokens and obj_ptr that Python stored for obj_id=1. Dump them directly (no forward hook needed):

```python
        # State-based dump of what Python stored in the memory bank
        # for obj 1 after frame 0's cond run.
        obj_output = state["output_dict_per_obj"][0]["cond_frame_outputs"].get(0)
        if obj_output:
            maskmem = obj_output.get("maskmem_features")
            if maskmem is not None:
                _dump("/tmp/py_trk_bank_maskmem_entry0_f1.bin", maskmem)
            objptr = obj_output.get("obj_ptr")
            if objptr is not None:
                _dump("/tmp/py_trk_bank_objptr_row0_f1.bin", objptr)
```

(The exact keys of `obj_output` depend on the upstream — grep for `cond_frame_outputs` in `video_tracking_multiplex_demo.py` to confirm.)

- [ ] **Step 5: Extend the comparator**

Append to LEVEL0:

```python
    # Memory bank state (level-1 path γ)
    ("/tmp/py_trk_bank_maskmem_entry0_f1.bin",
     "/tmp/dbg_trk_bank_maskmem_entry0_f1.bin", (4096, 64)),
    ("/tmp/py_trk_bank_objptr_row0_f1.bin",
     "/tmp/dbg_trk_bank_objptr_row0_f1.bin", (1, 256)),
```

- [ ] **Step 6: Rebuild, re-run, log, commit**

Same sequence as Task 5α steps 6-8 but labeled "Iteration 2 — path γ".

---

## Task 6 — Patch the first divergent C op

**Scope:** This is the payoff task. What the patch looks like depends entirely on what the level-1 diff revealed. The plan captures the general procedure; concrete code goes in the matching commit.

**Files:** Variable — whichever `src/model/tracker_multiplex.c` function (or rarely `src/model/sam3_video.c`) hosts the first-divergent op.

- [ ] **Step 1: Identify the specific op from the diff**

From `compare_tracker_layers.py` output, the lowest-level divergent slot points to a function or even a specific op inside it. Open that function in `src/model/tracker_multiplex.c` and identify:
- What Python's reference does at that point (look at the matching Python module in `reference/sam3/sam3/model/`).
- What the C code does.
- The specific divergence (wrong index, transposed tensor, missing scalar, NHWC/NCHW confusion).

- [ ] **Step 2: Write a focused C patch**

Apply the minimum change that aligns C with Python. Do NOT take the opportunity to refactor adjacent code — keep the diff small so the cause-and-effect is clear.

Example (hypothetical — replace with the real fix when Task 4-5 pin it):

```c
/* Before */
extra_per_object[slot * 256 + c] = invalid[c];

/* After — slot 0 is the valid slot; all others use invalid */
extra_per_object[slot * 256 + c] =
    (slot == 0) ? valid[c] : invalid[slot * 256 + c];
```

- [ ] **Step 3: Build**

```bash
cd /Users/rbisri/Documents/sam3/build && cmake --build . -j8 2>&1 | tail -5
```

Expected: clean build.

- [ ] **Step 4: Re-run the level-1 dumps + comparator**

```bash
cd /Users/rbisri/Documents/sam3 && \
  ulimit -s unlimited && \
  ./build/sam3_1_dump_seed \
    --model models/sam3.1.sam3 --video assets/kids.mp4 \
    --point 0.5,0.5,1 --out /tmp/seed_lvl0.png \
    --propagate-frames 2 --frames-dir /tmp/c_frames_lvl0 && \
  SAM3_CKPT=models/sam3.1_multiplex.pt \
    python3 scripts/dump_tracker_layers.py \
      --video assets/kids.mp4 --seed-mask /tmp/seed_lvl0.png --frames 2 && \
  python3 scripts/compare_tracker_layers.py
```

Expected outcomes:
- Best case: the previously-divergent slot's cosine climbs above 0.99, and downstream slots (including `mask_dec_score_f1`) also recover. IoU should improve on the next parity-test run.
- Partial case: the first divergent slot moves. A different op is now the next bottleneck — re-enter Task 5 for that path.
- Regression case: a previously-matching slot now diverges. The fix was wrong or overbroad; revert and retry.

- [ ] **Step 5: Smoke test — ensure no SAM 3 / smoke regressions**

```bash
cd /Users/rbisri/Documents/sam3/build && ctest -E "test_video_parity_kids|test_fixture_compare" --output-on-failure 2>&1 | tail -20
```

Expected: 73/73 tests pass (the parity test is excluded here because it's the gate; `test_fixture_compare` is the pre-existing NaN failure flagged in TODO.md sub-project 1 polish and is unrelated).

- [ ] **Step 6: Append iteration N to the parity log**

In `docs/superpowers/notes/2026-04-20-sam3-1-tracker-parity-log.md`:

```markdown
## Iteration N — fix (YYYY-MM-DD)

### Patch

<brief description of the op that was fixed>
File: src/model/tracker_multiplex.c:<line>
Diff: <paste the minimal diff>

### Post-fix comparator output

Paste the relevant rows from compare_tracker_layers.py.

### Next step

- All slots above 0.99 → run parity test (Task 7).
- New divergent slot → enter Task 5 for the matching path.
```

- [ ] **Step 7: Commit**

```bash
git add src/model/tracker_multiplex.c \
        docs/superpowers/notes/2026-04-20-sam3-1-tracker-parity-log.md
git commit -m "$(cat <<'EOF'
tracker/multiplex: fix <specific op> on propagation frames

<1-2 sentences on what was wrong>. Verified via layer-parity diff
(see docs/superpowers/notes/2026-04-20-sam3-1-tracker-parity-log.md
iteration N) — the previously-divergent slot now matches Python at
cos >= 0.99.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 8: Repeat Tasks 4-6 until all level-0 slots match**

Each Task 6 iteration advances the bisection. Continue until
`compare_tracker_layers.py` shows every level-0 slot at cos ≥ 0.99.
Then proceed to Task 7.

---

## Task 7 — Terminal validation

**Files:** None modified (this task validates the work).

- [ ] **Step 1: Ensure DEBUG_DUMP is off for the parity test run**

The parity test doesn't care about dumps; strip them from the build to avoid stale-binary confusion:

```bash
cd /Users/rbisri/Documents/sam3/build && \
  cmake -DSAM3_DEBUG_DUMP=OFF -DSAM3_BUILD_PARITY_TESTS=ON \
        -DSAM3_TEST_MODEL="$PWD/../models/sam3.1.sam3" \
        -DSAM3_PARITY_VARIANT=sam3_1 .. && \
  cmake --build . --target test_video_parity_kids -j8 2>&1 | tail -5
```

Expected: clean build.

- [ ] **Step 2: Run the parity test**

```bash
cd /Users/rbisri/Documents/sam3/build && \
  ulimit -s unlimited && \
  ctest -R test_video_parity_kids --output-on-failure 2>&1 | tail -30
```

Expected: PASS. If any per-frame IoU is still below 0.75, return to Task 5 (the bisection isn't done).

- [ ] **Step 3: Run the full suite once**

```bash
cd /Users/rbisri/Documents/sam3/build && \
  ctest --output-on-failure 2>&1 | tail -15
```

Expected: everything except the known pre-existing `test_fixture_compare` failure passes.

- [ ] **Step 4: Final log entry**

Close the parity log with a summary section:

```markdown
## Conclusion

- First-divergent root cause(s): <list>
- Fix commits: <list of short-sha + subject lines>
- `test_video_parity_kids` SAM 3.1 variant: PASS at per-frame IoU
  {F1, F2, F3}
```

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/notes/2026-04-20-sam3-1-tracker-parity-log.md
git commit -m "$(cat <<'EOF'
docs/notes: SAM 3.1 tracker parity — close the loop

test_video_parity_kids now passes the SAM 3.1 variant at per-frame
IoU >= 0.75. Root causes + fix commits documented in the parity log.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: Update TODO.md (local-only; not committed)**

In `TODO.md`, under sub-project 2's Phase 2.5b section, add a final
entry noting that propagation-frame parity against the Python
reference is now green, referencing the new parity log at
`docs/superpowers/notes/2026-04-20-sam3-1-tracker-parity-log.md`
and the closing commit.

---

## Self-Review

**1. Spec coverage**
- §3.1 bisection strategy → Tasks 4-5-6 iterative loop.
- §3.2 level-0 (6 tensors) → Task 1 adds exactly those slots.
- §3.3 level-1 drill paths (α/β/γ) → Task 5 has explicit subtasks for all three.
- §3.4 iteration loop → Tasks 4, 5, 6 are the loop; Task 7 is the exit.
- §3.5 thresholds → Task 3 codifies `COS_THRESHOLD = 0.99`; Task 7 gates on IoU ≥ 0.75.
- §4.1 new files → Task 2 creates `dump_tracker_layers.py`; Task 3 creates `compare_tracker_layers.py`; Task 4 creates the log file.
- §4.2 level-0 modifications → Task 1 steps 1-3 cover `sam3_video.c` + `tracker_multiplex.c`.
- §4.3 level-1 drill files → Task 5α/β/γ touch matching files.
- §4.4 files NOT modified → confirmed: no changes planned to Python reference, `gen_video_parity_fixtures.py`, or `test_video_parity_kids.c`.
- §6 edge cases → Task 1 step 5 handles missing slots; Task 6 step 5 handles regressions.
- §10 deliverables → all 9 items mapped to tasks.

**2. Placeholder scan**
- Task 5γ step 2 has "Adapt variable names to match the actual loop in that function" — that's a hedge rather than full code, but the outer structure is complete and the hedge is honest about C-code shape uncertainty that's only resolvable with the source in hand. Acceptable.
- Task 6 is intentionally shape-only because its code depends on Task 4-5 findings; this is the plan admitting debugging is not fully prescribable. The steps themselves (identify → patch → rebuild → re-diff → regression-check → log → commit) are concrete.

**3. Type consistency**
- `sam3_dbg_trk_*` slots defined in Task 1 are referenced by the same names in Tasks 5α/β/γ.
- `DUMP_TRK(...)` macro defined once in Task 1 step 3 and reused by name in Tasks 5α/β/γ step 3.
- Python-side `captures` dict keys (`memattn_out`, `mask_decoder_tuple`, `memattn_layer{0..3}`) are consistent between `_register_hooks` and `_flush_captures`.
- LEVEL0 list entries in Task 3 are extended (not overwritten) by Task 5 subtasks.
