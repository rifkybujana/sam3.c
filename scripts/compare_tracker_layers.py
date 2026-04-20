#!/usr/bin/env python3
"""Compare C-dump vs Python-reference dump for the SAM 3.1 tracker
per-layer binaries. First row with cosine < 0.99 is the bisection
divergence point.

Usage:
    python scripts/compare_tracker_layers.py

Expects dumps at /tmp/dbg_trk_*.bin (C side, from SAM3_DEBUG_DUMP) and
/tmp/py_trk_*.bin (Python side, from scripts/dump_tracker_layers.py).

Frame numbering note: C writes _f0/_f1/_f2 (frame 0 is the cond frame
with mask-decoder only; memory-attn path is skipped). Python writes
only _f1/_f2 (its add_new_masks seed path doesn't forward). This
script only compares the overlap -- frames 1 and 2.
"""
import os
import sys
import numpy as np


COS_THRESHOLD = 0.99


# (label, c_path, py_path, reshape_fn) where reshape_fn(arr, side) is
# called after load to bring both sides into a common shape. `side`
# is 'c' or 'py'.
def _reshape_masks(arr, side):
    # C: NHWC [16, 4H, 4W, 3]; Py: NCHW 5-D [1, 16, 3, 4H, 4W].
    # Target: flat [16 * 3 * 4H * 4W] for bytewise cos.
    # Compute 4H = sqrt(arr.size // (16*3)) to stay shape-agnostic.
    total = arr.size
    chw = total // 16   # = 3 * 4H * 4W
    k = 3
    hw = chw // k
    hw_side = int(round(hw ** 0.5))
    assert hw_side * hw_side == hw, f"masks not square: hw={hw}"
    if side == 'c':
        # C NHWC [16, H, W, K] -> [16, K, H, W]
        a = arr.reshape(16, hw_side, hw_side, k).transpose(0, 3, 1, 2)
    else:
        # Py [1, 16, K, H, W] -> [16, K, H, W]
        a = arr.reshape(1, 16, k, hw_side, hw_side).squeeze(0)
    return a.reshape(-1)


def _reshape_memattn(arr, side):
    # C: [1, 5184, 256]; Py: [5184, 1, 256]. Both contiguous, same bytes.
    return arr.reshape(5184, 256).reshape(-1)


def _reshape_tgt(arr, side):
    # Both sides are batch-first [1, HW=5184, 256] at encoder entry
    # (Py's encoder has batch_first=True so the forward-hook kwargs see
    # the pre-transpose layout). Identical reshape on both.
    return arr.reshape(5184, 256).reshape(-1)


def _reshape_bank(arr, side):
    # Bank tensors: C emits [1, Nm, 256]; Py emits [1, Nm, 256] too
    # (encoder has batch_first=True, so hook kwargs see pre-transpose).
    # Nm can differ between memory (includes obj_ptrs) and memory_image
    # (pre-pad) on the Py side — _report handles size-mismatch truncation.
    assert arr.size % 256 == 0, f"bank size {arr.size} not divisible by 256"
    Nm = arr.size // 256
    return arr.reshape(Nm, 256).reshape(-1)


def _reshape_iou(arr, side):
    return arr.reshape(16, 3).reshape(-1)


def _reshape_score(arr, side):
    return arr.reshape(16, 1).reshape(-1)


def _reshape_sam(arr, side):
    return arr.reshape(16, 3, 256).reshape(-1)


LEVEL0_FRAMES = [1, 2]  # overlap of C and Python dumps

PAIRS = []
for f in LEVEL0_FRAMES:
    PAIRS.extend([
        # Task 5γ bank-input slots: inputs to memory-attn layer 0.
        # `tgt` (= encoder's `src` kwarg) has fixed HW=5184 rows;
        # `memory` / `memory_image` / `memory_image_pos` have a
        # per-frame Nm — the reshape is size-agnostic and _report
        # truncates mismatched sides to their common row prefix.
        (f"tgt_f{f}",
         f"/tmp/dbg_trk_tgt_f{f}.bin",
         f"/tmp/py_trk_tgt_f{f}.bin",
         _reshape_tgt),
        (f"memory_f{f}",
         f"/tmp/dbg_trk_memory_f{f}.bin",
         f"/tmp/py_trk_memory_f{f}.bin",
         _reshape_bank),
        (f"memory_image_f{f}",
         f"/tmp/dbg_trk_memory_image_f{f}.bin",
         f"/tmp/py_trk_memory_image_f{f}.bin",
         _reshape_bank),
        (f"memory_image_pos_f{f}",
         f"/tmp/dbg_trk_memory_image_pos_f{f}.bin",
         f"/tmp/py_trk_memory_image_pos_f{f}.bin",
         _reshape_bank),
        (f"memattn_layer0_f{f}",
         f"/tmp/dbg_trk_memattn_layer0_f{f}.bin",
         f"/tmp/py_trk_memattn_layer0_f{f}.bin",
         _reshape_memattn),
        (f"memattn_layer1_f{f}",
         f"/tmp/dbg_trk_memattn_layer1_f{f}.bin",
         f"/tmp/py_trk_memattn_layer1_f{f}.bin",
         _reshape_memattn),
        (f"memattn_layer2_f{f}",
         f"/tmp/dbg_trk_memattn_layer2_f{f}.bin",
         f"/tmp/py_trk_memattn_layer2_f{f}.bin",
         _reshape_memattn),
        (f"memattn_layer3_f{f}",
         f"/tmp/dbg_trk_memattn_layer3_f{f}.bin",
         f"/tmp/py_trk_memattn_layer3_f{f}.bin",
         _reshape_memattn),
        (f"memattn_out_f{f}",
         f"/tmp/dbg_trk_memattn_out_f{f}.bin",
         f"/tmp/py_trk_memattn_out_f{f}.bin",
         _reshape_memattn),
        (f"mask_dec_masks_f{f}",
         f"/tmp/dbg_trk_mask_dec_masks_f{f}.bin",
         f"/tmp/py_trk_mask_dec_masks_f{f}.bin",
         _reshape_masks),
        (f"mask_dec_iou_f{f}",
         f"/tmp/dbg_trk_mask_dec_iou_f{f}.bin",
         f"/tmp/py_trk_mask_dec_iou_f{f}.bin",
         _reshape_iou),
        (f"mask_dec_score_f{f}",
         f"/tmp/dbg_trk_mask_dec_score_f{f}.bin",
         f"/tmp/py_trk_mask_dec_score_f{f}.bin",
         _reshape_score),
        (f"mask_dec_sam_f{f}",
         f"/tmp/dbg_trk_mask_dec_sam_f{f}.bin",
         f"/tmp/py_trk_mask_dec_sam_f{f}.bin",
         _reshape_sam),
    ])


def _load(path):
    if not os.path.exists(path):
        return None
    return np.fromfile(path, dtype=np.float32)


def _cosine(a, b):
    af, bf = a.flatten(), b.flatten()
    denom = np.linalg.norm(af) * np.linalg.norm(bf) + 1e-9
    return float(af @ bf / denom)


def _report(name, py, c, reshape_fn):
    if py is None or c is None:
        print(f"{name:40s} MISSING (py={py is not None}, c={c is not None})")
        return None  # treat as no signal
    size_note = ""
    if py.size != c.size:
        # Bank tensors (memory_image, memory_image_pos) differ in row
        # count: C pre-pads with obj_ptr rows; Py pre-pads inside the
        # encoder forward, so the hook sees the shorter pre-concat
        # tensor. Truncate both to the common row prefix (assumed
        # row-major with last dim = 256) so we still compare content.
        if reshape_fn is _reshape_bank and py.size % 256 == 0 and c.size % 256 == 0:
            py_rows = py.size // 256
            c_rows = c.size // 256
            common_rows = min(py_rows, c_rows)
            py = py[: common_rows * 256]
            c = c[: common_rows * 256]
            size_note = (f" [truncated to {common_rows} rows "
                         f"(py_rows={py_rows}, "
                         f"c_rows={c_rows})]")
        else:
            print(f"{name:40s} size mismatch py={py.size} c={c.size}")
            return False
    try:
        py_flat = reshape_fn(py, 'py')
        c_flat  = reshape_fn(c, 'c')
    except AssertionError as e:
        print(f"{name:40s} reshape failed: {e}")
        return False
    diff = py_flat - c_flat
    abs_diff = np.abs(diff)
    py_abs = np.abs(py_flat)
    cos = _cosine(py_flat, c_flat)
    marker = " <--- FIRST DIVERGENCE" if cos < COS_THRESHOLD else ""
    print(f"{name:40s} cos={cos:.5f} "
          f"abs_max={abs_diff.max():.4g} "
          f"abs_mean={abs_diff.mean():.4g} "
          f"rel={abs_diff.mean()/(py_abs.mean()+1e-9):.3%}"
          f"{marker}{size_note}")
    return cos >= COS_THRESHOLD


def main():
    print(f"# Tracker layer parity diff (cosine threshold: {COS_THRESHOLD})")
    print(f"# Frames compared: {LEVEL0_FRAMES} "
          f"(C also writes _f0 for mask-decoder only; no Python counterpart)")
    print()
    first_diverged = None
    for name, c_path, py_path, reshape_fn in PAIRS:
        c = _load(c_path)
        py = _load(py_path)
        ok = _report(name, py, c, reshape_fn)
        if ok is False and first_diverged is None:
            first_diverged = name

    print()
    if first_diverged:
        print(f"*** First divergent slot: {first_diverged}")
        print("    -> drill down with the level-1 dumps for that path")
    else:
        print("All level-0 slots match within threshold. If IoU still "
              "fails, expand LEVEL0 with finer-grained slots.")


if __name__ == "__main__":
    main()
