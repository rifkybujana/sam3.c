#!/usr/bin/env python3
"""
Slot-by-slot diff between the Python reference dump
(output/sam3_1_layer_dumps/) and the C reference dump
(output/sam3_1_c_layer_dumps/). For each overlapping slot, reports:

  shape_match | cos_sim | max_abs_diff | mean_abs_diff | rel_l2

Writes the raw per-slot table to stdout as JSON + a human-readable
Markdown section that can be pasted into REVIEW.md.

Cosine similarity is computed on flattened F32 contents (no reshape),
which is valid whenever shapes match — every slot is dumped with the
same convention on both sides (see the _NHWC_SLOTS set in both dump
scripts). For shape-mismatched slots we note the divergence but skip
numeric comparison.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))


def _load_bin(path):
    return np.fromfile(path, dtype=np.float32)


def _cos_sim(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _stats(py, c):
    diff = py - c
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    npy = float(np.linalg.norm(py))
    rel_l2 = (float(np.linalg.norm(diff)) / npy) if npy > 0 else float("nan")
    return _cos_sim(py, c), max_abs, mean_abs, rel_l2


# --- Category buckets ------------------------------------------------

def _category(slot):
    """Assign a slot name to a pipeline stage bucket."""
    if slot.startswith("frame_rgb"):
        return "00_input"
    if slot.startswith("vit_patch") or slot.startswith("vit_ln"):
        return "10_vit_entry"
    if slot.startswith("vit_block_"):
        return "11_vit_blocks"
    if slot.startswith("neck_") or slot.startswith("sam2_fpn"):
        return "20_neck"
    if slot.startswith("mdec_conv_"):
        return "21_mdec_high_res_side"
    if slot == "image_embed":
        return "30_image_embed"
    if slot.startswith("memattn_in_"):
        return "40_memattn_inputs"
    if slot.startswith("memattn_layer_"):
        return "41_memattn_layers"
    if slot.startswith("memattn_encoder_out") or slot == "memattn_final_norm":
        return "42_memattn_output"
    if slot.startswith("memattn_l0_"):
        return "43_memattn_l0_subdrill"
    if slot.startswith("mdec_out_"):
        return "60_mdec_outputs"
    if slot.startswith("mdec_"):
        return "50_mdec_internals"
    return "99_other"


def _slot_name_from_key(key):
    """'frame_0001/45_vit_ln_pre.bin' -> 'vit_ln_pre'."""
    fname = key.split("/", 1)[1]
    # fname = 'NN_<slot>.bin' -> strip 'NN_' and '.bin'
    if "_" in fname:
        stem = fname.split("_", 1)[1]
    else:
        stem = fname
    return stem.rsplit(".bin", 1)[0]


def _frame_from_key(key):
    return key.split("/", 1)[0]


def _dump_exists(root, key):
    return os.path.exists(os.path.join(root, "bins", key))


def _dual_load(py_root, c_root, key):
    return _load_bin(os.path.join(py_root, "bins", key)), \
           _load_bin(os.path.join(c_root, "bins", key))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py-dir",
                    default=os.path.join(_ROOT, "output",
                                         "sam3_1_layer_dumps"))
    ap.add_argument("--c-dir",
                    default=os.path.join(_ROOT, "output",
                                         "sam3_1_c_layer_dumps"))
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    py_shapes = json.load(
        open(os.path.join(args.py_dir, "shapes.json")))
    c_shapes = json.load(
        open(os.path.join(args.c_dir, "shapes.json")))

    py_keys = set(py_shapes)
    c_keys = set(c_shapes)
    overlap = sorted(py_keys & c_keys)
    only_py = sorted(py_keys - c_keys)
    only_c = sorted(c_keys - py_keys)

    rows = []
    for key in overlap:
        py_shape = py_shapes[key]
        c_shape = c_shapes[key]
        slot = _slot_name_from_key(key)
        frame = _frame_from_key(key)
        py_size = int(np.prod(py_shape))
        c_size = int(np.prod(c_shape))
        # On-disk byte size sanity-check
        py_fs = os.path.getsize(
            os.path.join(args.py_dir, "bins", key)) // 4
        c_fs = os.path.getsize(
            os.path.join(args.c_dir, "bins", key)) // 4
        size_match = (py_fs == c_fs)
        shape_match = (py_shape == c_shape)

        entry = {
            "frame": frame, "slot": slot,
            "category": _category(slot),
            "py_shape": py_shape, "c_shape": c_shape,
            "shape_match": shape_match,
            "py_nelts": py_fs, "c_nelts": c_fs,
            "size_match": size_match,
        }
        if size_match:
            py_a, c_a = _dual_load(args.py_dir, args.c_dir, key)
            cos, max_abs, mean_abs, rel_l2 = _stats(py_a, c_a)
            entry.update(cos_sim=cos, max_abs_diff=max_abs,
                         mean_abs_diff=mean_abs, rel_l2=rel_l2)
        rows.append(entry)

    # --- Grouped human-readable summary -----------------------------
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    print("# C vs Python per-layer diff (SAM 3.1 video tracker, "
          "3 frames)\n")
    print(f"- Python dump: `{os.path.relpath(args.py_dir, _ROOT)}`")
    print(f"- C dump:      `{os.path.relpath(args.c_dir, _ROOT)}`")
    print(f"- Overlapping slots: **{len(overlap)}** / "
          f"{len(py_keys)} (Py) / {len(c_keys)} (C)")
    n_shape_match = sum(1 for r in rows if r["shape_match"])
    print(f"- Shape matches: **{n_shape_match}/{len(rows)}**\n")

    for cat in sorted(by_cat.keys()):
        items = by_cat[cat]
        print(f"## `{cat}` ({len(items)} entries)\n")
        print("| frame | slot | shape_match | cos_sim | max_abs | "
              "mean_abs | rel_l2 |")
        print("|---|---|---|---|---|---|---|")
        for r in items:
            if r.get("cos_sim") is not None:
                cs = r.get("cos_sim")
                mx = r.get("max_abs_diff")
                mn = r.get("mean_abs_diff")
                rl = r.get("rel_l2")
                cs_s = f"{cs:.4f}" if cs == cs else "nan"
                mx_s = f"{mx:.3e}"
                mn_s = f"{mn:.3e}"
                rl_s = f"{rl:.3e}"
            else:
                cs_s = mx_s = mn_s = rl_s = "—"
            sm = "✓" if r["shape_match"] else f"✗ Py{r['py_shape']} / C{r['c_shape']}"
            print(f"| `{r['frame']}` | `{r['slot']}` | {sm} | "
                  f"{cs_s} | {mx_s} | {mn_s} | {rl_s} |")
        print()

    if only_py:
        print(f"## Only in Python ({len(only_py)})\n")
        for k in only_py:
            print(f"- `{k}`")
        print()
    if only_c:
        print(f"## Only in C ({len(only_c)})\n")
        for k in only_c:
            print(f"- `{k}`")
        print()

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({
                "rows": rows,
                "only_py": only_py,
                "only_c": only_c,
            }, f, indent=2, default=str)


if __name__ == "__main__":
    sys.exit(main() or 0)
