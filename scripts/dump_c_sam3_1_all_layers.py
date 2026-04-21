#!/usr/bin/env python3
"""
Dump per-layer inputs/outputs from the C SAM 3.1 video tracker for
the first 3 frames of a video (default: assets/kids.mp4). Mirrors the
output layout of `scripts/dump_sam3_1_all_layers.py` (Python reference
dump) so C and Python can be compared slot-for-slot.

Approach:
  1. Run `./build/sam3_1_dump_seed --propagate-frames 2 --frames-dir ...`
     with a center point prompt (same as the committed fixture seed).
     That tool is already wired to write PNGs for frame 0 (seed) and
     frames 1..N (propagation) and — with -DSAM3_DEBUG_DUMP — to emit
     raw-F32 binaries for every intermediate tensor at /tmp/dbg_*.bin.
  2. Reorganize /tmp/dbg_trk_<slot>_f<N>.bin (frame-indexed) and
     /tmp/dbg_vit_*.bin / dbg_neck_*.bin / dbg_sam2_*.bin
     (non-indexed; last frame only) into
     output/sam3_1_c_layer_dumps/bins/frame_{NNNN}/NN_<pyname>.bin.
  3. Infer shapes from the committed Python shapes.json (same
     architecture, so matching slots must share shapes) plus a
     fallback heuristic on file size.
  4. Decode input RGB frames from kids.mp4 with ffmpeg and write
     each as pngs/frame_{NNNN}_input.png — matches the Python PNG
     layout.

Gaps relative to Python (documented in README.md):
  - per-frame ViT block dumps are shared across frames (image encoder's
    dumps in src/model/image_encoder.c use non-indexed paths); we
    attribute them to the last frame only. C-side per-frame ViT dumps
    would require frame-suffix plumbing in image_encoder.c.
  - several decoder sub-slots (mdec_twt_*, mdec_upscale_*, mdec_iou/
    obj_score_head, mdec_conv_s0/s1, memattn_final_norm,
    neck_sam3_conv_*, neck_prop_conv_*) are not yet hooked in C.

Usage:
  python3 scripts/dump_c_sam3_1_all_layers.py \\
    --model models/sam3.1.sam3 \\
    --video assets/kids.mp4 \\
    --binary build/sam3_1_dump_seed
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))


# ---------------------------------------------------------------------
# Slot mapping — C dump name -> Python slot name
# ---------------------------------------------------------------------
#
# Per-frame slots: sam3_video.c writes these as /tmp/dbg_trk_<C>_f<N>.bin.
# Non-indexed slots: image_encoder.c / sam3_image.c write to fixed paths;
# they reflect the LAST frame processed (frame 2 here).
#
# Some Python slots have no direct C analog — noted in `README.md`.

_PER_FRAME = {
    # C slot name         -> Python slot name
    "frame_rgb":             "frame_rgb",
    "feat_s1":               "sam2_fpn_2",   # 1x = last FPN level
    "vit_out":               "vit_out_per_frame",  # C has per-frame
    "sam3_feat_s1":          "sam3_feat_s1_per_frame",
    "image_embed":           "image_embed",
    "tgt":                   "memattn_in_tgt",
    "memory":                "memattn_in_memory",
    "memory_image":          "memattn_in_memory_image",
    "memory_image_pos":      "memattn_in_memory_image_pos",
    "memattn_layer0":        "memattn_layer_0_out",
    "memattn_layer1":        "memattn_layer_1_out",
    "memattn_layer2":        "memattn_layer_2_out",
    "memattn_layer3":        "memattn_layer_3_out",
    "memattn_out":           "memattn_encoder_out",
    "memattn_l0_sa_out":     "memattn_l0_sa_out",
    "memattn_l0_ca_q":       "memattn_l0_ca_q",
    "memattn_l0_ca_k":       "memattn_l0_ca_k",
    "memattn_l0_ca_v":       "memattn_l0_ca_v",
    "memattn_l0_ca_attn":    "memattn_l0_ca_attn",
    "mask_dec_masks":        "mdec_out_masks",
    "mask_dec_iou":          "mdec_out_iou",
    "mask_dec_score":        "mdec_out_obj_score",
    "mask_dec_sam":          "mdec_out_sam_tokens",
    "maskmem_out":           "maskmem_out",
}

# Non-frame-indexed C dumps (image pipeline runs per forward_image,
# but with fixed paths that overwrite each frame). We attribute these
# to the last frame (2).
_LAST_FRAME_ONLY = {
    "vit_patch_only":   "vit_patch_embed",
    "vit_patch":        "vit_ln_pre",
    "vit_nhwc":         "vit_ln_post",
    "neck_4x":          "neck_sam3_conv_0",   # 4x = first level
    "neck_2x":          "neck_sam3_conv_1",
    "neck_1x":          "neck_sam3_conv_2",
    "sam2_4x":          "sam2_fpn_0",
    "sam2_2x":          "sam2_fpn_1",
    "sam2_1x":          "sam2_fpn_2_image",
}
# Per-block (32 blocks) — also not frame-indexed.
for _i in range(32):
    _LAST_FRAME_ONLY[f"vit_block{_i:02d}"] = f"vit_block_{_i:02d}"


def _clear_tmp_dumps():
    """Remove stale dbg_*.bin files from /tmp so we only pick up the
    current run's output."""
    for fn in os.listdir("/tmp"):
        if fn.startswith("dbg_") and fn.endswith(".bin"):
            try:
                os.remove(os.path.join("/tmp", fn))
            except OSError:
                pass


def _run_c_tool(binary, model, video, seed_out, frames_dir,
                prop_frames):
    """Invoke the C tool. It writes:
       - seed_out: frame-0 mask PNG
       - frames_dir/frame_NNNN_obj_1.png: propagation PNGs
       - /tmp/dbg_*.bin: per-layer raw binaries (SAM3_DEBUG_DUMP build)
    """
    cmd = [binary,
           "--model", model,
           "--video", video,
           "--point", "0.5,0.5,1",
           "--out", seed_out,
           "--propagate-frames", str(prop_frames),
           "--frames-dir", frames_dir]
    print(f"[run] {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)


def _decode_input_rgbs(video, out_dir, n_frames):
    """Use ffmpeg to extract frames 0..n_frames-1 as PNGs."""
    # ffmpeg frame numbering in -vf select is 0-indexed. Extract
    # separately per frame to keep naming deterministic.
    for i in range(n_frames):
        out = os.path.join(out_dir, f"frame_{i:04d}_input.png")
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-i", video,
               "-vf", f"select=eq(n\\,{i})",
               "-vframes", "1",
               out]
        subprocess.run(cmd, check=True)


def _load_python_shapes():
    """Return the Python-side shapes dict keyed by
    `frame_NNNN/NN_<slot>.bin` so we can derive shape metadata for
    matching C slots."""
    p = os.path.join(_ROOT, "output", "sam3_1_layer_dumps",
                     "shapes.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def _index_python_shape_by_slot(py_shapes):
    """Reindex Python shapes: { "frame_NNNN": {slot_name: shape, ...} }.
    slot_name is stripped of the NN_ prefix so it can be matched to
    the C→Python mapping."""
    out = {}
    for full_key, shape in py_shapes.items():
        frame, fname = full_key.split("/", 1)
        # fname = "NN_<slot>.bin" -> strip NN_ + .bin
        if "_" in fname:
            slot = fname.split("_", 1)[1].rsplit(".bin", 1)[0]
        else:
            slot = fname.rsplit(".bin", 1)[0]
        out.setdefault(frame, {})[slot] = shape
    return out


def _write_ordered_slot_list():
    """Same ordering as the Python script's _ordered_slot_names."""
    names = ["frame_rgb", "vit_patch_embed", "vit_ln_pre"]
    names += [f"vit_block_{i:02d}" for i in range(32)]
    names += ["vit_ln_post"]
    names += [f"neck_sam3_conv_{i}" for i in range(3)]
    names += [f"neck_prop_conv_{i}" for i in range(3)]
    names += [f"sam2_fpn_{i}" for i in range(3)]
    names += ["mdec_conv_s0", "mdec_conv_s1",
              "maskmem_downsampled", "maskmem_pix_proj",
              "maskmem_fuser_out", "maskmem_out",
              "image_embed"]
    names += ["memattn_in_tgt", "memattn_in_image",
              "memattn_in_memory", "memattn_in_memory_image",
              "memattn_in_memory_image_pos", "memattn_in_memory_pos",
              "memattn_in_src_pos"]
    names += [f"memattn_layer_{i}_out" for i in range(4)]
    names += ["memattn_final_norm", "memattn_encoder_out"]
    for i in range(2):
        names += [f"mdec_twt_{i}_queries", f"mdec_twt_{i}_keys"]
    names += ["mdec_twt_final_attn", "mdec_twt_norm_final",
              "mdec_upscale_dc1", "mdec_upscale_ln1",
              "mdec_upscale_act1", "mdec_upscale_dc2",
              "mdec_upscale_act2",
              "mdec_iou_head", "mdec_obj_score_head",
              "mdec_out_masks", "mdec_out_iou",
              "mdec_out_sam_tokens", "mdec_out_obj_score"]
    return names


def _lookup_shape_any_frame(py_name, py_shapes_by_frame,
                            prefer_frame=None):
    """Find the shape for a slot. If `prefer_frame` is given and has
    the slot, return that frame's shape (so growing memory banks
    don't get stuck at the earliest frame's size). Otherwise fall
    back to any frame where the slot fired."""
    if prefer_frame is not None:
        shape = py_shapes_by_frame.get(prefer_frame, {}).get(py_name)
        if shape is not None:
            return shape
    for frame_shapes in py_shapes_by_frame.values():
        shape = frame_shapes.get(py_name)
        if shape is not None:
            return shape
    return None


def _copy_per_frame(src_dir, dst_dir, frame_idx, slot_order,
                    c_to_py, shapes_out, py_shapes_by_frame):
    """Copy /tmp/dbg_trk_<C>_f<frame>.bin into dst_dir under the
    Python naming convention. Returns the list of resolved slot names
    for the manifest."""
    py_frame = f"frame_{frame_idx:04d}"

    resolved = {}
    for c_name, py_name in c_to_py.items():
        src = os.path.join(src_dir,
                           f"dbg_trk_{c_name}_f{frame_idx}.bin")
        if not os.path.exists(src):
            continue
        resolved[py_name] = src

    # Emit in Python's canonical slot order (gives aligned NN_ prefix
    # between C and Python outputs). Only slots that resolve to a file
    # produce output; others are skipped (but still take their index
    # slot — matching Python behaviour where unfired hooks create
    # gaps in numbering).
    for idx, py_name in enumerate(slot_order):
        src = resolved.get(py_name)
        if not src:
            continue
        fname = f"{idx:02d}_{py_name}.bin"
        dst = os.path.join(dst_dir, fname)
        shutil.copyfile(src, dst)

        key = f"{py_frame}/{fname}"
        shape = _lookup_shape_any_frame(py_name, py_shapes_by_frame,
                                        prefer_frame=py_frame)
        if shape is None:
            # Architecture-invariant lookup failed — fall back to
            # a flat [N] shape from file size.
            size = os.path.getsize(dst)
            shape = [size // 4]
        shapes_out[key] = shape


def _copy_last_frame_only(src_dir, dst_dir, frame_idx, slot_order,
                          c_to_py, shapes_out, py_shapes_by_frame,
                          already_written):
    """Copy non-frame-indexed dumps (vit_*, neck_*, sam2_*) into the
    designated last-frame dir."""
    py_frame = f"frame_{frame_idx:04d}"

    for c_name, py_name in c_to_py.items():
        if py_name in already_written:
            continue
        src = os.path.join(src_dir, f"dbg_{c_name}.bin")
        if not os.path.exists(src):
            continue
        try:
            idx = slot_order.index(py_name)
        except ValueError:
            continue
        fname = f"{idx:02d}_{py_name}.bin"
        dst = os.path.join(dst_dir, fname)
        shutil.copyfile(src, dst)
        key = f"{py_frame}/{fname}"
        shape = _lookup_shape_any_frame(py_name, py_shapes_by_frame,
                                        prefer_frame=py_frame)
        if shape is None:
            size = os.path.getsize(dst)
            shape = [size // 4]
        shapes_out[key] = shape
        already_written.add(py_name)


def _copy_mask_pngs(frames_dir, pngs_dir, n_frames, seed_png):
    """Copy the C tool's output PNGs into the structured pngs/ dir."""
    # Frame 0: seed mask
    shutil.copyfile(seed_png,
                    os.path.join(pngs_dir, "frame_0000_mask_obj1.png"))
    for i in range(1, n_frames):
        src = os.path.join(frames_dir, f"frame_{i:04d}_obj_1.png")
        if os.path.exists(src):
            shutil.copyfile(
                src, os.path.join(pngs_dir,
                                  f"frame_{i:04d}_mask_obj1.png"))


def _write_readme(out_dir, n_frames, slot_order, covered_slots):
    missing = [s for s in slot_order if s not in covered_slots]
    path = os.path.join(out_dir, "README.md")
    with open(path, "w") as f:
        f.write("# SAM 3.1 C reference — per-layer dumps\n\n")
        f.write("Generated by `scripts/dump_c_sam3_1_all_layers.py` "
                "by running `build/sam3_1_dump_seed` with "
                f"`--propagate-frames {n_frames - 1}` and a center "
                "point prompt (matches the committed "
                "`tests/fixtures/video_kids/sam3_1/seed_mask.png`).\n\n")
        f.write("## Layout\n\n")
        f.write("Mirrors `output/sam3_1_layer_dumps/` (the Python-side "
                "dump) so C and Python slots line up by "
                "`NN_<slot>.bin` filename for direct diffing.\n\n")
        f.write("- `pngs/frame_{NNNN}_input.png` — input RGB "
                "(ffmpeg-decoded at native video resolution).\n")
        f.write("- `pngs/frame_{NNNN}_mask_obj1.png` — binary "
                "object-1 mask from the C tracker (288x288 or native "
                "decoder resolution).\n")
        f.write("- `bins/frame_{NNNN}/NN_<slot>.bin` — F32 raw "
                "binary for that layer's output. NN prefixes match "
                "`output/sam3_1_layer_dumps/`.\n")
        f.write("- `shapes.json` — `frame_NNNN/NN_<slot>.bin` -> "
                "shape map. Shapes come from the Python side's "
                "shapes.json (same architecture); mismatches indicate "
                "an arch divergence.\n\n")
        f.write("## Known gaps vs Python\n\n")
        f.write("### ViT backbone per-frame\n\n"
                "Image-encoder dumps in `src/model/image_encoder.c` "
                "use fixed paths (no `_f<N>` suffix), so per-block "
                f"ViT tensors only exist for the **last** frame "
                f"(frame {n_frames - 1}). Adding frame-indexed paths "
                "would require plumbing `frame_idx` into the image "
                "encoder's dump macros.\n\n")
        f.write("### Mask-decoder sub-layer hooks not yet in C\n\n"
                "The C tracker dumps decoder OUTPUTS "
                "(`mdec_out_masks/iou/sam_tokens/obj_score`) but not "
                "the two-way transformer internals, upscaling sub-"
                "ops, IoU/obj-score head pre-activation, or "
                "`conv_s0/s1`. Adding them requires new `sam3_dbg_*` "
                "extern slots populated from "
                "`src/model/mask_decoder.c` + "
                "`src/model/tracker_multiplex.c`.\n\n")
        if missing:
            f.write("### Slots missing from this dump\n\n")
            for s in missing:
                f.write(f"- `{s}`\n")
            f.write("\n")
        f.write("## Dump order (per frame)\n\n")
        f.write("| NN | slot | C source | notes |\n"
                "|---|---|---|---|\n")
        # Build reverse lookup for doc purposes.
        py_to_c = {v: k for k, v in _PER_FRAME.items()}
        py_to_c_last = {v: k for k, v in _LAST_FRAME_ONLY.items()}
        for i, s in enumerate(slot_order):
            if s in py_to_c:
                f.write(f"| {i:02d} | `{s}` | "
                        f"`/tmp/dbg_trk_{py_to_c[s]}_f<N>.bin` | "
                        f"per-frame |\n")
            elif s in py_to_c_last:
                f.write(f"| {i:02d} | `{s}` | "
                        f"`/tmp/dbg_{py_to_c_last[s]}.bin` | "
                        f"last-frame only |\n")
            else:
                f.write(f"| {i:02d} | `{s}` | — | **not yet hooked "
                        "in C** |\n")


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary",
                    default=os.path.join(_ROOT, "build",
                                         "sam3_1_dump_seed"))
    ap.add_argument("--model",
                    default=os.path.join(_ROOT, "models",
                                         "sam3.1.sam3"))
    ap.add_argument("--video",
                    default=os.path.join(_ROOT, "assets",
                                         "kids.mp4"))
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--out",
                    default=os.path.join(_ROOT, "output",
                                         "sam3_1_c_layer_dumps"))
    args = ap.parse_args()

    if not os.path.exists(args.binary):
        print(f"ERROR: C binary not found: {args.binary}. "
              "Rebuild with -DSAM3_DEBUG_DUMP=ON.", file=sys.stderr)
        return 2
    if not os.path.exists(args.model):
        print(f"ERROR: model not found: {args.model}",
              file=sys.stderr)
        return 2

    # Sanity-check the binary was built with SAM3_DEBUG_DUMP.
    dbg = subprocess.run(
        ["nm", "-U", args.binary],
        capture_output=True, text=True)
    if "sam3_dbg_trk_image_embed" not in dbg.stdout:
        print("WARN: C binary does not export sam3_dbg_trk_* symbols "
              "— was it built with -DSAM3_DEBUG_DUMP=ON?",
              file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    bins_root = os.path.join(args.out, "bins")
    pngs_dir = os.path.join(args.out, "pngs")
    os.makedirs(bins_root, exist_ok=True)
    os.makedirs(pngs_dir, exist_ok=True)
    for i in range(args.frames):
        os.makedirs(os.path.join(bins_root, f"frame_{i:04d}"),
                    exist_ok=True)

    _clear_tmp_dumps()

    # Staging dir for the C tool's PNG outputs (--frames-dir).
    stage_png = os.path.join(args.out, "_c_stage_pngs")
    os.makedirs(stage_png, exist_ok=True)
    seed_png = os.path.join(stage_png, "frame_0000_seed.png")

    _run_c_tool(args.binary, args.model, args.video,
                seed_png, stage_png,
                prop_frames=args.frames - 1)

    slot_order = _write_ordered_slot_list()
    py_shapes = _load_python_shapes()
    py_by_frame = _index_python_shape_by_slot(py_shapes)

    shapes_out = {}
    covered = set()
    for fi in range(args.frames):
        frame_dir = os.path.join(bins_root, f"frame_{fi:04d}")
        _copy_per_frame("/tmp", frame_dir, fi, slot_order,
                        _PER_FRAME, shapes_out, py_by_frame)
        for c_name, py_name in _PER_FRAME.items():
            if os.path.exists(
                    os.path.join("/tmp",
                                 f"dbg_trk_{c_name}_f{fi}.bin")):
                covered.add(py_name)

    # Non-frame-indexed dumps: attribute to last frame.
    last_frame = args.frames - 1
    last_dir = os.path.join(bins_root, f"frame_{last_frame:04d}")
    already_written = set()
    _copy_last_frame_only("/tmp", last_dir, last_frame, slot_order,
                          _LAST_FRAME_ONLY, shapes_out, py_by_frame,
                          already_written)
    covered.update(already_written)

    # Input RGB PNGs via ffmpeg.
    _decode_input_rgbs(args.video, pngs_dir, args.frames)

    # Mask PNGs from C tool (seed + propagation frames).
    _copy_mask_pngs(stage_png, pngs_dir, args.frames, seed_png)
    shutil.rmtree(stage_png, ignore_errors=True)

    with open(os.path.join(args.out, "shapes.json"), "w") as f:
        json.dump(shapes_out, f, indent=2, sort_keys=True)

    _write_readme(args.out, args.frames, slot_order, covered)
    print(f"[done] C dumps landed in {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
