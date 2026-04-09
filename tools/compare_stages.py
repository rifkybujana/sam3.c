#!/usr/bin/env python3
"""
tools/compare_stages.py - Compare C debug dumps against Python fixtures.

Reads the /tmp/dbg_*.bin files (raw F32 arrays from the C engine) and
compares them against the SafeTensors fixture files from gen_fixtures.py.

Usage:
    python3 tools/compare_stages.py
"""

import sys
import struct
import json
import numpy as np
from pathlib import Path

FIXTURE_DIR = Path("tests/fixtures")


def load_safetensors_all(path):
    """Load all tensors from a safetensors file, return dict {name: np.array}."""
    result = {}
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
        data_start = 8 + header_size

        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype_str = info["dtype"]
            shape = info["shape"]
            offsets = info["data_offsets"]
            f.seek(data_start + offsets[0])
            nbytes = offsets[1] - offsets[0]
            raw = f.read(nbytes)

            if dtype_str == "F32":
                arr = np.frombuffer(raw, dtype=np.float32).copy().reshape(shape)
            elif dtype_str == "F16":
                arr = np.frombuffer(raw, dtype=np.float16).copy().reshape(shape).astype(np.float32)
            elif dtype_str == "BF16":
                u16 = np.frombuffer(raw, dtype=np.uint16).copy()
                u32 = u16.astype(np.uint32) << 16
                arr = u32.view(np.float32).reshape(shape)
            else:
                continue
            result[name] = arr
    return result


def load_bin(path):
    """Load a raw float32 binary dump."""
    return np.fromfile(str(path), dtype=np.float32)


def compare(label, c_data, py_data, atol=0.01):
    """Compare two arrays and print statistics. Returns max diff."""
    c = c_data.ravel().astype(np.float64)
    p = py_data.ravel().astype(np.float64)
    if c.size != p.size:
        print(f"  {label:45s} SIZE MISMATCH C={c.size} Py={p.size}")
        return -1.0

    diff = np.abs(c - p)
    maxd = diff.max()
    meand = diff.mean()
    pct_ok = 100.0 * np.mean(diff < atol)

    status = ("EXACT" if maxd < 0.001 else "GOOD" if maxd < 0.1 else
              "OK" if maxd < 1.0 else "FAIR" if maxd < 5.0 else
              "POOR" if maxd < 20.0 else "DIVERGED")

    print(f"  {label:45s} max={maxd:<10.4f} mean={meand:<10.6f} ok={pct_ok:5.1f}%  {status}")
    return maxd


def main():
    print("=" * 90)
    print("Stage-by-stage: C debug dumps vs Python fixtures")
    print("=" * 90)

    # ── 01 ViT ──
    print("\n── 01 ViT ──")
    vit_bin = Path("/tmp/dbg_vit_nchw.bin")
    blk31_st = FIXTURE_DIR / "01_vit" / "block_31.safetensors"
    if vit_bin.exists() and blk31_st.exists():
        c = load_bin(vit_bin)  # [1, 1024, 72, 72] NCHW
        py = load_safetensors_all(blk31_st)
        if "output" in py:
            # Py output is [5184, 1024] (HW, C) — the raw ViT block output
            py_out = py["output"]
            # C is NCHW [1, 1024, 72, 72], convert to [5184, 1024]
            c_hwc = c.reshape(1, 1024, 72, 72).transpose(0, 2, 3, 1).reshape(-1, 1024)
            compare("vit block_31 (final)", c_hwc, py_out)

    # ── 02 Neck ──
    print("\n── 02 Neck ──")
    for scale, shape in [("4x", (1, 256, 288, 288)), ("2x", (1, 256, 144, 144)),
                          ("1x", (1, 256, 72, 72)), ("05x", (1, 256, 36, 36))]:
        bin_p = Path(f"/tmp/dbg_neck_{scale}.bin")
        st_p = FIXTURE_DIR / "02_neck" / f"scale_{scale}.safetensors"
        if bin_p.exists() and st_p.exists():
            c = load_bin(bin_p)
            py = load_safetensors_all(st_p)
            if "features" in py:
                compare(f"neck scale_{scale}", c, py["features"])

    # ── 04 Text Encoder ──
    print("\n── 04 Text Encoder ──")
    txt_bin = Path("/tmp/dbg_text_features.bin")
    txt_st = FIXTURE_DIR / "04_text_encoder" / "output.safetensors"
    if txt_bin.exists() and txt_st.exists():
        c = load_bin(txt_bin)
        py = load_safetensors_all(txt_st)
        if "language_features" in py:
            py_lf = py["language_features"]
            # C text_features may be truncated to real tokens
            # Py shape is [1, seq_len, 256]
            n = min(c.size, py_lf.ravel().size)
            compare("text features (truncated)", c[:n], py_lf.ravel()[:n])
            print(f"    C size={c.size}, Py shape={py_lf.shape}")

    # ── 05 Geometry Encoder ──
    print("\n── 05 Geometry Encoder ──")
    for li in range(3):
        c_bin = Path(f"/tmp/dbg_geom_layer_{li:02d}.bin")
        st_p = FIXTURE_DIR / "05_geometry_encoder" / f"layer_{li:02d}.safetensors"
        if c_bin.exists() and st_p.exists():
            c = load_bin(c_bin)
            py = load_safetensors_all(st_p)
            # Hook captures output (the layer output) and tgt (the input)
            for k, v in py.items():
                n = min(c.size, v.ravel().size)
                if n > 0:
                    compare(f"geom layer {li} {k}", c[:n], v.ravel()[:n])

    geom_out_bin = Path("/tmp/dbg_geom_out.bin")
    if geom_out_bin.exists():
        c = load_bin(geom_out_bin)
        print(f"    geom_out: size={c.size}, range=[{c.min():.4f}, {c.max():.4f}]")

    # ── 06 Encoder Fusion ──
    print("\n── 06 Encoder Fusion ──")
    # Compare encoder input
    enc_in_bin = Path("/tmp/dbg_enc_input.bin")
    l0_st = FIXTURE_DIR / "06_encoder_fusion" / "layer_00.safetensors"
    if enc_in_bin.exists() and l0_st.exists():
        c = load_bin(enc_in_bin)
        py = load_safetensors_all(l0_st)
        if "tgt" in py:
            compare("enc input (= L0 tgt)", c, py["tgt"])

    # Per-layer outputs
    for li in range(6):
        c_bin = Path(f"/tmp/dbg_enc_layer_{li:02d}.bin")
        st_p = FIXTURE_DIR / "06_encoder_fusion" / f"layer_{li:02d}.safetensors"
        if c_bin.exists() and st_p.exists():
            c = load_bin(c_bin)
            py = load_safetensors_all(st_p)
            if "output" in py:
                compare(f"enc layer {li} output", c, py["output"])

    # Final output
    fused_bin = Path("/tmp/dbg_fused.bin")
    fused_st = FIXTURE_DIR / "06_encoder_fusion" / "output.safetensors"
    if fused_bin.exists() and fused_st.exists():
        c = load_bin(fused_bin)
        py = load_safetensors_all(fused_st)
        if "encoder_hidden_states" in py:
            compare("enc fusion FINAL", c, py["encoder_hidden_states"])

    # ── 07 Decoder ──
    print("\n── 07 Decoder ──")
    queries_bin = Path("/tmp/dbg_queries.bin")
    dec_out_st = FIXTURE_DIR / "07_decoder" / "output.safetensors"
    if queries_bin.exists() and dec_out_st.exists():
        c = load_bin(queries_bin)
        py = load_safetensors_all(dec_out_st)
        for k, v in sorted(py.items()):
            n = v.ravel().size
            if n <= c.size:
                compare(f"decoder {k}", c[:n], v)
            else:
                print(f"  decoder {k}: Py={v.shape} (too large for C dump)")

    # ── 09 Segmentation Head ──
    print("\n── 09 Segmentation Head ──")
    for sub, c_path in [("pixel_decoder", "/tmp/dbg_pixel_embed.bin"),
                         ("instance_proj", "/tmp/dbg_inst.bin"),
                         ("mask_predictor", "/tmp/dbg_mask_embed.bin")]:
        c_bin = Path(c_path)
        st_p = FIXTURE_DIR / "09_seg_head" / f"{sub}.safetensors"
        if c_bin.exists() and st_p.exists():
            c = load_bin(c_bin)
            py = load_safetensors_all(st_p)
            for k, v in py.items():
                n = min(c.size, v.ravel().size)
                compare(f"seg {sub}/{k}", c[:n], v.ravel()[:n])

    # ── 10 Final Masks ──
    print("\n── 10 Final ──")
    final_st = FIXTURE_DIR / "10_final" / "tensors.safetensors"
    if final_st.exists():
        py = load_safetensors_all(final_st)
        for k, v in sorted(py.items()):
            print(f"  Py {k}: shape={v.shape} range=[{v.min():.4f}, {v.max():.4f}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
