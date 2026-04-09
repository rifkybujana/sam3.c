#!/usr/bin/env python3
"""Compare C text encoder per-block outputs against Python fixtures.

Run: python3 tools/compare_text_encoder.py

Requires the C code to dump per-block outputs (dbg_te_block_XX.bin).
If dumps are absent, just prints the Python fixture stats.
"""

import sys
import os
import struct
import json
import numpy as np

FIXTURE_DIR = "tests/fixtures/04_text_encoder"


def load_safetensors(path, tensor_name=None):
    """Load first tensor (or named tensor) from a safetensors file."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
        data_start = 8 + header_size

        for name, info in header.items():
            if name == "__metadata__":
                continue
            if tensor_name and name != tensor_name:
                continue
            dtype_str = info["dtype"]
            shape = info["shape"]
            offsets = info["data_offsets"]
            f.seek(data_start + offsets[0])
            nbytes = offsets[1] - offsets[0]
            raw = f.read(nbytes)

            if dtype_str == "F32":
                arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            elif dtype_str == "F16":
                arr = np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
            elif dtype_str == "BF16":
                u16 = np.frombuffer(raw, dtype=np.uint16)
                u32 = u16.astype(np.uint32) << 16
                arr = u32.view(np.float32).reshape(shape)
            else:
                continue

            return name, arr, header

    return None, None, None


def load_bin(path):
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.float32)


def compare(label, c_data, py_data, atol=0.01):
    if c_data.size != py_data.size:
        print(f"  {label:40s} SKIP (size: C={c_data.size} vs Py={py_data.size})")
        return -1
    diff = np.abs(c_data.flatten() - py_data.flatten())
    maxd = diff.max()
    meand = diff.mean()
    n_exceed = (diff > atol).sum()
    pct_ok = 100.0 * (1.0 - n_exceed / diff.size)
    print(f"  {label:40s} max={maxd:<12.6f} mean={meand:<12.8f} ok={pct_ok:.1f}%")
    return maxd


def list_tensors(path):
    """List all tensor names and shapes in a safetensors file."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    result = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        result.append((name, info["shape"], info["dtype"]))
    return result


def main():
    print("=== Text Encoder: Per-block fixture analysis ===\n")

    # Token embedding fixture
    print("--- Token Embedding ---")
    try:
        name, arr, _ = load_safetensors(f"{FIXTURE_DIR}/token_embed.safetensors")
        if arr is not None:
            print(f"  {name}: shape={arr.shape} min={arr.min():.4f} max={arr.max():.4f}")
        c = load_bin("/tmp/dbg_te_token_embed.bin")
        compare("token_embed", c, arr.flatten())
    except Exception as e:
        print(f"  SKIP: {e}")

    # Per-block
    print("\n--- Per-Block Outputs ---")
    for bi in range(24):
        try:
            name, py_arr, _ = load_safetensors(
                f"{FIXTURE_DIR}/block_{bi:02d}.safetensors")
            py_flat = py_arr.flatten()
            py_stats = (f"shape={py_arr.shape} "
                       f"min={py_arr.min():.4f} max={py_arr.max():.4f}")

            c_path = f"/tmp/dbg_te_block_{bi:02d}.bin"
            if os.path.exists(c_path):
                c_data = load_bin(c_path)
                maxd = compare(f"block_{bi:02d}", c_data, py_flat)
            else:
                print(f"  block_{bi:02d}: Py {py_stats} (no C dump)")
        except Exception as e:
            print(f"  block_{bi:02d} SKIP: {e}")

    # ln_final
    print("\n--- Final Layer Norm ---")
    try:
        name, py_arr, _ = load_safetensors(f"{FIXTURE_DIR}/ln_final.safetensors")
        print(f"  Py {name}: shape={py_arr.shape} "
              f"min={py_arr.min():.4f} max={py_arr.max():.4f}")
        c_path = "/tmp/dbg_te_ln_final.bin"
        if os.path.exists(c_path):
            c_data = load_bin(c_path)
            compare("ln_final", c_data, py_arr.flatten())
    except Exception as e:
        print(f"  SKIP: {e}")

    # Resizer output
    print("\n--- Resizer Output ---")
    try:
        name, py_arr, _ = load_safetensors(f"{FIXTURE_DIR}/resizer.safetensors")
        print(f"  Py {name}: shape={py_arr.shape} "
              f"min={py_arr.min():.4f} max={py_arr.max():.4f}")
    except Exception as e:
        print(f"  SKIP: {e}")

    # Final output
    print("\n--- Output Tensors ---")
    try:
        tensors = list_tensors(f"{FIXTURE_DIR}/output.safetensors")
        for tname, tshape, tdtype in tensors:
            name, arr, _ = load_safetensors(
                f"{FIXTURE_DIR}/output.safetensors", tensor_name=tname)
            print(f"  {tname}: shape={tshape} dtype={tdtype} "
                  f"min={arr.min():.4f} max={arr.max():.4f}")
    except Exception as e:
        print(f"  SKIP: {e}")

    # C text features
    print("\n--- C Text Features ---")
    try:
        c_text = load_bin("/tmp/dbg_text_features.bin")
        name, py_lf, _ = load_safetensors(
            f"{FIXTURE_DIR}/output.safetensors",
            tensor_name="language_features")
        if py_lf is not None:
            compare("language_features", c_text, py_lf.flatten())
    except Exception as e:
        print(f"  SKIP: {e}")


if __name__ == "__main__":
    main()
