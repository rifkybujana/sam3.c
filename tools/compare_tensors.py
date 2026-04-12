#!/usr/bin/env python3
"""
Compare tensor dumps from C and Python SAM3 implementations.

Usage:
    python compare_tensors.py --c-dir <c_dump> --py-dir <py_dump>

Compares matching .bin files and reports MAE, max error, cosine similarity.
"""
import argparse
import os
import struct
import sys
import numpy as np


def load_tensor(path: str) -> np.ndarray:
    """Load tensor from sam3 dump format."""
    with open(path, "rb") as f:
        n_dims = struct.unpack("<i", f.read(4))[0]
        shape = []
        for _ in range(n_dims):
            shape.append(struct.unpack("<i", f.read(4))[0])
        n_elems = 1
        for d in shape:
            n_elems *= d
        data = np.frombuffer(f.read(n_elems * 4), dtype=np.float32)
        return data.reshape(shape)


def compare(name: str, a: np.ndarray, b: np.ndarray) -> dict:
    """Compare two tensors and return metrics."""
    if a.shape != b.shape:
        return {"name": name, "error": f"shape mismatch: {a.shape} vs {b.shape}"}

    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    mae = float(np.mean(diff))
    max_err = float(np.max(diff))

    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    cos_sim = float(dot / (norm_a * norm_b + 1e-12))

    return {
        "name": name,
        "shape": list(a.shape),
        "mae": mae,
        "max_err": max_err,
        "cos_sim": cos_sim,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare SAM3 tensor dumps")
    parser.add_argument("--c-dir", required=True, help="C tensor dump directory")
    parser.add_argument("--py-dir", required=True, help="Python tensor dump directory")
    args = parser.parse_args()

    c_files = {f for f in os.listdir(args.c_dir) if f.endswith(".bin")}
    py_files = {f for f in os.listdir(args.py_dir) if f.endswith(".bin")}
    common = sorted(c_files & py_files)

    if not common:
        print("No matching .bin files found.")
        print(f"  C dir:  {sorted(c_files)}")
        print(f"  Py dir: {sorted(py_files)}")
        sys.exit(1)

    print(f"Comparing {len(common)} tensors:\n")
    print(f"{'Tensor':<30} {'Shape':<25} {'MAE':>12} {'MaxErr':>12} {'CosSim':>10}")
    print("-" * 92)

    any_diverged = False
    for name in common:
        c_tensor = load_tensor(os.path.join(args.c_dir, name))
        py_tensor = load_tensor(os.path.join(args.py_dir, name))
        result = compare(name, c_tensor, py_tensor)

        if "error" in result:
            print(f"{name:<30} {result['error']}")
            any_diverged = True
            continue

        shape_str = str(result["shape"])
        mae = result["mae"]
        max_err = result["max_err"]
        cos_sim = result["cos_sim"]

        flag = ""
        if max_err > 0.1:
            flag = " *** DIVERGED"
            any_diverged = True
        elif max_err > 0.01:
            flag = " * WARNING"

        print(f"{name:<30} {shape_str:<25} {mae:>12.6f} {max_err:>12.6f} {cos_sim:>10.6f}{flag}")

    print()
    if any_diverged:
        print("RESULT: Significant divergence detected. Investigate flagged tensors.")
        sys.exit(1)
    else:
        print("RESULT: All tensors within tolerance. C matches Python reference.")


if __name__ == "__main__":
    main()
