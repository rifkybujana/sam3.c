#!/usr/bin/env python3
"""Compare C-dump vs Python-reference dump for each layer."""
import os
import numpy as np

PAIRS = [
    ("/tmp/py_vit_nhwc.bin", "/tmp/dbg_vit_nhwc.bin", (1, 72, 72, 1024)),
    ("/tmp/py_neck_4x.bin",  "/tmp/dbg_neck_4x.bin",  (1, 288, 288, 256)),
    ("/tmp/py_neck_2x.bin",  "/tmp/dbg_neck_2x.bin",  (1, 144, 144, 256)),
    ("/tmp/py_neck_1x.bin",  "/tmp/dbg_neck_1x.bin",  (1, 72, 72, 256)),
    ("/tmp/py_neck_05x.bin", "/tmp/dbg_neck_05x.bin", (1, 36, 36, 256)),
]

def load(path, shape):
    if not os.path.exists(path):
        return None
    a = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(shape))
    if a.size != expected:
        print(f"  WARN size mismatch: {a.size} vs {expected}")
        return a
    return a.reshape(shape)

def cmp(py, c, name):
    if py is None or c is None:
        print(f"{name:30s} MISSING ({py is not None}, {c is not None})")
        return
    if py.shape != c.shape:
        print(f"{name:30s} shape mismatch py={py.shape} c={c.shape}")
        return
    diff = py - c
    abs_diff = np.abs(diff)
    py_abs = np.abs(py)
    print(f"{name:30s} shape={py.shape}")
    print(f"    py  min={py.min():+.4f} max={py.max():+.4f} "
          f"mean={py.mean():+.4f} std={py.std():.4f}")
    print(f"    c   min={c.min():+.4f}  max={c.max():+.4f}  "
          f"mean={c.mean():+.4f} std={c.std():.4f}")
    print(f"    abs_err max={abs_diff.max():.4f} "
          f"mean={abs_diff.mean():.4f} "
          f"rel={abs_diff.mean()/(py_abs.mean()+1e-6):.3%}")
    cos = (py.flatten() @ c.flatten()) / (
          np.linalg.norm(py.flatten()) * np.linalg.norm(c.flatten()) + 1e-9)
    print(f"    cosine_sim={cos:.5f}")

for py_path, c_path, shape in PAIRS:
    name = os.path.basename(py_path).replace("py_", "").replace(".bin", "")
    py = load(py_path, shape)
    c  = load(c_path, shape)
    cmp(py, c, name)
    print()
