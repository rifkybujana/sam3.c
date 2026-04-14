#!/usr/bin/env python3
"""
tools/export_efficientsam3.py - Export EfficientSAM3 .pt checkpoint to SafeTensors

Loads a pre-merged EfficientSAM3 checkpoint (EfficientViT-B2 vision backbone +
SAM3 text encoder) and writes all tensors to SafeTensors format. The C rename
reader handles all name mapping; this script just does format conversion.

Usage:
    python tools/export_efficientsam3.py <input.pt> <output.safetensors>

Copyright (c) 2026 Rifky Bujana Bisri
SPDX-License-Identifier: MIT
"""

import sys
import argparse

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(
        description="Export EfficientSAM3 .pt checkpoint to SafeTensors"
    )
    parser.add_argument("input", help="Input .pt checkpoint path")
    parser.add_argument("output", help="Output .safetensors path")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print tensor names and shapes"
    )
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)

    # Extract model state dict (may be nested under 'model' key)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Filter to only torch.Tensor entries (skip optimizer state, etc.)
    tensors = {}
    skipped = 0
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            skipped += 1
            continue
        # Contiguous copy for SafeTensors compatibility
        tensors[key] = value.contiguous()

    print(f"  tensors: {len(tensors)}")
    if skipped:
        print(f"  skipped: {skipped} (non-tensor entries)")

    if args.verbose:
        for name, t in sorted(tensors.items()):
            print(f"  {name}: {list(t.shape)} {t.dtype}")

    print(f"Writing {args.output}...")
    save_file(tensors, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
