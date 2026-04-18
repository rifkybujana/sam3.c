#!/usr/bin/env python3
"""
tools/pt_to_safetensors.py - Convert a PyTorch .pt checkpoint to .safetensors.

Used as the first step in the SAM 3.1 conversion pipeline because
sam3_convert only reads .safetensors. Also normalizes the checkpoint:
unwraps the `{"model": ...}` outer dict and remaps Facebook's internal
prefixes `sam3_model.*` / `sam2_predictor.*` to the OSS layout
`detector.*` / `tracker.*` (matches the Python reference in
reference/sam3/sam3/model_builder.py needs_remap logic).

Usage:
    python tools/pt_to_safetensors.py input.pt output.safetensors
"""
import argparse
import sys

import torch
from safetensors.torch import save_file

# Mirrors reference/sam3/sam3/model_builder.py:1209-1221
FB_TO_OSS_PREFIXES = [
    ("sam3_model.", "detector."),
    ("sam2_predictor.", "tracker."),
]


def remap(key: str) -> str:
    for src, dst in FB_TO_OSS_PREFIXES:
        if key.startswith(src):
            return dst + key[len(src):]
    return key


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("input", help="Input .pt path")
    ap.add_argument("output", help="Output .safetensors path")
    args = ap.parse_args()

    print(f"Loading {args.input} ...", file=sys.stderr)
    ckpt = torch.load(args.input, map_location="cpu", weights_only=True)

    # Unwrap {"model": ...}
    if isinstance(ckpt, dict) and "model" in ckpt \
            and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]

    if not isinstance(ckpt, dict):
        print(f"error: checkpoint is not a dict (got {type(ckpt).__name__})",
              file=sys.stderr)
        return 1

    # Remap if any FB-internal prefixes are present
    needs_remap = any(
        k.startswith(src) for src, _ in FB_TO_OSS_PREFIXES for k in ckpt
    )
    if needs_remap:
        print("Remapping sam3_model.* -> detector.*, "
              "sam2_predictor.* -> tracker.*", file=sys.stderr)
        ckpt = {remap(k): v for k, v in ckpt.items()}
    else:
        print("No FB-internal prefixes detected; writing as-is.",
              file=sys.stderr)

    # Force contiguous tensors (safetensors requires it)
    ckpt = {k: v.contiguous() for k, v in ckpt.items()}

    print(f"Writing {len(ckpt)} tensors to {args.output} ...", file=sys.stderr)
    save_file(ckpt, args.output)
    print("Done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
