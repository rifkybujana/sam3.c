#!/usr/bin/env python3
"""
scripts/inspect_mobileclip_pt.py - Dump key/shape/dtype for all tensors in
an EfficientSAM3 .pt checkpoint, focusing on the MobileCLIP text-encoder
sub-tree. Used during Phase 0 of the MobileCLIP plan to confirm exact
key names that the spec left as TBD.

Usage:
    python scripts/inspect_mobileclip_pt.py models/efficient_sam3_image_encoder_mobileclip_s0_ctx16.pt
"""
import argparse
import sys
import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("input", help="Input .pt path")
    ap.add_argument("--prefix", default="",
                    help="Filter to keys starting with this prefix")
    ap.add_argument("--grep", default=None,
                    help="Substring filter on key names")
    args = ap.parse_args()

    print(f"Loading {args.input} ...", file=sys.stderr)
    ckpt = torch.load(args.input, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    if not isinstance(ckpt, dict):
        print(f"error: not a state_dict (got {type(ckpt).__name__})",
              file=sys.stderr)
        return 1

    keys = list(ckpt.keys())
    if args.prefix:
        keys = [k for k in keys if k.startswith(args.prefix)]
    if args.grep:
        keys = [k for k in keys if args.grep in k]

    for k in sorted(keys):
        v = ckpt[k]
        if isinstance(v, torch.Tensor):
            print(f"{k}\t{tuple(v.shape)}\t{v.dtype}")
        else:
            print(f"{k}\t<{type(v).__name__}>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
