#!/usr/bin/env python3
"""
Dump intermediate tensors from the SAM3 Python reference for comparison
with the C implementation.

Usage:
    python dump_reference.py --image <path> --model <checkpoint> --out <dir>
                             [--text <prompt>] [--point <x,y,label>]

Requires: pip install git+https://github.com/facebookresearch/sam3.git
"""
import argparse
import os
import struct
import numpy as np
import torch

# NOTE: The SAM3 Python API names used below (build_sam3_hiera_b_plus,
# SAM3ImagePredictor, predictor._features, etc.) are approximate and may
# need adjustment based on the actual upstream facebookresearch/sam3 repo.
# Verify function signatures, class names, and internal attribute names
# against the installed version before running.


def dump_tensor(path: str, t: torch.Tensor) -> None:
    """Write tensor in sam3 dump format: int32 n_dims, int32[] shape, f32 data."""
    t = t.detach().float().cpu().contiguous()
    with open(path, "wb") as f:
        f.write(struct.pack("<i", t.ndim))
        for d in t.shape:
            f.write(struct.pack("<i", d))
        f.write(t.numpy().tobytes())


def main():
    parser = argparse.ArgumentParser(description="Dump SAM3 reference tensors")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--model", required=True,
                        help="SAM3 checkpoint path (e.g. sam3_hiera_base_plus.pt)")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--text", default=None, help="Text prompt")
    parser.add_argument("--point", default=None,
                        help="Point prompt as x,y,label")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Import sam3 — user must have it installed
    from sam3.build_sam import build_sam3_hiera_b_plus
    from sam3.sam3_image_predictor import SAM3ImagePredictor
    from PIL import Image

    # Build model + predictor
    model = build_sam3_hiera_b_plus(checkpoint=args.model)
    model.eval()
    predictor = SAM3ImagePredictor(model)

    # Load and set image
    image = np.array(Image.open(args.image).convert("RGB"))
    predictor.set_image(image)

    # Dump cached backbone features
    features = predictor._features
    for key, val in features.items():
        if isinstance(val, torch.Tensor):
            name = f"backbone_{key}.bin"
            dump_tensor(os.path.join(args.out, name), val)
            print(f"  wrote {name} {list(val.shape)}")

    # Run prediction to get mask logits + scores
    if args.text:
        masks, scores, logits = predictor.predict(
            text_prompt=args.text,
            multimask_output=True,
            return_logits=True,
        )
    elif args.point:
        x, y, label = args.point.split(",")
        point_coords = np.array([[int(x), int(y)]])
        point_labels = np.array([int(label)])
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
            return_logits=True,
        )
    else:
        print("Error: specify --text or --point")
        return

    # Dump mask logits and scores
    dump_tensor(os.path.join(args.out, "mask_logits.bin"),
                torch.from_numpy(logits) if isinstance(logits, np.ndarray)
                else logits)
    print(f"  wrote mask_logits.bin {list(logits.shape)}")

    if scores is not None:
        dump_tensor(os.path.join(args.out, "scores.bin"),
                    torch.from_numpy(scores) if isinstance(scores, np.ndarray)
                    else scores)
        print(f"  wrote scores.bin {list(scores.shape)}")

    print(f"\nAll tensors written to {args.out}")


if __name__ == "__main__":
    main()
