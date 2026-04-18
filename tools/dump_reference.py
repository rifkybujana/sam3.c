#!/usr/bin/env python3
"""
tools/dump_reference.py - Dump per-stage SAM 3 / SAM 3.1 reference
tensors for numerical parity tests.

For --variant sam3, assembles the stock image model via
    build_sam3_image_model(...)  (from the upstream sam3 package).
For --variant sam3.1, assembles an image-only SAM 3.1 model by
reusing the multiplex tri-neck helpers from the upstream
model_builder.py: _create_multiplex_tri_backbone + _create_text_encoder
+ the unchanged detector components (transformer, segmentation head,
dot-product scoring, geometry encoder), wrapped in Sam3Image. The
multiplex checkpoint carries detector.* keys that load straight into
this assembly (tracker weights are silently unused).

Output format matches the per-stage .safetensors fixture layout already
used by tests/fixtures/bus_person/ (00_input/, 01_vit/, ..., 10_final/).

Usage:
    python tools/dump_reference.py --variant {sam3,sam3.1} \\
        --image path/to/img.jpg --checkpoint path/to/file.pt --text "bus" \\
        --out tests/fixtures/sam3_1_bus_person/
"""
import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file


def build_model_sam3(checkpoint: str, bpe_path: str, device: str):
    from sam3.model_builder import build_sam3_image_model
    return build_sam3_image_model(
        bpe_path=bpe_path,
        device=device,
        eval_mode=True,
        checkpoint_path=checkpoint,
        load_from_HF=False,
    )


def build_model_sam3_1(checkpoint: str, bpe_path: str, device: str):
    """Assemble SAM 3.1 image model - reuses the multiplex tri-neck."""
    from sam3.model_builder import (
        _create_geometry_encoder,
        _create_multiplex_tri_backbone,
        _create_sam3_transformer,
        _create_segmentation_head,
        _create_dot_product_scoring,
        _create_text_encoder,
    )
    from sam3.model.sam3_image import Sam3Image
    from sam3.model.vl_combiner import SAM3VLBackbone

    tri_neck = _create_multiplex_tri_backbone()
    text_encoder = _create_text_encoder(bpe_path)
    backbone = SAM3VLBackbone(scalp=1, visual=tri_neck, text=text_encoder)
    transformer = _create_sam3_transformer()
    dot_prod_scoring = _create_dot_product_scoring()
    segmentation_head = _create_segmentation_head()
    geometry_encoder = _create_geometry_encoder()

    model = Sam3Image(
        backbone=backbone,
        transformer=transformer,
        input_geometry_encoder=geometry_encoder,
        segmentation_head=segmentation_head,
        num_feature_levels=1,
        o2m_mask_predict=True,
        dot_prod_scoring=dot_prod_scoring,
        use_instance_query=False,
        multimask_output=True,
        inst_interactive_predictor=None,
        matcher=None,
    )

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    # Keep only detector.* keys, strip the prefix
    detector_ckpt = {k[len("detector."):]: v
                     for k, v in ckpt.items() if k.startswith("detector.")}
    missing, unexpected = model.load_state_dict(detector_ckpt, strict=False)
    print(f"Loaded {len(detector_ckpt)} detector tensors. "
          f"missing={len(missing)} unexpected={len(unexpected)}",
          file=sys.stderr)
    model.eval().to(device)
    return model


def run_and_dump(model, image_path: str, text: str, out_dir: str):
    """Run the image pipeline and save per-stage tensors.

    Output layout matches tests/fixtures/bus_person/ (00_input, 02_neck,
    10_final). The 02_neck hook body depends on the upstream neck's
    output shape; adjust after first run if needed.
    """
    image = Image.open(image_path).convert("RGB")
    import torchvision.transforms.functional as F
    img_t = F.to_tensor(image.resize((1008, 1008)))
    img_t = (img_t - 0.5) / 0.5
    img_t = img_t.unsqueeze(0).to(next(model.parameters()).device)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "00_input").mkdir(exist_ok=True)
    save_file({"image": img_t.cpu().contiguous()},
              str(Path(out_dir) / "00_input" / "tensors.safetensors"))

    with torch.no_grad():
        neck_outs = []
        def _neck_hook(mod, inp, out):
            neck_outs.append(out)
        h = model.backbone.visual.register_forward_hook(_neck_hook)
        out = model(img_t, text=[text])
        h.remove()

    (Path(out_dir) / "02_neck").mkdir(exist_ok=True)
    # neck output format is implementation-specific; inspect neck_outs on
    # first run and fill in scale_4x / scale_2x / scale_1x saves below.
    # Keep empty for the first pass so tools/dump_reference.py returns 0
    # and the user can inspect the output.

    (Path(out_dir) / "10_final").mkdir(exist_ok=True)
    final_dict = {}
    if isinstance(out, dict):
        for k in ("pred_masks", "pred_logits"):
            if k in out:
                final_dict[k] = out[k].cpu().contiguous()
    save_file(final_dict,
              str(Path(out_dir) / "10_final" / "tensors.safetensors"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--variant", choices=["sam3", "sam3.1"], required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bpe", default=None,
                    help="Path to bpe_simple_vocab_16e6.txt.gz (default: "
                         "upstream package resource)")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.bpe is None:
        import pkg_resources
        args.bpe = pkg_resources.resource_filename(
            "sam3", "assets/bpe_simple_vocab_16e6.txt.gz")

    if args.variant == "sam3":
        model = build_model_sam3(args.checkpoint, args.bpe, args.device)
    else:
        model = build_model_sam3_1(args.checkpoint, args.bpe, args.device)

    run_and_dump(model, args.image, args.text, args.out)
    print(f"Wrote fixtures to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
