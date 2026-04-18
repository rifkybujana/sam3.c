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
import types
from pathlib import Path

# sam3's tracker utils import triton (CUDA-only). The SAM 3.1 image path
# we assemble below doesn't actually execute any triton kernels.
#
# Two things must happen before sam3 is imported:
# 1. Force torch.utils._triton.has_triton_package() to return False, so
#    torch._inductor (loaded transitively via torchvision) doesn't try
#    to import triton.backends.compiler.
# 2. Stub the `triton` module so sam3/model/edt.py's bare `import triton`
#    succeeds without the kernels ever being called.
import torch  # isort:skip
from torch.utils import _triton as _torch_triton

_torch_triton.has_triton_package = lambda: False
_torch_triton.has_triton = lambda: False

# The upstream sam3 package hardcodes device="cuda" in several tensor
# creation calls (PositionEmbeddingSine, TransformerDecoder._get_coords,
# etc.). Redirect `device="cuda"` (or torch.device("cuda")) to CPU for
# every tensor factory when CUDA isn't available.
if not torch.cuda.is_available():
    def _cpu_redirect(kwargs):
        dev = kwargs.get("device")
        if dev is None:
            return
        if isinstance(dev, torch.device) and dev.type == "cuda":
            kwargs["device"] = torch.device("cpu")
        elif isinstance(dev, str) and dev.startswith("cuda"):
            kwargs["device"] = "cpu"

    for _fn in ("zeros", "ones", "empty", "arange", "randn", "rand",
                "full", "linspace", "eye", "tensor", "as_tensor"):
        _orig = getattr(torch, _fn)

        def _wrap(orig):
            def _wrapped(*a, **kw):
                _cpu_redirect(kw)
                return orig(*a, **kw)
            return _wrapped

        setattr(torch, _fn, _wrap(_orig))

    # .cuda() no-ops to self (CPU tensor stays on CPU).
    torch.Tensor.cuda = lambda self, *a, **kw: self
    # Module.cuda() likewise returns self without moving.
    torch.nn.Module.cuda = lambda self, *a, **kw: self


if "triton" not in sys.modules:
    class _TritonStubType:
        pass

    triton_stub = types.ModuleType("triton")
    triton_stub.jit = lambda f=None, **kw: (
        f if callable(f) else (lambda g: g)
    )
    triton_stub.heuristics = lambda *a, **kw: (lambda f: f)
    triton_stub.autotune = lambda *a, **kw: (lambda f: f)

    triton_lang_stub = types.ModuleType("triton.language")
    for _name in ("dtype", "tensor", "pointer_type", "void", "int1",
                  "int8", "int16", "int32", "int64",
                  "uint8", "uint16", "uint32", "uint64",
                  "float16", "float32", "float64",
                  "bfloat16", "block_type", "constexpr"):
        setattr(triton_lang_stub, _name, _TritonStubType)

    triton_stub.language = triton_lang_stub
    sys.modules["triton"] = triton_stub
    sys.modules["triton.language"] = triton_lang_stub

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


def _install_addmm_act_fp32_patch():
    """Replace sam3.perflib.fused.addmm_act with an fp32-preserving
    version that works on CPU. Must be called AFTER importing sam3 (so
    sam3.perflib.fused exists) but BEFORE vitdet captures it. Since
    vitdet does `from sam3.perflib.fused import addmm_act` at import
    time, we also patch vitdet's module-level reference.
    """
    import sam3.perflib.fused as _fused

    def _addmm_act_fp32(activation, linear, mat1):
        if torch.is_grad_enabled():
            raise ValueError("Expected grad to be disabled.")
        w = linear.weight.detach()
        b = linear.bias.detach()
        y = torch.nn.functional.linear(mat1, w, b)
        if activation in (torch.nn.functional.relu, torch.nn.ReLU):
            return torch.nn.functional.relu(y)
        if activation in (torch.nn.functional.gelu, torch.nn.GELU):
            return torch.nn.functional.gelu(y)
        raise ValueError(f"Unexpected activation {activation}")

    _fused.addmm_act = _addmm_act_fp32
    # vitdet imports the symbol directly; patch its copy too.
    import sam3.model.vitdet as _vitdet
    _vitdet.addmm_act = _addmm_act_fp32


def build_model_sam3_1(checkpoint: str, bpe_path: str, device: str):
    """Assemble SAM 3.1 image model - reuses the multiplex tri-neck."""
    if not torch.cuda.is_available():
        _install_addmm_act_fp32_patch()
    from sam3.model_builder import (
        _create_geometry_encoder,
        _create_multiplex_tri_backbone,
        _create_sam3_transformer,
        _create_segmentation_head,
        _create_dot_product_scoring,
        _create_text_encoder,
    )
    from sam3.model.sam3_image import Sam3Image
    from sam3.model.vl_combiner import SAM3VLBackboneTri

    tri_neck = _create_multiplex_tri_backbone()
    text_encoder = _create_text_encoder(bpe_path)
    backbone = SAM3VLBackboneTri(scalp=0, visual=tri_neck, text=text_encoder)
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
    # CPU + bfloat16 is flaky (some torch ops on CPU silently upcast to
    # float32, mismatching weights). Cast the entire model to float32
    # when running on CPU.
    if device == "cpu":
        model = model.float()
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
    param = next(model.parameters())
    img_t = img_t.unsqueeze(0).to(device=param.device, dtype=param.dtype)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "00_input").mkdir(exist_ok=True)
    save_file({"image": img_t.cpu().contiguous()},
              str(Path(out_dir) / "00_input" / "tensors.safetensors"))

    with torch.no_grad():
        # Direct vision-only path: SAM3VLBackbone.forward_image produces
        # the FPN feature dict used by the detector. No text, no
        # BatchedDatapoint wrapping needed.
        try:
            vision_out = model.backbone.forward_image(img_t)
        except Exception as exc:
            print(f"forward_image failed ({exc}); wrote input only",
                  file=sys.stderr)
            vision_out = None
        out = None

    (Path(out_dir) / "02_neck").mkdir(exist_ok=True)
    if isinstance(vision_out, dict):
        # SAM3VLBackboneTri.forward_image returns:
        #   vision_features   - tensor, last FPN scale [1, 256, 72, 72]
        #   vision_pos_enc    - list of N pos-enc tensors
        #   backbone_fpn      - list of N NestedTensor per-scale features
        # Plus 'interactive' and 'sam2_backbone_out' sub-dicts with the
        # same shape (tracker-specific; skip them for the image parity
        # fixture — the C side only consumes the detector FPN).
        neck_dict = {}
        scale_names = ["scale_4x", "scale_2x", "scale_1x"]
        fpn = vision_out.get("backbone_fpn")
        if isinstance(fpn, (list, tuple)):
            for i, nt in enumerate(fpn):
                t = getattr(nt, "tensors", nt)
                if isinstance(t, torch.Tensor) and i < len(scale_names):
                    neck_dict[scale_names[i]] = t.cpu().contiguous().float()
        pos_enc = vision_out.get("vision_pos_enc")
        if isinstance(pos_enc, (list, tuple)):
            for i, t in enumerate(pos_enc):
                if isinstance(t, torch.Tensor) and i < len(scale_names):
                    neck_dict[f"pos_{scale_names[i]}"] = (
                        t.cpu().contiguous().float())
        if neck_dict:
            save_file(neck_dict,
                      str(Path(out_dir) / "02_neck" / "tensors.safetensors"))
            print(f"Wrote {len(neck_dict)} neck tensors: "
                  f"{sorted(neck_dict)}", file=sys.stderr)

    (Path(out_dir) / "10_final").mkdir(exist_ok=True)
    final_dict = {}
    if isinstance(out, dict):
        for k in ("pred_masks", "pred_logits"):
            if k in out:
                final_dict[k] = out[k].cpu().contiguous()
    if not final_dict:
        # Running the full Sam3Image.forward requires a BatchedDatapoint
        # wrapper around image + prompts that's upstream-internal; for
        # now we dump only the neck output and let the integration test
        # treat the final stage as optional.
        final_dict["placeholder"] = torch.zeros(1)
    save_file(final_dict,
              str(Path(out_dir) / "10_final" / "tensors.safetensors"))

    # Marker file the C integration test uses to gate on fixture
    # availability.
    import json
    meta = {
        "variant": "sam3.1",
        "image": Path(image_path).name,
        "text": text,
        "image_size": 1008,
        "n_fpn_scales": 3,
        "has_final_masks": "pred_masks" in final_dict,
    }
    (Path(out_dir) / "metadata.json").write_text(json.dumps(meta, indent=2))


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
