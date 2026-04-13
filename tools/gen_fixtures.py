#!/usr/bin/env python3
"""
tools/gen_fixtures.py - Generate per-layer test fixtures from reference SAM3 model.

Loads the real SAM3 checkpoint, runs inference on an image with a text prompt
and optional point prompt, and saves every intermediate tensor (per-layer)
as SafeTensors files.

Usage:
    cd /path/to/sam3

    # Point + text prompt:
    python tools/gen_fixtures.py \
        --checkpoint models/sam3.pt \
        --bpe models/bpe_simple_vocab_16e6.txt.gz \
        --image assets/cat.jpeg \
        --output tests/fixtures \
        --text "cat" \
        --point 500,400,1

    # Text-only prompt:
    python tools/gen_fixtures.py \
        --image assets/bus.jpg \
        --text "person" \
        --output tests/fixtures/bus_person

Requires: torch, safetensors, Pillow, torchvision
          Plus the reference sam3 package (add reference/ to PYTHONPATH)
"""

import argparse
import json
import os
import sys
import types
from collections import OrderedDict
from pathlib import Path

import PIL.Image
import torch
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# Stub out triton (not available on macOS, only needed for EDT kernel in
# tracker code which we don't use for image-only inference)
# ---------------------------------------------------------------------------
try:
    import triton  # noqa: F401
except (ImportError, ModuleNotFoundError):
    # Comprehensive triton stub for macOS where triton isn't available.
    # torch._dynamo and torch._inductor probe triton deeply, so we need
    # a stub that auto-creates arbitrary sub-modules on access.
    class _TritonStubFinder:
        """sys.meta_path finder that intercepts any 'triton.*' import."""
        def find_module(self, fullname, path=None):
            if fullname == "triton" or fullname.startswith("triton."):
                return self
            return None

        def load_module(self, fullname):
            if fullname in sys.modules:
                return sys.modules[fullname]
            mod = types.ModuleType(fullname)
            mod.__loader__ = self
            mod.__path__ = []
            mod.__package__ = fullname
            sys.modules[fullname] = mod
            return mod

    sys.meta_path.insert(0, _TritonStubFinder())
    # Create the root triton module and key attributes
    import triton  # noqa: F811
    triton.jit = lambda *a, **kw: (lambda f: f)
    triton.autotune = lambda *a, **kw: (lambda f: f)
    triton.Config = lambda *a, **kw: None

    import triton.language as _tl  # noqa: F811
    _tl.constexpr = int
    class _FakeDtype:
        pass
    _tl.dtype = _FakeDtype
    for _name in ["program_id", "arange", "zeros", "load", "store",
                   "where", "minimum", "maximum", "sqrt", "cdiv",
                   "float32", "int32"]:
        setattr(_tl, _name, lambda *a, **kw: None)

# Stub iopath if not installed
if "iopath" not in sys.modules:
    _iopath = types.ModuleType("iopath")
    _iopath_common = types.ModuleType("iopath.common")
    _iopath_file_io = types.ModuleType("iopath.common.file_io")

    class _FakePathManager:
        def open(self, path, mode="r"):
            return open(path, mode)
    _iopath_file_io.g_pathmgr = _FakePathManager()
    _iopath.common = _iopath_common
    _iopath_common.file_io = _iopath_file_io
    sys.modules["iopath"] = _iopath
    sys.modules["iopath.common"] = _iopath_common
    sys.modules["iopath.common.file_io"] = _iopath_file_io

# ---------------------------------------------------------------------------
# Add reference/ to path so we can import the upstream sam3 package
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "reference"))

from sam3.model_builder import build_sam3_image_model  # noqa: E402
from sam3.model.data_misc import FindStage              # noqa: E402
from sam3.model.geometry_encoders import Prompt          # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import contextlib


@contextlib.contextmanager
def _force_cpu_device():
    """Redirect any CUDA device references to CPU during model construction.

    The reference SAM3 code hardcodes device='cuda' in several __init__ methods
    (PositionEmbeddingSine, TransformerDecoder). This context manager patches
    torch tensor-creation functions to silently replace 'cuda' with 'cpu'.
    """
    _orig_zeros = torch.zeros
    _orig_ones = torch.ones
    _orig_arange = torch.arange
    _orig_empty = torch.empty
    _orig_full = torch.full
    _orig_tensor = torch.tensor

    def _cpu_device(kwargs):
        d = kwargs.get("device", None)
        if d is not None and str(d).startswith("cuda"):
            kwargs["device"] = "cpu"
        return kwargs

    def _wrap(fn):
        def wrapper(*args, **kwargs):
            return fn(*args, **_cpu_device(kwargs))
        return wrapper

    torch.zeros = _wrap(_orig_zeros)
    torch.ones = _wrap(_orig_ones)
    torch.arange = _wrap(_orig_arange)
    torch.empty = _wrap(_orig_empty)
    torch.full = _wrap(_orig_full)
    torch.tensor = _wrap(_orig_tensor)
    try:
        yield
    finally:
        torch.zeros = _orig_zeros
        torch.ones = _orig_ones
        torch.arange = _orig_arange
        torch.empty = _orig_empty
        torch.full = _orig_full
        torch.tensor = _orig_tensor


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _to_f32_dict(tensors: dict) -> OrderedDict:
    """Convert all tensors to contiguous float32 for SafeTensors."""
    out = OrderedDict()
    for k, v in tensors.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().float().contiguous()
    return out


def _save(path, tensors):
    """Save a dict of tensors as a .safetensors file."""
    d = _to_f32_dict(tensors)
    if not d:
        return
    _ensure_dir(os.path.dirname(path))
    save_file(d, path)
    nbytes = sum(t.nbytes for t in d.values())
    print(f"  {path}  ({len(d)} tensors, {nbytes / 1024:.1f} KiB)")


# ---------------------------------------------------------------------------
# Hook-based tensor capture
# ---------------------------------------------------------------------------

class TensorCapture:
    """Register forward hooks to capture intermediate tensors."""

    def __init__(self):
        self.captures = {}   # name -> dict of tensors
        self._hooks = []

    def hook(self, name, capture_input=False, capture_output=True,
             input_names=None, output_names=None):
        """Return a hook function that captures inputs/outputs under `name`."""
        def _hook(module, inp, out):
            d = {}
            if capture_input:
                if isinstance(inp, tuple):
                    names = input_names or [f"input_{i}" for i in range(len(inp))]
                    for i, t in enumerate(inp):
                        if isinstance(t, torch.Tensor) and i < len(names):
                            d[names[i]] = t
                elif isinstance(inp, torch.Tensor):
                    d["input"] = inp
            if capture_output:
                if isinstance(out, torch.Tensor):
                    onames = output_names or ["output"]
                    d[onames[0]] = out
                elif isinstance(out, tuple):
                    onames = output_names or [f"output_{i}" for i in range(len(out))]
                    for i, t in enumerate(out):
                        if isinstance(t, torch.Tensor) and i < len(onames):
                            d[onames[i]] = t
                elif isinstance(out, dict):
                    for k, v in out.items():
                        if isinstance(v, torch.Tensor):
                            d[k] = v
                        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                            for j, t in enumerate(v):
                                d[f"{k}_{j}"] = t
            self.captures[name] = d
        return _hook

    def register(self, module, name, **kwargs):
        h = module.register_forward_hook(self.hook(name, **kwargs))
        self._hooks.append(h)

    def remove_all(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Image preprocessing (mirrors Sam3Processor.set_image)
# ---------------------------------------------------------------------------

def preprocess_image(image_path, resolution=1008):
    """Load and normalize image exactly as Sam3Processor does."""
    from torchvision.transforms import v2
    img = PIL.Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    transform = v2.Compose([
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(resolution, resolution)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    tensor = v2.functional.to_image(img)   # uint8 [C, H, W]
    tensor = transform(tensor).unsqueeze(0)  # [1, 3, 1008, 1008] float in [-1,1]
    return tensor, orig_w, orig_h


# ---------------------------------------------------------------------------
# Register hooks on every module we care about
# ---------------------------------------------------------------------------

def register_all_hooks(model, cap):
    """Register forward hooks for per-layer capture on the full SAM3 model.

    Module paths (verified via model.named_modules()):
      backbone.vision_backbone.trunk          -> ViT
      backbone.vision_backbone.trunk.blocks   -> ViT blocks
      backbone.vision_backbone.convs          -> FPN neck convolutions
      backbone.language_backbone.encoder      -> TextTransformer
      backbone.language_backbone.resizer      -> Linear(1024, 256)
      geometry_encoder.encode                 -> 3 layers
      transformer.encoder.layers              -> 6 encoder layers
      transformer.decoder.layers              -> 6 decoder layers
      segmentation_head.pixel_decoder         -> PixelDecoder
      segmentation_head.instance_seg_head     -> Conv2d
      segmentation_head.mask_predictor        -> MaskPredictor
      dot_prod_scoring                        -> DotProductScoring
    """

    # -- ViT backbone --
    vit = model.backbone.vision_backbone.trunk
    cap.register(vit.patch_embed, "vit/patch_embed",
                 capture_input=True, capture_output=True,
                 input_names=["pixels"], output_names=["patches"])

    for i, blk in enumerate(vit.blocks):
        cap.register(blk, f"vit/block_{i:02d}",
                     capture_output=True, output_names=["output"])

    # -- Neck (FPN) --
    neck = model.backbone.vision_backbone
    for i, conv in enumerate(neck.convs):
        scale_name = {0: "4x", 1: "2x", 2: "1x", 3: "05x"}.get(i, f"s{i}")
        cap.register(conv, f"neck/scale_{scale_name}",
                     capture_output=True, output_names=["features"])

    # -- Text encoder --
    text_enc = model.backbone.language_backbone.encoder
    cap.register(text_enc.token_embedding, "text/token_embed",
                 capture_output=True, output_names=["embeddings"])
    for i, blk in enumerate(text_enc.transformer.resblocks):
        cap.register(blk, f"text/block_{i:02d}",
                     capture_output=True, output_names=["output"])
    cap.register(text_enc.ln_final, "text/ln_final",
                 capture_output=True, output_names=["output"])
    # text resizer (projection to d_model)
    cap.register(model.backbone.language_backbone.resizer, "text/resizer",
                 capture_output=True, output_names=["output"])

    # -- Geometry encoder --
    geo_enc = model.geometry_encoder
    if hasattr(geo_enc, 'encode') and geo_enc.encode is not None:
        for i, lay in enumerate(geo_enc.encode):
            cap.register(lay, f"geometry/layer_{i:02d}",
                         capture_input=True, capture_output=True,
                         input_names=["tgt"], output_names=["output"])

    # -- Encoder fusion --
    encoder = model.transformer.encoder
    for i, lay in enumerate(encoder.layers):
        cap.register(lay, f"encoder/layer_{i:02d}",
                     capture_input=True, capture_output=True,
                     input_names=["tgt"], output_names=["output"])

    # -- Decoder --
    decoder = model.transformer.decoder
    for i, lay in enumerate(decoder.layers):
        cap.register(lay, f"decoder/layer_{i:02d}",
                     capture_input=True, capture_output=True,
                     input_names=["tgt"], output_names=["output"])
    cap.register(decoder.norm, "decoder/final_norm",
                 capture_output=True, output_names=["output"])

    # -- Dot product scoring --
    cap.register(model.dot_prod_scoring, "dot_scorer",
                 capture_output=True, output_names=["scores"])

    # -- Segmentation head --
    seg = model.segmentation_head
    if seg is not None:
        cap.register(seg.pixel_decoder, "seg/pixel_decoder",
                     capture_output=True, output_names=["pixel_features"])
        if hasattr(seg, 'instance_seg_head') and seg.instance_seg_head is not None:
            cap.register(seg.instance_seg_head, "seg/instance_proj",
                         capture_output=True, output_names=["instance_features"])
        if hasattr(seg, 'mask_predictor'):
            cap.register(seg.mask_predictor, "seg/mask_predictor",
                         capture_output=True, output_names=["mask_logits"])


# ---------------------------------------------------------------------------
# Run inference and save fixtures
# ---------------------------------------------------------------------------

def run_and_save(model, image_tensor, text_prompt, point, output_dir, device):
    """Run full inference, capture all intermediates, save as SafeTensors."""

    cap = TensorCapture()
    register_all_hooks(model, cap)

    # --- Build inputs matching Sam3Processor ---
    image_tensor = image_tensor.to(device)

    find_stage = FindStage(
        img_ids=torch.tensor([0], device=device, dtype=torch.long),
        text_ids=torch.tensor([0], device=device, dtype=torch.long),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )

    # Phase 1: backbone forward (image)
    print("\n[1/4] Running vision backbone...")
    backbone_out = model.backbone.forward_image(image_tensor)

    # Phase 2: text forward
    print("[2/4] Running text encoder...")
    text_outputs = model.backbone.forward_text([text_prompt], device=device)
    backbone_out.update(text_outputs)

    # Phase 3: build geometric prompt (point or dummy for text-only)
    if point is not None:
        print("[3/4] Building geometric prompt (point)...")
        px, py, plabel = point
        # Points are expected as normalized [0,1] coordinates, seq-first: [N_pts, B, 2]
        point_xy = torch.tensor([[[px / 1008.0, py / 1008.0]]], device=device,
                                dtype=torch.float32).permute(1, 0, 2)  # [1, 1, 2]
        point_labels = torch.tensor([[plabel]], device=device, dtype=torch.long).permute(1, 0)  # [1, 1]
        point_mask = torch.zeros(1, 1, device=device, dtype=torch.bool)  # [B=1, N_pts=1] all valid

        geometric_prompt = Prompt(
            point_embeddings=point_xy,
            point_mask=point_mask,
            point_labels=point_labels,
            box_embeddings=torch.zeros(0, 1, 4, device=device),
            box_mask=torch.zeros(1, 0, device=device, dtype=torch.bool),
        )
    else:
        print("[3/4] Building dummy prompt (text-only)...")
        geometric_prompt = model._get_dummy_prompt()

    # Phase 4: forward_grounding (encoder + decoder + seg heads)
    print("[4/4] Running forward_grounding (encoder, decoder, seg head)...")
    outputs = model.forward_grounding(
        backbone_out=backbone_out,
        find_input=find_stage,
        geometric_prompt=geometric_prompt,
        find_target=None,
    )

    cap.remove_all()

    # --- Save everything ---
    print(f"\nSaving fixtures to {output_dir}/")

    # 00_input: the normalized image
    _save(f"{output_dir}/00_input/tensors.safetensors", {
        "image": image_tensor,
    })

    # 01_vit: patch embed + each block
    if "vit/patch_embed" in cap.captures:
        _save(f"{output_dir}/01_vit/patch_embed.safetensors",
              cap.captures["vit/patch_embed"])
    for i in range(32):
        key = f"vit/block_{i:02d}"
        if key in cap.captures:
            _save(f"{output_dir}/01_vit/block_{i:02d}.safetensors",
                  cap.captures[key])

    # 02_neck: FPN scales
    for scale in ["4x", "2x", "1x", "05x"]:
        key = f"neck/scale_{scale}"
        if key in cap.captures:
            _save(f"{output_dir}/02_neck/scale_{scale}.safetensors",
                  cap.captures[key])

    # 03_pos_encoding: save the position encodings from backbone_out
    if "vision_pos_enc" in backbone_out:
        pos_tensors = {}
        pos_list = backbone_out["vision_pos_enc"]
        if isinstance(pos_list, list):
            for j, p in enumerate(pos_list):
                if isinstance(p, torch.Tensor):
                    pos_tensors[f"pos_enc_{j}"] = p
        _save(f"{output_dir}/03_pos_encoding/tensors.safetensors", pos_tensors)

    # 04_text_encoder: token embed, each block, final
    if "text/token_embed" in cap.captures:
        _save(f"{output_dir}/04_text_encoder/token_embed.safetensors",
              cap.captures["text/token_embed"])
    for i in range(24):
        key = f"text/block_{i:02d}"
        if key in cap.captures:
            _save(f"{output_dir}/04_text_encoder/block_{i:02d}.safetensors",
                  cap.captures[key])
    for suffix in ["ln_final", "resizer"]:
        key = f"text/{suffix}"
        if key in cap.captures:
            _save(f"{output_dir}/04_text_encoder/{suffix}.safetensors",
                  cap.captures[key])

    # Also save the final text features from backbone_out
    text_tensors = {}
    for k in ["language_features", "language_mask", "language_embeds"]:
        if k in backbone_out and isinstance(backbone_out[k], torch.Tensor):
            text_tensors[k] = backbone_out[k]
    if text_tensors:
        _save(f"{output_dir}/04_text_encoder/output.safetensors", text_tensors)

    # 05_geometry_encoder: layers + output
    for i in range(3):
        key = f"geometry/layer_{i:02d}"
        if key in cap.captures:
            _save(f"{output_dir}/05_geometry_encoder/layer_{i:02d}.safetensors",
                  cap.captures[key])

    # 06_encoder_fusion: layers
    for i in range(6):
        key = f"encoder/layer_{i:02d}"
        if key in cap.captures:
            _save(f"{output_dir}/06_encoder_fusion/layer_{i:02d}.safetensors",
                  cap.captures[key])

    # Save encoder output from the outputs dict
    if "encoder_hidden_states" in outputs:
        _save(f"{output_dir}/06_encoder_fusion/output.safetensors", {
            "encoder_hidden_states": outputs["encoder_hidden_states"],
        })

    # 07_decoder: layers + output
    for i in range(6):
        key = f"decoder/layer_{i:02d}"
        if key in cap.captures:
            _save(f"{output_dir}/07_decoder/layer_{i:02d}.safetensors",
                  cap.captures[key])
    if "decoder/final_norm" in cap.captures:
        _save(f"{output_dir}/07_decoder/final_norm.safetensors",
              cap.captures["decoder/final_norm"])

    # Save decoder outputs
    dec_out = {}
    for k in ["pred_logits", "pred_boxes", "pred_boxes_xyxy",
              "queries", "presence_logit_dec"]:
        if k in outputs and isinstance(outputs[k], torch.Tensor):
            dec_out[k] = outputs[k]
    if dec_out:
        _save(f"{output_dir}/07_decoder/output.safetensors", dec_out)

    # 08_dot_scorer
    if "dot_scorer" in cap.captures:
        _save(f"{output_dir}/08_dot_scorer/tensors.safetensors",
              cap.captures["dot_scorer"])

    # 09_seg_head: pixel decoder, instance proj, mask logits
    for sub in ["pixel_decoder", "instance_proj", "mask_predictor"]:
        key = f"seg/{sub}"
        if key in cap.captures:
            _save(f"{output_dir}/09_seg_head/{sub}.safetensors",
                  cap.captures[key])

    # Save final mask outputs
    final = {}
    for k in ["pred_masks"]:
        if k in outputs and isinstance(outputs[k], torch.Tensor):
            final[k] = outputs[k]
    if final:
        _save(f"{output_dir}/09_seg_head/output.safetensors", final)

    # 10_final: complete outputs
    all_out = {}
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            all_out[k] = v
    if all_out:
        _save(f"{output_dir}/10_final/tensors.safetensors", all_out)

    # Print summary of captured vs missed hooks
    print(f"\n--- Capture summary ---")
    print(f"Registered hooks captured: {len(cap.captures)} groups")
    total_tensors = sum(len(d) for d in cap.captures.values())
    print(f"Total tensors saved: {total_tensors}")

    return outputs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate per-layer test fixtures from reference SAM3")
    parser.add_argument("--checkpoint", default="models/sam3.pt",
                        help="Path to sam3.pt checkpoint")
    parser.add_argument("--bpe", default="models/bpe_simple_vocab_16e6.txt.gz",
                        help="Path to BPE vocabulary file")
    parser.add_argument("--image", default="assets/cat.jpeg",
                        help="Input image path")
    parser.add_argument("--output", default="tests/fixtures",
                        help="Output directory for fixture files")
    parser.add_argument("--text", default="cat",
                        help="Text prompt")
    parser.add_argument("--point", default=None,
                        help="Point prompt as x,y,label (pixel coords, label 1=pos). "
                             "Omit for text-only prompts.")
    parser.add_argument("--device", default="cpu",
                        help="Device (cpu recommended for determinism)")
    args = parser.parse_args()

    # Parse point (None for text-only)
    point = None
    if args.point is not None:
        parts = args.point.split(",")
        point = (int(parts[0]), int(parts[1]), int(parts[2]))

    # Determinism
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    print("=" * 60)
    print("SAM3 Fixture Generator")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Image:      {args.image}")
    print(f"Text:       {args.text}")
    print(f"Point:      {point or '(text-only)'}")
    print(f"Device:     {args.device}")
    print(f"Output:     {args.output}")
    print()

    # Build model
    # The reference code hardcodes device="cuda" in several __init__ methods.
    # We redirect all CUDA device references to CPU during construction.
    #
    # Also patch addmm_act: the upstream fused kernel converts to bf16 for GPU
    # perf, which fails on CPU. Replace with a plain linear+activation in f32.
    import sam3.perflib.fused as _fused_mod
    def _addmm_act_f32(activation, linear, mat1):
        x = torch.nn.functional.linear(mat1, linear.weight, linear.bias)
        if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
            return torch.nn.functional.gelu(x)
        if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
            return torch.nn.functional.relu(x)
        raise ValueError(f"Unexpected activation {activation}")
    _fused_mod.addmm_act = _addmm_act_f32
    # Also patch the import in vitdet which may have cached the old ref
    import sam3.model.vitdet as _vitdet_mod
    _vitdet_mod.addmm_act = _addmm_act_f32

    # Patch pin_memory to be a no-op on CPU (upstream uses it for async CUDA
    # transfers; on macOS it tries MPS pinning which fails)
    _orig_pin_memory = torch.Tensor.pin_memory
    torch.Tensor.pin_memory = lambda self, *a, **kw: self

    print("Loading model...")
    with _force_cpu_device():
        model = build_sam3_image_model(
            bpe_path=args.bpe,
            device=args.device,
            eval_mode=True,
            checkpoint_path=args.checkpoint,
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        )
    # Convert all parameters to float32 for deterministic CPU inference
    # (checkpoint may have bf16 weights from GPU training)
    model = model.float()
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")

    # Preprocess image
    print("Preprocessing image...")
    image_tensor, orig_w, orig_h = preprocess_image(args.image)
    print(f"Image: {orig_w}x{orig_h} -> {image_tensor.shape}")

    # Run inference and save
    outputs = run_and_save(model, image_tensor, args.text, point,
                           args.output, args.device)

    # Save metadata
    meta = {
        "image": args.image,
        "original_width": orig_w,
        "original_height": orig_h,
        "text_prompt": args.text,
        "point_prompt": list(point) if point else None,
        "prompt_mode": "point_text" if point else "text_only",
        "image_size": 1008,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "torch_version": torch.__version__,
    }
    meta_path = os.path.join(args.output, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  {meta_path}")

    # Show final output shapes
    print("\n--- Final outputs ---")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {list(v.shape)}")
        elif isinstance(v, dict):
            print(f"  {k}: <dict with {len(v)} keys>")

    print("\nDone!")


if __name__ == "__main__":
    main()
