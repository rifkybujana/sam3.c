#!/usr/bin/env python3
"""
Dump Python-reference intermediate tensors for a single frame of
kids.mp4 so we can compare layer-by-layer against our C engine's
/tmp/dbg_*.bin files.

Writes NHWC f32 raw-binary files (matching C dump format):
  /tmp/py_vit_nhwc.bin     [1, 72, 72, 1024]  (after ViT, before neck)
  /tmp/py_neck_4x.bin      [1, 288, 288, 256]
  /tmp/py_neck_2x.bin      [1, 144, 144, 256]
  /tmp/py_neck_1x.bin      [1, 72, 72, 256]
  /tmp/py_neck_05x.bin     [1, 36, 36, 256]

Run with the same PYTHONPATH as gen_fixture.py.
"""
import os
import sys
import numpy as np
import torch

# Same patches as gen_fixture.py: redirect cuda to cpu, stub autocast.
TARGET = "cpu"
_tz = torch.zeros
def _tz_cpu(*a, **k):
    if k.get("device") == "cuda": k["device"] = TARGET
    return _tz(*a, **k)
torch.zeros = _tz_cpu
_te = torch.empty
def _te_cpu(*a, **k):
    if k.get("device") == "cuda": k["device"] = TARGET
    return _te(*a, **k)
torch.empty = _te_cpu
torch.nn.Module.cuda = lambda self, *a, **k: self.to(TARGET)
torch.Tensor.cuda = lambda self, *a, **k: self.to(TARGET)
_dev = torch.device
def _dev_cpu(spec, *a, **k):
    if isinstance(spec, str) and spec.startswith("cuda"): spec = TARGET
    return _dev(spec, *a, **k)
torch.device = _dev_cpu
class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __init__(self, *a, **k): pass
    def __call__(self, f): return f
torch.autocast = _NullCtx
torch.cuda.amp.autocast = _NullCtx
_to = torch.Tensor.to
def _to_f32_if_bf16(self, *a, **k):
    if k.get("dtype") == torch.bfloat16: k["dtype"] = torch.float32
    a = tuple(torch.float32 if x is torch.bfloat16 else x for x in a)
    return _to(self, *a, **k)
torch.Tensor.to = _to_f32_if_bf16
torch.Tensor.pin_memory = lambda self, *a, **k: self

CKPT = "/Users/rbisri/Documents/sam3/models/sam3.pt"
FRAME_JPG = "/tmp/kids_jpgs/0.jpg"

def dump_nhwc(path, tensor):
    """Convert a torch tensor to NHWC f32 and dump raw bytes."""
    t = tensor.detach().cpu().float()
    if t.dim() == 4 and t.shape[1] in (256, 1024):
        # [B, C, H, W] -> [B, H, W, C]
        t = t.permute(0, 2, 3, 1).contiguous()
    arr = t.numpy().astype(np.float32, copy=False)
    arr.tofile(path)
    print(f"dump {path}: shape={list(tensor.shape)} -> NHWC [{','.join(str(s) for s in arr.shape)}] "
          f"({arr.size} floats)", flush=True)

def main():
    from sam3.model_builder import build_tracker

    print("Building tracker...", flush=True)
    tracker = build_tracker(apply_temporal_disambiguation=False,
                            with_backbone=True, compile_mode=None)
    tracker = tracker.to("cpu").float().eval()

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    tracker_state = {}
    for k, v in state.items():
        if k.startswith("tracker."):
            tracker_state[k[len("tracker."):]] = v
        elif k.startswith("detector.backbone."):
            tracker_state[k[len("detector."):]] = v
    miss, unex = tracker.load_state_dict(tracker_state, strict=False)
    print(f"state loaded: missing={len(miss)} unexpected={len(unex)}", flush=True)

    # Load and preprocess frame 0 exactly as the reference JPEG loader does
    from PIL import Image
    img = Image.open(FRAME_JPG).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    # Resize to image_size x image_size (letterbox per reference; actually reference
    # JUST resizes square — load_video_frames_from_jpg_images simply loads into
    # [3, image_size, image_size] via _load_img_as_tensor which calls
    # transforms.Resize(size=(image_size, image_size))).
    from torchvision.transforms.functional import resize, to_tensor
    img_tensor = to_tensor(img)  # [3, H, W], f32 in [0, 1]
    img_tensor = resize(img_tensor, [tracker.image_size, tracker.image_size])
    # Normalize by mean/std (0.5 each)
    img_mean = torch.tensor([0.5, 0.5, 0.5])[:, None, None]
    img_std  = torch.tensor([0.5, 0.5, 0.5])[:, None, None]
    img_tensor = (img_tensor - img_mean) / img_std
    img_tensor = img_tensor.unsqueeze(0).float()   # [1, 3, H, W]
    print(f"input: {img_tensor.shape} range=[{img_tensor.min():.3f}, {img_tensor.max():.3f}]",
          flush=True)

    # Backbone forward. The backbone is SAM3VLBackbone wrapping a vision backbone
    # (ViT + neck). Its forward returns a dict. Let's run just the vision part.
    with torch.no_grad():
        # tracker.backbone is SAM3VLBackbone, vision_backbone is Sam3DualViTDetNeck.
        vis = tracker.backbone.vision_backbone
        # Trunk is the ViT itself (vis.trunk).
        vit = vis.trunk
        # Hook patch embedding + pos + ln_pre output. Attributes differ
        # per backbone; probe to find patch_embed / pos_embed / ln_pre.
        trunk = vit
        attrs = [a for a in dir(trunk) if not a.startswith('_')]
        print(f"trunk attrs: {attrs}", flush=True)
        # Find patch-embed-like attribute
        pe_mod = None
        for name in ("patch_embed", "embeddings", "conv1"):
            if hasattr(trunk, name):
                pe_mod = getattr(trunk, name)
                print(f"patch_embed-like: trunk.{name} = "
                      f"{type(pe_mod).__name__}", flush=True)
                break
        # Try calling patch_embed
        if pe_mod is not None:
            try:
                x = pe_mod(img_tensor)
                print(f"after trunk.{name}: "
                      f"{x.shape if hasattr(x,'shape') else type(x)}",
                      flush=True)
            except Exception as e:
                print(f"trunk.{name} failed: {e}", flush=True)
                x = None
        # Instead of reassembling, register a forward hook on the first block
        # to capture its INPUT — that's the output of patch_embed+pos+ln_pre.
        block0 = trunk.blocks[0]
        captured = {}
        def hook(module, inputs, outputs):
            captured["input"] = inputs[0] if isinstance(inputs, tuple) else inputs
            captured["output"] = outputs
        h = block0.register_forward_hook(hook)
        _ = trunk(img_tensor)  # full forward (wasteful but deterministic)
        h.remove()
        pre_blocks = captured.get("input")
        if pre_blocks is not None:
            print(f"block0 input shape: {pre_blocks.shape}", flush=True)
            arr = pre_blocks.detach().cpu().float().contiguous().numpy(
                ).astype(np.float32)
            arr.tofile("/tmp/py_vit_patch.bin")
            print(f"dump /tmp/py_vit_patch.bin: shape={list(pre_blocks.shape)} "
                  f"({arr.size} floats)", flush=True)

        vit_out_raw = vit(img_tensor)
        # Trunk may return list/tuple of feature maps; use the top-level one.
        if isinstance(vit_out_raw, (list, tuple)):
            print(f"vit returned {len(vit_out_raw)} tensors:", flush=True)
            for i, t in enumerate(vit_out_raw):
                print(f"  [{i}]: {getattr(t, 'shape', type(t).__name__)}",
                      flush=True)
            vit_out = vit_out_raw[-1]
        else:
            vit_out = vit_out_raw
        print(f"vit_out: {vit_out.shape}", flush=True)
        dump_nhwc("/tmp/py_vit_nhwc.bin", vit_out)

        # Sam3DualViTDetNeck.forward returns 4 things; the first list is
        # the SAM3 FPN pyramid at scales [4x, 2x, 1x, 05x].
        sam3_features, sam3_pos, sam2_features, sam2_pos = vis.forward(img_tensor)
        print(f"sam3 fpn n={len(sam3_features)}", flush=True)
        names = ["4x", "2x", "1x", "05x"]
        for i, t in enumerate(sam3_features):
            name = names[i] if i < len(names) else f"lvl{i}"
            print(f"  sam3[{i}] {name}: {t.shape}", flush=True)
            dump_nhwc(f"/tmp/py_neck_{name}.bin", t)
        if sam2_features is not None:
            print(f"sam2 features n={len(sam2_features)}", flush=True)
            for i, t in enumerate(sam2_features):
                print(f"  sam2[{i}]: {t.shape}", flush=True)

if __name__ == "__main__":
    main()
