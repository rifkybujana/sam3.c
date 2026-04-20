#!/usr/bin/env python3
"""
Dump per-layer inputs/outputs from the Python SAM 3.1 video reference
for the first 3 frames of a video (default: assets/kids.mp4), seeded
with a committed mask (default: tests/fixtures/video_kids/sam3_1/
seed_mask.png). Also writes PNG snapshots of the input frame and the
final segmentation mask for each frame.

Coverage (hooks fire in forward order):
  - backbone: patch_embed, ln_pre, each of the 32 ViT blocks, ln_post
  - neck: each of the 3 propagation_convs (sam2 side, feeds tracker)
  - memory-attn encoder: each of the 4 decoupled cross-attn layers
    (+ final norm), with encoder-level INPUT kwargs captured so the
    tgt / memory / memory_image / memory_image_pos tensors are
    dumpable
  - mask decoder: each of the 2 two-way transformer blocks, the final
    attn, output_upscaling, obj_score head, iou head, and the decoder
    dict return
  - output_dict: final mask + obj_score for each frame

Outputs (all under --out, default output/sam3_1_layer_dumps/):
  pngs/frame_{NNNN}_input.png    input RGB (uint8, native res)
  pngs/frame_{NNNN}_mask_obj1.png  binary mask at 288x288 (object 1)
  bins/frame_{NNNN}/NN_<name>.bin  F32 raw-binary for each layer
  shapes.json                    dump-name -> shape map
  README.md                      regenerated summary of what's inside

Usage:
  SAM3_CKPT=models/sam3.1_multiplex.pt \\
    python3 scripts/dump_sam3_1_all_layers.py \\
      --video assets/kids.mp4 \\
      --seed-mask tests/fixtures/video_kids/sam3_1/seed_mask.png \\
      --frames 3
"""
import argparse
import json
import os
import sys

import numpy as np

# Upstream sam3 reference and CPU shims.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "reference", "sam3"))
sys.path.insert(0, os.path.join(_ROOT, "tools"))

from _cpu_patches import (  # noqa: E402
    install_triton_stub, install_cuda_redirect, install_addmm_act_fp32,
)
install_triton_stub()
install_cuda_redirect()

import torch  # noqa: E402
from PIL import Image  # noqa: E402


# ---------------------------------------------------------------------
# Dump helpers
# ---------------------------------------------------------------------

def _to_numpy(t):
    if not isinstance(t, torch.Tensor):
        return None
    return t.detach().cpu().float().contiguous().numpy().astype(np.float32)


def _dump_bin(path, arr, shapes, dump_name):
    """Write F32 numpy to disk, record shape for shapes.json."""
    arr.tofile(path)
    shapes[dump_name] = list(arr.shape)


def _save_mask_png(path, logits_2d, size=288):
    """logits_2d is a [H,W] torch tensor; binarize at 0 and resize
    nearest. Accepts a decoder logits tensor (threshold at 0) or a
    binary/uint8 mask (threshold at 0.5, treated as already-binary)."""
    if logits_2d.dim() == 4:
        logits_2d = logits_2d.squeeze(0).squeeze(0)
    if logits_2d.dim() == 3:
        logits_2d = logits_2d[0]
    # If the tensor is already in [0, 1] / binary, threshold at 0.5.
    # Otherwise treat it as decoder logits and threshold at 0.
    mx = float(logits_2d.max()) if logits_2d.numel() > 0 else 0.0
    mn = float(logits_2d.min()) if logits_2d.numel() > 0 else 0.0
    threshold = 0.5 if (mn >= 0.0 and mx <= 1.5) else 0.0
    binary = (logits_2d > threshold).float()[None, None]
    resized = torch.nn.functional.interpolate(
        binary, size=(size, size), mode="nearest")
    arr = (resized.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    Image.fromarray(arr).save(path)


def _save_rgb_png(path, rgb_chw):
    """rgb_chw is [3,H,W] in ~[-1,1] after mean/std normalization.
    Invert the normalization for a human-readable preview."""
    # ImageNet-ish mean/std used by SAM's frame loader.
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    x = rgb_chw.cpu().float() * std + mean
    x = (x.clamp(0, 1) * 255).permute(1, 2, 0).numpy().astype(np.uint8)
    Image.fromarray(x).save(path)


# ---------------------------------------------------------------------
# Model build + state init
# ---------------------------------------------------------------------

def _build_model(checkpoint):
    from sam3.model_builder import build_sam3_multiplex_video_model
    model = build_sam3_multiplex_video_model(
        checkpoint_path=None, load_from_HF=False, multiplex_count=16,
        use_fa3=False, use_rope_real=False,
        strict_state_dict_loading=False, device="cpu", compile=False,
    )
    install_addmm_act_fp32()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    merged = {}
    for k, v in ckpt.items():
        if k.startswith("tracker.model."):
            merged[k[len("tracker.model."):]] = v
    for k, v in ckpt.items():
        if k.startswith("detector.backbone."):
            merged[k[len("detector."):]] = v
    missing, unexpected = model.load_state_dict(merged, strict=False)
    print(f"state_dict: loaded={len(merged)} missing={len(missing)} "
          f"unexpected={len(unexpected)}", file=sys.stderr)
    return model.float().eval()


def _install_video_patches():
    """Same shims as gen_video_parity_fixtures.py."""
    import gen_video_parity_fixtures as _gvp
    _gvp._patch_load_video_frames()
    _gvp._patch_forward_image_clone_loop()


def _init_state(model, video_path):
    from sam3.model.video_tracking_multiplex_demo import (
        VideoTrackingMultiplexDemo,
    )
    state = VideoTrackingMultiplexDemo.init_state(
        model, video_path=video_path,
        offload_video_to_cpu=True, offload_state_to_cpu=True,
        async_loading_frames=False, use_cv2=True,
    )
    if not torch.cuda.is_available():
        state["device"] = torch.device("cpu")
        state["storage_device"] = torch.device("cpu")
    return state


# ---------------------------------------------------------------------
# Hook wiring
# ---------------------------------------------------------------------

class _Captures:
    """Slot-indexed append-only capture store.

    Each slot is a list of torch.Tensors (or dicts for inputs). A
    cursor tracks how many entries have been flushed, so each
    propagate yield can emit only the deltas for that frame.
    """

    def __init__(self):
        self.d = {}
        self.cursors = {}

    def append(self, name, value):
        self.d.setdefault(name, []).append(value)

    def delta(self, name):
        start = self.cursors.get(name, 0)
        end = len(self.d.get(name, []))
        self.cursors[name] = end
        if end > start:
            return self.d[name][end - 1]
        return None


def _register_hooks(model, cap):
    hooks = []

    # --- backbone trunk: patch_embed, ln_pre, 32 blocks, ln_post ---
    trunk = model.backbone.vision_backbone.trunk

    def _cap_plain(name):
        def _h(_m, _inp, out):
            if isinstance(out, torch.Tensor):
                cap.append(name, out.detach().cpu().float().contiguous())
        return _h

    hooks.append(trunk.patch_embed.register_forward_hook(
        _cap_plain("vit_patch_embed")))
    hooks.append(trunk.ln_pre.register_forward_hook(
        _cap_plain("vit_ln_pre")))
    for i in range(len(trunk.blocks)):
        hooks.append(trunk.blocks[i].register_forward_hook(
            _cap_plain(f"vit_block_{i:02d}")))
    hooks.append(trunk.ln_post.register_forward_hook(
        _cap_plain("vit_ln_post")))

    # --- neck convs (propagation side feeds the tracker) ---
    vb = model.backbone.vision_backbone
    for i, conv in enumerate(vb.propagation_convs):
        hooks.append(conv.register_forward_hook(
            _cap_plain(f"neck_prop_conv_{i}")))
    for i, conv in enumerate(vb.convs):
        hooks.append(conv.register_forward_hook(
            _cap_plain(f"neck_sam3_conv_{i}")))

    # --- forward_image sam2/sam3 outputs + frame RGB input ---
    _orig_fwd_image = model.forward_image

    def _patched_forward_image(img_batch, **kwargs):
        out = _orig_fwd_image(img_batch, **kwargs)
        img_t = img_batch.tensors if hasattr(img_batch, "tensors") \
            else img_batch
        cap.append("frame_rgb",
                   img_t.detach().cpu().float().contiguous())
        sam2 = out.get("sam2_backbone_out") if isinstance(out, dict) \
            else None
        if isinstance(sam2, dict):
            fpn = sam2.get("backbone_fpn")
            if isinstance(fpn, list) and len(fpn) > 0:
                for i, fm in enumerate(fpn):
                    t = fm.tensors if hasattr(fm, "tensors") else fm
                    cap.append(f"sam2_fpn_{i}",
                               t.detach().cpu().float().contiguous())
        return out

    model.forward_image = _patched_forward_image

    class _UnpatchFwdImage:
        def remove(self_inner):
            model.forward_image = _orig_fwd_image
    hooks.append(_UnpatchFwdImage())

    # --- _prepare_memory_conditioned_features captures image_embed ---
    _orig_prep = model._prepare_memory_conditioned_features

    def _patched_prep(*args, **kwargs):
        cvf = kwargs.get("current_vision_feats")
        if isinstance(cvf, (list, tuple)) and len(cvf) > 0 \
                and isinstance(cvf[-1], torch.Tensor):
            cap.append("image_embed",
                       cvf[-1].detach().cpu().float().contiguous())
        return _orig_prep(*args, **kwargs)

    model._prepare_memory_conditioned_features = _patched_prep

    class _UnpatchPrep:
        def remove(self_inner):
            model._prepare_memory_conditioned_features = _orig_prep
    hooks.append(_UnpatchPrep())

    # --- memory-attention encoder: per-layer + encoder inputs ---
    enc = model.transformer.encoder

    def _cap_enc(_m, _inp, kwargs, out):
        # Encoder output.
        if isinstance(out, dict):
            mem = out.get("memory")
            if isinstance(mem, torch.Tensor):
                cap.append("memattn_encoder_out",
                           mem.detach().cpu().float().contiguous())
        # Encoder INPUT kwargs (tgt/memory/memory_image/memory_image_pos).
        names = {"src": "memattn_in_tgt", "memory": "memattn_in_memory",
                 "memory_image": "memattn_in_memory_image",
                 "memory_image_pos": "memattn_in_memory_image_pos",
                 "image": "memattn_in_image",
                 "memory_pos": "memattn_in_memory_pos",
                 "src_pos": "memattn_in_src_pos"}
        for key, slot in names.items():
            v = kwargs.get(key)
            if isinstance(v, torch.Tensor):
                cap.append(slot,
                           v.detach().cpu().float().contiguous())
    hooks.append(enc.register_forward_hook(
        _cap_enc, with_kwargs=True))

    for i, layer in enumerate(enc.layers):
        def _cap_layer(_m, _inp, out, _i=i):
            # Layer returns (image, output).
            t = out[1] if isinstance(out, tuple) and len(out) >= 2 \
                else (out if isinstance(out, torch.Tensor) else None)
            if isinstance(t, torch.Tensor):
                cap.append(f"memattn_layer_{_i}_out",
                           t.detach().cpu().float().contiguous())
        hooks.append(layer.register_forward_hook(_cap_layer))

    if hasattr(enc, "norm"):
        hooks.append(enc.norm.register_forward_hook(
            _cap_plain("memattn_final_norm")))

    # --- mask decoder: two-way transformer blocks + heads ---
    md = model.sam_mask_decoder

    for i, layer in enumerate(md.transformer.layers):
        def _cap_mdlayer(_m, _inp, out, _i=i):
            # TwoWayAttentionBlock returns (queries, keys).
            if isinstance(out, tuple) and len(out) >= 2:
                q, k = out[0], out[1]
                if isinstance(q, torch.Tensor):
                    cap.append(f"mdec_twt_{_i}_queries",
                               q.detach().cpu().float().contiguous())
                if isinstance(k, torch.Tensor):
                    cap.append(f"mdec_twt_{_i}_keys",
                               k.detach().cpu().float().contiguous())
        hooks.append(layer.register_forward_hook(_cap_mdlayer))

    if hasattr(md.transformer, "final_attn_token_to_image"):
        hooks.append(
            md.transformer.final_attn_token_to_image.register_forward_hook(
                _cap_plain("mdec_twt_final_attn")))
    if hasattr(md.transformer, "norm_final_attn"):
        hooks.append(
            md.transformer.norm_final_attn.register_forward_hook(
                _cap_plain("mdec_twt_norm_final")))

    # With use_high_res_features=True the mask decoder unpacks
    # output_upscaling into (dc1, ln1, act1, dc2, act2) and calls the
    # sub-modules individually, so the Sequential wrapper's forward
    # hook never fires. Hook each sub-module instead.
    _up_names = ["dc1", "ln1", "act1", "dc2", "act2"]
    for i, sub in enumerate(md.output_upscaling):
        hooks.append(sub.register_forward_hook(
            _cap_plain(f"mdec_upscale_{_up_names[i]}")))
    # High-res side-path convs (applied to backbone_fpn[0/1] during
    # forward_image). Fire once per frame.
    if hasattr(md, "conv_s0"):
        hooks.append(md.conv_s0.register_forward_hook(
            _cap_plain("mdec_conv_s0")))
    if hasattr(md, "conv_s1"):
        hooks.append(md.conv_s1.register_forward_hook(
            _cap_plain("mdec_conv_s1")))
    hooks.append(md.iou_prediction_head.register_forward_hook(
        _cap_plain("mdec_iou_head")))
    hooks.append(md.pred_obj_score_head.register_forward_hook(
        _cap_plain("mdec_obj_score_head")))

    # Top-level mask decoder dict (masks/iou/sam_tokens/obj_score).
    def _cap_md_top(_m, _inp, out):
        if isinstance(out, dict):
            for key, slot in (("masks", "mdec_out_masks"),
                              ("iou_pred", "mdec_out_iou"),
                              ("sam_tokens_out", "mdec_out_sam_tokens"),
                              ("object_score_logits",
                               "mdec_out_obj_score")):
                v = out.get(key)
                if isinstance(v, torch.Tensor):
                    cap.append(slot,
                               v.detach().cpu().float().contiguous())
    hooks.append(md.register_forward_hook(_cap_md_top))

    return hooks


# ---------------------------------------------------------------------
# Flushing captures per frame
# ---------------------------------------------------------------------

# Slots that are image-shaped NCHW and should be permuted to NHWC on
# dump (matches the C engine's byte layout). Everything else dumps
# verbatim.
_NHWC_SLOTS = {
    "frame_rgb", "sam2_fpn_0", "sam2_fpn_1", "sam2_fpn_2",
    "neck_prop_conv_0", "neck_prop_conv_1", "neck_prop_conv_2",
    "neck_sam3_conv_0", "neck_sam3_conv_1", "neck_sam3_conv_2",
    "mdec_upscale_dc1", "mdec_upscale_ln1", "mdec_upscale_act1",
    "mdec_upscale_dc2", "mdec_upscale_act2",
    "mdec_conv_s0", "mdec_conv_s1", "mdec_out_masks",
}


def _ordered_slot_names(trunk_nblocks, memattn_nlayers,
                       mdec_nlayers):
    """Return the dump order for bin filenames (NN_<slot> prefix).

    Order roughly follows execution: input -> backbone -> neck ->
    memory-attn -> mask decoder -> output.
    """
    names = ["frame_rgb",
             "vit_patch_embed", "vit_ln_pre"]
    names += [f"vit_block_{i:02d}" for i in range(trunk_nblocks)]
    names += ["vit_ln_post"]
    names += [f"neck_sam3_conv_{i}" for i in range(3)]
    names += [f"neck_prop_conv_{i}" for i in range(3)]
    names += [f"sam2_fpn_{i}" for i in range(3)]
    names += ["mdec_conv_s0", "mdec_conv_s1"]
    names += ["image_embed"]
    names += ["memattn_in_tgt", "memattn_in_image",
              "memattn_in_memory", "memattn_in_memory_image",
              "memattn_in_memory_image_pos", "memattn_in_memory_pos",
              "memattn_in_src_pos"]
    names += [f"memattn_layer_{i}_out"
              for i in range(memattn_nlayers)]
    names += ["memattn_final_norm", "memattn_encoder_out"]
    for i in range(mdec_nlayers):
        names += [f"mdec_twt_{i}_queries", f"mdec_twt_{i}_keys"]
    names += ["mdec_twt_final_attn", "mdec_twt_norm_final",
              "mdec_upscale_dc1", "mdec_upscale_ln1",
              "mdec_upscale_act1", "mdec_upscale_dc2",
              "mdec_upscale_act2",
              "mdec_iou_head", "mdec_obj_score_head",
              "mdec_out_masks", "mdec_out_iou",
              "mdec_out_sam_tokens", "mdec_out_obj_score"]
    return names


def _flush_frame(cap, frame_idx, out_dir, shapes, slot_names):
    """Write every captured-this-frame tensor to
    bins/frame_{NNNN}/NN_<name>.bin. Also write input RGB + output
    mask PNGs."""
    bins_dir = os.path.join(out_dir, "bins", f"frame_{frame_idx:04d}")
    os.makedirs(bins_dir, exist_ok=True)

    for idx, slot in enumerate(slot_names):
        t = cap.delta(slot)
        if t is None:
            continue
        if not isinstance(t, torch.Tensor):
            continue
        x = t.contiguous()
        if slot in _NHWC_SLOTS and x.dim() == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
        arr = x.numpy().astype(np.float32)
        fname = f"{idx:02d}_{slot}.bin"
        dump_name = f"frame_{frame_idx:04d}/{fname}"
        _dump_bin(os.path.join(bins_dir, fname), arr, shapes, dump_name)

    # PNGs from raw captures (frame_rgb + mdec_out_masks were appended
    # already and may have been consumed by the loop above — re-read
    # from the raw store rather than the delta cursor).
    pngs_dir = os.path.join(out_dir, "pngs")
    os.makedirs(pngs_dir, exist_ok=True)

    # frame_rgb: index is frame_idx+1 because frame 0's backbone runs
    # both during add_new_masks(preflight=True) setup... actually
    # add_new_masks doesn't call forward_image; forward_image is
    # called once per frame during propagate. The entries arrive in
    # frame order, so entry `frame_idx` is this frame's RGB.
    raw_rgb = cap.d.get("frame_rgb", [])
    if len(raw_rgb) > frame_idx:
        rgb = raw_rgb[frame_idx]
        if rgb.dim() == 4:
            rgb = rgb.squeeze(0)
        _save_rgb_png(os.path.join(pngs_dir,
                                   f"frame_{frame_idx:04d}_input.png"),
                      rgb)

    # Mask PNGs are saved separately from the yielded
    # `video_res_masks` in the propagate loop — that's the full
    # production output (post-gating, best-slot-selected) and so
    # matches what the C side's sam3_video_propagate returns. Using
    # the raw `mdec_out_masks[0][0]` hook capture here would show
    # an arbitrary slot/head, which is not what the tracker
    # actually commits.


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

def _write_readme(out_dir, n_frames, slot_names):
    path = os.path.join(out_dir, "README.md")
    with open(path, "w") as f:
        f.write("# SAM 3.1 Python reference — per-layer dumps\n\n")
        f.write("Generated by `scripts/dump_sam3_1_all_layers.py` "
                f"for the first {n_frames} frame(s) of the input "
                "video, seeded with the committed mask.\n\n")
        f.write("## Layout\n\n")
        f.write("- `pngs/frame_{NNNN}_input.png` — input RGB "
                "(de-normalized for preview)\n")
        f.write("- `pngs/frame_{NNNN}_mask_obj1.png` — "
                "binary object-1 mask at 288x288\n")
        f.write("- `bins/frame_{NNNN}/NN_<slot>.bin` — F32 raw "
                "binary of that layer's output. NHWC for image-shaped "
                "4-D slots; verbatim for token/score tensors with the "
                "leading singleton batch dim squeezed.\n")
        f.write("- `shapes.json` — slot -> numpy shape map\n\n")
        f.write("## Frame-by-frame coverage note\n\n")
        f.write("The seed frame (`frame_0000`) is handled by "
                "`add_new_masks + propagate_in_video_preflight`. "
                "This path runs the image encoder and neck to build "
                "the image features needed for later frames' memory "
                "attention, but does **not** invoke the memory "
                "attention or mask decoder — the seed mask IS the "
                "frame-0 output. So `bins/frame_0000/` only contains "
                "backbone and neck slots; `pngs/frame_0000_mask_obj1"
                ".png` is the seed mask downsampled to 288×288.\n\n"
                "Frames 1..N (`propagate_in_video`) go through the "
                "full pipeline and include every slot listed below.\n\n")
        f.write("## Dump order (per frame)\n\n")
        f.write("| NN | slot | kind |\n|---|---|---|\n")
        for i, s in enumerate(slot_names):
            kind = "NHWC image" if s in _NHWC_SLOTS else "tokens"
            f.write(f"| {i:02d} | `{s}` | {kind} |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(_ROOT,
                    "assets", "kids.mp4"))
    ap.add_argument("--seed-mask", default=os.path.join(_ROOT,
                    "tests", "fixtures", "video_kids", "sam3_1",
                    "seed_mask.png"))
    ap.add_argument("--frames", type=int, default=3,
                    help="Number of frames (starting at 0) to dump")
    ap.add_argument("--out", default=os.path.join(_ROOT,
                    "output", "sam3_1_layer_dumps"))
    ap.add_argument("--checkpoint",
                    default=os.environ.get("SAM3_CKPT",
                        os.path.join(_ROOT, "models",
                            "sam3.1_multiplex.pt")))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "bins"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "pngs"), exist_ok=True)

    _install_video_patches()
    print(f"[build] loading checkpoint {args.checkpoint}",
          file=sys.stderr)
    model = _build_model(args.checkpoint)
    print(f"[init] init_state video={args.video}", file=sys.stderr)
    state = _init_state(model, args.video)

    seed = np.array(Image.open(args.seed_mask).convert("L"))
    seed_bool = (seed > 127).astype(np.float32)
    seed_t = torch.from_numpy(seed_bool)[None, ...]

    cap = _Captures()
    hooks = _register_hooks(model, cap)

    n_trunk_blocks = len(model.backbone.vision_backbone.trunk.blocks)
    n_memattn = len(model.transformer.encoder.layers)
    n_mdec = len(model.sam_mask_decoder.transformer.layers)
    slot_names = _ordered_slot_names(n_trunk_blocks, n_memattn, n_mdec)

    shapes = {}
    try:
        print("[seed] add_new_masks + preflight (frame 0 forward)",
              file=sys.stderr)
        model.add_new_masks(inference_state=state, frame_idx=0,
                            obj_ids=[1], masks=seed_t)
        model.propagate_in_video_preflight(state)
        _flush_frame(cap, 0, args.out, shapes, slot_names)

        # The mask decoder is not invoked for the seed frame — the
        # seed mask IS the frame-0 mask. Save the seed (downsampled
        # to 288 to match the other frames) so every frame has a
        # visible mask PNG.
        _save_mask_png(os.path.join(
            args.out, "pngs", "frame_0000_mask_obj1.png"),
            torch.from_numpy((seed > 127).astype(np.float32)))

        print(f"[propagate] running {args.frames} frames",
              file=sys.stderr)
        count = 0
        max_track = args.frames  # 0-indexed: 0..frames-1
        for frame_idx, obj_ids, _low, video_res_masks, _score in \
                model.propagate_in_video(
                    inference_state=state,
                    start_frame_idx=0,
                    max_frame_num_to_track=max_track,
                    reverse=False):
            if frame_idx == 0:
                # Frame 0 already dumped after preflight. Skip the
                # re-yield so we don't overwrite with its cached
                # outputs.
                count += 1
                continue
            _flush_frame(cap, frame_idx, args.out, shapes, slot_names)

            # Two PNGs per frame, both for object 1:
            #   {frame}_mask_obj1.png         — production output
            #       (`video_res_masks` from propagate_in_video, post-
            #        obj_score gate, best-slot-selected). If Python's
            #        obj_score gate fires, this file is all-black.
            #   {frame}_mask_obj1_raw.png     — raw decoder best-slot
            #       pick with NO gating (argmax over all 16 multiplex
            #        slots by obj_score; mask head 0). Useful when the
            #        gate empties the production output — reveals what
            #        the decoder actually predicted before suppression.
            for obj_id, m in zip(obj_ids, video_res_masks):
                if int(obj_id) != 1:
                    continue
                _save_mask_png(os.path.join(
                    args.out, "pngs",
                    f"frame_{frame_idx:04d}_mask_obj1.png"), m)

            # Raw best-slot preview from the decoder hook capture.
            raw_masks = cap.d.get("mdec_out_masks", [])
            raw_scores = cap.d.get("mdec_out_obj_score", [])
            if len(raw_masks) > 0 and len(raw_scores) > 0:
                md = raw_masks[-1]  # [1, 16, 3, 288, 288]
                sc = raw_scores[-1]  # [1, 16, 1]
                best_slot = int(sc.squeeze().argmax())
                if md.dim() == 5:
                    md = md[0]  # [16, 3, H, W]
                best_mask = md[best_slot][0]  # mask head 0
                _save_mask_png(os.path.join(
                    args.out, "pngs",
                    f"frame_{frame_idx:04d}_mask_obj1_raw.png"),
                    best_mask)

            count += 1
            if count >= args.frames:
                break
    finally:
        for h in hooks:
            h.remove()

    with open(os.path.join(args.out, "shapes.json"), "w") as f:
        json.dump(shapes, f, indent=2, sort_keys=True)

    _write_readme(args.out, args.frames, slot_names)
    print(f"[done] dumped {args.frames} frames to {args.out}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
