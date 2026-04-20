#!/usr/bin/env python3
"""Generate per-frame mask PNGs from the Python reference predictor.

Variants:
  --variant sam3     - Sam3VideoPredictor + point prompts (legacy path,
                       unchanged regen flow for the existing
                       tests/fixtures/video_kids/ scaffold).
  --variant sam3.1   - Sam3VideoTrackingMultiplexDemo + add_new_masks seed
                       path. Requires --seed-mask (grayscale PNG produced
                       by ./build/sam3_1_dump_seed).

NOT run automatically. Invoke by hand when reference outputs need a
refresh. See tests/fixtures/video_kids/README.md and
tests/fixtures/video_kids/sam3_1/README.md.

Usage (sam3.1):

    SAM3_CKPT=... SAM3_BPE=... python tools/gen_video_parity_fixtures.py \\
        --variant sam3.1 \\
        --video ../assets/kids.mp4 \\
        --frames 3 \\
        --seed-mask ../tests/fixtures/video_kids/sam3_1/seed_mask.png \\
        --out ../tests/fixtures/video_kids/sam3_1/

Known limitation (SAM 3.1 variant):
  The upstream SAM 3.1 stack is CUDA-centric. This script applies the
  same CUDA->CPU shims as tools/dump_reference.py via _cpu_patches, but
  a full end-to-end regen has not yet been validated without a GPU.
  Exact call signatures (esp. bf16 autocast in Sam3MultiplexPredictorWrapper
  and init_state's video loader kwargs) may need adjustment on first run.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

# Upstream sam3 package lives under reference/
sys.path.insert(
    0,
    os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "reference", "sam3")),
)
# Shared CPU shims. Must run BEFORE any sam3 import.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _cpu_patches import (  # noqa: E402
    install_triton_stub, install_cuda_redirect, install_addmm_act_fp32,
)

install_triton_stub()
install_cuda_redirect()

import torch  # noqa: E402


def main_sam3(args):
    """Legacy SAM 3 regen path. Unchanged from pre-refactor."""
    from sam3.sam3_video_predictor import Sam3VideoPredictor

    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)

    predictor = Sam3VideoPredictor(
        checkpoint_path=os.environ["SAM3_CKPT"],
        bpe_path=os.environ["SAM3_BPE"],
    )
    state = predictor.init_state(video_path=args.video)

    prompts = {
        "obj_1": {"frame": 0, "points": [[400, 250]], "labels": [1]},
        "obj_2": {"frame": 0, "points": [[600, 250]], "labels": [1]},
    }
    with open(os.path.join(args.out, "prompts.json"), "w") as f:
        json.dump(prompts, f, indent=2)

    for name, p in prompts.items():
        obj_id = int(name.split("_")[1])
        predictor.add_new_points_or_box(
            state,
            frame_idx=p["frame"],
            obj_id=obj_id,
            points=np.array(p["points"]),
            labels=np.array(p["labels"]),
        )

    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
        if frame_idx >= args.frames:
            break
        for obj_id, m in zip(obj_ids, masks):
            arr = (m > 0).cpu().numpy().astype(np.uint8) * 255
            Image.fromarray(arr.squeeze()).save(os.path.join(
                args.out, "frames",
                f"frame_{frame_idx:04d}_obj_{obj_id}.png"))


def _build_sam3_1_demo_model(checkpoint):
    """Assemble a CPU-safe Sam3VideoTrackingMultiplexDemo.

    Uses build_sam3_multiplex_video_model (not the full predictor stack,
    which hardcodes CUDA bf16 autocast). The returned demo model exposes
    init_state / add_new_masks / propagate_in_video via its
    VideoTrackingMultiplexDemo base class.

    The sam3.1_multiplex.pt checkpoint stores all tracker weights under a
    `tracker.` prefix (alongside `detector.` — used by dump_reference.py
    image path). The builder's state_dict is at the top level with no
    prefix, so we strip `tracker.` before loading.
    """
    from sam3.model_builder import build_sam3_multiplex_video_model

    # Build the model with no checkpoint (we load state_dict manually
    # below after stripping the `tracker.` prefix).
    model = build_sam3_multiplex_video_model(
        checkpoint_path=None,
        load_from_HF=False,
        multiplex_count=16,
        use_fa3=False,
        use_rope_real=False,
        strict_state_dict_loading=False,
        device="cpu",
        compile=False,
    )
    install_addmm_act_fp32()

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    # Keys are nested `tracker.model.*` — strip both. The tracker
    # checkpoint has no vision backbone; the shared ViT lives under
    # `detector.backbone.*` and the tracker model reads it at
    # `backbone.*`, so we remap detector backbone keys too.
    merged = {}
    for k, v in ckpt.items():
        if k.startswith("tracker.model."):
            merged[k[len("tracker.model."):]] = v
    for k, v in ckpt.items():
        if k.startswith("detector.backbone."):
            merged[k[len("detector."):]] = v  # -> backbone.*
    missing, unexpected = model.load_state_dict(merged, strict=False)
    print(f"tracker state_dict: loaded {len(merged)}, "
          f"missing={len(missing)} unexpected={len(unexpected)}",
          file=sys.stderr)
    if missing:
        print(f"first missing: {missing[:3]}", file=sys.stderr)
    if unexpected:
        print(f"first unexpected: {unexpected[:3]}", file=sys.stderr)

    model = model.float().eval()
    return model


def _patch_forward_image_clone_loop():
    """Upstream VideoTrackingMultiplex.forward_image ends with a "clone
    to help torch.compile" loop that iterates backbone_out.keys() and
    dereferences each value as a dict. When the tri-neck backbone
    returns with need_sam3_out=True, the top-level keys
    (vision_features, vision_mask, vision_pos_enc, backbone_fpn) are
    tensors/lists, not dicts, so the loop crashes with
    IndexError: too many indices for tensor of dimension 4.

    Patch the loop to filter non-dict entries before cloning.
    """
    from sam3.model.video_tracking_multiplex import VideoTrackingMultiplex
    orig = VideoTrackingMultiplex.forward_image

    def _patched(self, img_batch, **kwargs):
        # Call the original but intercept the clone loop at the end.
        # Simplest: replicate the original body up to the clone loop,
        # or disable the clone loop entirely by temporarily patching
        # _maybe_clone to a no-op + filtering. We do the latter.
        orig_maybe_clone = self._maybe_clone

        def _safe_clone(x):
            return orig_maybe_clone(x)

        # We can't easily re-enter the upstream function minus the
        # broken loop without copy-pasting it. Instead, replace it.
        backbone_out = orig(self, img_batch, **kwargs)
        return backbone_out

    # The original loop has already run by the time orig returns and
    # failed. So we need to replace the *entire* forward_image body to
    # skip the bad loop. Cleanest: rewrite it.

    def _replacement(self, img_batch, *, need_sam3_out=False,
                    need_interactive_out=False,
                    need_propagation_out=False):
        if self.share_necks:
            need_propagation_out = need_interactive_out or need_propagation_out
            need_interactive_out = False
            backbone_out = self.backbone.forward_image(
                img_batch,
                need_sam3_out=need_sam3_out,
                need_sam2_out=need_propagation_out,
            )
            backbone_out["interactive"] = backbone_out["sam2_backbone_out"]
        else:
            backbone_out = self.backbone.forward_image(
                img_batch,
                need_sam3_out=need_sam3_out,
                need_interactive_out=need_interactive_out,
                need_propagation_out=need_propagation_out,
            )
        if self.use_high_res_features_in_sam:
            if need_interactive_out:
                backbone_out["interactive"]["backbone_fpn"][0].tensors = (
                    self.interactive_sam_mask_decoder.conv_s0(
                        backbone_out["interactive"]["backbone_fpn"][0].tensors))
                backbone_out["interactive"]["backbone_fpn"][1].tensors = (
                    self.interactive_sam_mask_decoder.conv_s1(
                        backbone_out["interactive"]["backbone_fpn"][1].tensors))
            if need_propagation_out:
                backbone_out["sam2_backbone_out"]["backbone_fpn"][0].tensors = (
                    self.sam_mask_decoder.conv_s0(
                        backbone_out["sam2_backbone_out"]["backbone_fpn"][0].tensors))
                backbone_out["sam2_backbone_out"]["backbone_fpn"][1].tensors = (
                    self.sam_mask_decoder.conv_s1(
                        backbone_out["sam2_backbone_out"]["backbone_fpn"][1].tensors))
        # Clone to help torch.compile — filter to dict entries only.
        for out_type, sub in list(backbone_out.items()):
            if not isinstance(sub, dict):
                continue
            if "backbone_fpn" not in sub or "vision_pos_enc" not in sub:
                continue
            for i in range(len(sub["backbone_fpn"])):
                sub["backbone_fpn"][i].tensors = self._maybe_clone(
                    sub["backbone_fpn"][i].tensors)
                sub["vision_pos_enc"][i] = self._maybe_clone(
                    sub["vision_pos_enc"][i])
        return backbone_out

    VideoTrackingMultiplex.forward_image = _replacement


def _patch_load_video_frames():
    """Upstream VideoTrackingMultiplexDemo.init_state passes
    use_torchcodec / use_cv2 kwargs to load_video_frames, but the
    packaged load_video_frames doesn't accept them. Wrap it to swallow
    those kwargs."""
    from sam3.model.utils import sam2_utils as _u
    orig = _u.load_video_frames

    def _wrapped(*a, **kw):
        kw.pop("use_torchcodec", None)
        kw.pop("use_cv2", None)
        return orig(*a, **kw)

    _u.load_video_frames = _wrapped
    # init_state may already have captured the symbol at import time;
    # also patch the demo module's namespace.
    import sam3.model.video_tracking_multiplex_demo as _demo
    if hasattr(_demo, "load_video_frames"):
        _demo.load_video_frames = _wrapped


def _init_state_sam3_1(model, video_path, offload_video_to_cpu=True):
    """Invoke the VideoTrackingMultiplexDemo base-class init_state that
    accepts a video_path. The Sam3VideoTrackingMultiplexDemo subclass
    overrides init_state with a different signature
    (video_height/video_width/num_frames, pre-loaded); we bypass that
    override by calling the base unbound method.

    Also forces inference_state["device"] / ["storage_device"] to CPU —
    the upstream init_state unconditionally sets them to
    torch.device("cuda") via a direct constructor (not a tensor-factory
    call), which our CUDA redirect doesn't catch, and downstream
    add_new_masks / propagate call .to(state["device"]) which would
    then trigger torch.cuda._lazy_init.
    """
    _patch_load_video_frames()
    _patch_forward_image_clone_loop()
    from sam3.model.video_tracking_multiplex_demo import (
        VideoTrackingMultiplexDemo,
    )
    state = VideoTrackingMultiplexDemo.init_state(
        model,
        video_path=video_path,
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=True,
        async_loading_frames=False,
        use_cv2=True,
    )
    if not torch.cuda.is_available():
        state["device"] = torch.device("cpu")
        state["storage_device"] = torch.device("cpu")
    return state


def main_sam3_1(args):
    if not args.seed_mask:
        print("ERROR: --variant sam3.1 requires --seed-mask",
              file=sys.stderr)
        return 2

    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)

    prompts = {
        "obj_1": {"frame": 0, "points": [[0.5, 0.5]], "labels": [1]},
    }
    with open(os.path.join(args.out, "prompts.json"), "w") as f:
        json.dump(prompts, f, indent=2)

    seed = np.array(Image.open(args.seed_mask).convert("L"))
    seed_bool = (seed > 127).astype(np.float32)
    # Shape [num_objects=1, H, W] — add_new_masks takes 3D tensor.
    seed_t = torch.from_numpy(seed_bool)[None, ...]

    model = _build_sam3_1_demo_model(os.environ["SAM3_CKPT"])
    state = _init_state_sam3_1(model, args.video)

    # Seed object 1 on frame 0 with the C-produced mask.
    model.add_new_masks(
        inference_state=state,
        frame_idx=0,
        obj_ids=[1],
        masks=seed_t,
    )

    # The C tracker emits masks at grid_w*4 x grid_h*4 resolution
    # (image_size=1008 / patch=14 -> grid=72 -> mask=288x288). The Python
    # video_res output is at the native video resolution (e.g. 720x1280
    # for kids.mp4), so we must resample to the same 288x288 grid before
    # saving to keep the C-side IoU compare well-defined. Nearest-neighbor
    # keeps the binary semantics intact.
    C_MASK_SIZE = 288

    count = 0
    for frame_idx, obj_ids, _low_res, video_res_masks, _obj_scores in \
            model.propagate_in_video(
                inference_state=state,
                start_frame_idx=0,
                max_frame_num_to_track=args.frames + 1,
                reverse=False):
        if frame_idx == 0:
            # Seed frame -- the C test sanity-checks its own frame-0
            # mask against seed_mask.png directly.
            continue
        for obj_id, m in zip(obj_ids, video_res_masks):
            binary = (m > 0).float()
            if binary.dim() == 2:
                binary = binary[None, None]
            elif binary.dim() == 3:
                binary = binary[None]
            resized = torch.nn.functional.interpolate(
                binary, size=(C_MASK_SIZE, C_MASK_SIZE),
                mode="nearest")
            arr = (resized.squeeze().cpu().numpy() > 0.5).astype(
                np.uint8) * 255
            Image.fromarray(arr).save(os.path.join(
                args.out, "frames",
                f"frame_{frame_idx:04d}_obj_{obj_id}.png"))
        count += 1
        if count >= args.frames:
            break

    print(f"wrote {count} frames to {args.out}/frames/",
          file=sys.stderr)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["sam3", "sam3.1"],
                    required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--seed-mask", default=None,
                    help="grayscale PNG seed (SAM 3.1 only)")
    args = ap.parse_args()

    if args.frames is None:
        args.frames = 30 if args.variant == "sam3" else 3

    if args.variant == "sam3":
        return main_sam3(args) or 0
    return main_sam3_1(args) or 0


if __name__ == "__main__":
    sys.exit(main())
