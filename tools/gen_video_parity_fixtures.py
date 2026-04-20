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
    """
    from sam3.model_builder import build_sam3_multiplex_video_model

    # Build the multiplex tracker model (returns Sam3VideoTrackingMultiplexDemo).
    model = build_sam3_multiplex_video_model(
        checkpoint_path=checkpoint,
        load_from_HF=False,
        multiplex_count=16,
        use_fa3=False,
        use_rope_real=False,
        strict_state_dict_loading=False,
        device="cpu",
        compile=False,
    )
    install_addmm_act_fp32()
    model = model.float().eval()
    return model


def _init_state_sam3_1(model, video_path, offload_video_to_cpu=True):
    """Invoke the VideoTrackingMultiplexDemo base-class init_state that
    accepts a video_path. The Sam3VideoTrackingMultiplexDemo subclass
    overrides init_state with a different signature
    (video_height/video_width/num_frames, pre-loaded); we bypass that
    override by calling the base unbound method.
    """
    from sam3.model.video_tracking_multiplex_demo import (
        VideoTrackingMultiplexDemo,
    )
    return VideoTrackingMultiplexDemo.init_state(
        model,
        video_path=video_path,
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=True,
        async_loading_frames=False,
        use_cv2=True,
    )


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
            arr = (m > 0).cpu().numpy().astype(np.uint8) * 255
            Image.fromarray(arr.squeeze()).save(os.path.join(
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
