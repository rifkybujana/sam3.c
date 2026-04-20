#!/usr/bin/env python3
"""
Dump Python-reference intermediate tensors for the two propagation
frames of kids.mp4 (seeded with a C-produced seed_mask) so we can
compare layer-by-layer against our C engine's /tmp/dbg_trk_*.bin files.

Only `_f1.bin` and `_f2.bin` are produced: the seed frame (frame 0)
does not invoke forward hooks on memory_attention or mask_decoder
(its outputs are cached during preflight and re-yielded by
propagate_in_video without re-running the forward), so there is no
`_f0.bin`.

Writes f32 raw-binary files matching the C dump format. Image-shaped
4-D tensors (NCHW) are permuted to NHWC before writing; token-shaped
tensors (e.g. sam_tokens_out [B, M, K, C]) are written verbatim with
any leading singleton batch dim squeezed so the byte layout matches
the C side's 3-D [M, K, C] dump.

  /tmp/py_trk_memattn_out_fN.bin       [5184, 256]
  /tmp/py_trk_mask_dec_masks_fN.bin    NHWC of [16, 3, 288, 288]
  /tmp/py_trk_mask_dec_iou_fN.bin      [16, 3]
  /tmp/py_trk_mask_dec_score_fN.bin    [16]
  /tmp/py_trk_mask_dec_sam_fN.bin      [16, 3, 256]

Usage:
  SAM3_CKPT=models/sam3.1_multiplex.pt \\
    python scripts/dump_tracker_layers.py \\
      --video assets/kids.mp4 \\
      --seed-mask /tmp/seed_lvl0.png \\
      --frames 2
"""
import argparse
import os
import sys
import numpy as np

# Upstream sam3 package lives under reference/
sys.path.insert(
    0,
    os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "reference", "sam3")),
)
# Share the CPU patches with gen_video_parity_fixtures.py.
sys.path.insert(
    0,
    os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "tools")),
)
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

def _dump(path, t, nhwc=False):
    """Write a torch.Tensor (any shape) as f32 raw binary.

    When `nhwc=True`, 4-D tensors are assumed to be in NCHW layout and
    are permuted to NHWC so the C engine's NHWC dumps align pixel-for-
    pixel. This is only correct for image-shaped tensors; token-shaped
    4-D tensors (e.g. sam_tokens_out [B, M, K, C]) must be dumped with
    `nhwc=False` (the default) to preserve their layout.

    When `nhwc=False`, a leading singleton batch dim is squeezed so
    e.g. [1, 16, 3, 256] becomes [16, 3, 256] to match the C side's
    3-D tensor dumps.
    """
    x = t.detach().cpu().float().contiguous()
    if nhwc and x.dim() == 4:
        x = x.permute(0, 2, 3, 1).contiguous()
    elif not nhwc and x.dim() >= 2 and x.size(0) == 1:
        x = x.squeeze(0).contiguous()
    arr = x.numpy().astype(np.float32)
    arr.tofile(path)
    print(f"dump: {path} shape={tuple(arr.shape)} dtype={arr.dtype}",
          file=sys.stderr)


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

def _build_model(checkpoint):
    from sam3.model_builder import build_sam3_multiplex_video_model
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
    merged = {}
    for k, v in ckpt.items():
        if k.startswith("tracker.model."):
            merged[k[len("tracker.model."):]] = v
    for k, v in ckpt.items():
        if k.startswith("detector.backbone."):
            merged[k[len("detector."):]] = v
    model.load_state_dict(merged, strict=False)
    return model.float().eval()


def _patches():
    """Same shims gen_video_parity_fixtures.py applies."""
    import gen_video_parity_fixtures as _gvp  # type: ignore
    _gvp._patch_load_video_frames()
    _gvp._patch_forward_image_clone_loop()


def _init_state(model, video_path):
    from sam3.model.video_tracking_multiplex_demo import (
        VideoTrackingMultiplexDemo,
    )
    state = VideoTrackingMultiplexDemo.init_state(
        model,
        video_path=video_path,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=False,
        use_cv2=True,
    )
    if not torch.cuda.is_available():
        state["device"] = torch.device("cpu")
        state["storage_device"] = torch.device("cpu")
    return state


def _register_hooks(model, captures):
    """Register forward_hooks on memory-attention + mask decoder.

    `captures` is a dict keyed by slot name; each value is a list the
    hook appends to. After each frame we pop the last appended and
    write it to /tmp/py_trk_<slot>_f<frame>.bin.

    In SAM 3.1 multiplex the memory-attention transformer lives at
    `model.transformer.encoder` (a TransformerEncoderDecoupledCross-
    Attention that returns {"memory": ...}). The mask head lives at
    `model.sam_mask_decoder` (a MultiplexMaskDecoder that returns a
    dict with keys: masks / iou_pred / sam_tokens_out /
    object_score_logits).
    """
    hooks = []

    def _cap(name):
        def _h(_m, _inp, out):
            captures.setdefault(name, []).append(out)
        return _h

    hooks.append(model.transformer.encoder.register_forward_hook(
        _cap("memattn_out")))
    hooks.append(model.sam_mask_decoder.register_forward_hook(
        _cap("mask_decoder_tuple")))
    return hooks


def _dump_mem(raw, frame_idx):
    """Dump a memattn_out capture (dict or tensor) for this frame."""
    if isinstance(raw, dict):
        t = raw.get("memory")
    else:
        t = raw
    if isinstance(t, torch.Tensor):
        _dump(f"/tmp/py_trk_memattn_out_f{frame_idx}.bin", t)


def _dump_mask_decoder(raw, frame_idx):
    """Dump a sam_mask_decoder capture for this frame.

    MultiplexMaskDecoder returns a dict with keys:
      masks / iou_pred / sam_tokens_out / object_score_logits
    Only `masks` is image-shaped (NCHW -> NHWC); the rest are
    token/score tensors and are written verbatim (with the leading
    singleton batch dim squeezed) so the byte layout matches the C
    side's dumps.
    """
    if not isinstance(raw, dict):
        return
    # image-shaped tensors get NHWC'd; everything else stays as-is.
    nhwc_keys = {"masks"}
    named = {
        "masks": "mask_dec_masks",
        "iou_pred": "mask_dec_iou",
        "sam_tokens_out": "mask_dec_sam",
        "object_score_logits": "mask_dec_score",
    }
    for key, slot in named.items():
        t = raw.get(key)
        if isinstance(t, torch.Tensor):
            _dump(f"/tmp/py_trk_{slot}_f{frame_idx}.bin", t,
                  nhwc=(key in nhwc_keys))


def _flush_captures_delta(captures, cursors, frame_idx):
    """Dump captures that arrived for the current propagate-yield.

    `cursors` tracks the last-consumed index per slot. Any new entries
    (index >= cursors[slot]) belong to the just-finished frame. This
    preserves frame-index numbering regardless of which path each
    frame took inside the upstream forward.
    """
    slot = "memattn_out"
    start = cursors.get(slot, 0)
    end = len(captures.get(slot, []))
    if end > start:
        # Multiple calls in one frame is unusual; dump the last one.
        _dump_mem(captures[slot][end - 1], frame_idx)
    cursors[slot] = end

    slot = "mask_decoder_tuple"
    start = cursors.get(slot, 0)
    end = len(captures.get(slot, []))
    if end > start:
        _dump_mask_decoder(captures[slot][end - 1], frame_idx)
    cursors[slot] = end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--seed-mask", required=True,
                    help="C-produced seed PNG (frame-0 mask)")
    ap.add_argument("--frames", type=int, default=2,
                    help="number of propagation frames to capture")
    args = ap.parse_args()

    _patches()
    model = _build_model(os.environ["SAM3_CKPT"])
    state = _init_state(model, args.video)

    seed = np.array(Image.open(args.seed_mask).convert("L"))
    seed_bool = (seed > 127).astype(np.float32)
    seed_t = torch.from_numpy(seed_bool)[None, ...]

    captures = {}
    cursors = {}
    hooks = _register_hooks(model, captures)
    try:
        # add_new_masks does NOT run the mask decoder; it just stores
        # the mask in the inference state.
        model.add_new_masks(
            inference_state=state,
            frame_idx=0,
            obj_ids=[1],
            masks=seed_t,
        )

        # Preflight consolidates per-object temp outputs into
        # output_dict["cond_frame_outputs"]. This is where the seed
        # frame (frame 0) actually forwards through the mask decoder
        # and memory encoder; propagate_in_video reuses those outputs
        # later without re-running them.
        model.propagate_in_video_preflight(state)
        _flush_captures_delta(captures, cursors, 0)

        # propagate_in_video yields (frame_idx, obj_ids, low_res, vid, score)
        # in ascending frame order starting at start_frame_idx=0. Frame 0
        # is re-yielded from the cond_frame_outputs cache (no new forward);
        # frames 1..N actually run the full forward including memory-attn.
        count = 0
        for frame_idx, obj_ids, _low, _vid, _score in \
                model.propagate_in_video(
                    inference_state=state,
                    start_frame_idx=0,
                    max_frame_num_to_track=args.frames + 1,
                    reverse=False):
            _flush_captures_delta(captures, cursors, frame_idx)
            count += 1
            if count >= args.frames + 1:
                break
    finally:
        for h in hooks:
            h.remove()

    print("done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
