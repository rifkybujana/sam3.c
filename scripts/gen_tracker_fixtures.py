#!/usr/bin/env python3
"""
scripts/gen_tracker_fixtures.py - Generate per-component test fixtures for SAM3 video tracker.

Two clip modes are supported:

  --clip-type real   (default)
      Loads the real SAM3 video model checkpoint, creates a synthetic 3-frame
      video (solid red, green, blue frames at 1008x1008), runs frame 0 with
      a center point prompt, propagates to frames 1 and 2, and saves every
      intermediate tensor at component boundaries as SafeTensors files.

  --clip-type square
      No model required. Synthesises an 8-frame clip of a 32x32 white square
      sliding diagonally 8 px/frame across a noisy gray 1008x1008 background,
      writes the frames as PNGs, and emits ground_truth.json with per-frame
      square centroids. Used by the video tracker integration test (Task 15)
      to validate end-to-end motion tracking.

This is the video-tracking counterpart to tools/gen_fixtures.py (which generates
fixtures for the image model). For --clip-type real the fixture directory
structure mirrors the tracker pipeline stages: backbone, prompt encoder, memory
attention, mask decoder, memory encoder, object pointer, integration, and
memory bank.

Usage:
    cd /path/to/sam3

    # Generate real-model tracker fixtures (default):
    python scripts/gen_tracker_fixtures.py \
        --checkpoint models/sam3.pt \
        --output tests/fixtures/tracker

    # Generate the synthetic moving-square clip (no checkpoint needed):
    python scripts/gen_tracker_fixtures.py \
        --clip-type square \
        --output tests/fixtures/tracker/moving_square

Requires: torch, safetensors, torchvision  (for --clip-type real)
          numpy, Pillow                     (for --clip-type square)
          Plus the reference sam3 package (add reference/sam3/ to PYTHONPATH)
          when using --clip-type real.
"""

import argparse
import contextlib
import json
import os
import sys
import types
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# Stub out triton (not available on macOS, only needed for EDT kernel in
# tracker code which we don't use for CPU fixture generation)
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
# Add reference/sam3/ to path so we can import the upstream sam3 package
# when --clip-type real needs it. Imports themselves are deferred to main()
# so --clip-type square works in environments without the reference package.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "reference" / "sam3"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _force_cpu_device():
    """Redirect any CUDA device references to CPU during model construction.

    The reference SAM3 code hardcodes device='cuda' in several __init__ methods
    (PositionEmbeddingSine, TransformerDecoder, etc.). This context manager
    patches torch tensor-creation functions to silently replace 'cuda' with
    'cpu'.
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

    def clear_captures(self):
        """Clear all captured tensors without removing hooks."""
        self.captures.clear()


# ---------------------------------------------------------------------------
# Synthetic video creation
# ---------------------------------------------------------------------------

# Moving-square clip parameters. These constants define the ground truth for
# the integration test in Task 15 and MUST stay in sync with the expected
# centroid trajectory consumed by that test.
SQUARE_NUM_FRAMES = 8
SQUARE_SIZE = 32
SQUARE_START_X = 100
SQUARE_START_Y = 100
SQUARE_STEP = 8


def synthesize_moving_square(output_dir, image_size=1008, seed=0):
    """Write an 8-frame clip of a moving 32x32 white square on noisy gray.

    Frame i (0-indexed) places the top-left corner of the square at
    (SQUARE_START_X + i * SQUARE_STEP, SQUARE_START_Y + i * SQUARE_STEP),
    so the square slides diagonally at SQUARE_STEP px/frame while staying
    fully inside the 1008x1008 frame for all 8 frames.

    The background is uniform noise drawn with numpy.random.default_rng(seed)
    in the range [100, 156) (roughly mean 128, +-28), providing enough visual
    texture to exercise a real tracker without swamping the square.

    Outputs (all written under output_dir):
      frame_0000.png ... frame_0007.png   (RGB 1008x1008 PNGs)
      ground_truth.json                   (per-frame square centroids)

    No model checkpoint is required: this path is intentionally standalone so
    the integration test in Task 15 can regenerate fixtures without weights.
    """
    # Lazy imports so --clip-type real does not require numpy/Pillow.
    import numpy as np
    from PIL import Image

    _ensure_dir(output_dir)

    rng = np.random.default_rng(seed)
    gt_frames = []

    for i in range(SQUARE_NUM_FRAMES):
        # Noisy gray background, fresh per frame so the tracker cannot cheat
        # by matching a static backdrop.
        frame = rng.integers(
            100, 156, size=(image_size, image_size, 3), dtype=np.uint8
        )

        x0 = SQUARE_START_X + i * SQUARE_STEP
        y0 = SQUARE_START_Y + i * SQUARE_STEP
        x1 = x0 + SQUARE_SIZE
        y1 = y0 + SQUARE_SIZE

        # Sanity: assert the square is fully inside the frame.
        assert 0 <= x0 and x1 <= image_size, (
            f"square x out of bounds on frame {i}: [{x0}, {x1}) vs {image_size}"
        )
        assert 0 <= y0 and y1 <= image_size, (
            f"square y out of bounds on frame {i}: [{y0}, {y1}) vs {image_size}"
        )

        frame[y0:y1, x0:x1, :] = 255

        out_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        Image.fromarray(frame, mode="RGB").save(out_path)

        gt_frames.append({
            "frame": i,
            "x0": int(x0),
            "y0": int(y0),
            "x1": int(x1),
            "y1": int(y1),
            "cx": int(x0 + SQUARE_SIZE / 2),
            "cy": int(y0 + SQUARE_SIZE / 2),
        })

        print(
            f"  frame_{i:04d}.png  "
            f"square=[{x0},{y0},{x1},{y1}]  "
            f"centroid=({gt_frames[-1]['cx']},{gt_frames[-1]['cy']})"
        )

    gt = {
        "clip_type": "square",
        "num_frames": SQUARE_NUM_FRAMES,
        "image_size": image_size,
        "square_size": SQUARE_SIZE,
        "start_x": SQUARE_START_X,
        "start_y": SQUARE_START_Y,
        "step_px_per_frame": SQUARE_STEP,
        "rng_seed": seed,
        "background": "uniform noise in [100, 156) per channel, per frame",
        "foreground": "32x32 white (255,255,255) square, sliding diagonally",
        "frames": gt_frames,
    }
    gt_path = os.path.join(output_dir, "ground_truth.json")
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2)
    print(f"  {gt_path}")


def create_synthetic_frames(num_frames=3, image_size=1008):
    """Create synthetic solid-color frames (red, green, blue).

    Returns a list of [1, 3, H, W] float32 tensors normalized to [-1, 1]
    (matching SAM3 preprocessing with mean=0.5, std=0.5).
    """
    colors = [
        [1.0, -1.0, -1.0],   # red:   R=1 (raw 255), G=-1 (raw 0), B=-1 (raw 0)
        [-1.0, 1.0, -1.0],   # green: R=-1 (raw 0), G=1 (raw 255), B=-1 (raw 0)
        [-1.0, -1.0, 1.0],   # blue:  R=-1 (raw 0), G=-1 (raw 0), B=1 (raw 255)
    ]
    frames = []
    for i in range(num_frames):
        color = colors[i % len(colors)]
        frame = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
        for c in range(3):
            frame[0, c, :, :] = color[c]
        frames.append(frame)
    return frames


# ---------------------------------------------------------------------------
# Register hooks on tracker-specific modules
# ---------------------------------------------------------------------------

def register_tracker_hooks(tracker, cap, prefix=""):
    """Register forward hooks on the SAM3 tracker's key components.

    The tracker (Sam3TrackerBase / Sam3TrackerPredictor) has these submodules:
      tracker.sam_prompt_encoder   -> SAM-style prompt encoder
      tracker.sam_mask_decoder     -> SAM-style mask decoder
      tracker.maskmem_backbone     -> memory encoder (mask + features -> memory)
      tracker.transformer.encoder  -> memory attention (cross-attention with memory)
    """
    pfx = f"{prefix}/" if prefix else ""

    # Prompt encoder
    if hasattr(tracker, "sam_prompt_encoder"):
        cap.register(
            tracker.sam_prompt_encoder,
            f"{pfx}prompt_encoder",
            capture_input=True,
            capture_output=True,
            output_names=["sparse_embeddings", "dense_embeddings"],
        )

    # Memory attention (encoder-only transformer)
    if hasattr(tracker, "transformer") and hasattr(tracker.transformer, "encoder"):
        encoder = tracker.transformer.encoder
        if hasattr(encoder, "layers"):
            for i, layer in enumerate(encoder.layers):
                cap.register(
                    layer,
                    f"{pfx}memory_attention/layer_{i:02d}",
                    capture_input=True,
                    capture_output=True,
                    input_names=["tgt"],
                    output_names=["output"],
                )
        # Also capture the full encoder output
        cap.register(
            encoder,
            f"{pfx}memory_attention",
            capture_output=True,
            output_names=["output"],
        )

    # SAM mask decoder
    if hasattr(tracker, "sam_mask_decoder"):
        cap.register(
            tracker.sam_mask_decoder,
            f"{pfx}mask_decoder",
            capture_input=True,
            capture_output=True,
        )

    # Memory encoder (maskmem_backbone)
    if hasattr(tracker, "maskmem_backbone"):
        cap.register(
            tracker.maskmem_backbone,
            f"{pfx}memory_encoder",
            capture_input=True,
            capture_output=True,
        )


def register_backbone_hooks(model, cap):
    """Register hooks on the vision backbone (shared detector backbone).

    The video model has: model.detector.backbone.vision_backbone.trunk
    """
    try:
        vit = model.detector.backbone.vision_backbone.trunk
        cap.register(vit.patch_embed, "backbone/patch_embed",
                     capture_input=True, capture_output=True,
                     input_names=["pixels"], output_names=["patches"])
        for i, blk in enumerate(vit.blocks):
            cap.register(blk, f"backbone/block_{i:02d}",
                         capture_output=True, output_names=["output"])
    except AttributeError:
        print("  [warn] Could not register backbone hooks (structure mismatch)")

    # Neck (FPN)
    try:
        neck = model.detector.backbone.vision_backbone
        for i, conv in enumerate(neck.convs):
            scale_name = {0: "4x", 1: "2x", 2: "1x", 3: "05x"}.get(i, f"s{i}")
            cap.register(conv, f"backbone/neck_scale_{scale_name}",
                         capture_output=True, output_names=["features"])
    except AttributeError:
        print("  [warn] Could not register neck hooks (structure mismatch)")


# ---------------------------------------------------------------------------
# Save captured tensors per stage
# ---------------------------------------------------------------------------

def save_per_frame_captures(cap, frame_idx, output_dir, capture_prefix,
                            stage_num, stage_label):
    """Save all captures matching a prefix for a given frame.

    Args:
        cap: TensorCapture instance
        frame_idx: Frame index for the output filename
        output_dir: Root output directory
        capture_prefix: Prefix to match in cap.captures keys
        stage_num: Numeric prefix for the stage directory (e.g. 2)
        stage_label: Human-readable stage label (e.g. "prompt_encoder")
    """
    stage_dir = f"{output_dir}/{stage_num:02d}_{stage_label}"
    relevant = {}
    for key, tensors in cap.captures.items():
        if key.startswith(capture_prefix) or key == capture_prefix:
            for tname, tval in tensors.items():
                flat_key = f"{key}/{tname}" if key != capture_prefix else tname
                # Clean up the key for SafeTensors (no slashes)
                flat_key = flat_key.replace("/", ".")
                relevant[flat_key] = tval
    if relevant:
        _save(f"{stage_dir}/frame_{frame_idx}.safetensors", relevant)
    return relevant


# ---------------------------------------------------------------------------
# Main tracking loop with fixture capture
# ---------------------------------------------------------------------------

def run_tracker_and_save(model, frames, point_prompt, output_dir, device,
                         image_size):
    """Run the tracker on synthetic frames, capture intermediates, save fixtures.

    This function drives the SAM3 video model through its tracking pipeline:
      1. Process each frame through the backbone
      2. On frame 0, apply point prompt -> prompt encoder -> mask decoder
      3. Encode frame 0's mask into memory
      4. On frames 1+, use memory attention to fuse current features with memory
      5. Run mask decoder to predict masks
      6. Encode new memory for each frame

    We use forward hooks to capture tensors at each stage boundary.
    """
    cap = TensorCapture()

    # Get the tracker submodule from the video model
    # The video model is Sam3VideoInferenceWithInstanceInteractivity which
    # has a .tracker attribute (Sam3TrackerPredictor)
    tracker = model.tracker
    register_tracker_hooks(tracker, cap, prefix="tracker")
    register_backbone_hooks(model, cap)

    print(f"\nRegistered hooks on tracker and backbone")

    # --- Step 0: Save input frames ---
    print("\n[0/7] Saving input frames...")
    frame_tensors = {}
    for i, f in enumerate(frames):
        frame_tensors[f"frame_{i}"] = f.squeeze(0)  # [3, H, W]
    _save(f"{output_dir}/00_input/frames.safetensors", frame_tensors)

    # --- Initialize tracker state ---
    # We need to create an inference state manually since we're working with
    # synthetic frames (not loading from a video file).
    # The Sam3TrackerPredictor.init_state() can accept pre-loaded frames.
    print("\n[1/7] Initializing tracker state...")

    # Stack frames into the format expected by the tracker:
    # images is a list of [1, 3, H, W] tensors on the target device
    images = [f.to(device) for f in frames]
    num_frames = len(images)

    # Initialize the tracker's inference state directly
    inference_state = tracker.init_state(
        video_height=image_size,
        video_width=image_size,
        num_frames=num_frames,
    )
    # Manually set the images in the inference state
    inference_state["images"] = images

    # --- Step 1: Run backbone on frame 0 ---
    print("\n[2/7] Running backbone + prompt on frame 0...")
    cap.clear_captures()

    px, py, plabel = point_prompt
    # Scale point from pixel coordinates to the range expected by the tracker.
    # add_new_points_or_box with rel_coordinates=False expects absolute pixel
    # coords in the image_size space.
    _, obj_ids, video_res_masks = tracker.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=[[px, py]],
        labels=[plabel],
        rel_coordinates=False,
        normalize_coords=False,
    )

    # Save backbone captures for frame 0
    print("\n  Saving backbone features for frame 0...")
    backbone_tensors = {}
    for key, tensors in cap.captures.items():
        if key.startswith("backbone/"):
            for tname, tval in tensors.items():
                flat_key = f"{key}.{tname}".replace("/", ".")
                backbone_tensors[flat_key] = tval
    if backbone_tensors:
        _save(f"{output_dir}/01_backbone/frame_0.safetensors", backbone_tensors)

    # Save prompt encoder captures for frame 0
    print("  Saving prompt encoder output for frame 0...")
    save_per_frame_captures(
        cap, 0, output_dir, "tracker/prompt_encoder", 2, "prompt_encoder"
    )

    # Save memory attention captures for frame 0 (no-memory case)
    print("  Saving memory attention output for frame 0 (no-memory)...")
    save_per_frame_captures(
        cap, 0, output_dir, "tracker/memory_attention", 3, "memory_attention"
    )

    # Save mask decoder captures for frame 0
    print("  Saving mask decoder output for frame 0...")
    save_per_frame_captures(
        cap, 0, output_dir, "tracker/mask_decoder", 4, "mask_decoder"
    )

    # Save memory encoder captures for frame 0
    print("  Saving memory encoder output for frame 0...")
    save_per_frame_captures(
        cap, 0, output_dir, "tracker/memory_encoder", 5, "memory_encoder"
    )

    # Save object pointer captures for frame 0
    # The obj_ptr is stored in the output_dict by track_step
    print("  Saving object pointer for frame 0...")
    output_dict = inference_state["output_dict"]
    obj_ptr_tensors = {}
    for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
        if 0 in output_dict.get(storage_key, {}):
            frame_out = output_dict[storage_key][0]
            if "obj_ptr" in frame_out and frame_out["obj_ptr"] is not None:
                obj_ptr_tensors["obj_ptr"] = frame_out["obj_ptr"]
            if "object_score_logits" in frame_out and frame_out["object_score_logits"] is not None:
                obj_ptr_tensors["object_score_logits"] = frame_out["object_score_logits"]
    if obj_ptr_tensors:
        _save(f"{output_dir}/06_obj_pointer/frame_0.safetensors", obj_ptr_tensors)

    # Save integration output (final masks) for frame 0
    print("  Saving integration masks for frame 0...")
    integration_tensors = {}
    for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
        if 0 in output_dict.get(storage_key, {}):
            frame_out = output_dict[storage_key][0]
            for k in ["pred_masks", "pred_masks_high_res"]:
                if k in frame_out and frame_out[k] is not None:
                    integration_tensors[k] = frame_out[k]
    if integration_tensors:
        _save(f"{output_dir}/07_integration/frame_0_masks.safetensors",
              integration_tensors)

    # Save memory bank state after frame 0
    print("  Saving memory bank state after frame 0...")
    membank_tensors = {}
    for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
        if 0 in output_dict.get(storage_key, {}):
            frame_out = output_dict[storage_key][0]
            if "maskmem_features" in frame_out and frame_out["maskmem_features"] is not None:
                membank_tensors["maskmem_features"] = frame_out["maskmem_features"]
            if "maskmem_pos_enc" in frame_out and frame_out["maskmem_pos_enc"] is not None:
                pos_enc = frame_out["maskmem_pos_enc"]
                if isinstance(pos_enc, list):
                    for j, p in enumerate(pos_enc):
                        if isinstance(p, torch.Tensor):
                            membank_tensors[f"maskmem_pos_enc_{j}"] = p
                elif isinstance(pos_enc, torch.Tensor):
                    membank_tensors["maskmem_pos_enc"] = pos_enc
    if membank_tensors:
        _save(f"{output_dir}/08_memory_bank/frame_0.safetensors", membank_tensors)

    # --- Steps 3-7: Propagate to frames 1 and 2 ---
    print("\n[3/7] Propagating to remaining frames...")

    for frame_idx, obj_ids_out, low_res_masks, video_res_masks_out, obj_scores in \
            tracker.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=0,
                max_frame_num_to_track=num_frames,
                reverse=False,
            ):
        if frame_idx == 0:
            # Frame 0 was already processed above; skip saving duplicates
            # but still clear captures for the next frame
            cap.clear_captures()
            continue

        print(f"\n  Processing frame {frame_idx}...")

        # Save backbone features for this frame
        backbone_tensors = {}
        for key, tensors in cap.captures.items():
            if key.startswith("backbone/"):
                for tname, tval in tensors.items():
                    flat_key = f"{key}.{tname}".replace("/", ".")
                    backbone_tensors[flat_key] = tval
        if backbone_tensors:
            _save(f"{output_dir}/01_backbone/frame_{frame_idx}.safetensors",
                  backbone_tensors)

        # Save memory attention for this frame (with memory)
        save_per_frame_captures(
            cap, frame_idx, output_dir,
            "tracker/memory_attention", 3, "memory_attention"
        )

        # Save mask decoder for this frame
        save_per_frame_captures(
            cap, frame_idx, output_dir,
            "tracker/mask_decoder", 4, "mask_decoder"
        )

        # Save memory encoder for this frame
        save_per_frame_captures(
            cap, frame_idx, output_dir,
            "tracker/memory_encoder", 5, "memory_encoder"
        )

        # Save object pointer
        obj_ptr_tensors = {}
        for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
            if frame_idx in output_dict.get(storage_key, {}):
                frame_out = output_dict[storage_key][frame_idx]
                if "obj_ptr" in frame_out and frame_out["obj_ptr"] is not None:
                    obj_ptr_tensors["obj_ptr"] = frame_out["obj_ptr"]
                if "object_score_logits" in frame_out and frame_out["object_score_logits"] is not None:
                    obj_ptr_tensors["object_score_logits"] = frame_out["object_score_logits"]
        if obj_ptr_tensors:
            _save(f"{output_dir}/06_obj_pointer/frame_{frame_idx}.safetensors",
                  obj_ptr_tensors)

        # Save integration masks
        integration_tensors = {}
        for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
            if frame_idx in output_dict.get(storage_key, {}):
                frame_out = output_dict[storage_key][frame_idx]
                for k in ["pred_masks", "pred_masks_high_res"]:
                    if k in frame_out and frame_out[k] is not None:
                        integration_tensors[k] = frame_out[k]
        if integration_tensors:
            _save(f"{output_dir}/07_integration/frame_{frame_idx}_masks.safetensors",
                  integration_tensors)

        # Save memory bank state
        membank_tensors = {}
        for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
            if frame_idx in output_dict.get(storage_key, {}):
                frame_out = output_dict[storage_key][frame_idx]
                if "maskmem_features" in frame_out and frame_out["maskmem_features"] is not None:
                    membank_tensors["maskmem_features"] = frame_out["maskmem_features"]
                if "maskmem_pos_enc" in frame_out and frame_out["maskmem_pos_enc"] is not None:
                    pos_enc = frame_out["maskmem_pos_enc"]
                    if isinstance(pos_enc, list):
                        for j, p in enumerate(pos_enc):
                            if isinstance(p, torch.Tensor):
                                membank_tensors[f"maskmem_pos_enc_{j}"] = p
                    elif isinstance(pos_enc, torch.Tensor):
                        membank_tensors["maskmem_pos_enc"] = pos_enc
        if membank_tensors:
            _save(f"{output_dir}/08_memory_bank/frame_{frame_idx}.safetensors",
                  membank_tensors)

        # Clear for next frame
        cap.clear_captures()

    cap.remove_all()

    # Print summary of captured data
    print(f"\n--- Capture summary ---")
    total_files = 0
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith(".safetensors"):
                total_files += 1
    print(f"Total .safetensors files saved: {total_files}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate per-component test fixtures for SAM3 video tracker")
    parser.add_argument("--clip-type", choices=["real", "square"], default="real",
                        help="Clip mode: 'real' drives the full SAM3 video model on "
                             "synthetic RGB frames (requires --checkpoint); 'square' "
                             "writes an 8-frame moving-square PNG clip plus "
                             "ground_truth.json (no checkpoint required).")
    parser.add_argument("--checkpoint", default="models/sam3.pt",
                        help="Path to sam3.pt checkpoint (required for --clip-type real)")
    parser.add_argument("--output", default="tests/fixtures/tracker",
                        help="Output directory for fixture files")
    parser.add_argument("--device", default="cpu",
                        help="Device for inference (cpu recommended for determinism)")
    parser.add_argument("--image-size", type=int, default=1008,
                        help="Input resolution (default: 1008)")
    args = parser.parse_args()

    # --- Short-circuit: synthetic moving-square clip ---
    # This path is intentionally independent of torch/sam3 weight loading so
    # the integration test fixtures can be regenerated without a checkpoint.
    if args.clip_type == "square":
        print("=" * 60)
        print("SAM3 Video Tracker Fixture Generator (moving-square clip)")
        print("=" * 60)
        print(f"Output:      {args.output}")
        print(f"Image size:  {args.image_size}")
        print(f"Num frames:  {SQUARE_NUM_FRAMES}")
        print(f"Square:      {SQUARE_SIZE}x{SQUARE_SIZE} white, "
              f"start=({SQUARE_START_X},{SQUARE_START_Y}), "
              f"step={SQUARE_STEP} px/frame diagonal")
        print()
        synthesize_moving_square(
            output_dir=args.output,
            image_size=args.image_size,
            seed=0,
        )
        print("\nDone!")
        return

    # Determinism
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    image_size = args.image_size
    # Center point prompt: (x=504, y=504, label=1 for positive)
    point_prompt = (image_size // 2, image_size // 2, 1)

    print("=" * 60)
    print("SAM3 Video Tracker Fixture Generator")
    print("=" * 60)
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Device:      {args.device}")
    print(f"Image size:  {image_size}")
    print(f"Point:       ({point_prompt[0]}, {point_prompt[1]}, label={point_prompt[2]})")
    print(f"Num frames:  3 (synthetic red/green/blue)")
    print(f"Output:      {args.output}")
    print()

    # --- Patch fused kernels for CPU compatibility ---
    # The reference code uses bf16 fused kernels that fail on CPU. Replace
    # with plain f32 linear+activation.
    try:
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
        try:
            import sam3.model.vitdet as _vitdet_mod
            _vitdet_mod.addmm_act = _addmm_act_f32
        except (ImportError, AttributeError):
            pass
    except (ImportError, AttributeError):
        pass

    # Patch pin_memory to be a no-op on CPU (upstream uses it for async CUDA
    # transfers; on macOS it tries MPS pinning which fails)
    _orig_pin_memory = torch.Tensor.pin_memory
    torch.Tensor.pin_memory = lambda self, *a, **kw: self

    # --- Build model ---
    # Import here (not at module load) so --clip-type square does not require
    # the reference sam3 package to be on PYTHONPATH.
    from sam3.model_builder import build_sam3_video_model  # noqa: E402

    print("Loading SAM3 video model...")
    with _force_cpu_device():
        model = build_sam3_video_model(
            checkpoint_path=args.checkpoint,
            load_from_HF=False,
            device=args.device,
            strict_state_dict_loading=True,
            apply_temporal_disambiguation=True,
            compile=False,
        )
    # Convert to float32 for deterministic CPU inference
    # (checkpoint may have bf16 weights from GPU training)
    model = model.float()
    model.eval()

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded ({total_params:.1f}M params)")

    # Extract the tracker submodule for direct use
    tracker = model.tracker
    tracker_params = sum(p.numel() for p in tracker.parameters()) / 1e6
    print(f"Tracker submodule: {tracker_params:.1f}M params")

    # --- Create synthetic frames ---
    print("\nCreating synthetic frames...")
    frames = create_synthetic_frames(num_frames=3, image_size=image_size)
    print(f"Created {len(frames)} frames of shape {list(frames[0].shape)}")

    # --- Run tracking and save fixtures ---
    run_tracker_and_save(
        model=model,
        frames=frames,
        point_prompt=point_prompt,
        output_dir=args.output,
        device=args.device,
        image_size=image_size,
    )

    # --- Save metadata ---
    meta = {
        "generator": "scripts/gen_tracker_fixtures.py",
        "model_type": "sam3_video",
        "checkpoint": args.checkpoint,
        "device": args.device,
        "image_size": image_size,
        "num_frames": 3,
        "frame_colors": ["red", "green", "blue"],
        "point_prompt": list(point_prompt),
        "point_prompt_description": "(x, y, label) center point, label=1 positive",
        "torch_version": torch.__version__,
        "torch_manual_seed": 0,
        "tracker_components": {
            "backbone": "ViT with FPN neck (shared detector backbone)",
            "prompt_encoder": "SAM-style prompt encoder (sam_prompt_encoder)",
            "memory_attention": "Encoder-only transformer (transformer.encoder)",
            "mask_decoder": "SAM-style mask decoder (sam_mask_decoder)",
            "memory_encoder": "Mask memory backbone (maskmem_backbone)",
            "obj_pointer": "Object pointer from mask decoder output token",
            "memory_bank": "Temporal memory features (maskmem_features + pos_enc)",
        },
        "fixture_layout": {
            "00_input": "Preprocessed input frames",
            "01_backbone": "Vision backbone (ViT + neck) features per frame",
            "02_prompt_encoder": "SAM prompt encoder output (frame 0 only)",
            "03_memory_attention": "Memory attention output per frame",
            "04_mask_decoder": "Mask decoder logits + IoU scores per frame",
            "05_memory_encoder": "Memory encoder output per frame",
            "06_obj_pointer": "Object pointer vectors per frame",
            "07_integration": "Final predicted masks per frame",
            "08_memory_bank": "Memory bank state (features + pos encoding) per frame",
        },
    }
    meta_path = os.path.join(args.output, "metadata.json")
    _ensure_dir(args.output)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  {meta_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
