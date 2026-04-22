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

    # Encoder hook captures BOTH the output (for memattn_out) and the
    # inputs (for tgt / memory / memory_image / memory_image_pos bank
    # comparisons in Task 5γ). Upstream call-site at
    # video_tracking_multiplex.py:1590 uses kwargs, so we register with
    # with_kwargs=True to receive them in the kwargs dict.
    def _cap_encoder(_m, _inp, kwargs, out):
        captures.setdefault("memattn_out", []).append(out)
        captures.setdefault("memattn_inputs", []).append(dict(kwargs))
    hooks.append(model.transformer.encoder.register_forward_hook(
        _cap_encoder, with_kwargs=True))

    # Capture _encode_new_memory's pred_masks_high_res to verify C's
    # resolution-matching vs Python's multi-hop path on cond frames.
    # Capture the final video_res output yielded by propagate_in_video
    # — this is what gen_video_parity_fixtures.py writes to the ref
    # fixture PNG. The raw `_forward_sam_heads` output gets gated to
    # NO_OBJ_SCORE on multiplex propagation frames, but somewhere
    # between that and the yielded mask the content is restored (parity
    # log iter 15 — exact path not yet traced; empirically the C side
    # skipping apply_occlusion_gating on track_masks matches Python at
    # IoU ~0.47).
    _orig_gvr = model._get_orig_video_res_output
    def _patched_gvr(inference_state, any_res_masks):
        any_r, vrm = _orig_gvr(inference_state, any_res_masks)
        captures.setdefault("gvr_output", []).append(
            vrm.detach().clone().cpu().float().contiguous())
        return any_r, vrm
    model._get_orig_video_res_output = _patched_gvr

    class _UnpatchGVR:
        def remove(self_inner):
            model._get_orig_video_res_output = _orig_gvr
    hooks.append(_UnpatchGVR())

    # Capture _forward_sam_heads output to see gating effect

    _orig_enm = model._encode_new_memory
    def _patched_enm(image, current_vision_feats, feat_sizes,
                     pred_masks_high_res, object_score_logits,
                     is_mask_from_pts, *, conditioning_objects=None,
                     multiplex_state=None):
        captures.setdefault("enm_pred_masks_high_res", []).append(
            pred_masks_high_res.detach().cpu().float().contiguous())
        return _orig_enm(
            image=image,
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=pred_masks_high_res,
            object_score_logits=object_score_logits,
            is_mask_from_pts=is_mask_from_pts,
            conditioning_objects=conditioning_objects,
            multiplex_state=multiplex_state,
        )
    model._encode_new_memory = _patched_enm

    class _UnpatchENM:
        def remove(self_inner):
            model._encode_new_memory = _orig_enm
    hooks.append(_UnpatchENM())

    # Iteration 15: Maskmem bisection. Capture:
    #   - maskmem_out: final maskmem_backbone output (pre no_obj_embed)
    #   - maskmem_proj: post-downsampler (pre pix_proj add)
    #   - maskmem_pix_proj: pix_feat_proj output (for add bisection)
    #   - maskmem_stage{0..3}_conv/act: per-stage Conv and post-GELU out
    # These let us pinpoint which sub-op of maskmem amplifies the ~2%
    # feat_s1 input gap into the observed 10% cos drop at memory[:5184].
    def _cap_maskmem_out(_m, _inp, out):
        # SimpleMaskEncoder returns dict {"vision_features", "vision_pos_enc"}
        if isinstance(out, dict) and "vision_features" in out:
            captures.setdefault("maskmem_out", []).append(
                out["vision_features"].detach().cpu().float().contiguous())
    hooks.append(model.maskmem_backbone.register_forward_hook(_cap_maskmem_out))

    # Downsampler output: feeds `x + masks` add in SimpleMaskEncoder.forward.
    def _cap_maskmem_proj(_m, _inp, out):
        if isinstance(out, torch.Tensor):
            captures.setdefault("maskmem_proj", []).append(
                out.detach().cpu().float().contiguous())
    hooks.append(model.maskmem_backbone.mask_downsampler.register_forward_hook(
        _cap_maskmem_proj))

    # pix_feat_proj (1x1 conv on feat_s1) output.
    def _cap_maskmem_pix(_m, _inp, out):
        if isinstance(out, torch.Tensor):
            captures.setdefault("maskmem_pix_proj", []).append(
                out.detach().cpu().float().contiguous())
    hooks.append(model.maskmem_backbone.pix_feat_proj.register_forward_hook(
        _cap_maskmem_pix))

    # Per-stage intermediates inside mask_downsampler.encoder (Sequential).
    # Layout: [Conv, LN, GELU] × 4, then final 1x1 Conv.
    # Indices: 0,3,6,9 = Conv per stage; 2,5,8,11 = GELU per stage; 12 = final proj.
    _enc = model.maskmem_backbone.mask_downsampler.encoder
    # Capture post-bilinear input to encoder[0] (the Conv)
    def _cap_downsampler_in(_m, inp, out):
        if isinstance(inp, tuple) and len(inp) > 0 and isinstance(inp[0], torch.Tensor):
            captures.setdefault("maskmem_downsampler_in", []).append(
                inp[0].detach().cpu().float().contiguous())
    hooks.append(_enc[0].register_forward_pre_hook(
        lambda _m, inp: captures.setdefault(
            "maskmem_downsampler_in", []).append(
                inp[0].detach().cpu().float().contiguous())))
    for si in range(4):
        conv_idx = si * 3
        act_idx = si * 3 + 2
        def _cap_conv(_m, _inp, out, _si=si):
            if isinstance(out, torch.Tensor):
                captures.setdefault(f"maskmem_stage{_si}_conv", []).append(
                    out.detach().cpu().float().contiguous())
        def _cap_act(_m, _inp, out, _si=si):
            if isinstance(out, torch.Tensor):
                captures.setdefault(f"maskmem_stage{_si}_act", []).append(
                    out.detach().cpu().float().contiguous())
        hooks.append(_enc[conv_idx].register_forward_hook(_cap_conv))
        hooks.append(_enc[act_idx].register_forward_hook(_cap_act))
    hooks.append(model.sam_mask_decoder.register_forward_hook(
        _cap("mask_decoder_tuple")))

    # Per-layer hooks on the 4 decoupled transformer layers. These fire
    # independently of the encoder-level hook above; both capture.
    for i, layer in enumerate(model.transformer.encoder.layers):
        hooks.append(layer.register_forward_hook(
            _cap(f"memattn_layer{i}")))

    # Sub-drill (iteration 9): bisect layer-0 memattn amplification.
    # The memattn inputs (tgt, memory_image, memory, memory_image_pos)
    # match Python at cos ≈ 0.98, but memattn_layer0 drops to cos ≈ 0.004.
    # A ~2% input residual cannot legally explode to 100% output error
    # through one decoupled cross-attention layer — structural bug.
    # Patch layer[0]._forward_sa and ._forward_ca to capture the
    # intermediate tensors after each major sub-op so we can localize
    # whether self-attn, Q/K projections, or the attention body is the
    # culprit. _forward_ca's Q/K/V naming mirrors C's memory_attn_layer
    # (q post img+tgt sum, k post img+mem+pos, v post v_proj, attn post
    # cross_attn_out_proj pre-residual).
    _l0 = model.transformer.encoder.layers[0]
    _l0_orig_sa = _l0._forward_sa
    _l0_orig_ca = _l0._forward_ca

    # NOTE (iter 15): the _l0 sa/ca replacements below alter Python's
    # runtime behavior — propagation scores diverge from a clean run,
    # e.g. frame 1 obj_score is -0.20 under this script but +2.93 under
    # gen_video_parity_fixtures.py. Exact cause not yet isolated (likely
    # Python closure/state interaction with model.transformer's
    # activation_ckpt_wrapper). For parity drills where the *result*
    # tensors matter, keep this flag True; flip to False only when you
    # need the sa/ca intermediates (memattn_l0_{sa_out,ca_q,ca_k,ca_v,
    # ca_attn} slots). Don't diff fsh_ / mask_dec_ slots while patched.
    _DISABLE_L0_PATCH = True

    def _l0_patched_sa(tgt, query_pos):
        out = _l0_orig_sa(tgt, query_pos)
        captures.setdefault("memattn_l0_sa_out", []).append(
            out.detach().cpu().float().contiguous())
        return out

    def _l0_patched_ca(*, image, tgt, memory_image, memory, query_pos,
                       memory_image_pos, num_k_exclude_rope=0):
        # Replicate the body of _forward_ca so we can sample
        # intermediates. Keep in sync with
        # reference/sam3/sam3/model/decoder.py:_forward_ca.
        tgt2 = _l0.norm2(tgt)
        q = _l0.image_cross_attn_q_proj(image) + _l0.cross_attn_q_proj(tgt2)
        if _l0.pos_enc_at_cross_attn_queries:
            q = q + query_pos
        # Capture memory_image_pos AT CA ENTRY (post encoder transpose+pad)
        k_img = _l0.image_cross_attn_k_proj(memory_image)
        k_mem = _l0.cross_attn_k_proj(memory)
        captures.setdefault("memattn_l0_ca_k_img", []).append(
            k_img.detach().cpu().float().contiguous())
        captures.setdefault("memattn_l0_ca_k_mem", []).append(
            k_mem.detach().cpu().float().contiguous())
        k = k_img + k_mem
        if _l0.pos_enc_at_cross_attn_keys:
            k = k + memory_image_pos
        v = _l0.cross_attn_v_proj(memory)
        # Clone BEFORE cross_attention_rope to prevent in-place mutation
        # from corrupting our post-pos k capture (RoPE applies rotations
        # in-place on q/k via torch.view_as_complex).
        captures.setdefault("memattn_l0_ca_q", []).append(
            q.detach().clone().cpu().float().contiguous())
        captures.setdefault("memattn_l0_ca_k", []).append(
            k.detach().clone().cpu().float().contiguous())
        captures.setdefault("memattn_l0_ca_v", []).append(
            v.detach().clone().cpu().float().contiguous())
        kwds = {}
        if num_k_exclude_rope > 0:
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}
        out = _l0.cross_attention_rope(q, k, v, **kwds)
        tgt2 = _l0.cross_attn_out_proj(out)
        captures.setdefault("memattn_l0_ca_attn", []).append(
            tgt2.detach().cpu().float().contiguous())
        tgt = tgt + _l0.dropout2(tgt2)
        return tgt

    if not _DISABLE_L0_PATCH:
        _l0._forward_sa = _l0_patched_sa
        _l0._forward_ca = _l0_patched_ca

    class _UnpatchL0:
        def remove(self_inner):
            _l0._forward_sa = _l0_orig_sa
            _l0._forward_ca = _l0_orig_ca
    hooks.append(_UnpatchL0())

    # Sub-drill (iteration 4): capture image features pre-flatten by
    # monkey-patching `_prepare_memory_conditioned_features` on the
    # model instance. `current_vision_feats[-1]` is [HW, 1, C] (the
    # top-level feature map after backbone+neck, already in (HW)BC
    # layout but BEFORE the expand(-1, B, -1) broadcast to 16 buckets).
    # Byte-layout matches C's [1, H, W, C] NHWC image_embed since HW
    # is contiguous and C is innermost.
    _orig = model._prepare_memory_conditioned_features

    def _patched_prepare(*args, **kwargs):
        cvf = kwargs.get("current_vision_feats")
        if isinstance(cvf, (list, tuple)) and len(cvf) > 0 \
                and isinstance(cvf[-1], torch.Tensor):
            captures.setdefault("image_embed", []).append(cvf[-1])
        return _orig(*args, **kwargs)

    model._prepare_memory_conditioned_features = _patched_prepare

    class _UnpatchHandle:
        def remove(self_inner):
            model._prepare_memory_conditioned_features = _orig
    hooks.append(_UnpatchHandle())

    # Image-pipeline drill: capture backbone.forward_image input
    # (img_batch) + its sam2 1x output per call. forward_image is
    # invoked exactly once per frame in upstream order (frame 0 during
    # add_new_masks / preflight, then frames 1..N during
    # propagate_in_video), so the capture index is the frame index.
    _orig_fwd_image = model.forward_image

    def _patched_forward_image(img_batch, **kwargs):
        out = _orig_fwd_image(img_batch, **kwargs)
        # Capture img_batch verbatim (NCHW [1, 3, H, W] F32, [-1, 1]
        # after mean/std) — matches C's make_frame_tensor CHW F32.
        # img_batch may be a NestedTensor; unwrap via .tensors.
        img_t = img_batch
        if hasattr(img_t, "tensors"):
            img_t = img_t.tensors
        captures.setdefault("frame_rgb", []).append(
            img_t.detach().cpu().float().contiguous())
        # Capture the sam2 1x backbone+neck output for this frame —
        # corresponds to C's cached_sam2_1x_nhwc after the sam2 neck.
        sam2 = out.get("sam2_backbone_out") if isinstance(out, dict) \
            else None
        if isinstance(sam2, dict):
            fpn = sam2.get("backbone_fpn")
            if isinstance(fpn, list) and len(fpn) > 0:
                shapes = [tuple(x.tensors.shape) if hasattr(x, "tensors")
                          else None for x in fpn]
                print(f"[hook] sam2_backbone_fpn len={len(fpn)} "
                      f"shapes={shapes}", file=sys.stderr)
                if hasattr(fpn[-1], "tensors"):
                    captures.setdefault("feat_s1", []).append(
                        fpn[-1].tensors.detach().cpu().float().contiguous())
        # Sub-drill (iteration 8): capture the sam3-side 1x neck
        # output. `out["backbone_fpn"]` is `sam3_features` in the
        # tri-neck (see vl_combiner.py:243-252) — the detector-side
        # neck that shares the ViT input with propagation_convs but
        # uses a different conv stack. If vit_out matches and
        # sam3_feat_s1 matches but feat_s1 (sam2/propagation side)
        # doesn't, the propagation_convs compute itself is the bug.
        if isinstance(out, dict):
            sam3_fpn = out.get("backbone_fpn")
            if isinstance(sam3_fpn, list) and len(sam3_fpn) > 0 and \
                    hasattr(sam3_fpn[-1], "tensors"):
                captures.setdefault("sam3_feat_s1", []).append(
                    sam3_fpn[-1].tensors.detach().cpu().float()
                    .contiguous())
        return out

    model.forward_image = _patched_forward_image

    class _UnpatchFwdImage:
        def remove(self_inner):
            model.forward_image = _orig_fwd_image
    hooks.append(_UnpatchFwdImage())

    # Sub-drill (iteration 8): capture ViT trunk output pre-neck.
    # `Sam3TriViTDetNeck.forward` calls `xs = self.trunk(tensor_list)`
    # and then applies convs/interactive_convs/propagation_convs to
    # `xs[-1]`. Registering a forward_hook on the trunk gives us
    # direct access to that shared input — so we can bisect whether
    # the feat_s1 divergence originates in the backbone or in the
    # sam2-side neck's propagation_convs compute.
    def _cap_trunk(_m, _inp, out):
        # `out` is a list/tuple of feature maps at the FPN scales,
        # possibly wrapped as NestedTensors. Grab the last one — it's
        # what the neck convs consume in Sam3TriViTDetNeck.forward.
        if not isinstance(out, (list, tuple)) or len(out) == 0:
            return
        last = out[-1]
        t = getattr(last, "tensors", last)
        if isinstance(t, torch.Tensor):
            captures.setdefault("vit_out", []).append(
                t.detach().cpu().float().contiguous())

    trunk = model.backbone.vision_backbone.trunk
    hooks.append(trunk.register_forward_hook(_cap_trunk))

    # Sub-drill (iteration 8.5): capture ViT pre-blocks state (after
    # patch embed + pos_embed + ln_pre) and selected block outputs so
    # we can bisect where in the backbone the ~3% cosine gap starts.
    # The pre-blocks tensor matches C's /tmp/dbg_vit_patch.bin; each
    # block matches C's /tmp/dbg_vit_block{NN}.bin. Both sides dump
    # the same flat [np, e] byte layout — Python's block output is
    # [B, H, W, C] NHWC which is already byte-equivalent to [HW, e].
    def _cap_ln_pre(_m, _inp, out):
        if isinstance(out, torch.Tensor):
            captures.setdefault("vit_pre_blocks", []).append(
                out.detach().cpu().float().contiguous())
    hooks.append(trunk.ln_pre.register_forward_hook(_cap_ln_pre))

    # Sub-drill (iteration 8.5+): capture patch_embed output alone
    # (pre-pos_embed, pre-ln_pre) to further bisect the ~18% cosine
    # gap at pre_blocks into patch_embed vs pos_embed/ln_pre. Matches
    # C's /tmp/dbg_vit_patch_only.bin.
    def _cap_patch(_m, _inp, out):
        if isinstance(out, torch.Tensor):
            captures.setdefault("vit_patch_only", []).append(
                out.detach().cpu().float().contiguous())
    hooks.append(trunk.patch_embed.register_forward_hook(_cap_patch))

    _BLOCK_DUMP_IDS = (0, 3, 7, 11, 13, 14, 15, 16, 17, 19, 23, 27, 31)
    for bi in _BLOCK_DUMP_IDS:
        def _cap_block(_m, _inp, out, _bi=bi):
            if isinstance(out, torch.Tensor):
                captures.setdefault(
                    f"vit_block{_bi:02d}", []).append(
                    out.detach().cpu().float().contiguous())
        hooks.append(
            trunk.blocks[bi].register_forward_hook(_cap_block))

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
    # Sub-drill (iteration 4): image_embed = current_vision_feats[-1]
    # (HW, 1, C), captured pre-expand by the monkey-patch on
    # _prepare_memory_conditioned_features. Same byte layout as C's
    # [1, H, W, C] NHWC image_embed.
    slot = "image_embed"
    start = cursors.get(slot, 0)
    end = len(captures.get(slot, []))
    if end > start:
        t = captures[slot][end - 1]
        if isinstance(t, torch.Tensor):
            _dump(f"/tmp/py_trk_image_embed_f{frame_idx}.bin", t)
    cursors[slot] = end

    # ViT sub-drill (iteration 8.5): pre-blocks state + per-block
    # outputs. Only dump for the last propagation frame (frame 2)
    # to match the C side's fixed-path dumps that get overwritten
    # each encode. Dumped as raw [np, e] F32 with no permute.
    if frame_idx == 2:
        for slot in (["vit_patch_only", "vit_pre_blocks"] +
                     [f"vit_block{bi:02d}"
                      for bi in (0, 3, 7, 11, 13, 14, 15, 16, 17,
                                 19, 23, 27, 31)]):
            entries = captures.get(slot, [])
            if len(entries) > 0:
                t = entries[-1]
                if isinstance(t, torch.Tensor):
                    arr = t.contiguous().numpy().astype(np.float32)
                    path = f"/tmp/py_{slot}.bin"
                    arr.tofile(path)
                    print(f"dump: {path} shape={tuple(arr.shape)}",
                          file=sys.stderr)

    # Image-pipeline drill: frame_rgb (NCHW img_batch input) and
    # feat_s1 (NCHW sam2 1x post-neck). Captured once per frame via
    # forward_image forward-hook, in frame-index order. Dump the
    # latest unconsumed one — if >1 new entries, prefer the last.
    # Iteration 8 adds two more NCHW-captured slots: vit_out (ViT
    # trunk output, shared input to all three neck heads) and
    # sam3_feat_s1 (sam3-side 1x neck output — same ViT, different
    # conv weights). Both are dumped NHWC to match the C side.
    for slot in ("frame_rgb", "feat_s1", "vit_out", "sam3_feat_s1"):
        start = cursors.get(slot, 0)
        end = len(captures.get(slot, []))
        if end > start:
            t = captures[slot][end - 1]
            if isinstance(t, torch.Tensor):
                # Drop the batch dim and permute NCHW -> CHW for
                # frame_rgb to match C's make_frame_tensor layout;
                # NCHW -> NHWC for feat_s1 / sam3_feat_s1 / vit_out
                # to match the C-side NHWC wraps.
                if slot == "frame_rgb" and t.dim() == 4:
                    out = t.squeeze(0).contiguous()
                elif t.dim() == 4:
                    out = t.permute(0, 2, 3, 1).contiguous()
                else:
                    out = t
                arr = out.numpy().astype(np.float32)
                path = f"/tmp/py_trk_{slot}_f{frame_idx}.bin"
                arr.tofile(path)
                print(f"dump: {path} shape={tuple(arr.shape)}",
                      file=sys.stderr)
        cursors[slot] = end

    slot = "memattn_out"
    start = cursors.get(slot, 0)
    end = len(captures.get(slot, []))
    if end > start:
        # Multiple calls in one frame is unusual; dump the last one.
        _dump_mem(captures[slot][end - 1], frame_idx)
    cursors[slot] = end

    # Iteration 15: Maskmem bisection dumps. `maskmem_backbone` runs once
    # per frame (cond + prop) to compute that frame's bank entry. Dump
    # the final output + internal intermediates so compare_tracker_layers
    # can localize maskmem internal divergence vs input gap amplification.
    for slot in ("maskmem_out", "maskmem_proj", "maskmem_pix_proj",
                 "maskmem_downsampler_in", "enm_pred_masks_high_res",
                 "maskmem_stage0_conv", "maskmem_stage0_act",
                 "maskmem_stage1_conv", "maskmem_stage1_act",
                 "maskmem_stage2_conv", "maskmem_stage2_act",
                 "maskmem_stage3_conv", "maskmem_stage3_act",
                 "gvr_output"):
        start = cursors.get(slot, 0)
        end = len(captures.get(slot, []))
        if end > start:
            t = captures[slot][end - 1]
            if isinstance(t, torch.Tensor):
                # All are NCHW image-shaped ([1, C, H, W]). Permute to
                # NHWC to match the C-side dump byte layout.
                if t.dim() == 4:
                    t = t.permute(0, 2, 3, 1).contiguous()
                arr = t.numpy().astype(np.float32)
                path = f"/tmp/py_trk_{slot}_f{frame_idx}.bin"
                arr.tofile(path)
                print(f"dump: {path} shape={tuple(arr.shape)}",
                      file=sys.stderr)
        cursors[slot] = end

    # Task 5γ: dump memory-attn encoder INPUTS (tgt / memory /
    # memory_image / memory_image_pos) for bank parity. Upstream
    # forward(image, src, memory_image, memory, image_pos=None,
    # src_pos=None, memory_image_pos=None, memory_pos=None, ...).
    # `src` is the object-query stream (= C's `tgt`).
    slot = "memattn_inputs"
    start = cursors.get(slot, 0)
    end = len(captures.get(slot, []))
    if end > start:
        kw = captures[slot][end - 1]
        tgt_t = kw.get("src")
        memory_t = kw.get("memory")
        memory_image_t = kw.get("memory_image")
        memory_image_pos_t = kw.get("memory_image_pos")
        import torch as _torch  # local alias to avoid top-level churn
        if isinstance(tgt_t, _torch.Tensor):
            _dump(f"/tmp/py_trk_tgt_f{frame_idx}.bin", tgt_t)
        if isinstance(memory_t, _torch.Tensor):
            _dump(f"/tmp/py_trk_memory_f{frame_idx}.bin", memory_t)
        if isinstance(memory_image_t, _torch.Tensor):
            _dump(f"/tmp/py_trk_memory_image_f{frame_idx}.bin",
                  memory_image_t)
        if isinstance(memory_image_pos_t, _torch.Tensor):
            _dump(f"/tmp/py_trk_memory_image_pos_f{frame_idx}.bin",
                  memory_image_pos_t)
        # Iteration 15: Python has a SEPARATE `memory_pos` kwarg for
        # the obj_ptr tail. The encoder concats memory_image_pos +
        # memory_pos[-num_obj_ptr_tokens:] internally (decoder.py:
        # 1340-1346), so the post-pad shape equals memory's row count.
        # Dump the raw memory_pos so we can reconstruct the padded
        # shape on the C side for parity checks.
        memory_pos_t = kw.get("memory_pos")
        if isinstance(memory_pos_t, _torch.Tensor):
            _dump(f"/tmp/py_trk_memory_pos_f{frame_idx}.bin",
                  memory_pos_t)
    cursors[slot] = end

    slot = "mask_decoder_tuple"
    start = cursors.get(slot, 0)
    end = len(captures.get(slot, []))
    if end > start:
        _dump_mask_decoder(captures[slot][end - 1], frame_idx)
    cursors[slot] = end

    # Per-layer memory-attention dumps. Each encoder layer returns a
    # `(image, output)` tuple (DecoupledTransformerDecoderLayerv2.forward
    # — `output` is the running [HW, B, C] object-feature stream that
    # threads through all 4 layers and matches C's per-layer residual
    # accumulator). We dump the `output` element only.
    for i in range(4):
        slot = f"memattn_layer{i}"
        start = cursors.get(slot, 0)
        end = len(captures.get(slot, []))
        if end > start:
            raw = captures[slot][end - 1]
            t = None
            if isinstance(raw, tuple) and len(raw) >= 2:
                t = raw[1]
            elif isinstance(raw, torch.Tensor):
                t = raw
            if isinstance(t, torch.Tensor):
                _dump(f"/tmp/py_trk_{slot}_f{frame_idx}.bin", t)
        cursors[slot] = end

    # Layer-0 sub-drill (iteration 9): intermediate slots captured by
    # the patched _forward_sa / _forward_ca. Byte layout matches C's
    # NHWC/3-D [1, N, C] dumps (Python side is [N, 1, C] pre-transpose
    # but HW contiguous and C innermost — same byte order).
    for slot in ("memattn_l0_sa_out", "memattn_l0_ca_q",
                 "memattn_l0_ca_k", "memattn_l0_ca_v",
                 "memattn_l0_ca_attn",
                 "memattn_l0_ca_k_img", "memattn_l0_ca_k_mem"):
        start = cursors.get(slot, 0)
        end = len(captures.get(slot, []))
        if end > start:
            t = captures[slot][end - 1]
            if isinstance(t, torch.Tensor):
                _dump(f"/tmp/py_trk_{slot}_f{frame_idx}.bin", t)
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
