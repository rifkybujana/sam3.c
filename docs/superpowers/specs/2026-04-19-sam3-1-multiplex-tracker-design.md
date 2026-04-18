# SAM 3.1 multiplex tracker (sub-project 2)

Date: 2026-04-19
Status: Design in progress

## Context

Sub-project 1 (2026-04-18) landed variant-aware weight loading, the tri-neck,
and image-path inference for `sam3.1_multiplex.pt`. It also rejected video
tracking for SAM 3.1 because the tracker architecture differs substantially
from SAM 3's. This spec is the plan to implement that tracker.

**Motivation**: Object Multiplex is the headline feature of SAM 3.1 — joint
multi-object video tracking at ~7× SAM 3's speed. Image-only SAM 3.1 barely
changes segmentation quality; the tracker is where the value sits.

## Scope

### In scope (sub-project 2)

Correctness — NOT the 7× multiplex speedup. Objects are processed one at a
time; no joint forward. The memory attention module takes a single object's
query/memory at a time. This is still meaningfully useful: it unblocks
`sam3 track -m models/sam3.1.sam3` to produce correct masks.

Concretely:

- New 8-head memory-attention transformer
  (`DecoupledTransformerDecoderLayerv2` + `SimpleRoPEAttention`,
  4 encoder layers, 122 weight tensors).
- New maskmem backbone (`SimpleMaskEncoder` with multiplex-aware
  `SimpleMaskDownSampler`: `out_dim=256`, `multiplex_count=16`,
  `starting_out_chan=4`, `input_channel_multiplier=2`).
- New mask decoder (`sam_mask_decoder`, SAM-style two-way transformer
  with 2 layers + final token→image attention + 3-stage upscaler +
  3 hypernetwork MLPs + IoU head + obj-score head; 125 tensors).
- Object pointer plumbing: `obj_ptr_proj` (3-layer MLP),
  `obj_ptr_tpos_proj`, `no_obj_ptr_linear`, and the embedding tensors
  `maskmem_tpos_enc`, `no_obj_embed_spatial`, `output_valid_embed`,
  `output_invalid_embed`, `interactivity_no_mem_embed`,
  `image_pe_layer`.
- `sam3_video_start_ex` branches on variant; SAM 3.1 routes to a new
  `sam3_tracker_v2` subsystem.
- Per-frame propagation updated for the new tracker, one object per pass.
- `tests/test_sam3_1_track` — end-to-end `sam3 track` smoke test.

### Explicitly out of scope (later sub-projects)

- **Sub-project 3**: The interactive pathway (`interactive_sam_mask_decoder`,
  131 tensors; `interactive_sam_prompt_encoder`, 17 tensors;
  `interactive_obj_ptr_proj`, 6 tensors; `interactive_mask_downsample`) is
  for SAM 1-task single-image interactive prompts. Ship per-object video
  tracking first; interactive comes after.
- **Sub-project 4**: Multiplex joint forward — `MultiplexController`
  groups 16 objects into shared memory attention forwards. The 7× speedup
  happens here. Requires sub-project 2 to be correct first.
- **Sub-project 5**: Detector/tracker association heuristics —
  `Sam3MultiplexTrackingWithInteractivity` (hotstart, masklet
  confirmation, IoM recondition, batched grounding). Quality polish on
  top of 2+4.

## Weight inventory (from real checkpoint)

All under `tracker_v2.*` after the rename handler landed in the feature
branch. Totals per module: 457 tensors, 16 modules.

### 1. `transformer` — 122 tensors

4 encoder layers × 29 tensors + `encoder.norm.{weight,bias}`.

Per-layer breakdown (`DecoupledTransformerDecoderLayerv2`):
- `self_attn_{q,k,v,out}_proj.{weight,bias}` — 8 tensors, 8-head
  `SimpleRoPEAttention` over the per-object query stream.
- `cross_attn_{q,k,v,out}_proj.{weight,bias}` — 8 tensors, object
  query ← memory-token cross attention.
- `image_cross_attn_{q,k}_proj.{weight,bias}` — 4 tensors; note V and
  out are absent. This is a RoPE-only query/key side attention that
  shares V+out with `cross_attn`.
- `linear1.{weight,bias}`, `linear2.{weight,bias}` — FFN dim 2048.
- `norm1`, `norm2`, `norm3` — three layer norms (pre-self-attn,
  pre-cross-attn, pre-FFN).

All weights are `[256, 256]` or `[256]` except the FFN
(`[2048, 256]` / `[256, 2048]`).

### 2. `sam_mask_decoder` — 125 tensors

Separate SAM-style mask decoder instance (not shared with the image
model's seg head):

- `transformer.layers.0..1.self_attn.{q,k,v,out}_proj.{weight,bias}`
- `transformer.layers.0..1.cross_attn_{token_to_image,image_to_token}.{q,k,v,out}_proj.{weight,bias}`
- `transformer.layers.0..1.mlp.{lin1,lin2}.{weight,bias}`
- `transformer.layers.0..1.norm{1..4}.{weight,bias}`
- `transformer.final_attn_token_to_image.{q,k,v,out}_proj.{weight,bias}`
- `transformer.norm_final_attn.{weight,bias}`
- `output_upscaling.{0,1,3}.{weight,bias}` — 3-stage upscaler (conv +
  layer norm + deconv pattern).
- `output_hypernetworks_mlps.{0..2}.layers.{0..2}.{weight,bias}` —
  3 hypernets × 3 layers = 9 weight+bias pairs.
- `iou_prediction_head.layers.{0..2}.{weight,bias}`
- `pred_obj_score_head.layers.{0..2}.{weight,bias}`
- `conv_s0.{weight,bias}` — `[32, 256, 1, 1]` high-res feature conv
  (for 4× scale).
- `conv_s1.{weight,bias}` — `[64, 256, 1, 1]` for 2× scale.
- `iou_token.weight`, `mask_tokens.weight`, `obj_score_token.weight` —
  learned output tokens.

### 3. `maskmem_backbone` — 38 tensors

- `mask_downsampler.encoder.{0,3,6,9}.{weight,bias}` — 4 convs
  (32→16→64→256→1024, k=3, s=2).
- `mask_downsampler.encoder.{1,4,7,10}.{weight,bias}` — 4
  `LayerNorm2d`s (channels-first; applied per-pixel across channels).
- `mask_downsampler.encoder.12.{weight,bias}` — final 1×1 conv
  projection (1024 → 256).
- `fuser.layers.{0,1}.dwconv.{weight,bias}` — depthwise 7×7 conv
  (`[256, 1, 7, 7]`).
- `fuser.layers.{0,1}.norm.{weight,bias}` — `LayerNorm(256)` in
  channels-last.
- `fuser.layers.{0,1}.pwconv1.{weight,bias}` — `Linear(256 → 1024)`.
- `fuser.layers.{0,1}.pwconv2.{weight,bias}` — `Linear(1024 → 256)`.
- `fuser.layers.{0,1}.gamma` — layer-scale parameter `[256]`.
- `pix_feat_proj.{weight,bias}` — `Conv2d(256 → 256, 1×1)`.

Forward: `masks` (multiplex-packed, `[B, 32, 1152, 1152]`) →
sigmoid → 4× strided conv downsample + norm + GELU → 1×1 proj →
`[B, 256, 72, 72]`; `pix_feat` → `pix_feat_proj` →
add → two `CXBlock` (ConvNeXt) → out.

### 4. `obj_ptr_proj` — 6 tensors

3-layer MLP (`[256 → 256 → 256 → 256]` with bias). Takes an object's
mask-token embedding and produces its inter-frame "object pointer".

### 5. `obj_ptr_tpos_proj` — 2 tensors

`Linear(256 → 256)` on the temporal pos embedding added to obj ptrs.

### 6. `no_obj_ptr_linear` — 2 tensors

`Linear(256 → 256)` producing the "no-object" pointer when an object
is fully occluded.

### 7. `image_pe_layer` — 1 tensor

`positional_encoding_gaussian_matrix` `[2, 128]`. SAM 3.1 ships a
learned Gaussian positional-encoding basis rather than the sinusoid
the rest of the repo uses.

### 8. Embedding tensors — 6 tensors total

- `maskmem_tpos_enc` `[7, 1, 1, 256]` — temporal position for each of
  the 7 memory slots (num_maskmem=7).
- `no_obj_embed_spatial` `[16, 256]` — per-multiplex-slot occluded
  embedding.
- `output_valid_embed` `[16, 256]`, `output_invalid_embed` `[16, 256]`
  — object-presence modulation.
- `interactivity_no_mem_embed` `[1, 1, 256]` — used when the
  interactive path hasn't accumulated memory yet.

### Out of scope (sub-project 3)

- `interactive_sam_mask_decoder` — 131 tensors, same structure as
  `sam_mask_decoder`.
- `interactive_sam_prompt_encoder` — 17 tensors (`pe_layer`,
  `mask_downscaling` stack, `no_mask_embed`, `point_embeddings`,
  `shared_embedding.positional_embedding`, `not_a_point_embed`).
- `interactive_obj_ptr_proj` — 6 tensors, MLP.
- `interactive_mask_downsample` — 2 tensors, a single 4×4 conv that
  turns a full-res mask into a low-res probability map for the
  interactive memory path.

## Architecture

```
+---------------------------------------------+
|             sam3_ctx (variant)              |
+---------------------------------------------+
                    |
                    v
+---------------------------------------------+
|       sam3_video_session (variant aware)    |
|                                             |
|  sam3_tracker            sam3_tracker_v2    |
|  (existing, SAM 3)       (new, SAM 3.1)     |
+---------------------------------------------+
                              |
                              v
  +----------------------------------------------------+
  |  sam3_tracker_v2                                   |
  |                                                    |
  |  memory_attn_v2  (8-head decoupled, 4 layers)      |
  |  maskmem_v2      (256-out, multiplex downsampler)  |
  |  mask_decoder_v2 (new SAM decoder)                 |
  |  obj_ptr_proj    (3-layer MLP)                     |
  |  obj_ptr_tpos_proj + no_obj_ptr_linear             |
  |  maskmem_tpos_enc (parameter)                      |
  |  no_obj_embed_spatial (parameter)                  |
  |  output_{valid,invalid}_embed (parameter)          |
  |  image_pe_layer (Gaussian PE basis)                |
  +----------------------------------------------------+
```

Per-frame propagate loop (simplified, single-object):

```
for frame_idx in propagate_order:
    pix_feat = frame_cache[frame_idx]  /* neck output */
    /* memory attention: object query <- pix_feat + memory bank */
    obj_query = memory_attn_v2(
        obj_query=last_mask_embed,
        memory_tokens=memory_bank_tokens(obj_id),
        memory_pos=maskmem_tpos_enc,
        obj_ptrs=obj_ptr_bank(obj_id),
        image_feat=pix_feat,
        image_pe=image_pe_layer(pix_feat.shape),
    )
    /* mask decoder: produce masks from conditioned image features */
    masks, iou, obj_score = mask_decoder_v2(
        image_embeddings=obj_query,
        image_pe=image_pe,
        prompts=None,  /* non-conditional frame */
        high_res_features=[conv_s0_out, conv_s1_out],
    )
    /* update memory bank */
    new_mask_embed = maskmem_v2(pix_feat, masks)
    memory_bank.push(obj_id, new_mask_embed)
    obj_ptr = obj_ptr_proj(mask_tokens_out)
    obj_ptr_bank.push(obj_id, obj_ptr)
```

For conditioning (prompted) frames, the mask decoder additionally takes
geometric prompt tokens through the SAM-style prompt encoder path —
which for the non-interactive entry point reuses the existing
`sam3_prompt_encoder` machinery. Sub-project 3 adds the separate
interactive decoder.

## File layout (proposed)

### Create

- `src/model/memory_attn_v2.h` / `.c` — new 8-head decoupled memory
  attention (122 tensors).
- `src/model/maskmem_v2.h` / `.c` — multiplex-aware mask encoder
  (38 tensors).
- `src/model/mask_decoder_v2.h` / `.c` — new SAM-style mask decoder
  with conv_s0/conv_s1 high-res features (125 tensors).
- `src/model/tracker_v2.h` / `.c` — top-level SAM 3.1 tracker struct
  that owns the three modules above + embedding params + object-pointer
  projections.
- `tests/test_tracker_v2_load.c` — round-trip: convert, open, verify
  all 457 tensors resolve (no warnings).
- `tests/test_sam3_1_track.c` — end-to-end `sam3 track` on a short
  video from assets/kids.mp4, asserts no crash + plausible mask
  shapes.

### Modify

- `src/model/sam3_video.c` — remove the `SAM3_VARIANT_SAM3_1 reject`;
  branch on variant at `sam3_video_start_ex` to allocate either
  `sam3_tracker` (SAM 3) or `sam3_tracker_v2` (SAM 3.1); update the
  propagate loop analogously.
- `src/model/video_session.h` — union of tracker types OR add a
  `variant` field and pointer to either old or new struct.
- `tools/weight_rename.c` — already done: `tracker.model.` →
  `tracker_v2.`

## Risks

- **Scale**: ~3 000 – 5 000 new C LOC across four modules. Probably
  4–6 work sessions, not one. Committing in thin vertical slices
  (load one subsystem at a time + a tiny forward smoke test) keeps
  the branch green throughout.
- **RoPE dimensions**: `SimpleRoPEAttention` uses 8 heads at d_head=32
  on a 72×72 feature grid with `rope_theta=10000`. The existing
  single-head RoPE implementation cannot be reused as-is; we likely
  need a new graph builder.
- **Gaussian PE**: `image_pe_layer.positional_encoding_gaussian_matrix`
  is a learned basis that the rest of sam3 does not currently use.
  Needs a new PE builder that multiplies grid coords against that
  `[2, 128]` matrix.
- **`LayerNorm2d`**: The mask downsampler applies per-pixel layer norm
  over channels. The existing `cpu_groupnorm.c` / `cpu_layernorm.c`
  kernels assume NHWC channels-last. Either reshape or add a
  `LayerNorm2d` helper.
- **Multiplex packing**: Even per-object (sub-project 2), the mask
  downsampler's first conv expects 32 input channels because the
  checkpoint was trained with `multiplex_count=16,
  input_channel_multiplier=2`. Single-object input must be packed into
  the 32-channel tensor with appropriate zero-padding.
- **Testing**: The Python reference path is doubly blocked (CUDA +
  BatchedDatapoint). Sub-project 2's parity test will start as a
  smoke test ("runs, masks are in [0,1], IoU ≥ 0"); full parity lands
  when one of those blockers clears or we hand-build a reference.

## Phasing

The implementation is split into thin vertical slices. Each slice
commits a compiling, test-green branch so we never sit on a broken
main line for days.

| Phase | Subsystem                         | LOC est. | New tests                    |
|-------|-----------------------------------|----------|------------------------------|
| 2.1   | tracker_v2 struct + weight load   |   300    | test_tracker_v2_load (shape) |
| 2.2   | maskmem_v2 forward                |   500    | unit test w/ synthetic input |
| 2.3   | memory_attn_v2 forward            |   700    | unit test w/ synthetic input |
| 2.4   | mask_decoder_v2 forward           |   600    | unit test w/ synthetic input |
| 2.5   | sam3_video_start + propagate wire |   400    | test_sam3_1_track (smoke)    |
| 2.6   | image_pe_layer + LayerNorm2d      |   200    | unit tests each              |

Phase 2.1 is the next commit after this spec. It lands the struct +
loader + `test_tracker_v2_load` but exposes no forward functions.
After that phase is green, 2.2 lands the first real forward.

## Open questions

1. **Tokenizer / text encoder reuse**: the SAM 3.1 detector uses the
   same CLIP text encoder as SAM 3. Sub-project 1 already loads it.
   No action needed for the tracker.
2. **Frame cache reuse**: the existing `sam3_frame_cache` in
   `sam3_video.c` caches neck outputs per frame. The tracker_v2 path
   consumes the same neck outputs, so the cache carries over.
3. **`num_maskmem` = 7**: matches SAM 3. No change.
4. **Metal kernels**: no new Metal kernels are strictly required —
   all new modules decompose into existing ops (matmul, conv, norm,
   softmax, GELU). The SimpleRoPEAttention uses a rotary transform
   that the existing `cpu_rope.c` / Metal RoPE kernels already
   implement; the per-head dim just needs to be configurable.
