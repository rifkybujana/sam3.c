# Mask Head: Pixel Decoder + UniversalSegmentationHead

The mask head converts the 200 refined object queries from the decoder into
per-instance binary masks at 288×288 resolution. It has two components:

1. A **pixel decoder** that merges multi-scale FPN features into a single
   high-resolution pixel-embedding map.
2. A **mask predictor** that takes a dot product between each object query
   and the pixel map to produce one mask per query.

## Files

- `reference/sam3/sam3/model/maskformer_segmentation.py`
  - `PixelDecoder` (lines 184-231)
  - `MaskPredictor` (lines 25-53)
  - `UniversalSegmentationHead` (lines 234-337)
  - `SegmentationHead` base class (lines 56-181)
- Builder: `reference/sam3/sam3/model_builder.py:217-243`

## Pipeline

```
backbone_fpn [288², 144², 72²]   encoder_hidden_states (5184, B, 256)
         |                                      |
         +--------------- prompt ---------------+
                                                |
                         +----------------------+
                         |
                         v
           +-----------------------------+
           | UniversalSegmentationHead   |
           |  cross_attend_prompt        |  encoder_hidden updates
           |  on encoder_hidden_states   |  via cross-attn to prompt
           +-----------------------------+
                         |
                         v
           +-----------------------------+
           | PixelDecoder                |
           |  3 FPN levels, 2 upsample   |
           |  iterations                 |
           +-----------------------------+
                         |
                         v
             pixel_embed (B, 256, 288, 288)
                         |
                         v
           +-----------------------------+
           | instance_seg_head (1x1)     |
           | (256->256)                  |
           +-----------------------------+
                         |
                         v
             instance_embeds (B, 256, 288, 288)
                         |
              obj_queries (B, 200, 256)
                         |
                         v
           +-----------------------------+
           | MaskPredictor (MLP + einsum)|
           +-----------------------------+
                         |
                         v
             pred_masks (B, 200, 288, 288)
```

## Invocation (`sam3_image.py:386-424`)

```python
seg_head_outputs = self.segmentation_head(
    backbone_feats=backbone_out["backbone_fpn"],   # 3 levels after scalp
    obj_queries=obj_queries,                       # (6, B, 200, 256)
    image_ids=image_ids,
    encoder_hidden_states=encoder_hidden_states,   # (5184, B, 256)
    prompt=prompt,                                 # (33, B, 256)
    prompt_mask=prompt_mask,                       # (B, 33)
)
```

Because `self.o2m_mask_predict=True` but `apply_dac=False` at inference,
`obj_queries` is the full `hs` tensor `(6, B, 200, 256)`.

## UniversalSegmentationHead (`maskformer_segmentation.py:234-337`)

Configured in `model_builder.py:233-242`:

| Parameter           | Value               |
|---------------------|---------------------|
| hidden_dim          | 256                 |
| upsampling_stages   | 3                   |
| aux_masks           | False               |
| presence_head       | False               |
| dot_product_scorer  | None                |
| act_ckpt            | True                |
| cross_attend_prompt | MultiheadAttention  |

Because `presence_head=False`, `presence_logit` returned from this module is
always `None`. The presence score comes from the **decoder's** presence
token (`presence_logit_dec`), not from here.

### Step 1 — Cross-attend encoder features to the prompt

```python
if self.cross_attend_prompt is not None:
    tgt2 = self.cross_attn_norm(encoder_hidden_states)   # LayerNorm(256)
    tgt2 = self.cross_attend_prompt(
        query=tgt2,                    # (5184, B, 256)
        key=prompt,                    # (33, B, 256)
        value=prompt,
        key_padding_mask=prompt_mask,  # (B, 33)
    )[0]
    encoder_hidden_states = tgt2 + encoder_hidden_states
```

This is a **second** round of image→text cross-attention (the first is
inside the encoder). It re-grounds the image features on the prompt before
they enter the pixel decoder.

Configuration from `model_builder.py:226-231`:

| Parameter  | Value |
|------------|-------|
| num_heads  | 8     |
| embed_dim  | 256   |
| dropout    | 0     |

### Step 2 — Embed pixels

```python
pixel_embed = self._embed_pixels(
    backbone_feats=backbone_feats,       # 3 levels
    image_ids=image_ids,
    encoder_hidden_states=encoder_hidden_states,  # post cross-attn
)
```

See `_embed_pixels` below.

### Step 3 — Instance embedding (1x1 conv)

```python
instance_embeds = self.instance_seg_head(pixel_embed)  # Conv2d(256, 256, 1)
# shape: (B, 256, 288, 288)
```

### Step 4 — Mask prediction

```python
mask_pred = self.mask_predictor(obj_queries[-1], instance_embeds)
# obj_queries[-1] is the final decoder layer's queries: (B, 200, 256)
# shape: (B, 200, 288, 288)
```

### Step 5 — Semantic segmentation (side head)

```python
semantic_seg = self.semantic_seg_head(pixel_embed)  # Conv2d(256, 1, 1)
# shape: (B, 1, 288, 288)
```

This is a single-channel class-agnostic foreground mask, not used by the
primary inference path.

### Return

```python
return {
    "pred_masks":      mask_pred,     # (B, 200, 288, 288)
    "semantic_seg":    semantic_seg,  # (B, 1, 288, 288)
    "presence_logit":  None,
}
```

## _embed_pixels (`maskformer_segmentation.py:104-155`)

With `use_encoder_inputs=True` (set by `UniversalSegmentationHead.__init__`):

```python
# Clone backbone features (per-query broadcast for bs=1)
backbone_visual_feats = [_unwrap(bb_feat).clone() for bb_feat in backbone_feats]

# Reshape encoder hidden states back to a spatial grid
# encoder_hidden_states: (5184, B, 256)
encoder_hidden_states = encoder_hidden_states.permute(1, 2, 0)  # (B, 256, 5184)
spatial_dim = 72 * 72                                           # = 5184
encoder_visual_embed = encoder_hidden_states[..., :spatial_dim].reshape(
    -1, 256, 72, 72,
)                                                               # (B, 256, 72, 72)

# Replace the 72x72 backbone level with the encoder output
backbone_visual_feats[-1] = encoder_visual_embed

pixel_embed = self.pixel_decoder(backbone_visual_feats)
```

**Crucially**, the lowest-resolution level passed to the pixel decoder is
**not** the raw backbone 72x72 feature map — it is replaced with the
**encoder's output**, so the fusion between text and image propagates into
the mask head.

## PixelDecoder (`maskformer_segmentation.py:184-231`)

Simple FPN top-down path: starting from the lowest-resolution level,
repeatedly upsample by 2x (nearest neighbor), add the next-higher backbone
level, then apply `Conv3x3 → GroupNorm(8) → ReLU`.

Config from `model_builder.py:219-224`:

| Parameter             | Value     |
|-----------------------|-----------|
| hidden_dim            | 256       |
| num_upsampling_stages | 3         |
| interpolation_mode    | "nearest" |
| shared_conv           | False     |

Despite `num_upsampling_stages=3` (which creates 3 conv+norm layers), the
forward loop only iterates `len(fpn_feats) = 2` times because only 3 input
levels are passed after scalp. The third conv layer is **never invoked at
inference**.

### Forward (`maskformer_segmentation.py:215-231`)

```python
prev_fpn = backbone_feats[-1]     # (B, 256, 72, 72) -- encoder output
fpn_feats = backbone_feats[:-1]   # [(B,256,288,288), (B,256,144,144)]

for layer_idx, bb_feat in enumerate(fpn_feats[::-1]):
    # iter 0: bb_feat = 144x144, prev_fpn = 72x72
    # iter 1: bb_feat = 288x288, prev_fpn = 144x144
    curr_fpn = bb_feat
    prev_fpn = curr_fpn + F.interpolate(
        prev_fpn, size=curr_fpn.shape[-2:], mode="nearest",
    )
    prev_fpn = self.conv_layers[layer_idx](prev_fpn)
    prev_fpn = F.relu(self.norms[layer_idx](prev_fpn))

return prev_fpn   # (B, 256, 288, 288)
```

Each step:
```
   prev                          curr
(256, h, w) --upsample 2x--> (256, 2h, 2w)  +  bb_feat (256, 2h, 2w)
                                      |
                                      v
                       Conv3x3(256, 256) + GN(8) + ReLU
                                      |
                                      v
                              (256, 2h, 2w)
```

Parameters per conv: 256*256*3*3 + 256 (bias) + 16 (GN) ≈ 590k
Total: 2 used + 1 unused = ~1.77M weights, 1.18M executed.

## MaskPredictor (`maskformer_segmentation.py:25-53`)

The final mask prediction is a dot product between each object query's
embedding and each pixel of the pixel-embedding map.

```python
self.mask_embed = MLP(
    input_dim=256, hidden_dim=256, output_dim=256, num_layers=3,
)
```

### Forward

```python
obj_queries: (B, 200, 256)
pixel_embed: (B, 256, 288, 288)

# Project queries through a 3-layer MLP
mask_embed = self.mask_embed(obj_queries)   # (B, 200, 256)

# Dot product: 'bqc,bchw->bqhw'
mask_preds = torch.einsum("bqc,bchw->bqhw", mask_embed, pixel_embed)
# shape: (B, 200, 288, 288)
```

The output logits are then thresholded at 0 (equivalently, sigmoid at 0.5)
during post-processing to produce binary masks.

## Upsampling to Original Resolution

The mask at `(288, 288)` corresponds to the 1008×1008 model input with a
stride of `1008/288 = 3.5`. The processor's postprocessing step
(`Sam3Processor._post_process_masks`) bilinearly resamples these logits to
the original image size:

```python
pred_masks_orig = F.interpolate(
    pred_masks, size=(original_height, original_width), mode="bilinear",
    align_corners=False,
)
binary_masks = (pred_masks_orig > 0)
```

## Shape Table (Text-Only Inference, B=1)

| Stage                                    | Tensor shape          | Layout |
|------------------------------------------|-----------------------|--------|
| encoder_hidden_states (input)            | (5184, 1, 256)        | SBD    |
| prompt                                   | (33, 1, 256)          | SBD    |
| After cross_attend_prompt residual       | (5184, 1, 256)        | SBD    |
| encoder_hidden_states reshape to grid    | (1, 256, 72, 72)      | NCHW   |
| backbone_feats[0] (288²)                 | (1, 256, 288, 288)    | NCHW   |
| backbone_feats[1] (144²)                 | (1, 256, 144, 144)    | NCHW   |
| backbone_feats[2] (72², replaced)        | (1, 256, 72, 72)      | NCHW   |
| PixelDecoder iter 0 output               | (1, 256, 144, 144)    | NCHW   |
| PixelDecoder iter 1 output (final)       | (1, 256, 288, 288)    | NCHW   |
| instance_embeds                          | (1, 256, 288, 288)    | NCHW   |
| obj_queries[-1]                          | (1, 200, 256)         | BQD    |
| mask_embed                               | (1, 200, 256)         | BQD    |
| pred_masks                               | (1, 200, 288, 288)    | BQHW   |

## Parameter Count

UniversalSegmentationHead (excluding pixel decoder):
- cross_attn_norm (LayerNorm 256): 512
- cross_attend_prompt (MHA 256, 8 heads): 4 × 256 × 256 ≈ 263k
- instance_seg_head (Conv2d 256→256, 1×1): 65.8k
- semantic_seg_head (Conv2d 256→1, 1×1): 257
- mask_embed MLP (3 layers, 256→256→256→256): 3 × (256×256 + 256) ≈ 197k
- **Subtotal: ~527k**

PixelDecoder (num_upsampling_stages=3):
- 3 × Conv3x3(256, 256): 3 × (256×256×9 + 256) ≈ 1.77M
- 3 × GroupNorm(8, 256): 3 × 512 ≈ 1.5k
- **Subtotal: ~1.77M** (of which ~590k is never invoked at inference)

**Mask head total: ~2.3M**

This is tiny compared to the fusion transformer (~22M) and the backbones
(~800M). The mask head is cheap because the heavy lifting — aligning image
and text — has already been done by the transformer encoder and decoder.
