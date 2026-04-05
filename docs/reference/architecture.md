# End-to-End Forward Pass

This document traces a single image+text inference through SAM3 stage by
stage, showing the exact tensor shapes and the source-file call sites for
each step. For detailed documentation of each stage, see the per-component
docs ([image-encoder.md](image-encoder.md), [text-encoder.md](text-encoder.md),
[fusion.md](fusion.md), [mask-head.md](mask-head.md),
[processor.md](processor.md)).

## Running Example

```python
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model     = build_sam3_image_model()
processor = Sam3Processor(model)

image  = Image.open("cat.jpg")          # 640 x 480
state  = processor.set_image(image)     # stage 1-2
output = processor.set_text_prompt("cat", state)  # stage 3-8
```

Final outputs:
```
state["masks"]   : (N, 1, 480, 640) bool
state["boxes"]   : (N, 4) xyxy, original coords
state["scores"]  : (N,) float in [0, 1]
```

## Stage 1 — Image Preprocessing

**File:** `sam3_image_processor.py:21-28` (transform), `41-73` (set_image)

```
PIL.Image (640, 480)
  -> to_image()               -> (3, 480, 640) uint8
  -> ToDtype(uint8, scale)    -> (3, 480, 640) uint8
  -> Resize((1008, 1008))     -> (3, 1008, 1008) uint8
  -> ToDtype(float32, scale)  -> (3, 1008, 1008) float32 [0, 1]
  -> Normalize(0.5, 0.5)      -> (3, 1008, 1008) float32 [-1, 1]
  -> unsqueeze(0)             -> (1, 3, 1008, 1008)
```

Stored: `state["original_height"] = 480`, `state["original_width"] = 640`.

## Stage 2 — Image Backbone (ViT + FPN Neck)

**Files:** `vitdet.py`, `necks.py`, `vl_combiner.py:80-121`

### 2a — PatchEmbed + ViT
```
(1, 3, 1008, 1008)
  -> Conv2d(3, 1024, k=14, s=14)    -> (1, 1024, 72, 72)
  -> permute to NHWC                -> (1, 72, 72, 1024)
  -> + abs_pos_embed (tiled)        -> (1, 72, 72, 1024)
  -> ln_pre                         -> (1, 72, 72, 1024)
  -> 32 x Block
       (window attn blocks 0-6,8-14,16-22,24-30)
       (global attn blocks 7,15,23,31)
       (2D RoPE in every block)
       (ln_post after block 31)      -> (1, 72, 72, 1024)
  -> permute to NCHW                -> (1, 1024, 72, 72)
```

### 2b — Sam3DualViTDetNeck
Four parallel conv stacks at different scales:
```
(1, 1024, 72, 72)
  -> scale 4.0  convT+GELU+convT+1x1+3x3  -> (1, 256, 288, 288)
  -> scale 2.0  convT+1x1+3x3             -> (1, 256, 144, 144)
  -> scale 1.0  identity+1x1+3x3          -> (1, 256, 72, 72)
  -> scale 0.5  maxpool+1x1+3x3           -> (1, 256, 36, 36)
```

### 2c — Scalp (`SAM3VLBackbone` with `scalp=1`)
The lowest-resolution level (36×36) is dropped:
```
backbone_fpn      = [288², 144², 72²]            # 3 levels
vision_pos_enc    = [pos_288², pos_144², pos_72²]
```

Stored in `state["backbone_out"]`. Cost: ~800M params, runs **once** per
image.

## Stage 3 — Text Encoding

**Files:** `tokenizer_ve.py`, `text_encoder_ve.py`, called from
`sam3_image_processor.py:113-125`.

### 3a — Tokenization
```
"cat"
  -> _clean_lower       -> "cat"
  -> regex pretokenize  -> ["cat"]
  -> BPE merge          -> [2368]       (token id for "cat</w>")
  -> add SOT/EOT        -> [49406, 2368, 49407]
  -> pad to 32          -> [49406, 2368, 49407, 0, 0, ..., 0]
  -> torch.tensor       -> (1, 32) int64
```

### 3b — Text Transformer
```
(1, 32) int64
  -> token_embedding                           -> (1, 32, 1024)
  -> + positional_embedding (learned)          -> (1, 32, 1024)
  -> 24 x ResidualAttentionBlock (causal mask) -> (1, 32, 1024)
  -> ln_final                                  -> (1, 32, 1024)
```

### 3c — VETextEncoder projection
```
(1, 32, 1024)
  -> transpose to seq-first   -> (32, 1, 1024)
  -> Linear(1024, 256)        -> (32, 1, 256)   = language_features
mask = (tokens != 0).ne(1)    -> (1, 32) bool, True = pad
                                = language_mask
```

Stored in `state["backbone_out"]`. Cost: ~353M params per prompt.

## Stage 4 — Prompt Assembly

**File:** `sam3_image.py:167-210` (`_encode_prompt`)

### 4a — Geometry encoding
With no geometric prompts, `SequenceGeometryEncoder` returns just a CLS
token:
```
geo_feats  -> (1, 1, 256)     -- single learned CLS
geo_masks  -> (1, 1) bool     -- False = valid
```

### 4b — Concatenation
```
prompt      = cat([txt_feats, geo_feats, visual_prompt_embed], dim=0)
            -> (32 + 1 + 0, 1, 256) = (33, 1, 256)

prompt_mask = cat([txt_masks, geo_masks, visual_prompt_mask], dim=1)
            -> (1, 33) bool
```

## Stage 5 — Fusion Encoder

**File:** `encoder.py`, called from `sam3_image.py:212-250` (`_run_encoder`)

### 5a — Flatten image features
```
img_feats[-1]: (5184, 1, 256) = (72*72, B, 256)    -- seq-first
img_pos[-1]:   (5184, 1, 256)
```

### 5b — 6 layer cross-attention loop
Each layer: image self-attn → image cross-attn to text prompt → FFN
(pre-norm, ReLU, 8 heads, dim_feedforward=2048).

```
for layer in encoder.layers:   # 6 layers
    output = layer(
        tgt=output,                        # image
        memory=prompt_batch_first,         # text
        tgt_key_padding_mask=None,
        memory_key_padding_mask=prompt_mask,
        pos=None,                          # (text pos is zero)
        query_pos=img_pos,
    )
```

Output: `encoder_out.memory` = (5184, 1, 256).
Text prompt is **not** modified → `encoder_out.memory_text = prompt`.

Cost: ~9.5M params.

## Stage 6 — Fusion Decoder

**File:** `decoder.py`, called from `sam3_image.py:252-298` (`_run_decoder`)

### 6a — Initialize queries
```
query_embed.weight: (200, 256)
tgt               : (200, 1, 256) = query_embed.unsqueeze(1)
reference_boxes   : (200, 1, 4) = reference_points.weight.sigmoid()
presence_token    : (1, 1, 256)
```

### 6b — 6 layer refinement loop
Each iteration (layer `i` of 6):
1. Generate sine position embed from `reference_boxes` → `query_pos`
2. Compute boxRPB bias (log scale) → `memory_mask` (B*8, 200, 5184)
3. Run `TransformerDecoderLayer`:
   - self-attn: concat(presence, queries) self-attend
   - cross-attn to **text**: queries attend to (32, 1, 256) text
   - cross-attn to **image**: queries attend to (5184, 1, 256) image
     with boxRPB bias added
   - FFN
4. Refine reference_boxes: `sigmoid(inverse_sigmoid(ref) + bbox_embed(output))`
5. Record intermediate outputs

After 6 iterations:
```
hs               : (6, 1, 200, 256)   -- all layers
reference_boxes  : (6, 1, 200, 4)
dec_presence_out : (6, 1, 1)
```

Cost: ~11.5M params.

## Stage 7 — Score + Box Prediction Heads

**File:** `sam3_image.py:300-384` (`_update_scores_and_boxes`)

### 7a — DotProductScoring
```
outputs_class = dot_prod_scoring(hs, prompt, prompt_mask)
            -> (6, 1, 200, 1)
```

Internally: mean-pool masked text → MLP → dot product with each query.

### 7b — Box refinement
```
anchor_box_offsets       = bbox_embed(hs)                         # (6, 1, 200, 4)
reference_boxes_inv_sig  = inverse_sigmoid(reference_boxes)
outputs_coord            = (reference_boxes_inv_sig + offsets).sigmoid()
                        -> (6, 1, 200, 4) cxcywh in [0, 1]
outputs_boxes_xyxy       = box_cxcywh_to_xyxy(outputs_coord)
```

Only the final layer (index -1) is taken for inference outputs:
```
out["pred_logits"]        = outputs_class[-1]       -> (1, 200, 1)
out["pred_boxes"]         = outputs_coord[-1]       -> (1, 200, 4)
out["presence_logit_dec"] = dec_presence_out[-1]    -> (1, 1)
```

## Stage 8 — Mask Head

**File:** `maskformer_segmentation.py`, called from
`sam3_image.py:386-424` (`_run_segmentation_heads`)

### 8a — Re-ground image features on prompt
```
encoder_hidden_states (5184, 1, 256)
  -> cross_attn_norm (LN)
  -> cross_attend_prompt (q=img, k=v=prompt)   # 8 heads
  -> residual add                              -> (5184, 1, 256)
```

### 8b — PixelDecoder (FPN top-down)
Replace backbone 72x72 level with (refined) encoder output, then upsample:
```
backbone_feats[2] <- encoder_hidden reshaped  -> (1, 256, 72, 72)
prev_fpn = backbone_feats[2]

# iter 0: upsample 72->144, add backbone 144
prev_fpn = conv0(interp(prev, 144) + bb_144)  -> (1, 256, 144, 144)

# iter 1: upsample 144->288, add backbone 288
prev_fpn = conv1(interp(prev, 288) + bb_288)  -> (1, 256, 288, 288)
```

### 8c — Instance head + mask predictor
```
instance_embeds = Conv2d(256, 256, 1)(pixel_embed)     -> (1, 256, 288, 288)
mask_embed      = MLP(3)(obj_queries[-1])              -> (1, 200, 256)
pred_masks      = einsum('bqc,bchw->bqhw', ...)        -> (1, 200, 288, 288)
```

Cost: ~2.3M params.

## Stage 9 — Postprocessing

**File:** `sam3_image_processor.py:182-222` (`_forward_grounding`)

### 9a — Fuse scores with presence
```
out_probs      = pred_logits.sigmoid()                       # (1, 200, 1)
presence_score = presence_logit_dec.sigmoid().unsqueeze(1)   # (1, 1, 1)
out_probs      = (out_probs * presence_score).squeeze(-1)    # (1, 200)
```

### 9b — Filter by confidence
```
keep = out_probs > 0.5    # default threshold
surviving: N queries (say N=3)
```

### 9c — Rescale boxes to original coords
```
boxes (cxcywh, [0,1]) -> xyxy * [W, H, W, H]
                    -> (3, 4) in original pixel coords
```

### 9d — Resize masks to original size
```
pred_masks (N, 288, 288)
  -> unsqueeze(1)                           -> (3, 1, 288, 288)
  -> bilinear interpolate to (480, 640)     -> (3, 1, 480, 640)
  -> sigmoid                                -> (3, 1, 480, 640) in [0, 1]
  -> > 0.5                                  -> (3, 1, 480, 640) bool
```

### 9e — Assemble output state
```
state["masks_logits"] = sigmoid, (3, 1, 480, 640) float
state["masks"]        = > 0.5,   (3, 1, 480, 640) bool
state["boxes"]        = xyxy,    (3, 4) in original px
state["scores"]       = fused,   (3,)
```

## Summary of Costs

| Stage | Component                  | Params   | Runs per         |
|-------|----------------------------|----------|------------------|
| 1     | Preprocessing              | 0        | per image        |
| 2     | ViT backbone + FPN neck    | ~446M    | per image        |
| 3     | Tokenizer + text transformer | ~353M  | per prompt       |
| 4     | Geometry encoder           | ~1.3M    | per prompt       |
| 5     | Fusion encoder             | ~9.5M    | per prompt       |
| 6     | Fusion decoder             | ~11.5M   | per prompt       |
| 7     | Score/box heads            | ~1.0M    | per prompt       |
| 8     | Mask head + pixel decoder  | ~2.3M    | per prompt       |
| 9     | Postprocessing             | 0        | per prompt       |
| —     | **Total**                  | ~848M    | —                |

The `set_image()` API caches the Stage 2 output so stages 3-9 can be re-run
cheaply with different text or box prompts. Re-prompting costs roughly
~380M params of forward pass (text + fusion + heads), versus ~800M for a
fresh image.
