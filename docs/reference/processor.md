# Processor: Preprocessing + Postprocessing

`Sam3Processor` is the user-facing wrapper around `Sam3Image`. It handles
image preprocessing, prompt routing (text / box / point), and mask/box
postprocessing back to the original image coordinates. All method calls
update a mutable `state` dict that threads through the API.

## Files

- `reference/sam3/sam3/model/sam3_image_processor.py`
- Dummy prompt helper: `reference/sam3/sam3/model/sam3_image.py:_get_dummy_prompt`

## Public API

```python
processor = Sam3Processor(model, resolution=1008, device="cuda",
                          confidence_threshold=0.5)

state = processor.set_image(image)                            # bake image
state = processor.set_text_prompt("a cat", state)             # get masks
state = processor.add_geometric_prompt(box, label=True, state) # refine
state = processor.set_confidence_threshold(0.7, state)        # re-filter
processor.reset_all_prompts(state)                            # clear prompts
```

## Preprocessing (`__init__`, `set_image`)

The preprocessing pipeline (`sam3_image_processor.py:21-28`):

```python
self.transform = v2.Compose([
    v2.ToDtype(torch.uint8, scale=True),      # PIL -> uint8 (0..255)
    v2.Resize(size=(1008, 1008)),             # bilinear resize
    v2.ToDtype(torch.float32, scale=True),    # uint8 -> float32 (0..1)
    v2.Normalize(mean=[0.5, 0.5, 0.5],
                 std =[0.5, 0.5, 0.5]),       # to [-1, 1]
])
```

**Steps:**
1. Convert input to uint8 with value scaling (PIL → uint8 Tensor)
2. Bilinearly resize to **1008×1008** (non-uniform aspect ratio squash; no
   letterboxing or padding)
3. Convert uint8 → float32 with value scaling: `0..255 → 0..1`
4. Normalize: `(x - 0.5) / 0.5 = 2*x - 1`, i.e. `[0,1] → [-1, 1]`

**Shape after preprocessing:** `(1, 3, 1008, 1008)` float32 on device.

### Aspect-Ratio Warning

Because resizing is done with `Resize(size=(1008, 1008))` **without**
preserving aspect ratio, non-square images are stretched. The mask output
is always at 288×288 relative to the stretched input, and the postprocessor
undoes the stretch by resizing back to `(original_height, original_width)`.

### set_image (`set_image`, lines 41-73)

```python
state["original_height"] = height
state["original_width"]  = width
state["backbone_out"]    = self.model.backbone.forward_image(image)
```

`backbone.forward_image` runs the ViT + FPN neck, producing a 3-level FPN
(after scalp=1):

```python
backbone_out = {
    "vision_features":    (B, 256, 72, 72),      # = backbone_fpn[-1]
    "vision_pos_enc":     [pos_288, pos_144, pos_72],
    "backbone_fpn":       [(B,256,288,288), (B,256,144,144), (B,256,72,72)],
    "sam2_backbone_out":  None | {...},          # enabled only if SAM2
                                                 # interactive predictor used
}
```

This is the expensive call (~800M parameters on the image path). The
`state` is typically reused across many text/box prompts.

### set_image_batch

Same as `set_image` but takes a list of PIL images and batches them after
per-image `transform`. Stores `original_heights` / `original_widths` as
lists.

## Text Prompt (`set_text_prompt`, lines 112-125)

```python
text_outputs = self.model.backbone.forward_text([prompt], device=self.device)
state["backbone_out"].update(text_outputs)       # adds language_features etc.
if "geometric_prompt" not in state:
    state["geometric_prompt"] = self.model._get_dummy_prompt()
return self._forward_grounding(state)
```

After this call, `state["backbone_out"]` contains:

| Key                       | Shape              | Meaning                        |
|---------------------------|--------------------|--------------------------------|
| `language_features`       | (32, 1, 256)       | text memory (post-projection)  |
| `language_mask`           | (1, 32) bool       | True = pad position            |
| `language_embeds`         | (32, 1, 1024)      | raw text embeddings            |

The **single** text prompt is always batch-size 1 in this API. Multiple
prompts in one image are not supported by the processor (the model supports
it, but the processor's `find_stage.text_ids = [0]` hardcodes one prompt).

## Geometric Prompt (`add_geometric_prompt`, lines 127-152)

Accepts a single box in **normalized cxcywh** format `[cx, cy, w, h] ∈ [0,1]`
with a polarity label (`True` = positive / keep, `False` = negative / suppress).

If no text prompt has been set, the processor automatically encodes the
string `"visual"` as a placeholder, since the model's fusion architecture
requires a text prompt:

```python
if "language_features" not in state["backbone_out"]:
    dummy_text_outputs = self.model.backbone.forward_text(["visual"], ...)
    state["backbone_out"].update(dummy_text_outputs)
```

The box is appended to `state["geometric_prompt"]` (a `Prompt` object from
`geometry_encoders.py`) and the grounding forward is run.

## FindStage (`__init__`, lines 31-39)

`FindStage` is a config container passed to `model.forward_grounding`.
The processor hardcodes a single-image, single-prompt configuration:

```python
self.find_stage = FindStage(
    img_ids=torch.tensor([0]),      # which image in the batch
    text_ids=torch.tensor([0]),     # which text prompt to use
    input_boxes=None,               # geometry handled via geometric_prompt
    input_boxes_mask=None,
    input_boxes_label=None,
    input_points=None,
    input_points_mask=None,
)
```

## Postprocessing (`_forward_grounding`, lines 182-222)

This runs the full fusion+decoder stack and turns raw logits into final
predictions in original image coordinates.

### Step 1 — Forward pass

```python
outputs = self.model.forward_grounding(
    backbone_out=state["backbone_out"],
    find_input=self.find_stage,
    geometric_prompt=state["geometric_prompt"],
    find_target=None,
)
# outputs keys:
#   pred_logits:         (1, 200, 1)     class logits
#   pred_boxes:          (1, 200, 4)     cxcywh in [0, 1]
#   pred_masks:          (1, 200, 288, 288)
#   presence_logit_dec:  (1, 1)          scalar presence per image
```

### Step 2 — Fuse scores with presence

```python
out_probs       = out_logits.sigmoid()                  # (1, 200, 1)
presence_score  = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)  # (1, 1, 1)
out_probs       = (out_probs * presence_score).squeeze(-1)  # (1, 200)
```

The final per-query score is the product of:
- query-text dot-product score (from `DotProductScoring`)
- image-level presence score (from the decoder's presence token)

Both are already in [0, 1] after sigmoid. Their product naturally
down-weights queries from images that look unlikely to contain the object
at all.

### Step 3 — Confidence threshold

```python
keep       = out_probs > self.confidence_threshold       # default 0.5
out_probs  = out_probs[keep]                             # (N,)
out_masks  = out_masks[keep]                             # (N, 288, 288)
out_bbox   = out_bbox[keep]                              # (N, 4)
```

All filtering is based on the fused score. Queries below threshold are
discarded entirely; no NMS is run in the processor (the model already
produces distinct per-query predictions due to the Hungarian training).

### Step 4 — Box denormalization

```python
boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)            # still in [0,1]
scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
boxes = boxes * scale_fct[None, :]                      # pixels, original
```

Boxes come out of the decoder normalized to the 1008×1008 working
resolution. Multiplying by `[W, H, W, H]` in the original image coordinates
rescales them correctly because the preprocessing resize was uniform scale
per axis.

### Step 5 — Mask resize + sigmoid + threshold

```python
out_masks = interpolate(
    out_masks.unsqueeze(1),       # (N, 1, 288, 288)
    (img_h, img_w),
    mode="bilinear",
    align_corners=False,
).sigmoid()                        # (N, 1, img_h, img_w) in [0, 1]

state["masks_logits"] = out_masks          # probabilities, float
state["masks"]        = out_masks > 0.5    # binary
```

Notes:
- The logits are **bilinearly** interpolated to the original image size
  **before** the sigmoid, so the resize operates in logit space where
  values are unbounded. This is slightly different from resizing after
  sigmoid (where values are squashed to [0,1] first).
- The final threshold is `> 0.5` on the sigmoid output, i.e. `> 0` on the
  logit.
- Despite the variable name, `masks_logits` is **not** logits — it is
  probabilities in `[0,1]`.

### Step 6 — Assemble output state

```python
state["masks_logits"]  # (N, 1, H, W) float in [0, 1]
state["masks"]         # (N, 1, H, W) bool
state["boxes"]         # (N, 4) xyxy in original pixel coords
state["scores"]        # (N,) float in [0, 1]
```

## set_confidence_threshold

Updates `self.confidence_threshold` and, if a prior inference result exists
in `state`, reruns `_forward_grounding` to re-filter. The model forward is
re-invoked each time — the processor does not cache the raw 200-query
outputs. A caching variant would avoid recomputing the segmentation head
but would require a deeper API change.

## reset_all_prompts

Drops all prompts and results from the state, but **keeps** the image
backbone output. Specifically, clears:

- `backbone_out.language_features`, `language_mask`, `language_embeds`
- `geometric_prompt`, `boxes`, `masks`, `masks_logits`, `scores`

After reset, `set_text_prompt` or `add_geometric_prompt` can be called
again without re-running the image backbone.

## End-to-End Shape Trace

For a 640×480 input image with threshold 0.5 matching N=3 instances:

| Stage                             | Shape                   | Dtype          |
|-----------------------------------|-------------------------|----------------|
| Input PIL image                   | 480 × 640 × 3           | uint8          |
| After resize + normalize          | (1, 3, 1008, 1008)      | float32, [-1,1]|
| `pred_masks` (pre-resize)         | (1, 200, 288, 288)      | float32, logits|
| `pred_boxes` (cxcywh norm)        | (1, 200, 4)             | float32, [0,1] |
| Fused scores                      | (1, 200)                | float32, [0,1] |
| After threshold (N=3)             | 3 surviving queries     |                |
| `masks_logits` (after interp+sig) | (3, 1, 480, 640)        | float32, [0,1] |
| `masks` (binary)                  | (3, 1, 480, 640)        | bool           |
| `boxes` (xyxy, original px)       | (3, 4)                  | float32        |
| `scores`                          | (3,)                    | float32        |

## Normalization Sanity Check

```
input pixel:  127  (8-bit mid-grey)
uint8 scaled: 127
float32:      127 / 255 ≈ 0.498
normalized:   (0.498 - 0.5) / 0.5 ≈ -0.004  (near zero, expected)
```

SAM3 uses `mean=std=0.5` across all channels — this is **not** the CLIP
normalization (`mean=[0.48, 0.46, 0.41], std=[0.27, 0.26, 0.28]`) nor the
ImageNet normalization. It's a simple symmetric `[-1, 1]` mapping.
