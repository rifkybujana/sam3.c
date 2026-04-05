# SAM3 Python Reference Implementation

This directory documents Facebook's official SAM3 reference implementation
(`reference/sam3/`, upstream `facebookresearch/sam3`). These documents describe
what the Python code does, the exact shapes at each stage, and where the
canonical definitions live. They are the source of truth for the C port in this
repository.

All paths in these documents are relative to `/Users/rbisri/Documents/sam3/`.
The Python package root is `reference/sam3/sam3/`.

## What SAM3 Does

SAM3 takes an image and a short text phrase (e.g. `"a red apple"`) and returns
per-instance segmentation masks, bounding boxes, and confidence scores for
every object in the image that matches the phrase. It is an open-vocabulary,
promptable segmenter — the text prompt can refer to any noun phrase, not a
fixed class list.

```
PIL.Image + "a red apple"
         |
         v
   +------------+
   | Sam3Image  |   848M parameters
   +------------+
         |
         v
{ masks: [N, H, W] bool,
  boxes: [N, 4] xyxy,
  scores: [N] in [0, 1] }
```

## Architecture at a Glance

```
               IMAGE PATH                          TEXT PATH
   +----------------------------+      +----------------------------+
   | Preprocess (1008x1008)     |      | Tokenize (32 BPE tokens)   |
   +----------------------------+      +----------------------------+
                 |                                    |
                 v                                    v
   +----------------------------+      +----------------------------+
   | ViT-L/14 (32 blocks)       |      | Text Transformer           |
   | embed_dim=1024             |      | (24 blocks, width=1024)    |
   +----------------------------+      +----------------------------+
                 |                                    |
                 v                                    v
   +----------------------------+      +----------------------------+
   | Dual ViTDet Neck (FPN)     |      | Linear 1024 -> 256         |
   | 4 scales @ d_model=256     |      | (seq_len, B, 256)          |
   +----------------------------+      +----------------------------+
                 |                                    |
                 +----------------+  +----------------+
                                  |  |
                                  v  v
                       +----------------------------+
                       | Transformer Encoder        |   6 layers
                       | (prompt cross-attends      |   d_model=256
                       |  to image features)        |   heads=8
                       +----------------------------+
                                     |
                                     v
                       +----------------------------+
                       | Transformer Decoder        |   6 layers
                       | 200 object queries +       |   box refinement
                       | presence token             |
                       +----------------------------+
                                     |
                                     v
                       +----------------------------+
                       | Segmentation Head          |   pixel decoder
                       | (mask @ pixel features)    |   144x144 masks
                       +----------------------------+
                                     |
                                     v
                            { pred_logits,
                              pred_boxes,
                              pred_masks,
                              presence_logit_dec }
```

## Entry Points

The public API is two-layered: `build_sam3_image_model()` constructs the
`Sam3Image` nn.Module, and `Sam3Processor` wraps it with pre/post-processing.

### Canonical "hello world"

From `reference/sam3/README.md:121-137`:

```python
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()           # load weights from HF
processor = Sam3Processor(model)

image = Image.open("cat.jpg")
state = processor.set_image(image)         # runs vision backbone, caches features
output = processor.set_text_prompt("cat", state)  # runs text + fusion + decode

masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
```

The two-phase API exists because the vision backbone is the expensive part
(~848M params minus text path). `set_image()` runs it once and caches the
result in `state["backbone_out"]`. Subsequent calls to `set_text_prompt()` or
`add_geometric_prompt()` reuse that cache.

### Inference State Dictionary

The `state` dict threaded through these calls accumulates:

| Key                          | Set by              | Shape / type                |
|------------------------------|---------------------|-----------------------------|
| `original_height`            | `set_image`         | int                         |
| `original_width`             | `set_image`         | int                         |
| `backbone_out`               | `set_image`         | dict (see below)            |
| `backbone_out.backbone_fpn`  | `set_image`         | list of 4 tensors           |
| `backbone_out.vision_pos_enc`| `set_image`         | list of 4 tensors           |
| `backbone_out.language_features` | `set_text_prompt` | (seq, B, 256)             |
| `backbone_out.language_mask` | `set_text_prompt`   | (B, seq) bool               |
| `backbone_out.language_embeds` | `set_text_prompt` | (seq, B, 1024)              |
| `geometric_prompt`           | processor internal  | Prompt object (boxes/points)|
| `masks`                      | `_forward_grounding`| (N, H, W) bool              |
| `masks_logits`               | `_forward_grounding`| (N, H, W) float in [0, 1]   |
| `boxes`                      | `_forward_grounding`| (N, 4) xyxy, orig coords    |
| `scores`                     | `_forward_grounding`| (N,) float in [0, 1]        |

## Model Configuration

All hyperparameters are hardcoded in `reference/sam3/sam3/model_builder.py`.
There is no config file — the builder functions are the config.

| Component            | Param              | Value       | Source (file:line)               |
|----------------------|--------------------|-------------|----------------------------------|
| Image                | input size         | 1008x1008   | `model_builder.py:80`            |
| Image                | patch size         | 14          | `model_builder.py:82`            |
| Image                | mean / std         | 0.5 / 0.5   | `sam3_image_processor.py:26`     |
| ViT backbone         | embed_dim          | 1024        | `model_builder.py:83`            |
| ViT backbone         | depth              | 32          | `model_builder.py:84`            |
| ViT backbone         | num_heads          | 16          | `model_builder.py:85`            |
| ViT backbone         | mlp_ratio          | 4.625       | `model_builder.py:86`            |
| ViT backbone         | window_size        | 24          | `model_builder.py:96`            |
| ViT backbone         | global attn blocks | (7,15,23,31)| `model_builder.py:92`            |
| FPN neck             | d_model            | 256         | `model_builder.py:113`           |
| FPN neck             | scale_factors      | [4,2,1,0.5] | `model_builder.py:114`           |
| Text encoder         | context_length     | 32          | `model_builder.py:500-509`       |
| Text encoder         | width              | 1024        | `model_builder.py:506`           |
| Text encoder         | heads              | 16          | `model_builder.py:507`           |
| Text encoder         | layers             | 24          | `model_builder.py:508`           |
| Text encoder         | vocab_size         | 49408       | `text_encoder_ve.py:264`         |
| Transformer encoder  | layers             | 6           | `model_builder.py:154`           |
| Transformer encoder  | d_model            | 256         | `model_builder.py:129`           |
| Transformer encoder  | num_heads          | 8           | `model_builder.py:137`           |
| Transformer encoder  | dim_feedforward    | 2048        | `model_builder.py:130`           |
| Transformer decoder  | layers             | 6           | `model_builder.py:184`           |
| Transformer decoder  | num_queries        | 200         | `model_builder.py:185`           |
| Transformer decoder  | presence_token     | True        | `model_builder.py:198`           |
| Segmentation head    | upsampling_stages  | 3           | `model_builder.py:235`           |
| Segmentation head    | hidden_dim         | 256         | `model_builder.py:234`           |

## Checkpoint

- HuggingFace repo: `facebook/sam3` → `sam3.pt`
- Alternate: `facebook/sam3.1` → `sam3.1_multiplex.pt` (Object Multiplex variant)
- Format: PyTorch state dict. Keys relevant to image inference are prefixed
  `detector.*`; tracker keys are `tracker.*` and can be ignored.
- Loaded by `_load_checkpoint` at `model_builder.py:539`.

## Document Map

| File                                              | Covers                                   |
|---------------------------------------------------|------------------------------------------|
| [architecture.md](architecture.md)                | End-to-end forward pass, stage-by-stage  |
| [image-encoder.md](image-encoder.md)              | ViT backbone + FPN neck                  |
| [text-encoder.md](text-encoder.md)                | Tokenizer + text transformer             |
| [fusion.md](fusion.md)                            | Encoder + decoder, cross-attention       |
| [mask-head.md](mask-head.md)                      | Pixel decoder + mask prediction          |
| [processor.md](processor.md)                      | Preprocessing, postprocessing, API       |

## Source File Map

Relative to `reference/sam3/sam3/`:

| File                             | Role                                            |
|----------------------------------|-------------------------------------------------|
| `model_builder.py`               | Constructs every component; hyperparameters     |
| `model/sam3_image.py`            | `Sam3Image` — forward orchestrator              |
| `model/sam3_image_processor.py`  | `Sam3Processor` — user-facing API               |
| `model/vl_combiner.py`           | `SAM3VLBackbone` wraps vision + text            |
| `model/vitdet.py`                | `ViT` image backbone                            |
| `model/necks.py`                 | `Sam3DualViTDetNeck` FPN                        |
| `model/position_encoding.py`     | Sine 2D position embeddings                     |
| `model/text_encoder_ve.py`       | `VETextEncoder`, `TextTransformer`              |
| `model/tokenizer_ve.py`          | `SimpleTokenizer` (CLIP BPE)                    |
| `model/encoder.py`               | `TransformerEncoderFusion` + layer              |
| `model/decoder.py`               | `TransformerDecoder` + layers                   |
| `model/maskformer_segmentation.py` | `PixelDecoder`, `UniversalSegmentationHead`   |
| `model/geometry_encoders.py`     | `SequenceGeometryEncoder` (boxes/points)        |
| `model/model_misc.py`            | `MLP`, `DotProductScoring`, attention wrappers  |
| `model/box_ops.py`               | Box format conversions                          |
| `assets/bpe_simple_vocab_16e6.txt.gz` | CLIP BPE merges                            |
