# Image Encoder: ViT Backbone + FPN Neck

The image encoder transforms a 1008x1008 RGB tensor into a 4-level feature
pyramid at d_model=256. Only the lowest-resolution level (36x36) goes to the
fusion transformer; the higher-resolution levels feed the pixel decoder that
produces output masks.

## Files

- Backbone: `reference/sam3/sam3/model/vitdet.py`
- Neck: `reference/sam3/sam3/model/necks.py`
- Position encoding: `reference/sam3/sam3/model/position_encoding.py`
- Assembled in: `reference/sam3/sam3/model_builder.py:77-117`

## Pipeline

```
(B, 3, 1008, 1008)
        |
        v
   PatchEmbed (Conv2d kernel=14 stride=14)
        |
        v
(B, 72, 72, 1024)     -- HWC layout, 72*72 = 5184 patches
        |
        + abs_pos_embed (tiled from 336/14=24 to 72)
        |
        v
   ln_pre (LayerNorm)
        |
        v
   32 x Block                          -- window attn (size 24) on 28 blocks
        |                              -- global attn on blocks 7, 15, 23, 31
        |                              -- RoPE 2D in every block
        v
   ln_post (applied after block 31)
        |
        v
(B, 72, 72, 1024)  ->  permute to (B, 1024, 72, 72)
        |
        v
   Sam3DualViTDetNeck            -- 4 parallel conv stacks
        |
        v
   [ (B, 256, 288, 288)    -- scale 4.0
     (B, 256, 144, 144)    -- scale 2.0
     (B, 256,  72,  72)    -- scale 1.0
     (B, 256,  36,  36) ]  -- scale 0.5  (discarded by scalp=1)
        |
        v
   SAM3VLBackbone (scalp=1) drops the 36x36 level
        |
        v
   backbone_fpn = [
     (B, 256, 288, 288),    -- scale 4.0 (for mask head)
     (B, 256, 144, 144),    -- scale 2.0 (for mask head)
     (B, 256,  72,  72),    -- scale 1.0 (for encoder + mask head)
   ]
   vision_pos_enc = [matching sine 2D pos embeddings, same shapes]
```

## Patch Embedding (`vitdet.py:346-383`)

A single `Conv2d(3, 1024, kernel_size=14, stride=14)` followed by an NCHW→NHWC
permutation.

```python
self.proj = nn.Conv2d(3, 1024, kernel_size=14, stride=14, bias=False)
x = self.proj(x)          # (B, 1024, 72, 72)
x = x.permute(0, 2, 3, 1) # (B, 72, 72, 1024)
```

Bias is disabled (`bias_patch_embed=False`). SAM3 keeps the HWC layout through
every ViT block; only the neck converts back to NCHW.

## Absolute Position Embedding (`vitdet.py:861-875, 967-975`)

Pretrained at 336x336 (24x24 patches) with a CLS token slot, then tiled to
72x72 at inference time via `get_abs_pos(..., tiling=True)`.

- `self.pos_embed`: shape `(1, 1 + 24*24, 1024)` (pretrain with CLS token)
- At inference: CLS slot stripped, remaining `(1, 576, 1024)` tiled to
  `(1, 72, 72, 1024)` and added to the patch grid.
- `retain_cls_token=False` → no CLS token kept during forward.

## Transformer Block (`vitdet.py:635-741`)

Each of the 32 blocks is standard pre-norm with LayerScale disabled
(`init_values=None`), drop path rate linearly scaled 0 → 0.1:

```python
shortcut = x
x = norm1(x)
if window_size > 0:
    x, pad_hw = window_partition(x, window_size=24)
x = attn(x)                 # MHA with 2D RoPE
if window_size > 0:
    x = window_unpartition(x, window_size, pad_hw, (H, W))
x = shortcut + drop_path(x)
x = x + drop_path(mlp(norm2(x)))
```

**Attention variant:**
- `num_heads=16`, `head_dim=64`, `scale=1/8`
- Linear qkv: `nn.Linear(1024, 3072, bias=True)` (`vitdet.py:433`)
- 2D RoPE applied to Q and K (`use_rope=True`, `use_interp_rope=True`)
  with `rope_pt_size=(24, 24)`.
- No relative position bias (`use_rel_pos=False`, `rel_pos_blocks=()`).
- Output projection: `nn.Linear(1024, 1024)` (`vitdet.py:434`).

**MLP:**
- Hidden dim: `int(1024 * 4.625) = 4736`
- `Linear(1024, 4736) → GELU → Linear(4736, 1024)`

**Windowed vs global attention.** 28 of the 32 blocks use window attention
with a 24x24 window (`window_partition` slices the 72x72 patch grid into
non-overlapping 24x24 tiles; 72/24=3, so 9 tiles per image). The 4 blocks at
indices 7, 15, 23, 31 perform full global attention over all 5184 patches.

## Layer Norm (`vitdet.py:929-930, 990`)

- `ln_pre`: applied once before block 0 (`eps=1e-5`)
- `ln_post`: applied once, at the end of block 31 only (inside the forward
  loop when `i == full_attn_ids[-1]`)

## Feature Pyramid Neck (`necks.py:15-127`)

`Sam3DualViTDetNeck` takes the last ViT output `(B, 1024, 72, 72)` and produces
four parallel projections at different spatial scales. The "dual" part of the
name refers to the optional SAM2 neck used for instance interactivity; for
pure text-prompted inference only the SAM3 branch is needed (set
`add_sam2_neck=False`, or ignore `sam2_backbone_out`).

Each of the four conv stacks transforms the input differently:

| Scale | Spatial output | First op                              | Intermediate C |
|-------|----------------|---------------------------------------|----------------|
| 4.0   | 288 x 288      | ConvT(1024→512, k=2,s=2) + GELU + ConvT(512→256, k=2,s=2) | 256 |
| 2.0   | 144 x 144      | ConvT(1024→512, k=2, s=2)             | 512            |
| 1.0   |  72 x 72       | (identity)                            | 1024           |
| 0.5   |  36 x 36       | MaxPool2d(k=2, s=2)                   | 1024           |

Every stack ends with the same 1x1 → 3x3 projection to d_model=256:

```python
conv_1x1 = nn.Conv2d(intermediate_C, 256, kernel_size=1, bias=True)
conv_3x3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
```

A 2D sine positional encoding is generated per scale with
`PositionEmbeddingSine(num_pos_feats=256, temperature=10000, normalize=True)`
(`position_encoding.py`, `model_builder.py:66-74`). The positional tensor has
the same shape as the feature map at that scale.

## Scalp (`vl_combiner.py:91-96`)

`SAM3VLBackbone` is constructed with `scalp=1` in `model_builder.py:122`.
Immediately after the neck runs, `scalp=1` drops the lowest-resolution FPN
level (the 36x36 one) from both `sam3_features` and `sam3_pos`:

```python
if self.scalp > 0:
    sam3_features = sam3_features[: -self.scalp]
    sam3_pos      = sam3_pos[: -self.scalp]
```

All downstream consumers see only the 3 remaining levels: 288², 144², 72².

## Output Consumption

The FPN output list is stored in `backbone_out["backbone_fpn"]` (post-scalp),
ordered from highest to lowest resolution. Two downstream consumers use
different slices:

1. **Transformer encoder** (`sam3_image.py:115-133`):
   reads only `backbone_fpn[-num_feature_levels:]`. With `num_feature_levels=1`
   and the 36x36 level already dropped, this is `backbone_fpn[-1]` — the
   72x72 level. It gets flattened:

   ```
   (B, 256, 72, 72) -> flatten spatial -> (5184, B, 256)
   ```

2. **Pixel decoder / mask head** (`maskformer_segmentation.py:215-231`):
   reads all 3 levels. The encoder output replaces the 72x72 level, then
   upsamples 72 → 144 → 288, adding each higher-res level. Output pixel
   embedding is `(B, 256, 288, 288)`.

## Shape Table

| Stage                      | Tensor shape           | Layout |
|----------------------------|------------------------|--------|
| Raw image                  | (B, 3, 1008, 1008)     | NCHW   |
| After PatchEmbed           | (B, 72, 72, 1024)      | NHWC   |
| After 32 ViT blocks        | (B, 72, 72, 1024)      | NHWC   |
| After permute in trunk out | (B, 1024, 72, 72)      | NCHW   |
| FPN level 0 (scale 4.0)    | (B, 256, 288, 288)     | NCHW   |
| FPN level 1 (scale 2.0)    | (B, 256, 144, 144)     | NCHW   |
| FPN level 2 (scale 1.0)    | (B, 256, 72, 72)       | NCHW   |
| FPN level 3 (scale 0.5)    | (B, 256, 36, 36)       | NCHW (dropped by scalp=1) |
| Flattened for encoder      | (5184, B, 256)         | SBD    |

## Parameter Count

Approximate ViT-L/14 count at these settings:
- Patch embed: 14*14*3*1024 = 602k
- Per block: qkv 3.15M + proj 1.05M + MLP 9.7M + 2*LN = ~13.9M
- 32 blocks: ~445M
- Abs pos embed: 591k
- Total vision trunk: ~446M
- FPN neck (sam3 side only, no SAM2 clone):
  - Scale 4.0: 1024*512*4 + 512*256*4 + 256*256 + 256*256*9 ≈ 3.2M
  - Scale 2.0: 1024*512*4 + 512*256 + 256*256*9 ≈ 2.8M
  - Scale 1.0: 1024*256 + 256*256*9 ≈ 0.86M
  - Scale 0.5: same as 1.0 ≈ 0.86M
  - Neck total: ~7.7M

The vision path dominates the 848M total model size.
