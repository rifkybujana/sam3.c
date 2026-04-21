# MobileCLIP key audit — 2026-04-20

Resolutions of TBD items from the design spec, derived from inspecting
the three .pt checkpoints in `models/`.

## Standard-block keys (S1/L, S0 blocks 1-4)

Verified prefix: `detector.backbone.language_backbone.encoder.transformer.<N>`

| Sub-key                              | Shape (S1)       | Shape (L)        |
|--------------------------------------|------------------|------------------|
| `pre_norm_mha.0.weight` (LN1 w)      | `(512,)`         | `(768,)`         |
| `pre_norm_mha.0.bias` (LN1 b)        | `(512,)`         | `(768,)`         |
| `pre_norm_mha.1.qkv_proj.weight`     | `(1536, 512)`    | `(2304, 768)`    |
| `pre_norm_mha.1.qkv_proj.bias`       | `(1536,)`        | `(2304,)`        |
| `pre_norm_mha.1.out_proj.weight`     | `(512, 512)`     | `(768, 768)`     |
| `pre_norm_mha.1.out_proj.bias`       | `(512,)`         | `(768,)`         |
| `pre_norm_ffn.0.weight` (LN2 w)      | `(512,)`         | `(768,)`         |
| `pre_norm_ffn.0.bias` (LN2 b)        | `(512,)`         | `(768,)`         |
| `pre_norm_ffn.1.weight` (fc1 w)      | `(2048, 512)`    | `(3072, 768)`    |
| `pre_norm_ffn.1.bias` (fc1 b)        | `(2048,)`        | `(3072,)`        |
| `pre_norm_ffn.4.weight` (fc2 w)      | `(512, 2048)`    | `(768, 3072)`    |
| `pre_norm_ffn.4.bias` (fc2 b)        | `(512,)`         | `(768,)`         |

S1 has 12 standard blocks (0–11), all identical structure.
L has 12 standard blocks (0–11), all identical structure, width scaled to 768.

## Embedding/finalisation keys

All three variants share the same key names under
`detector.backbone.language_backbone.encoder.*`:

| Key                                                                    | Shape (S0/S1)    | Shape (L)        |
|------------------------------------------------------------------------|------------------|------------------|
| `embedding_layer.weight`                                               | `(49408, 512)`   | `(49408, 768)`   |
| `positional_embedding.pos_embed.pos_embed`                             | `(1, 1, 77, 512)`| `(1, 1, 77, 768)`|
| `final_layer_norm.weight`                                              | `(512,)`         | `(768,)`         |
| `final_layer_norm.bias`                                                | `(512,)`         | `(768,)`         |
| `projection_layer`                                                     | `(512, 512)`     | `(768, 768)`     |

Note: `positional_embedding.pos_embed.pos_embed` is a doubly-nested name —
the sub-module is `.pos_embed` and the leaf parameter is also `.pos_embed`.

## External 256-dim projector

Confirmed key: `detector.backbone.language_backbone.projector.weight`
Shape: `[256, 512]` (S0/S1), `[256, 768]` (L)

Companion bias: `detector.backbone.language_backbone.projector.bias`
Shape: `(256,)` for all variants.

The projector sits one level above the encoder (at `language_backbone.*`,
not `language_backbone.encoder.*`).

## RepMixer keys (S0 only — blocks 0 AND 5)

**Correction from design spec**: the design spec anticipated RepMixer only
in block 0. In the S0 checkpoint, blocks **0 and 5** are RepMixer blocks;
blocks 1–4 are standard transformer blocks. S0 has **6 transformer blocks
total** (indices 0–5), confirmed by `inspect_mobileclip_pt.py` and by the
reference Python (`_create_student_text_encoder` instantiates a 6-layer
RepMixer-augmented MobileCLIP for S0).

Verbatim inspector output for block 0 and block 5 (lines 7–34 and 83–110
of /tmp/s0_keys.txt):

```
detector.backbone.language_backbone.encoder.transformer.0.convffn.conv.bn.bias	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.0.convffn.conv.bn.num_batches_tracked	()	torch.int64
detector.backbone.language_backbone.encoder.transformer.0.convffn.conv.bn.running_mean	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.0.convffn.conv.bn.running_var	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.0.convffn.conv.bn.weight	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.0.convffn.conv.conv.weight	(512, 1, 1, 11)	torch.float32
detector.backbone.language_backbone.encoder.transformer.0.convffn.fc1.bias	(2048,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.0.convffn.fc1.weight	(2048, 512, 1, 1)	torch.float32
detector.backbone.language_backbone.encoder.transformer.0.convffn.fc2.bias	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.0.convffn.fc2.weight	(512, 2048, 1, 1)	torch.float32
detector.backbone.language_backbone.encoder.transformer.0.layer_scale	(512, 1, 1)	torch.float32
detector.backbone.language_backbone.encoder.transformer.0.token_mixer.layer_scale	(512, 1, 1)	torch.float32
detector.backbone.language_backbone.encoder.transformer.0.token_mixer.mixer.rbr_conv.0.bn.bias	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.0.token_mixer.mixer.rbr_conv.0.bn.num_batches_tracked	()	torch.int64
detector.backbone.language_backbone.encoder.transformer.0.token_mixer.mixer.rbr_conv.0.bn.running_mean	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.0.token_mixer.mixer.rbr_conv.0.bn.running_var	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.0.token_mixer.mixer.rbr_conv.0.bn.weight	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.0.token_mixer.mixer.rbr_conv.0.conv.weight	(512, 1, 1, 11)	torch.float32
detector.backbone.language_backbone.encoder.transformer.0.token_mixer.mixer.rbr_skip.bias	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.0.token_mixer.mixer.rbr_skip.num_batches_tracked	()	torch.int64
detector.backbone.language_backbone.encoder.transformer.0.token_mixer.mixer.rbr_skip.running_mean	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.0.token_mixer.mixer.rbr_skip.running_var	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.0.token_mixer.mixer.rbr_skip.weight	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.0.token_mixer.norm.rbr_skip.bias	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.0.token_mixer.norm.rbr_skip.num_batches_tracked	()	torch.int64
detector.backbone.language_backbone.encoder.transformer.0.token_mixer.norm.rbr_skip.running_mean	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.0.token_mixer.norm.rbr_skip.running_var	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.0.token_mixer.norm.rbr_skip.weight	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.5.convffn.conv.bn.bias	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.5.convffn.conv.bn.num_batches_tracked	()	torch.int64
detector.backbone.language_backbone.encoder.transformer.5.convffn.conv.bn.running_mean	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.5.convffn.conv.bn.running_var	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.5.convffn.conv.bn.weight	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.5.convffn.conv.conv.weight	(512, 1, 1, 11)	torch.float32
detector.backbone.language_backbone.encoder.transformer.5.convffn.fc1.bias	(2048,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.5.convffn.fc1.weight	(2048, 512, 1, 1)	torch.float32
detector.backbone.language_backbone.encoder.transformer.5.convffn.fc2.bias	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.5.convffn.fc2.weight	(512, 2048, 1, 1)	torch.float32
detector.backbone.language_backbone.encoder.transformer.5.layer_scale	(512, 1, 1)	torch.float32
detector.backbone.language_backbone.encoder.transformer.5.token_mixer.layer_scale	(512, 1, 1)	torch.float32
detector.backbone.language_backbone.encoder.transformer.5.token_mixer.mixer.rbr_conv.0.bn.bias	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.5.token_mixer.mixer.rbr_conv.0.bn.num_batches_tracked	()	torch.int64
detector.backbone.language_backbone.encoder.transformer.5.token_mixer.mixer.rbr_conv.0.bn.running_mean	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.5.token_mixer.mixer.rbr_conv.0.bn.running_var	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.5.token_mixer.mixer.rbr_conv.0.bn.weight	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.5.token_mixer.mixer.rbr_conv.0.conv.weight	(512, 1, 1, 11)	torch.float32
detector.backbone.language_backbone.encoder.transformer.5.token_mixer.mixer.rbr_skip.bias	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.5.token_mixer.mixer.rbr_skip.num_batches_tracked	()	torch.int64
detector.backbone.language_backbone.encoder.transformer.5.token_mixer.mixer.rbr_skip.running_mean	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.5.token_mixer.mixer.rbr_skip.running_var	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.5.token_mixer.mixer.rbr_skip.weight	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.5.token_mixer.norm.rbr_skip.bias	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.5.token_mixer.norm.rbr_skip.num_batches_tracked	()	torch.int64
detector.backbone.language_backbone.encoder.transformer.5.token_mixer.norm.rbr_skip.running_mean	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.5.token_mixer.norm.rbr_skip.running_var	(512,)	torch.float32
detector.backbone.language_backbone.encoder.transformer.5.token_mixer.norm.rbr_skip.weight	(512,)	torch.float32
```

**BN folding decision**: BN running stats are PRESENT (`running_mean`,
`running_var`, `num_batches_tracked` all ship with the checkpoint).
BN ops are NOT folded into conv weights at export. The C loader must
emit BN ops for every RepMixer block, using these running stats to
perform inference-time normalisation.

Key observations for the RepMixer block structure:
- `token_mixer.mixer.rbr_conv.0.conv.weight`: `(512, 1, 1, 11)` — depthwise
  conv with kernel 11, stored as `(C_out, C_in, H, W)` = `(512, 1, 1, 11)`.
  This is a 1D causal depthwise conv over sequence dimension.
- `token_mixer.mixer.rbr_skip.*`: BN on the identity (skip) branch.
- `token_mixer.norm.rbr_skip.*`: BN on the norm branch (no conv weight,
  just a standalone BN — this is the RepBN identity path).
- `convffn.fc1.weight`: `(2048, 512, 1, 1)` — pointwise conv (not Linear).
- `convffn.fc2.weight`: `(512, 2048, 1, 1)` — pointwise conv (not Linear).
- `layer_scale`: `(512, 1, 1)` — per-channel scale for the block output.
- `token_mixer.layer_scale`: `(512, 1, 1)` — per-channel scale for mixer
  output.

## L vs S1/S0 prefix

MobileCLIP2-L uses the exact same `detector.backbone.language_backbone.encoder`
prefix as S1 and S0; the `mobileclip2` distinction only appears in the
checkpoint filename, not in any state-dict key.
