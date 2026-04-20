# MobileCLIP text encoder

The SAM3 inference engine supports three MobileCLIP text-encoder variants
in addition to the standard CLIP encoder. They are wired through the
`sam3_text_encoder_iface` vtable and selected at conversion time via the
`.sam3` v4 header's `text_backbone` field.

## Variants

| Variant         | Layers | Width | Heads | MLP  | Ctx | RepMixer blocks |
|-----------------|--------|-------|-------|------|-----|-----------------|
| `clip`          | 24     | 1024  | 16    | 4096 | 32  | (none)          |
| `mobileclip_s0` | 6      | 512   | 8     | 2048 | 16  | {0, 5}          |
| `mobileclip_s1` | 12     | 512   | 8     | 2048 | 16  | (none)          |
| `mobileclip_l`  | 12     | 768   | 12    | 3072 | 16  | (none)          |

S0 uses RepMixer (depthwise conv + ConvFFN, MobileOne-style multi-branch
BN/conv at inference) for blocks 0 and 5; the other indices use standard
pre-norm transformer blocks. S1 and L use only standard blocks.

The vision side is unchanged for all three EfficientSAM3 checkpoints
(HIERA 32-layer ViT, 1008×1008 input).

## Conversion

The EfficientSAM3 `.pt` checkpoints in `models/` bundle both vision and
text encoders. Conversion is two-step:

```bash
# 1. .pt -> .safetensors (Python; remaps FB-internal prefixes to OSS)
python tools/pt_to_safetensors.py \
  models/efficient_sam3_image_encoder_mobileclip_s1_ctx16.pt \
  /tmp/mc_s1.safetensors

# 2. .safetensors -> .sam3 (C; writes v4 header with text_backbone tag)
sam3 convert -i /tmp/mc_s1.safetensors \
  -o models/sam3_mobileclip_s1.sam3 \
  --backbone hiera --variant sam3 \
  --text-backbone mobileclip_s1
```

Substitute `mobileclip_s0` / `mobileclip_l` (and the matching `.pt` file)
for the other variants. The default `--text-backbone clip` keeps existing
CLIP-based conversions unchanged.

## Runtime

No public API change. `sam3_load_model` reads `text_backbone` from the
v4 header and instantiates the right encoder. `sam3_set_text(ctx, "...")`
runs at the variant's native context length (16 tokens for MobileCLIP);
longer prompts are truncated.

## Implementation notes

- Variant configs live in `src/model/mobileclip_text.c`; the iface
  vtable lives in `src/model/text_encoder_iface.c`.
- RepMixer blocks compute
  `x + tm_layer_scale * (mixer(x) - norm(x))` for the token-mixer
  residual (note the subtraction), then
  `x + outer_layer_scale * convffn(x)` for the outer block residual.
- BN epsilon is 1e-5 throughout (PyTorch default).
- Per-block parity vs the PyTorch reference is verified by
  `tests/test_mobileclip_text.c` to within 1e-2 max-abs-error.
