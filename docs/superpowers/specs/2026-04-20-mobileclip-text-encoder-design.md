# MobileCLIP text encoder for EfficientSAM3

Date: 2026-04-20
Status: Design in progress

## Context

Three new EfficientSAM3 checkpoints landed in `models/`:

- `efficient_sam3_image_encoder_mobileclip_s0_ctx16.pt`
- `efficient_sam3_image_encoder_mobileclip_s1_ctx16.pt`
- `efficient_sam3_image_encoder_mobileclip2_l_ctx16.pt`

These bundle a vision backbone + a MobileCLIP text encoder run at a
16-token context (vs. the standard CLIP 77). An audit of the `.pt` state
dicts confirmed all three pair with the existing `SAM3_BACKBONE_HIERA`
vision encoder (32-layer ViT, 1008×1008, 447M params); only the **text**
side differs from what the C inference engine already supports.

**Motivation**: enabling MobileCLIP unlocks the EfficientSAM3 checkpoints
without adding any vision-side variants. The tradeoff is roughly
2-3% mAP for ~5-10× faster text encoding on mobile-class hardware, which
matters for use cases that re-embed text frequently (multi-prompt, video
relabeling, interactive workflows).

The existing C text encoder is hardcoded as a single 24-layer/1024-dim
CLIP architecture wired into `sam3_vl_backbone`. There is no
text-encoder variant infrastructure today; the image encoder already has
a clean union+enum dispatch (`SAM3_BACKBONE_HIERA/EFFICIENTVIT/TINYVIT`)
that this spec mirrors for the text side.

## Scope

### In scope

- New text-encoder variant enum: `SAM3_TEXT_CLIP`, `SAM3_TEXT_MOBILECLIP_S0`,
  `_S1`, `_L`. Stored in a new field of the `.sam3` header (format v4).
- New `sam3_mobileclip_text_encoder` struct with config-driven layer
  count / width / heads / mlp-dim, covering S0 / S1 / L from a single
  code path.
- RepMixer block (S0 only, block 0): conv-based token mixer + ConvFFN
  + layer scale. Implemented entirely with the existing
  `SAM3_OP_CONV2D` (groups + 1×11 kernel) and `SAM3_OP_BATCHNORM` ops;
  no new kernels required.
- Vtable-style dispatch at `vl_combiner` boundary
  (`sam3_text_encoder_iface`) so callers don't switch on backbone.
- Tokenizer parameterized to support arbitrary `ctx_len` (currently
  hardcoded at 77).
- Weight conversion pipeline: existing `tools/pt_to_safetensors.py` is
  extended to detect the variant and emit a sidecar JSON; `cli_convert`
  reads the sidecar and writes `text_backbone` into the v4 header. The
  C-side loader resolves source-style weight keys directly
  (`detector.backbone.language_backbone.encoder.*`) — no rename
  inside the converter.
- v4 header bump with backward-compat read (`version == 3` defaults
  `text_backbone = SAM3_TEXT_CLIP`).
- Per-block parity tests against the PyTorch reference for all three
  MobileCLIP variants, plus end-to-end golden embedding tests.

### Explicitly out of scope

- Training-time MobileOne reparameterization. Only the inference-fused
  form shipped in the checkpoint is supported.
- Quantization (Q8_0 etc.) of MobileCLIP weights. Defer.
- Joint vision-text fine-tuning. Load-only.
- New vision-side backbones — audit confirmed all three checkpoints use
  the existing HIERA encoder.
- Backward-compat **writing** of v3 files. Bump is one-way; old
  binaries refuse v4 cleanly with an unsupported-version error.

## Variant configs

All three MobileCLIP variants share the same code path, parameterized
by a config struct loaded from a `static const` table at module init:

| Variant            | n_layers | width | n_heads | mlp_dim | ctx_len | out_dim | repmixer block 0 |
|--------------------|----------|-------|---------|---------|---------|---------|------------------|
| `MOBILECLIP_S0`    | 6        | 512   | 8       | 2048    | 16      | 256     | yes              |
| `MOBILECLIP_S1`    | 12       | 512   | 8       | 2048    | 16      | 256     | no               |
| `MOBILECLIP_L`     | 12       | 768   | 12      | 3072    | 16      | 256     | no               |

Vocab size = 49408 (shared CLIP BPE). Output contract is `[ctx_len, 256]`
per-token plus `[256]` pooled, matching the existing CLIP encoder so
downstream code (`vl_combiner`, `sam3_processor`, mask decoder) is
untouched.

## Architecture

```
                              sam3_vl_backbone
                                     │
                          sam3_text_encoder_iface  ◄── vtable
                                ┌────┴────┐
                                ▼         ▼
                       sam3_text_encoder   sam3_mobileclip_text_encoder
                       (existing CLIP      (new — S0/S1/L,
                        24L/1024d/77ctx)    config-driven, RepMixer
                                            conditional in block 0)
```

### Standard MobileCLIP block (S0 blocks 1-5; S1/L all blocks)

Pre-norm transformer, no causal mask, no RoPE.

```
x_in
  ├─ ln1(x_in)                    LayerNorm, FP32 stats
  │  ├─ qkv_proj                  [width → 3*width], pre-fused in checkpoint
  │  ├─ split q,k,v
  │  ├─ scaled_dot_product        (no causal mask, no RoPE)
  │  └─ out_proj                  [width → width]
  │  → attn_out
  ├─ x_in + attn_out              → x_after_attn
  ├─ ln2(x_after_attn)
  │  ├─ fc1 [width → mlp_dim]
  │  ├─ GELU
  │  └─ fc2 [mlp_dim → width]
  │  → ffn_out
  └─ x_after_attn + ffn_out       → x_out
```

Reuses the existing `gh_multihead_attention` helper with `attn_mask =
NULL` and `rope = false`. The current CLIP path hardcodes a causal mask;
parameterize the helper call site (the underlying graph helper already
accepts `NULL`).

### RepMixer block (S0 block 0 only)

```
x_in: [seq_len, width=512]
  reshape → [N=1, C=512, H=1, W=seq_len]   (channels = width, spatial = seq)

  ── token_mixer (RepMixer) ──
    norm:   BatchNorm2d on C=512               (omit if BN folded into conv)
    mixer:  depthwise Conv2D, kernel=[1,11],
            padding=[0,5], groups=512
    layer_scale1: per-channel mul, gamma1: [512,1,1]
    + residual

  ── convffn ──
    fc1:    Conv2D 1×1, [512 → 2048]
    BatchNorm2d on 2048                        (omit if folded)
    GELU
    fc2:    Conv2D 1×1, [2048 → 512]
    BatchNorm2d on 512                         (omit if folded)
    layer_scale2: per-channel mul, gamma2: [512,1,1]
    + residual

  reshape → [seq_len, 512]
```

All ops already exist in the C codebase. Whether BN is folded into the
conv weights at export is determined by inspecting the actual `.pt`
state dict during implementation; the loader handles both cases by
checking for the BN parameter keys.

### Final stack (all variants)

```
final LayerNorm
  → projection_layer matmul (width → width, in-place projection from reference)
  → external 256-dim projector  (TextStudentEncoder.projector,
                                 width → 256, located adjacent to the
                                 language_backbone.encoder prefix)
  → output [ctx_len, 256]
  → pooled = output[eot_idx]  (idx = argmax(input_ids == EOT))
```

## File layout

### New files

- `src/model/mobileclip_text.h` — public type
  `struct sam3_mobileclip_text_encoder`, config struct, init/load/build/
  free declarations, the three variant configs as `static const` tables.
- `src/model/mobileclip_text.c` — implementation: weight loading, graph
  construction (standard pre-norm block + RepMixer block-0 branch),
  per-block evaluator mirroring `sam3_text_encoder_build_perblock`.
- `src/model/text_encoder_iface.h` — vtable struct
  `sam3_text_encoder_iface` (init/load/build/build_perblock/free),
  tagged union holding either concrete impl. Factory
  `sam3_text_encoder_iface_init(enum sam3_text_backbone, ...)`.
- `src/model/text_encoder_iface.c` — vtable wiring + factory dispatch.
- `tests/test_mobileclip_text.c` — golden parity tests for S0/S1/L.
- `scripts/dump_mobileclip_text_layers.py` — Python reference dumper for
  fixture generation, mirroring `scripts/dump_tracker_layers.py`.

### Modified files

- `include/sam3/sam3_types.h` — add `enum sam3_text_backbone`; add
  `text_backbone` to the public config struct holding `backbone_type` /
  `variant`.
- `src/core/weight.h` / `weight.c` — bump format version to v4, add
  `text_backbone` field to header struct, write/read it; v3 reads
  default `text_backbone = SAM3_TEXT_CLIP`.
- `src/model/vl_combiner.h` / `vl_combiner.c` — replace the embedded
  `sam3_text_encoder` field with a `sam3_text_encoder_iface*`.
  Init/load/build all go through the iface.
- `src/model/text_encoder.h` / `text_encoder.c` — expose existing
  functions through the iface vtable. Stop hardcoding the causal mask
  in the attention helper call (graph helper already accepts `NULL`).
- `src/model/tokenizer.h` / `tokenizer.c` — add
  `sam3_tokenizer_encode_ctx(tok, text, ctx_len, out)`. Existing
  `sam3_tokenizer_encode` becomes a thin wrapper passing
  `ctx_len = 77`. Output array length is caller-controlled.
- `src/model/sam3_processor.c` — pass the iface's `ctx_len` to the
  tokenizer instead of assuming 77.
- `tools/cli_convert.c` — read the new `text_backbone` value (from
  sidecar JSON or CLI flag); write it into the v4 header.
- `tools/pt_to_safetensors.py` — detect text-encoder variant from key
  prefixes; emit `*.meta.json` sidecar with
  `{text_backbone, image_backbone, image_size, ctx_len}`.

### Unchanged

All image encoder code, mask decoder, processor public API, BPE vocab
file, weight reader infrastructure.

## Header v4 layout

Bump version 3 → 4. Header grows from 48 to 52 bytes. New field:

```
offset 48: uint32_t text_backbone   (0=CLIP, 1=MOBILECLIP_S0,
                                     2=MOBILECLIP_S1, 3=MOBILECLIP_L)
```

Reader logic:

```c
if (version == 3) text_backbone = SAM3_TEXT_CLIP;       /* backcompat */
if (version == 4) text_backbone = read_u32(header + 48);
if (version  > 4) return SAM3_EUNSUPPORTED_VERSION;
```

Writer always emits v4. Old `.sam3` files keep working unchanged.

## Conversion pipeline

```
efficient_sam3_image_encoder_mobileclip_s1_ctx16.pt
  │
  ▼ tools/pt_to_safetensors.py  (extended to detect variant + emit sidecar)
  │
*.safetensors  +  *.meta.json
       {"text_backbone": "mobileclip_s1",
        "image_backbone": "hiera",
        "image_size": 1008,
        "ctx_len": 16}
  │
  ▼ tools/cli_convert -f safetensors --meta *.meta.json
  │
sam3_mobileclip_s1.sam3   (v4 header, text_backbone = MOBILECLIP_S1)
```

### Source key map (standard blocks)

```
language_backbone.encoder.embedding_layer.weight
  → token_embedding                  [49408, width]

language_backbone.encoder.positional_embedding.pos_embed.pos_embed
  → pos_embed                        [77, width]   (slice to ctx_len at build)

language_backbone.encoder.transformer.{i}.pre_norm_mha.0.{weight,bias}
  → layers[i].ln1.{w,b}              [width]

language_backbone.encoder.transformer.{i}.pre_norm_mha.1.qkv_proj.{weight,bias}
  → layers[i].qkv.{w,b}              [3*width, width] / [3*width]
  (already fused — no fuse_3 dance needed, unlike the CLIP loader)

language_backbone.encoder.transformer.{i}.pre_norm_mha.1.out_proj.{weight,bias}
  → layers[i].out.{w,b}              [width, width]

language_backbone.encoder.transformer.{i}.pre_norm_ffn.0.{weight,bias}
  → layers[i].ln2.{w,b}

language_backbone.encoder.transformer.{i}.pre_norm_ffn.1.{weight,bias}
  → layers[i].fc1.{w,b}              [mlp_dim, width]

language_backbone.encoder.transformer.{i}.pre_norm_ffn.4.{weight,bias}
  → layers[i].fc2.{w,b}              [width, mlp_dim]

language_backbone.encoder.final_layer_norm.{weight,bias}
  → final_ln.{w,b}

language_backbone.encoder.projection_layer
  → proj                             [width, width]
```

The `pre_norm_ffn.1` and `pre_norm_ffn.4` numbering (skipping 2, 3)
reflects the reference's `nn.Sequential(LN, Linear, GELU, Dropout, Linear)`
layout; GELU/Dropout slots have no parameters.

The external 256-dim projector lives outside `language_backbone.encoder.*`
in the `.pt`, in the `TextStudentEncoder.projector` Linear. Likely key
prefix is `detector.backbone.language_backbone.projector.*` (peer of
`encoder` under `language_backbone`) but exact key must be confirmed
by greping the state dict during implementation.

### RepMixer key map (S0 block 0 only)

The audit reported the following structure but did **not** drill into
the leaf-level field names:

```
language_backbone.encoder.transformer.0.layer_scale         [512,1,1]
language_backbone.encoder.transformer.0.token_mixer.layer_scale
language_backbone.encoder.transformer.0.token_mixer.norm.*  (BatchNorm2d)
language_backbone.encoder.transformer.0.token_mixer.mixer.* (depthwise conv,
                                                             possibly with
                                                             multi-branch
                                                             MobileOne keys)
language_backbone.encoder.transformer.0.convffn.conv.*
language_backbone.encoder.transformer.0.convffn.fc1.*       (1×1 conv +
                                                             optional BN)
language_backbone.encoder.transformer.0.convffn.act.*       (GELU, no params)
language_backbone.encoder.transformer.0.convffn.fc2.*       (1×1 conv +
                                                             optional BN)
```

The leaf field names (`weight`, `bias`, `running_mean`, `running_var`,
or MobileOne's `reparam_conv.weight` after reparameterization) and
whether BN is folded into conv at export must be confirmed at first
checkpoint inspection during implementation. Loader treats BN params
as optional (skip the BN op if its keys are absent, indicating
fold-into-conv).

## Tokenizer changes

Add `sam3_tokenizer_encode_ctx(tok, text, ctx_len, out)`. Internal
logic identical (BPE merges, lowercase, SOT prefix, EOT padding) but
output array length is caller-controlled. Existing
`sam3_tokenizer_encode` becomes a thin wrapper passing `ctx_len = 77`.

Truncation: when text exceeds `ctx_len - 2` BPE tokens, keep the first
`ctx_len - 2` content tokens, prepend SOT, append EOT. Matches
reference `SimpleTokenizer` behavior at ctx=16.

Pooled-output index: `argmax(input_ids == EOT)` (same as existing CLIP
path, smaller search range).

## Loader integration & public API

`sam3_load_model` reads the v4 header, picks `text_backbone`, calls
`sam3_text_encoder_iface_init(text_backbone, ...)`. Public API surface
is **unchanged** — callers still receive a `sam3_ctx*` and call
`sam3_processor_set_text(ctx, "a cat")`. No new public symbols beyond
the new enum values.

CLI tools (`sam3_run`, `sam3_video`) need zero code changes; they load
whatever's in the `.sam3` file. Variant selection happens at conversion
time, not at runtime.

## Testing strategy

Three layers, all under `tests/`:

1. **Unit tests** — `test_mobileclip_text.c`: weight loader resolves
   all expected keys for each of S0/S1/L; graph builds without error;
   output shapes match `[ctx_len, 256]` and pooled `[256]`.
2. **Per-block parity** — `dump_mobileclip_text_layers.py` dumps
   PyTorch reference activations after each block; the C test loads
   each, runs the same block, asserts max-abs-diff < 1e-3 (FP16
   tolerance) or 1e-5 (FP32). Three fixture sets (one per variant).
3. **End-to-end golden** — fixed prompt ("a person riding a bike"),
   assert final pooled embedding matches reference within tolerance
   for each variant. Ties into `test_text_prompt.c`.

Tokenizer at ctx=16 gets its own test in `test_tokenizer.c`
(truncation, EOT placement, padding). All new tests added to CTest.

## Risks and open questions

- **External 256-dim projector key location**: not visible in the
  `language_backbone.encoder.*` prefix. Likely
  `detector.backbone.language_backbone.projector.*` based on the
  reference's `TextStudentEncoder.projector` field, but must be
  confirmed during implementation. If naming diverges, the loader
  needs a small per-variant lookup table.
- **BN folding in MobileOne blocks**: the `.pt` may ship pre-folded
  (no BN params) or unfolded (BN params present). Loader must handle
  both. Mitigation: per-key existence check at load time, conditional
  graph emission.
- **Position embedding interpolation vs slicing**: reference uses
  `resize_pos_embed(16)` which is a slice for the down-from-77 case
  (no interpolation needed when target ≤ source). Confirm at
  implementation that pure slicing matches reference output.
- **Non-causal attention helper**: the underlying graph helper
  accepts `attn_mask = NULL`, but the call site in the existing
  CLIP encoder builds a causal mask unconditionally. Refactoring
  must keep CLIP behavior bit-identical (no functional change to
  the existing path).
