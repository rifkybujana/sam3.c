# SAM 3.1 image path (weight pipeline + tri-neck)

Date: 2026-04-18
Status: Design approved

## Summary

Add first-class support for Facebook's SAM 3.1 checkpoint
(`sam3.1_multiplex.pt`, ~3.5 GB) to this C engine, scoped to the image
inference path only. When done, a user can run

```
sam3 segment -m models/sam3.1.sam3 -i input.jpg -t "cat"
```

and get masks that numerically match the Python reference within IoU
≥ 0.99. SAM 3 keeps working exactly as today — SAM 3.1 is a sibling
variant, selected by a new enum field in the `.sam3` header. Tracker,
memory attention, and video inference are explicitly out of scope here
and deferred to sub-projects 2–5.

## Scope

### In scope

- End-to-end conversion pipeline from the upstream `.pt` checkpoint to a
  new `.sam3` file marked as the `sam3.1` variant.
- Runtime support for the SAM 3.1 detector, which is architecturally
  identical to the SAM 3 detector except that the FPN neck has **three
  scales `[4×, 2×, 1×]` instead of four `[4×, 2×, 1×, 0.5×]`**. All other
  detector pieces (ViT backbone, CLIP text encoder, DETR encoder/decoder,
  geometry encoder, segmentation head) are unchanged.
- CLI integration: `sam3_cli segment` and `sam3_cli info` both learn the
  variant field without new flags on `segment`.
- Correctness: a parity test against a Python-reference baseline.

### Out of scope

- SAM 3.1 tracker transformer (`DecoupledTransformerDecoderLayerv2`,
  `SimpleRoPEAttention`, 8-head memory attention).
- Multiplex-aware mask-memory backbone (`out_dim=256`,
  `multiplex_count=16`, `starting_out_chan=4`, `input_channel_multiplier=2`).
- Object-pointer bank, `sincos_tpos_enc`, `use_maskmem_tpos_v2`,
  `no_obj_embed_spatial`, `add_output_suppression_embeddings`.
- Multiplex controller and joint multi-object forward pass.
- Detector/tracker association heuristics introduced in 3.1
  (`Sam3MultiplexTrackingWithInteractivity`).

Tracker weights from the checkpoint are still converted and stored in
the `.sam3` file, passthrough, so sub-project 2 can consume them without
a second conversion pass. The image-path loader simply ignores them.

## Architecture inputs

Source of truth for what SAM 3.1 actually is:

- `reference/sam3/sam3/model_builder.py` — `build_sam3_multiplex_video_model`,
  `_create_multiplex_tri_backbone`, `download_ckpt_from_hf(version="sam3.1")`
  (`reference/sam3/sam3/model_builder.py:920-1067`).
- `reference/sam3/RELEASE_SAM3p1.md` — paper/release notes for
  Object Multiplex.
- The HF `config.json` for `facebook/sam3.1` — confirms detector config
  equals SAM 3's and `num_feature_levels: 3` with
  `backbone_feature_sizes [[288,288],[144,144],[72,72]]`.

Findings that drive this spec:

- SAM 3.1 image detector is byte-for-byte SAM 3 detector **except** the
  neck drops the 0.5× level. `scale_factors` in HF config still lists
  four values for legacy reasons; the authoritative shape is
  `num_feature_levels=3`.
- HF distribution may or may not have pre-remapped the checkpoint from
  the in-house `sam3_model.*` / `sam2_predictor.*` prefixes to the OSS
  `detector.*` / `tracker.*` prefixes. Both shapes must be handled.
- Image size stays at 1008, patch size 14, ViT hidden 1024 with 32
  layers, CLIP text 1024/24L, DETR 6L/200 queries. No change to any of
  these.

## Weight-format changes

The `.sam3` header is already at `SAM3_WEIGHT_VERSION = 3` and its
48-byte layout is locked by `_Static_assert`. `reserved[0]` already
carries `backbone_type`. We reuse the remaining two 32-bit slots:

```c
/* src/core/weight.c, sam3_weight_write */
hdr.reserved[0] = (uint32_t)config->backbone_type;  /* existing */
hdr.reserved[1] = (uint32_t)config->variant;        /* new: 0=SAM3, 1=SAM3_1 */
hdr.reserved[2] = (uint32_t)config->n_fpn_scales;   /* new: 3 or 4 */
```

No version bump, no struct layout change, no `_Static_assert` touch.

Loader behavior (existing `sam3_weight_open`):

- `reserved[1] == 0, reserved[2] == 0`: old file predating SAM 3.1;
  treat as `variant = SAM3_VARIANT_SAM3`, `n_fpn_scales = 4` for
  backwards compatibility.
- `reserved[1] == 1`: SAM 3.1. Require `reserved[2] in {3, 4}`.
- `reserved[1] > 1`: return `SAM3_EMODEL` (unknown variant).

Public-API additions in `include/sam3/sam3_types.h`:

```c
enum sam3_variant {
        SAM3_VARIANT_SAM3   = 0,
        SAM3_VARIANT_SAM3_1 = 1,
};

struct sam3_model_config {
        /* existing fields unchanged */
        int            n_fpn_scales;   /* 3 or 4 */
        enum sam3_variant variant;
};
```

`sam3_model_config` is already part of the public API; the additions are
at the struct tail to keep binary compatibility with any callers that
stat-allocate one. (We do not promise binary compat across releases, but
layout-at-tail costs nothing and avoids surprises.)

## Converter pipeline

```
sam3.1_multiplex.pt
        |
        |  pt_to_safetensors.py  (Python, one-shot)
        |    - torch.load(weights_only=True)
        |    - unwrap {"model": ...} if present
        |    - remap sam3_model.* -> detector.*
        |    - remap sam2_predictor.* -> tracker.*
        v
sam3.1_multiplex.safetensors
        |
        |  sam3_convert -i ... -o ... --variant sam3.1
        |    - existing safetensors reader
        |    - existing rename_reader (handle_neck handles 3 or 4 levels)
        |    - existing conv_perm_reader (unchanged)
        |    - sam3_weight_write writes variant=1, n_fpn_scales=3
        v
sam3.1.sam3
```

### `tools/pt_to_safetensors.py` (new, ~80 LOC)

Mirrors the `needs_remap` logic in
`reference/sam3/sam3/model_builder.py:1209-1221`. Pseudocode:

```
ckpt = torch.load(path, map_location="cpu", weights_only=True)
if "model" in ckpt and isinstance(ckpt["model"], dict):
    ckpt = ckpt["model"]
if any(k.startswith(("sam3_model.", "sam2_predictor.")) for k in ckpt):
    ckpt = {rename(k): v for k, v in ckpt.items()}
safetensors.torch.save_file(ckpt, out_path)
```

Accept `--help`, `input.pt`, `output.safetensors`; exit non-zero on
missing input or incompatible keys. No C integration — this is a
pre-processing step, documented in the converter help text and README.

### `sam3_convert.c` changes

- New flag `--variant {sam3,sam3.1}`, default `sam3`.
- When `--variant sam3.1`:
  - `args.n_fpn_scales = 3`.
  - `config.variant = SAM3_VARIANT_SAM3_1`.
  - No backbone-default change (`--backbone hiera` still the default;
    SAM 3.1 uses the same ViT backbone).
- Print the variant in the conversion summary.

### `tools/weight_rename.c` changes

`handle_neck()` already processes each FPN level independently — the
tri-neck produces three `convs.i.*` sets instead of four, and the
existing per-level rename rules (`dconv_2x2{,_0,_1}`, `conv_1x1`,
`conv_3x3`) apply unchanged. No new rename rules are needed for the
detector. Tracker-specific rename rules (memory attention, maskmem,
multiplex controller) are **added as passthrough prefix remaps only**:
the weights end up in the `.sam3` file with predictable names but are
not consumed by any module in this sub-project. The full tracker
rename table is the subject of sub-project 2.

## Runtime changes

### `struct sam3_weight_header` (`src/core/weight.h`)

Add the variant/n_fpn_scales fields. Keep the struct 16-byte aligned.
`sam3_weight_load` initializes `ctx->config.variant` and
`ctx->config.n_fpn_scales` from the header, with the v2 fallback
described above.

### `src/model/necks.c` / `necks.h`

- `struct sam3_neck` gets `int n_scales` replacing the hard-coded 4.
- Allocation loop uses `n_scales`.
- Forward loop uses `n_scales`.
- `sam3_neck_load` reads only `fpn_layers.0..n_scales-1.*` from the
  weight map. The historical `sam2_fpn_layers.*` path is left untouched
  (tracker-only, not used by the image path).

No other image-encoder changes. The ViT, text encoder, DETR, geometry
encoder, and segmentation head don't branch on `variant` because their
weights and forward passes are identical.

### `sam3_ctx` initialization

In `sam3_init` → image-model construction, pass `config.n_fpn_scales`
into the neck builder. No other code reads `variant` in this
sub-project; future sub-projects will branch on it when building tracker
modules.

## CLI integration

### `sam3_cli info`

Add `variant` and `n_fpn_scales` to the info dump:

```
variant:        sam3.1
n_fpn_scales:   3
```

### `sam3_cli segment`

No new flags. The variant is read from the `.sam3` file; the segment
pipeline is identical for SAM 3 and SAM 3.1 at the image-detector
level, which is precisely the point.

### `sam3_cli convert`

Passes through the new `--variant` flag to `sam3_convert`.

## Testing

Three layers, each built on top of the existing test infrastructure.

### Unit — `tests/test_weight_sam3_1.c`

Runs only when `models/sam3.1.sam3` is present (guarded, like the
existing large-model tests). Asserts:

- `sam3_weight_load` returns a config with
  `variant == SAM3_VARIANT_SAM3_1` and `n_fpn_scales == 3`.
- The 0.5× FPN tensors
  (`detector_model.vision_encoder.neck.fpn_layers.3.*`) are **absent**.
- Detector-side ViT, DETR, and segmentation-head tensors are present
  with the same shapes as in the SAM 3 baseline (cross-checked against
  a tiny shape table compiled into the test).

A synthetic fixture covers the header round-trip in isolation: write a
minimal variant-=1, n_fpn_scales=3 `.sam3` file with a single 1×1
tensor, load it, assert fields round-trip. This runs always.

### Integration — extend `tests/test_fixture_compare.c`

The existing harness in `tests/test_fixture_compare.c` compares each
pipeline stage against reference tensors dumped from Python (ViT
blocks, neck scales, text encoder, DETR, seg head). We add a parallel
SAM 3.1 fixture directory:

```
tests/fixtures/sam3_1_bus_person/
    00_input/tensors.safetensors
    02_neck/scale_4x.safetensors
    02_neck/scale_2x.safetensors
    02_neck/scale_1x.safetensors        # no scale_05x
    10_final/tensors.safetensors
    metadata.json
```

and a new test function gated on `models/sam3.1.sam3` being present.
Neck comparison iterates only the three scales when `variant == SAM3_1`.
Final-mask comparison asserts IoU ≥ 0.99 against `10_final/tensors.safetensors`
and `iou_scores[0]` within 1e-3.

The fixture is produced by extending `tools/dump_reference.py` with a
SAM 3.1 image-model assembly path (see §9).

### Baseline refresh — `scripts/refresh_baselines.sh`

Learn a new fixture key `sam3_1_bus_person` so the script can re-dump
the per-stage tensors when weights or the Python reference change.

### Python reference assembly — `tools/dump_reference.py`

Python's `build_sam3_image_model` does not accept `version="sam3.1"`.
We assemble the SAM 3.1 image model by composing the multiplex
tri-neck from `_create_multiplex_tri_backbone`
(`reference/sam3/sam3/model_builder.py:920-934`) with the unchanged
detector components — essentially a ~50 line helper that re-builds
`Sam3Image(...)` with the tri-neck in place of the dual-neck. The
helper lives in `tools/dump_reference.py` and is not expected to ship
as part of the runtime.

## Build & dependencies

No new C dependencies. The Python helper requires `torch` and
`safetensors` (already used by other scripts in `tools/`).

## Risks & open questions

- **Python reference shape**: we assemble the SAM 3.1 image model by
  hand in `dump_reference.py` because upstream only exposes the video
  path with that variant. Low risk — all the pieces exist; we just
  pass them to `Sam3Image(...)`.
- **Header layout drift**: version 3 adds four bytes. Any code reading
  `sizeof(struct sam3_weight_header)` from a v2 file would be wrong;
  `sam3_weight_load` already reads field-by-field so this is safe, but
  any future reader must be careful.
- **Remap detection**: `pt_to_safetensors.py` uses the same heuristic
  as upstream (`k.startswith("sam3_model.")` or `"sam2_predictor."`);
  if HF changes the layout in a future checkpoint, the script will
  skip remapping and produce a correctly-named safetensors already.
- **Checkpoint provenance**: we rely on the HF token in the user's
  environment / command line. Nothing is persisted; the download
  lands in `models/` and the token should be rotated afterwards.

## Open work for future sub-projects (not this spec)

- Sub-project 2 will consume the passthrough tracker tensors and build
  the new memory attention / maskmem backbone.
- Sub-project 4 will add `SAM3_VARIANT_SAM3_1_MULTIPLEX` as a third
  enum value if the multiplex joint pass needs to be gated separately
  from "SAM 3.1 correctness, per-object".
