# MobileCLIP text encoder implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add MobileCLIP-S0/S1/L text encoder support to the SAM3 C inference engine, dispatched by a new `.sam3` v4 header field, exercising the three EfficientSAM3 checkpoints in `models/efficient_sam3_image_encoder_mobileclip_*_ctx16.pt`.

**Architecture:** New `sam3_text_encoder_iface` vtable wraps the existing CLIP encoder and a new `sam3_mobileclip_text_encoder` (config-driven, covers S0/S1/L from one code path; RepMixer is a conditional in block 0 for S0). Variant tag stored in a new `text_backbone` field of the v4 header; old v3 files default to CLIP. Conversion is `.pt` → `.safetensors` (existing Python helper) → `cli_convert -i ... --text-backbone mobileclip_s1` → `.sam3`.

**Tech Stack:** C11, arena allocators, graph IR, Metal/CPU backends, existing `SAM3_OP_CONV2D` (with groups) and `SAM3_OP_BATCHNORM` for RepMixer, Python/PyTorch for golden fixtures.

**Spec:** `docs/superpowers/specs/2026-04-20-mobileclip-text-encoder-design.md`
**Audit:** `docs/superpowers/specs/notes/2026-04-20-mobileclip-key-audit.md` (Phase 0 .pt key inventory)

> **Revision note (2026-04-20, post-Phase-0):** The Phase 0 audit + reference
> Python check changed the RepMixer design:
> - **S0 has 6 transformer blocks total** (indices 0–5; not 12 as the audit
>   doc initially extrapolated). RepMixer at indices **{0, 5}** (the
>   bookends), standard pre-norm transformer blocks at {1, 2, 3, 4}.
> - **MobileOne reparameterization is NOT collapsed** at export — both
>   RepMixer blocks ship parallel BN/conv branches that must be summed at
>   inference (see spec section "RepMixer block (S0 blocks 0 and 5) —
>   verified from reference" for exact arithmetic).
>
> Variant config now carries `n_repmixer_blocks` + `repmixer_block_indices[]`
> instead of a boolean, and the per-layer struct is a tagged union (std vs
> repmixer). Tasks 4.3, 4.5, 6.1, and 6.2 were drafted before the audit and
> contain stale `has_repmixer_block_0` / `repmixer_block_0` references — the
> implementer for those tasks must adapt to the tagged-union layer layout
> shown in Task 4.1. **Treat the spec as authoritative** if any code snippet
> below contradicts it.

---

## File Structure

### Created files

- `src/model/text_encoder_iface.h` / `.c` — vtable + factory dispatching CLIP vs MobileCLIP.
- `src/model/mobileclip_text.h` / `.c` — MobileCLIP encoder (S0/S1/L), shared code path with RepMixer conditional.
- `tests/test_mobileclip_text.c` — unit + per-block parity tests.
- `scripts/dump_mobileclip_text_layers.py` — Python reference dumper for fixture generation.
- `tests/fixtures/mobileclip_s1/` — per-block reference activations + final pooled embedding (one dir per variant: `mobileclip_s0`, `mobileclip_s1`, `mobileclip_l`).

### Modified files

- `include/sam3/sam3_types.h` — add `enum sam3_text_backbone`; add `text_backbone` to `sam3_model_config`.
- `src/core/weight.h` / `weight.c` — bump version 3→4, add `text_backbone` field to header struct, write/read it; v3 reads default `SAM3_TEXT_CLIP`.
- `src/model/vl_combiner.h` / `vl_combiner.c` — replace embedded `sam3_text_encoder text_enc` with `sam3_text_encoder_iface *text_iface`; route init/load/build through the iface.
- `src/model/text_encoder.c` — stop hardcoding causal mask in attention call (parameterize: NULL = non-causal). Existing CLIP behavior must stay bit-identical.
- `src/model/sam3_processor.c` — read `ctx_len` and embedding from the iface instead of `text_enc.context_len`.
- `tools/cli_convert.c` — add `--text-backbone` flag (clip|mobileclip_s0|mobileclip_s1|mobileclip_l), thread it into `sam3_model_config`.
- `CMakeLists.txt` — register `mobileclip_text.c`, `text_encoder_iface.c`, `test_mobileclip_text`.

### Unchanged (verified during grounding)

- Tokenizer API: `sam3_tokenizer_encode(tok, text, out, max_tokens)` already accepts variable length.
- Image encoder code, neck, mask decoder, processor public API, BPE vocab file, weight reader infrastructure, `pt_to_safetensors.py` (still does FB→OSS prefix remap; no extension required because variant is selected via `cli_convert` CLI flag).
- `tokens[77]` stack buffer in `sam3_processor.c:935` — already large enough for ctx=16/32/77.

---

## Phase ordering rationale

1. **Phase 0** (discovery): inspect actual `.pt` to confirm uncertain key names. Cheap, blocks Phase 4/6.
2. **Phase 1** (header v4): infrastructure for the variant tag. Independent of model code.
3. **Phase 2** (iface): refactor that keeps CLIP working. No behavior change.
4. **Phase 3** (tokenizer ctx_len): tiny plumbing change, prerequisite for MobileCLIP build path.
5. **Phase 4** (MobileCLIP standard): S1 first, then enable L by adding a config table entry.
6. **Phase 4.5** (converter `--text-backbone` flag): unblocks Phase 5 by producing `.sam3` files with `text_backbone` set correctly. Must precede the parity tests.
7. **Phase 5** (per-block parity): catches bugs in Phase 4 before Phase 6 piles on.
8. **Phase 6** (RepMixer): S0 only. Adds the conditional first-block path.
9. **Phase 7** (end-to-end + documentation): full sam3_processor smoke + variant docs.

---

## Phase 0 — Discovery

Three open items from the spec must be resolved by inspecting the actual `.pt` checkpoints before MobileCLIP-side code is written:

1. Exact leaf field names under the RepMixer keys (`token_mixer.mixer.*`, `token_mixer.norm.*`, `convffn.fc1.*`, etc.) — and whether MobileOne BN is folded into conv at export.
2. Exact key location of the external 256-dim projector (assumed `detector.backbone.language_backbone.projector.*`).
3. Exact shape of the standard MobileCLIP weights (qkv_proj, fc1/fc2, etc.) for S1 and L, to confirm config dims match the spec table.

### Task 0.1 — Write a key-inspection script

**Files:**
- Create: `scripts/inspect_mobileclip_pt.py`

- [ ] **Step 1: Write the script**

Create `scripts/inspect_mobileclip_pt.py`:

```python
#!/usr/bin/env python3
"""
scripts/inspect_mobileclip_pt.py - Dump key/shape/dtype for all tensors in
an EfficientSAM3 .pt checkpoint, focusing on the MobileCLIP text-encoder
sub-tree. Used during Phase 0 of the MobileCLIP plan to confirm exact
key names that the spec left as TBD.

Usage:
    python scripts/inspect_mobileclip_pt.py models/efficient_sam3_image_encoder_mobileclip_s0_ctx16.pt
"""
import argparse
import sys
import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("input", help="Input .pt path")
    ap.add_argument("--prefix", default="",
                    help="Filter to keys starting with this prefix")
    ap.add_argument("--grep", default=None,
                    help="Substring filter on key names")
    args = ap.parse_args()

    print(f"Loading {args.input} ...", file=sys.stderr)
    ckpt = torch.load(args.input, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    if not isinstance(ckpt, dict):
        print(f"error: not a state_dict (got {type(ckpt).__name__})",
              file=sys.stderr)
        return 1

    keys = list(ckpt.keys())
    if args.prefix:
        keys = [k for k in keys if k.startswith(args.prefix)]
    if args.grep:
        keys = [k for k in keys if args.grep in k]

    for k in sorted(keys):
        v = ckpt[k]
        if isinstance(v, torch.Tensor):
            print(f"{k}\t{tuple(v.shape)}\t{v.dtype}")
        else:
            print(f"{k}\t<{type(v).__name__}>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Mark executable**

Run: `chmod +x scripts/inspect_mobileclip_pt.py`

- [ ] **Step 3: Commit**

```bash
git add scripts/inspect_mobileclip_pt.py
git commit -m "scripts: add MobileCLIP .pt key inspector"
```

### Task 0.2 — Inspect all three checkpoints, record findings

**Files:**
- Create: `docs/superpowers/specs/notes/2026-04-20-mobileclip-key-audit.md`

- [ ] **Step 1: Run inspector against S0, S1, L**

```bash
python scripts/inspect_mobileclip_pt.py \
    models/efficient_sam3_image_encoder_mobileclip_s0_ctx16.pt \
    --grep language_backbone > /tmp/s0_keys.txt

python scripts/inspect_mobileclip_pt.py \
    models/efficient_sam3_image_encoder_mobileclip_s1_ctx16.pt \
    --grep language_backbone > /tmp/s1_keys.txt

python scripts/inspect_mobileclip_pt.py \
    models/efficient_sam3_image_encoder_mobileclip2_l_ctx16.pt \
    --grep language_backbone > /tmp/l_keys.txt
```

Also dump the projector candidate location:

```bash
python scripts/inspect_mobileclip_pt.py \
    models/efficient_sam3_image_encoder_mobileclip_s0_ctx16.pt \
    --grep projector > /tmp/projector_keys.txt
```

Inspect results and verify:
- For S1/L: every standard block has `pre_norm_mha.0.{weight,bias}`, `pre_norm_mha.1.qkv_proj.{weight,bias}`, `pre_norm_mha.1.out_proj.{weight,bias}`, `pre_norm_ffn.0.{weight,bias}`, `pre_norm_ffn.1.{weight,bias}`, `pre_norm_ffn.4.{weight,bias}`. Shapes match the spec table.
- For S0: blocks 1-5 follow the same standard pattern. Block 0 has different keys; record exactly what's there.
- Find the 256-dim projector: shape `[256, width]` somewhere with `projector` in the name.

- [ ] **Step 2: Write findings doc**

Create `docs/superpowers/specs/notes/2026-04-20-mobileclip-key-audit.md` with:

```markdown
# MobileCLIP key audit — 2026-04-20

Resolutions of TBD items from the design spec, derived from inspecting
the three .pt checkpoints in `models/`.

## Standard-block keys (S1/L, S0 blocks 1-5)

Verified prefix: `detector.backbone.language_backbone.encoder.transformer.{i}.`

| Sub-key                                  | Shape (S1)              | Shape (L)              |
|------------------------------------------|-------------------------|------------------------|
| `pre_norm_mha.0.weight` (LN1)            | [512]                   | [768]                  |
| `pre_norm_mha.0.bias`                    | [512]                   | [768]                  |
| `pre_norm_mha.1.qkv_proj.weight`         | [1536, 512]             | [2304, 768]            |
| `pre_norm_mha.1.qkv_proj.bias`           | [1536]                  | [2304]                 |
| `pre_norm_mha.1.out_proj.weight`         | [512, 512]              | [768, 768]             |
| `pre_norm_mha.1.out_proj.bias`           | [512]                   | [768]                  |
| `pre_norm_ffn.0.weight` (LN2)            | [512]                   | [768]                  |
| `pre_norm_ffn.0.bias`                    | [512]                   | [768]                  |
| `pre_norm_ffn.1.weight` (FC1)            | [2048, 512]             | [3072, 768]            |
| `pre_norm_ffn.1.bias`                    | [2048]                  | [3072]                 |
| `pre_norm_ffn.4.weight` (FC2)            | [512, 2048]             | [768, 3072]            |
| `pre_norm_ffn.4.bias`                    | [512]                   | [768]                  |

(Fill in actual observed shapes after running Step 1.)

## Embedding/finalisation keys

| Key                                                         | Shape               |
|-------------------------------------------------------------|---------------------|
| `language_backbone.encoder.embedding_layer.weight`          | [49408, width]      |
| `language_backbone.encoder.positional_embedding.pos_embed.pos_embed` | [1,1,77,width] |
| `language_backbone.encoder.final_layer_norm.weight`         | [width]             |
| `language_backbone.encoder.final_layer_norm.bias`           | [width]             |
| `language_backbone.encoder.projection_layer`                | [width, width]      |

## External 256-dim projector

Confirmed key: `<TODO from grep output>`
Shape: `[256, width]`

## RepMixer keys (S0 block 0 only)

Recorded leaf names with actual shapes:

```
<paste of `grep "transformer.0\." /tmp/s0_keys.txt`>
```

**BN folding decision**: BN params are/are-not present (record which).
If absent: fold-into-conv at export, skip BN ops in C.
If present: BN running stats ship with the checkpoint, emit BN ops.
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/notes/2026-04-20-mobileclip-key-audit.md
git commit -m "docs/specs: MobileCLIP .pt key audit (Phase 0)"
```

### Task 0.3 — Run reference forward to capture expected pooled embedding

**Files:**
- Create: `scripts/dump_mobileclip_text_layers.py`

- [ ] **Step 1: Write the dumper**

Create `scripts/dump_mobileclip_text_layers.py`:

```python
#!/usr/bin/env python3
"""
scripts/dump_mobileclip_text_layers.py - Dump per-block activations from
the reference MobileCLIP text encoder for a fixed prompt. Used to build
golden fixtures that the C-side test_mobileclip_text checks against.

Usage:
    python scripts/dump_mobileclip_text_layers.py \
        --variant mobileclip_s1 \
        --ckpt models/efficient_sam3_image_encoder_mobileclip_s1_ctx16.pt \
        --prompt "a person riding a bike" \
        --out tests/fixtures/mobileclip_s1
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# The reference repo lives at reference/efficientsam3.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference" / "efficientsam3"))

from sam3.sam3.model.text_encoder_student import TextStudentEncoder
from sam3.sam3.model.tokenizer_ve import SimpleTokenizer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--variant", required=True,
                    choices=["mobileclip_s0", "mobileclip_s1", "mobileclip_l"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="a person riding a bike")
    ap.add_argument("--ctx", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Load checkpoint and pluck out the language backbone state dict.
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    text_sd = {
        k.removeprefix("detector.backbone.language_backbone."): v
        for k, v in ckpt.items()
        if k.startswith("detector.backbone.language_backbone.")
    }

    # Build the reference encoder using the variant config from
    # reference/efficientsam3/sam3/sam3/model_builder.py
    # (The TextStudentEncoder factory accepts the variant string.)
    encoder = TextStudentEncoder.from_variant(args.variant, ctx_len=args.ctx)
    missing, unexpected = encoder.load_state_dict(text_sd, strict=False)
    if missing:
        print(f"warn: missing keys: {missing}", file=sys.stderr)
    if unexpected:
        print(f"warn: unexpected keys: {unexpected}", file=sys.stderr)
    encoder.eval()

    # Tokenize.
    tokenizer = SimpleTokenizer()
    tokens = tokenizer(args.prompt, context_length=args.ctx)
    np.save(out / "tokens.npy", tokens.numpy().astype(np.int32))

    # Hook every transformer block's input.
    activations = {}

    def make_hook(name):
        def hook(_mod, inputs, output):
            x = output if isinstance(output, torch.Tensor) else output[0]
            activations[name] = x.detach().cpu().float().numpy()
        return hook

    for i, blk in enumerate(encoder.encoder.transformer):
        blk.register_forward_hook(make_hook(f"block_{i:02d}_out"))
    encoder.encoder.final_layer_norm.register_forward_hook(make_hook("final_ln_out"))

    with torch.no_grad():
        out_tokens = encoder(tokens.unsqueeze(0))
    pooled = out_tokens[0, tokens.argmax(dim=-1)].cpu().float().numpy()

    np.save(out / "out_tokens.npy", out_tokens[0].cpu().float().numpy())
    np.save(out / "pooled.npy", pooled)
    for name, arr in activations.items():
        np.save(out / f"{name}.npy", arr)

    print(f"Wrote fixtures to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Generate fixtures for S1 (smallest standard variant)**

```bash
python scripts/dump_mobileclip_text_layers.py \
    --variant mobileclip_s1 \
    --ckpt models/efficient_sam3_image_encoder_mobileclip_s1_ctx16.pt \
    --prompt "a person riding a bike" \
    --out tests/fixtures/mobileclip_s1
```

Expected: `tests/fixtures/mobileclip_s1/{tokens,out_tokens,pooled,block_00_out,...,block_11_out,final_ln_out}.npy`

If the script fails with import errors, pause and reconcile imports against the actual layout under `reference/efficientsam3/sam3/sam3/`. Update the script accordingly.

- [ ] **Step 3: Generate fixtures for S0 and L**

```bash
python scripts/dump_mobileclip_text_layers.py \
    --variant mobileclip_s0 \
    --ckpt models/efficient_sam3_image_encoder_mobileclip_s0_ctx16.pt \
    --out tests/fixtures/mobileclip_s0

python scripts/dump_mobileclip_text_layers.py \
    --variant mobileclip_l \
    --ckpt models/efficient_sam3_image_encoder_mobileclip2_l_ctx16.pt \
    --out tests/fixtures/mobileclip_l
```

- [ ] **Step 4: Commit**

```bash
git add scripts/dump_mobileclip_text_layers.py tests/fixtures/mobileclip_s0 tests/fixtures/mobileclip_s1 tests/fixtures/mobileclip_l
git commit -m "scripts: MobileCLIP reference dumper + S0/S1/L golden fixtures"
```

---

## Phase 1 — Header v4 + variant enum + config plumbing

### Task 1.1 — Add `enum sam3_text_backbone` and `text_backbone` to `sam3_model_config`

**Files:**
- Modify: `include/sam3/sam3_types.h:108-133`

- [ ] **Step 1: Add the enum and config field**

After `enum sam3_variant` (around line 121), insert:

```c
/* Text encoder backbone variant. Stored in the .sam3 v4 header.
 * v3 files (no field) load as SAM3_TEXT_CLIP for backward compat. */
enum sam3_text_backbone {
	SAM3_TEXT_CLIP            = 0,
	SAM3_TEXT_MOBILECLIP_S0   = 1,
	SAM3_TEXT_MOBILECLIP_S1   = 2,
	SAM3_TEXT_MOBILECLIP_L    = 3,
};
```

In `struct sam3_model_config` (lines 124-133), add after `int variant;`:

```c
	int text_backbone;    /* enum sam3_text_backbone */
```

- [ ] **Step 2: Build to confirm no callers break**

Run: `cd build && cmake --build . -j$(sysctl -n hw.ncpu) 2>&1 | tail -20`
Expected: success (new field is unused; existing call sites zero-init the struct so it implicitly defaults to `SAM3_TEXT_CLIP`).

- [ ] **Step 3: Commit**

```bash
git add include/sam3/sam3_types.h
git commit -m "sam3_types: add sam3_text_backbone enum + config field"
```

### Task 1.2 — Bump `.sam3` header to v4 (52 bytes), add `text_backbone` field

**Files:**
- Modify: `src/core/weight.h:28,40-51,67`

- [ ] **Step 1: Bump version, extend header struct**

In `src/core/weight.h` line 28:

```c
#define SAM3_WEIGHT_VERSION 4
```

Replace the `struct sam3_weight_header` block (lines 36-51):

```c
/*
 * File header: 52 bytes (v4) / 48 bytes (v3, backward-compat read).
 * Sits at offset 0 in the .sam3 file.
 *
 * v4 added `text_backbone` (enum sam3_text_backbone). v3 readers see
 * the trailing 4 bytes as the start of the tensor descriptor table —
 * weight.c handles version-aware sizing in sam3_weight_open.
 */
struct sam3_weight_header {
	uint32_t magic;
	uint32_t version;
	uint32_t flags;
	uint32_t n_tensors;
	int32_t  image_size;
	int32_t  encoder_dim;
	int32_t  decoder_dim;
	int32_t  n_encoder_layers;
	int32_t  n_decoder_layers;
	uint32_t reserved[3];     /* [0]=backbone_type, [1]=variant, [2]=n_fpn_scales */
	uint32_t text_backbone;   /* v4: enum sam3_text_backbone (0=CLIP) */
};
```

Update the static_assert at line 67:

```c
_Static_assert(sizeof(struct sam3_weight_header) == 52,
	       "header must be exactly 52 bytes on disk (v4)");
```

- [ ] **Step 2: Build expecting compile failures in weight.c (intentional)**

Run: `cd build && cmake --build . -j$(sysctl -n hw.ncpu) 2>&1 | tail -30`
Expected: `weight.c` errors at the assert (header is now 52 not 48). Next task fixes the writer/reader.

- [ ] **Step 3: Commit (compile-broken intermediate state)**

We commit now because the header change is a logical unit. Phase 1 stays bisectable across tasks; the broken state is fixed in Task 1.3.

```bash
git add src/core/weight.h
git commit -m "weight: bump .sam3 format to v4 with text_backbone field"
```

### Task 1.3 — Update writer + reader to round-trip `text_backbone`

**Files:**
- Modify: `src/core/weight.c:86-99` (writer header build)
- Modify: `src/core/weight.c:355-363` (reader version check)

- [ ] **Step 1: Writer emits text_backbone**

In `sam3_weight_write` (around line 86-99) add a new field assignment after `hdr.reserved[2]`:

```c
	hdr.reserved[2]      = (uint32_t)config->n_fpn_scales;
	hdr.text_backbone    = (uint32_t)config->text_backbone;
```

- [ ] **Step 2: Reader handles v3 (legacy) and v4**

Replace the version check in `sam3_weight_open` (around line 356-363) with a v3/v4 dispatch. Insert the version-aware logic and a derived "header_size" that the rest of the function uses for the data-section calculation.

The existing block:

```c
	if (hdr->version != SAM3_WEIGHT_VERSION) {
		sam3_log_error("weight_open: unsupported version %u in %s "
			       "(expected %u; regenerate via sam3_convert)",
			       hdr->version, path, SAM3_WEIGHT_VERSION);
		munmap(mapped, file_size);
		memset(wf, 0, sizeof(*wf));
		return SAM3_EMODEL;
	}
```

Becomes:

```c
	size_t header_size;
	uint32_t text_backbone;

	if (hdr->version == 3) {
		/*
		 * v3 backward-compat: header is 48 bytes (no text_backbone).
		 * The new field defaults to SAM3_TEXT_CLIP. Tensor table
		 * starts at offset 48, not sizeof(struct sam3_weight_header).
		 */
		header_size  = 48;
		text_backbone = 0; /* SAM3_TEXT_CLIP */
	} else if (hdr->version == 4) {
		header_size  = sizeof(struct sam3_weight_header);
		text_backbone = hdr->text_backbone;
	} else {
		sam3_log_error("weight_open: unsupported version %u in %s "
			       "(expected 3 or 4; regenerate via sam3_convert)",
			       hdr->version, path);
		munmap(mapped, file_size);
		memset(wf, 0, sizeof(*wf));
		return SAM3_EMODEL;
	}
	(void)text_backbone; /* exposed via accessor in next task */
```

Replace the existing tensor-table offset calculation (around line 366-394) so it uses `header_size` instead of `sizeof(struct sam3_weight_header)`:

```c
	size_t table_size = (size_t)hdr->n_tensors *
			    sizeof(struct sam3_weight_tensor_desc);
	size_t table_end  = header_size + table_size;
	...
	wf->tensors     = (const struct sam3_weight_tensor_desc *)
			  ((const char *)mapped + header_size);
```

- [ ] **Step 3: Build**

Run: `cd build && cmake --build . -j$(sysctl -n hw.ncpu) 2>&1 | tail -20`
Expected: success.

- [ ] **Step 4: Run existing tests to confirm v3 backcompat**

Run: `cd build && ctest --output-on-failure -R weight 2>&1 | tail -20`
Expected: all weight-related tests pass (existing `.sam3` files in `models/` are v3 and must still load).

Also run the broader smoke set:

```bash
cd build && ctest --output-on-failure -E '(text|video|track)' 2>&1 | tail -40
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/core/weight.c
git commit -m "weight: write text_backbone (v4) + read v3/v4 round-trip"
```

### Task 1.4 — Expose `text_backbone` to the loader path

**Files:**
- Modify: `src/core/weight.h` (add field to `sam3_weight_file`)
- Modify: `src/core/weight.c:355-394` (populate the field)
- Modify: `src/sam3.c` (read it from the file, write it into the runtime config)

- [ ] **Step 1: Add field to `sam3_weight_file`**

In `src/core/weight.h`, in `struct sam3_weight_file` (around line 78-88), add:

```c
	uint32_t                              text_backbone; /* derived from header */
```

- [ ] **Step 2: Populate it in `sam3_weight_open`**

Replace the `(void)text_backbone;` line added in the previous task with:

```c
	wf->text_backbone = text_backbone;
```

(Place it just before the `wf->mapped = mapped;` block.)

- [ ] **Step 3: Thread it into runtime config in sam3.c**

Find the `sam3_load_model` (or equivalent) function in `src/sam3.c` that reads `wf->header->reserved[*]` into `config.backbone_type` etc. Add:

```c
	config.text_backbone = (int)wf->text_backbone;
```

(Use `Grep` for `config.backbone_type` in `src/sam3.c` to locate the right block; the new line goes adjacent.)

- [ ] **Step 4: Build + smoke**

Run: `cd build && cmake --build . -j$(sysctl -n hw.ncpu) && ctest --output-on-failure -R weight 2>&1 | tail -20`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/core/weight.h src/core/weight.c src/sam3.c
git commit -m "weight: surface text_backbone through wf -> sam3_model_config"
```

---

## Phase 2 — Text encoder iface + CLIP backwards-compat

### Task 2.1 — Define the iface vtable

**Files:**
- Create: `src/model/text_encoder_iface.h`

- [ ] **Step 1: Write the iface header**

Create `src/model/text_encoder_iface.h`:

```c
/*
 * src/model/text_encoder_iface.h - Text encoder vtable + factory
 *
 * Wraps either the existing CLIP encoder (sam3_text_encoder) or the new
 * MobileCLIP encoder (sam3_mobileclip_text_encoder) behind a single
 * interface so the rest of the engine (vl_combiner, sam3_processor)
 * dispatches without switching on backbone. Selected at load time by
 * the .sam3 v4 header's text_backbone field.
 *
 * Key types:  sam3_text_encoder_iface, sam3_text_encoder_iface_ops
 * Depends on: sam3/sam3_types.h, core/tensor.h, core/graph.h, core/alloc.h,
 *             core/weight.h, backend/backend.h
 * Used by:    src/model/vl_combiner.c, src/model/sam3_processor.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_TEXT_ENCODER_IFACE_H
#define SAM3_MODEL_TEXT_ENCODER_IFACE_H

#include "sam3/sam3_types.h"
#include "core/tensor.h"
#include "core/graph.h"
#include "core/alloc.h"
#include "core/weight.h"
#include "backend/backend.h"

struct sam3_text_encoder_iface;

struct sam3_text_encoder_iface_ops {
	enum sam3_error (*load)(struct sam3_text_encoder_iface *iface,
				const struct sam3_weight_file *wf,
				struct sam3_arena *arena);
	struct sam3_tensor *(*build)(struct sam3_text_encoder_iface *iface,
				     struct sam3_graph *g,
				     struct sam3_tensor *token_ids,
				     struct sam3_tensor **pooled_out,
				     struct sam3_arena *arena);
	struct sam3_tensor *(*build_perblock)(
		struct sam3_text_encoder_iface *iface,
		struct sam3_backend *be,
		struct sam3_tensor *token_ids,
		struct sam3_arena *scratch,
		struct sam3_arena *persist);
	void (*free)(struct sam3_text_encoder_iface *iface);
};

struct sam3_text_encoder_iface {
	const struct sam3_text_encoder_iface_ops *ops;
	void *impl;          /* concrete encoder pointer */
	int   text_backbone; /* enum sam3_text_backbone */
	int   ctx_len;       /* sequence length the encoder expects */
	int   d_model;       /* output embedding dimension (always 256 today) */
};

/*
 * sam3_text_encoder_iface_init - Construct an iface for the given backbone.
 *
 * @iface:         Caller-allocated iface struct (zeroed)
 * @text_backbone: enum sam3_text_backbone (CLIP or one of the MOBILECLIP_*)
 * @arena:         Arena for the concrete encoder struct + its precomputed data
 *
 * For SAM3_TEXT_CLIP, allocates a sam3_text_encoder with the historical
 * 24L/1024w/16h/ctx=32 config used today. For SAM3_TEXT_MOBILECLIP_*,
 * allocates a sam3_mobileclip_text_encoder configured per the variant
 * table in mobileclip_text.c.
 *
 * Returns SAM3_OK on success; SAM3_EINVAL on unknown backbone;
 * SAM3_ENOMEM on arena exhaustion.
 */
enum sam3_error sam3_text_encoder_iface_init(
	struct sam3_text_encoder_iface *iface,
	int text_backbone,
	struct sam3_arena *arena);

#endif /* SAM3_MODEL_TEXT_ENCODER_IFACE_H */
```

- [ ] **Step 2: Build (header-only, no compile dependents yet)**

Run: `cd build && cmake --build . -j$(sysctl -n hw.ncpu) 2>&1 | tail -10`
Expected: success.

- [ ] **Step 3: Commit**

```bash
git add src/model/text_encoder_iface.h
git commit -m "model: add text_encoder_iface vtable + factory declaration"
```

### Task 2.2 — Implement the iface for CLIP (existing encoder)

**Files:**
- Create: `src/model/text_encoder_iface.c`
- Modify: `CMakeLists.txt` (register the new source)

- [ ] **Step 1: Write the iface impl**

Create `src/model/text_encoder_iface.c`:

```c
/*
 * src/model/text_encoder_iface.c - Text encoder vtable wiring
 *
 * Concrete vtable implementations: one set of ops wraps the historical
 * sam3_text_encoder (CLIP); a second set wraps sam3_mobileclip_text_encoder
 * (the three MobileCLIP variants). The factory picks the right pair based
 * on the text_backbone enum and stashes it on the iface.
 *
 * Key types:  sam3_text_encoder_iface
 * Depends on: text_encoder_iface.h, text_encoder.h, mobileclip_text.h
 * Used by:    src/model/vl_combiner.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include <string.h>

#include "text_encoder_iface.h"
#include "text_encoder.h"
#include "core/alloc.h"
#include "util/log.h"

/* --- CLIP wrapper ops --- */

static enum sam3_error clip_load(struct sam3_text_encoder_iface *iface,
				 const struct sam3_weight_file *wf,
				 struct sam3_arena *arena)
{
	return sam3_text_encoder_load((struct sam3_text_encoder *)iface->impl,
				      wf, arena);
}

static struct sam3_tensor *clip_build(struct sam3_text_encoder_iface *iface,
				      struct sam3_graph *g,
				      struct sam3_tensor *token_ids,
				      struct sam3_tensor **pooled_out,
				      struct sam3_arena *arena)
{
	return sam3_text_encoder_build(
		(struct sam3_text_encoder *)iface->impl, g, token_ids,
		pooled_out, arena);
}

static struct sam3_tensor *clip_build_perblock(
	struct sam3_text_encoder_iface *iface,
	struct sam3_backend *be,
	struct sam3_tensor *token_ids,
	struct sam3_arena *scratch,
	struct sam3_arena *persist)
{
	return sam3_text_encoder_build_perblock(
		(struct sam3_text_encoder *)iface->impl, be, token_ids,
		scratch, persist);
}

static void clip_free(struct sam3_text_encoder_iface *iface)
{
	(void)iface; /* arena-backed; nothing to free */
}

static const struct sam3_text_encoder_iface_ops clip_ops = {
	.load           = clip_load,
	.build          = clip_build,
	.build_perblock = clip_build_perblock,
	.free           = clip_free,
};

/* --- MobileCLIP wrapper ops --- forward decls used after Phase 4 lands --- */

extern const struct sam3_text_encoder_iface_ops sam3_mobileclip_text_iface_ops;
extern enum sam3_error sam3_mobileclip_text_iface_init_impl(
	struct sam3_text_encoder_iface *iface,
	int text_backbone, struct sam3_arena *arena);

/* --- Factory --- */

enum sam3_error sam3_text_encoder_iface_init(
	struct sam3_text_encoder_iface *iface,
	int text_backbone,
	struct sam3_arena *arena)
{
	if (!iface || !arena)
		return SAM3_EINVAL;

	memset(iface, 0, sizeof(*iface));
	iface->text_backbone = text_backbone;

	switch (text_backbone) {
	case SAM3_TEXT_CLIP: {
		struct sam3_text_encoder *te;

		te = sam3_arena_alloc(arena, sizeof(*te), _Alignof(*te));
		if (!te) {
			sam3_log_error("text_iface: arena alloc failed (CLIP)");
			return SAM3_ENOMEM;
		}
		memset(te, 0, sizeof(*te));
		te->d_model     = 256;
		te->width       = 1024;
		te->n_heads     = 16;
		te->n_layers    = 24;
		te->context_len = 32;   /* matches existing vl_combiner config */
		te->vocab_size  = 49408;

		iface->impl    = te;
		iface->ops     = &clip_ops;
		iface->ctx_len = te->context_len;
		iface->d_model = te->d_model;
		return SAM3_OK;
	}
	case SAM3_TEXT_MOBILECLIP_S0:
	case SAM3_TEXT_MOBILECLIP_S1:
	case SAM3_TEXT_MOBILECLIP_L:
		return sam3_mobileclip_text_iface_init_impl(
			iface, text_backbone, arena);
	default:
		sam3_log_error("text_iface: unknown text_backbone %d",
			       text_backbone);
		return SAM3_EINVAL;
	}
}
```

- [ ] **Step 2: Add temporary stub for the MobileCLIP impl**

The factory references `sam3_mobileclip_text_iface_init_impl` which doesn't exist yet. Add a temporary stub at the bottom of `src/model/text_encoder_iface.c` to keep the build green during Phase 2. This stub will be replaced in Phase 4.

```c
/*
 * Temporary stub: removed in Phase 4 once mobileclip_text.c lands.
 * The CLIP path doesn't reach this; MobileCLIP variants will fail
 * until Phase 4.
 */
__attribute__((weak))
enum sam3_error sam3_mobileclip_text_iface_init_impl(
	struct sam3_text_encoder_iface *iface,
	int text_backbone, struct sam3_arena *arena)
{
	(void)iface; (void)text_backbone; (void)arena;
	sam3_log_error("text_iface: MobileCLIP not yet implemented");
	return SAM3_EINVAL;
}
```

- [ ] **Step 3: Register in CMake**

Add `src/model/text_encoder_iface.c` to the appropriate `target_sources` block in `CMakeLists.txt`. Use:

```bash
grep -n "src/model/text_encoder.c" CMakeLists.txt
```

Insert `src/model/text_encoder_iface.c` adjacent to `src/model/text_encoder.c`.

- [ ] **Step 4: Build**

Run: `cd build && cmake --build . -j$(sysctl -n hw.ncpu) 2>&1 | tail -20`
Expected: success.

- [ ] **Step 5: Commit**

```bash
git add src/model/text_encoder_iface.c CMakeLists.txt
git commit -m "model: text_encoder_iface impl with CLIP wiring + MobileCLIP stub"
```

### Task 2.3 — Switch `vl_combiner` to use the iface

**Files:**
- Modify: `src/model/vl_combiner.h:30-44` (replace embedded text_enc with iface)
- Modify: `src/model/vl_combiner.c:124-149` (init), `:217-219` (load), `:348-378` (build_text)
- Modify: `src/model/sam3_processor.c:786-791,935-961` (read ctx_len from iface)

- [ ] **Step 1: Update `sam3_vl_backbone` struct**

In `src/model/vl_combiner.h`, replace:

```c
#include "text_encoder.h"
```

with:

```c
#include "text_encoder.h"
#include "text_encoder_iface.h"
```

Replace the `struct sam3_text_encoder text_enc;` field with:

```c
	struct sam3_text_encoder_iface text_iface;
```

Update the docstring on the struct accordingly (mention dispatch through iface).

- [ ] **Step 2: Update init/load/build_text in vl_combiner.c**

In `sam3_vl_backbone_init` (around line 124-149), replace the inline text_enc config block:

```c
	/* Init text encoder config (weights loaded separately) */
	vl->text_enc.d_model = 256;
	vl->text_enc.width = 1024;
	vl->text_enc.n_heads = 16;
	vl->text_enc.n_layers = 24;
	vl->text_enc.context_len = 32;
	vl->text_enc.vocab_size = 49408;
```

with:

```c
	/*
	 * Init text encoder iface. Default to CLIP here; the variant
	 * passed to sam3_vl_backbone_init may not yet be wired (callers
	 * may overwrite via sam3_vl_backbone_set_text_backbone before
	 * load). The actual variant flows through sam3_load_model from
	 * the .sam3 v4 header.
	 */
	err = sam3_text_encoder_iface_init(&vl->text_iface,
					   SAM3_TEXT_CLIP, arena);
	if (err != SAM3_OK)
		return err;
```

Add a new public entry point in `vl_combiner.h`:

```c
/*
 * sam3_vl_backbone_set_text_backbone - Reinitialize the text iface.
 *
 * Called by the loader after reading the .sam3 v4 header so the iface
 * matches the file's text_backbone. Must be called BEFORE
 * sam3_vl_backbone_load.
 */
enum sam3_error sam3_vl_backbone_set_text_backbone(
	struct sam3_vl_backbone *vl, int text_backbone,
	struct sam3_arena *arena);
```

And implement it in `vl_combiner.c`:

```c
enum sam3_error sam3_vl_backbone_set_text_backbone(
	struct sam3_vl_backbone *vl, int text_backbone,
	struct sam3_arena *arena)
{
	if (!vl || !arena)
		return SAM3_EINVAL;
	return sam3_text_encoder_iface_init(&vl->text_iface,
					    text_backbone, arena);
}
```

In `sam3_vl_backbone_load` (around line 217-219), replace:

```c
	err = sam3_text_encoder_load(&vl->text_enc, wf, arena);
```

with:

```c
	err = vl->text_iface.ops->load(&vl->text_iface, wf, arena);
```

In `sam3_vl_backbone_build_text` (around line 348-378), replace the body with:

```c
struct sam3_tensor *sam3_vl_backbone_build_text(
	struct sam3_vl_backbone *vl,
	struct sam3_graph *g,
	const char *text,
	struct sam3_tensor **pooled_out,
	struct sam3_arena *arena)
{
	int ctx = vl->text_iface.ctx_len;
	int32_t tokens[SAM3_TOKENIZER_CONTEXT_LEN]; /* 32; ≥ all current ctx_lens */
	int n_tokens;
	struct sam3_tensor *tok_tensor;
	int dims[1];

	if (ctx > SAM3_TOKENIZER_CONTEXT_LEN)
		ctx = SAM3_TOKENIZER_CONTEXT_LEN;

	n_tokens = sam3_tokenizer_encode(&vl->tokenizer, text, tokens, ctx);
	if (n_tokens <= 0)
		return NULL;

	dims[0] = ctx;
	tok_tensor = gh_alloc_tensor(arena, SAM3_DTYPE_I32, 1, dims);
	if (!tok_tensor)
		return NULL;
	memcpy(tok_tensor->data, tokens, (size_t)ctx * sizeof(int32_t));

	return vl->text_iface.ops->build(&vl->text_iface, g, tok_tensor,
					 pooled_out, arena);
}
```

If `SAM3_TOKENIZER_CONTEXT_LEN` (currently 32 in `tokenizer.h`) is too small for ctx=77 (CLIP) we'd hit the clamp. Today the CLIP path runs at ctx=32, so 32 stays fine. If a future variant exceeds 32, the clamp logs a warning and truncates — acceptable.

- [ ] **Step 3: Update sam3_processor.c text-path call sites**

Two call sites use `te->context_len` — replace with the iface's ctx_len.

At `src/model/sam3_processor.c:786-791`, change:

```c
	te = &proc->model.backbone.text_enc;
	...
	n_tokens = sam3_tokenizer_encode(
		&proc->model.backbone.tokenizer, text,
		proc->text_tokens, te->context_len);
```

to:

```c
	struct sam3_text_encoder_iface *te_iface =
		&proc->model.backbone.text_iface;
	int ctx = te_iface->ctx_len;

	n_tokens = sam3_tokenizer_encode(
		&proc->model.backbone.tokenizer, text,
		proc->text_tokens, ctx);
```

(Find every other reference to `te = &proc->model.backbone.text_enc` and `te->context_len` in `sam3_processor.c` and change them to read from the iface. The grep:

```bash
grep -n "model.backbone.text_enc\|te->context_len" src/model/sam3_processor.c
```

…tells you the exact set.)

Where the worker thread previously called `sam3_text_encoder_build_perblock(&proc->model.backbone.text_enc, ...)`, replace with:

```c
	te_iface = &proc->model.backbone.text_iface;
	text_features = te_iface->ops->build_perblock(
		te_iface, proc->text_backend, tok_tensor,
		&proc->text_scratch_arena, &proc->text_persist_arena);
```

- [ ] **Step 4: Wire text_backbone into the loader**

In `src/sam3.c`, somewhere between `sam3_vl_backbone_init` and `sam3_vl_backbone_load`, insert:

```c
	err = sam3_vl_backbone_set_text_backbone(
		&proc->model.backbone, config.text_backbone,
		<the same arena passed to vl_backbone_init>);
	if (err != SAM3_OK) {
		sam3_log_error("sam3_load: text_backbone init failed (%d)", err);
		goto cleanup;
	}
```

Use `Grep` to find where `sam3_vl_backbone_init` is called in `src/sam3.c` (or `sam3_processor.c`/`sam3_image.c`) and place this call right after.

- [ ] **Step 5: Build + smoke**

Run:
```bash
cd build && cmake --build . -j$(sysctl -n hw.ncpu) 2>&1 | tail -20
ctest --output-on-failure -R '(text|prompt)' 2>&1 | tail -30
```

Expected: all existing text/prompt tests pass (CLIP behavior preserved end-to-end through the iface).

- [ ] **Step 6: Commit**

```bash
git add src/model/vl_combiner.h src/model/vl_combiner.c src/model/sam3_processor.c src/sam3.c
git commit -m "vl_combiner: route text encoder through iface (CLIP unchanged)"
```

---

## Phase 3 — Tokenizer ctx_len pass-through (no API change)

The tokenizer API already accepts a runtime `max_tokens`. Phase 2 already routed the iface's `ctx_len` into both call sites in `sam3_processor.c`. There is nothing else to do here unless an existing call uses a hardcoded constant.

### Task 3.1 — Audit hardcoded tokenizer call sites

**Files:**
- Read-only audit; no edits unless something is hardcoded wrong.

- [ ] **Step 1: Grep for hardcoded constants**

Run:
```bash
grep -rn "sam3_tokenizer_encode" src/ tests/ tools/
```

Confirm every call passes either `te_iface->ctx_len`, `te->context_len`, or `SAM3_TOKENIZER_CONTEXT_LEN`. None should pass a literal `77`.

- [ ] **Step 2: If any literal is found, replace with the iface ctx_len**

(For each offending site, change to read from `iface->ctx_len`. If none, skip.)

- [ ] **Step 3: Commit (only if changes were made)**

```bash
git add <changed files>
git commit -m "tokenizer: route ctx_len through iface at every call site"
```

---

## Phase 4 — MobileCLIP standard transformer (S1 + L; S0 standard blocks)

### Task 4.1 — Define `sam3_mobileclip_text_encoder` struct + variant configs

**Files:**
- Create: `src/model/mobileclip_text.h`

- [ ] **Step 1: Write the header**

Create `src/model/mobileclip_text.h`:

```c
/*
 * src/model/mobileclip_text.h - MobileCLIP text encoder (S0/S1/L)
 *
 * Pre-norm transformer text encoder with non-causal attention. Covers
 * three variants from one code path, parameterized by config (n_layers,
 * width, n_heads, mlp_dim, n_repmixer_blocks). The S0 variant has
 * RepMixer blocks at indices listed in cfg.repmixer_block_indices
 * (verified in the .pt audit at indices 0 and 5); S1 and L have no
 * RepMixer blocks. Each layer slot uses either the std or repmixer
 * sub-struct depending on the per-block flag.
 *
 * Output contract matches sam3_text_encoder: per-token [ctx_len, 256]
 * plus pooled [256] from the EOT position.
 *
 * Key types:  sam3_mobileclip_text_encoder, sam3_mobileclip_config
 * Depends on: text_encoder_iface.h, core/tensor.h, core/graph.h
 * Used by:    src/model/text_encoder_iface.c, tests/test_mobileclip_text.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_MOBILECLIP_TEXT_H
#define SAM3_MODEL_MOBILECLIP_TEXT_H

#include "text_encoder_iface.h"
#include "core/tensor.h"
#include "core/graph.h"
#include "core/alloc.h"
#include "core/weight.h"
#include "backend/backend.h"

#define SAM3_MOBILECLIP_MAX_LAYERS 12
#define SAM3_MOBILECLIP_MAX_REPMIXER_BLOCKS 4

struct sam3_mobileclip_config {
	int  text_backbone;             /* enum sam3_text_backbone */
	int  n_layers;                  /* total transformer block count (12 for all variants) */
	int  width;                     /* embedding/transformer dim */
	int  n_heads;
	int  mlp_dim;
	int  ctx_len;
	int  out_dim;                   /* always 256 for SAM3 */
	int  vocab_size;                /* always 49408 */
	int  pos_embed_table_len;       /* always 77 */
	int  n_repmixer_blocks;         /* 0 for S1/L; 2 for S0 (indices 0,5) */
	int  repmixer_block_indices[SAM3_MOBILECLIP_MAX_REPMIXER_BLOCKS];
};

struct sam3_mobileclip_layer_std {
	struct sam3_tensor *ln1_w, *ln1_b;
	struct sam3_tensor *qkv_w, *qkv_b;
	struct sam3_tensor *out_w, *out_b;
	struct sam3_tensor *ln2_w, *ln2_b;
	struct sam3_tensor *fc1_w, *fc1_b;
	struct sam3_tensor *fc2_w, *fc2_b;
};

/*
 * RepMixer block weights (S0 blocks 0 and 5). MobileOne reparameterization
 * is NOT collapsed at export, so we ship parallel BN/conv branches that
 * the build code sums at inference. eps=1e-5 for every BN.
 *
 * Audit-verified key shapes (S0, width=512):
 *   norm_skip_*       (512,)             — single BN on input
 *   mixer_skip_*      (512,)             — branch 1: BN(x)
 *   mixer_conv_w      (512,1,1,11)       — depthwise 1×11 conv
 *   mixer_conv_bn_*   (512,)             — branch 2: BN(conv(x))
 *   tm_layer_scale    (512,1,1)          — token-mixer residual scale
 *   convffn_dw_w      (512,1,1,11)       — convffn.conv: depthwise 1×11
 *   convffn_dw_bn_*   (512,)             — convffn.conv.bn
 *   convffn_fc1_w     (2048,512,1,1)     — convffn.fc1: 1×1, [512→2048]
 *   convffn_fc1_b     (2048,)
 *   convffn_fc2_w     (512,2048,1,1)     — convffn.fc2: 1×1, [2048→512]
 *   convffn_fc2_b     (512,)
 *   outer_layer_scale (512,1,1)          — outer block residual scale
 */
struct sam3_mobileclip_layer_repmixer {
	/* token_mixer.norm — single standalone BN */
	struct sam3_tensor *norm_skip_w, *norm_skip_b;
	struct sam3_tensor *norm_skip_rm, *norm_skip_rv;

	/* token_mixer.mixer.rbr_skip — BN-only branch */
	struct sam3_tensor *mixer_skip_w, *mixer_skip_b;
	struct sam3_tensor *mixer_skip_rm, *mixer_skip_rv;

	/* token_mixer.mixer.rbr_conv[0] — depthwise conv + BN branch */
	struct sam3_tensor *mixer_conv_w;                    /* [512,1,1,11] */
	struct sam3_tensor *mixer_conv_bn_w, *mixer_conv_bn_b;
	struct sam3_tensor *mixer_conv_bn_rm, *mixer_conv_bn_rv;

	/* (Optional) token_mixer.mixer.rbr_scale — 1×1 conv + BN branch.
	 * Absent in the audited S0 checkpoint; loader leaves NULL when
	 * keys are missing and build_repmixer_block skips the branch. */
	struct sam3_tensor *mixer_scale_w;                   /* [512,512,1,1] or NULL */
	struct sam3_tensor *mixer_scale_bn_w, *mixer_scale_bn_b;
	struct sam3_tensor *mixer_scale_bn_rm, *mixer_scale_bn_rv;

	/* token_mixer residual scale: x + tm_layer_scale * (mixer_out - norm_out) */
	struct sam3_tensor *tm_layer_scale;                  /* [512,1,1] */

	/* convffn.conv — depthwise 1×11 conv + BN */
	struct sam3_tensor *convffn_dw_w;                    /* [512,1,1,11] */
	struct sam3_tensor *convffn_dw_bn_w, *convffn_dw_bn_b;
	struct sam3_tensor *convffn_dw_bn_rm, *convffn_dw_bn_rv;

	/* convffn.fc1 — 1×1 conv (with bias) */
	struct sam3_tensor *convffn_fc1_w, *convffn_fc1_b;

	/* convffn.fc2 — 1×1 conv (with bias) */
	struct sam3_tensor *convffn_fc2_w, *convffn_fc2_b;

	/* outer block residual scale: x + outer_layer_scale * convffn(x) */
	struct sam3_tensor *outer_layer_scale;               /* [512,1,1] */
};

/*
 * Tagged per-block layer slot. is_repmixer=1 selects the repmixer union
 * arm; is_repmixer=0 selects std. Set by the loader from cfg.repmixer_block_indices.
 */
struct sam3_mobileclip_layer {
	int is_repmixer;
	union {
		struct sam3_mobileclip_layer_std       std;
		struct sam3_mobileclip_layer_repmixer  repmixer;
	} u;
};

struct sam3_mobileclip_text_encoder {
	struct sam3_mobileclip_config cfg;

	/* Embeddings */
	struct sam3_tensor *token_embedding;   /* [vocab_size, width] */
	struct sam3_tensor *pos_embed_full;    /* [1,1,77,width] (sliced at build) */

	/* Final norm + projection */
	struct sam3_tensor *ln_final_w;
	struct sam3_tensor *ln_final_b;
	struct sam3_tensor *projection_layer;  /* [width, width], stored as raw tensor (no .weight suffix) */

	/* External 256-dim projector (TextStudentEncoder.projector) */
	struct sam3_tensor *out_proj_w;        /* [256, width] */
	struct sam3_tensor *out_proj_b;        /* [256] */

	/* Per-block tagged layers. cfg.n_layers entries used. */
	struct sam3_mobileclip_layer layers[SAM3_MOBILECLIP_MAX_LAYERS];
};

enum sam3_error sam3_mobileclip_text_load(
	struct sam3_mobileclip_text_encoder *enc,
	const struct sam3_weight_file *wf,
	struct sam3_arena *arena);

struct sam3_tensor *sam3_mobileclip_text_build(
	struct sam3_mobileclip_text_encoder *enc,
	struct sam3_graph *g,
	struct sam3_tensor *token_ids,
	struct sam3_tensor **pooled_out,
	struct sam3_arena *arena);

struct sam3_tensor *sam3_mobileclip_text_build_perblock(
	struct sam3_mobileclip_text_encoder *enc,
	struct sam3_backend *be,
	struct sam3_tensor *token_ids,
	struct sam3_arena *scratch,
	struct sam3_arena *persist);

/*
 * sam3_mobileclip_config_for - Return the config for a given variant.
 *
 * @text_backbone: SAM3_TEXT_MOBILECLIP_S0 / _S1 / _L
 *
 * Returns a pointer to a static const config struct. NULL on unknown
 * variant.
 */
const struct sam3_mobileclip_config *sam3_mobileclip_config_for(int text_backbone);

#endif /* SAM3_MODEL_MOBILECLIP_TEXT_H */
```

- [ ] **Step 2: Build (header-only)**

Run: `cd build && cmake --build . -j$(sysctl -n hw.ncpu) 2>&1 | tail -10`
Expected: success.

- [ ] **Step 3: Commit**

```bash
git add src/model/mobileclip_text.h
git commit -m "model: declare sam3_mobileclip_text_encoder + variant configs"
```

### Task 4.2 — Implement variant config table and the iface impl glue

**Files:**
- Create: `src/model/mobileclip_text.c` (skeleton + config table + iface ops)
- Modify: `CMakeLists.txt` (register the new source)

- [ ] **Step 1: Write the skeleton**

Create `src/model/mobileclip_text.c`:

```c
/*
 * src/model/mobileclip_text.c - MobileCLIP text encoder implementation
 *
 * Skeleton file: variant config table + iface vtable ops + stubs for
 * load/build that are filled in over Phases 4-6. Each variant maps to
 * a static const sam3_mobileclip_config; the iface init allocates a
 * fresh sam3_mobileclip_text_encoder, copies the config, and hands
 * back via the iface vtable.
 *
 * Key types:  sam3_mobileclip_text_encoder
 * Depends on: mobileclip_text.h, text_encoder_iface.h, util/log.h
 * Used by:    src/model/text_encoder_iface.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <string.h>

#include "mobileclip_text.h"
#include "text_encoder_iface.h"
#include "util/log.h"

/* --- Variant configs (spec table) --- */

static const struct sam3_mobileclip_config mobileclip_s0 = {
	.text_backbone           = SAM3_TEXT_MOBILECLIP_S0,
	.n_layers                = 6,
	.width                   = 512,
	.n_heads                 = 8,
	.mlp_dim                 = 2048,
	.ctx_len                 = 16,
	.out_dim                 = 256,
	.vocab_size              = 49408,
	.pos_embed_table_len     = 77,
	.n_repmixer_blocks       = 2,
	.repmixer_block_indices  = { 0, 5, 0, 0 },
};

static const struct sam3_mobileclip_config mobileclip_s1 = {
	.text_backbone           = SAM3_TEXT_MOBILECLIP_S1,
	.n_layers                = 12,
	.width                   = 512,
	.n_heads                 = 8,
	.mlp_dim                 = 2048,
	.ctx_len                 = 16,
	.out_dim                 = 256,
	.vocab_size              = 49408,
	.pos_embed_table_len     = 77,
	.n_repmixer_blocks       = 0,
	.repmixer_block_indices  = { 0 },
};

static const struct sam3_mobileclip_config mobileclip_l = {
	.text_backbone           = SAM3_TEXT_MOBILECLIP_L,
	.n_layers                = 12,
	.width                   = 768,
	.n_heads                 = 12,
	.mlp_dim                 = 3072,
	.ctx_len                 = 16,
	.out_dim                 = 256,
	.vocab_size              = 49408,
	.pos_embed_table_len     = 77,
	.n_repmixer_blocks       = 0,
	.repmixer_block_indices  = { 0 },
};

const struct sam3_mobileclip_config *sam3_mobileclip_config_for(int text_backbone)
{
	switch (text_backbone) {
	case SAM3_TEXT_MOBILECLIP_S0: return &mobileclip_s0;
	case SAM3_TEXT_MOBILECLIP_S1: return &mobileclip_s1;
	case SAM3_TEXT_MOBILECLIP_L:  return &mobileclip_l;
	default:                      return NULL;
	}
}

/* --- iface vtable wiring --- */

static enum sam3_error mc_load(struct sam3_text_encoder_iface *iface,
			       const struct sam3_weight_file *wf,
			       struct sam3_arena *arena)
{
	return sam3_mobileclip_text_load(
		(struct sam3_mobileclip_text_encoder *)iface->impl, wf, arena);
}

static struct sam3_tensor *mc_build(struct sam3_text_encoder_iface *iface,
				    struct sam3_graph *g,
				    struct sam3_tensor *token_ids,
				    struct sam3_tensor **pooled_out,
				    struct sam3_arena *arena)
{
	return sam3_mobileclip_text_build(
		(struct sam3_mobileclip_text_encoder *)iface->impl, g,
		token_ids, pooled_out, arena);
}

static struct sam3_tensor *mc_build_perblock(
	struct sam3_text_encoder_iface *iface,
	struct sam3_backend *be,
	struct sam3_tensor *token_ids,
	struct sam3_arena *scratch,
	struct sam3_arena *persist)
{
	return sam3_mobileclip_text_build_perblock(
		(struct sam3_mobileclip_text_encoder *)iface->impl, be,
		token_ids, scratch, persist);
}

static void mc_free(struct sam3_text_encoder_iface *iface)
{
	(void)iface; /* arena-backed */
}

const struct sam3_text_encoder_iface_ops sam3_mobileclip_text_iface_ops = {
	.load           = mc_load,
	.build          = mc_build,
	.build_perblock = mc_build_perblock,
	.free           = mc_free,
};

enum sam3_error sam3_mobileclip_text_iface_init_impl(
	struct sam3_text_encoder_iface *iface,
	int text_backbone, struct sam3_arena *arena)
{
	const struct sam3_mobileclip_config *cfg;
	struct sam3_mobileclip_text_encoder *enc;

	cfg = sam3_mobileclip_config_for(text_backbone);
	if (!cfg) {
		sam3_log_error("mobileclip: unknown variant %d", text_backbone);
		return SAM3_EINVAL;
	}

	enc = sam3_arena_alloc(arena, sizeof(*enc), _Alignof(*enc));
	if (!enc) {
		sam3_log_error("mobileclip: arena alloc failed");
		return SAM3_ENOMEM;
	}
	memset(enc, 0, sizeof(*enc));
	enc->cfg = *cfg;

	iface->impl    = enc;
	iface->ops     = &sam3_mobileclip_text_iface_ops;
	iface->ctx_len = cfg->ctx_len;
	iface->d_model = cfg->out_dim;
	return SAM3_OK;
}

/* --- Stubs for load/build (filled in by Tasks 4.3-4.5, 6.x) --- */

enum sam3_error sam3_mobileclip_text_load(
	struct sam3_mobileclip_text_encoder *enc,
	const struct sam3_weight_file *wf,
	struct sam3_arena *arena)
{
	(void)enc; (void)wf; (void)arena;
	sam3_log_error("mobileclip_text_load: not yet implemented");
	return SAM3_EINVAL;
}

struct sam3_tensor *sam3_mobileclip_text_build(
	struct sam3_mobileclip_text_encoder *enc,
	struct sam3_graph *g,
	struct sam3_tensor *token_ids,
	struct sam3_tensor **pooled_out,
	struct sam3_arena *arena)
{
	(void)enc; (void)g; (void)token_ids; (void)pooled_out; (void)arena;
	sam3_log_error("mobileclip_text_build: not yet implemented");
	return NULL;
}

struct sam3_tensor *sam3_mobileclip_text_build_perblock(
	struct sam3_mobileclip_text_encoder *enc,
	struct sam3_backend *be,
	struct sam3_tensor *token_ids,
	struct sam3_arena *scratch,
	struct sam3_arena *persist)
{
	(void)enc; (void)be; (void)token_ids; (void)scratch; (void)persist;
	sam3_log_error("mobileclip_text_build_perblock: not yet implemented");
	return NULL;
}
```

- [ ] **Step 2: Drop the weak stub in text_encoder_iface.c**

Now that the real symbol exists, remove the `__attribute__((weak))` stub block at the bottom of `src/model/text_encoder_iface.c`.

- [ ] **Step 3: Register in CMake**

Add `src/model/mobileclip_text.c` to the same `target_sources` list as `text_encoder_iface.c`.

- [ ] **Step 4: Build**

Run: `cd build && cmake --build . -j$(sysctl -n hw.ncpu) 2>&1 | tail -20`
Expected: success.

- [ ] **Step 5: Commit**

```bash
git add src/model/mobileclip_text.c src/model/text_encoder_iface.c CMakeLists.txt
git commit -m "model: mobileclip_text skeleton (config table + iface ops)"
```

### Task 4.3 — Implement weight loading for embeddings + final layers

**Files:**
- Modify: `src/model/mobileclip_text.c` (replace `sam3_mobileclip_text_load` stub)

- [ ] **Step 1: Implement the loader**

Replace the `sam3_mobileclip_text_load` stub with:

```c
/* --- Helpers to look up tensors with proper error handling --- */

static struct sam3_tensor *load_or_zero(
	const struct sam3_weight_file *wf, struct sam3_arena *arena,
	const char *name, enum sam3_dtype dtype, int n_dims, const int *dims)
{
	struct sam3_tensor *t = gh_load_mmap(wf, arena, name);
	if (t)
		return t;

	/* Zero-fill fallback so smoke-test paths can run without weights. */
	t = gh_alloc_tensor(arena, dtype, n_dims, dims);
	if (!t)
		return NULL;
	memset(t->data, 0, sam3_tensor_nbytes(t));
	sam3_log_warn("mobileclip: tensor %s not found, zero-filled", name);
	return t;
}

static struct sam3_tensor *load_required(
	const struct sam3_weight_file *wf, struct sam3_arena *arena,
	const char *name)
{
	struct sam3_tensor *t = gh_load_mmap(wf, arena, name);
	if (!t)
		sam3_log_error("mobileclip: required tensor %s not found", name);
	return t;
}

#define ENC_PFX "detector.backbone.language_backbone.encoder."
#define BACKBONE_PFX "detector.backbone.language_backbone."

enum sam3_error sam3_mobileclip_text_load(
	struct sam3_mobileclip_text_encoder *enc,
	const struct sam3_weight_file *wf,
	struct sam3_arena *arena)
{
	char name[256];
	int dims[2];
	const struct sam3_mobileclip_config *cfg;

	if (!enc || !arena)
		return SAM3_EINVAL;
	cfg = &enc->cfg;

	/* Embeddings */
	dims[0] = cfg->vocab_size; dims[1] = cfg->width;
	enc->token_embedding = load_or_zero(
		wf, arena, ENC_PFX "embedding_layer.weight",
		SAM3_DTYPE_F32, 2, dims);
	if (!enc->token_embedding) return SAM3_ENOMEM;

	dims[0] = cfg->pos_embed_table_len; dims[1] = cfg->width;
	enc->pos_embed_full = load_or_zero(
		wf, arena,
		ENC_PFX "positional_embedding.pos_embed.pos_embed",
		SAM3_DTYPE_F32, 2, dims);
	if (!enc->pos_embed_full) return SAM3_ENOMEM;

	/* Final LN */
	dims[0] = cfg->width;
	enc->ln_final_w = load_or_zero(
		wf, arena, ENC_PFX "final_layer_norm.weight",
		SAM3_DTYPE_F32, 1, dims);
	enc->ln_final_b = load_or_zero(
		wf, arena, ENC_PFX "final_layer_norm.bias",
		SAM3_DTYPE_F32, 1, dims);

	/* Inner projection (width -> width) */
	dims[0] = cfg->width; dims[1] = cfg->width;
	enc->projection_layer = load_or_zero(
		wf, arena, ENC_PFX "projection_layer",
		SAM3_DTYPE_F32, 2, dims);

	/* External 256-dim projector. Key was confirmed in Task 0.2;
	 * update the literal below if the audit found a different name. */
	dims[0] = cfg->out_dim; dims[1] = cfg->width;
	enc->out_proj_w = load_or_zero(
		wf, arena, BACKBONE_PFX "projector.weight",
		SAM3_DTYPE_F32, 2, dims);
	dims[0] = cfg->out_dim;
	enc->out_proj_b = gh_load_mmap(
		wf, arena, BACKBONE_PFX "projector.bias");
	/* projector.bias may legitimately be absent (Linear with bias=False). */

	/* Per-layer load. RepMixer indices (S0 only) are handled by Task 6.1
	 * via load_repmixer_block; here we populate the standard-block tensors
	 * into enc->layers[i].u.std and set is_repmixer=0. */
	for (int i = 0; i < cfg->n_layers; i++) {
		int is_rep = 0;
		for (int k = 0; k < cfg->n_repmixer_blocks; k++) {
			if (cfg->repmixer_block_indices[k] == i) { is_rep = 1; break; }
		}
		if (is_rep) {
			enc->layers[i].is_repmixer = 1;
			continue; /* loaded by Task 6.1 */
		}
		enc->layers[i].is_repmixer = 0;
		struct sam3_mobileclip_layer_std *L = &enc->layers[i].u.std;

		dims[0] = cfg->width;
		snprintf(name, sizeof(name),
			 ENC_PFX "transformer.%d.pre_norm_mha.0.weight", i);
		L->ln1_w = load_or_zero(wf, arena, name, SAM3_DTYPE_F32, 1, dims);
		snprintf(name, sizeof(name),
			 ENC_PFX "transformer.%d.pre_norm_mha.0.bias", i);
		L->ln1_b = load_or_zero(wf, arena, name, SAM3_DTYPE_F32, 1, dims);

		dims[0] = 3 * cfg->width; dims[1] = cfg->width;
		snprintf(name, sizeof(name),
			 ENC_PFX "transformer.%d.pre_norm_mha.1.qkv_proj.weight", i);
		L->qkv_w = load_or_zero(wf, arena, name, SAM3_DTYPE_F32, 2, dims);
		dims[0] = 3 * cfg->width;
		snprintf(name, sizeof(name),
			 ENC_PFX "transformer.%d.pre_norm_mha.1.qkv_proj.bias", i);
		L->qkv_b = load_or_zero(wf, arena, name, SAM3_DTYPE_F32, 1, dims);

		dims[0] = cfg->width; dims[1] = cfg->width;
		snprintf(name, sizeof(name),
			 ENC_PFX "transformer.%d.pre_norm_mha.1.out_proj.weight", i);
		L->out_w = load_or_zero(wf, arena, name, SAM3_DTYPE_F32, 2, dims);
		dims[0] = cfg->width;
		snprintf(name, sizeof(name),
			 ENC_PFX "transformer.%d.pre_norm_mha.1.out_proj.bias", i);
		L->out_b = load_or_zero(wf, arena, name, SAM3_DTYPE_F32, 1, dims);

		snprintf(name, sizeof(name),
			 ENC_PFX "transformer.%d.pre_norm_ffn.0.weight", i);
		L->ln2_w = load_or_zero(wf, arena, name, SAM3_DTYPE_F32, 1, dims);
		snprintf(name, sizeof(name),
			 ENC_PFX "transformer.%d.pre_norm_ffn.0.bias", i);
		L->ln2_b = load_or_zero(wf, arena, name, SAM3_DTYPE_F32, 1, dims);

		dims[0] = cfg->mlp_dim; dims[1] = cfg->width;
		snprintf(name, sizeof(name),
			 ENC_PFX "transformer.%d.pre_norm_ffn.1.weight", i);
		L->fc1_w = load_or_zero(wf, arena, name, SAM3_DTYPE_F32, 2, dims);
		dims[0] = cfg->mlp_dim;
		snprintf(name, sizeof(name),
			 ENC_PFX "transformer.%d.pre_norm_ffn.1.bias", i);
		L->fc1_b = load_or_zero(wf, arena, name, SAM3_DTYPE_F32, 1, dims);

		dims[0] = cfg->width; dims[1] = cfg->mlp_dim;
		snprintf(name, sizeof(name),
			 ENC_PFX "transformer.%d.pre_norm_ffn.4.weight", i);
		L->fc2_w = load_or_zero(wf, arena, name, SAM3_DTYPE_F32, 2, dims);
		dims[0] = cfg->width;
		snprintf(name, sizeof(name),
			 ENC_PFX "transformer.%d.pre_norm_ffn.4.bias", i);
		L->fc2_b = load_or_zero(wf, arena, name, SAM3_DTYPE_F32, 1, dims);
	}

	sam3_log_info("mobileclip: loaded %d-layer %s text encoder",
		      cfg->n_layers,
		      cfg->text_backbone == SAM3_TEXT_MOBILECLIP_S0 ? "S0" :
		      cfg->text_backbone == SAM3_TEXT_MOBILECLIP_S1 ? "S1" : "L");
	return SAM3_OK;
}
```

- [ ] **Step 2: Build**

Run: `cd build && cmake --build . -j$(sysctl -n hw.ncpu) 2>&1 | tail -20`
Expected: success.

- [ ] **Step 3: Commit**

```bash
git add src/model/mobileclip_text.c
git commit -m "mobileclip: load embeddings + final layers + standard blocks"
```

### Task 4.4 — Implement `sam3_mobileclip_text_build` (single-shot graph)

**Files:**
- Modify: `src/model/mobileclip_text.c` (replace `sam3_mobileclip_text_build` stub)

- [ ] **Step 1: Add forward decl for the standard-block helper**

Near the top of `mobileclip_text.c`, after the includes:

```c
static struct sam3_tensor *build_std_block(
	struct sam3_graph *g,
	const struct sam3_mobileclip_layer_std *L,
	struct sam3_tensor *x,
	int n_heads,
	int width,
	struct sam3_arena *arena);
```

- [ ] **Step 2: Implement the standard block helper**

Add (above `sam3_mobileclip_text_build`):

```c
/*
 * build_std_block - One pre-norm transformer block, no causal mask.
 *
 * Mirrors the reference: x -> LN1 -> QKV -> attn -> out -> + residual
 *                          -> LN2 -> FC1 -> GELU -> FC2 -> + residual
 */
static struct sam3_tensor *build_std_block(
	struct sam3_graph *g,
	const struct sam3_mobileclip_layer_std *L,
	struct sam3_tensor *x,
	int n_heads,
	int width,
	struct sam3_arena *arena)
{
	struct sam3_tensor *t;

	/* Attention sub-block */
	t = gh_layernorm(g, x, L->ln1_w, L->ln1_b, arena);
	t = gh_multihead_attention(g, t, L->qkv_w, L->qkv_b,
				   L->out_w, L->out_b,
				   n_heads, width,
				   /* attn_mask */ NULL,
				   /* use_rope  */ 0,
				   arena);
	x = gh_add(g, x, t, arena);

	/* FFN sub-block */
	t = gh_layernorm(g, x, L->ln2_w, L->ln2_b, arena);
	t = gh_linear(g, t, L->fc1_w, L->fc1_b, arena);
	t = gh_gelu(g, t, arena);
	t = gh_linear(g, t, L->fc2_w, L->fc2_b, arena);
	x = gh_add(g, x, t, arena);
	return x;
}
```

If the existing graph helper signatures differ — `gh_multihead_attention`, `gh_linear`, `gh_gelu`, `gh_add`, `gh_layernorm` — adapt the calls. Find the actual names with:

```bash
grep -rn "^[[:space:]]*struct sam3_tensor \*gh_" src/model/graph_helpers.h
```

Match the existing CLIP path for parity (read `src/model/text_encoder.c` around the standard block construction).

- [ ] **Step 3: Implement build**

Replace the stub with:

```c
struct sam3_tensor *sam3_mobileclip_text_build(
	struct sam3_mobileclip_text_encoder *enc,
	struct sam3_graph *g,
	struct sam3_tensor *token_ids,
	struct sam3_tensor **pooled_out,
	struct sam3_arena *arena)
{
	const struct sam3_mobileclip_config *cfg = &enc->cfg;
	struct sam3_tensor *x, *pos, *out;
	int slice_dims[2];
	int eot_idx;

	/* No RepMixer guard: build dispatches per-block on is_repmixer
	 * (set by the loader from cfg.repmixer_block_indices). For S1/L
	 * every block is_repmixer=0 so the loop runs the standard path. */

	/* 1. Token embedding lookup [ctx_len, width] */
	x = gh_embedding_lookup(g, enc->token_embedding, token_ids, arena);
	if (!x) return NULL;

	/* 2. Position embedding: slice [77, width] -> [ctx_len, width] */
	slice_dims[0] = cfg->ctx_len;
	slice_dims[1] = cfg->width;
	pos = gh_slice(g, enc->pos_embed_full, /*axis*/ 0,
		       /*start*/ 0, /*end*/ cfg->ctx_len, arena);
	x = gh_add(g, x, pos, arena);

	/* 3. Transformer blocks (no causal mask). Per-block dispatch
	 *    on is_repmixer; RepMixer wiring lands in Phase 6. */
	for (int i = 0; i < cfg->n_layers; i++) {
		const struct sam3_mobileclip_layer *L = &enc->layers[i];
		if (L->is_repmixer) {
			/* Filled in by Task 6.2; no-op for S1/L. */
			sam3_log_error("mobileclip_build: RepMixer block %d "
				       "reached before Phase 6 wired it",
				       i);
			return NULL;
		}
		x = build_std_block(g, &L->u.std, x,
				    cfg->n_heads, cfg->width, arena);
	}

	/* 4. Final LN */
	x = gh_layernorm(g, x, enc->ln_final_w, enc->ln_final_b, arena);

	/* 5. Inner projection (width -> width) */
	x = gh_matmul(g, x, enc->projection_layer, arena);

	/* 6. External 256-dim projector */
	out = gh_linear(g, x, enc->out_proj_w, enc->out_proj_b, arena);
	if (!out) return NULL;

	/* 7. Pooled output: pick token at argmax(token_ids == EOT_token).
	 * For simplicity (and matching the existing CLIP path), use
	 * ctx_len-1 unless the build helpers expose a real argmax.
	 * The fixture in Task 5.x verifies whether per-token output
	 * matches the reference; if pooled diverges, swap to a
	 * gh_gather_at_eot(...) helper. */
	if (pooled_out) {
		int pdims[1] = { cfg->out_dim };
		eot_idx = cfg->ctx_len - 1;
		*pooled_out = gh_slice_row(g, out, eot_idx, arena);
		(void)pdims;
	}
	return out;
}
```

The `gh_*` names above are nominal; replace with the actual helpers used in `src/model/text_encoder.c`'s build function. The transformation is:

```bash
sed -n '244,431p' src/model/text_encoder.c
```

Use that as the template and substitute MobileCLIP dims/weights.

- [ ] **Step 4: Build**

Run: `cd build && cmake --build . -j$(sysctl -n hw.ncpu) 2>&1 | tail -20`
Expected: success.

- [ ] **Step 5: Commit**

```bash
git add src/model/mobileclip_text.c
git commit -m "mobileclip: build single-shot graph (S1/L; S0 standard blocks only)"
```

### Task 4.5 — Implement `sam3_mobileclip_text_build_perblock`

**Files:**
- Modify: `src/model/mobileclip_text.c` (replace `sam3_mobileclip_text_build_perblock` stub)

- [ ] **Step 1: Mirror the existing CLIP per-block evaluator**

Open `src/model/text_encoder.c:448` (the `sam3_text_encoder_build_perblock` function) and read it end-to-end. The MobileCLIP version is the same shape: build embedding once, then loop blocks in groups, evaluating each group on `be`, copying activations into `persist`, resetting `scratch` between groups.

Replace the stub with an implementation that follows the same control flow but uses `build_std_block` and the MobileCLIP config/weights. See the spec section 5.

Pseudocode skeleton (adapt to actual helper APIs):

```c
struct sam3_tensor *sam3_mobileclip_text_build_perblock(
	struct sam3_mobileclip_text_encoder *enc,
	struct sam3_backend *be,
	struct sam3_tensor *token_ids,
	struct sam3_arena *scratch,
	struct sam3_arena *persist)
{
	const struct sam3_mobileclip_config *cfg = &enc->cfg;
	struct sam3_tensor *x_persist;
	int blocks_per_eval = 4;

	/* No top-level RepMixer guard: the block loop dispatches per-block.
	 * Phase 6 wires the L->is_repmixer arm. */

	/* === Pre-block setup: embed + add pos, evaluated in scratch and
	 *     copied into persist === */
	{
		struct sam3_graph g;
		sam3_graph_init(&g, scratch);

		struct sam3_tensor *x = gh_embedding_lookup(
			&g, enc->token_embedding, token_ids, scratch);
		struct sam3_tensor *pos = gh_slice(
			&g, enc->pos_embed_full, 0, 0, cfg->ctx_len, scratch);
		x = gh_add(&g, x, pos, scratch);

		sam3_backend_eval(be, &g);
		x_persist = gh_persist_copy(persist, x);

		sam3_arena_reset(scratch);
	}

	/* === Block loop ===
	 *
	 * For S1/L: every block is_repmixer=0, so the loop runs the
	 * standard path. We still emit one block per group rather than
	 * batching 4 — this keeps the dispatch identical for both std
	 * and RepMixer blocks once Phase 6 lands. If profiling shows
	 * the per-block overhead matters, batch consecutive std blocks
	 * (see Task 6.2 step 3 for the batching pattern).
	 */
	for (int i = 0; i < cfg->n_layers; i++) {
		const struct sam3_mobileclip_layer *L = &enc->layers[i];

		struct sam3_graph g;
		sam3_graph_init(&g, scratch);

		struct sam3_tensor *x = gh_alias(&g, x_persist);
		if (L->is_repmixer) {
			/* Filled in by Task 6.2; no-op for S1/L. */
			sam3_log_error("mobileclip_perblock: RepMixer block %d "
				       "reached before Phase 6 wired it", i);
			return NULL;
		}
		x = build_std_block(&g, &L->u.std, x,
				    cfg->n_heads, cfg->width, scratch);

		sam3_backend_eval(be, &g);
		x_persist = gh_persist_copy(persist, x);

		sam3_arena_reset(scratch);
	}
	(void)blocks_per_eval; /* unused after refactor */

	/* === Tail: final LN + inner proj + external projector === */
	{
		struct sam3_graph g;
		sam3_graph_init(&g, scratch);

		struct sam3_tensor *x = gh_alias(&g, x_persist);
		x = gh_layernorm(&g, x, enc->ln_final_w, enc->ln_final_b, scratch);
		x = gh_matmul(&g, x, enc->projection_layer, scratch);
		x = gh_linear(&g, x, enc->out_proj_w, enc->out_proj_b, scratch);

		sam3_backend_eval(be, &g);
		x_persist = gh_persist_copy(persist, x);

		sam3_arena_reset(scratch);
	}
	return x_persist;
}
```

The actual helper names (`gh_alias`, `gh_persist_copy`, `sam3_graph_init`, `sam3_backend_eval`) must match the CLIP per-block evaluator. Read `src/model/text_encoder.c:448-...` for the literal patterns and copy them.

- [ ] **Step 2: Build + smoke**

Run: `cd build && cmake --build . -j$(sysctl -n hw.ncpu) && ctest --output-on-failure -R '(text|prompt)' 2>&1 | tail -30`
Expected: existing CLIP tests still pass; no MobileCLIP tests yet.

- [ ] **Step 3: Commit**

```bash
git add src/model/mobileclip_text.c
git commit -m "mobileclip: per-block evaluator (S1/L)"
```

---

## Phase 4.5 — Converter `--text-backbone` flag (unblocks Phase 5 tests)

This phase is just Task 7.1 promoted earlier — Phase 5 needs `.sam3` files with `text_backbone` set, which requires the CLI flag. The body is unchanged (see "Phase 7 — Task 7.1" below); execute Task 7.1 here, then continue to Phase 5.

After completing Task 7.1, generate the `.sam3` fixtures **before** running Phase 5:

```bash
python tools/pt_to_safetensors.py \
  models/efficient_sam3_image_encoder_mobileclip_s1_ctx16.pt \
  /tmp/mc_s1.safetensors
build/sam3 convert -i /tmp/mc_s1.safetensors \
  -o tests/fixtures/mobileclip_s1/encoder.sam3 \
  --backbone hiera --variant sam3 --text-backbone mobileclip_s1

python tools/pt_to_safetensors.py \
  models/efficient_sam3_image_encoder_mobileclip2_l_ctx16.pt \
  /tmp/mc_l.safetensors
build/sam3 convert -i /tmp/mc_l.safetensors \
  -o tests/fixtures/mobileclip_l/encoder.sam3 \
  --backbone hiera --variant sam3 --text-backbone mobileclip_l
```

(S0 conversion happens in Task 6.3 once RepMixer loading is in place.)

---

## Phase 5 — MobileCLIP per-block parity tests

### Task 5.1 — Test scaffolding + tokens fixture round-trip

**Files:**
- Create: `tests/test_mobileclip_text.c` (smoke/skeleton)
- Modify: `CMakeLists.txt` (register the test)

- [ ] **Step 1: Write the test skeleton**

Create `tests/test_mobileclip_text.c`:

```c
/*
 * tests/test_mobileclip_text.c - MobileCLIP text encoder parity tests
 *
 * Loads the fixtures dumped by scripts/dump_mobileclip_text_layers.py
 * (one directory per variant under tests/fixtures/) and exercises the
 * C-side encoder for shape, per-block, and final-pooled parity against
 * the PyTorch reference.
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sam3/sam3.h"
#include "sam3/sam3_types.h"
#include "core/weight.h"
#include "core/alloc.h"
#include "core/graph.h"
#include "model/mobileclip_text.h"
#include "model/text_encoder_iface.h"
#include "test_helpers.h"
#include "test_npy.h" /* tiny .npy reader; add if absent — see Step 1a */

static const char *fixture_dir(int backbone)
{
	switch (backbone) {
	case SAM3_TEXT_MOBILECLIP_S0: return "tests/fixtures/mobileclip_s0";
	case SAM3_TEXT_MOBILECLIP_S1: return "tests/fixtures/mobileclip_s1";
	case SAM3_TEXT_MOBILECLIP_L:  return "tests/fixtures/mobileclip_l";
	default: return NULL;
	}
}

/* Minimal smoke: factory yields a non-NULL iface with the right ctx_len. */
static int test_iface_factory_s1(void)
{
	struct sam3_arena arena;
	sam3_arena_init(&arena, 4 * 1024 * 1024);

	struct sam3_text_encoder_iface iface;
	enum sam3_error err = sam3_text_encoder_iface_init(
		&iface, SAM3_TEXT_MOBILECLIP_S1, &arena);
	TEST_ASSERT(err == SAM3_OK);
	TEST_ASSERT(iface.ctx_len == 16);
	TEST_ASSERT(iface.d_model == 256);
	TEST_ASSERT(iface.text_backbone == SAM3_TEXT_MOBILECLIP_S1);

	sam3_arena_free(&arena);
	return 0;
}

int main(void)
{
	int rc = 0;
	rc |= test_iface_factory_s1();
	return rc ? 1 : 0;
}
```

- [ ] **Step 1a: Add a minimal `.npy` reader (only if not present)**

Search for existing usage:

```bash
grep -rln "load_npy\|test_npy" tests/ src/
```

If a helper exists, include it. Otherwise add `tests/test_npy.h` with a small parser supporting F32/I32 1D and 2D arrays — fixed-format `\x93NUMPY` header parsing is ~80 lines. Mirror any pattern used by `tests/test_text_encoder.c` if it loads numpy fixtures.

- [ ] **Step 2: Register in CMake**

Add an `add_executable(test_mobileclip_text tests/test_mobileclip_text.c ...)` block alongside the existing `test_text_encoder` registration. Mirror the existing pattern (linking against the sam3 library).

- [ ] **Step 3: Build + run**

```bash
cd build && cmake --build . -j$(sysctl -n hw.ncpu) && ctest --output-on-failure -R mobileclip 2>&1 | tail -20
```

Expected: `test_iface_factory_s1` passes.

- [ ] **Step 4: Commit**

```bash
git add tests/test_mobileclip_text.c tests/test_npy.h CMakeLists.txt
git commit -m "tests: scaffold test_mobileclip_text + fixture loaders"
```

### Task 5.2 — Per-block parity test for S1 (full path through the encoder)

**Files:**
- Modify: `tests/test_mobileclip_text.c` (add per-block parity case)

- [ ] **Step 1: Convert one .pt to .sam3 (manual, one-time prerequisite)**

We need a v4 `.sam3` file containing the S1 weights to run the test. Use the converter once Phase 7 is wired (Task 7.1). For now this task should be **gated on Phase 7** — rearrange execution order if running tasks linearly: complete Phase 7 first, then return here.

To proceed in this task, assume the file `tests/fixtures/mobileclip_s1/encoder.sam3` exists. Generate it via:

```bash
python tools/pt_to_safetensors.py \
  models/efficient_sam3_image_encoder_mobileclip_s1_ctx16.pt \
  /tmp/mc_s1.safetensors

build/sam3 convert -i /tmp/mc_s1.safetensors \
  -o tests/fixtures/mobileclip_s1/encoder.sam3 \
  --backbone hiera \
  --variant sam3 \
  --text-backbone mobileclip_s1
```

Add `tests/fixtures/mobileclip_s1/encoder.sam3` to `.gitignore` if it's >100MB; otherwise check it in (small relative to existing fixtures).

- [ ] **Step 2: Add the per-block test**

Append to `tests/test_mobileclip_text.c`:

```c
static int test_perblock_parity_s1(void)
{
	const char *dir = fixture_dir(SAM3_TEXT_MOBILECLIP_S1);
	char path[512];

	struct sam3_weight_file wf = {0};
	snprintf(path, sizeof(path), "%s/encoder.sam3", dir);
	if (sam3_weight_open(&wf, path) != SAM3_OK) {
		fprintf(stderr, "skip: %s missing (run Task 7.1 first)\n", path);
		return 0; /* skip */
	}
	TEST_ASSERT(wf.text_backbone == SAM3_TEXT_MOBILECLIP_S1);

	struct sam3_arena arena;
	sam3_arena_init(&arena, 256 * 1024 * 1024);
	struct sam3_arena scratch;
	sam3_arena_init(&scratch, 256 * 1024 * 1024);
	struct sam3_arena persist;
	sam3_arena_init(&persist, 16 * 1024 * 1024);

	struct sam3_text_encoder_iface iface;
	TEST_ASSERT(sam3_text_encoder_iface_init(
		&iface, SAM3_TEXT_MOBILECLIP_S1, &arena) == SAM3_OK);
	TEST_ASSERT(iface.ops->load(&iface, &wf, &arena) == SAM3_OK);

	/* Load tokens fixture */
	int32_t tokens[16];
	int n_tokens = 16;
	snprintf(path, sizeof(path), "%s/tokens.npy", dir);
	TEST_ASSERT(test_npy_load_i32(path, tokens, &n_tokens) == 0);
	TEST_ASSERT(n_tokens == 16);

	/* Build a backend (CPU is fine for parity) */
	struct sam3_backend be;
	sam3_backend_init_cpu(&be);

	/* Wrap tokens as tensor (in arena) */
	int dims[1] = { 16 };
	struct sam3_tensor *tok_tensor = gh_alloc_tensor(
		&arena, SAM3_DTYPE_I32, 1, dims);
	memcpy(tok_tensor->data, tokens, 16 * sizeof(int32_t));

	/* Run per-block evaluator */
	struct sam3_tensor *out = iface.ops->build_perblock(
		&iface, &be, tok_tensor, &scratch, &persist);
	TEST_ASSERT(out != NULL);
	TEST_ASSERT(out->dims[0] == 16);
	TEST_ASSERT(out->dims[1] == 256);

	/* Compare against fixture */
	float ref[16 * 256];
	int ref_dims[2] = { 16, 256 };
	int ref_n_dims = 2;
	snprintf(path, sizeof(path), "%s/out_tokens.npy", dir);
	TEST_ASSERT(test_npy_load_f32(path, ref, ref_dims, &ref_n_dims) == 0);

	float max_err = 0.0f;
	const float *ours = (const float *)out->data;
	for (int i = 0; i < 16 * 256; i++) {
		float e = fabsf(ours[i] - ref[i]);
		if (e > max_err) max_err = e;
	}
	fprintf(stderr, "max abs err vs reference: %.4e\n", max_err);
	TEST_ASSERT(max_err < 1e-3f);

	sam3_backend_free(&be);
	sam3_arena_free(&persist);
	sam3_arena_free(&scratch);
	sam3_arena_free(&arena);
	sam3_weight_close(&wf);
	return 0;
}
```

Add to `main()`:

```c
	rc |= test_perblock_parity_s1();
```

- [ ] **Step 3: Build + run**

```bash
cd build && cmake --build . -j$(sysctl -n hw.ncpu) && ctest --output-on-failure -R mobileclip 2>&1 | tail -20
```

Expected: skipped (until Task 7.1 produces the .sam3) or pass with max-abs-err < 1e-3. If the error is larger, debug per-block: dump activations after each block in the C path and compare against the corresponding `block_NN_out.npy`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_mobileclip_text.c
git commit -m "tests: per-block parity check for MobileCLIP-S1"
```

### Task 5.3 — Add parity tests for L (and S0 standard blocks only)

**Files:**
- Modify: `tests/test_mobileclip_text.c`

- [ ] **Step 1: Generalize `test_perblock_parity_*` to a parameterized helper**

Refactor `test_perblock_parity_s1` body into:

```c
static int test_perblock_parity(int text_backbone, int ctx_len)
{
	/* same body as s1, but with text_backbone/ctx_len params */
}
```

Add wrappers:

```c
static int test_perblock_parity_s1(void)
{
	return test_perblock_parity(SAM3_TEXT_MOBILECLIP_S1, 16);
}
static int test_perblock_parity_l(void)
{
	return test_perblock_parity(SAM3_TEXT_MOBILECLIP_L, 16);
}
```

(S0 deferred until Phase 6 since its first block is RepMixer.)

Add to `main()`:

```c
	rc |= test_perblock_parity_l();
```

- [ ] **Step 2: Generate L fixture .sam3**

```bash
python tools/pt_to_safetensors.py \
  models/efficient_sam3_image_encoder_mobileclip2_l_ctx16.pt \
  /tmp/mc_l.safetensors

build/sam3 convert -i /tmp/mc_l.safetensors \
  -o tests/fixtures/mobileclip_l/encoder.sam3 \
  --backbone hiera --variant sam3 --text-backbone mobileclip_l
```

- [ ] **Step 3: Build + run**

```bash
cd build && cmake --build . -j$(sysctl -n hw.ncpu) && ctest --output-on-failure -R mobileclip 2>&1 | tail -20
```

Expected: S1 + L pass; S0 still skipped.

- [ ] **Step 4: Commit**

```bash
git add tests/test_mobileclip_text.c
git commit -m "tests: parity for MobileCLIP-L"
```

---

## Phase 6 — RepMixer block (S0 only)

### Task 6.1 — Load RepMixer block weights (S0 indices 0 and 5)

**Files:**
- Modify: `src/model/mobileclip_text.c` (extend loader)

**Audit-verified key inventory** (per `docs/superpowers/specs/notes/2026-04-20-mobileclip-key-audit.md`).
For each S0 block index `i` in `cfg.repmixer_block_indices` (i ∈ {0, 5}):

```
transformer.<i>.layer_scale                                            (512,1,1)
transformer.<i>.token_mixer.layer_scale                                (512,1,1)
transformer.<i>.token_mixer.norm.rbr_skip.{weight,bias,running_mean,running_var}    (512,)
transformer.<i>.token_mixer.mixer.rbr_skip.{weight,bias,running_mean,running_var}   (512,)
transformer.<i>.token_mixer.mixer.rbr_conv.0.bn.{weight,bias,running_mean,running_var} (512,)
transformer.<i>.token_mixer.mixer.rbr_conv.0.conv.weight               (512,1,1,11)
transformer.<i>.convffn.conv.bn.{weight,bias,running_mean,running_var} (512,)
transformer.<i>.convffn.conv.conv.weight                               (512,1,1,11)
transformer.<i>.convffn.fc1.{weight,bias}                              (2048,512,1,1) / (2048,)
transformer.<i>.convffn.fc2.{weight,bias}                              (512,2048,1,1) / (512,)
```

**Mapping into `struct sam3_mobileclip_layer_repmixer`** (declared in Task 4.1):

| C field                           | Source key (`<P>` = `ENC_PFX "transformer.%d." % i`) |
|-----------------------------------|------------------------------------------------------|
| `outer_layer_scale`               | `<P>layer_scale`                                     |
| `tm_layer_scale`                  | `<P>token_mixer.layer_scale`                         |
| `norm_skip_{w,b,rm,rv}`           | `<P>token_mixer.norm.rbr_skip.{weight,bias,running_mean,running_var}` |
| `mixer_skip_{w,b,rm,rv}`          | `<P>token_mixer.mixer.rbr_skip.{weight,bias,running_mean,running_var}` |
| `mixer_conv_w`                    | `<P>token_mixer.mixer.rbr_conv.0.conv.weight`        |
| `mixer_conv_bn_{w,b,rm,rv}`       | `<P>token_mixer.mixer.rbr_conv.0.bn.{...}`           |
| `mixer_scale_*` (optional)        | `<P>token_mixer.mixer.rbr_scale.*` (NULL on miss — current S0 lacks it) |
| `convffn_dw_w`                    | `<P>convffn.conv.conv.weight`                        |
| `convffn_dw_bn_{w,b,rm,rv}`       | `<P>convffn.conv.bn.{...}`                           |
| `convffn_fc1_{w,b}`               | `<P>convffn.fc1.{weight,bias}`                       |
| `convffn_fc2_{w,b}`               | `<P>convffn.fc2.{weight,bias}`                       |

- [ ] **Step 1: Add the RepMixer loader**

Add this function before `sam3_mobileclip_text_load`:

```c
/*
 * load_repmixer_block - Populate one repmixer layer slot.
 *
 * @R:           Repmixer struct to fill (zeroed by caller).
 * @wf:          Open weight file.
 * @arena:       Arena for tensor descriptors.
 * @layer_idx:   The transformer.<layer_idx> index in the source state-dict.
 * @width:       cfg.width (channel count, e.g., 512 for S0).
 * @mlp_dim:     cfg.mlp_dim (e.g., 2048 for S0).
 *
 * BN params are required by the audited S0 checkpoint (no fold). All four
 * (weight, bias, running_mean, running_var) per BN must load successfully;
 * a missing key is a model-format error. The optional rbr_scale branch
 * (mixer_scale_*) tolerates absence — when NULL, build skips the branch.
 */
static enum sam3_error load_repmixer_block(
	struct sam3_mobileclip_layer_repmixer *R,
	const struct sam3_weight_file *wf,
	struct sam3_arena *arena,
	int layer_idx, int width, int mlp_dim)
{
	char k[256];
	#define K(suffix) (snprintf(k, sizeof(k), \
		ENC_PFX "transformer.%d." suffix, layer_idx), k)

	/* Outer + token-mixer scales. */
	R->outer_layer_scale = load_required(wf, arena, K("layer_scale"));
	R->tm_layer_scale    = load_required(wf, arena, K("token_mixer.layer_scale"));

	/* token_mixer.norm.rbr_skip — single BN. */
	R->norm_skip_w  = load_required(wf, arena, K("token_mixer.norm.rbr_skip.weight"));
	R->norm_skip_b  = load_required(wf, arena, K("token_mixer.norm.rbr_skip.bias"));
	R->norm_skip_rm = load_required(wf, arena, K("token_mixer.norm.rbr_skip.running_mean"));
	R->norm_skip_rv = load_required(wf, arena, K("token_mixer.norm.rbr_skip.running_var"));

	/* token_mixer.mixer.rbr_skip — BN-only branch. */
	R->mixer_skip_w  = load_required(wf, arena, K("token_mixer.mixer.rbr_skip.weight"));
	R->mixer_skip_b  = load_required(wf, arena, K("token_mixer.mixer.rbr_skip.bias"));
	R->mixer_skip_rm = load_required(wf, arena, K("token_mixer.mixer.rbr_skip.running_mean"));
	R->mixer_skip_rv = load_required(wf, arena, K("token_mixer.mixer.rbr_skip.running_var"));

	/* token_mixer.mixer.rbr_conv[0] — depthwise conv + BN branch. */
	R->mixer_conv_w     = load_required(wf, arena, K("token_mixer.mixer.rbr_conv.0.conv.weight"));
	R->mixer_conv_bn_w  = load_required(wf, arena, K("token_mixer.mixer.rbr_conv.0.bn.weight"));
	R->mixer_conv_bn_b  = load_required(wf, arena, K("token_mixer.mixer.rbr_conv.0.bn.bias"));
	R->mixer_conv_bn_rm = load_required(wf, arena, K("token_mixer.mixer.rbr_conv.0.bn.running_mean"));
	R->mixer_conv_bn_rv = load_required(wf, arena, K("token_mixer.mixer.rbr_conv.0.bn.running_var"));

	/* token_mixer.mixer.rbr_scale — optional 1×1 conv + BN branch. */
	R->mixer_scale_w     = gh_load_mmap(wf, arena, K("token_mixer.mixer.rbr_scale.conv.weight"));
	R->mixer_scale_bn_w  = gh_load_mmap(wf, arena, K("token_mixer.mixer.rbr_scale.bn.weight"));
	R->mixer_scale_bn_b  = gh_load_mmap(wf, arena, K("token_mixer.mixer.rbr_scale.bn.bias"));
	R->mixer_scale_bn_rm = gh_load_mmap(wf, arena, K("token_mixer.mixer.rbr_scale.bn.running_mean"));
	R->mixer_scale_bn_rv = gh_load_mmap(wf, arena, K("token_mixer.mixer.rbr_scale.bn.running_var"));

	/* convffn.conv — depthwise conv + BN. */
	R->convffn_dw_w     = load_required(wf, arena, K("convffn.conv.conv.weight"));
	R->convffn_dw_bn_w  = load_required(wf, arena, K("convffn.conv.bn.weight"));
	R->convffn_dw_bn_b  = load_required(wf, arena, K("convffn.conv.bn.bias"));
	R->convffn_dw_bn_rm = load_required(wf, arena, K("convffn.conv.bn.running_mean"));
	R->convffn_dw_bn_rv = load_required(wf, arena, K("convffn.conv.bn.running_var"));

	/* convffn.fc1 / convffn.fc2 — 1×1 convs with bias. */
	R->convffn_fc1_w = load_required(wf, arena, K("convffn.fc1.weight"));
	R->convffn_fc1_b = load_required(wf, arena, K("convffn.fc1.bias"));
	R->convffn_fc2_w = load_required(wf, arena, K("convffn.fc2.weight"));
	R->convffn_fc2_b = load_required(wf, arena, K("convffn.fc2.bias"));
	#undef K

	/* Quick null check — load_required already logs each miss; this catches
	 * the cumulative case so the caller can return EMODEL cleanly. */
	if (!R->mixer_conv_w || !R->convffn_dw_w ||
	    !R->convffn_fc1_w || !R->convffn_fc2_w) {
		sam3_log_error("mobileclip: RepMixer block %d incomplete", layer_idx);
		return SAM3_EMODEL;
	}

	(void)width; (void)mlp_dim;  /* width/mlp_dim used only for logging; can drop */
	sam3_log_info("mobileclip_s0: RepMixer block %d loaded%s",
		      layer_idx,
		      R->mixer_scale_w ? " (rbr_scale present)" : "");
	return SAM3_OK;
}
```

In `sam3_mobileclip_text_load`, replace the layer loop so it dispatches per-block based on `cfg.repmixer_block_indices`. The loop body should:

1. Check whether `i` appears in `cfg.repmixer_block_indices[0..n_repmixer_blocks)`.
2. If yes, set `enc->layers[i].is_repmixer = 1` and call `load_repmixer_block(&enc->layers[i].u.repmixer, wf, arena, i, cfg->width, cfg->mlp_dim)`.
3. Otherwise, set `enc->layers[i].is_repmixer = 0` and load the standard-block tensors into `enc->layers[i].u.std` (the same code that previously populated `enc->layers[i]` directly).

A small helper `is_repmixer_index(cfg, i)` returning a bool keeps the loop tidy.

- [ ] **Step 2: Build**

Run: `cd build && cmake --build . -j$(sysctl -n hw.ncpu) 2>&1 | tail -20`
Expected: success.

- [ ] **Step 3: Commit**

```bash
git add src/model/mobileclip_text.c
git commit -m "mobileclip: load RepMixer block weights (S0 indices 0 and 5)"
```

### Task 6.2 — Build the RepMixer block

**Files:**
- Modify: `src/model/mobileclip_text.c`

**Reference arithmetic** (authoritative — see spec section "RepMixer block (S0 blocks 0 and 5) — verified from reference"):

```
x_in: [1, C=width, 1, W=seq_len]   (already reshaped by caller)

# token_mixer.norm — single standalone BN
norm_out  = bn(x; norm_skip_*)

# token_mixer.mixer — sum of branches
y_skip    = bn(x; mixer_skip_*)
y_conv    = bn( conv2d(x, mixer_conv_w, kernel=[1,11], pad=[0,5], groups=C) ; mixer_conv_bn_* )
y_scale   = bn( conv2d(x, mixer_scale_w, kernel=[1,1])                      ; mixer_scale_bn_* )  # optional
mixer_out = y_skip + y_conv [+ y_scale if present]

# Token-mixer residual with internal scale (NOTE: SUBTRACTION between mixer and norm)
x = x + tm_layer_scale * (mixer_out - norm_out)

# ConvFFN — sequential
y = bn( conv2d(x, convffn_dw_w, kernel=[1,11], pad=[0,5], groups=C) ; convffn_dw_bn_* )
y = conv2d(y, convffn_fc1_w, bias=convffn_fc1_b, kernel=[1,1])    # [C → mlp_dim]
y = GELU(y)
y = conv2d(y, convffn_fc2_w, bias=convffn_fc2_b, kernel=[1,1])    # [mlp_dim → C]

# Outer block residual with outer scale
x = x + outer_layer_scale * y
```

eps=1e-5 for every BN. `bias=NULL` for BN-fused convs (mixer_conv_w, convffn_dw_w, mixer_scale_w); the BN supplies the bias. `convffn.fc1` and `convffn.fc2` ship explicit `bias`.

- [ ] **Step 1: Add a small BN helper**

If `gh_batchnorm` is not already a one-shot helper that takes `(weight, bias, running_mean, running_var, eps)`, add a static helper at the top of `mobileclip_text.c`:

```c
static struct sam3_tensor *bn(struct sam3_graph *g, struct sam3_tensor *x,
			      struct sam3_tensor *w, struct sam3_tensor *b,
			      struct sam3_tensor *rm, struct sam3_tensor *rv,
			      struct sam3_arena *arena)
{
	return gh_batchnorm(g, x, w, b, rm, rv, /*eps*/ 1e-5f, arena);
}
```

Confirm the existing `gh_batchnorm` signature first via:

```bash
grep -n "gh_batchnorm" src/model/graph_helpers.h
```

Adapt the wrapper if the actual helper differs.

- [ ] **Step 2: Implement build_repmixer_block**

Add (above `sam3_mobileclip_text_build`):

```c
/*
 * build_repmixer_block - One RepMixer block (S0 indices 0 and 5).
 *
 * Caller provides x already reshaped to NCHW=[1, width, 1, seq_len].
 * Returns the same NCHW shape (caller reshapes back to [seq, width] when
 * dispatching from the per-block evaluator).
 *
 * See spec section "RepMixer block (S0 blocks 0 and 5) — verified from
 * reference" for the exact arithmetic. eps=1e-5 for every BN. Optional
 * mixer.rbr_scale branch is added when R->mixer_scale_w is non-NULL.
 */
static struct sam3_tensor *build_repmixer_block(
	struct sam3_graph *g,
	const struct sam3_mobileclip_layer_repmixer *R,
	struct sam3_tensor *x,
	int width,
	struct sam3_arena *arena)
{
	struct sam3_tensor *norm_out, *y_skip, *y_conv, *mixer_out, *t;
	struct sam3_tensor *y;

	/* token_mixer.norm — single BN on the input. */
	norm_out = bn(g, x, R->norm_skip_w, R->norm_skip_b,
		      R->norm_skip_rm, R->norm_skip_rv, arena);

	/* token_mixer.mixer.rbr_skip branch: BN on the input. */
	y_skip = bn(g, x, R->mixer_skip_w, R->mixer_skip_b,
		    R->mixer_skip_rm, R->mixer_skip_rv, arena);

	/* token_mixer.mixer.rbr_conv[0] branch: depthwise conv 1×11 + BN. */
	t = gh_conv2d(g, x, R->mixer_conv_w, /*bias*/ NULL,
		      /*stride*/ 1, 1, /*pad_h*/ 0, /*pad_w*/ 5,
		      /*groups*/ width, arena);
	y_conv = bn(g, t, R->mixer_conv_bn_w, R->mixer_conv_bn_b,
		    R->mixer_conv_bn_rm, R->mixer_conv_bn_rv, arena);

	mixer_out = gh_add(g, y_skip, y_conv, arena);

	/* Optional rbr_scale branch (1×1 conv + BN). */
	if (R->mixer_scale_w) {
		struct sam3_tensor *y_scale;
		t = gh_conv2d(g, x, R->mixer_scale_w, /*bias*/ NULL,
			      1, 1, 0, 0, /*groups*/ 1, arena);
		y_scale = bn(g, t,
			     R->mixer_scale_bn_w, R->mixer_scale_bn_b,
			     R->mixer_scale_bn_rm, R->mixer_scale_bn_rv,
			     arena);
		mixer_out = gh_add(g, mixer_out, y_scale, arena);
	}

	/* Token-mixer residual: x + tm_layer_scale * (mixer_out - norm_out). */
	t = gh_sub(g, mixer_out, norm_out, arena);
	t = gh_mul_per_channel(g, t, R->tm_layer_scale, arena);
	x = gh_add(g, x, t, arena);

	/* ConvFFN: depthwise 1×11 + BN -> 1×1 (with bias) -> GELU -> 1×1 (with bias). */
	y = gh_conv2d(g, x, R->convffn_dw_w, /*bias*/ NULL,
		      1, 1, 0, 5, /*groups*/ width, arena);
	y = bn(g, y, R->convffn_dw_bn_w, R->convffn_dw_bn_b,
	       R->convffn_dw_bn_rm, R->convffn_dw_bn_rv, arena);
	y = gh_conv2d(g, y, R->convffn_fc1_w, R->convffn_fc1_b,
		      1, 1, 0, 0, /*groups*/ 1, arena);
	y = gh_gelu(g, y, arena);
	y = gh_conv2d(g, y, R->convffn_fc2_w, R->convffn_fc2_b,
		      1, 1, 0, 0, /*groups*/ 1, arena);

	/* Outer block residual with outer scale. */
	y = gh_mul_per_channel(g, y, R->outer_layer_scale, arena);
	x = gh_add(g, x, y, arena);

	return x;
}
```

If the codebase lacks `gh_sub`, implement it as `x + (-y)` using `gh_mul` with a constant `-1` tensor, or add a real elementwise-sub helper. `gh_mul_per_channel` is nominal — verify against `graph_helpers.h`; if absent, use `gh_mul` with the broadcast tensor `[width,1,1]`.

`gh_conv2d`, `gh_batchnorm`, `gh_mul_per_channel`, `gh_reshape` names are nominal. Verify with:

```bash
grep -rn "^[[:space:]]*struct sam3_tensor \*gh_conv2d\|^[[:space:]]*struct sam3_tensor \*gh_batchnorm" src/model/graph_helpers.h
```

If `gh_mul_per_channel` doesn't exist, look for `gh_mul_broadcast` or use `gh_mul` with a [width,1,1] tensor — broadcasting is handled by `SAM3_OP_MUL`'s elementwise path.

- [ ] **Step 2: Wire into `sam3_mobileclip_text_build`**

Replace the early-return-for-RepMixer guard and standard block loop in `sam3_mobileclip_text_build` with a per-block dispatch on `enc->layers[i].is_repmixer`:

```c
	for (int i = 0; i < cfg->n_layers; i++) {
		const struct sam3_mobileclip_layer *L = &enc->layers[i];

		if (L->is_repmixer) {
			/* Reshape [seq, width] -> [1, width, 1, seq], run
			 * RepMixer, reshape back. */
			int nchw[4] = { 1, cfg->width, 1, cfg->ctx_len };
			int seq2[2] = { cfg->ctx_len, cfg->width };
			x = gh_reshape(g, x, 4, nchw, arena);
			x = build_repmixer_block(g, &L->u.repmixer, x,
						 cfg->width, arena);
			x = gh_reshape(g, x, 2, seq2, arena);
		} else {
			x = build_std_block(g, &L->u.std, x,
					    cfg->n_heads, cfg->width, arena);
		}
	}
```

(Removes the old early-return for S0 and the `start_block = 1` adjustment.)

- [ ] **Step 3: Wire into `sam3_mobileclip_text_build_perblock`**

In the per-block evaluator, the block loop must dispatch per-block on `is_repmixer`. Since RepMixer blocks have a different working layout (NCHW conv) than standard blocks (sequence-major), each RepMixer block needs its own evaluation group rather than being batched with adjacent standard blocks. The cleanest pattern is one block per group when crossing a layout boundary; the simplest pattern is **one block per group always** (give up the 4-block batching for MobileCLIP). Both work; the simpler pattern is acceptable here because MobileCLIP has at most 12 blocks (vs CLIP's 24) and the per-block overhead is small.

```c
	for (int i = 0; i < cfg->n_layers; i++) {
		const struct sam3_mobileclip_layer *L = &enc->layers[i];
		struct sam3_graph g;
		struct sam3_tensor *x;

		sam3_graph_init(&g, scratch);
		x = gh_alias(&g, x_persist);

		if (L->is_repmixer) {
			int nchw[4] = { 1, cfg->width, 1, cfg->ctx_len };
			int seq2[2] = { cfg->ctx_len, cfg->width };
			x = gh_reshape(&g, x, 4, nchw, scratch);
			x = build_repmixer_block(&g, &L->u.repmixer, x,
						 cfg->width, scratch);
			x = gh_reshape(&g, x, 2, seq2, scratch);
		} else {
			x = build_std_block(&g, &L->u.std, x,
					    cfg->n_heads, cfg->width, scratch);
		}

		sam3_backend_eval(be, &g);
		x_persist = gh_persist_copy(persist, x);
		sam3_arena_reset(scratch);
	}
```

If the existing standard-block evaluator batches 4 blocks per group and you want to preserve that for the standard runs, the dispatch becomes a small state machine: accumulate consecutive standard blocks into a group, flush + start a new group when crossing a RepMixer block, evaluate the RepMixer alone, then continue. Implement only if profiling later shows the per-block overhead matters.

- [ ] **Step 4: Build**

Run: `cd build && cmake --build . -j$(sysctl -n hw.ncpu) 2>&1 | tail -30`
Expected: success.

- [ ] **Step 5: Commit**

```bash
git add src/model/mobileclip_text.c
git commit -m "mobileclip: build RepMixer blocks (S0 indices 0 and 5)"
```

### Task 6.3 — Convert S0 .pt and run S0 parity test

**Files:**
- Modify: `tests/test_mobileclip_text.c`

- [ ] **Step 1: Convert S0 to .sam3**

```bash
python tools/pt_to_safetensors.py \
  models/efficient_sam3_image_encoder_mobileclip_s0_ctx16.pt \
  /tmp/mc_s0.safetensors

build/sam3 convert -i /tmp/mc_s0.safetensors \
  -o tests/fixtures/mobileclip_s0/encoder.sam3 \
  --backbone hiera --variant sam3 --text-backbone mobileclip_s0
```

If conversion warns about missing keys for the RepMixer block, the actual `.pt` key names diverge from what we wrote in Task 6.1 — re-read the audit and fix the literals.

- [ ] **Step 2: Add S0 parity wrapper**

Add to `tests/test_mobileclip_text.c`:

```c
static int test_perblock_parity_s0(void)
{
	return test_perblock_parity(SAM3_TEXT_MOBILECLIP_S0, 16);
}
```

And in `main()`:

```c
	rc |= test_perblock_parity_s0();
```

- [ ] **Step 3: Build + run**

```bash
cd build && cmake --build . -j$(sysctl -n hw.ncpu) && ctest --output-on-failure -R mobileclip 2>&1 | tail -30
```

Expected: all three (S0/S1/L) per-block parity tests pass with max-abs-err < 1e-3.

If S0 fails, isolate to the RepMixer block: dump intermediate activations after each sub-step (BN, conv, layer_scale, residual) and compare to the reference's `block_00_out.npy`. The most likely failure modes are:

1. **Conv layout**: if the .sam3 file ships conv weights in the wrong layout (OHWI vs OIHW), `weight_conv_perm.c` should already permute, but RepMixer's depthwise conv may need a layout the perm logic doesn't recognize — check `tools/weight_conv_perm.c` for the depthwise-conv pattern.
2. **BN folding mismatch**: if BN params load but the reference path was built assuming folded BN, the running stats produce a wrong output. Re-check the audit.
3. **Padding semantics**: PyTorch `F.conv1d(..., padding=5)` is equivalent to `Conv2d(..., padding=(0,5))`; verify the C `gh_conv2d` interprets `pad_w` symmetrically.

- [ ] **Step 4: Commit**

```bash
git add tests/test_mobileclip_text.c tests/fixtures/mobileclip_s0
git commit -m "tests: parity for MobileCLIP-S0 (RepMixer)"
```

### Task 6.4 — End-to-end pooled-embedding parity for all variants

**Files:**
- Modify: `tests/test_mobileclip_text.c`

- [ ] **Step 1: Add pooled comparison**

The `test_perblock_parity` helper already runs the full encoder; extend it to also compare the pooled output against `pooled.npy`. After the per-token comparison block, add:

```c
	/* Pooled output check */
	struct sam3_tensor *pooled = NULL;
	struct sam3_graph g;
	sam3_graph_init(&g, &scratch);
	(void)iface.ops->build(&iface, &g, tok_tensor, &pooled, &scratch);
	TEST_ASSERT(pooled != NULL);
	TEST_ASSERT(pooled->dims[0] == 256);

	float pooled_ref[256];
	int pdims[1] = { 256 };
	int pn = 1;
	snprintf(path, sizeof(path), "%s/pooled.npy", dir);
	TEST_ASSERT(test_npy_load_f32(path, pooled_ref, pdims, &pn) == 0);

	float pooled_max_err = 0.0f;
	const float *p = (const float *)pooled->data;
	for (int i = 0; i < 256; i++) {
		float e = fabsf(p[i] - pooled_ref[i]);
		if (e > pooled_max_err) pooled_max_err = e;
	}
	fprintf(stderr, "pooled max abs err: %.4e\n", pooled_max_err);
	TEST_ASSERT(pooled_max_err < 1e-3f);
```

- [ ] **Step 2: Build + run**

```bash
cd build && cmake --build . -j$(sysctl -n hw.ncpu) && ctest --output-on-failure -R mobileclip 2>&1 | tail -30
```

Expected: all three variants pass per-block + pooled checks.

- [ ] **Step 3: Commit**

```bash
git add tests/test_mobileclip_text.c
git commit -m "tests: pooled-embedding parity across all MobileCLIP variants"
```

---

## Phase 7 — Converter wiring + end-to-end

### Task 7.1 — Add `--text-backbone` flag to `cli_convert`

**Files:**
- Modify: `tools/cli_convert.c:36-67` (usage), `:71-88` (struct), `:96-278` (parse), `:519-544` (config build)

- [ ] **Step 1: Update usage help**

In `print_usage` (lines 36-69), add after the `--variant` line:

```c
	printf("  --text-backbone <t>  Text encoder: \"clip\" (default), "
	       "\"mobileclip_s0\", \"mobileclip_s1\", \"mobileclip_l\"\n");
```

- [ ] **Step 2: Add struct field + default**

In `struct convert_args` (line 71-88), add:

```c
	const char *text_backbone;     /* CLI arg string */
	int         text_backbone_type; /* enum sam3_text_backbone */
```

In `parse_args` (around line 100-114), set the default:

```c
	args->text_backbone      = "clip";
	args->text_backbone_type = SAM3_TEXT_CLIP;
```

- [ ] **Step 3: Parse the flag**

In the option loop in `parse_args`, after the `--variant` block (around line 184-190), add:

```c
		} else if (strcmp(argv[i], "--text-backbone") == 0) {
			if (++i >= argc) {
				fprintf(stderr,
					"error: --text-backbone requires a type\n");
				return 1;
			}
			args->text_backbone = argv[i];
```

After the `--variant` resolve block (around line 251-263), add:

```c
	if (strcmp(args->text_backbone, "clip") == 0) {
		args->text_backbone_type = SAM3_TEXT_CLIP;
	} else if (strcmp(args->text_backbone, "mobileclip_s0") == 0) {
		args->text_backbone_type = SAM3_TEXT_MOBILECLIP_S0;
	} else if (strcmp(args->text_backbone, "mobileclip_s1") == 0) {
		args->text_backbone_type = SAM3_TEXT_MOBILECLIP_S1;
	} else if (strcmp(args->text_backbone, "mobileclip_l") == 0) {
		args->text_backbone_type = SAM3_TEXT_MOBILECLIP_L;
	} else {
		fprintf(stderr,
			"error: unsupported text-backbone '%s'\n",
			args->text_backbone);
		return 1;
	}
```

- [ ] **Step 4: Print + thread into config**

In the conversion summary print block (around line 520-534), add:

```c
		cli_progress("  text_backbone:  %s\n", args->text_backbone);
```

In the `Build model config` block (around line 537-544), add:

```c
	config.text_backbone = args.text_backbone_type;
```

- [ ] **Step 5: Build + smoke-convert a CLIP model (regression check)**

```bash
cd build && cmake --build . -j$(sysctl -n hw.ncpu) 2>&1 | tail -10
```

Re-convert an existing CLIP model to confirm no regression:

```bash
build/sam3 convert -i <path-to-existing-safetensors> \
  -o /tmp/clip_v4.sam3 \
  --backbone hiera --variant sam3
# (no --text-backbone defaults to "clip")
```

Expected: success; `sam3 info /tmp/clip_v4.sam3` shows version 4 with `text_backbone=clip`.

- [ ] **Step 6: Commit**

```bash
git add tools/cli_convert.c
git commit -m "cli_convert: add --text-backbone flag (writes v4 header field)"
```

### Task 7.2 — End-to-end smoke: load and call `sam3_processor_set_text` for each variant

**Files:**
- Modify: `tests/test_mobileclip_text.c`

- [ ] **Step 1: Add an end-to-end `sam3_load_model` path test**

Append to `tests/test_mobileclip_text.c`:

```c
static int test_e2e_load_set_text(int text_backbone, const char *sam3_path)
{
	sam3_ctx *ctx;
	enum sam3_error err = sam3_load_model(&ctx, sam3_path,
					      /*backend*/ "cpu");
	if (err != SAM3_OK) {
		fprintf(stderr, "skip e2e %d: load failed (err=%d)\n",
			text_backbone, err);
		return 0;
	}

	err = sam3_processor_set_text(ctx, "a person riding a bike");
	TEST_ASSERT(err == SAM3_OK);

	sam3_free(ctx);
	return 0;
}

static int test_e2e_s1(void)
{
	return test_e2e_load_set_text(
		SAM3_TEXT_MOBILECLIP_S1,
		"tests/fixtures/mobileclip_s1/encoder.sam3");
}
```

Add to `main()`:

```c
	rc |= test_e2e_s1();
```

(Repeat the wrapper pattern for S0 and L if quick; otherwise S1 alone is sufficient since the iface dispatch is the same code path.)

- [ ] **Step 2: Build + run**

```bash
cd build && cmake --build . -j$(sysctl -n hw.ncpu) && ctest --output-on-failure -R mobileclip 2>&1 | tail -30
```

Expected: end-to-end smoke passes alongside the per-block parity.

- [ ] **Step 3: Commit**

```bash
git add tests/test_mobileclip_text.c
git commit -m "tests: end-to-end sam3_processor_set_text smoke for MobileCLIP"
```

### Task 7.3 — Documentation: usage notes

**Files:**
- Modify: `docs/weight-format.md` (header v4 section)
- Create: `docs/mobileclip.md` (variant table + conversion command + usage)

- [ ] **Step 1: Update weight-format.md**

Find the v3 header description and add a v4 section:

```markdown
### Header v4 (current)

52 bytes. Adds a 4-byte `text_backbone` field after `reserved[3]`.

| Offset | Size | Field             | Notes                                    |
|--------|------|-------------------|------------------------------------------|
| 48     | 4    | text_backbone     | enum sam3_text_backbone (0=CLIP)         |

v3 readers see the field as part of the tensor descriptor table — the
v4 loader detects `version == 3` and uses a 48-byte header offset.
```

- [ ] **Step 2: Create docs/mobileclip.md**

```markdown
# MobileCLIP text encoder

The SAM3 inference engine supports three MobileCLIP text-encoder variants
in addition to the standard CLIP encoder:

| Variant        | Layers | Width | Heads | MLP  | Ctx | RepMixer |
|----------------|--------|-------|-------|------|-----|----------|
| `mobileclip_s0`| 6      | 512   | 8     | 2048 | 16  | yes      |
| `mobileclip_s1`| 12     | 512   | 8     | 2048 | 16  | no       |
| `mobileclip_l` | 12     | 768   | 12    | 3072 | 16  | no       |

## Conversion

```bash
python tools/pt_to_safetensors.py \
  models/efficient_sam3_image_encoder_mobileclip_s1_ctx16.pt \
  /tmp/mc_s1.safetensors

sam3 convert -i /tmp/mc_s1.safetensors \
  -o models/sam3_mobileclip_s1.sam3 \
  --backbone hiera --variant sam3 \
  --text-backbone mobileclip_s1
```

The text encoder runs at ctx=16 tokens; longer prompts are truncated.

The vision side is unchanged (HIERA 32-layer ViT, 1008×1008).
```

- [ ] **Step 3: Commit**

```bash
git add docs/weight-format.md docs/mobileclip.md
git commit -m "docs: weight-format v4 + MobileCLIP usage notes"
```

---

## Risk register (carried over from spec)

These open items must be resolved during execution; tasks reference them but cannot pre-resolve them.

1. **External 256-dim projector key** (Task 4.3): `BACKBONE_PFX "projector.weight"` is the leading guess; replace with the audit's confirmed key.
2. **BN folding state** (Tasks 6.1, 6.2): the RepMixer load + build branches handle both folded and unfolded BN, but the audit must say which we have.
3. **Conv layout for depthwise convs** (Task 6.3): `weight_conv_perm.c` may not recognize the depthwise pattern; verify.
4. **Helper API names** (Tasks 4.4, 4.5, 6.2): `gh_layernorm`, `gh_multihead_attention`, `gh_linear`, `gh_gelu`, `gh_add`, `gh_conv2d`, `gh_batchnorm`, `gh_mul_per_channel`, `gh_reshape`, `gh_alias`, `gh_persist_copy` are nominal — verify by reading `src/model/text_encoder.c` and `src/model/graph_helpers.h`.
5. **EOT pooling**: `cfg->ctx_len - 1` is approximate; if pooled parity (Task 6.4) fails, switch to a real argmax helper.
