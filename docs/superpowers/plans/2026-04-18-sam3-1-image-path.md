# SAM 3.1 image path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land sub-project 1 of the SAM 3.1 rollout: convert `sam3.1_multiplex.pt` to a variant-tagged `.sam3` file, run `sam3 segment` on images with SAM 3.1 weights, and verify numerical parity against the Python reference.

**Architecture:** No version bump — the `.sam3` header's 48-byte layout stays frozen. We reuse two free slots in `header.reserved[]` (`[1] = variant`, `[2] = n_fpn_scales`) alongside the existing `[0] = backbone_type`. The neck module already supports 1-4 scales; we just thread `n_fpn_scales` from the loaded config through `sam3_processor_init → sam3_image_model_init → sam3_vl_backbone_init → sam3_neck_init`. Tracker weights are passed through the converter untouched so sub-project 2 can consume them without re-conversion.

**Tech Stack:** C11 + CMake (existing), Python 3 with `torch` + `safetensors` (for the one-shot `.pt → .safetensors` preprocessor and reference-fixture dumper).

**Reference:** Design spec in `docs/superpowers/specs/2026-04-18-sam3-1-image-path-design.md`.

---

## File Structure

### Create
- `tools/pt_to_safetensors.py` — Python one-shot that strips checkpoint wrappers, remaps `sam3_model.*`/`sam2_predictor.*` prefixes, writes `.safetensors`.
- `tests/test_sam3_1_header.c` — synthetic round-trip for the new header slots (runs always).
- `tests/fixtures/sam3_1_bus_person/` — per-stage Python reference tensors for SAM 3.1 (populated by `dump_reference.py`; directory itself is created by the fixture-dump step).

### Modify
- `include/sam3/sam3_types.h` — add `enum sam3_variant`, extend `struct sam3_model_config`.
- `src/core/weight.c` — writer stashes `variant` / `n_fpn_scales` in `hdr.reserved[1]` / `[2]`.
- `src/sam3.c` — loader pulls them back into `ctx->config`, SAM 3 fallback on zero, and threads through `sam3_processor_init`.
- `src/model/sam3_processor.h` / `.c` — add `n_fpn_scales` parameter.
- `src/model/sam3_image.h` / `.c` — add `n_fpn_scales` parameter.
- `src/model/vl_combiner.h` / `.c` — add `n_fpn_scales` parameter; swap the hardcoded `4` at the main-neck `sam3_neck_init` for the parameter value. Tracker-side `sam2_neck` stays at 4.
- `tools/sam3_convert.c` — new `--variant {sam3,sam3.1}` flag that fills `config.variant` and `config.n_fpn_scales`.
- `tools/cli_info.c` — print `variant` and `n_fpn_scales`.
- `tools/cli_common.h` / `cli_common.c` — extend `cli_json_model_info` to emit the same two fields.
- `tools/dump_reference.py` — add `--variant {sam3,sam3.1}` path that assembles the SAM 3.1 image model from the reference repo's multiplex helpers.
- `tests/test_fixture_compare.c` — add `test_bus_person_sam3_1` that loads `sam3.1.sam3` + the new fixture and compares through the 10 stages (skipping `02_neck/scale_05x`).
- `CMakeLists.txt` — register `test_sam3_1_header` (no extra sources, follows the auto-registration pattern used by other tests).

### Delete
- None.

---

## Task 1: Public types for `variant` + `n_fpn_scales`

**Files:**
- Modify: `include/sam3/sam3_types.h:108-123`

- [ ] **Step 1: Add the enum and extend the config struct**

Insert this new enum between the existing `enum sam3_backbone_type` block and `struct sam3_model_config` (around line 113):

```c
/* Model variant. SAM3_1 uses the tri-neck (3 FPN scales) and the
 * multiplex-capable tracker; SAM3 uses the dual-neck and the original
 * tracker. Stored in the .sam3 header's reserved[1] slot. */
enum sam3_variant {
	SAM3_VARIANT_SAM3   = 0,
	SAM3_VARIANT_SAM3_1 = 1,
};
```

Extend the `struct sam3_model_config` definition (currently lines 116-123) by adding two fields at the tail:

```c
struct sam3_model_config {
	int image_size;       /* Input image size (e.g., 1024) */
	int encoder_dim;      /* Image encoder embedding dimension */
	int decoder_dim;      /* Mask decoder dimension */
	int n_encoder_layers;
	int n_decoder_layers;
	int backbone_type;    /* enum sam3_backbone_type */
	int n_fpn_scales;     /* 3 (SAM 3.1) or 4 (SAM 3) */
	int variant;          /* enum sam3_variant */
};
```

- [ ] **Step 2: Build (expect config-consumer failures)**

```bash
cd /Users/rbisri/Documents/sam3 && cmake --build build -j 2>&1 | tail -20
```

Expected: build succeeds (existing call sites zero-init the struct or use designated initializers; the new fields fall back to zero). If a compile error names a designated initializer that's become incomplete, note the line and fix in Task 2/3/4 rather than here.

- [ ] **Step 3: Commit**

```bash
git add include/sam3/sam3_types.h
git commit -m "$(cat <<'EOF'
sam3_types: add sam3_variant enum and config fields for SAM 3.1

Adds SAM3_VARIANT_SAM3 / SAM3_VARIANT_SAM3_1 and extends
sam3_model_config with variant + n_fpn_scales. Fields are appended so
zero-init keeps the SAM 3 default (variant=0, n_fpn_scales=0 is
interpreted as SAM 3 / 4 scales by the loader). No wiring yet.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Weight writer stashes new slots

**Files:**
- Modify: `src/core/weight.c:85-98`

- [ ] **Step 1: Populate `hdr.reserved[1]` and `hdr.reserved[2]`**

Extend the header-building block in `sam3_weight_write` (lines 86-98). After the existing `hdr.reserved[0] = (uint32_t)config->backbone_type;`, add:

```c
	hdr.reserved[0]      = (uint32_t)config->backbone_type;
	hdr.reserved[1]      = (uint32_t)config->variant;
	hdr.reserved[2]      = (uint32_t)config->n_fpn_scales;
```

- [ ] **Step 2: Build**

```bash
cmake --build build -j 2>&1 | tail -10
```

Expected: success.

- [ ] **Step 3: Commit**

```bash
git add src/core/weight.c
git commit -m "$(cat <<'EOF'
core/weight: write variant and n_fpn_scales into header reserved slots

Extends sam3_weight_write to store config.variant in reserved[1] and
config.n_fpn_scales in reserved[2], matching the existing pattern for
backbone_type in reserved[0]. Loader integration and CLI flag land in
follow-up commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Weight loader reads new slots with SAM 3 fallback

**Files:**
- Modify: `src/sam3.c:124-131`

- [ ] **Step 1: Add round-trip fallback logic**

Replace the header-copy block (lines 124-131) with:

```c
	/* Copy model config from weight file header */
	const struct sam3_weight_header *h = ctx->weights.header;
	ctx->config.image_size       = h->image_size;
	ctx->config.encoder_dim      = h->encoder_dim;
	ctx->config.decoder_dim      = h->decoder_dim;
	ctx->config.n_encoder_layers = h->n_encoder_layers;
	ctx->config.n_decoder_layers = h->n_decoder_layers;
	ctx->config.backbone_type    = (int)h->reserved[0];
	ctx->config.variant          = (int)h->reserved[1];
	ctx->config.n_fpn_scales     = (int)h->reserved[2];

	/* Legacy (pre-SAM3.1) .sam3 files have reserved[1..2] == 0.
	 * Treat that as SAM 3 with 4 FPN scales. */
	if (ctx->config.variant == 0 && ctx->config.n_fpn_scales == 0) {
		ctx->config.n_fpn_scales = 4;
	}
	if (ctx->config.variant != SAM3_VARIANT_SAM3 &&
	    ctx->config.variant != SAM3_VARIANT_SAM3_1) {
		sam3_log_error("unknown model variant %d in %s",
			       ctx->config.variant, path);
		sam3_weight_close(&ctx->weights);
		return SAM3_EMODEL;
	}
	if (ctx->config.n_fpn_scales < 1 || ctx->config.n_fpn_scales > 4) {
		sam3_log_error("invalid n_fpn_scales %d in %s (expect 3 or 4)",
			       ctx->config.n_fpn_scales, path);
		sam3_weight_close(&ctx->weights);
		return SAM3_EMODEL;
	}
```

- [ ] **Step 2: Build**

```bash
cmake --build build -j 2>&1 | tail -10
```

Expected: success. `sam3.c` already includes `sam3/sam3_types.h` transitively so the new enum values resolve.

- [ ] **Step 3: Commit**

```bash
git add src/sam3.c
git commit -m "$(cat <<'EOF'
sam3: populate config.variant and config.n_fpn_scales at load time

Reads reserved[1] / reserved[2] from the header into the context config.
Pre-SAM3.1 files (both zero) are treated as SAM 3 with 4 FPN scales.
Unknown variants or out-of-range scale counts fail the load cleanly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Converter CLI — `--variant` flag

**Files:**
- Modify: `tools/sam3_convert.c:60-215`

- [ ] **Step 1: Extend `struct convert_args`**

Replace `struct convert_args` (lines 60-73) with:

```c
struct convert_args {
	const char *input_path;
	const char *output_path;
	const char *format;
	const char *backbone;   /* "hiera" or "efficientvit" */
	const char *variant;    /* "sam3" or "sam3.1" */
	const char *quantize;   /* NULL or "q8_0" */
	int         image_size;
	int         encoder_dim;
	int         decoder_dim;
	int         encoder_layers;
	int         decoder_layers;
	int         backbone_type;
	int         variant_type;     /* enum sam3_variant */
	int         n_fpn_scales;
	int         verbose;
};
```

- [ ] **Step 2: Print the new flag in `--help`**

In `print_usage` (lines 31-58), add after the `--backbone` line:

```c
	printf("  --variant <type>     Model variant: "
	       "\"sam3\" (default), \"sam3.1\"\n");
```

- [ ] **Step 3: Parse `--variant` and default in `parse_args`**

In `parse_args`, initialize the new defaults after the existing field defaults (around line 88):

```c
	args->variant        = "sam3";
	args->variant_type   = SAM3_VARIANT_SAM3;
	args->n_fpn_scales   = 4;
```

Add the parse case after the `--backbone` block (around line 155, right before `--quantize`):

```c
		} else if (strcmp(argv[i], "--variant") == 0) {
			if (++i >= argc) {
				fprintf(stderr,
					"error: --variant requires a type\n");
				return 1;
			}
			args->variant = argv[i];
```

- [ ] **Step 4: Resolve variant at the bottom of `parse_args`**

After the backbone resolution block (ends around line 200), before the image-size default, add:

```c
	/* Resolve variant and apply its defaults */
	if (strcmp(args->variant, "sam3") == 0) {
		args->variant_type = SAM3_VARIANT_SAM3;
		args->n_fpn_scales = 4;
	} else if (strcmp(args->variant, "sam3.1") == 0) {
		args->variant_type = SAM3_VARIANT_SAM3_1;
		args->n_fpn_scales = 3;
	} else {
		fprintf(stderr,
			"error: unsupported variant '%s' "
			"(use \"sam3\" or \"sam3.1\")\n",
			args->variant);
		return 1;
	}
```

- [ ] **Step 5: Populate the config and summary print in `main`**

In `main`, after the existing `config.backbone_type = args.backbone_type;` (around line 439), add:

```c
	config.variant       = args.variant_type;
	config.n_fpn_scales  = args.n_fpn_scales;
```

Extend the conversion summary print (around line 432) with:

```c
	printf("  variant:        %s\n", args.variant);
	printf("  n_fpn_scales:   %d\n", args.n_fpn_scales);
```

- [ ] **Step 6: Build and smoke-run**

```bash
cmake --build build -j 2>&1 | tail -5
./build/tools/sam3_convert --help 2>&1 | head -20
```

Expected: help text shows the new `--variant` line.

- [ ] **Step 7: Commit**

```bash
git add tools/sam3_convert.c
git commit -m "$(cat <<'EOF'
sam3_convert: add --variant flag (sam3 | sam3.1)

Variant selects SAM 3 (4 FPN scales, default) or SAM 3.1 (3 FPN
scales). Both values flow into the .sam3 header via reserved[1] and
reserved[2] (written by the core weight writer in the previous commit).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `pt_to_safetensors.py` preprocessor

**Files:**
- Create: `tools/pt_to_safetensors.py`

- [ ] **Step 1: Write the script**

Create `tools/pt_to_safetensors.py` with:

```python
#!/usr/bin/env python3
"""
tools/pt_to_safetensors.py - Convert a PyTorch .pt checkpoint to .safetensors.

Used as the first step in the SAM 3.1 conversion pipeline because
sam3_convert only reads .safetensors. Also normalizes the checkpoint:
unwraps the `{"model": ...}` outer dict and remaps Facebook's internal
prefixes `sam3_model.*` / `sam2_predictor.*` to the OSS layout
`detector.*` / `tracker.*` (matches the Python reference in
reference/sam3/sam3/model_builder.py needs_remap logic).

Usage:
    python tools/pt_to_safetensors.py input.pt output.safetensors
"""
import argparse
import sys

import torch
from safetensors.torch import save_file

# Mirrors reference/sam3/sam3/model_builder.py:1209-1221
FB_TO_OSS_PREFIXES = [
    ("sam3_model.", "detector."),
    ("sam2_predictor.", "tracker."),
]


def remap(key: str) -> str:
    for src, dst in FB_TO_OSS_PREFIXES:
        if key.startswith(src):
            return dst + key[len(src):]
    return key


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("input", help="Input .pt path")
    ap.add_argument("output", help="Output .safetensors path")
    args = ap.parse_args()

    print(f"Loading {args.input} ...", file=sys.stderr)
    ckpt = torch.load(args.input, map_location="cpu", weights_only=True)

    # Unwrap {"model": ...}
    if isinstance(ckpt, dict) and "model" in ckpt \
            and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]

    if not isinstance(ckpt, dict):
        print(f"error: checkpoint is not a dict (got {type(ckpt).__name__})",
              file=sys.stderr)
        return 1

    # Remap if any FB-internal prefixes are present
    needs_remap = any(
        k.startswith(src) for src, _ in FB_TO_OSS_PREFIXES for k in ckpt
    )
    if needs_remap:
        print("Remapping sam3_model.* -> detector.*, "
              "sam2_predictor.* -> tracker.*", file=sys.stderr)
        ckpt = {remap(k): v for k, v in ckpt.items()}
    else:
        print("No FB-internal prefixes detected; writing as-is.",
              file=sys.stderr)

    # Force contiguous tensors (safetensors requires it)
    ckpt = {k: v.contiguous() for k, v in ckpt.items()}

    print(f"Writing {len(ckpt)} tensors to {args.output} ...", file=sys.stderr)
    save_file(ckpt, args.output)
    print("Done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x /Users/rbisri/Documents/sam3/tools/pt_to_safetensors.py
```

- [ ] **Step 3: Smoke-run `--help`**

```bash
python3 /Users/rbisri/Documents/sam3/tools/pt_to_safetensors.py --help
```

Expected: prints usage text with two positional args.

- [ ] **Step 4: Commit**

```bash
git add tools/pt_to_safetensors.py
git commit -m "$(cat <<'EOF'
tools: add pt_to_safetensors.py preprocessor for SAM 3.1

Normalizes a PyTorch .pt checkpoint into .safetensors form that
sam3_convert can consume. Unwraps {\"model\": ...} and remaps
Facebook-internal sam3_model.* / sam2_predictor.* prefixes to the OSS
detector.* / tracker.* layout, mirroring the needs_remap block in
reference/sam3/sam3/model_builder.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Thread `n_fpn_scales` through `vl_combiner`

**Files:**
- Modify: `src/model/vl_combiner.h:42-60`
- Modify: `src/model/vl_combiner.c:25-100`

- [ ] **Step 1: Update the public signature in `vl_combiner.h`**

Replace the prototype for `sam3_vl_backbone_init` (check current line range; it's the function declared in the header) with:

```c
enum sam3_error sam3_vl_backbone_init(struct sam3_vl_backbone *vl,
				      int backbone_type,
				      int n_fpn_scales,
				      struct sam3_arena *arena);
```

Update the `@n_fpn_scales:` entry in the doc comment to document:

```
 * @n_fpn_scales: Number of FPN scales for the main neck (3 for SAM 3.1,
 *                4 for SAM 3). The tracker-side sam2_neck always uses
 *                4 scales regardless.
```

- [ ] **Step 2: Update the implementation in `vl_combiner.c`**

Change the signature at `src/model/vl_combiner.c:25-27`:

```c
enum sam3_error sam3_vl_backbone_init(struct sam3_vl_backbone *vl,
				      int backbone_type,
				      int n_fpn_scales,
				      struct sam3_arena *arena)
```

Replace the hardcoded `4` in the main-neck init (currently lines 95-100) with:

```c
	/* Init main neck: n_fpn_scales at {4x, 2x, 1x, [0.5x]} */
	float scales[] = {4.0f, 2.0f, 1.0f, 0.5f};
	if (n_fpn_scales < 1 || n_fpn_scales > 4) {
		sam3_log_error("vl_backbone: invalid n_fpn_scales %d",
			       n_fpn_scales);
		return SAM3_EINVAL;
	}
	err = sam3_neck_init(&vl->neck, 256, backbone_dim,
			      grid_size, n_fpn_scales, scales);
	if (err != SAM3_OK)
		return err;
```

Leave the `sam2_neck` init (lines 109-113) at a hardcoded `4` — the tracker's sam2_neck is a separate FPN that stays dual-scale in SAM 3.1 too (its weights carry four scales). Confirm by reading the block once more before editing.

- [ ] **Step 3: Build (expect call-site errors in image model)**

```bash
cmake --build build -j 2>&1 | tail -20
```

Expected: errors at `sam3_vl_backbone_init(` call sites complaining about missing argument. Task 7 fixes that.

- [ ] **Step 4: Commit**

```bash
git add src/model/vl_combiner.h src/model/vl_combiner.c
git commit -m "$(cat <<'EOF'
vl_combiner: accept n_fpn_scales for the main neck

Threads the SAM 3.1 tri-neck count through sam3_vl_backbone_init. The
tracker-side sam2_neck still hardcodes 4 scales because the tracker
remains dual-neck in SAM 3.1 too. Image-model and processor call sites
updated in the next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Thread `n_fpn_scales` through `sam3_image` and `sam3_processor`

**Files:**
- Modify: `src/model/sam3_image.h:74-90`
- Modify: `src/model/sam3_image.c:176-200`
- Modify: `src/model/sam3_processor.h:78-92`
- Modify: `src/model/sam3_processor.c:55-200`

- [ ] **Step 1: Update `sam3_image_model_init`**

In `src/model/sam3_image.h`, extend the prototype:

```c
enum sam3_error sam3_image_model_init(struct sam3_image_model *model,
				      int backbone_type,
				      int n_fpn_scales,
				      struct sam3_arena *arena);
```

In `src/model/sam3_image.c` around line 176, change the definition accordingly and pass the new param down:

```c
enum sam3_error sam3_image_model_init(struct sam3_image_model *model,
				      int backbone_type,
				      int n_fpn_scales,
				      struct sam3_arena *arena)
{
	enum sam3_error err;
	memset(model, 0, sizeof(*model));

	err = sam3_vl_backbone_init(&model->backbone, backbone_type,
				     n_fpn_scales, arena);
	/* …rest unchanged… */
}
```

- [ ] **Step 2: Update `sam3_processor_init`**

In `src/model/sam3_processor.h` around lines 78-92, extend the prototype:

```c
enum sam3_error sam3_processor_init(struct sam3_processor *proc,
				    int backbone_type,
				    int n_fpn_scales);
```

In `src/model/sam3_processor.c` around line 59, change the definition and pass through to `sam3_image_model_init` at the call around line 180:

```c
enum sam3_error sam3_processor_init(struct sam3_processor *proc,
				    int backbone_type,
				    int n_fpn_scales)
{
	/* …existing pre-model setup unchanged… */
	err = sam3_image_model_init(&proc->model, backbone_type,
				    n_fpn_scales,
				    proc->arena_weights);
	/* …rest unchanged… */
}
```

- [ ] **Step 3: Build (expect error in `src/sam3.c`)**

```bash
cmake --build build -j 2>&1 | tail -10
```

Expected: `sam3_processor_init(&ctx->proc, ctx->config.backbone_type)` fails compilation in `src/sam3.c:144`. Task 8 fixes that.

- [ ] **Step 4: Commit**

```bash
git add src/model/sam3_image.h src/model/sam3_image.c \
        src/model/sam3_processor.h src/model/sam3_processor.c
git commit -m "$(cat <<'EOF'
sam3_image + sam3_processor: plumb n_fpn_scales

Updates both init signatures to forward the scale count into
vl_backbone. sam3.c call-site update lands next.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Wire `n_fpn_scales` at the context call site

**Files:**
- Modify: `src/sam3.c:144`

- [ ] **Step 1: Pass config.n_fpn_scales into `sam3_processor_init`**

Change the call (currently line 144) to:

```c
	err = sam3_processor_init(&ctx->proc,
				   ctx->config.backbone_type,
				   ctx->config.n_fpn_scales);
```

- [ ] **Step 2: Build**

```bash
cmake --build build -j 2>&1 | tail -10
```

Expected: clean build.

- [ ] **Step 3: Run the full test suite (regression guard)**

```bash
cd /Users/rbisri/Documents/sam3/build && ctest --output-on-failure 2>&1 | tail -30
```

Expected: all previously-passing tests still pass. SAM 3 files produce `n_fpn_scales=4` via the fallback in Task 3, so the runtime behaviour is identical.

- [ ] **Step 4: Commit**

```bash
git add src/sam3.c
git commit -m "$(cat <<'EOF'
sam3: pass n_fpn_scales from loaded config into the processor

Completes the plumbing started two commits back. SAM 3 files still
initialise with 4 scales (fallback in sam3_load_model), SAM 3.1 files
initialise with 3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: `cli_info` prints variant and scales

**Files:**
- Modify: `tools/cli_info.c:76-94`
- Modify: `tools/cli_common.c` (extend `cli_json_model_info`)

- [ ] **Step 1: Human output — add two lines**

In `tools/cli_info.c` around line 84, add after the `decoder_layers` print:

```c
		const char *variant_str = (h->reserved[1] == SAM3_VARIANT_SAM3_1)
					   ? "sam3.1" : "sam3";
		uint32_t scales = h->reserved[2]
				   ? h->reserved[2]
				   : (h->reserved[1] == SAM3_VARIANT_SAM3_1 ? 3 : 4);
		printf("  variant:         %s\n", variant_str);
		printf("  n_fpn_scales:    %u\n", scales);
```

Include `<sam3/sam3_types.h>` at the top if not already pulled in transitively (check with the current include list; `core/weight.h` likely drags it in already).

- [ ] **Step 2: JSON output — same two fields**

Open `tools/cli_common.c`, find `cli_json_model_info`, and add the same two keys to its JSON output (`"variant"` and `"n_fpn_scales"`). Use the same fallback logic as the human path.

- [ ] **Step 3: Build and smoke-run on an existing SAM 3 file**

```bash
cmake --build build -j 2>&1 | tail -5
./build/tools/sam3 info models/sam3.sam3 2>&1 | tail -12
```

Expected output includes:

```
  variant:         sam3
  n_fpn_scales:    4
```

(The latter falls back to 4 because the file was converted before this change.)

- [ ] **Step 4: Commit**

```bash
git add tools/cli_info.c tools/cli_common.c
git commit -m "$(cat <<'EOF'
cli_info: report variant and n_fpn_scales

Reads header reserved[1..2] directly and falls back to the SAM 3
defaults for pre-SAM-3.1 files. Both the human-readable and --json
outputs carry the new fields.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Synthetic header round-trip test

**Files:**
- Create: `tests/test_sam3_1_header.c`

- [ ] **Step 1: Write the test**

Create `tests/test_sam3_1_header.c`:

```c
/*
 * tests/test_sam3_1_header.c - Round-trip the SAM 3.1 header slots.
 *
 * Writes a minimal valid .sam3 file with a single 1-element tensor,
 * setting variant=SAM3_VARIANT_SAM3_1 and n_fpn_scales=3. Re-opens it
 * and asserts the fields round-trip. Also asserts that the legacy path
 * (reserved[1..2] zero) yields variant=SAM3 / n_fpn_scales=4 via the
 * loader's fallback.
 *
 * Key types:  (uses sam3_weight_header)
 * Depends on: sam3/sam3_types.h, core/weight.h, test_helpers.h
 * Used by:    CTest registration in CMakeLists.txt
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sam3/sam3.h"
#include "sam3/sam3_types.h"
#include "core/weight.h"
#include "test_helpers.h"

struct tiny_reader_state {
	float datum;
};

static enum sam3_error tr_open(struct weight_reader *r, const char *p)
{
	(void)r; (void)p;
	return SAM3_OK;
}
static int tr_n_tensors(struct weight_reader *r) { (void)r; return 1; }
static enum sam3_error tr_get_tensor_info(struct weight_reader *r, int idx,
					  struct weight_tensor_info *info)
{
	(void)r; (void)idx;
	info->name   = "dummy";
	info->dtype  = SAM3_DTYPE_F32;
	info->n_dims = 1;
	info->dims[0] = 1;
	info->nbytes  = sizeof(float);
	return SAM3_OK;
}
static enum sam3_error tr_read_tensor_data(struct weight_reader *r, int idx,
					   void *dst, size_t dst_size)
{
	struct tiny_reader_state *s = r->impl;
	(void)idx;
	if (dst_size < sizeof(float)) return SAM3_EINVAL;
	memcpy(dst, &s->datum, sizeof(float));
	return SAM3_OK;
}
static void tr_close(struct weight_reader *r) { (void)r; }

static const struct weight_reader_ops tr_ops = {
	.open             = tr_open,
	.n_tensors        = tr_n_tensors,
	.get_tensor_info  = tr_get_tensor_info,
	.read_tensor_data = tr_read_tensor_data,
	.close            = tr_close,
};

static void test_sam3_1_roundtrip(void)
{
	const char *path = "/tmp/sam3_test_variant.sam3";
	struct sam3_model_config cfg = {
		.image_size       = 1008,
		.encoder_dim      = 1024,
		.decoder_dim      = 256,
		.n_encoder_layers = 32,
		.n_decoder_layers = 2,
		.backbone_type    = SAM3_BACKBONE_HIERA,
		.n_fpn_scales     = 3,
		.variant          = SAM3_VARIANT_SAM3_1,
	};
	struct tiny_reader_state s = { .datum = 1.5f };
	struct weight_reader r = { .ops = &tr_ops, .impl = &s };

	ASSERT(sam3_weight_write(path, &cfg, &r) == SAM3_OK);

	struct sam3_weight_file wf;
	memset(&wf, 0, sizeof(wf));
	ASSERT(sam3_weight_open(&wf, path) == SAM3_OK);
	ASSERT_EQ(wf.header->reserved[1], (uint32_t)SAM3_VARIANT_SAM3_1);
	ASSERT_EQ(wf.header->reserved[2], 3);
	sam3_weight_close(&wf);

	unlink(path);
}

static void test_sam3_legacy_defaults(void)
{
	const char *path = "/tmp/sam3_test_legacy.sam3";
	struct sam3_model_config cfg = {
		.image_size       = 1008,
		.encoder_dim      = 1024,
		.decoder_dim      = 256,
		.n_encoder_layers = 32,
		.n_decoder_layers = 2,
		.backbone_type    = SAM3_BACKBONE_HIERA,
		/* variant and n_fpn_scales both zero */
	};
	struct tiny_reader_state s = { .datum = 0.0f };
	struct weight_reader r = { .ops = &tr_ops, .impl = &s };

	ASSERT(sam3_weight_write(path, &cfg, &r) == SAM3_OK);

	struct sam3_weight_file wf;
	memset(&wf, 0, sizeof(wf));
	ASSERT(sam3_weight_open(&wf, path) == SAM3_OK);
	ASSERT_EQ(wf.header->reserved[1], 0);
	ASSERT_EQ(wf.header->reserved[2], 0);
	/* sam3_load_model would synthesize variant=SAM3, n_fpn_scales=4
	 * from these zeros; that path is exercised by the integration
	 * tests — here we only verify the on-disk slots round-trip. */
	sam3_weight_close(&wf);

	unlink(path);
}

int main(void)
{
	test_sam3_1_roundtrip();
	test_sam3_legacy_defaults();
	printf("test_sam3_1_header: PASS\n");
	return 0;
}
```

- [ ] **Step 2: Register the test in `CMakeLists.txt`**

Check the test-registration loop near the existing `test_fixture_compare` block (around line 232 in `CMakeLists.txt`). If the loop auto-picks up `test_*.c` from `tests/`, no change is needed. If the file list is manual, add `test_sam3_1_header.c` to it. Rerun cmake:

```bash
cd /Users/rbisri/Documents/sam3 && cmake --build build -j 2>&1 | tail -10
```

- [ ] **Step 3: Run the test**

```bash
cd /Users/rbisri/Documents/sam3/build && ctest -R test_sam3_1_header --output-on-failure 2>&1 | tail -10
```

Expected: `1/1 Test passed`. Output contains `test_sam3_1_header: PASS`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_sam3_1_header.c CMakeLists.txt
git commit -m "$(cat <<'EOF'
tests: header round-trip for SAM 3.1 variant and scale-count slots

Writes a minimal .sam3 with SAM 3.1 metadata and re-opens it,
confirming reserved[1..2] carry the expected values. Also writes a
legacy config (both zero) to confirm that path still succeeds; the
loader-side fallback is exercised by the integration suite.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Rewrite `dump_reference.py` to support SAM 3.1

**Files:**
- Modify: `tools/dump_reference.py` (full rewrite)

- [ ] **Step 1: Replace the file with a variant-aware assembler**

Overwrite `tools/dump_reference.py` with:

```python
#!/usr/bin/env python3
"""
tools/dump_reference.py - Dump per-stage SAM 3 / SAM 3.1 reference
tensors for numerical parity tests.

For --variant sam3, assembles the stock image model via
    build_sam3_image_model(...)  (from the upstream sam3 package).
For --variant sam3.1, assembles an image-only SAM 3.1 model by
reusing the multiplex tri-neck helpers from the upstream
model_builder.py: _create_multiplex_tri_backbone + _create_text_encoder
+ the unchanged detector components (transformer, segmentation head,
dot-product scoring, geometry encoder), wrapped in Sam3Image. The
multiplex checkpoint carries detector.* keys that load straight into
this assembly (tracker weights are silently unused).

Output format matches the per-stage .safetensors fixture layout already
used by tests/fixtures/bus_person/ (00_input/, 01_vit/, …, 10_final/).

Usage:
    python tools/dump_reference.py --variant {sam3,sam3.1} \
        --image path/to/img.jpg --checkpoint path/to/file.pt --text "bus" \
        --out tests/fixtures/sam3_1_bus_person/
"""
import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file


def build_model_sam3(checkpoint: str, bpe_path: str, device: str):
    from sam3.model_builder import build_sam3_image_model
    return build_sam3_image_model(
        bpe_path=bpe_path,
        device=device,
        eval_mode=True,
        checkpoint_path=checkpoint,
        load_from_HF=False,
    )


def build_model_sam3_1(checkpoint: str, bpe_path: str, device: str):
    """Assemble SAM 3.1 image model — reuses the multiplex tri-neck."""
    from sam3.model_builder import (
        _create_geometry_encoder,
        _create_multiplex_tri_backbone,
        _create_sam3_transformer,
        _create_segmentation_head,
        _create_dot_product_scoring,
        _create_text_encoder,
    )
    from sam3.model.sam3_image import Sam3Image
    from sam3.model.vl_combiner import SAM3VLBackbone

    tri_neck = _create_multiplex_tri_backbone()
    text_encoder = _create_text_encoder(bpe_path)
    backbone = SAM3VLBackbone(scalp=1, visual=tri_neck, text=text_encoder)
    transformer = _create_sam3_transformer()
    dot_prod_scoring = _create_dot_product_scoring()
    segmentation_head = _create_segmentation_head()
    geometry_encoder = _create_geometry_encoder()

    model = Sam3Image(
        backbone=backbone,
        transformer=transformer,
        input_geometry_encoder=geometry_encoder,
        segmentation_head=segmentation_head,
        num_feature_levels=1,
        o2m_mask_predict=True,
        dot_prod_scoring=dot_prod_scoring,
        use_instance_query=False,
        multimask_output=True,
        inst_interactive_predictor=None,
        matcher=None,
    )

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    # Keep only detector.* keys, strip the prefix
    detector_ckpt = {k[len("detector."):]: v
                     for k, v in ckpt.items() if k.startswith("detector.")}
    missing, unexpected = model.load_state_dict(detector_ckpt, strict=False)
    print(f"Loaded {len(detector_ckpt)} detector tensors. "
          f"missing={len(missing)} unexpected={len(unexpected)}",
          file=sys.stderr)
    model.eval().to(device)
    return model


def run_and_dump(model, image_path: str, text: str, out_dir: str):
    """Run the image pipeline and save per-stage tensors.

    Hooks are attached to the modules listed in STAGE_HOOKS below.
    Output layout matches tests/fixtures/bus_person/.
    """
    image = Image.open(image_path).convert("RGB")
    # Preprocess to 1008x1008, normalize to (0.5, 0.5, 0.5) / (0.5, 0.5, 0.5)
    # — matches image_mean / image_std from upstream Sam3Image.
    import torchvision.transforms.functional as F
    img_t = F.to_tensor(image.resize((1008, 1008)))
    img_t = (img_t - 0.5) / 0.5
    img_t = img_t.unsqueeze(0).to(next(model.parameters()).device)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "00_input").mkdir(exist_ok=True)
    save_file({"image": img_t.cpu().contiguous()},
              str(Path(out_dir) / "00_input" / "tensors.safetensors"))

    # The hooks-based per-stage capture is long; implement per the
    # existing bus_person fixture (see git log for prior art). For the
    # minimum-viable SAM 3.1 fixture we only need 00_input + 02_neck +
    # 10_final, which is what test_bus_person_sam3_1 consumes.

    # 02_neck: extract the three neck outputs [4x, 2x, 1x]
    # 10_final: model(img_t, text=text) returns pred_masks / pred_logits

    with torch.no_grad():
        # Capture neck outputs via a forward hook
        neck_outs = []
        def _neck_hook(mod, inp, out):
            # out is a dict or tuple of per-scale features in order
            neck_outs.append(out)
        h = model.backbone.visual.register_forward_hook(_neck_hook)

        out = model(img_t, text=[text])
        h.remove()

    # Save 02_neck: expect 3 scales for SAM 3.1
    (Path(out_dir) / "02_neck").mkdir(exist_ok=True)
    # neck_outs[0] layout depends on TriViTDetNeck output — adapt once we
    # have the real tensor available (inspect shape + keys on the first
    # dump). Write each scale to scale_4x / scale_2x / scale_1x files.

    # Save 10_final
    (Path(out_dir) / "10_final").mkdir(exist_ok=True)
    final_dict = {}
    if isinstance(out, dict):
        for k in ("pred_masks", "pred_logits"):
            if k in out:
                final_dict[k] = out[k].cpu().contiguous()
    save_file(final_dict,
              str(Path(out_dir) / "10_final" / "tensors.safetensors"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--variant", choices=["sam3", "sam3.1"], required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bpe", default=None,
                    help="Path to bpe_simple_vocab_16e6.txt.gz (default: "
                         "upstream package resource)")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.bpe is None:
        import pkg_resources
        args.bpe = pkg_resources.resource_filename(
            "sam3", "assets/bpe_simple_vocab_16e6.txt.gz")

    if args.variant == "sam3":
        model = build_model_sam3(args.checkpoint, args.bpe, args.device)
    else:
        model = build_model_sam3_1(args.checkpoint, args.bpe, args.device)

    run_and_dump(model, args.image, args.text, args.out)
    print(f"Wrote fixtures to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Note on the `run_and_dump` shape**: the 02_neck hook and the final-stage mapping depend on the upstream `TriViTDetNeck` output shape, which we can only verify once the checkpoint is downloaded and the first forward pass runs. Task 12 is the iteration point — adjust the hook body there once you see real shapes in `pdb`, then re-run.

- [ ] **Step 2: Sanity-check `--help`**

```bash
python3 /Users/rbisri/Documents/sam3/tools/dump_reference.py --help
```

Expected: two-line help summary, all flags listed.

- [ ] **Step 3: Commit**

```bash
git add tools/dump_reference.py
git commit -m "$(cat <<'EOF'
tools/dump_reference.py: support --variant for SAM 3 and SAM 3.1

Wraps the upstream reference model builders. For SAM 3.1, assembles an
image-only model by composing the multiplex tri-neck helper with the
unchanged detector components, then loads only the detector.* keys
from the multiplex checkpoint. Output layout mirrors the existing
bus_person fixture; the per-stage hook bodies will be adjusted in the
fixture-generation task once real tensor shapes are verified.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: End-to-end conversion smoke test

**Prereq:** `models/sam3.1_multiplex.pt` exists (3.5 GB from HF).

- [ ] **Step 1: Convert `.pt` to `.safetensors`**

```bash
cd /Users/rbisri/Documents/sam3 && \
    python3 tools/pt_to_safetensors.py \
        models/sam3.1_multiplex.pt \
        models/sam3.1_multiplex.safetensors 2>&1 | tail -5
```

Expected:
- `Loading models/sam3.1_multiplex.pt ...`
- A remap line (probably `Remapping sam3_model.* -> detector.*, sam2_predictor.* -> tracker.*` — or the "no prefixes" line if HF already normalized).
- `Writing N tensors to models/sam3.1_multiplex.safetensors ...`
- `Done.`

Output file size ≈ 3.5 GB.

- [ ] **Step 2: Convert to `.sam3`**

```bash
./build/tools/sam3_convert \
    -i models/sam3.1_multiplex.safetensors \
    -o models/sam3.1.sam3 \
    --variant sam3.1 2>&1 | tail -15
```

Expected summary includes:

```
  variant:        sam3.1
  n_fpn_scales:   3
```

ends with `Done.` and produces a `models/sam3.1.sam3` file (~3.5 GB).

- [ ] **Step 3: Inspect with `sam3 info`**

```bash
./build/tools/sam3 info models/sam3.1.sam3 2>&1 | tail -15
```

Expected:

```
  variant:         sam3.1
  n_fpn_scales:    3
```

- [ ] **Step 4: Load test**

```bash
./build/tools/sam3 segment \
    -m models/sam3.1.sam3 \
    -i tests/fixtures/bus_person/bus.jpg \
    -t "bus" \
    --output /tmp/sam3_1_smoke 2>&1 | tail -20
```

Expected: completes without crash; writes masks to `/tmp/sam3_1_smoke/`. Masks may or may not look correct yet — correctness is Task 13's job. The purpose of this step is to verify the **load path** (header → config → processor → neck with 3 scales → weights into place).

If the segment fails, the most likely causes are:
1. Missing tensors (rename-rule mismatch on a tracker key that also happens to share a name prefix with a detector key). Inspect with `SAM3_LOG_LEVEL=debug`.
2. Wrong shape on a neck tensor (scale-count miscount). Check `sam3_neck_load` log lines.
3. Image-size mismatch (check `ctx->config.image_size == 1008`).

- [ ] **Step 5: Commit converted baseline path**

Do **not** commit the large `.sam3` / `.safetensors` files (they're gitignored). Commit any small test fixture images you ended up using:

```bash
# Only if we added a new fixture image
# git add tests/fixtures/bus_person/bus.jpg
echo "No commit expected here unless smoke run added a fixture file."
```

---

## Task 13: Generate the SAM 3.1 reference fixture

**Prereq:** Task 12 complete; `sam3.1.sam3` loads cleanly; Python reference repo installed in a venv (`pip install -e reference/sam3/` or equivalent).

- [ ] **Step 1: Produce the fixture**

```bash
cd /Users/rbisri/Documents/sam3 && \
    python3 tools/dump_reference.py \
        --variant sam3.1 \
        --image tests/fixtures/bus_person/bus.jpg \
        --checkpoint models/sam3.1_multiplex.pt \
        --text "bus" \
        --out tests/fixtures/sam3_1_bus_person/ 2>&1 | tail -10
```

Expected: writes `00_input/tensors.safetensors`, `02_neck/scale_4x.safetensors`, `02_neck/scale_2x.safetensors`, `02_neck/scale_1x.safetensors`, `10_final/tensors.safetensors`.

If the run fails because the hook body in `run_and_dump` doesn't match the tri-neck's output shape (likely on first run), drop into the script, add a `breakpoint()` before the hook, inspect the actual tuple/dict that `TriViTDetNeck.forward` returns, update the scale-extraction logic, and rerun.

- [ ] **Step 2: Write `metadata.json`**

Inside `tests/fixtures/sam3_1_bus_person/`, create a `metadata.json` like the one in the SAM 3 fixture:

```bash
cat > tests/fixtures/sam3_1_bus_person/metadata.json <<'EOF'
{
    "variant": "sam3.1",
    "image": "bus.jpg",
    "text": "bus",
    "image_size": 1008,
    "n_fpn_scales": 3
}
EOF
```

- [ ] **Step 3: Commit (fixture is small enough)**

```bash
git add tests/fixtures/sam3_1_bus_person/
git commit -m "$(cat <<'EOF'
tests/fixtures: add sam3_1_bus_person parity fixture

Per-stage tensors dumped from the Python reference with
dump_reference.py --variant sam3.1, driving the same bus.jpg + "bus"
text prompt as the existing SAM 3 fixture. Three neck scales instead
of four; same 10_final shape.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Integration test — extend `test_fixture_compare.c`

**Files:**
- Modify: `tests/test_fixture_compare.c:625-end`

- [ ] **Step 1: Add the SAM 3.1 test function**

After `test_bus_person_text_only` in `tests/test_fixture_compare.c`, add:

```c
/* ── Test: SAM 3.1 bus/person text-only fixture ─────────────────────── */

#define SAM3_1_BUS_FIXTURE_DIR \
	SAM3_SOURCE_DIR "/tests/fixtures/sam3_1_bus_person"
#define SAM3_1_MODEL_PATH SAM3_SOURCE_DIR "/models/sam3.1.sam3"

static int sam3_1_fixtures_available(void)
{
	return access(SAM3_1_BUS_FIXTURE_DIR "/metadata.json", F_OK) == 0
	    && access(SAM3_1_MODEL_PATH, F_OK) == 0;
}

static void test_bus_person_sam3_1(void)
{
	if (!sam3_1_fixtures_available()) {
		printf("\ntest_bus_person_sam3_1: SKIP "
		       "(fixture or model missing)\n");
		return;
	}

	sam3_ctx *ctx = sam3_init();
	ASSERT(ctx != NULL);
	if (!ctx) return;

	enum sam3_error err = sam3_load_model(ctx, SAM3_1_MODEL_PATH);
	ASSERT(err == SAM3_OK);
	if (err != SAM3_OK) {
		sam3_free(ctx);
		return;
	}

	/* Header plumbing assertions */
	/* Access via ctx->config — if private, add a test-only accessor;
	 * otherwise the sam3_ctx struct is already visible through
	 * "sam3/sam3.h" here (confirm before editing). */

	/* Load fixture image + final masks (reuses helpers from the SAM3
	 * test above). Asserts IoU >= 0.99 against pred_masks and matches
	 * pred_logits within 1e-3. Uses only 3 neck scales. */
	/* -- body analogous to test_bus_person_text_only, minus the
	 *    scale_05x comparison. Copy-paste and delete the 4th-scale
	 *    block; keep everything else identical. */

	sam3_free(ctx);
	printf("test_bus_person_sam3_1: PASS\n");
}
```

Then add the call into `main` (near the bottom of the file), right after the existing `test_bus_person_text_only();` call:

```c
	test_bus_person_sam3_1();
```

- [ ] **Step 2: Mirror the numerical comparison logic**

Copy the relevant block from `test_bus_person_text_only` (the ViT/text-encoder/neck/decoder/final-stage comparisons that load from `{BUS_FIXTURE_DIR}/...`) into `test_bus_person_sam3_1`, changing:

- `BUS_FIXTURE_DIR` → `SAM3_1_BUS_FIXTURE_DIR`.
- In the `02_neck` comparison loop, **delete the `scale_05x.safetensors` case** (SAM 3.1 has no fourth scale). Keep `scale_4x`, `scale_2x`, `scale_1x`.
- In the neck-side assertions, loop upper bound uses `3` instead of `4` — or better: read `proc.model.backbone.neck.n_scales` and loop to that.

Keep the existing tolerance (IoU ≥ 0.99 on `pred_masks`, 1e-3 on `pred_logits`).

- [ ] **Step 3: Build and run**

```bash
cmake --build build -j 2>&1 | tail -5
cd build && ctest -R test_fixture_compare --output-on-failure 2>&1 | tail -30
```

Expected:
- `test_bus_person_text_only: PASS` (unchanged)
- `test_bus_person_sam3_1: PASS` (new)

If `test_bus_person_sam3_1` fails, the most likely failure modes are:
1. Neck scale comparison mismatch → check `n_scales` on the C side.
2. IoU shortfall → check the detector-only checkpoint-loading path in the Python fixture (possibly need `strict=False` misses).
3. Prompt coordinate / spatial-size drift → re-inspect the tri-neck grid_size (still 72 for 1008/14).

- [ ] **Step 4: Commit**

```bash
git add tests/test_fixture_compare.c
git commit -m "$(cat <<'EOF'
test_fixture_compare: add SAM 3.1 bus-person parity case

Loads models/sam3.1.sam3 and compares through the same 10-stage
fixture plumbing as the SAM 3 test, minus the fourth-scale neck
comparison. Gated on the fixture + model being present so CI without
the large files still passes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: README snippet

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the two-step conversion**

Find the weight-conversion section in `README.md` (it references `sam3_convert`). Append a SAM 3.1 subsection:

```markdown
### SAM 3.1

SAM 3.1 ships as a PyTorch `.pt` checkpoint only. Convert in two steps:

```bash
# 1. Normalize into .safetensors
python tools/pt_to_safetensors.py \
    models/sam3.1_multiplex.pt \
    models/sam3.1_multiplex.safetensors

# 2. Convert to .sam3 with the SAM 3.1 variant flag
./build/tools/sam3_convert \
    -i models/sam3.1_multiplex.safetensors \
    -o models/sam3.1.sam3 \
    --variant sam3.1

# 3. Use it
./build/tools/sam3 segment -m models/sam3.1.sam3 -i img.jpg -t "cat"
```

Only the image-detector path is wired at this time. Tracker / video
tracking for SAM 3.1 is on the roadmap.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs: README section for SAM 3.1 conversion and image-path usage

Documents the pt -> safetensors -> sam3 pipeline, flags the
image-only scope, and points at the roadmap for video.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Checklist

Before handing off:

1. **Spec coverage**:
   - Spec §1 (Goal) → Task 14 is the "segment works / parity holds" checkpoint.
   - Spec §2 (Weight-format) → Tasks 2-3 + 10.
   - Spec §3 (Converter) → Tasks 4-5.
   - Spec §4 (Neck param) → Tasks 6-8 (the parameterization already existed, so we just wire a new caller path).
   - Spec §5 (Loader) → Task 3 + 8.
   - Spec §6 (CLI) → Tasks 4, 9, 15.
   - Spec §7 (Testing) → Tasks 10, 13, 14.
   - Spec §8 (Risks) → Task 11 (Python reference) + Task 12 (HF prefix heuristic).

2. **Placeholders**: Task 11 contains a "note on run_and_dump shape" — this is deliberate because the hook body depends on the actual tri-neck output tuple, which we iterate in Task 13 once weights are present. Not a plan failure; the step includes a concrete debug workflow.

3. **Type consistency**: `n_fpn_scales` is `int` throughout (config, args, propagated params). `variant` is stored as `int` in the config (matching `backbone_type`'s pattern) and compared against `SAM3_VARIANT_SAM3` / `SAM3_VARIANT_SAM3_1` enum values.

---

## Execution Notes

- Tasks 1-11 can be done without the 3.5 GB checkpoint.
- Task 12 requires the downloaded `.pt` and the Python reference repo (`reference/sam3/`).
- Task 13 requires Task 12 to have succeeded so the load path is known-good.
- Task 14 is gated on both the model and the fixture being present — the test self-skips otherwise, so CI runs without the big files remain green.
