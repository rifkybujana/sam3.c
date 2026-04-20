# SAM 3.1 Tracker Parity Fixture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-frame IoU parity test for the SAM 3.1 multiplex tracker (C-seeded, Python-propagated), plus delete the dead `tgt_pos` branch in the memory-attention entry point.

**Architecture:** A small offline regeneration pipeline (C seed-dumper → Python propagator) produces committed PNG fixtures. The C parity test loads those PNGs, runs the tracker fresh, and asserts per-frame IoU ≥ 0.75 on frames 1..3. The Python side is seeded with the C frame-0 output via `Sam3MultiplexTracking.add_new_mask`, bypassing the not-yet-ported interactive decoder. Frame-0 sanity check enforces fixture freshness without being load-bearing.

**Tech Stack:** C11, CMake, Python 3.10+ (PyTorch + upstream sam3 package), `stb_image.h`/`stb_image_write.h` (already vendored at `src/util/vendor/`).

**Spec:** `docs/superpowers/specs/2026-04-20-sam3-1-parity-fixture-design.md`

---

## File Structure

- **Modify** `src/model/tracker_multiplex.c:1085-1099` — delete dead `tgt_pos` branch.
- **Modify** `src/model/tracker_multiplex.c:1009-1020` + `tracker_multiplex.h:455-466` — drop `tgt_pos` parameter.
- **Modify** `src/model/tracker_multiplex.c:2108-2114` — drop `NULL` arg at call site.
- **Create** `tools/_cpu_patches.py` — shared CUDA→CPU redirect + triton stub + `addmm_act` fp32 patch.
- **Modify** `tools/dump_reference.py` — import from `_cpu_patches` instead of duplicating.
- **Modify** `tools/gen_video_parity_fixtures.py` — add `--variant` and `--seed-mask`; SAM 3.1 branch uses `Sam3MultiplexTracking.add_new_mask`.
- **Create** `tools/sam3_1_dump_seed.c` — minimal CLI: init → load → video_start → add_points → save grayscale PNG of frame-0 mask.
- **Create** `tests/test_helpers_png.h` — declarations for `load_png_grayscale` / `save_png_grayscale`.
- **Create** `tests/test_helpers_png.c` — stb-based implementations (stb impls already in `sam3` library).
- **Modify** `tests/test_video_parity_kids.c` — variant dispatch via `SAM3_PARITY_VARIANT`; SAM 3.1 branch implements full parity run.
- **Create** `tests/fixtures/video_kids/sam3_1/README.md` — SAM 3.1 regen instructions.
- **Create** `tests/fixtures/video_kids/sam3_1/prompts.json` — prompt specification.
- **Modify** `CMakeLists.txt` — add `SAM3_PARITY_VARIANT` option, wire `sam3_1_dump_seed` tool, link `test_helpers_png.c` into `test_video_parity_kids`.
- **Modify** `TODO.md` — mark Phase 2.5b IoU item and `tgt_pos` item done.

---

## Task 1 — Delete dead `tgt_pos` branch in memory-attention

**Files:**
- Modify: `src/model/tracker_multiplex.c:1085-1099` (remove branch)
- Modify: `src/model/tracker_multiplex.c:1009-1020` (drop parameter)
- Modify: `src/model/tracker_multiplex.c:2108-2114` (drop `NULL` at call site)
- Modify: `src/model/tracker_multiplex.h:378-466` (drop parameter from docstring + prototype)

- [ ] **Step 1: Delete the dead branch body**

In `src/model/tracker_multiplex.c`, replace lines 1085–1100 (the
`/* --- pos_enc_at_input=True: output = tgt + 0.1 * tgt_pos --- */`
block) with:

```c
	/*
	 * pos_enc_at_input=True but the current single-object call path
	 * always passes tgt_pos=NULL; the tgt + 0.1 * tgt_pos add is a
	 * no-op for our use. If a future refactor needs this path, pull
	 * it back from commit history.
	 */
	struct sam3_tensor *output = tgt;
```

- [ ] **Step 2: Drop the parameter from the function signature**

In `src/model/tracker_multiplex.c:1009-1020`, remove the `tgt_pos`
parameter so the signature becomes:

```c
struct sam3_tensor *sam3_multiplex_memory_attn_forward(
		struct sam3_graph *g,
		struct sam3_arena *arena,
		const struct sam3_multiplex_memory_attn *ma,
		struct sam3_tensor *tgt,
		struct sam3_tensor *image,
		struct sam3_tensor *memory,
		struct sam3_tensor *memory_image,
		struct sam3_tensor *memory_image_pos,
		int grid_w,
		int num_k_exclude_rope)
```

- [ ] **Step 3: Update the header prototype + docstring**

In `src/model/tracker_multiplex.h`:
- Remove the `@tgt_pos:` paragraph (lines 398–403) from the docstring.
- Remove the `struct sam3_tensor *tgt_pos,` line from the prototype
  (line 460).

- [ ] **Step 4: Drop NULL at the only call site**

In `src/model/tracker_multiplex.c:2109-2114`, remove the second
argument (the `NULL`) so the call becomes:

```c
			cond_2d =
				sam3_multiplex_memory_attn_forward(
					g, arena, &trk->transformer,
					tgt, tgt,
					memory, memory_image,
					memory_image_pos,
					W, num_k_exclude);
```

- [ ] **Step 5: Build and confirm no warnings**

Run:

```bash
cd /Users/rbisri/Documents/sam3/build && cmake --build . -j8 2>&1 | tail -40
```

Expected: clean build, no unused-parameter warnings on the function.

- [ ] **Step 6: Run the existing memory-attn forward test**

Run:

```bash
cd /Users/rbisri/Documents/sam3/build && ctest -R test_memory_attn_multiplex_forward --output-on-failure
```

Expected: PASS. (This test already passes both `NULL` tgt_pos and a
real tensor; after the refactor it should only pass NULL. If the test
still references `tgt_pos`, update it as described in step 7.)

- [ ] **Step 7: Update the test if needed**

If `tests/test_memory_attn_multiplex_forward.c` calls
`sam3_multiplex_memory_attn_forward(...)` with a `tgt_pos` argument,
remove that argument. The test header comment in `TODO.md` notes two
passes (with and without `tgt_pos`); delete the non-NULL pass since
the parameter no longer exists. Re-run the test.

- [ ] **Step 8: Run the full test suite**

Run:

```bash
cd /Users/rbisri/Documents/sam3/build && ctest --output-on-failure
```

Expected: no regressions. Especially verify `test_sam3_1_track` still
passes its mixed-sign / fg-frac invariants.

- [ ] **Step 9: Commit**

```bash
git add src/model/tracker_multiplex.c src/model/tracker_multiplex.h \
        tests/test_memory_attn_multiplex_forward.c
git commit -m "$(cat <<'EOF'
tracker/multiplex: drop dead tgt_pos branch in memory-attn forward

The pos_enc_at_input add was plumbed through the entry point but the
only caller (sam3_tracker_multiplex_track_frame) passes NULL after
Phase 2.5b landed. Remove the parameter + branch; reference trace
shows no consumer.
EOF
)"
```

---

## Task 2 — Extract shared CPU patches module

**Files:**
- Create: `tools/_cpu_patches.py`
- Modify: `tools/dump_reference.py` (replace inlined patches with import)

- [ ] **Step 1: Write `tools/_cpu_patches.py`**

Create `tools/_cpu_patches.py` with:

```python
"""
tools/_cpu_patches.py - Shared CUDA→CPU shims for dumping SAM 3 /
SAM 3.1 reference outputs on CPU-only machines.

The upstream sam3 package hardcodes CUDA in several tensor factories
and imports triton (CUDA-only). This module provides:

  install_triton_stub()   - Make `import triton` succeed without the
                            kernels ever running.
  install_cuda_redirect() - Redirect device="cuda" / torch.device("cuda")
                            to CPU in tensor factories and .cuda() ops,
                            and force has_triton_package() false for
                            torch._inductor.
  install_addmm_act_fp32() - Replace sam3.perflib.fused.addmm_act with a
                             CPU/fp32-preserving implementation (must be
                             called AFTER sam3 is imported).

Call install_triton_stub() and install_cuda_redirect() BEFORE importing
any sam3 submodule. Call install_addmm_act_fp32() after.
"""
import sys
import types

import torch
from torch.utils import _triton as _torch_triton


def install_triton_stub():
    _torch_triton.has_triton_package = lambda: False
    _torch_triton.has_triton = lambda: False

    if "triton" in sys.modules:
        return

    class _TritonStubType:
        pass

    triton_stub = types.ModuleType("triton")
    triton_stub.jit = lambda f=None, **kw: (
        f if callable(f) else (lambda g: g)
    )
    triton_stub.heuristics = lambda *a, **kw: (lambda f: f)
    triton_stub.autotune = lambda *a, **kw: (lambda f: f)

    triton_lang_stub = types.ModuleType("triton.language")
    for _name in (
        "dtype", "tensor", "pointer_type", "void", "int1",
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "float16", "float32", "float64",
        "bfloat16", "block_type", "constexpr",
    ):
        setattr(triton_lang_stub, _name, _TritonStubType)
    triton_stub.language = triton_lang_stub
    sys.modules["triton"] = triton_stub
    sys.modules["triton.language"] = triton_lang_stub


def install_cuda_redirect():
    if torch.cuda.is_available():
        return

    def _cpu_redirect(kwargs):
        dev = kwargs.get("device")
        if dev is None:
            return
        if isinstance(dev, torch.device) and dev.type == "cuda":
            kwargs["device"] = torch.device("cpu")
        elif isinstance(dev, str) and dev.startswith("cuda"):
            kwargs["device"] = "cpu"

    for _fn in ("zeros", "ones", "empty", "arange", "randn", "rand",
                "full", "linspace", "eye", "tensor", "as_tensor"):
        _orig = getattr(torch, _fn)

        def _wrap(orig):
            def _wrapped(*a, **kw):
                _cpu_redirect(kw)
                return orig(*a, **kw)
            return _wrapped
        setattr(torch, _fn, _wrap(_orig))

    torch.Tensor.cuda = lambda self, *a, **kw: self
    torch.nn.Module.cuda = lambda self, *a, **kw: self


def install_addmm_act_fp32():
    import sam3.perflib.fused as _fused

    def _addmm_act_fp32(activation, linear, mat1):
        if torch.is_grad_enabled():
            raise ValueError("Expected grad to be disabled.")
        w = linear.weight.detach()
        b = linear.bias.detach()
        y = torch.nn.functional.linear(mat1, w, b)
        if activation in (torch.nn.functional.relu, torch.nn.ReLU):
            return torch.nn.functional.relu(y)
        if activation in (torch.nn.functional.gelu, torch.nn.GELU):
            return torch.nn.functional.gelu(y)
        raise ValueError(f"Unexpected activation {activation}")

    _fused.addmm_act = _addmm_act_fp32
    import sam3.model.vitdet as _vitdet
    _vitdet.addmm_act = _addmm_act_fp32
```

- [ ] **Step 2: Swap `dump_reference.py` to import from the module**

In `tools/dump_reference.py`:
- Replace lines 30–98 (the block between the docstring and
  `from PIL import Image`) with:

```python
from _cpu_patches import (
    install_triton_stub, install_cuda_redirect, install_addmm_act_fp32,
)

# Must run before importing any sam3 submodule.
install_triton_stub()
install_cuda_redirect()
```

- Replace the body of `_install_addmm_act_fp32_patch()` (lines
  115–139) with a thin shim:

```python
def _install_addmm_act_fp32_patch():
    install_addmm_act_fp32()
```

- Add `import os, sys` at the top and prepend `tools/` to the path if
  needed so the relative import works:

```python
sys.path.insert(0, os.path.dirname(__file__))
```

(If `tools/` is already on the Python path when invoked from the repo
root, this is redundant but harmless.)

- [ ] **Step 3: Smoke-test the refactor**

Run (no checkpoint needed — argparse alone exercises the import graph):

```bash
cd /Users/rbisri/Documents/sam3 && python tools/dump_reference.py --help
```

Expected: help text printed, exit 0.

- [ ] **Step 4: Commit**

```bash
git add tools/_cpu_patches.py tools/dump_reference.py
git commit -m "$(cat <<'EOF'
tools: extract CUDA->CPU shims into shared _cpu_patches module

dump_reference.py's CUDA redirect + triton stub + addmm_act fp32
patch will be reused by gen_video_parity_fixtures.py's SAM 3.1
branch. Pull them into tools/_cpu_patches.py.
EOF
)"
```

---

## Task 3 — PNG grayscale helpers for tests

**Files:**
- Create: `tests/test_helpers_png.h`
- Create: `tests/test_helpers_png.c`

- [ ] **Step 1: Write the header**

Create `tests/test_helpers_png.h`:

```c
/*
 * tests/test_helpers_png.h - 8-bit grayscale PNG load/save for tests.
 *
 * Thin wrappers around the stb_image.h / stb_image_write.h APIs whose
 * implementations are already defined once in src/util/image.c (linked
 * into the sam3 library). Test translation units only need to #include
 * this header; they do not re-declare the stb implementations.
 *
 * Key types:  uint8_t buffers + int dimensions
 * Depends on: <stdint.h>, src/util/vendor/stb_image.h (impl in sam3 lib)
 * Used by:    tests/test_video_parity_kids.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_TEST_HELPERS_PNG_H
#define SAM3_TEST_HELPERS_PNG_H

#include <stdint.h>

/*
 * load_png_grayscale - Load an 8-bit single-channel PNG.
 *
 * Returns a malloc'd buffer of size (*out_h) * (*out_w) uint8_t, or
 * NULL on error. Caller frees via free(). PNGs with >1 channel are
 * forced to grayscale by the decoder.
 */
uint8_t *load_png_grayscale(const char *path, int *out_h, int *out_w);

/*
 * save_png_grayscale - Write an 8-bit single-channel PNG.
 *
 * Returns 0 on success, -1 on error (errors are logged via
 * sam3_log_error). @data is row-major, size h*w.
 */
int save_png_grayscale(const char *path, const uint8_t *data,
                       int h, int w);

#endif /* SAM3_TEST_HELPERS_PNG_H */
```

- [ ] **Step 2: Write the implementation**

Create `tests/test_helpers_png.c`:

```c
/*
 * tests/test_helpers_png.c - PNG helpers for parity fixture tests.
 *
 * stb_image / stb_image_write implementations are defined once in
 * src/util/image.c (compiled into the sam3 library). This TU only
 * calls the extern symbols.
 *
 * Key types:  (none)
 * Depends on: test_helpers_png.h, src/util/vendor/stb_image.h
 * Used by:    tests/test_video_parity_kids.c (SAM 3.1 variant)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_helpers_png.h"
#include "util/log.h"

/* stb_image / stb_image_write declarations without re-defining the
 * implementation (implementation lives in src/util/image.c). */
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#endif
#include "util/vendor/stb_image.h"
#include "util/vendor/stb_image_write.h"
#ifdef __clang__
#pragma clang diagnostic pop
#endif

uint8_t *load_png_grayscale(const char *path, int *out_h, int *out_w)
{
	int w, h, channels_in_file;

	if (!path || !out_h || !out_w) {
		sam3_log_error("load_png_grayscale: NULL arg");
		return NULL;
	}

	/* Request 1 channel; stb down-converts automatically. */
	uint8_t *data = stbi_load(path, &w, &h, &channels_in_file, 1);
	if (!data) {
		sam3_log_error("load_png_grayscale: %s: %s",
			       path, stbi_failure_reason());
		return NULL;
	}
	*out_h = h;
	*out_w = w;
	return data;
}

int save_png_grayscale(const char *path, const uint8_t *data,
		       int h, int w)
{
	if (!path || !data || h <= 0 || w <= 0) {
		sam3_log_error("save_png_grayscale: bad arg");
		return -1;
	}
	int stride = w;
	if (!stbi_write_png(path, w, h, 1, data, stride)) {
		sam3_log_error("save_png_grayscale: stbi_write_png failed "
			       "for %s", path);
		return -1;
	}
	return 0;
}
```

- [ ] **Step 3: Build — the TU is not yet in any test target, so this is
  a syntactic check only**

No CMake change in this task. Run:

```bash
cd /Users/rbisri/Documents/sam3/build && \
  clang -std=c11 -Wall -Wextra -Wpedantic \
        -I../include -I../src \
        -c ../tests/test_helpers_png.c -o /tmp/t.o
```

Expected: compiles with no warnings. Discard `/tmp/t.o`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_helpers_png.h tests/test_helpers_png.c
git commit -m "$(cat <<'EOF'
tests: add grayscale PNG load/save helpers

Declaration-only include of stb_image / stb_image_write. The
implementations are already defined in src/util/image.c and linked
into sam3, so test TUs can call stbi_load / stbi_write_png directly.
Used next by the SAM 3.1 parity fixture test.
EOF
)"
```

---

## Task 4 — `tools/sam3_1_dump_seed.c` CLI

**Files:**
- Create: `tools/sam3_1_dump_seed.c`
- Modify: `CMakeLists.txt` (register the executable)

- [ ] **Step 1: Write the CLI source**

Create `tools/sam3_1_dump_seed.c`:

```c
/*
 * tools/sam3_1_dump_seed.c - Dump the C frame-0 mask as a grayscale PNG.
 *
 * Drives sam3_init -> sam3_load_model -> sam3_video_start ->
 * sam3_video_add_points on frame 0 of a video with a single
 * point prompt, binarizes the resulting logits (>0 -> 255, else 0),
 * and writes the mask as an 8-bit grayscale PNG.
 *
 * Used exclusively to seed tests/fixtures/video_kids/sam3_1/seed_mask.png
 * so the Python reference propagator (tools/gen_video_parity_fixtures.py
 * --variant sam3.1) can feed the same seed into
 * Sam3MultiplexTracking.add_new_mask. See
 * docs/superpowers/plans/2026-04-20-sam3-1-parity-fixture.md.
 *
 * Key types:  (uses public sam3.h API)
 * Depends on: sam3/sam3.h, src/util/vendor/stb_image_write.h (impl in sam3 lib)
 * Used by:    manual fixture regeneration only (not CI, not CTest)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sam3/sam3.h"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#endif
#include "util/vendor/stb_image_write.h"
#ifdef __clang__
#pragma clang diagnostic pop
#endif

static void usage(const char *argv0)
{
	fprintf(stderr,
		"Usage: %s --model PATH --video PATH "
		"--point X,Y,LABEL --out PATH\n\n"
		"  X, Y are normalized [0,1] click coordinates.\n"
		"  LABEL is 1 (positive) or 0 (negative).\n",
		argv0);
}

static int parse_point(const char *s, float *x, float *y, int *label)
{
	if (sscanf(s, "%f,%f,%d", x, y, label) != 3)
		return -1;
	if (*x < 0.0f || *x > 1.0f || *y < 0.0f || *y > 1.0f)
		return -1;
	if (*label != 0 && *label != 1)
		return -1;
	return 0;
}

int main(int argc, char **argv)
{
	const char *model_path = NULL;
	const char *video_path = NULL;
	const char *out_path   = NULL;
	float px = 0.5f, py = 0.5f;
	int   plabel = 1;
	int   have_point = 0;

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--model") && i + 1 < argc) {
			model_path = argv[++i];
		} else if (!strcmp(argv[i], "--video") && i + 1 < argc) {
			video_path = argv[++i];
		} else if (!strcmp(argv[i], "--out") && i + 1 < argc) {
			out_path = argv[++i];
		} else if (!strcmp(argv[i], "--point") && i + 1 < argc) {
			if (parse_point(argv[++i], &px, &py, &plabel) != 0) {
				usage(argv[0]);
				return 1;
			}
			have_point = 1;
		} else {
			usage(argv[0]);
			return 1;
		}
	}
	if (!model_path || !video_path || !out_path || !have_point) {
		usage(argv[0]);
		return 1;
	}

	sam3_ctx *ctx = sam3_init();
	if (!ctx) {
		fprintf(stderr, "sam3_init failed\n");
		return 2;
	}
	if (sam3_load_model(ctx, model_path) != SAM3_OK) {
		fprintf(stderr, "sam3_load_model(%s) failed\n", model_path);
		sam3_free(ctx);
		return 2;
	}

	sam3_video_session *sess = NULL;
	if (sam3_video_start(ctx, video_path, &sess) != SAM3_OK) {
		fprintf(stderr, "sam3_video_start(%s) failed\n", video_path);
		sam3_free(ctx);
		return 2;
	}

	struct sam3_point pt;
	memset(&pt, 0, sizeof(pt));
	pt.x = px; pt.y = py; pt.label = plabel;

	struct sam3_video_frame_result r;
	memset(&r, 0, sizeof(r));
	if (sam3_video_add_points(sess, 0, 1, &pt, 1, &r) != SAM3_OK) {
		fprintf(stderr, "sam3_video_add_points failed\n");
		sam3_video_end(sess);
		sam3_free(ctx);
		return 3;
	}
	if (r.n_objects < 1 || !r.objects || !r.objects[0].mask) {
		fprintf(stderr, "add_points returned no mask\n");
		sam3_video_frame_result_free(&r);
		sam3_video_end(sess);
		sam3_free(ctx);
		return 3;
	}

	int H = r.objects[0].mask_h;
	int W = r.objects[0].mask_w;
	const float *logits = r.objects[0].mask;
	uint8_t *bin = malloc((size_t)H * (size_t)W);
	if (!bin) {
		fprintf(stderr, "malloc failed\n");
		sam3_video_frame_result_free(&r);
		sam3_video_end(sess);
		sam3_free(ctx);
		return 4;
	}
	for (int i = 0; i < H * W; i++)
		bin[i] = (logits[i] > 0.0f) ? (uint8_t)255 : (uint8_t)0;

	int ok = stbi_write_png(out_path, W, H, 1, bin, W);
	free(bin);
	sam3_video_frame_result_free(&r);
	sam3_video_end(sess);
	sam3_free(ctx);

	if (!ok) {
		fprintf(stderr, "stbi_write_png(%s) failed\n", out_path);
		return 4;
	}
	fprintf(stderr, "wrote seed mask %dx%d to %s\n", W, H, out_path);
	return 0;
}
```

- [ ] **Step 2: Register the executable in CMake**

In `CMakeLists.txt`, after the `gen_nhwc_fixtures` block (around line
205), add:

```cmake
add_executable(sam3_1_dump_seed tools/sam3_1_dump_seed.c)
target_link_libraries(sam3_1_dump_seed sam3)
target_include_directories(sam3_1_dump_seed PRIVATE
	${CMAKE_SOURCE_DIR}/src)
```

The `src/` include path is needed so the `util/vendor/stb_image_write.h`
include resolves.

- [ ] **Step 3: Build**

Run:

```bash
cd /Users/rbisri/Documents/sam3/build && cmake .. && \
  cmake --build . --target sam3_1_dump_seed -j8 2>&1 | tail -30
```

Expected: clean build, `sam3_1_dump_seed` executable in
`build/sam3_1_dump_seed`.

- [ ] **Step 4: Skip-path smoke test (no model required)**

Run:

```bash
./build/sam3_1_dump_seed
```

Expected: exit 1 with the usage message printed to stderr.

- [ ] **Step 5: Commit**

```bash
git add tools/sam3_1_dump_seed.c CMakeLists.txt
git commit -m "$(cat <<'EOF'
tools: add sam3_1_dump_seed CLI for parity fixture seeds

Single-shot helper that dumps the C frame-0 mask as a grayscale PNG.
Used by tools/gen_video_parity_fixtures.py --variant sam3.1 to seed
Sam3MultiplexTracking.add_new_mask with the same placeholder mask the
C tracker will produce at test time.
EOF
)"
```

---

## Task 5 — Extend `gen_video_parity_fixtures.py` for SAM 3.1

**Files:**
- Modify: `tools/gen_video_parity_fixtures.py` (full rewrite — the
  existing 69-line scaffold becomes variant-aware)

- [ ] **Step 1: Rewrite the generator**

Replace the contents of `tools/gen_video_parity_fixtures.py` with:

```python
#!/usr/bin/env python3
"""Generate per-frame mask PNGs from the Python reference predictor.

Variants:
  --variant sam3     - Sam3VideoPredictor + point prompts (legacy path,
                       unchanged regen flow for the existing
                       tests/fixtures/video_kids/ scaffold).
  --variant sam3.1   - Sam3MultiplexTracking + add_new_mask seed path.
                       Requires --seed-mask (grayscale PNG produced by
                       ./build/sam3_1_dump_seed).

NOT run automatically. Invoke by hand when reference outputs need a
refresh. See tests/fixtures/video_kids/README.md and
tests/fixtures/video_kids/sam3_1/README.md.

Usage (sam3.1):

    SAM3_CKPT=... SAM3_BPE=... python tools/gen_video_parity_fixtures.py \\
        --variant sam3.1 \\
        --video ../assets/kids.mp4 \\
        --frames 3 \\
        --seed-mask ../tests/fixtures/video_kids/sam3_1/seed_mask.png \\
        --out ../tests/fixtures/video_kids/sam3_1/
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

# Upstream sam3 package lives under reference/
sys.path.insert(
    0,
    os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "reference", "sam3")),
)
# Shared CPU shims (triton stub, CUDA redirect). Must run before any
# sam3 import.
sys.path.insert(0, os.path.dirname(__file__))

from _cpu_patches import (  # noqa: E402
    install_triton_stub, install_cuda_redirect, install_addmm_act_fp32,
)

install_triton_stub()
install_cuda_redirect()

import torch  # noqa: E402


def main_sam3(args):
    """Legacy SAM 3 regen path. Unchanged from pre-refactor."""
    from sam3.sam3_video_predictor import Sam3VideoPredictor

    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)

    predictor = Sam3VideoPredictor(
        checkpoint_path=os.environ["SAM3_CKPT"],
        bpe_path=os.environ["SAM3_BPE"],
    )
    state = predictor.init_state(video_path=args.video)

    prompts = {
        "obj_1": {"frame": 0, "points": [[400, 250]], "labels": [1]},
        "obj_2": {"frame": 0, "points": [[600, 250]], "labels": [1]},
    }
    with open(os.path.join(args.out, "prompts.json"), "w") as f:
        json.dump(prompts, f, indent=2)

    for name, p in prompts.items():
        obj_id = int(name.split("_")[1])
        predictor.add_new_points_or_box(
            state,
            frame_idx=p["frame"],
            obj_id=obj_id,
            points=np.array(p["points"]),
            labels=np.array(p["labels"]),
        )

    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
        if frame_idx >= args.frames:
            break
        for obj_id, m in zip(obj_ids, masks):
            arr = (m > 0).cpu().numpy().astype(np.uint8) * 255
            Image.fromarray(arr.squeeze()).save(os.path.join(
                args.out, "frames",
                f"frame_{frame_idx:04d}_obj_{obj_id}.png"))


def _build_sam3_1_tracker(checkpoint, bpe_path):
    """Assemble a CPU-safe Sam3MultiplexTracking from the checkpoint.

    The user-facing Sam3MultiplexVideoPredictor hardcodes CUDA autocast;
    skip it and instantiate Sam3MultiplexTracking directly so CPU runs
    work. Re-uses the same multiplex builders dump_reference.py uses.
    """
    install_addmm_act_fp32()  # Patch perflib.addmm_act before eval runs

    from sam3.model_builder import build_sam3_multiplex_tracker
    model = build_sam3_multiplex_tracker(
        bpe_path=bpe_path,
        device="cpu",
        eval_mode=True,
        checkpoint_path=checkpoint,
        load_from_HF=False,
    )
    # bf16 is flaky on CPU for some ops; force fp32.
    model = model.float().eval()
    return model


def main_sam3_1(args):
    if not args.seed_mask:
        print("ERROR: --variant sam3.1 requires --seed-mask",
              file=sys.stderr)
        return 2

    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)

    # Prompt record mirrors the C test: single object, center point.
    prompts = {
        "obj_1": {"frame": 0, "points": [[0.5, 0.5]], "labels": [1]},
    }
    with open(os.path.join(args.out, "prompts.json"), "w") as f:
        json.dump(prompts, f, indent=2)

    seed = np.array(Image.open(args.seed_mask).convert("L"))
    seed_bool = (seed > 127)
    seed_t = torch.from_numpy(seed_bool).bool()

    bpe = os.environ.get("SAM3_BPE")
    if not bpe:
        import pkg_resources
        bpe = pkg_resources.resource_filename(
            "sam3", "assets/bpe_simple_vocab_16e6.txt.gz")

    model = _build_sam3_1_tracker(os.environ["SAM3_CKPT"], bpe)

    # Mirrors Sam3MultiplexVideoPredictor.init_state but on CPU.
    state = model.init_state(
        resource_path=args.video,
        offload_video_to_cpu=True,
        async_loading_frames=False,
        use_cv2=True,
    )

    # Seed object 1 on frame 0 with the C-produced mask.
    model.add_new_mask(
        inference_state=state,
        frame_idx=0,
        obj_id=1,
        mask=seed_t,
    )

    count = 0
    for frame_idx, obj_ids, masks in model.propagate_in_video(
            state, start_frame_idx=0, max_frame_num_to_track=args.frames,
            reverse=False):
        if frame_idx == 0:
            # Seed frame -- skipped to keep fixture semantics clean.
            # The C test sanity-checks its own frame-0 mask against
            # seed_mask.png directly.
            continue
        for obj_id, m in zip(obj_ids, masks):
            arr = (m > 0).cpu().numpy().astype(np.uint8) * 255
            Image.fromarray(arr.squeeze()).save(os.path.join(
                args.out, "frames",
                f"frame_{frame_idx:04d}_obj_{obj_id}.png"))
        count += 1
        if count >= args.frames:
            break

    print(f"wrote {count} frames to {args.out}/frames/",
          file=sys.stderr)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["sam3", "sam3.1"],
                    required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--seed-mask", default=None,
                    help="grayscale PNG seed (SAM 3.1 only)")
    args = ap.parse_args()

    if args.frames is None:
        args.frames = 30 if args.variant == "sam3" else 3

    if args.variant == "sam3":
        return main_sam3(args) or 0
    return main_sam3_1(args) or 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Validate the Python reference entry points exist**

Some details (e.g. `build_sam3_multiplex_tracker`, exact kwargs of
`init_state` and `add_new_mask`) come from the upstream reference.
Verify at implementation time:

```bash
cd /Users/rbisri/Documents/sam3 && \
  grep -n "^def build_sam3_multiplex_tracker\|^def add_new_mask\|^    def add_new_mask" \
       reference/sam3/sam3/model_builder.py \
       reference/sam3/sam3/model/sam3_multiplex_tracking.py \
       reference/sam3/sam3/model/sam3_tracker_base.py
```

If `build_sam3_multiplex_tracker` doesn't exist under that exact name,
use whatever the multiplex checkpoint-loader helper is called upstream
(check for nearby `_create_sam3_multiplex_*` / `build_sam3_multiplex_*`
functions in `model_builder.py` and adjust the import accordingly).
Similarly for `add_new_mask` — if the kwargs differ (e.g. `mask` vs
`mask_logits`), match the upstream signature.

- [ ] **Step 3: Dry-run the generator with --help**

```bash
cd /Users/rbisri/Documents/sam3 && \
  python tools/gen_video_parity_fixtures.py --help
```

Expected: argparse help printed, exit 0. Confirms the imports and CPU
patches load without a real checkpoint.

- [ ] **Step 4: Commit**

```bash
git add tools/gen_video_parity_fixtures.py
git commit -m "$(cat <<'EOF'
tools: extend gen_video_parity_fixtures for SAM 3.1

--variant sam3.1 seeds Sam3MultiplexTracking.add_new_mask with the
C frame-0 mask (produced by ./build/sam3_1_dump_seed) and propagates
forward --frames frames, writing per-frame PNGs. Bypasses the
interactive decoder that sub-project 3 hasn't ported yet.
EOF
)"
```

---

## Task 6 — SAM 3.1 parity branch in `test_video_parity_kids.c`

**Files:**
- Modify: `tests/test_video_parity_kids.c` (add SAM 3.1 variant dispatch)
- Modify: `CMakeLists.txt` (add `SAM3_PARITY_VARIANT`, link
  `test_helpers_png.c`)

- [ ] **Step 1: Add the variant option to CMake**

In `CMakeLists.txt`, after the existing `SAM3_BUILD_PARITY_TESTS`
block (around line 23), add:

```cmake
set(SAM3_PARITY_VARIANT "sam3_1" CACHE STRING
	"Variant for test_video_parity_kids {sam3, sam3_1}")
set_property(CACHE SAM3_PARITY_VARIANT PROPERTY STRINGS sam3 sam3_1)
```

Update the `test_video_parity_kids` target block (line 351–356) to:

```cmake
	# test_video_parity_kids needs source dir, model path, variant
	if(TARGET test_video_parity_kids)
		target_compile_definitions(test_video_parity_kids PRIVATE
			SAM3_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
			SAM3_TEST_MODEL="${SAM3_TEST_MODEL}"
			SAM3_PARITY_VARIANT_${SAM3_PARITY_VARIANT}=1)
		target_sources(test_video_parity_kids PRIVATE
			${CMAKE_SOURCE_DIR}/tests/test_helpers_png.c)
		target_include_directories(test_video_parity_kids PRIVATE
			${CMAKE_SOURCE_DIR}/src
			${CMAKE_SOURCE_DIR}/tests)
	endif()
```

`SAM3_PARITY_VARIANT_sam3_1=1` → `#ifdef SAM3_PARITY_VARIANT_sam3_1`
switches the C branch.

- [ ] **Step 2: Write the SAM 3.1 branch in `test_video_parity_kids.c`**

Replace the contents of `tests/test_video_parity_kids.c` with:

```c
/*
 * tests/test_video_parity_kids.c - End-to-end parity vs Python on kids.mp4
 *
 * Two variants selected at build time via SAM3_PARITY_VARIANT_{sam3,sam3_1}:
 *
 *   sam3    - Legacy scaffold (Sam3VideoPredictor text prompt). Still a
 *             stub: PNG compare is not wired. Skip at runtime when
 *             fixtures absent.
 *   sam3_1  - C-seeded, Python-propagated parity:
 *               1. sam3_init + load sam3.1 + video_start(kids.mp4)
 *               2. add_points(center, frame=0) -> C frame-0 mask
 *               3. Load seed_mask.png; IoU(C_f0, seed) warn < 0.9,
 *                                       fail  < 0.5
 *               4. propagate(FORWARD, callback stops at 3 frames)
 *               5. For n in {1,2,3}: IoU(C_fn, frames/frame_000n_obj_1.png)
 *                  >= 0.75
 *
 * Gated on SAM3_BUILD_PARITY_TESTS=ON. Self-skips on missing fixtures or
 * model so the CI default profile stays clean.
 *
 * Key types: sam3_video_session
 * Depends on: sam3/sam3.h, test_helpers.h, test_helpers_png.h
 * Used by:    CTest (opt-in)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "test_helpers.h"
#include "test_helpers_png.h"
#include "sam3/sam3.h"

#ifndef SAM3_SOURCE_DIR
#error "SAM3_SOURCE_DIR must be defined (via CMake)"
#endif
#ifndef SAM3_TEST_MODEL
#error "SAM3_TEST_MODEL must be defined (via -DSAM3_TEST_MODEL=<path>)"
#endif

#if !defined(SAM3_PARITY_VARIANT_sam3) && !defined(SAM3_PARITY_VARIANT_sam3_1)
#error "SAM3_PARITY_VARIANT_{sam3,sam3_1} must be set (CMake cache)"
#endif

#ifdef SAM3_PARITY_VARIANT_sam3

/* --- SAM 3 variant: original scaffold, unchanged --- */

static int fixture_dir_exists(void)
{
	struct stat st;
	const char *path =
		SAM3_SOURCE_DIR "/tests/fixtures/video_kids/frames";
	return (stat(path, &st) == 0) && S_ISDIR(st.st_mode);
}

int main(void)
{
	if (!fixture_dir_exists()) {
		fprintf(stderr,
			"SKIP: fixtures absent. See "
			"tests/fixtures/video_kids/README.md\n");
		return 0;
	}
	if (SAM3_TEST_MODEL[0] == '\0') {
		fprintf(stderr, "SKIP: SAM3_TEST_MODEL is empty\n");
		return 0;
	}
	fprintf(stderr,
		"NOTE: test_video_parity_kids sam3 variant is a scaffold. "
		"See tests/fixtures/video_kids/README.md.\n");
	return 0;
}

#else /* SAM3_PARITY_VARIANT_sam3_1 */

/* --- SAM 3.1 variant: C-seeded parity run --- */

#define MODEL_PATH   SAM3_SOURCE_DIR "/models/sam3.1.sam3"
#define VIDEO_PATH   SAM3_SOURCE_DIR "/assets/kids.mp4"
#define FIXTURE_DIR  SAM3_SOURCE_DIR "/tests/fixtures/video_kids/sam3_1"
#define SEED_PATH    FIXTURE_DIR "/seed_mask.png"

#define N_PROP_FRAMES 3
#define IOU_FRAME_THRESH 0.75f
#define IOU_SEED_WARN    0.90f
#define IOU_SEED_FAIL    0.50f

static float
mask_iou_logits_vs_png(const float *logits, int h, int w,
		       const uint8_t *png, int png_h, int png_w)
{
	if (h != png_h || w != png_w) {
		fprintf(stderr,
			"iou: dim mismatch logits %dx%d vs png %dx%d\n",
			h, w, png_h, png_w);
		return -1.0f;
	}
	size_t inter = 0, uni = 0;
	for (int i = 0; i < h * w; i++) {
		int a = (logits[i] > 0.0f);
		int b = (png[i] > 127);
		inter += (a & b);
		uni   += (a | b);
	}
	if (uni == 0)
		return 1.0f;   /* both empty — treat as matching */
	return (float)inter / (float)uni;
}

struct cb_state {
	int        frames_seen;
	int        passed;      /* 0 on any per-frame IoU failure */
	const char *fixture_dir;
};

static int
frame_cb(const struct sam3_video_frame_result *r, void *ud)
{
	struct cb_state *s = (struct cb_state *)ud;
	s->frames_seen++;

	if (r->n_objects < 1 || !r->objects || !r->objects[0].mask) {
		fprintf(stderr,
			"parity: frame %d missing mask\n", r->frame_idx);
		s->passed = 0;
		return 1;
	}

	char path[1024];
	snprintf(path, sizeof(path), "%s/frames/frame_%04d_obj_1.png",
		 s->fixture_dir, r->frame_idx);

	int ph = 0, pw = 0;
	uint8_t *png = load_png_grayscale(path, &ph, &pw);
	if (!png) {
		fprintf(stderr,
			"parity: frame %d fixture %s missing / unreadable\n",
			r->frame_idx, path);
		s->passed = 0;
		return 1;
	}

	float iou = mask_iou_logits_vs_png(
		r->objects[0].mask, r->objects[0].mask_h,
		r->objects[0].mask_w, png, ph, pw);
	free(png);
	if (iou < 0.0f) {
		s->passed = 0;
		return 1;
	}
	fprintf(stderr, "parity: frame %d IoU=%.4f\n",
		r->frame_idx, (double)iou);
	if (iou < IOU_FRAME_THRESH) {
		fprintf(stderr,
			"parity: frame %d IoU %.4f < %.4f — FAIL\n",
			r->frame_idx, (double)iou, (double)IOU_FRAME_THRESH);
		s->passed = 0;
	}
	return (s->frames_seen >= N_PROP_FRAMES) ? 1 : 0;
}

int main(void)
{
	if (access(MODEL_PATH, F_OK) != 0 ||
	    access(VIDEO_PATH, F_OK) != 0 ||
	    access(SEED_PATH,  F_OK) != 0) {
		fprintf(stderr,
			"SKIP: model/video/seed missing. See %s/README.md\n",
			FIXTURE_DIR);
		return 0;
	}

	sam3_ctx *ctx = sam3_init();
	ASSERT_NOT_NULL(ctx);
	ASSERT_EQ(sam3_load_model(ctx, MODEL_PATH), SAM3_OK);

	sam3_video_session *sess = NULL;
	ASSERT_EQ(sam3_video_start(ctx, VIDEO_PATH, &sess), SAM3_OK);
	ASSERT_NOT_NULL(sess);

	struct sam3_point pt;
	memset(&pt, 0, sizeof(pt));
	pt.x = 0.5f; pt.y = 0.5f; pt.label = 1;

	struct sam3_video_frame_result r0;
	memset(&r0, 0, sizeof(r0));
	ASSERT_EQ(sam3_video_add_points(sess, 0, 1, &pt, 1, &r0), SAM3_OK);
	ASSERT(r0.n_objects == 1);
	ASSERT_NOT_NULL(r0.objects);
	ASSERT_NOT_NULL(r0.objects[0].mask);

	/* Frame-0 sanity: compare C output to committed seed */
	int sh = 0, sw = 0;
	uint8_t *seed = load_png_grayscale(SEED_PATH, &sh, &sw);
	ASSERT_NOT_NULL(seed);
	float seed_iou = mask_iou_logits_vs_png(
		r0.objects[0].mask, r0.objects[0].mask_h,
		r0.objects[0].mask_w, seed, sh, sw);
	free(seed);
	ASSERT(seed_iou >= 0.0f);
	fprintf(stderr, "parity: frame-0 vs seed IoU=%.4f\n",
		(double)seed_iou);
	if (seed_iou < IOU_SEED_FAIL) {
		fprintf(stderr,
			"parity: frame-0 IoU %.4f < %.4f — fixture is stale, "
			"regenerate via sam3_1_dump_seed + "
			"gen_video_parity_fixtures.py\n",
			(double)seed_iou, (double)IOU_SEED_FAIL);
		ASSERT(0);
	} else if (seed_iou < IOU_SEED_WARN) {
		fprintf(stderr,
			"parity: WARN frame-0 IoU %.4f < %.4f "
			"(numerical drift tolerated)\n",
			(double)seed_iou, (double)IOU_SEED_WARN);
	}
	sam3_video_frame_result_free(&r0);

	struct cb_state cbs = {0};
	cbs.passed = 1;
	cbs.fixture_dir = FIXTURE_DIR;

	ASSERT_EQ(sam3_video_propagate(sess, SAM3_PROPAGATE_FORWARD,
				       frame_cb, &cbs), SAM3_OK);
	fprintf(stderr, "parity: frames_seen=%d passed=%d\n",
		cbs.frames_seen, cbs.passed);
	ASSERT(cbs.frames_seen >= N_PROP_FRAMES);
	ASSERT(cbs.passed == 1);

	sam3_video_end(sess);
	sam3_free(ctx);
	TEST_REPORT();
}

#endif /* SAM3_PARITY_VARIANT_* */
```

- [ ] **Step 3: Confirm build compiles both variants**

The test is only added to `TEST_SOURCES` when
`SAM3_BUILD_PARITY_TESTS=ON`. Configure with the default variant:

```bash
cd /Users/rbisri/Documents/sam3/build && \
  cmake -DSAM3_BUILD_PARITY_TESTS=ON \
        -DSAM3_TEST_MODEL="$PWD/../models/sam3.1.sam3" \
        -DSAM3_PARITY_VARIANT=sam3_1 .. && \
  cmake --build . --target test_video_parity_kids -j8 2>&1 | tail -40
```

Expected: clean build.

Then re-configure with the legacy variant to confirm both compile:

```bash
cd /Users/rbisri/Documents/sam3/build && \
  cmake -DSAM3_PARITY_VARIANT=sam3 .. && \
  cmake --build . --target test_video_parity_kids -j8 2>&1 | tail -20
```

Expected: clean build.

Leave the config as `sam3_1` for Task 7:

```bash
cd /Users/rbisri/Documents/sam3/build && \
  cmake -DSAM3_PARITY_VARIANT=sam3_1 ..
```

- [ ] **Step 4: Skip-path test (fixtures absent)**

Before the fixture directory exists, the SAM 3.1 branch should skip
cleanly:

```bash
cd /Users/rbisri/Documents/sam3/build && ./test_video_parity_kids 2>&1
```

Expected: `SKIP: model/video/seed missing...` and exit 0.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt tests/test_video_parity_kids.c
git commit -m "$(cat <<'EOF'
tests: SAM 3.1 variant of test_video_parity_kids

Compile-time dispatch via SAM3_PARITY_VARIANT_sam3_1. Runs the
C-seeded, Python-propagated parity check: sanity-compare frame-0
vs seed_mask.png (warn < 0.9, fail < 0.5), then propagate 3 frames
and assert per-frame IoU >= 0.75 against committed PNGs.
EOF
)"
```

---

## Task 7 — Regenerate fixtures and validate

**Files:**
- Create: `tests/fixtures/video_kids/sam3_1/README.md`
- Create: `tests/fixtures/video_kids/sam3_1/prompts.json` (via generator)
- Create: `tests/fixtures/video_kids/sam3_1/seed_mask.png` (via dumper)
- Create: `tests/fixtures/video_kids/sam3_1/frames/frame_000{1,2,3}_obj_1.png`
  (via generator)
- Modify: `tests/fixtures/video_kids/README.md` (add pointer to SAM 3.1)

**Note:** this task requires a working `models/sam3.1.sam3` (~several
GB) and the Python reference environment. If either is absent, stop
after step 1 and note that fixture regeneration is pending hardware
availability; the CI parity test will remain in its SKIP path until
then. Task 8 (TODO.md update) can still proceed.

- [ ] **Step 1: Write the SAM 3.1 fixture README**

Create `tests/fixtures/video_kids/sam3_1/README.md`:

```markdown
# SAM 3.1 parity fixtures: kids.mp4

C-seeded, Python-propagated parity fixtures for
`test_video_parity_kids` (SAM 3.1 variant).

## Files

- `prompts.json` — single object, center point (0.5, 0.5) on frame 0.
  Matches the C test's fixed prompt.
- `seed_mask.png` — C frame-0 mask (grayscale, >0 → 255). Produced by
  `./build/sam3_1_dump_seed`. Committed as-is.
- `frames/frame_NNNN_obj_1.png` — Python reference propagation for
  frames 1..3, seeded via `Sam3MultiplexTracking.add_new_mask(seed_mask)`.

## Regenerating

    # Step 1: Build the C helper and dump the frame-0 seed
    cmake -DSAM3_BUILD_PARITY_TESTS=ON .. && \
      cmake --build . --target sam3_1_dump_seed -j8
    ./sam3_1_dump_seed \
        --model ../models/sam3.1.sam3 \
        --video ../assets/kids.mp4 \
        --point 0.5,0.5,1 \
        --out ../tests/fixtures/video_kids/sam3_1/seed_mask.png

    # Step 2: Propagate Python from that seed
    cd ../tools
    SAM3_CKPT=/path/to/sam3.1_multiplex.pt \
    SAM3_BPE=/path/to/bpe_simple_vocab_16e6.txt.gz \
      python gen_video_parity_fixtures.py \
        --variant sam3.1 \
        --video ../assets/kids.mp4 \
        --frames 3 \
        --seed-mask ../tests/fixtures/video_kids/sam3_1/seed_mask.png \
        --out ../tests/fixtures/video_kids/sam3_1/

Requires Python 3.10+, PyTorch (CPU-only is supported via
`tools/_cpu_patches.py`), and the upstream reference at
`reference/sam3/`.

## Gating

The SAM 3.1 parity test is the `SAM3_PARITY_VARIANT=sam3_1` variant
of `tests/test_video_parity_kids.c`, compiled only under
`SAM3_BUILD_PARITY_TESTS=ON`. Without that option the test is
excluded from the build. At runtime it skips cleanly when any of
`models/sam3.1.sam3`, `assets/kids.mp4`, or `seed_mask.png` are
absent.

## Assertions (C side)

- Frame-0 IoU (C vs seed_mask): warn if < 0.90, fail if < 0.50.
- Propagation frames 1..3: per-frame IoU ≥ 0.75 vs committed PNGs.

A failure on frame 1..3 indicates a regression in the memory
stream (Phase 2.5b items B1–B6 or later changes to the multiplex
mask decoder / memory-attn). Frame-0 drift indicates either a
change in the placeholder obj_ptr path or that the interactive
decoder (sub-project 3) has landed, in which case the fixture
needs regeneration.
```

- [ ] **Step 2: Regenerate the seed mask**

Assuming `models/sam3.1.sam3` is present:

```bash
cd /Users/rbisri/Documents/sam3/build && \
  ./sam3_1_dump_seed \
    --model ../models/sam3.1.sam3 \
    --video ../assets/kids.mp4 \
    --point 0.5,0.5,1 \
    --out ../tests/fixtures/video_kids/sam3_1/seed_mask.png 2>&1
```

Expected: `wrote seed mask 1008x1008 to .../seed_mask.png` (or
whatever the native mask size is for the variant).

If the model file isn't available, print a note and stop here — the
SAM 3.1 parity test remains in SKIP until someone with the
checkpoint completes steps 2–4. Proceed to Task 8.

- [ ] **Step 3: Regenerate Python propagation PNGs**

```bash
cd /Users/rbisri/Documents/sam3 && \
  SAM3_CKPT=/path/to/sam3.1_multiplex.pt \
  SAM3_BPE=/path/to/bpe_simple_vocab_16e6.txt.gz \
    python tools/gen_video_parity_fixtures.py \
      --variant sam3.1 \
      --video assets/kids.mp4 \
      --frames 3 \
      --seed-mask tests/fixtures/video_kids/sam3_1/seed_mask.png \
      --out tests/fixtures/video_kids/sam3_1/ 2>&1 | tail -20
```

Expected: `wrote 3 frames to .../frames/` and
`frame_0001_obj_1.png .. frame_0003_obj_1.png` on disk.

- [ ] **Step 4: Run the parity test (should pass)**

```bash
cd /Users/rbisri/Documents/sam3/build && \
  ctest -R test_video_parity_kids --output-on-failure 2>&1 | tail -30
```

Expected: PASS, with stderr showing frame-by-frame IoU lines (all
≥ 0.75).

- [ ] **Step 5: Regression-check by reverting one B1–B6 commit locally**

Verify the test *fails* when memory data-flow regresses. Example
(revert a single line of commit 97c6148 locally — do NOT commit):

```bash
cd /Users/rbisri/Documents/sam3 && \
  git stash || true
# Edit src/model/sam3_video.c:1030-1043 to remove the ch 16
# conditioning indicator (set to 0.0 on cond frames instead of 1.0).
# Rebuild + run:
cd build && cmake --build . -j8 && \
  ctest -R test_video_parity_kids --output-on-failure 2>&1 | tail -30
```

Expected: test FAILS with "frame N IoU < 0.75 — FAIL".

Then restore:

```bash
git checkout src/model/sam3_video.c
cd build && cmake --build . -j8
```

- [ ] **Step 6: Update top-level fixture README pointer**

In `tests/fixtures/video_kids/README.md`, add at the bottom:

```markdown
## Variants

- SAM 3 (this directory's `frames/`): see regen steps above. Still
  scaffolded — the C parity test is a stub pending a PNG loader and
  blessed reference outputs.
- SAM 3.1: see `sam3_1/README.md`. C-seeded, Python-propagated;
  fully wired.
```

- [ ] **Step 7: Commit fixtures and README changes**

```bash
git add tests/fixtures/video_kids/sam3_1/ \
        tests/fixtures/video_kids/README.md
git commit -m "$(cat <<'EOF'
tests: SAM 3.1 parity fixtures for kids.mp4

C-seeded frame-0 mask + Python propagation for frames 1..3 via
Sam3MultiplexTracking.add_new_mask. Consumed by the SAM 3.1 variant
of test_video_parity_kids.
EOF
)"
```

If fixture regeneration is blocked (no model/checkpoint available),
commit the README only and note in the commit message that
`seed_mask.png` + `frames/*.png` are pending.

---

## Task 8 — Update `TODO.md`

**Files:**
- Modify: `TODO.md`

- [ ] **Step 1: Strike the two closed items**

In `TODO.md`, Phase 2.5b section:

Change the `[~] **Strengthen \`tests/test_sam3_1_track.c\`.**` item
to `[x] ...` and replace the bulleted option list with:

```markdown
- [x] **Strengthen `tests/test_sam3_1_track.c`.**
  Per-frame mixed-sign-logits and foreground-fraction invariants
  (from commit 97c6148) catch gross collapse. A dedicated
  reference-IoU fixture now lives in
  `tests/fixtures/video_kids/sam3_1/` and is consumed by the
  `SAM3_PARITY_VARIANT=sam3_1` build of
  `tests/test_video_parity_kids.c`: C frame-0 seeds Python via
  `Sam3MultiplexTracking.add_new_mask`, propagation frames 1..3 are
  committed PNGs, C asserts per-frame IoU ≥ 0.75. See
  `docs/superpowers/specs/2026-04-20-sam3-1-parity-fixture-design.md`.
```

Change the `[ ] **Minor: dead \`tgt_pos\` branch in memory-attn.**`
item to `[x] ...` and replace the body with:

```markdown
- [x] **Minor: dead `tgt_pos` branch in memory-attn.**
  Deleted. The `tgt_pos` parameter no longer exists on
  `sam3_multiplex_memory_attn_forward`; the only caller (which
  passed NULL) was updated in the same commit.
```

- [ ] **Step 2: Update the Phase 2.5b header summary**

Change the Phase 2.5b header from:

```markdown
### Phase 2.5b — per-frame data-flow gaps — **DONE (B1–B6 + I1)**
```

to:

```markdown
### Phase 2.5b — per-frame data-flow gaps — **DONE**
```

And update the section preamble to remove the "remaining" paragraph
(the IoU fixture and tgt_pos cleanup are now shipped).

- [ ] **Step 3: Commit**

```bash
git add TODO.md
git commit -m "$(cat <<'EOF'
TODO: close Phase 2.5b IoU fixture + tgt_pos cleanup

Phase 2.5b is now fully done: data-flow gaps B1-B6 + I1, the IoU
fixture test, and the dead tgt_pos branch deletion.
EOF
)"
```

---

## Self-Review Checklist

- **Spec coverage.** Each spec section maps to tasks:
  - §3.1 file changes → Tasks 1–7
  - §3.2 fixture layout → Task 7 (README + regen)
  - §4.1 regen pipeline → Task 7 steps 2–3
  - §4.2 CI test flow → Task 6 step 2 (main body)
  - §5.1 `sam3_1_dump_seed.c` → Task 4
  - §5.2 gen_video_parity_fixtures flags → Task 5
  - §5.3 PNG helpers → Task 3
  - §5.4 test variant dispatch → Task 6 steps 1–2
  - §6 error handling (skip, seed drift, CPU-only) → Tasks 5 (CPU
    patches) + 6 (skip + tiered IoU check)
  - §7 testing validation → Task 7 steps 4–5
  - Deliverables checklist (§10) → Tasks 1–8
- **Placeholder scan.** No "TBD"/"TODO" in task bodies. Task 7 step 2
  has an explicit "stop here if model absent" escape hatch, not a
  placeholder.
- **Type consistency.** All functions used in later tasks are defined
  in earlier tasks:
  - `load_png_grayscale` / `save_png_grayscale` defined Task 3, used
    Tasks 4 (indirectly via stbi_write_png in the tool itself) + 6.
  - `sam3_1_dump_seed` binary built Task 4, invoked Task 7 step 2.
  - `gen_video_parity_fixtures.py --variant sam3.1` added Task 5,
    invoked Task 7 step 3.
  - `install_triton_stub` / `install_cuda_redirect` /
    `install_addmm_act_fp32` defined Task 2, consumed Task 5.
- **Upstream API name drift (Task 5 step 2).** The plan explicitly
  requires verifying `build_sam3_multiplex_tracker` and `add_new_mask`
  kwargs at implementation time rather than assuming their exact
  names — this is a known unknown documented as a verification step,
  not a placeholder.
