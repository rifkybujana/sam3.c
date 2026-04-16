# Rust Binding Design

**Date:** 2026-04-16
**Status:** Approved, pending implementation plan
**Related:** `2026-04-15-python-bindings-design.md`

## Goal

Provide safe, idiomatic Rust bindings to the SAM3 inference engine, structured
so they can be developed in-repo and eventually published to crates.io with
minimal changes.

## Non-Goals

- Porting the C CLI (`sam3_cli`) to Rust.
- Exposing internal headers (`include/sam3/internal/*.h`) in v1.
- `ndarray` or `image` crate integration in v1 — plain `Vec<f32>` / `&[f32]`
  with explicit shape fields is the baseline.
- Windows support beyond best-effort (libsam3 itself is Unix-focused; paths
  with non-UTF-8 bytes return `Error::Invalid`).
- CTest integration for Rust tests.

## Key Decisions

| Decision | Choice |
|---|---|
| Audience | Both — in-repo dev now, crates.io-ready layout |
| Linking | Dynamic (`libsam3.dylib` / `libsam3.so`) |
| FFI generation | `bindgen` at build time |
| API level | Idiomatic Rust, mirrors the Python binding |
| Array types | Plain `Vec<f32>` / `&[f32]` with shape fields |
| Thread safety | `!Send + !Sync` on `Ctx` |
| CLI scope | Library only (no bundled binary) |
| MSRV | 1.75, `edition = 2021` |
| Extras | Move `python/` to `bindings/python/` in the same change |

## Repository Layout

```
bindings/
├── python/                  # moved from top-level python/
│   ├── setup.py
│   ├── pyproject.toml
│   ├── sam3/
│   ├── tests/
│   └── exports.txt
└── rust/
    ├── Cargo.toml           # workspace root
    ├── README.md
    ├── rust-toolchain.toml
    ├── rustfmt.toml
    ├── sam3-sys/
    │   ├── Cargo.toml
    │   ├── build.rs
    │   ├── wrapper.h
    │   └── src/lib.rs
    └── sam3/
        ├── Cargo.toml
        ├── README.md
        ├── src/
        │   ├── lib.rs
        │   ├── error.rs
        │   ├── ctx.rs
        │   ├── prompt.rs
        │   ├── result.rs
        │   ├── image.rs
        │   └── log.rs
        ├── examples/segment.rs
        └── tests/integration.rs
```

### Python move follow-ups

- `bindings/python/setup.py`: `CMakeBuild` resolves the repo root from
  `setup.py`'s location — the `..` hop becomes `../..`.
- Top-level `README.md`: update any `cd python && pip install -e .` snippets.
- `.gitignore`: update `python/build`, `python/dist`, `python/*.egg-info`.
- `docs/`: grep for `python/` path references, update.
- CI (`.github/workflows/*.yml`): update `working-directory: python` entries.
- Use `git mv` to preserve history.

## `sam3-sys` (raw FFI)

### `Cargo.toml`

```toml
[package]
name         = "sam3-sys"
version      = "0.1.0"
edition      = "2021"
rust-version = "1.75"
license      = "MIT"
description  = "Raw FFI bindings to libsam3"
links        = "sam3"
build        = "build.rs"

[build-dependencies]
bindgen = "0.70"

[lib]
doctest = false
```

`links = "sam3"` prevents duplicate native linking if multiple crates
transitively depend on us.

### `wrapper.h`

```c
#include <sam3/sam3.h>
```

Transitively pulls in `sam3_types.h`. Internal headers are intentionally
excluded from v1; they can be added behind a feature later.

### `build.rs`

Probe order for locating libsam3 (first hit wins):

1. `SAM3_LIB_DIR` + `SAM3_INCLUDE_DIR` env vars — explicit override.
2. `SAM3_BUILD_DIR` env var → `libsam3.{dylib,so}` inside; headers inferred
   from `../include`.
3. Auto-detect: walk up from `CARGO_MANIFEST_DIR` to the repo root, check
   for `build/libsam3.{dylib,so}` and `include/sam3/sam3.h`.
4. Emit actionable error listing the env vars above.

Bindgen configuration:

- `allowlist_function("sam3_.*")`, `allowlist_type("sam3_.*")`,
  `allowlist_var("SAM3_.*")`.
- `rustified_enum` for `sam3_error`, `sam3_log_level`, `sam3_dtype`,
  `sam3_prompt_type`, `sam3_backbone_type`.
- `derive_default(true)`, `derive_debug(true)` on structs.
- `layout_tests(true)` — compile-time struct-layout verification.

Emitted directives:

- `cargo:rustc-link-search=native={lib_dir}`
- `cargo:rustc-link-lib=dylib=sam3`
- `cargo:rustc-link-arg=-Wl,-rpath,@loader_path` (macOS)
- `cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN` (Linux)
- `cargo:rerun-if-changed=wrapper.h`
- `cargo:rerun-if-env-changed=SAM3_LIB_DIR`, `SAM3_BUILD_DIR`,
  `SAM3_INCLUDE_DIR`

### `src/lib.rs`

```rust
#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case)]
#![allow(deref_nullptr, clippy::missing_safety_doc)]
#![doc = "Raw FFI bindings to libsam3. See the `sam3` crate for a safe API."]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
```

## `sam3` (safe API)

### `Cargo.toml`

```toml
[package]
name         = "sam3"
version      = "0.1.0"
edition      = "2021"
rust-version = "1.75"
license      = "MIT"
description  = "Safe Rust bindings to the SAM3 inference engine"

[dependencies]
sam3-sys  = { version = "0.1.0", path = "../sam3-sys" }
thiserror = "1"
```

### `error.rs`

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid argument")]                    Invalid,
    #[error("out of memory")]                       NoMemory,
    #[error("I/O error")]                           Io,
    #[error("backend initialization failed")]       Backend,
    #[error("model format error")]                  Model,
    #[error("unsupported or mismatched dtype")]     Dtype,
    #[error("unknown SAM3 error ({0})")]            Unknown(i32),
}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn check(code: sam3_sys::sam3_error) -> Result<()> { /* ... */ }
```

Mirrors `python/sam3/errors.py`. Every `SAM3_E*` code maps to a distinct
variant; `Unknown(i32)` is a future-proofing fallback.

### `ctx.rs`

```rust
pub struct Ctx {
    raw: NonNull<sys::sam3_ctx>,
    _not_send_sync: PhantomData<*mut ()>, // !Send + !Sync
}

impl Ctx {
    pub fn new() -> Result<Self>;
    pub fn load_model<P: AsRef<Path>>(&mut self, path: P) -> Result<()>;
    pub fn load_bpe<P: AsRef<Path>>(&mut self, path: P) -> Result<()>;
    pub fn set_image(&mut self, img: &ImageData<'_>) -> Result<()>;
    pub fn set_image_rgb(&mut self, pixels: &[u8], width: u32, height: u32) -> Result<()>;
    pub fn set_image_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()>;
    pub fn set_prompt_space(&mut self, width: u32, height: u32);
    pub fn set_text(&mut self, text: &str) -> Result<()>;
    pub fn segment(&mut self, prompts: &[Prompt<'_>]) -> Result<SegmentResult>;
    pub fn image_size(&self) -> u32;
}

impl Drop for Ctx { /* sam3_free */ }
```

- `PhantomData<*mut ()>` enforces `!Send + !Sync`.
- Every mutating call takes `&mut self`; the borrow checker prevents
  aliased misuse of the internally mutable arenas and async worker.
- Paths convert to `CString` via `OsStr::as_bytes()` on Unix. Non-UTF-8
  paths → `Error::Invalid`.

### `prompt.rs`

```rust
#[derive(Debug, Clone, Copy)]
pub struct Point { pub x: f32, pub y: f32, pub label: PointLabel }

#[derive(Debug, Clone, Copy)]
pub enum PointLabel { Foreground, Background }

#[derive(Debug, Clone, Copy)]
pub struct Box { pub x1: f32, pub y1: f32, pub x2: f32, pub y2: f32 }

pub struct MaskPrompt<'a> {
    pub data:   &'a [f32],
    pub width:  u32,
    pub height: u32,
}

pub enum Prompt<'a> {
    Point(Point),
    Box(Box),
    Mask(MaskPrompt<'a>),
    Text(&'a str),
}
```

The `'a` lifetime ties mask slices and text borrows to caller data, so
`segment()` cannot outlive them. Lowering to `sys::sam3_prompt` happens
inside `segment()`; `CString`s for text prompts live in a scratch
`Vec<CString>` for the duration of the FFI call.

### `result.rs`

```rust
pub struct SegmentResult {
    masks:       Vec<f32>,       // len = n_masks * mask_height * mask_width
    iou_scores:  Vec<f32>,       // len = n_masks
    boxes:       Option<Vec<[f32; 4]>>,
    n_masks:     usize,
    mask_height: usize,
    mask_width:  usize,
    iou_valid:   bool,
    best_mask:   Option<usize>,
}
```

Accessors: `n_masks`, `mask_height`, `mask_width`, `iou_valid`, `iou_scores`,
`boxes`, `best_mask`, `masks`, `mask(i)`, `best()`.

Construction inside `Ctx::segment` copies all C-side buffers into owned
`Vec`s (same rationale as the Python binding — the next `segment()` call
reuses arenas), then runs `sam3_result_free` via an RAII guard to survive
panics during copy. Shape multiplications are checked for overflow.

### `image.rs`

```rust
pub struct ImageData<'a> {
    pub pixels: &'a [u8],
    pub width:  u32,
    pub height: u32,
}
```

Thin helper; no `image` crate dep.

### `log.rs`

```rust
pub enum LogLevel { Debug, Info, Warn, Error }

pub fn set_log_level(level: LogLevel);
pub fn version() -> &'static str;
```

### `lib.rs`

Public surface re-exports `Ctx`, `Error`, `Result`, `ImageData`,
`LogLevel`, `set_log_level`, `version`, `Box`, `MaskPrompt`, `Point`,
`PointLabel`, `Prompt`, `SegmentResult`. `sam3-sys` is not re-exported.

## Build Integration

### Rust-side

The `sam3-sys` build script auto-detects a sibling `build/` directory so
the in-repo flow is:

```sh
cmake -S . -B build -DSAM3_SHARED=ON
cmake --build build -j
cd bindings/rust
cargo build
```

### Optional CMake target

Add `-DSAM3_RUST=ON` (default OFF) to introduce a `sam3_rust` CMake target
that invokes `cargo build` with `SAM3_LIB_DIR` and `SAM3_INCLUDE_DIR`
pointing at the current build tree. Requires `SAM3_SHARED=ON`. Non-blocking
for existing CMake users.

### Runtime library resolution

| Scenario | How |
|---|---|
| In-repo dev | `DYLD_LIBRARY_PATH=build cargo test` (macOS) / `LD_LIBRARY_PATH=build cargo test` (Linux) |
| System install | `cmake --install build` puts libsam3 on the default loader path |
| Distribution | Ship `libsam3.{dylib,so}` next to the Rust binary; rpath (`@loader_path` / `$ORIGIN`) handles the rest |

## Testing

1. **`sam3-sys` sanity** — `#[test]` calls `sam3_version()` through raw FFI,
   asserts non-null. Verifies bindgen linkage without a model.
2. **`sam3` unit tests** — colocated `#[cfg(test)] mod tests` per module:
   - `error.rs`: every `sam3_error` variant round-trips through `check()`.
   - `prompt.rs`: lowering produces the expected union tag and field values.
   - `result.rs`: construction handles `n_masks == 0`, `boxes_valid == 0`,
     `best_mask == -1`.
   - `log.rs`: `version()` returns a non-empty string.
3. **`sam3/tests/integration.rs`** — end-to-end tests gated by env vars,
   matching `python/tests/conftest.py`: `SAM3_MODEL_PATH`,
   `SAM3_TEST_IMAGE`, optional `SAM3_BPE_PATH`. Skip (not fail) when unset.
   Scenarios: load → set image → segment with point / text / box; multiple
   contexts sequentially.

## Dev Workflow

- `bindings/rust/rustfmt.toml`: `edition = "2021"`, `max_width = 100`,
  `tab_spaces = 4`. Rust lives in its own subtree so the Rust-idiomatic
  4-space indent doesn't conflict with the 8-space-tab C style.
- Lints: `#![warn(missing_docs)]` on `sam3`;
  `#![deny(unsafe_op_in_unsafe_fn)]` — every FFI call is explicitly
  `unsafe { ... }` even inside `unsafe fn`s.
- `sam3-sys` suppresses the standard bindgen lint noise.

## Open Questions

None blocking implementation. Items worth revisiting post-v1:

- Whether to expose `include/sam3/internal/*` behind a feature flag.
- Whether to add `ndarray`/`image` integration behind features.
- A Rust-side CLI binary once the library is stable.

## Copyright

Copyright (c) 2026 Rifky Bujana Bisri
SPDX-License-Identifier: MIT
