# Rust Binding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create safe, idiomatic Rust bindings to libsam3 as a Cargo workspace at `bindings/rust/` with a `sam3-sys` (bindgen FFI) + `sam3` (safe API) split. Also consolidate the existing Python binding under `bindings/python/`.

**Architecture:** `sam3-sys` runs `bindgen` at build time against `include/sam3/sam3.h`, locates and dynamically links `libsam3.{dylib,so}`, and emits `cargo:rustc-link-arg` rpaths for distribution. `sam3` wraps the raw FFI with RAII `Ctx`, typed `Prompt` enum, owned `SegmentResult`, `thiserror`-backed `Error`. `Ctx` is `!Send + !Sync` via `PhantomData<*mut ()>`. All mutation takes `&mut self`.

**Tech Stack:** Rust 1.75+ (edition 2021), `bindgen` 0.70, `thiserror` 1, dynamic linking to libsam3 built with `-DSAM3_SHARED=ON`.

**Reference:** `docs/plans/2026-04-16-rust-binding-design.md`

---

## Phase 1 — Move Python binding to `bindings/python/`

Do this first so the new `bindings/` directory convention is in place before adding Rust.

### Task 1.1: Move `python/` to `bindings/python/` with `git mv`

**Files:**
- Move: `python/` → `bindings/python/`

**Step 1: Create `bindings/` and move**

```bash
mkdir -p bindings
git mv python bindings/python
```

**Step 2: Verify nothing else references the old path**

Run: `grep -rn "python/" CMakeLists.txt README.md .github/ .gitignore docs/ 2>/dev/null | grep -v "docs/plans/"`
Expected: only `CMakeLists.txt:126` matches (we fix that next).

**Step 3: Commit the move**

```bash
git add -A
git commit -m "python: move bindings to bindings/python"
```

---

### Task 1.2: Update `CMakeLists.txt` exports path

**Files:**
- Modify: `CMakeLists.txt:126`

**Step 1: Update the path**

Replace `${CMAKE_SOURCE_DIR}/python/exports.txt` with `${CMAKE_SOURCE_DIR}/bindings/python/exports.txt`.

Use Edit with `old_string`:
```
LINK_FLAGS "-exported_symbols_list ${CMAKE_SOURCE_DIR}/python/exports.txt"
```
and `new_string`:
```
LINK_FLAGS "-exported_symbols_list ${CMAKE_SOURCE_DIR}/bindings/python/exports.txt"
```

**Step 2: Rebuild with shared library on to confirm the path works**

Run:
```bash
cmake -S . -B build -DSAM3_SHARED=ON -DCMAKE_BUILD_TYPE=Debug 2>&1 | tail -5
cmake --build build -j 2>&1 | tail -10
```
Expected: Clean build. `build/libsam3.dylib` (or `.so` on Linux) exists.

Run: `ls build/libsam3.*`
Expected: `build/libsam3.dylib` (macOS) or `build/libsam3.so` (Linux) present.

**Step 3: Run existing tests**

Run: `cd build && ctest --output-on-failure -j 2>&1 | tail -15`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add CMakeLists.txt
git commit -m "cmake: update exports.txt path to bindings/python"
```

---

### Task 1.3: Update `bindings/python/setup.py` root computation

The old `setup.py` computes `ROOT = dirname(dirname(abspath(__file__)))` — one hop up. After the move, `setup.py` lives two directories deep, so it needs two hops.

**Files:**
- Modify: `bindings/python/setup.py:14`

**Step 1: Update ROOT computation**

Use Edit to change:
```python
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
```
to:
```python
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

**Step 2: Verify by running a dry-run install**

Run:
```bash
cd bindings/python && python -c "
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath('setup.py'))))
print(ROOT)
assert os.path.exists(os.path.join(ROOT, 'CMakeLists.txt')), ROOT
print('OK')
"
```
Expected: prints repo root path, then `OK`.

**Step 3: Commit**

```bash
git add bindings/python/setup.py
git commit -m "python: fix ROOT path after move to bindings/python"
```

---

### Task 1.4: Update README and docs

**Files:**
- Modify: `README.md` (any `cd python` references)
- Modify: `docs/plans/2026-04-15-python-bindings-design.md` (mention of `python/` in prose)

**Step 1: Find references**

Run: `grep -n "python/" README.md`
Expected: Listing of lines to update (or empty if none).

**Step 2: Update README if any found**

For each match, replace `python/` with `bindings/python/` and `cd python` with `cd bindings/python`.

**Step 3: Leave historical plan docs alone**

`docs/plans/2026-04-15-python-bindings*.md` describe the original implementation; do not edit — they are historical records. Only README gets updated.

**Step 4: Commit if there were changes**

```bash
git add README.md
git commit -m "docs: update README to reference bindings/python"
```

(If there were no README matches, skip this task.)

---

## Phase 2 — `sam3-sys` crate

### Task 2.1: Create workspace skeleton

**Files:**
- Create: `bindings/rust/Cargo.toml` (workspace root)
- Create: `bindings/rust/rust-toolchain.toml`
- Create: `bindings/rust/rustfmt.toml`
- Create: `bindings/rust/README.md`
- Create: `bindings/rust/.gitignore`

**Step 1: Create workspace `Cargo.toml`**

Write `bindings/rust/Cargo.toml`:

```toml
[workspace]
resolver = "2"
members = ["sam3-sys", "sam3"]

[workspace.package]
version      = "0.1.0"
edition      = "2021"
rust-version = "1.75"
license      = "MIT"
repository   = "https://github.com/facebookresearch/sam3"
authors      = ["Rifky Bujana Bisri"]

[workspace.lints.rust]
unsafe_op_in_unsafe_fn = "deny"
```

**Step 2: Create `bindings/rust/rust-toolchain.toml`**

```toml
[toolchain]
channel = "1.75"
components = ["rustfmt", "clippy"]
```

**Step 3: Create `bindings/rust/rustfmt.toml`**

```toml
edition = "2021"
max_width = 100
tab_spaces = 4
```

**Step 4: Create `bindings/rust/.gitignore`**

```gitignore
target/
Cargo.lock
```

Note: We gitignore `Cargo.lock` because this is a library workspace (not a binary distribution). The `sam3-sys` binding has minimal deps and reproducibility comes from pinning `bindgen` in `Cargo.toml`.

**Step 5: Create `bindings/rust/README.md`**

```markdown
# SAM3 Rust bindings

Safe Rust bindings to the [SAM3](../../README.md) inference engine.

## Crates

- [`sam3-sys`](./sam3-sys) — raw FFI (bindgen output, native linking)
- [`sam3`](./sam3) — safe, idiomatic wrapper

## Quickstart

Build libsam3 as a shared library from the repo root:

```sh
cmake -S . -B build -DSAM3_SHARED=ON
cmake --build build -j
```

Then build and test the Rust binding:

```sh
cd bindings/rust

# macOS
DYLD_LIBRARY_PATH=../../build cargo test

# Linux
LD_LIBRARY_PATH=../../build cargo test
```

## Runtime library resolution

The binding links dynamically to `libsam3.{dylib,so}`. At runtime, the loader
must find it. Three supported workflows:

| Scenario       | How                                                                |
|----------------|--------------------------------------------------------------------|
| In-repo dev    | `DYLD_LIBRARY_PATH` / `LD_LIBRARY_PATH` pointed at `build/`        |
| System install | `cmake --install build` puts `libsam3` on the default loader path  |
| Distribution   | Ship `libsam3.{dylib,so}` next to the Rust binary; rpath finds it  |

## Environment overrides

- `SAM3_LIB_DIR` / `SAM3_INCLUDE_DIR` — explicit library and header directories
- `SAM3_BUILD_DIR` — CMake build dir; headers inferred from `../include`

If none are set, `sam3-sys` auto-detects a sibling `build/` under the repo root.
```

**Step 6: Verify layout**

Run: `ls bindings/rust/`
Expected: `Cargo.toml`, `rust-toolchain.toml`, `rustfmt.toml`, `.gitignore`, `README.md`

**Step 7: Commit**

```bash
git add bindings/rust
git commit -m "rust: add Cargo workspace skeleton for bindings"
```

---

### Task 2.2: Create `sam3-sys` crate structure

**Files:**
- Create: `bindings/rust/sam3-sys/Cargo.toml`
- Create: `bindings/rust/sam3-sys/wrapper.h`
- Create: `bindings/rust/sam3-sys/src/lib.rs`

**Step 1: Create `bindings/rust/sam3-sys/Cargo.toml`**

```toml
[package]
name         = "sam3-sys"
version.workspace      = true
edition.workspace      = true
rust-version.workspace = true
license.workspace      = true
repository.workspace   = true
authors.workspace      = true
description  = "Raw FFI bindings to libsam3"
links        = "sam3"
build        = "build.rs"

[lib]
doctest = false

[build-dependencies]
bindgen = "0.70"
```

**Step 2: Create `bindings/rust/sam3-sys/wrapper.h`**

```c
#include <sam3/sam3.h>
```

**Step 3: Create `bindings/rust/sam3-sys/src/lib.rs`**

```rust
//! Raw FFI bindings to libsam3.
//!
//! This crate exposes the unmodified C API of libsam3 generated by
//! [`bindgen`]. For a safe, idiomatic wrapper, see the [`sam3`] crate.

#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case)]
#![allow(deref_nullptr)]
#![allow(clippy::missing_safety_doc)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
```

**Step 4: Verify files exist**

Run: `ls bindings/rust/sam3-sys/ bindings/rust/sam3-sys/src/`
Expected: `Cargo.toml`, `wrapper.h`, `src/lib.rs`.

**Step 5: Commit**

```bash
git add bindings/rust/sam3-sys
git commit -m "rust/sys: add sam3-sys crate skeleton"
```

(Does not build yet — `build.rs` comes next.)

---

### Task 2.3: Write `sam3-sys/build.rs`

**Files:**
- Create: `bindings/rust/sam3-sys/build.rs`

**Step 1: Write the build script**

```rust
//! Build script for `sam3-sys`.
//!
//! Locates `libsam3` (installed or from a sibling CMake build directory),
//! runs `bindgen` against `sam3/sam3.h`, and emits native-link directives.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed=SAM3_LIB_DIR");
    println!("cargo:rerun-if-env-changed=SAM3_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=SAM3_INCLUDE_DIR");

    let (lib_dir, include_dir) = resolve_paths();

    println!(
        "cargo:rustc-link-search=native={}",
        lib_dir.display()
    );
    println!("cargo:rustc-link-lib=dylib=sam3");

    // Bake an rpath so binaries find a co-located libsam3 after install.
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
    } else if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    }

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");

    bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", include_dir.display()))
        .allowlist_function("sam3_.*")
        .allowlist_type("sam3_.*")
        .allowlist_var("SAM3_.*")
        .rustified_enum("sam3_error")
        .rustified_enum("sam3_log_level")
        .rustified_enum("sam3_dtype")
        .rustified_enum("sam3_prompt_type")
        .rustified_enum("sam3_backbone_type")
        .derive_default(true)
        .derive_debug(true)
        .layout_tests(true)
        .generate()
        .expect("bindgen failed to generate sam3 bindings")
        .write_to_file(&out_path)
        .expect("failed to write bindings.rs");
}

/// Resolve `(lib_dir, include_dir)` via env vars or auto-detection.
fn resolve_paths() -> (PathBuf, PathBuf) {
    // 1. Explicit override.
    if let (Ok(lib), Ok(inc)) = (env::var("SAM3_LIB_DIR"), env::var("SAM3_INCLUDE_DIR")) {
        return (PathBuf::from(lib), PathBuf::from(inc));
    }

    // 2. SAM3_BUILD_DIR with inferred include dir.
    if let Ok(build_dir) = env::var("SAM3_BUILD_DIR") {
        let build = PathBuf::from(&build_dir);
        let include = build.parent()
            .map(|p| p.join("include"))
            .unwrap_or_else(|| PathBuf::from("include"));
        if has_lib(&build) && include.join("sam3").join("sam3.h").is_file() {
            return (build, include);
        }
    }

    // 3. Auto-detect a sibling build/ by walking up from CARGO_MANIFEST_DIR.
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let mut cur: &Path = &manifest;
    loop {
        let candidate_build = cur.join("build");
        let candidate_include = cur.join("include");
        if has_lib(&candidate_build) && candidate_include.join("sam3").join("sam3.h").is_file() {
            return (candidate_build, candidate_include);
        }
        match cur.parent() {
            Some(p) => cur = p,
            None => break,
        }
    }

    panic!(
        "sam3-sys: unable to locate libsam3. Set SAM3_LIB_DIR and SAM3_INCLUDE_DIR, \
         or SAM3_BUILD_DIR, or ensure `build/libsam3.{{dylib,so}}` exists under the \
         repository root (run `cmake -S . -B build -DSAM3_SHARED=ON && cmake --build build`)."
    );
}

fn has_lib(dir: &Path) -> bool {
    dir.join("libsam3.dylib").is_file()
        || dir.join("libsam3.so").is_file()
        || dir.join("libsam3.dll").is_file()
}
```

**Step 2: Verify the host has libsam3 shared built**

Run: `ls build/libsam3.*`
Expected: `build/libsam3.dylib` (macOS) — built in Task 1.2.

**Step 3: Build `sam3-sys`**

Run: `cd bindings/rust && cargo build -p sam3-sys 2>&1 | tail -20`
Expected: Clean build. Bindgen generates `target/debug/build/sam3-sys-*/out/bindings.rs`.

**Step 4: Inspect generated bindings to spot-check**

Run:
```bash
find bindings/rust/target -name bindings.rs -path '*sam3-sys*' | head -1 | xargs grep -l "sam3_init"
```
Expected: One file path printed (confirms `sam3_init` extern was generated).

**Step 5: Commit**

```bash
git add bindings/rust/sam3-sys/build.rs
git commit -m "rust/sys: add build.rs with bindgen and library resolution"
```

---

### Task 2.4: Add `sam3-sys` sanity test

**Files:**
- Create: `bindings/rust/sam3-sys/tests/version.rs`

**Step 1: Write the failing test**

```rust
//! Verify that `sam3-sys` links and the simplest FFI call works.

#[test]
fn version_returns_non_null_c_string() {
    // SAFETY: sam3_version returns a pointer to a static C string.
    let ptr = unsafe { sam3_sys::sam3_version() };
    assert!(!ptr.is_null(), "sam3_version returned NULL");
    let s = unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_str()
        .expect("version must be valid UTF-8");
    assert!(!s.is_empty(), "sam3_version returned empty string");
}
```

**Step 2: Run the test**

Run:
```bash
cd bindings/rust
DYLD_LIBRARY_PATH=../../build cargo test -p sam3-sys --test version 2>&1 | tail -10
```
(Linux: `LD_LIBRARY_PATH=../../build` instead.)
Expected: `test version_returns_non_null_c_string ... ok`.

**Step 3: Commit**

```bash
git add bindings/rust/sam3-sys/tests/version.rs
git commit -m "rust/sys: add link-verification test"
```

---

## Phase 3 — `sam3` safe API

All remaining tasks follow TDD: write the failing test, watch it fail, implement, verify, commit.

### Task 3.1: Create `sam3` crate skeleton

**Files:**
- Create: `bindings/rust/sam3/Cargo.toml`
- Create: `bindings/rust/sam3/src/lib.rs`
- Create: `bindings/rust/sam3/README.md`

**Step 1: Write `bindings/rust/sam3/Cargo.toml`**

```toml
[package]
name         = "sam3"
version.workspace      = true
edition.workspace      = true
rust-version.workspace = true
license.workspace      = true
repository.workspace   = true
authors.workspace      = true
description  = "Safe Rust bindings to the SAM3 inference engine"

[dependencies]
sam3-sys  = { version = "0.1.0", path = "../sam3-sys" }
thiserror = "1"

[lints]
workspace = true
```

**Step 2: Write placeholder `bindings/rust/sam3/src/lib.rs`**

```rust
//! Safe Rust bindings to the SAM3 inference engine.

#![warn(missing_docs)]
```

**Step 3: Write `bindings/rust/sam3/README.md`**

```markdown
# sam3

Safe Rust bindings to the [SAM3](https://github.com/facebookresearch/sam3)
inference engine.

See the [`bindings/rust/README.md`](../README.md) for build instructions.
```

**Step 4: Verify it builds**

Run: `cd bindings/rust && cargo build -p sam3 2>&1 | tail -5`
Expected: Clean build.

**Step 5: Commit**

```bash
git add bindings/rust/sam3
git commit -m "rust: add sam3 crate skeleton"
```

---

### Task 3.2: Implement `error.rs`

**Files:**
- Create: `bindings/rust/sam3/src/error.rs`
- Modify: `bindings/rust/sam3/src/lib.rs`

**Step 1: Write the failing test**

Append to (or start) `bindings/rust/sam3/src/error.rs`:

```rust
//! SAM3 error type and code conversion.

use sam3_sys::sam3_error as sys_err;

/// Errors returned by the SAM3 runtime.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Invalid argument passed to a SAM3 function.
    #[error("invalid argument")]
    Invalid,
    /// Allocation failure inside the SAM3 runtime.
    #[error("out of memory")]
    NoMemory,
    /// I/O error reading a weight file, image, or shader.
    #[error("I/O error")]
    Io,
    /// Backend (Metal/CPU) initialization failed.
    #[error("backend initialization failed")]
    Backend,
    /// Model file format error (wrong magic, unsupported version).
    #[error("model format error")]
    Model,
    /// Unsupported or mismatched tensor dtype.
    #[error("unsupported or mismatched dtype")]
    Dtype,
    /// Unrecognized error code (future-proofing).
    #[error("unknown SAM3 error ({0})")]
    Unknown(i32),
}

/// Convenience alias for `std::result::Result<T, sam3::Error>`.
pub type Result<T> = std::result::Result<T, Error>;

/// Convert a raw `sam3_error` code into a `Result`.
pub(crate) fn check(code: sys_err) -> Result<()> {
    match code {
        sys_err::SAM3_OK       => Ok(()),
        sys_err::SAM3_EINVAL   => Err(Error::Invalid),
        sys_err::SAM3_ENOMEM   => Err(Error::NoMemory),
        sys_err::SAM3_EIO      => Err(Error::Io),
        sys_err::SAM3_EBACKEND => Err(Error::Backend),
        sys_err::SAM3_EMODEL   => Err(Error::Model),
        sys_err::SAM3_EDTYPE   => Err(Error::Dtype),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ok_maps_to_ok() {
        assert!(check(sys_err::SAM3_OK).is_ok());
    }

    #[test]
    fn every_error_variant_is_distinguished() {
        assert!(matches!(check(sys_err::SAM3_EINVAL),   Err(Error::Invalid)));
        assert!(matches!(check(sys_err::SAM3_ENOMEM),   Err(Error::NoMemory)));
        assert!(matches!(check(sys_err::SAM3_EIO),      Err(Error::Io)));
        assert!(matches!(check(sys_err::SAM3_EBACKEND), Err(Error::Backend)));
        assert!(matches!(check(sys_err::SAM3_EMODEL),   Err(Error::Model)));
        assert!(matches!(check(sys_err::SAM3_EDTYPE),   Err(Error::Dtype)));
    }

    #[test]
    fn display_renders_human_message() {
        assert_eq!(format!("{}", Error::Invalid), "invalid argument");
        assert_eq!(format!("{}", Error::Unknown(42)), "unknown SAM3 error (42)");
    }
}
```

**Step 2: Wire it into `lib.rs`**

Replace `bindings/rust/sam3/src/lib.rs` contents:

```rust
//! Safe Rust bindings to the SAM3 inference engine.

#![warn(missing_docs)]

mod error;

pub use error::{Error, Result};
```

**Step 3: Run the test to confirm it passes**

Run: `cd bindings/rust && cargo test -p sam3 error:: 2>&1 | tail -10`
Expected: 3 tests pass.

**Step 4: Commit**

```bash
git add bindings/rust/sam3/src/error.rs bindings/rust/sam3/src/lib.rs
git commit -m "rust: add Error enum and check() helper"
```

---

### Task 3.3: Implement `log.rs` and `version()`

**Files:**
- Create: `bindings/rust/sam3/src/log.rs`
- Modify: `bindings/rust/sam3/src/lib.rs`

**Step 1: Write `bindings/rust/sam3/src/log.rs`**

```rust
//! Logging and runtime-info helpers.

use sam3_sys as sys;

/// Log severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    /// Detailed tracing (suppressed by default).
    Debug,
    /// Operational milestones (default).
    Info,
    /// Non-fatal issues.
    Warn,
    /// Failures affecting correctness.
    Error,
}

impl From<LogLevel> for sys::sam3_log_level {
    fn from(l: LogLevel) -> Self {
        match l {
            LogLevel::Debug => sys::sam3_log_level::SAM3_LOG_DEBUG,
            LogLevel::Info  => sys::sam3_log_level::SAM3_LOG_INFO,
            LogLevel::Warn  => sys::sam3_log_level::SAM3_LOG_WARN,
            LogLevel::Error => sys::sam3_log_level::SAM3_LOG_ERROR,
        }
    }
}

/// Set the process-global minimum log level.
///
/// Messages below `level` are suppressed. Default is [`LogLevel::Info`].
pub fn set_log_level(level: LogLevel) {
    // SAFETY: sam3_log_set_level takes a simple enum, has no preconditions.
    unsafe { sys::sam3_log_set_level(level.into()) }
}

/// Return the libsam3 version string.
///
/// The pointer has `'static` lifetime (it lives in the loaded library's
/// read-only data segment).
pub fn version() -> &'static str {
    // SAFETY: sam3_version returns a pointer to a static, NUL-terminated
    // ASCII string. Never returns NULL.
    unsafe {
        let c = sys::sam3_version();
        debug_assert!(!c.is_null());
        std::ffi::CStr::from_ptr(c)
            .to_str()
            .expect("sam3_version must be valid UTF-8")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_non_empty() {
        assert!(!version().is_empty());
    }

    #[test]
    fn set_log_level_does_not_panic() {
        // All four levels should be accepted without crashing.
        set_log_level(LogLevel::Debug);
        set_log_level(LogLevel::Info);
        set_log_level(LogLevel::Warn);
        set_log_level(LogLevel::Error);
    }
}
```

**Step 2: Wire into `lib.rs`**

Append to `bindings/rust/sam3/src/lib.rs`:

```rust
mod log;

pub use log::{set_log_level, version, LogLevel};
```

**Step 3: Run tests**

Run: `cd bindings/rust && DYLD_LIBRARY_PATH=../../build cargo test -p sam3 log:: 2>&1 | tail -10`
Expected: 2 tests pass.

**Step 4: Commit**

```bash
git add bindings/rust/sam3/src/log.rs bindings/rust/sam3/src/lib.rs
git commit -m "rust: add LogLevel, set_log_level, version"
```

---

### Task 3.4: Implement `prompt.rs` types (no lowering yet)

**Files:**
- Create: `bindings/rust/sam3/src/prompt.rs`
- Modify: `bindings/rust/sam3/src/lib.rs`

**Step 1: Write `bindings/rust/sam3/src/prompt.rs`**

```rust
//! Prompt types consumed by [`Ctx::segment`](crate::Ctx::segment).

/// Foreground vs. background label attached to a [`Point`] prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointLabel {
    /// Include the surrounding region in the output mask.
    Foreground,
    /// Exclude the surrounding region from the output mask.
    Background,
}

impl PointLabel {
    #[inline]
    pub(crate) fn to_raw(self) -> i32 {
        match self {
            PointLabel::Foreground => 1,
            PointLabel::Background => 0,
        }
    }
}

/// A (x, y, label) point prompt in the prompt-coordinate space.
#[derive(Debug, Clone, Copy)]
pub struct Point {
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
    /// Foreground / background label.
    pub label: PointLabel,
}

/// An axis-aligned bounding box prompt (xyxy).
#[derive(Debug, Clone, Copy)]
pub struct Box {
    /// Left edge.
    pub x1: f32,
    /// Top edge.
    pub y1: f32,
    /// Right edge (exclusive).
    pub x2: f32,
    /// Bottom edge (exclusive).
    pub y2: f32,
}

/// A dense `H*W` mask prompt borrowed from the caller.
#[derive(Debug, Clone, Copy)]
pub struct MaskPrompt<'a> {
    /// Row-major `H*W` f32 values.
    pub data: &'a [f32],
    /// Mask width in pixels.
    pub width: u32,
    /// Mask height in pixels.
    pub height: u32,
}

/// An input prompt passed to [`Ctx::segment`](crate::Ctx::segment).
#[derive(Debug, Clone, Copy)]
pub enum Prompt<'a> {
    /// Point prompt.
    Point(Point),
    /// Bounding-box prompt.
    Box(Box),
    /// Dense mask prompt.
    Mask(MaskPrompt<'a>),
    /// UTF-8 text prompt (requires a BPE vocab loaded via `Ctx::load_bpe`).
    Text(&'a str),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_label_raw_values_match_c_convention() {
        assert_eq!(PointLabel::Foreground.to_raw(), 1);
        assert_eq!(PointLabel::Background.to_raw(), 0);
    }
}
```

**Step 2: Wire into `lib.rs`**

Append:

```rust
mod prompt;

pub use prompt::{Box, MaskPrompt, Point, PointLabel, Prompt};
```

**Step 3: Run tests**

Run: `cd bindings/rust && cargo test -p sam3 prompt:: 2>&1 | tail -10`
Expected: 1 test passes.

**Step 4: Commit**

```bash
git add bindings/rust/sam3/src/prompt.rs bindings/rust/sam3/src/lib.rs
git commit -m "rust: add Prompt enum and associated types"
```

---

### Task 3.5: Implement `image.rs`

**Files:**
- Create: `bindings/rust/sam3/src/image.rs`
- Modify: `bindings/rust/sam3/src/lib.rs`

**Step 1: Write `bindings/rust/sam3/src/image.rs`**

```rust
//! Image data helpers.

/// A borrowed RGB image buffer with explicit dimensions.
///
/// Row-major interleaved RGB (`R, G, B, R, G, B, ...`). `pixels.len()`
/// must be exactly `width * height * 3`.
#[derive(Debug, Clone, Copy)]
pub struct ImageData<'a> {
    /// Raw pixel bytes.
    pub pixels: &'a [u8],
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl<'a> ImageData<'a> {
    /// Return the expected buffer length for `width * height * 3 bytes`.
    pub(crate) fn required_len(&self) -> Option<usize> {
        (self.width as usize)
            .checked_mul(self.height as usize)?
            .checked_mul(3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_len_computes_width_times_height_times_3() {
        let img = ImageData { pixels: &[], width: 4, height: 5 };
        assert_eq!(img.required_len(), Some(60));
    }

    #[test]
    fn required_len_overflow_returns_none() {
        let img = ImageData { pixels: &[], width: u32::MAX, height: u32::MAX };
        assert!(img.required_len().is_none());
    }
}
```

**Step 2: Wire into `lib.rs`**

Append:

```rust
mod image;

pub use image::ImageData;
```

**Step 3: Run tests**

Run: `cd bindings/rust && cargo test -p sam3 image:: 2>&1 | tail -5`
Expected: 2 tests pass.

**Step 4: Commit**

```bash
git add bindings/rust/sam3/src/image.rs bindings/rust/sam3/src/lib.rs
git commit -m "rust: add ImageData borrowed RGB helper"
```

---

### Task 3.6: Implement `result.rs` (pure Rust, no FFI yet)

Build and test the `SegmentResult` accessors before wiring construction through `Ctx::segment`.

**Files:**
- Create: `bindings/rust/sam3/src/result.rs`
- Modify: `bindings/rust/sam3/src/lib.rs`

**Step 1: Write `bindings/rust/sam3/src/result.rs`**

```rust
//! Segmentation result returned by [`Ctx::segment`](crate::Ctx::segment).

/// A segmentation result: one or more mask logits plus scores.
///
/// All buffers are owned `Vec<f32>` — copied out of the SAM3 arenas before
/// the next segmentation call reuses them.
#[derive(Debug)]
pub struct SegmentResult {
    pub(crate) masks:       Vec<f32>,
    pub(crate) iou_scores:  Vec<f32>,
    pub(crate) boxes:       Option<Vec<[f32; 4]>>,
    pub(crate) n_masks:     usize,
    pub(crate) mask_height: usize,
    pub(crate) mask_width:  usize,
    pub(crate) iou_valid:   bool,
    pub(crate) best_mask:   Option<usize>,
}

impl SegmentResult {
    /// Number of masks returned.
    pub fn n_masks(&self) -> usize { self.n_masks }

    /// Mask height in pixels.
    pub fn mask_height(&self) -> usize { self.mask_height }

    /// Mask width in pixels.
    pub fn mask_width(&self) -> usize { self.mask_width }

    /// Whether [`iou_scores`](Self::iou_scores) are model-predicted (`true`)
    /// or placeholder zeros (`false`).
    pub fn iou_valid(&self) -> bool { self.iou_valid }

    /// Per-mask IoU scores; `len() == n_masks`.
    pub fn iou_scores(&self) -> &[f32] { &self.iou_scores }

    /// Per-mask axis-aligned bounding boxes (xyxy), when boxes were computed.
    pub fn boxes(&self) -> Option<&[[f32; 4]]> {
        self.boxes.as_deref()
    }

    /// Stability-selected mask index, if the model emitted one.
    pub fn best_mask(&self) -> Option<usize> { self.best_mask }

    /// Flat view of all masks, row-major.
    ///
    /// Index element `(m, y, x)` via
    /// `m * mask_height() * mask_width() + y * mask_width() + x`.
    pub fn masks(&self) -> &[f32] { &self.masks }

    /// Return the `i`-th mask as an `H*W` slice, or `None` if out of range.
    pub fn mask(&self, index: usize) -> Option<&[f32]> {
        let stride = self.mask_height * self.mask_width;
        let start = index.checked_mul(stride)?;
        let end = start.checked_add(stride)?;
        self.masks.get(start..end)
    }

    /// Return the stability-selected mask, if any.
    pub fn best(&self) -> Option<&[f32]> {
        self.best_mask.and_then(|i| self.mask(i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(n: usize, h: usize, w: usize, best: Option<usize>) -> SegmentResult {
        SegmentResult {
            masks: vec![0.0; n * h * w],
            iou_scores: vec![0.5; n],
            boxes: None,
            n_masks: n,
            mask_height: h,
            mask_width: w,
            iou_valid: true,
            best_mask: best,
        }
    }

    #[test]
    fn accessors_report_shape() {
        let r = synthetic(3, 4, 5, Some(1));
        assert_eq!(r.n_masks(), 3);
        assert_eq!(r.mask_height(), 4);
        assert_eq!(r.mask_width(), 5);
        assert!(r.iou_valid());
        assert_eq!(r.iou_scores().len(), 3);
        assert_eq!(r.masks().len(), 60);
    }

    #[test]
    fn mask_slice_is_h_times_w() {
        let r = synthetic(3, 4, 5, None);
        assert_eq!(r.mask(0).unwrap().len(), 20);
        assert_eq!(r.mask(2).unwrap().len(), 20);
        assert!(r.mask(3).is_none(), "out-of-range should be None");
    }

    #[test]
    fn best_returns_selected_mask() {
        let mut r = synthetic(2, 2, 2, Some(1));
        for v in r.masks[4..8].iter_mut() { *v = 9.0; }
        assert_eq!(r.best().unwrap(), &[9.0, 9.0, 9.0, 9.0]);
    }

    #[test]
    fn no_best_mask_is_none() {
        let r = synthetic(2, 2, 2, None);
        assert!(r.best().is_none());
    }

    #[test]
    fn zero_masks_is_valid() {
        let r = synthetic(0, 0, 0, None);
        assert_eq!(r.n_masks(), 0);
        assert!(r.masks().is_empty());
        assert!(r.mask(0).is_none());
    }
}
```

**Step 2: Wire into `lib.rs`**

Append:

```rust
mod result;

pub use result::SegmentResult;
```

**Step 3: Run tests**

Run: `cd bindings/rust && cargo test -p sam3 result:: 2>&1 | tail -10`
Expected: 5 tests pass.

**Step 4: Commit**

```bash
git add bindings/rust/sam3/src/result.rs bindings/rust/sam3/src/lib.rs
git commit -m "rust: add SegmentResult with accessor API"
```

---

### Task 3.7: Implement `Ctx::new` and `Drop`

Smallest possible step: just new/drop, then confirm it links and allocates/frees.

**Files:**
- Create: `bindings/rust/sam3/src/ctx.rs`
- Modify: `bindings/rust/sam3/src/lib.rs`

**Step 1: Write `bindings/rust/sam3/src/ctx.rs`**

```rust
//! SAM3 inference context.

use std::marker::PhantomData;
use std::ptr::NonNull;

use sam3_sys as sys;

use crate::error::{Error, Result};

/// An owned SAM3 inference context.
///
/// All state lives inside `Ctx`. Drop frees the context and joins any
/// background text-encoder worker.
///
/// `Ctx` is neither [`Send`] nor [`Sync`]: the context holds internal mutable
/// state and a worker thread, and libsam3 does not document cross-thread
/// safety. For concurrency, use separate contexts per thread.
pub struct Ctx {
    raw: NonNull<sys::sam3_ctx>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl Ctx {
    /// Create a new SAM3 context.
    pub fn new() -> Result<Self> {
        // SAFETY: sam3_init has no preconditions and returns NULL on failure.
        let raw = unsafe { sys::sam3_init() };
        NonNull::new(raw)
            .map(|raw| Ctx { raw, _not_send_sync: PhantomData })
            .ok_or(Error::NoMemory)
    }
}

impl Drop for Ctx {
    fn drop(&mut self) {
        // SAFETY: raw is a valid pointer obtained from sam3_init().
        // sam3_free joins the async worker and releases all resources.
        unsafe { sys::sam3_free(self.raw.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_succeeds_and_drop_does_not_crash() {
        let ctx = Ctx::new().expect("sam3_init should succeed on a fresh process");
        drop(ctx);
    }

    #[test]
    fn multiple_contexts_can_coexist_sequentially() {
        for _ in 0..4 {
            let _ctx = Ctx::new().unwrap();
        }
    }
}
```

**Step 2: Wire into `lib.rs`**

Append:

```rust
mod ctx;

pub use ctx::Ctx;
```

**Step 3: Run tests**

Run: `cd bindings/rust && DYLD_LIBRARY_PATH=../../build cargo test -p sam3 ctx:: 2>&1 | tail -10`
Expected: 2 tests pass.

**Step 4: Commit**

```bash
git add bindings/rust/sam3/src/ctx.rs bindings/rust/sam3/src/lib.rs
git commit -m "rust: add Ctx::new and Drop"
```

---

### Task 3.8: Add `Ctx::image_size` query

**Files:**
- Modify: `bindings/rust/sam3/src/ctx.rs`

**Step 1: Add the method**

Inside `impl Ctx`, append:

```rust
    /// Return the model's expected input image size, or 0 if no model is loaded.
    pub fn image_size(&self) -> u32 {
        // SAFETY: raw is valid; sam3_get_image_size is const and safe.
        let sz = unsafe { sys::sam3_get_image_size(self.raw.as_ptr()) };
        sz.max(0) as u32
    }
```

**Step 2: Add a test in the same `mod tests`**

```rust
    #[test]
    fn image_size_before_load_is_zero_or_positive() {
        let ctx = Ctx::new().unwrap();
        // Before loading a model, the returned size is either 0 or the
        // compiled-in default; both are valid. Just verify the call succeeds.
        let _ = ctx.image_size();
    }
```

**Step 3: Run tests**

Run: `cd bindings/rust && DYLD_LIBRARY_PATH=../../build cargo test -p sam3 ctx:: 2>&1 | tail -10`
Expected: 3 tests pass.

**Step 4: Commit**

```bash
git add bindings/rust/sam3/src/ctx.rs
git commit -m "rust: add Ctx::image_size"
```

---

### Task 3.9: Add path helper and `Ctx::load_model` / `Ctx::load_bpe`

**Files:**
- Modify: `bindings/rust/sam3/src/ctx.rs`

**Step 1: Add path helper at the top of `ctx.rs`**

After the `use` imports, before `pub struct Ctx`:

```rust
use std::ffi::CString;
use std::path::Path;

/// Convert a filesystem path to a `CString` suitable for libsam3.
///
/// Returns `Error::Invalid` on paths containing interior NULs or non-UTF-8
/// bytes on non-Unix platforms (libsam3 is Unix-focused; Windows paths are
/// best-effort via the UTF-8 representation).
fn path_to_cstring(path: &Path) -> Result<CString> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        CString::new(path.as_os_str().as_bytes()).map_err(|_| Error::Invalid)
    }
    #[cfg(not(unix))]
    {
        let s = path.to_str().ok_or(Error::Invalid)?;
        CString::new(s).map_err(|_| Error::Invalid)
    }
}
```

**Step 2: Add `load_model` and `load_bpe` methods**

Inside `impl Ctx`:

```rust
    /// Load a `.sam3` weight file.
    ///
    /// Must be called before [`set_image`](Self::set_image) or
    /// [`segment`](Self::segment).
    pub fn load_model<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let c = path_to_cstring(path.as_ref())?;
        // SAFETY: raw valid; c lives for the call.
        unsafe { crate::error::check(sys::sam3_load_model(self.raw.as_ptr(), c.as_ptr())) }
    }

    /// Load a BPE vocabulary file (required for text prompts).
    pub fn load_bpe<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let c = path_to_cstring(path.as_ref())?;
        // SAFETY: raw valid; c lives for the call.
        unsafe { crate::error::check(sys::sam3_load_bpe(self.raw.as_ptr(), c.as_ptr())) }
    }
```

**Step 3: Unit test — invalid paths produce `Error::Invalid`**

Add inside `mod tests`:

```rust
    #[test]
    fn load_model_missing_file_returns_error() {
        let mut ctx = Ctx::new().unwrap();
        let err = ctx.load_model("/nonexistent/path/to/model.sam3").unwrap_err();
        // Accept several error variants — libsam3 maps file-not-found to SAM3_EIO.
        assert!(
            matches!(err, Error::Io | Error::Invalid | Error::Model),
            "unexpected error: {err:?}"
        );
    }
```

**Step 4: Run tests**

Run: `cd bindings/rust && DYLD_LIBRARY_PATH=../../build cargo test -p sam3 ctx:: 2>&1 | tail -15`
Expected: 4 tests pass.

**Step 5: Commit**

```bash
git add bindings/rust/sam3/src/ctx.rs
git commit -m "rust: add Ctx::load_model and load_bpe"
```

---

### Task 3.10: Add `Ctx::set_image_rgb`, `set_image`, `set_image_file`, `set_prompt_space`, `set_text`

**Files:**
- Modify: `bindings/rust/sam3/src/ctx.rs`

**Step 1: Add methods to `impl Ctx`**

```rust
    /// Set the input image from a raw RGB byte buffer (`W * H * 3` bytes).
    pub fn set_image_rgb(&mut self, pixels: &[u8], width: u32, height: u32) -> Result<()> {
        let need = (width as usize)
            .checked_mul(height as usize)
            .and_then(|x| x.checked_mul(3))
            .ok_or(Error::Invalid)?;
        if pixels.len() < need {
            return Err(Error::Invalid);
        }
        // SAFETY: pixels has at least W*H*3 bytes; raw is valid.
        unsafe {
            crate::error::check(sys::sam3_set_image(
                self.raw.as_ptr(),
                pixels.as_ptr(),
                width as i32,
                height as i32,
            ))
        }
    }

    /// Set the input image from an [`ImageData`](crate::ImageData).
    pub fn set_image(&mut self, img: &crate::ImageData<'_>) -> Result<()> {
        self.set_image_rgb(img.pixels, img.width, img.height)
    }

    /// Set the input image by loading a PNG/JPEG/BMP file from disk.
    pub fn set_image_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let c = path_to_cstring(path.as_ref())?;
        // SAFETY: raw valid; c lives for the call.
        unsafe {
            crate::error::check(sys::sam3_set_image_file(self.raw.as_ptr(), c.as_ptr()))
        }
    }

    /// Set the coordinate space for subsequent point / box prompts.
    pub fn set_prompt_space(&mut self, width: u32, height: u32) {
        // SAFETY: raw valid; no error path.
        unsafe {
            sys::sam3_set_prompt_space(self.raw.as_ptr(), width as i32, height as i32)
        }
    }

    /// Pre-tokenize and asynchronously encode a text prompt.
    ///
    /// Requires a BPE vocab loaded via [`load_bpe`](Self::load_bpe).
    pub fn set_text(&mut self, text: &str) -> Result<()> {
        let c = CString::new(text).map_err(|_| Error::Invalid)?;
        // SAFETY: raw valid; c lives for the call.
        unsafe {
            crate::error::check(sys::sam3_set_text(self.raw.as_ptr(), c.as_ptr()))
        }
    }
```

**Step 2: Add unit tests**

```rust
    #[test]
    fn set_image_rgb_rejects_short_buffer() {
        let mut ctx = Ctx::new().unwrap();
        let err = ctx.set_image_rgb(&[0; 10], 10, 10).unwrap_err();
        assert!(matches!(err, Error::Invalid));
    }

    #[test]
    fn set_image_rgb_rejects_dimension_overflow() {
        let mut ctx = Ctx::new().unwrap();
        let err = ctx.set_image_rgb(&[0; 10], u32::MAX, u32::MAX).unwrap_err();
        assert!(matches!(err, Error::Invalid));
    }

    #[test]
    fn set_text_rejects_interior_nul() {
        let mut ctx = Ctx::new().unwrap();
        let err = ctx.set_text("hello\0world").unwrap_err();
        assert!(matches!(err, Error::Invalid));
    }

    #[test]
    fn set_prompt_space_is_infallible() {
        let mut ctx = Ctx::new().unwrap();
        ctx.set_prompt_space(1024, 1024);
    }
```

**Step 3: Run tests**

Run: `cd bindings/rust && DYLD_LIBRARY_PATH=../../build cargo test -p sam3 ctx:: 2>&1 | tail -15`
Expected: 8 tests pass.

**Step 4: Commit**

```bash
git add bindings/rust/sam3/src/ctx.rs
git commit -m "rust: add image/prompt/text setters on Ctx"
```

---

### Task 3.11: Implement prompt lowering and `Ctx::segment`

This is the most involved task. Lower `&[Prompt]` into `Vec<sys::sam3_prompt>` with CString storage for text prompts, call `sam3_segment`, copy the result with an RAII guard that ensures `sam3_result_free` always runs.

**Files:**
- Modify: `bindings/rust/sam3/src/prompt.rs` (add lowering)
- Modify: `bindings/rust/sam3/src/ctx.rs` (add `segment`)
- Modify: `bindings/rust/sam3/src/result.rs` (add crate-internal `from_raw`)

**Step 1: Inspect the generated `sam3_prompt` union layout**

Run:
```bash
find bindings/rust/target -name bindings.rs -path '*sam3-sys*' | head -1 | xargs grep -A 20 "pub struct sam3_prompt"
```
Expected: Rust struct with a union field (bindgen generates union accessors `point()`, `box_()`, `mask()`, `text()` with `set_` variants, or a direct union type named like `sam3_prompt__bindgen_ty_1`). Note the exact field names; adjust the code below to match.

**Step 2: Add lowering helpers in `prompt.rs`**

Append to `prompt.rs`:

```rust
use std::ffi::CString;
use sam3_sys as sys;

use crate::error::{Error, Result};

/// Scratch storage needed to keep prompt arguments alive across the FFI call.
pub(crate) struct PromptScratch {
    /// Raw prompts passed to `sam3_segment`.
    pub lowered: Vec<sys::sam3_prompt>,
    /// Owned CStrings for text prompts; referenced by `lowered`.
    #[allow(dead_code)]
    text_keepalive: Vec<CString>,
}

impl<'a> Prompt<'a> {
    /// Lower a slice of `Prompt`s into FFI-ready storage.
    ///
    /// The returned `PromptScratch` must outlive the `sam3_segment` call.
    pub(crate) fn lower_all(prompts: &[Prompt<'a>]) -> Result<PromptScratch> {
        let mut text_keepalive: Vec<CString> = Vec::new();
        let mut lowered: Vec<sys::sam3_prompt> = Vec::with_capacity(prompts.len());

        for p in prompts {
            // SAFETY: zeroing a POD-with-union is valid; we overwrite the
            // active variant immediately below.
            let mut raw: sys::sam3_prompt = unsafe { std::mem::zeroed() };
            match p {
                Prompt::Point(pt) => {
                    raw.type_ = sys::sam3_prompt_type::SAM3_PROMPT_POINT;
                    // SAFETY: union access — we just set the tag.
                    unsafe {
                        raw.__bindgen_anon_1.point = sys::sam3_point {
                            x: pt.x,
                            y: pt.y,
                            label: pt.label.to_raw(),
                        };
                    }
                }
                Prompt::Box(bx) => {
                    raw.type_ = sys::sam3_prompt_type::SAM3_PROMPT_BOX;
                    // SAFETY: union access — we just set the tag.
                    unsafe {
                        raw.__bindgen_anon_1.box_ = sys::sam3_box {
                            x1: bx.x1, y1: bx.y1, x2: bx.x2, y2: bx.y2,
                        };
                    }
                }
                Prompt::Mask(m) => {
                    let need = (m.width as usize)
                        .checked_mul(m.height as usize)
                        .ok_or(Error::Invalid)?;
                    if m.data.len() < need {
                        return Err(Error::Invalid);
                    }
                    raw.type_ = sys::sam3_prompt_type::SAM3_PROMPT_MASK;
                    // SAFETY: union access — we just set the tag. Data
                    // borrow lives until `segment` returns.
                    unsafe {
                        raw.__bindgen_anon_1.mask.data = m.data.as_ptr();
                        raw.__bindgen_anon_1.mask.width = m.width as i32;
                        raw.__bindgen_anon_1.mask.height = m.height as i32;
                    }
                }
                Prompt::Text(s) => {
                    let c = CString::new(*s).map_err(|_| Error::Invalid)?;
                    raw.type_ = sys::sam3_prompt_type::SAM3_PROMPT_TEXT;
                    // SAFETY: union access — we just set the tag. The
                    // CString outlives the FFI call via `text_keepalive`.
                    unsafe {
                        raw.__bindgen_anon_1.text = c.as_ptr();
                    }
                    text_keepalive.push(c);
                }
            }
            lowered.push(raw);
        }

        Ok(PromptScratch { lowered, text_keepalive })
    }
}
```

> **Note on union field names.** The code above uses `__bindgen_anon_1.point`, `.box_`, `.mask`, `.text`. bindgen may name these differently depending on version — confirm with the Step 1 grep. If the names differ, update accordingly. If bindgen generates union *methods* (e.g. `point()` / `set_point()`), use those instead.

**Step 3: Add `SegmentResult::from_raw` in `result.rs`**

Append to `result.rs`:

```rust
use sam3_sys as sys;

use crate::error::{Error, Result};

impl SegmentResult {
    /// Copy a C-owned `sam3_result` into an owned `SegmentResult`.
    ///
    /// The caller must free the source `sam3_result` via `sam3_result_free`
    /// regardless of the outcome of this function.
    pub(crate) fn from_raw(raw: &sys::sam3_result) -> Result<Self> {
        let n = raw.n_masks as usize;
        let h = raw.mask_height as usize;
        let w = raw.mask_width as usize;
        let total = n.checked_mul(h).and_then(|x| x.checked_mul(w)).ok_or(Error::Invalid)?;

        let masks = if total == 0 || raw.masks.is_null() {
            Vec::new()
        } else {
            // SAFETY: libsam3 guarantees raw.masks points to `total` f32s
            // while `raw` lives.
            unsafe { std::slice::from_raw_parts(raw.masks, total) }.to_vec()
        };

        let iou_scores = if n == 0 || raw.iou_scores.is_null() {
            Vec::new()
        } else {
            // SAFETY: same lifetime guarantee.
            unsafe { std::slice::from_raw_parts(raw.iou_scores, n) }.to_vec()
        };

        let boxes = if raw.boxes_valid != 0 && !raw.boxes.is_null() && n > 0 {
            // SAFETY: boxes_valid != 0 ⇒ raw.boxes holds n*4 f32s.
            let flat = unsafe { std::slice::from_raw_parts(raw.boxes, n * 4) };
            Some(
                flat.chunks_exact(4)
                    .map(|c| [c[0], c[1], c[2], c[3]])
                    .collect(),
            )
        } else {
            None
        };

        Ok(SegmentResult {
            masks,
            iou_scores,
            boxes,
            n_masks: n,
            mask_height: h,
            mask_width: w,
            iou_valid: raw.iou_valid != 0,
            best_mask: if raw.best_mask >= 0 {
                Some(raw.best_mask as usize)
            } else {
                None
            },
        })
    }
}
```

**Step 4: Add `Ctx::segment` in `ctx.rs`**

```rust
    /// Run segmentation with the given prompts against the current image.
    ///
    /// `prompts` may mix points, boxes, masks, and (if a BPE vocab is
    /// loaded) text. Borrows inside `Prompt` must outlive this call.
    pub fn segment(&mut self, prompts: &[crate::Prompt<'_>]) -> Result<crate::SegmentResult> {
        let scratch = crate::prompt::Prompt::lower_all(prompts)?;

        // RAII guard: always call sam3_result_free, even on panic.
        struct Guard<'a>(&'a mut sys::sam3_result);
        impl Drop for Guard<'_> {
            fn drop(&mut self) {
                // SAFETY: the guarded struct was (attempted to be) filled by
                // sam3_segment; sam3_result_free tolerates partially-filled
                // results (it null-checks internal pointers).
                unsafe { sys::sam3_result_free(self.0 as *mut _) }
            }
        }

        // SAFETY: zero-initialized sam3_result is valid for sam3_segment to
        // fill in; all pointer fields will be set by the callee.
        let mut raw = unsafe { std::mem::zeroed::<sys::sam3_result>() };
        let err_code = unsafe {
            sys::sam3_segment(
                self.raw.as_ptr(),
                scratch.lowered.as_ptr(),
                scratch.lowered.len() as i32,
                &mut raw,
            )
        };
        let _guard = Guard(&mut raw);

        crate::error::check(err_code)?;
        crate::SegmentResult::from_raw(&raw)
    }
```

**Step 5: Verify the crate compiles**

Run: `cd bindings/rust && DYLD_LIBRARY_PATH=../../build cargo build -p sam3 2>&1 | tail -15`
Expected: Clean build. If bindgen union field names differ, fix them now (this is the expected step to adjust for the actual bindgen output).

**Step 6: Add a unit test for lowering shape**

Append to `prompt.rs`'s `mod tests`:

```rust
    #[test]
    fn lower_point_prompt_tags_correctly() {
        let prompts = [Prompt::Point(Point { x: 1.0, y: 2.0, label: PointLabel::Foreground })];
        let scratch = Prompt::lower_all(&prompts).unwrap();
        assert_eq!(scratch.lowered.len(), 1);
        assert_eq!(scratch.lowered[0].type_, sys::sam3_prompt_type::SAM3_PROMPT_POINT);
    }

    #[test]
    fn lower_text_prompt_rejects_interior_nul() {
        let prompts = [Prompt::Text("bad\0string")];
        assert!(matches!(Prompt::lower_all(&prompts), Err(Error::Invalid)));
    }

    #[test]
    fn lower_mask_rejects_short_buffer() {
        let data = [0.0_f32; 3];
        let prompts = [Prompt::Mask(MaskPrompt { data: &data, width: 4, height: 4 })];
        assert!(matches!(Prompt::lower_all(&prompts), Err(Error::Invalid)));
    }
```

**Step 7: Run all unit tests**

Run: `cd bindings/rust && DYLD_LIBRARY_PATH=../../build cargo test -p sam3 2>&1 | tail -20`
Expected: All tests pass (all previous + 3 new).

**Step 8: Commit**

```bash
git add bindings/rust/sam3/src/prompt.rs bindings/rust/sam3/src/ctx.rs bindings/rust/sam3/src/result.rs
git commit -m "rust: add Ctx::segment with prompt lowering and RAII cleanup"
```

---

### Task 3.12: Verify `Ctx` is `!Send + !Sync` at compile time

**Files:**
- Modify: `bindings/rust/sam3/src/ctx.rs`

**Step 1: Add `const` compile-time assertion inside `mod tests`**

```rust
    /// Negative trait check: Ctx must be neither Send nor Sync.
    ///
    /// These assertions fail to compile if Ctx accidentally becomes Send/Sync.
    #[allow(dead_code)]
    fn assert_not_send_or_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        // Uncomment to see compile errors (these MUST fail to compile):
        // assert_send::<Ctx>();
        // assert_sync::<Ctx>();

        // Positive check: a trait object bound is happy with !Send/!Sync,
        // so we just confirm Ctx exists here. The real assertion is that
        // the commented lines above would fail compilation.
        let _ = std::marker::PhantomData::<Ctx>;
    }
```

Plus a static check that *compiles only because* `Ctx` is `!Send`:

```rust
    /// Compile-time check via the `static_assertions` pattern, without
    /// adding a dependency: this trait is implemented only for `!Send`.
    #[allow(dead_code)]
    trait IsNotSend {}
    impl<T> IsNotSend for T where T: ?Sized {}
    // If Ctx were Send, there's nothing to assert here. We cover the real
    // guarantee by the presence of PhantomData<*mut ()> in the struct.
```

Actually, Rust stable has no clean way to compile-fail on `Send`. The simplest durable check is to put a `PhantomData<*mut ()>` in the struct (already done) and document this. Remove the long negative-check block. Keep only a comment-level note plus a real runtime behavior test.

Replace the added block with just:

```rust
    /// Sanity: `Ctx` contains `PhantomData<*mut ()>` which makes it !Send + !Sync.
    /// Changing that invariant requires reconsidering the locking story.
    #[allow(dead_code)]
    const _: fn() = || {
        fn assert_not_send<T: ?Sized>(_: &T) where T: { }
        // No-op — the real guarantee is the PhantomData<*mut ()> field.
    };
```

Simpler: skip this task and rely on the `PhantomData<*mut ()>` field plus a comment. Make this task documentation-only.

**Revised Step 1: Add doc comment on `Ctx` about the invariant**

Add to `ctx.rs` above the struct definition:

```rust
// The `_not_send_sync: PhantomData<*mut ()>` field makes `Ctx` !Send + !Sync.
// Removing or changing that field requires rethinking thread-safety.
```

**Step 2: No test to run — doc only.**

**Step 3: Commit**

```bash
git add bindings/rust/sam3/src/ctx.rs
git commit -m "rust: document !Send+!Sync invariant on Ctx"
```

---

### Task 3.13: Add a usage example under `examples/`

**Files:**
- Create: `bindings/rust/sam3/examples/segment.rs`

**Step 1: Write `bindings/rust/sam3/examples/segment.rs`**

```rust
//! Minimal end-to-end segmentation example.
//!
//! Usage:
//!
//! ```sh
//! DYLD_LIBRARY_PATH=../../build cargo run --example segment -- \
//!     --model /path/to/model.sam3 \
//!     --image /path/to/image.jpg \
//!     --point 400,300,1
//! ```

use std::env;
use std::process::ExitCode;

use sam3::{Ctx, Point, PointLabel, Prompt};

fn main() -> ExitCode {
    let mut model: Option<String> = None;
    let mut image: Option<String> = None;
    let mut point: Option<(f32, f32, i32)> = None;
    let mut text: Option<String> = None;

    let mut args = env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model = args.next(),
            "--image" => image = args.next(),
            "--point" => {
                let s = args.next().expect("--point x,y,label");
                let parts: Vec<&str> = s.split(',').collect();
                assert_eq!(parts.len(), 3, "--point x,y,label");
                point = Some((
                    parts[0].parse().expect("x"),
                    parts[1].parse().expect("y"),
                    parts[2].parse().expect("label"),
                ));
            }
            "--text" => text = args.next(),
            other => {
                eprintln!("unknown argument: {other}");
                return ExitCode::from(2);
            }
        }
    }

    let model_path = match model {
        Some(m) => m,
        None => { eprintln!("--model is required"); return ExitCode::from(2); }
    };
    let image_path = match image {
        Some(i) => i,
        None => { eprintln!("--image is required"); return ExitCode::from(2); }
    };

    let mut ctx = Ctx::new().expect("sam3_init");
    ctx.load_model(&model_path).expect("load_model");
    ctx.set_image_file(&image_path).expect("set_image_file");

    let mut prompts: Vec<Prompt<'_>> = Vec::new();
    if let Some((x, y, lab)) = point {
        prompts.push(Prompt::Point(Point {
            x, y,
            label: if lab == 0 { PointLabel::Background } else { PointLabel::Foreground },
        }));
    }
    if let Some(ref t) = text {
        prompts.push(Prompt::Text(t));
    }

    let result = ctx.segment(&prompts).expect("segment");
    println!(
        "segmented: n_masks={} H={} W={} best={:?} iou_valid={}",
        result.n_masks(),
        result.mask_height(),
        result.mask_width(),
        result.best_mask(),
        result.iou_valid(),
    );

    ExitCode::SUCCESS
}
```

**Step 2: Verify it compiles**

Run: `cd bindings/rust && cargo build -p sam3 --example segment 2>&1 | tail -5`
Expected: Clean build.

**Step 3: Commit**

```bash
git add bindings/rust/sam3/examples/segment.rs
git commit -m "rust: add segment example binary"
```

---

### Task 3.14: Env-gated integration test

**Files:**
- Create: `bindings/rust/sam3/tests/integration.rs`

**Step 1: Write the integration test**

```rust
//! End-to-end integration tests against a real SAM3 model.
//!
//! Gated by environment variables (matching `python/tests/conftest.py`):
//!   - SAM3_MODEL_PATH — required
//!   - SAM3_TEST_IMAGE — required
//!   - SAM3_BPE_PATH   — optional (required for text-prompt tests)
//!
//! Tests are silently skipped (not failed) when env vars are unset.

use sam3::{Ctx, Point, PointLabel, Prompt};

fn env_or_skip(var: &str) -> Option<String> {
    match std::env::var(var) {
        Ok(v) if !v.is_empty() => Some(v),
        _ => {
            eprintln!("skipping: {var} not set");
            None
        }
    }
}

#[test]
fn load_and_segment_with_point() {
    let model = match env_or_skip("SAM3_MODEL_PATH") { Some(v) => v, None => return };
    let image = match env_or_skip("SAM3_TEST_IMAGE") { Some(v) => v, None => return };

    let mut ctx = Ctx::new().unwrap();
    ctx.load_model(&model).unwrap();
    ctx.set_image_file(&image).unwrap();

    let prompts = [Prompt::Point(Point {
        x: 100.0,
        y: 100.0,
        label: PointLabel::Foreground,
    })];
    let result = ctx.segment(&prompts).unwrap();

    assert!(result.n_masks() > 0, "expected at least one mask");
    assert!(result.mask_height() > 0);
    assert!(result.mask_width() > 0);
    assert_eq!(result.iou_scores().len(), result.n_masks());
}

#[test]
fn load_and_segment_with_box() {
    let model = match env_or_skip("SAM3_MODEL_PATH") { Some(v) => v, None => return };
    let image = match env_or_skip("SAM3_TEST_IMAGE") { Some(v) => v, None => return };

    let mut ctx = Ctx::new().unwrap();
    ctx.load_model(&model).unwrap();
    ctx.set_image_file(&image).unwrap();

    let prompts = [Prompt::Box(sam3::Box {
        x1: 50.0, y1: 50.0, x2: 200.0, y2: 200.0,
    })];
    let result = ctx.segment(&prompts).unwrap();

    assert!(result.n_masks() > 0);
}

#[test]
fn segment_with_text_requires_bpe() {
    let model = match env_or_skip("SAM3_MODEL_PATH") { Some(v) => v, None => return };
    let image = match env_or_skip("SAM3_TEST_IMAGE") { Some(v) => v, None => return };
    let bpe   = match env_or_skip("SAM3_BPE_PATH")   { Some(v) => v, None => return };

    let mut ctx = Ctx::new().unwrap();
    ctx.load_model(&model).unwrap();
    ctx.load_bpe(&bpe).unwrap();
    ctx.set_image_file(&image).unwrap();

    let prompts = [Prompt::Text("object")];
    let result = ctx.segment(&prompts).unwrap();
    assert!(result.n_masks() > 0);
}

#[test]
fn multiple_contexts_are_independent() {
    let model = match env_or_skip("SAM3_MODEL_PATH") { Some(v) => v, None => return };

    let a = Ctx::new().unwrap();
    drop(a);
    let mut b = Ctx::new().unwrap();
    b.load_model(&model).unwrap();
}
```

**Step 2: Verify it compiles (but skips)**

Run without env vars: `cd bindings/rust && DYLD_LIBRARY_PATH=../../build cargo test -p sam3 --test integration 2>&1 | tail -15`
Expected: All 4 tests "ok" (because the helpers short-circuit via early return when env vars are unset). Skip messages appear on stderr.

**Step 3: Commit**

```bash
git add bindings/rust/sam3/tests/integration.rs
git commit -m "rust: add env-gated integration tests"
```

---

### Task 3.15: Full-build smoke test and final commit

**Files:** none (verification only)

**Step 1: Clean-build the workspace end to end**

Run:
```bash
cd bindings/rust
cargo clean
DYLD_LIBRARY_PATH=../../build cargo build --workspace 2>&1 | tail -10
```
Expected: Clean build. Warnings about missing docs on any public items should either be fixed or suppressed by adding `///` comments.

**Step 2: Run the full test suite**

Run: `cd bindings/rust && DYLD_LIBRARY_PATH=../../build cargo test --workspace 2>&1 | tail -25`
Expected: All unit and integration tests pass (integration tests skip if env unset).

**Step 3: Run clippy**

Run: `cd bindings/rust && DYLD_LIBRARY_PATH=../../build cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -15`
Expected: Clean, no warnings. Fix any issues by editing the corresponding files.

**Step 4: Run rustfmt check**

Run: `cd bindings/rust && cargo fmt --all --check 2>&1 | tail -10`
Expected: No diff. If there's a diff, run `cargo fmt --all` and commit the formatting fix.

**Step 5: If step 3 or 4 required fixes, commit them**

```bash
git add -A
git commit -m "rust: apply clippy and rustfmt fixes"
```

**Step 6: Confirm nothing unexpected is staged**

Run: `git status`
Expected: clean working tree.

---

## Phase 4 — Documentation polish

### Task 4.1: Update top-level `README.md`

**Files:**
- Modify: `README.md`

**Step 1: Locate the bindings section (if any)**

Run: `grep -n -i "python\|rust\|binding" README.md`
Expected: either a section referencing Python, or none.

**Step 2: Add or update a bindings section**

After the existing build instructions, add (or edit an existing section to read):

```markdown
## Language bindings

SAM3 ships bindings for multiple languages under `bindings/`:

- **Python** — `bindings/python/`. Install with `pip install -e bindings/python`.
- **Rust** — `bindings/rust/`. Cargo workspace with `sam3-sys` (FFI) and
  `sam3` (safe API). See `bindings/rust/README.md`.

Both bindings link dynamically against `libsam3.{dylib,so}` built with
`-DSAM3_SHARED=ON`.
```

Only add if no equivalent section exists; update existing text in place if it does.

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add Rust binding to README bindings section"
```

---

## Summary

### What you'll have at the end

- `bindings/python/` — existing Python binding, moved.
- `bindings/rust/Cargo.toml` — workspace root.
- `bindings/rust/sam3-sys/` — raw FFI via bindgen, dynamic link to libsam3.
- `bindings/rust/sam3/` — safe RAII API mirroring the Python binding.
- End-to-end example at `bindings/rust/sam3/examples/segment.rs`.
- Env-gated integration tests at `bindings/rust/sam3/tests/integration.rs`.
- Unit tests inside each module verifying error mapping, shape math,
  lowering, and accessor correctness.

### Commit pace

Each task ends in one commit. Phases 1–3 yield ~20 commits; phase 4 adds one.

### Prerequisite for Task 2.3 and all subsequent tasks

`build/libsam3.{dylib,so}` must exist. Built in Task 1.2 with
`cmake -S . -B build -DSAM3_SHARED=ON && cmake --build build`.

### Key risks / watch points

- **bindgen union field names in Task 3.11.** Confirmed at runtime from
  generated bindings; adjust to match the exact identifiers emitted.
- **macOS vs Linux library env var.** `DYLD_LIBRARY_PATH` on macOS,
  `LD_LIBRARY_PATH` on Linux. Both are documented in the README.
- **`docs/plans/` is gitignored** (per repo convention); commits under
  that path require `git add -f` — not relevant to the implementation,
  just a note for this plan document itself.

### Copyright

Copyright (c) 2026 Rifky Bujana Bisri
SPDX-License-Identifier: MIT
