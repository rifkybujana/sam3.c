# Cache API Bindings (Python + Rust) — Design

**Date:** 2026-04-22
**Status:** Approved for implementation planning
**Scope:** Expose the encoder feature-cache + persistence C API in both the
Python (`bindings/python/`) and Rust (`bindings/rust/`) bindings. No libsam3
C changes; no Tauri glue in this spec.

## Motivation

`include/sam3/sam3.h` ships a complete in-memory encoder cache and `.sam3cache`
persistence surface (commits `ba1651f`, `ceeada7`, `7e1f7b4`):

- `sam3_init_ex` / `struct sam3_cache_opts` — tune image/text slot counts
- `sam3_cache_clear(ctx, which)` / `sam3_cache_stats` — runtime control + counters
- `sam3_precache_image` / `_file` / `sam3_precache_text` — pre-warm entries
- `sam3_cache_save_image` / `sam3_cache_load_image`
- `sam3_cache_save_text` / `sam3_cache_load_text`

Neither binding currently wraps any of these. The only `cache` references in
`bindings/python/` and `bindings/rust/` are the unrelated video-session frame
cache budgets (`frame_cache_backend_budget`, `frame_cache_spill_budget`).

Downstream users building image-editor UIs on top of the Python or Rust
bindings (including a future Tauri application) cannot currently pre-warm the
encoder cache, observe hit/miss rates, or persist encoder features to disk
without writing their own C FFI.

## Non-Goals

- **No changes to libsam3 or `include/sam3/sam3.h`.** The C API is the source
  of truth; this work mirrors it into the two bindings.
- **No Tauri glue / no WebAssembly / no new bindings.** A Tauri host can depend
  directly on the `sam3` Rust crate once this work lands; that crate already
  covers image segmentation end-to-end.
- **No path-as-key variant of `cache_save_image` in Python.** Requiring Python
  to decode images would force a Pillow/OpenCV dependency that the current
  bindings avoid. See Follow-ups for the proper fix.
- **No async/threaded wrappers.** `sam3_precache_*` is blocking in C; the
  bindings call it blocking. Python/Rust callers that want concurrency spawn
  their own worker.

## Scope Overview

| API | Python surface | Rust surface |
|---|---|---|
| `sam3_init_ex` / `sam3_cache_opts` | `Model(..., image_cache_slots=, text_cache_slots=)` | `Ctx::new_with_cache_opts(&CacheOpts)` + `CacheOpts` struct |
| `sam3_cache_clear` | `Model.cache_clear(image=, text=)` | `Ctx::cache_clear(CacheKind)` + `bitflags` `CacheKind` |
| `sam3_cache_stats` | `Model.cache_stats()` → `CacheStats` dataclass | `Ctx::cache_stats()` → `CacheStats` struct |
| `sam3_precache_image` | `Model.precache_image(path_or_ndarray)` | `Ctx::precache_image_rgb`, `precache_image`, `precache_image_file` |
| `sam3_precache_text` | `Model.precache_text(text)` | `Ctx::precache_text(&str)` |
| `sam3_cache_save_image` | `Model.cache_save_image(ndarray, path)` | `Ctx::cache_save_image_rgb`, `cache_save_image` |
| `sam3_cache_load_image` | `Model.cache_load_image(path)` | `Ctx::cache_load_image` |
| `sam3_cache_save_text` | `Model.cache_save_text(text, path)` | `Ctx::cache_save_text` |
| `sam3_cache_load_text` | `Model.cache_load_text(path)` | `Ctx::cache_load_text` |

## Detailed API Surface

### 1. Cache slot tuning (`sam3_init_ex`)

**Python** — extend `Model.__init__` keyword args. `None` on either slot count
selects the C default (3 image slots, 16 text slots):

```python
with sam3.Model("model.sam3",
                bpe_path=None,
                image_cache_slots=None,
                text_cache_slots=None) as m:
    ...
```

Implementation: when both slot kwargs are `None`, call `sam3_init()` unchanged
(preserves existing behavior exactly). Otherwise build a `struct sam3_cache_opts`
via cffi and call `sam3_init_ex`. Reject negative or zero slot counts with
`ValueError` before calling into C.

**Rust** — a new constructor. `Ctx::new()` stays unchanged for existing
callers:

```rust
#[derive(Debug, Clone, Copy, Default)]
pub struct CacheOpts {
    pub image_slots: Option<u32>,   // None → C default
    pub text_slots:  Option<u32>,
}

impl Ctx {
    pub fn new() -> Result<Self>;                                 // sam3_init()
    pub fn new_with_cache_opts(opts: &CacheOpts) -> Result<Self>; // sam3_init_ex()
}
```

Implementation lowers `Option<u32>` to `i32` (0 = default in C). Overflow of
`u32` > `i32::MAX` → `Error::Invalid`.

### 2. Cache clear (`sam3_cache_clear`)

**Python:**

```python
model.cache_clear()                         # both (C semantics: which=0)
model.cache_clear(image=True)               # image only
model.cache_clear(text=True)                # text only
model.cache_clear(image=True, text=True)    # both, explicit
```

Lowering: when both args default to `False`, pass `which = 0` (C's "0 = both").
Otherwise OR together `SAM3_CACHE_IMAGE` and/or `SAM3_CACHE_TEXT`.

**Rust** — new dependency `bitflags = "2"` on the `sam3` crate:

```rust
bitflags::bitflags! {
    pub struct CacheKind: u32 {
        const IMAGE = 1 << 0;
        const TEXT  = 1 << 1;
    }
}

impl Ctx {
    pub fn cache_clear(&mut self, which: CacheKind);  // infallible, void in C
}
```

`CacheKind::empty()` maps to `which = 0` (clears both, matching C).

### 3. Cache stats (`sam3_cache_stats`)

**Python:**

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class CacheStats:
    image_hits: int
    image_misses: int
    image_evictions: int
    text_hits: int
    text_misses: int
    text_evictions: int

stats = model.cache_stats()          # infallible
```

`CacheStats` lives in `sam3/model.py` alongside `Model` and `Result`, and is
re-exported via `sam3/__init__.py`.

**Rust** — plain struct in a new `cache.rs` module, re-exported from
`lib.rs`:

```rust
#[derive(Debug, Clone, Copy, Default)]
pub struct CacheStats {
    pub image_hits: u64,
    pub image_misses: u64,
    pub image_evictions: u64,
    pub text_hits: u64,
    pub text_misses: u64,
    pub text_evictions: u64,
}

impl Ctx {
    pub fn cache_stats(&self) -> CacheStats;  // &self; C takes const ctx*
}
```

### 4. Precache (`sam3_precache_*`)

**Python** — mirror `set_image`'s path-or-ndarray overload:

```python
model.precache_image("photo.jpg")       # path → sam3_precache_image_file
model.precache_image(rgb_ndarray)       # array → sam3_precache_image
model.precache_text("cat")              # sam3_precache_text
```

Validation identical to `set_image`: `(H, W, 3)` uint8 required; other shapes
raise `ValueError` before calling C.

**Rust** — split methods matching `set_image_rgb` / `set_image_file`:

```rust
impl Ctx {
    pub fn precache_image_rgb(&mut self, pixels: &[u8],
                              width: u32, height: u32) -> Result<()>;
    pub fn precache_image(&mut self, img: &ImageData<'_>) -> Result<()>;  // convenience
    pub fn precache_image_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()>;
    pub fn precache_text(&mut self, text: &str) -> Result<()>;
}
```

Buffer-length validation and overflow handling mirror `Ctx::set_image_rgb`
exactly (return `Error::Invalid` on short buffers or `width * height * 3`
overflow).

### 5. Save / load `.sam3cache` (`sam3_cache_{save,load}_{image,text}`)

**Python** — pixel-keyed save, path-only load:

```python
model.cache_save_image(rgb_ndarray, "img.sam3cache")   # key = ndarray
model.cache_load_image("img.sam3cache")

model.cache_save_text("cat", "cat.sam3cache")
model.cache_load_text("cat.sam3cache")
```

`cache_save_image` rejects non-ndarray first args (`TypeError`). Users who
held a file path can decode once with Pillow/OpenCV and feed the ndarray.
Calling `cache_save_image` with pixels that were never precached/set surfaces
as `Sam3Error(SAM3_EINVAL)` from the C side; the binding does not pre-check.

**Rust** — two overloads for image save, one for text. All paths use
`AsRef<Path>`:

```rust
impl Ctx {
    pub fn cache_save_image_rgb<P: AsRef<Path>>(
        &mut self, pixels: &[u8], width: u32, height: u32, out_path: P,
    ) -> Result<()>;
    pub fn cache_save_image<P: AsRef<Path>>(
        &mut self, img: &ImageData<'_>, out_path: P,
    ) -> Result<()>;
    pub fn cache_load_image<P: AsRef<Path>>(&mut self, path: P) -> Result<()>;

    pub fn cache_save_text<P: AsRef<Path>>(
        &mut self, text: &str, out_path: P,
    ) -> Result<()>;
    pub fn cache_load_text<P: AsRef<Path>>(&mut self, path: P) -> Result<()>;
}
```

Both bindings forward all errors from the C layer unchanged (mismatched model
signature on load surfaces as `Error::Model` / `Sam3Error(SAM3_EMODEL)`;
missing cache entry on save surfaces as `Error::Invalid` / `SAM3_EINVAL`).

## Implementation Notes

### Python

- **`bindings/python/sam3/_ffi.py`** — append `ffi.cdef` declarations for
  `sam3_cache_opts`, `sam3_cache_stats`, `sam3_init_ex`, `sam3_cache_clear`,
  `sam3_cache_stats`, `sam3_precache_image`, `sam3_precache_image_file`,
  `sam3_precache_text`, `sam3_cache_save_image`, `sam3_cache_load_image`,
  `sam3_cache_save_text`, `sam3_cache_load_text`, and the
  `SAM3_CACHE_IMAGE` / `SAM3_CACHE_TEXT` constants.
- **`bindings/python/sam3/model.py`** — add `CacheStats` dataclass plus all new
  `Model` methods. Re-export `CacheStats` from `__init__.py`.
- **Tests** (`bindings/python/tests/`): a new `test_cache.py` covering
  init_ex parameter passing, precache hit/miss on stats, clear semantics,
  save/load round-trip using tmp_path fixtures.

### Rust

- **`bindings/rust/sam3-sys`** — *no manual changes*. `build.rs` runs
  `bindgen` with `allowlist_function("sam3_.*")` / `allowlist_type("sam3_.*")`
  / `allowlist_var("SAM3_.*")`, so the cache API auto-appears on the next
  build. Bump `sam3-sys` patch version (`0.1.0` → `0.1.1`) to reflect the
  expanded surface.
- **`bindings/rust/sam3/Cargo.toml`** — add `bitflags = "2"` dep; bump `sam3`
  patch version.
- **New file `bindings/rust/sam3/src/cache.rs`** — `CacheOpts`, `CacheKind`
  (bitflags), `CacheStats`. Re-exported from `lib.rs`.
- **`bindings/rust/sam3/src/ctx.rs`** — add `new_with_cache_opts`,
  `cache_clear`, `cache_stats`, `precache_*`, `cache_save_*`, `cache_load_*`
  methods on `Ctx`. Each method follows the existing pattern: SAFETY block,
  `check(...)` lowering, concise rustdoc.
- **Tests** — inline `#[cfg(test)] mod tests` in `ctx.rs` (matching the
  existing convention there), plus a dedicated `cache.rs` submodule for
  `CacheStats` `Default`/`Debug` checks. Long-running encode tests gated
  behind the existing model-file fixture convention used by
  `bindings/rust/sam3/tests/`.

### Error-path parity

The C API documents these cache-specific error codes:

- `SAM3_EINVAL` — no matching cache entry on save; bad slot counts on init_ex;
  zero dimensions on precache.
- `SAM3_EMODEL` — model-signature mismatch when loading a `.sam3cache` file
  produced against a different model.
- `SAM3_EIO` — file read/write failure on save/load.

Both bindings forward these unchanged: Python raises `Sam3Error(code)`,
Rust maps to `Error::Invalid` / `Error::Model` / `Error::Io`. Neither binding
adds wrapping error types.

## Testing

### Python

Add `bindings/python/tests/test_cache.py`:

- `test_init_ex_accepts_custom_slot_counts` — construct `Model` with explicit
  slot counts, verify no exception.
- `test_init_ex_rejects_zero_slots` — `ValueError` before the C call.
- `test_cache_stats_zero_on_fresh_ctx` — all six counters == 0 on a newly
  created `Model`.
- `test_precache_image_then_set_image_hits` — precache an ndarray, then
  `set_image` the same ndarray, assert `image_hits == 1`.
- `test_cache_clear_resets_counters_and_entries` — after clearing, a repeated
  `set_image` miss re-increments from zero.
- `test_cache_save_load_image_round_trip` — precache, save to
  `tmp_path / "x.sam3cache"`, open fresh `Model` with same weights, load,
  verify hit on matching pixels.
- `test_cache_save_image_requires_ndarray` — passing a `str` raises
  `TypeError`.
- `test_cache_save_image_missing_entry_raises` — saving pixels never
  precached raises `Sam3Error` with `SAM3_EINVAL`.

### Rust

Extend inline `#[cfg(test)]` mod in `ctx.rs`:

- `new_with_cache_opts_accepts_zero_as_default` — `CacheOpts::default()`
  (both None) produces a valid `Ctx` equivalent to `Ctx::new()`.
- `cache_clear_on_fresh_ctx_is_noop` — no panic, counters stay zero.
- `cache_stats_fresh_ctx_is_all_zero` — struct fields all `0`.

New `bindings/rust/sam3/tests/cache.rs` (model-backed integration):

- `precache_image_then_set_image_hits` — gated by the existing runtime
  model-file discovery; skipped if absent (match `tests/model.rs` pattern).
- `cache_save_load_image_round_trip` — two `Ctx` instances, same model,
  save from first, load into second, verify `image_hits == 1` on the second.

Benchmarks are out of scope for this work; the existing C-level cache tests
in `tests/test_processor_cache.c` already cover performance regressions.

## Follow-ups (out of scope)

1. **`sam3_cache_save_image_file`** — a C-side helper that decodes the file
   and looks up the cache key internally, so Python callers who used
   `precache_image_file` can save without holding decoded pixels. This is a
   small addition to `sam3.h` and `src/model/feature_cache.c`; it removes the
   only real usability gap this spec tolerates.

2. **Tauri bindings** — once the Rust `sam3` crate exposes the full cache
   surface, a Tauri app can depend on it directly via path or crates.io and
   wrap `#[tauri::command]` functions in its own `src-tauri/`. No further
   work in this repo is required unless we decide to ship a reusable
   `bindings/tauri/` template crate.

3. **WebAssembly** — out of scope for any Tauri use case (Tauri ships native
   binaries). If a browser-only deployment ever becomes a goal, revisit as a
   separate spec: CPU-only Emscripten build with WASM SIMD128, plus a JS
   glue layer over the public C API.

## References

- `include/sam3/sam3.h:44-257` — cache and persistence API declarations
- `bindings/python/sam3/model.py` — existing Python binding style
- `bindings/rust/sam3/src/ctx.rs` — existing Rust binding style
- `docs/superpowers/specs/2026-04-21-encoder-feature-cache-design.md` —
  in-memory encoder cache architecture (C side)
