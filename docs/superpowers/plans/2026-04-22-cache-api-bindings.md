# Cache API Bindings (Python + Rust) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the encoder feature-cache + `.sam3cache` persistence API in both the Python and Rust bindings, with full test coverage. No C changes.

**Architecture:** Python side is a manually-maintained cffi cdef + `Model` method additions; the Rust side lets bindgen regenerate `sam3-sys` automatically and adds a safe wrapper layer on `Ctx` plus a new `cache` module for the shared types.

**Tech Stack:** Python (cffi, numpy, pytest), Rust (bindgen, bitflags, cargo test), libsam3 shared library (`cmake -DSAM3_SHARED=ON`).

**Reference spec:** `docs/superpowers/specs/2026-04-22-cache-api-bindings-design.md`

---

## Task 0: Prerequisites

Confirm libsam3 builds as a shared library. Every subsequent task depends on this.

- [ ] **Step 1: Build libsam3 shared from the repo root**

```bash
cmake -S . -B build -DSAM3_SHARED=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```

Expected: build succeeds, `build/libsam3.{dylib,so}` exists.

- [ ] **Step 2: Sanity check existing bindings still work**

```bash
cd bindings/python && pip install -e . && cd ../..
cd bindings/rust && DYLD_LIBRARY_PATH=../../build cargo build --workspace && cd ../..
```

Expected: both complete without errors.

---

# Phase 1 — Python binding

## Task 1.1: Add cache API cdef declarations to `_ffi.py`

**Files:**
- Modify: `bindings/python/sam3/_ffi.py` (append to existing `ffi.cdef` block)

This exposes the C symbols to Python. No behavior yet — just makes `lib.sam3_init_ex`, `lib.sam3_cache_clear`, etc. resolve.

- [ ] **Step 1: Write the failing test**

Create `bindings/python/tests/test_cache_ffi.py`:

```python
"""Smoke tests that the cache API cdef declarations resolve."""
import sam3
from sam3._lib import ffi, lib


def test_cache_symbols_resolve():
    """All cache API symbols must be visible on lib."""
    # Lifecycle
    assert hasattr(lib, "sam3_init_ex")
    # Runtime
    assert hasattr(lib, "sam3_cache_clear")
    assert hasattr(lib, "sam3_cache_stats")
    # Precache
    assert hasattr(lib, "sam3_precache_image")
    assert hasattr(lib, "sam3_precache_image_file")
    assert hasattr(lib, "sam3_precache_text")
    # Persistence
    assert hasattr(lib, "sam3_cache_save_image")
    assert hasattr(lib, "sam3_cache_load_image")
    assert hasattr(lib, "sam3_cache_save_text")
    assert hasattr(lib, "sam3_cache_load_text")


def test_cache_bitmask_constants():
    """SAM3_CACHE_IMAGE and SAM3_CACHE_TEXT are defined with expected values."""
    assert int(lib.SAM3_CACHE_IMAGE) == 1
    assert int(lib.SAM3_CACHE_TEXT) == 2


def test_cache_opts_struct_has_expected_fields():
    """struct sam3_cache_opts can be allocated and fields assigned."""
    opts = ffi.new("struct sam3_cache_opts *")
    opts.n_image_slots = 5
    opts.n_text_slots = 32
    assert opts.n_image_slots == 5
    assert opts.n_text_slots == 32


def test_cache_stats_struct_has_expected_fields():
    """struct sam3_cache_stats exposes all six counters."""
    stats = ffi.new("struct sam3_cache_stats *")
    stats.image_hits = 1
    stats.image_misses = 2
    stats.image_evictions = 3
    stats.text_hits = 4
    stats.text_misses = 5
    stats.text_evictions = 6
    assert stats.image_hits == 1
    assert stats.text_evictions == 6
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd bindings/python && pytest tests/test_cache_ffi.py -v
```

Expected: failures on `hasattr(lib, "sam3_init_ex")` and on `ffi.new("struct sam3_cache_opts *")` because the declarations don't exist yet.

- [ ] **Step 3: Append cdef declarations in `_ffi.py`**

Add this block inside the existing `ffi.cdef("""...""")` call, **before** the closing `""")`. Put it after the `/* Debug */` section and before the `/* Video tracking */` section so the file stays organized by subsystem:

```c
    /* Cache tuning (sam3_init_ex) */
    struct sam3_cache_opts {
        int n_image_slots;
        int n_text_slots;
    };
    sam3_ctx *sam3_init_ex(const struct sam3_cache_opts *opts);

    /* Cache bitmask (anonymous enum in sam3.h) */
    #define SAM3_CACHE_IMAGE 1
    #define SAM3_CACHE_TEXT  2

    /* Cache runtime control */
    void sam3_cache_clear(sam3_ctx *ctx, unsigned which);

    struct sam3_cache_stats {
        uint64_t image_hits;
        uint64_t image_misses;
        uint64_t image_evictions;
        uint64_t text_hits;
        uint64_t text_misses;
        uint64_t text_evictions;
    };
    void sam3_cache_stats(const sam3_ctx *ctx,
                          struct sam3_cache_stats *out);

    /* Precache */
    enum sam3_error sam3_precache_image(sam3_ctx *ctx,
                                        const uint8_t *pixels,
                                        int width, int height);
    enum sam3_error sam3_precache_image_file(sam3_ctx *ctx,
                                             const char *path);
    enum sam3_error sam3_precache_text(sam3_ctx *ctx, const char *text);

    /* Persistence */
    enum sam3_error sam3_cache_save_image(sam3_ctx *ctx,
                                          const uint8_t *pixels,
                                          int width, int height,
                                          const char *path);
    enum sam3_error sam3_cache_load_image(sam3_ctx *ctx,
                                          const char *path);
    enum sam3_error sam3_cache_save_text(sam3_ctx *ctx,
                                         const char *text,
                                         const char *path);
    enum sam3_error sam3_cache_load_text(sam3_ctx *ctx,
                                         const char *path);
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd bindings/python && pytest tests/test_cache_ffi.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add bindings/python/sam3/_ffi.py bindings/python/tests/test_cache_ffi.py
git commit -m "bindings/python: declare cache API in cffi cdef"
```

---

## Task 1.2: Add `CacheStats` dataclass and `Model.cache_stats()`

**Files:**
- Modify: `bindings/python/sam3/model.py` (add `CacheStats` dataclass near top; add method on `Model`)
- Test: `bindings/python/tests/test_cache.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `bindings/python/tests/test_cache.py`:

```python
"""Tests for the encoder-cache API."""
import os
import numpy as np
import pytest

import sam3
from sam3.model import CacheStats


def test_cache_stats_dataclass_has_six_fields():
    s = CacheStats(
        image_hits=1, image_misses=2, image_evictions=3,
        text_hits=4, text_misses=5, text_evictions=6,
    )
    assert s.image_hits == 1
    assert s.image_misses == 2
    assert s.image_evictions == 3
    assert s.text_hits == 4
    assert s.text_misses == 5
    assert s.text_evictions == 6


def test_cache_stats_is_frozen():
    s = CacheStats(
        image_hits=0, image_misses=0, image_evictions=0,
        text_hits=0, text_misses=0, text_evictions=0,
    )
    with pytest.raises(Exception):  # FrozenInstanceError
        s.image_hits = 99


def test_cache_stats_zero_on_fresh_model(model_path):
    with sam3.Model(model_path) as m:
        stats = m.cache_stats()
        assert isinstance(stats, CacheStats)
        assert stats.image_hits == 0
        assert stats.image_misses == 0
        assert stats.image_evictions == 0
        assert stats.text_hits == 0
        assert stats.text_misses == 0
        assert stats.text_evictions == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd bindings/python && pytest tests/test_cache.py -v
```

Expected: `ImportError: cannot import name 'CacheStats' from 'sam3.model'`.

- [ ] **Step 3: Add `CacheStats` and `cache_stats()` in `model.py`**

At the top of `bindings/python/sam3/model.py`, **after** `import numpy as np` and **before** the `Result` class, add:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class CacheStats:
    """Hit/miss/eviction counters for the encoder feature caches.

    All counters are cumulative since the context was created or
    ``cache_clear`` was last called.
    """

    image_hits: int
    image_misses: int
    image_evictions: int
    text_hits: int
    text_misses: int
    text_evictions: int
```

Inside the `Model` class, add this method (place it after `profile_reset`):

```python
    def cache_stats(self):
        """Return a CacheStats snapshot of the encoder caches."""
        self._check_open()
        out = ffi.new("struct sam3_cache_stats *")
        lib.sam3_cache_stats(self._ctx, out)
        return CacheStats(
            image_hits=int(out.image_hits),
            image_misses=int(out.image_misses),
            image_evictions=int(out.image_evictions),
            text_hits=int(out.text_hits),
            text_misses=int(out.text_misses),
            text_evictions=int(out.text_evictions),
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd bindings/python && pytest tests/test_cache.py -v
```

Expected: 2 passed, 1 skipped (the `model_path` fixture skips when `SAM3_MODEL_PATH` is unset). If `SAM3_MODEL_PATH` is set, expect 3 passed.

- [ ] **Step 5: Commit**

```bash
git add bindings/python/sam3/model.py bindings/python/tests/test_cache.py
git commit -m "bindings/python: add CacheStats dataclass and Model.cache_stats"
```

---

## Task 1.3: Add `image_cache_slots` / `text_cache_slots` kwargs to `Model.__init__`

**Files:**
- Modify: `bindings/python/sam3/model.py:62-77` (`__init__`)
- Test: `bindings/python/tests/test_cache.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `bindings/python/tests/test_cache.py`:

```python
def test_init_ex_accepts_custom_slot_counts(model_path):
    with sam3.Model(model_path,
                    image_cache_slots=4,
                    text_cache_slots=8) as m:
        # Functional smoke: stats still work.
        stats = m.cache_stats()
        assert stats.image_hits == 0


def test_init_ex_rejects_zero_image_slots():
    with pytest.raises(ValueError, match="image_cache_slots"):
        sam3.Model("/dev/null", image_cache_slots=0)


def test_init_ex_rejects_negative_text_slots():
    with pytest.raises(ValueError, match="text_cache_slots"):
        sam3.Model("/dev/null", text_cache_slots=-1)


def test_default_init_unchanged(model_path):
    """Omitting slot kwargs must preserve sam3_init() path."""
    with sam3.Model(model_path) as m:
        assert m.get_image_size() > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd bindings/python && pytest tests/test_cache.py::test_init_ex_rejects_zero_image_slots -v
```

Expected: `TypeError: Model.__init__() got an unexpected keyword argument 'image_cache_slots'`.

- [ ] **Step 3: Rewrite `Model.__init__` to accept the new kwargs**

Replace lines 62-77 of `bindings/python/sam3/model.py`:

```python
    def __init__(self, model_path, bpe_path=None, *,
                 image_cache_slots=None, text_cache_slots=None):
        self._closed = True

        if (image_cache_slots is not None and image_cache_slots <= 0):
            raise ValueError(
                f"image_cache_slots must be positive, got {image_cache_slots}"
            )
        if (text_cache_slots is not None and text_cache_slots <= 0):
            raise ValueError(
                f"text_cache_slots must be positive, got {text_cache_slots}"
            )

        if image_cache_slots is None and text_cache_slots is None:
            self._ctx = lib.sam3_init()
        else:
            opts = ffi.new("struct sam3_cache_opts *")
            opts.n_image_slots = (0 if image_cache_slots is None
                                  else int(image_cache_slots))
            opts.n_text_slots = (0 if text_cache_slots is None
                                 else int(text_cache_slots))
            self._ctx = lib.sam3_init_ex(opts)

        if self._ctx == ffi.NULL:
            raise Sam3Error(0, "Failed to initialize sam3 context")
        self._closed = False

        try:
            check(lib.sam3_load_model(self._ctx, model_path.encode()))
            if bpe_path is not None:
                check(lib.sam3_load_bpe(self._ctx, bpe_path.encode()))
        except Exception:
            lib.sam3_free(self._ctx)
            self._ctx = ffi.NULL
            self._closed = True
            raise
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd bindings/python && pytest tests/test_cache.py -v
```

Expected: validation tests pass; model-requiring tests skipped without `SAM3_MODEL_PATH`.

- [ ] **Step 5: Commit**

```bash
git add bindings/python/sam3/model.py bindings/python/tests/test_cache.py
git commit -m "bindings/python: accept image_cache_slots/text_cache_slots on Model"
```

---

## Task 1.4: Add `Model.cache_clear(image=, text=)`

**Files:**
- Modify: `bindings/python/sam3/model.py` (add method after `cache_stats`)
- Test: `bindings/python/tests/test_cache.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `bindings/python/tests/test_cache.py`:

```python
def test_cache_clear_defaults_to_both(model_path):
    with sam3.Model(model_path) as m:
        m.cache_clear()  # must not raise


def test_cache_clear_image_only(model_path):
    with sam3.Model(model_path) as m:
        m.cache_clear(image=True)


def test_cache_clear_text_only(model_path):
    with sam3.Model(model_path) as m:
        m.cache_clear(text=True)


def test_cache_clear_both_explicit(model_path):
    with sam3.Model(model_path) as m:
        m.cache_clear(image=True, text=True)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd bindings/python && pytest tests/test_cache.py::test_cache_clear_defaults_to_both -v
```

Expected (with `SAM3_MODEL_PATH` set): `AttributeError: 'Model' object has no attribute 'cache_clear'`. Without the env var: skipped.

- [ ] **Step 3: Add `cache_clear` to `Model`**

In `bindings/python/sam3/model.py`, insert after the `cache_stats` method:

```python
    def cache_clear(self, *, image=False, text=False):
        """Clear encoder caches. Default (no args) clears both.

        Args:
            image: Clear the image encoder cache.
            text:  Clear the text encoder cache.
        """
        self._check_open()
        if not image and not text:
            which = 0  # C: 0 means clear both
        else:
            which = 0
            if image:
                which |= lib.SAM3_CACHE_IMAGE
            if text:
                which |= lib.SAM3_CACHE_TEXT
        lib.sam3_cache_clear(self._ctx, which)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd bindings/python && pytest tests/test_cache.py -v
```

Expected: passes or skips depending on `SAM3_MODEL_PATH`.

- [ ] **Step 5: Commit**

```bash
git add bindings/python/sam3/model.py bindings/python/tests/test_cache.py
git commit -m "bindings/python: add Model.cache_clear"
```

---

## Task 1.5: Add `Model.precache_image` and `Model.precache_text`

**Files:**
- Modify: `bindings/python/sam3/model.py` (add methods after `cache_clear`)
- Test: `bindings/python/tests/test_cache.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `bindings/python/tests/test_cache.py`:

```python
def test_precache_image_file_populates_cache(model_path, test_image_path):
    with sam3.Model(model_path) as m:
        m.precache_image(test_image_path)
        m.set_image(test_image_path)
        stats = m.cache_stats()
        assert stats.image_hits >= 1, \
            f"expected a hit after precache+set_image, got {stats}"


def test_precache_image_ndarray_populates_cache(model_path):
    img_size = 256
    rgb = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    rgb[100:150, 100:150] = 255
    with sam3.Model(model_path) as m:
        m.precache_image(rgb)
        m.set_image(rgb)
        stats = m.cache_stats()
        assert stats.image_hits >= 1


def test_precache_image_rejects_bad_shape(model_path):
    with sam3.Model(model_path) as m:
        with pytest.raises(ValueError, match="Expected.*RGB"):
            m.precache_image(np.zeros((32, 32), dtype=np.uint8))


def test_precache_text_is_noop_without_bpe(model_path):
    """Without a BPE vocab, precache_text still runs but tokenization
    falls back to byte-level — just verify the call returns cleanly."""
    with sam3.Model(model_path) as m:
        m.precache_text("cat")  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd bindings/python && pytest tests/test_cache.py::test_precache_image_rejects_bad_shape -v
```

Expected: `AttributeError: 'Model' object has no attribute 'precache_image'`.

- [ ] **Step 3: Add `precache_image` and `precache_text` to `Model`**

In `bindings/python/sam3/model.py`, insert after `cache_clear`:

```python
    def precache_image(self, image):
        """Populate the image cache without changing current-image state.

        Args:
            image: Either a file path (str) or a numpy array of shape
                   (H, W, 3) with dtype uint8 (RGB).
        """
        self._check_open()
        if isinstance(image, str):
            check(lib.sam3_precache_image_file(self._ctx, image.encode()))
        else:
            image = np.ascontiguousarray(image, dtype=np.uint8)
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    f"Expected (H, W, 3) RGB array, got shape {image.shape}"
                )
            h, w = image.shape[:2]
            ptr = ffi.cast("const uint8_t *", image.ctypes.data)
            check(lib.sam3_precache_image(self._ctx, ptr, w, h))

    def precache_text(self, text):
        """Populate the text cache without setting a pending prompt."""
        self._check_open()
        check(lib.sam3_precache_text(self._ctx, text.encode()))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd bindings/python && pytest tests/test_cache.py -v
```

Expected: all pass (integration tests skipped without env fixtures).

- [ ] **Step 5: Commit**

```bash
git add bindings/python/sam3/model.py bindings/python/tests/test_cache.py
git commit -m "bindings/python: add Model.precache_image and precache_text"
```

---

## Task 1.6: Add `Model.cache_save_image` / `cache_load_image` / `cache_save_text` / `cache_load_text`

**Files:**
- Modify: `bindings/python/sam3/model.py` (add methods after `precache_text`)
- Test: `bindings/python/tests/test_cache.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `bindings/python/tests/test_cache.py`:

```python
def test_cache_save_image_rejects_non_ndarray(model_path):
    with sam3.Model(model_path) as m:
        with pytest.raises(TypeError, match="ndarray"):
            m.cache_save_image("photo.jpg", "out.sam3cache")


def test_cache_save_image_rejects_bad_shape(model_path):
    with sam3.Model(model_path) as m:
        with pytest.raises(ValueError, match="Expected.*RGB"):
            m.cache_save_image(
                np.zeros((32, 32), dtype=np.uint8), "out.sam3cache")


def test_cache_save_load_image_round_trip(model_path, tmp_path):
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    rgb[50:100, 50:100] = 200
    cache_file = str(tmp_path / "img.sam3cache")

    with sam3.Model(model_path) as m:
        m.precache_image(rgb)
        m.cache_save_image(rgb, cache_file)

    assert os.path.isfile(cache_file)

    with sam3.Model(model_path) as m2:
        m2.cache_load_image(cache_file)
        m2.set_image(rgb)
        stats = m2.cache_stats()
        assert stats.image_hits >= 1


def test_cache_save_image_missing_entry_raises(model_path, tmp_path):
    """Saving pixels never precached must surface SAM3_EINVAL."""
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    with sam3.Model(model_path) as m:
        with pytest.raises(sam3.InvalidArgumentError):
            m.cache_save_image(rgb, str(tmp_path / "nope.sam3cache"))


def test_cache_save_load_text_round_trip(model_path, tmp_path):
    cache_file = str(tmp_path / "cat.sam3cache")
    with sam3.Model(model_path) as m:
        m.precache_text("cat")
        m.cache_save_text("cat", cache_file)
    assert os.path.isfile(cache_file)

    with sam3.Model(model_path) as m2:
        m2.cache_load_text(cache_file)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd bindings/python && pytest tests/test_cache.py::test_cache_save_image_rejects_non_ndarray -v
```

Expected: `AttributeError: 'Model' object has no attribute 'cache_save_image'`.

- [ ] **Step 3: Add the four persistence methods**

In `bindings/python/sam3/model.py`, insert after `precache_text`:

```python
    def cache_save_image(self, pixels, path):
        """Save a cached image entry to ``path``.

        Args:
            pixels: numpy array of shape (H, W, 3), dtype uint8 (RGB).
                    File-path strings are rejected; decode them in the
                    caller and pass the resulting array.
            path:   Destination .sam3cache file path.
        """
        self._check_open()
        if not isinstance(pixels, np.ndarray):
            raise TypeError(
                "cache_save_image requires a numpy ndarray for 'pixels', "
                f"got {type(pixels).__name__}. Decode the file first."
            )
        pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
        if pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError(
                f"Expected (H, W, 3) RGB array, got shape {pixels.shape}"
            )
        h, w = pixels.shape[:2]
        ptr = ffi.cast("const uint8_t *", pixels.ctypes.data)
        check(lib.sam3_cache_save_image(self._ctx, ptr, w, h, path.encode()))

    def cache_load_image(self, path):
        """Restore a previously-saved image cache entry from ``path``."""
        self._check_open()
        check(lib.sam3_cache_load_image(self._ctx, path.encode()))

    def cache_save_text(self, text, path):
        """Save a cached text entry to ``path``."""
        self._check_open()
        check(lib.sam3_cache_save_text(
            self._ctx, text.encode(), path.encode()))

    def cache_load_text(self, path):
        """Restore a previously-saved text cache entry from ``path``."""
        self._check_open()
        check(lib.sam3_cache_load_text(self._ctx, path.encode()))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd bindings/python && pytest tests/test_cache.py -v
```

Expected: pure-validation tests pass; model-requiring tests skipped or passed.

- [ ] **Step 5: Commit**

```bash
git add bindings/python/sam3/model.py bindings/python/tests/test_cache.py
git commit -m "bindings/python: add cache_{save,load}_{image,text} to Model"
```

---

## Task 1.7: Re-export `CacheStats` from the top-level `sam3` package

**Files:**
- Modify: `bindings/python/sam3/__init__.py:9` (extend existing `from sam3.model import ...`)
- Test: `bindings/python/tests/test_cache.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `bindings/python/tests/test_cache.py`:

```python
def test_cache_stats_reexported():
    """`sam3.CacheStats` must work without reaching into submodules."""
    assert hasattr(sam3, "CacheStats")
    assert sam3.CacheStats is sam3.model.CacheStats
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd bindings/python && pytest tests/test_cache.py::test_cache_stats_reexported -v
```

Expected: `AttributeError: module 'sam3' has no attribute 'CacheStats'`.

- [ ] **Step 3: Add the re-export**

Edit `bindings/python/sam3/__init__.py`:

- Change line 9 from `from sam3.model import Model, Result  # noqa: F401` to:
  ```python
  from sam3.model import Model, Result, CacheStats  # noqa: F401
  ```
- Add `"CacheStats"` to the `__all__` list, grouped with the other model types (e.g. after `"Result"`).

- [ ] **Step 4: Run test to verify it passes**

```bash
cd bindings/python && pytest tests/test_cache.py -v
```

Expected: all pass/skip as before plus the new re-export test passes.

- [ ] **Step 5: Commit**

```bash
git add bindings/python/sam3/__init__.py bindings/python/tests/test_cache.py
git commit -m "bindings/python: re-export CacheStats from sam3 package"
```

---

# Phase 2 — Rust binding

## Task 2.1: Verify `sam3-sys` picks up the cache API after regen

**Files:**
- Test: `bindings/rust/sam3-sys/tests/cache_ffi.rs` (new file)

bindgen's `allowlist_function("sam3_.*")` already matches the cache API. This task is a smoke test that regen actually produces the symbols.

- [ ] **Step 1: Write the failing test**

Create `bindings/rust/sam3-sys/tests/cache_ffi.rs`:

```rust
//! Smoke tests that the cache API is exposed via sam3-sys.

#[test]
fn cache_clear_is_linkable() {
    // Compile-time reference: if bindgen did not generate
    // `sam3_cache_clear`, this file fails to compile.
    let _fp: unsafe extern "C" fn(*mut sam3_sys::sam3_ctx, std::os::raw::c_uint) =
        sam3_sys::sam3_cache_clear;
}

#[test]
fn cache_stats_struct_and_function_exist() {
    let _ = std::mem::size_of::<sam3_sys::sam3_cache_stats>();
    let _fp: unsafe extern "C" fn(*const sam3_sys::sam3_ctx,
                                  *mut sam3_sys::sam3_cache_stats) =
        sam3_sys::sam3_cache_stats;
}

#[test]
fn cache_opts_struct_has_slot_fields() {
    let o = sam3_sys::sam3_cache_opts {
        n_image_slots: 4,
        n_text_slots: 8,
    };
    assert_eq!(o.n_image_slots, 4);
    assert_eq!(o.n_text_slots, 8);
}

#[test]
fn init_ex_is_linkable() {
    let _fp: unsafe extern "C" fn(*const sam3_sys::sam3_cache_opts)
        -> *mut sam3_sys::sam3_ctx = sam3_sys::sam3_init_ex;
}

#[test]
fn cache_bitmask_constants_have_expected_values() {
    assert_eq!(sam3_sys::SAM3_CACHE_IMAGE, 1);
    assert_eq!(sam3_sys::SAM3_CACHE_TEXT, 2);
}

#[test]
fn precache_functions_are_linkable() {
    let _a: unsafe extern "C" fn(*mut sam3_sys::sam3_ctx,
                                 *const u8,
                                 std::os::raw::c_int,
                                 std::os::raw::c_int)
        -> sam3_sys::sam3_error = sam3_sys::sam3_precache_image;
    let _b: unsafe extern "C" fn(*mut sam3_sys::sam3_ctx,
                                 *const std::os::raw::c_char)
        -> sam3_sys::sam3_error = sam3_sys::sam3_precache_image_file;
    let _c: unsafe extern "C" fn(*mut sam3_sys::sam3_ctx,
                                 *const std::os::raw::c_char)
        -> sam3_sys::sam3_error = sam3_sys::sam3_precache_text;
}

#[test]
fn save_load_functions_are_linkable() {
    let _sv_img: unsafe extern "C" fn(*mut sam3_sys::sam3_ctx,
                                      *const u8,
                                      std::os::raw::c_int,
                                      std::os::raw::c_int,
                                      *const std::os::raw::c_char)
        -> sam3_sys::sam3_error = sam3_sys::sam3_cache_save_image;
    let _ld_img: unsafe extern "C" fn(*mut sam3_sys::sam3_ctx,
                                      *const std::os::raw::c_char)
        -> sam3_sys::sam3_error = sam3_sys::sam3_cache_load_image;
    let _sv_txt: unsafe extern "C" fn(*mut sam3_sys::sam3_ctx,
                                      *const std::os::raw::c_char,
                                      *const std::os::raw::c_char)
        -> sam3_sys::sam3_error = sam3_sys::sam3_cache_save_text;
    let _ld_txt: unsafe extern "C" fn(*mut sam3_sys::sam3_ctx,
                                      *const std::os::raw::c_char)
        -> sam3_sys::sam3_error = sam3_sys::sam3_cache_load_text;
}
```

- [ ] **Step 2: Run the test to verify bindgen is already correct (or diagnose)**

```bash
cd bindings/rust && DYLD_LIBRARY_PATH=../../build cargo test -p sam3-sys cache_ffi
```

Expected: passes. The `build.rs` allowlist already covers these symbols.

If this **fails** with "cannot find function `sam3_cache_clear`", check:
- `bindings/rust/sam3-sys/wrapper.h` includes `<sam3/sam3.h>` ✓
- `cargo clean -p sam3-sys && cargo build -p sam3-sys` to force regen
- The installed libsam3 headers (`include/sam3/sam3.h`) have the cache API

- [ ] **Step 3: Commit**

```bash
git add bindings/rust/sam3-sys/tests/cache_ffi.rs
git commit -m "bindings/rust: smoke test cache API is exposed by sam3-sys"
```

---

## Task 2.2: Add `bitflags` dep to `sam3` crate

**Files:**
- Modify: `bindings/rust/sam3/Cargo.toml:11-13` (append to `[dependencies]`)

- [ ] **Step 1: Edit `Cargo.toml`**

Change `bindings/rust/sam3/Cargo.toml` `[dependencies]` block from:

```toml
[dependencies]
sam3-sys  = { version = "0.1.0", path = "../sam3-sys" }
thiserror = "1"
```

to:

```toml
[dependencies]
sam3-sys  = { version = "0.1.0", path = "../sam3-sys" }
thiserror = "1"
bitflags  = "2"
```

- [ ] **Step 2: Verify it resolves**

```bash
cd bindings/rust && cargo build -p sam3
```

Expected: builds cleanly, `bitflags` appears in `Cargo.lock`.

- [ ] **Step 3: Commit**

```bash
git add bindings/rust/Cargo.lock bindings/rust/sam3/Cargo.toml
git commit -m "bindings/rust: add bitflags 2 dep on sam3 crate"
```

---

## Task 2.3: Create `sam3/src/cache.rs` with `CacheOpts`, `CacheKind`, `CacheStats`

**Files:**
- Create: `bindings/rust/sam3/src/cache.rs`

- [ ] **Step 1: Write the file with inline tests**

Create `bindings/rust/sam3/src/cache.rs`:

```rust
//! Encoder-cache tuning, bitmask, and counter types.

/// Tunables for the encoder feature caches.
///
/// `None` on either slot count selects libsam3's compile-time default
/// (3 image slots, 16 text slots).
#[derive(Debug, Clone, Copy, Default)]
pub struct CacheOpts {
    /// Number of image-encoder cache slots.
    pub image_slots: Option<u32>,
    /// Number of text-encoder cache slots.
    pub text_slots: Option<u32>,
}

bitflags::bitflags! {
    /// Which encoder caches to clear in [`Ctx::cache_clear`](crate::Ctx::cache_clear).
    ///
    /// `CacheKind::empty()` maps to libsam3's "clear both" sentinel.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CacheKind: u32 {
        /// The image encoder cache.
        const IMAGE = 1 << 0;
        /// The text encoder cache.
        const TEXT  = 1 << 1;
    }
}

/// Cumulative hit/miss/eviction counters for both encoder caches.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Image-encoder cache hits.
    pub image_hits: u64,
    /// Image-encoder cache misses.
    pub image_misses: u64,
    /// Image-encoder cache evictions.
    pub image_evictions: u64,
    /// Text-encoder cache hits.
    pub text_hits: u64,
    /// Text-encoder cache misses.
    pub text_misses: u64,
    /// Text-encoder cache evictions.
    pub text_evictions: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_opts_default_is_all_none() {
        let o = CacheOpts::default();
        assert!(o.image_slots.is_none());
        assert!(o.text_slots.is_none());
    }

    #[test]
    fn cache_opts_is_copy() {
        let o1 = CacheOpts { image_slots: Some(4), text_slots: Some(8) };
        let o2 = o1;
        assert_eq!(o1.image_slots, o2.image_slots);
    }

    #[test]
    fn cache_kind_bitflags_or_combines() {
        let both = CacheKind::IMAGE | CacheKind::TEXT;
        assert!(both.contains(CacheKind::IMAGE));
        assert!(both.contains(CacheKind::TEXT));
        assert_eq!(both.bits(), 0b11);
    }

    #[test]
    fn cache_kind_empty_has_zero_bits() {
        assert_eq!(CacheKind::empty().bits(), 0);
    }

    #[test]
    fn cache_stats_default_is_all_zero() {
        let s = CacheStats::default();
        assert_eq!(s.image_hits, 0);
        assert_eq!(s.image_misses, 0);
        assert_eq!(s.image_evictions, 0);
        assert_eq!(s.text_hits, 0);
        assert_eq!(s.text_misses, 0);
        assert_eq!(s.text_evictions, 0);
    }
}
```

- [ ] **Step 2: Run the unit tests**

```bash
cd bindings/rust && DYLD_LIBRARY_PATH=../../build cargo test -p sam3 cache::tests
```

Expected: 5 passed. (Module is not wired into `lib.rs` yet, so the tests run via file-local compilation — they **will** fail at this step. Proceed to Task 2.4 to wire the module.)

Actually — without the module registration in `lib.rs`, `cargo test` won't even compile `cache.rs`. That registration is Task 2.4. So instead:

```bash
cd bindings/rust && cargo build -p sam3
```

Expected: builds, but `cache.rs` is unused. Real test run happens in Task 2.4.

- [ ] **Step 3: Commit**

```bash
git add bindings/rust/sam3/src/cache.rs
git commit -m "bindings/rust: add cache module types (CacheOpts, CacheKind, CacheStats)"
```

---

## Task 2.4: Register the `cache` module and re-export types from `lib.rs`

**Files:**
- Modify: `bindings/rust/sam3/src/lib.rs:23-31` (add module + re-exports)

- [ ] **Step 1: Write the failing test**

Create `bindings/rust/sam3/tests/cache_reexport.rs`:

```rust
//! Verify sam3 crate re-exports the cache types.

#[test]
fn cache_types_are_reachable_from_crate_root() {
    let _: sam3::CacheOpts = sam3::CacheOpts::default();
    let _: sam3::CacheKind = sam3::CacheKind::IMAGE;
    let _: sam3::CacheStats = sam3::CacheStats::default();
}
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd bindings/rust && DYLD_LIBRARY_PATH=../../build cargo test -p sam3 --test cache_reexport
```

Expected: `error[E0433]: failed to resolve: could not find CacheOpts in sam3`.

- [ ] **Step 3: Wire up `lib.rs`**

Add to `bindings/rust/sam3/src/lib.rs`, after the `mod image;` / `pub use image::ImageData;` block and before `mod result;`:

```rust
mod cache;

pub use cache::{CacheKind, CacheOpts, CacheStats};
```

- [ ] **Step 4: Run all cache tests**

```bash
cd bindings/rust && DYLD_LIBRARY_PATH=../../build cargo test -p sam3 cache
```

Expected: 5 unit tests in `cache::tests` plus `cache_types_are_reachable_from_crate_root` all pass.

- [ ] **Step 5: Commit**

```bash
git add bindings/rust/sam3/src/lib.rs bindings/rust/sam3/tests/cache_reexport.rs
git commit -m "bindings/rust: re-export cache types from crate root"
```

---

## Task 2.5: Add `Ctx::new_with_cache_opts`

**Files:**
- Modify: `bindings/rust/sam3/src/ctx.rs:70-81` (add method after `new`)

- [ ] **Step 1: Write the failing test**

Append to the `#[cfg(test)] mod tests` block near the bottom of `bindings/rust/sam3/src/ctx.rs` (after the existing tests, before the closing `}`):

```rust
    #[test]
    fn new_with_cache_opts_default_matches_new() {
        let opts = crate::CacheOpts::default();
        let ctx = Ctx::new_with_cache_opts(&opts)
            .expect("new_with_cache_opts should succeed on defaults");
        drop(ctx);
    }

    #[test]
    fn new_with_cache_opts_accepts_custom_slots() {
        let opts = crate::CacheOpts {
            image_slots: Some(4),
            text_slots: Some(8),
        };
        let ctx = Ctx::new_with_cache_opts(&opts).unwrap();
        drop(ctx);
    }

    #[test]
    fn new_with_cache_opts_rejects_u32_overflow() {
        let opts = crate::CacheOpts {
            image_slots: Some(u32::MAX),
            text_slots: None,
        };
        let err = Ctx::new_with_cache_opts(&opts).unwrap_err();
        assert!(matches!(err, Error::Invalid));
    }
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd bindings/rust && DYLD_LIBRARY_PATH=../../build cargo test -p sam3 new_with_cache_opts
```

Expected: `error[E0599]: no function or associated item named 'new_with_cache_opts' found for struct Ctx`.

- [ ] **Step 3: Add the method**

Insert into `impl Ctx { ... }` in `bindings/rust/sam3/src/ctx.rs`, directly after the existing `new()` method:

```rust
    /// Create a new SAM3 context with custom cache slot tuning.
    ///
    /// `CacheOpts::default()` (both slots `None`) is equivalent to
    /// [`Ctx::new`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if either slot count exceeds
    /// [`i32::MAX`] (libsam3 stores slot counts as `int`).
    pub fn new_with_cache_opts(opts: &crate::CacheOpts) -> Result<Self> {
        fn to_int(v: Option<u32>) -> Result<i32> {
            match v {
                None => Ok(0),
                Some(n) if n <= i32::MAX as u32 => Ok(n as i32),
                Some(_) => Err(Error::Invalid),
            }
        }
        let raw_opts = sys::sam3_cache_opts {
            n_image_slots: to_int(opts.image_slots)?,
            n_text_slots: to_int(opts.text_slots)?,
        };
        // SAFETY: sam3_init_ex dereferences `opts` during the call only;
        // &raw_opts is a valid, non-null pointer for that duration.
        let raw = unsafe { sys::sam3_init_ex(&raw_opts) };
        NonNull::new(raw)
            .map(|raw| Ctx {
                raw,
                _not_send_sync: PhantomData,
            })
            .ok_or(Error::NoMemory)
    }
```

- [ ] **Step 4: Run the tests**

```bash
cd bindings/rust && DYLD_LIBRARY_PATH=../../build cargo test -p sam3 ctx::tests
```

Expected: all existing `ctx` tests plus the three new ones pass.

- [ ] **Step 5: Commit**

```bash
git add bindings/rust/sam3/src/ctx.rs
git commit -m "bindings/rust: add Ctx::new_with_cache_opts"
```

---

## Task 2.6: Add `Ctx::cache_clear` and `Ctx::cache_stats`

**Files:**
- Modify: `bindings/rust/sam3/src/ctx.rs` (add methods in `impl Ctx`)

- [ ] **Step 1: Write the failing test**

Append inside the `#[cfg(test)] mod tests` block in `ctx.rs`:

```rust
    #[test]
    fn cache_clear_on_fresh_ctx_is_noop() {
        let mut ctx = Ctx::new().unwrap();
        ctx.cache_clear(crate::CacheKind::empty());
        ctx.cache_clear(crate::CacheKind::IMAGE);
        ctx.cache_clear(crate::CacheKind::TEXT);
        ctx.cache_clear(crate::CacheKind::IMAGE | crate::CacheKind::TEXT);
    }

    #[test]
    fn cache_stats_fresh_ctx_is_all_zero() {
        let ctx = Ctx::new().unwrap();
        let s = ctx.cache_stats();
        assert_eq!(s, crate::CacheStats::default());
    }
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd bindings/rust && DYLD_LIBRARY_PATH=../../build cargo test -p sam3 cache_clear_on_fresh_ctx_is_noop
```

Expected: `error[E0599]: no method named cache_clear found for struct Ctx`.

- [ ] **Step 3: Add the methods**

Insert into `impl Ctx { ... }` in `ctx.rs`, directly after `new_with_cache_opts`:

```rust
    /// Clear encoder caches.
    ///
    /// `CacheKind::empty()` clears both (matches libsam3's 0-means-both).
    pub fn cache_clear(&mut self, which: crate::CacheKind) {
        // SAFETY: self.raw is a non-null sam3_ctx; sam3_cache_clear
        // dereferences it only for the duration of the call.
        unsafe { sys::sam3_cache_clear(self.raw.as_ptr(), which.bits()) }
    }

    /// Return a snapshot of the encoder cache counters.
    pub fn cache_stats(&self) -> crate::CacheStats {
        let mut raw = sys::sam3_cache_stats::default();
        // SAFETY: self.raw is a non-null sam3_ctx; `raw` points to a
        // stack-allocated POD struct valid for the call duration.
        unsafe { sys::sam3_cache_stats(self.raw.as_ptr(), &mut raw) };
        crate::CacheStats {
            image_hits: raw.image_hits,
            image_misses: raw.image_misses,
            image_evictions: raw.image_evictions,
            text_hits: raw.text_hits,
            text_misses: raw.text_misses,
            text_evictions: raw.text_evictions,
        }
    }
```

- [ ] **Step 4: Run the tests**

```bash
cd bindings/rust && DYLD_LIBRARY_PATH=../../build cargo test -p sam3 ctx::tests
```

Expected: passes including the two new tests.

- [ ] **Step 5: Commit**

```bash
git add bindings/rust/sam3/src/ctx.rs
git commit -m "bindings/rust: add Ctx::cache_clear and Ctx::cache_stats"
```

---

## Task 2.7: Add precache methods (`precache_image_rgb`, `precache_image`, `precache_image_file`, `precache_text`)

**Files:**
- Modify: `bindings/rust/sam3/src/ctx.rs` (add methods in `impl Ctx`)

- [ ] **Step 1: Write the failing test**

Append inside the `#[cfg(test)] mod tests` block in `ctx.rs`:

```rust
    #[test]
    fn precache_image_rgb_rejects_short_buffer() {
        let mut ctx = Ctx::new().unwrap();
        let err = ctx.precache_image_rgb(&[0; 10], 10, 10).unwrap_err();
        assert!(matches!(err, Error::Invalid));
    }

    #[test]
    fn precache_image_rgb_rejects_dimension_overflow() {
        let mut ctx = Ctx::new().unwrap();
        let err = ctx
            .precache_image_rgb(&[0; 10], u32::MAX, u32::MAX)
            .unwrap_err();
        assert!(matches!(err, Error::Invalid));
    }

    #[test]
    fn precache_text_rejects_interior_nul() {
        let mut ctx = Ctx::new().unwrap();
        let err = ctx.precache_text("hello\0world").unwrap_err();
        assert!(matches!(err, Error::Invalid));
    }
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd bindings/rust && DYLD_LIBRARY_PATH=../../build cargo test -p sam3 precache_image_rgb_rejects_short_buffer
```

Expected: `error[E0599]: no method named precache_image_rgb found for struct Ctx`.

- [ ] **Step 3: Add the four methods**

Insert into `impl Ctx { ... }` in `ctx.rs`, directly after `cache_stats`:

```rust
    /// Populate the image cache from a raw RGB buffer.
    ///
    /// # Errors
    ///
    /// Same contract as [`Ctx::set_image_rgb`] — [`Error::Invalid`] on
    /// short buffers or `width * height * 3` overflow.
    pub fn precache_image_rgb(
        &mut self, pixels: &[u8], width: u32, height: u32,
    ) -> Result<()> {
        let need = crate::ImageData {
            pixels,
            width,
            height,
        }
        .required_len()
        .ok_or(Error::Invalid)?;
        if pixels.len() < need {
            return Err(Error::Invalid);
        }
        // SAFETY: self.raw is a non-null sam3_ctx; pixels has at least
        // width*height*3 bytes (verified above) and is not retained past
        // the call.
        unsafe {
            crate::error::check(sys::sam3_precache_image(
                self.raw.as_ptr(),
                pixels.as_ptr(),
                width as i32,
                height as i32,
            ))
        }
    }

    /// Populate the image cache from an [`ImageData`](crate::ImageData).
    pub fn precache_image(&mut self, img: &crate::ImageData<'_>) -> Result<()> {
        self.precache_image_rgb(img.pixels, img.width, img.height)
    }

    /// Populate the image cache by loading a PNG/JPEG/BMP file.
    pub fn precache_image_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let c = path_to_cstring(path.as_ref())?;
        // SAFETY: self.raw is a non-null sam3_ctx; c is a NUL-terminated
        // path string not retained beyond the call.
        unsafe {
            crate::error::check(sys::sam3_precache_image_file(
                self.raw.as_ptr(),
                c.as_ptr(),
            ))
        }
    }

    /// Populate the text cache by tokenizing and encoding ``text``.
    ///
    /// Requires a BPE vocab loaded via [`load_bpe`](Self::load_bpe).
    pub fn precache_text(&mut self, text: &str) -> Result<()> {
        let c = CString::new(text).map_err(|_| Error::Invalid)?;
        // SAFETY: self.raw is a non-null sam3_ctx; c is a NUL-terminated
        // string not retained beyond the call.
        unsafe {
            crate::error::check(sys::sam3_precache_text(
                self.raw.as_ptr(),
                c.as_ptr(),
            ))
        }
    }
```

- [ ] **Step 4: Run the tests**

```bash
cd bindings/rust && DYLD_LIBRARY_PATH=../../build cargo test -p sam3 ctx::tests
```

Expected: all tests pass including the three new ones.

- [ ] **Step 5: Commit**

```bash
git add bindings/rust/sam3/src/ctx.rs
git commit -m "bindings/rust: add Ctx::precache_image*/precache_text"
```

---

## Task 2.8: Add save/load methods

**Files:**
- Modify: `bindings/rust/sam3/src/ctx.rs` (add methods in `impl Ctx`)

- [ ] **Step 1: Write the failing unit tests**

Append inside the `#[cfg(test)] mod tests` block in `ctx.rs`:

```rust
    #[test]
    fn cache_save_image_rgb_rejects_short_buffer() {
        let mut ctx = Ctx::new().unwrap();
        let err = ctx
            .cache_save_image_rgb(&[0; 10], 10, 10, "/tmp/sam3-nope.sam3cache")
            .unwrap_err();
        assert!(matches!(err, Error::Invalid));
    }

    #[test]
    fn cache_save_text_rejects_interior_nul_text() {
        let mut ctx = Ctx::new().unwrap();
        let err = ctx
            .cache_save_text("bad\0text", "/tmp/sam3-nope.sam3cache")
            .unwrap_err();
        assert!(matches!(err, Error::Invalid));
    }

    #[test]
    fn cache_load_image_missing_file_returns_io_error() {
        let mut ctx = Ctx::new().unwrap();
        let err = ctx
            .cache_load_image("/nonexistent/path.sam3cache")
            .unwrap_err();
        assert!(matches!(err, Error::Io | Error::Invalid),
                "got {err:?}");
    }
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd bindings/rust && DYLD_LIBRARY_PATH=../../build cargo test -p sam3 cache_save_image_rgb_rejects_short_buffer
```

Expected: `error[E0599]: no method named cache_save_image_rgb`.

- [ ] **Step 3: Add the five methods**

Insert into `impl Ctx { ... }` in `ctx.rs`, directly after `precache_text`:

```rust
    /// Serialize a cached image entry (looked up by pixel hash) to ``out_path``.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] on short buffers, overflow, or when no cache
    /// entry matches the given pixels. [`Error::Io`] on write failure.
    pub fn cache_save_image_rgb<P: AsRef<Path>>(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        out_path: P,
    ) -> Result<()> {
        let need = crate::ImageData {
            pixels,
            width,
            height,
        }
        .required_len()
        .ok_or(Error::Invalid)?;
        if pixels.len() < need {
            return Err(Error::Invalid);
        }
        let c = path_to_cstring(out_path.as_ref())?;
        // SAFETY: self.raw is a non-null sam3_ctx; pixels has >= need
        // bytes (verified above); c is NUL-terminated; none are retained.
        unsafe {
            crate::error::check(sys::sam3_cache_save_image(
                self.raw.as_ptr(),
                pixels.as_ptr(),
                width as i32,
                height as i32,
                c.as_ptr(),
            ))
        }
    }

    /// Serialize a cached image entry via [`ImageData`](crate::ImageData).
    pub fn cache_save_image<P: AsRef<Path>>(
        &mut self,
        img: &crate::ImageData<'_>,
        out_path: P,
    ) -> Result<()> {
        self.cache_save_image_rgb(img.pixels, img.width, img.height, out_path)
    }

    /// Restore a previously-saved image cache entry from ``path``.
    ///
    /// # Errors
    ///
    /// [`Error::Model`] if the file's model signature does not match the
    /// currently loaded model. [`Error::Io`] on read failure.
    pub fn cache_load_image<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let c = path_to_cstring(path.as_ref())?;
        // SAFETY: self.raw is a non-null sam3_ctx; c is NUL-terminated.
        unsafe {
            crate::error::check(sys::sam3_cache_load_image(
                self.raw.as_ptr(),
                c.as_ptr(),
            ))
        }
    }

    /// Serialize a cached text entry (looked up by tokenization of ``text``)
    /// to ``out_path``.
    pub fn cache_save_text<P: AsRef<Path>>(
        &mut self,
        text: &str,
        out_path: P,
    ) -> Result<()> {
        let c_text = CString::new(text).map_err(|_| Error::Invalid)?;
        let c_path = path_to_cstring(out_path.as_ref())?;
        // SAFETY: self.raw is a non-null sam3_ctx; both c strings are
        // NUL-terminated and not retained past the call.
        unsafe {
            crate::error::check(sys::sam3_cache_save_text(
                self.raw.as_ptr(),
                c_text.as_ptr(),
                c_path.as_ptr(),
            ))
        }
    }

    /// Restore a previously-saved text cache entry from ``path``.
    pub fn cache_load_text<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let c = path_to_cstring(path.as_ref())?;
        // SAFETY: self.raw is a non-null sam3_ctx; c is NUL-terminated.
        unsafe {
            crate::error::check(sys::sam3_cache_load_text(
                self.raw.as_ptr(),
                c.as_ptr(),
            ))
        }
    }
```

- [ ] **Step 4: Run the unit tests**

```bash
cd bindings/rust && DYLD_LIBRARY_PATH=../../build cargo test -p sam3 ctx::tests
```

Expected: all existing + new unit tests pass.

- [ ] **Step 5: Commit**

```bash
git add bindings/rust/sam3/src/ctx.rs
git commit -m "bindings/rust: add Ctx::cache_{save,load}_{image,text}"
```

---

## Task 2.9: Add integration tests for cache round-trips

**Files:**
- Create: `bindings/rust/sam3/tests/cache_integration.rs`

- [ ] **Step 1: Write the integration tests**

Create `bindings/rust/sam3/tests/cache_integration.rs`:

```rust
//! End-to-end integration tests for the encoder cache.
//!
//! Gated by SAM3_MODEL_PATH and SAM3_TEST_IMAGE (matching
//! tests/integration.rs). Skipped silently when either is unset.

use sam3::{CacheKind, CacheOpts, CacheStats, Ctx};

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
fn cache_stats_after_fresh_load_is_zero() {
    let model = match env_or_skip("SAM3_MODEL_PATH") {
        Some(v) => v,
        None => return,
    };
    let mut ctx = Ctx::new().unwrap();
    ctx.load_model(&model).unwrap();
    assert_eq!(ctx.cache_stats(), CacheStats::default());
}

#[test]
fn precache_then_set_image_produces_a_hit() {
    let model = match env_or_skip("SAM3_MODEL_PATH") {
        Some(v) => v,
        None => return,
    };
    let image = match env_or_skip("SAM3_TEST_IMAGE") {
        Some(v) => v,
        None => return,
    };

    let mut ctx = Ctx::new().unwrap();
    ctx.load_model(&model).unwrap();
    ctx.precache_image_file(&image).unwrap();
    ctx.set_image_file(&image).unwrap();

    let stats = ctx.cache_stats();
    assert!(stats.image_hits >= 1, "expected at least one hit, got {stats:?}");
}

#[test]
fn cache_clear_image_resets_entries() {
    let model = match env_or_skip("SAM3_MODEL_PATH") {
        Some(v) => v,
        None => return,
    };
    let image = match env_or_skip("SAM3_TEST_IMAGE") {
        Some(v) => v,
        None => return,
    };

    let mut ctx = Ctx::new().unwrap();
    ctx.load_model(&model).unwrap();
    ctx.precache_image_file(&image).unwrap();
    ctx.cache_clear(CacheKind::IMAGE);
    ctx.set_image_file(&image).unwrap();

    let stats = ctx.cache_stats();
    // After clear, the first set_image is a miss.
    assert!(stats.image_misses >= 1);
}

#[test]
fn save_load_image_round_trip_across_contexts() {
    let model = match env_or_skip("SAM3_MODEL_PATH") {
        Some(v) => v,
        None => return,
    };

    // Synthetic RGB image — content doesn't matter, only that the same
    // pixel buffer is used for precache, save, and post-load lookup.
    let w: u32 = 256;
    let h: u32 = 256;
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    for (i, p) in pixels.iter_mut().enumerate() {
        *p = (i % 251) as u8;
    }

    let tmp_dir = std::env::temp_dir().join(format!(
        "sam3-cache-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&tmp_dir).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _guard = Cleanup(tmp_dir.clone());
    let cache_path = tmp_dir.join("img.sam3cache");

    // Source context: precache and save.
    let mut a = Ctx::new().unwrap();
    a.load_model(&model).unwrap();
    a.precache_image_rgb(&pixels, w, h).unwrap();
    a.cache_save_image_rgb(&pixels, w, h, &cache_path).unwrap();
    drop(a);

    assert!(cache_path.is_file(), "save did not write {:?}", cache_path);

    // Destination context: load and verify a hit on the same pixels.
    let mut b = Ctx::new().unwrap();
    b.load_model(&model).unwrap();
    b.cache_load_image(&cache_path).unwrap();
    b.set_image_rgb(&pixels, w, h).unwrap();
    let stats = b.cache_stats();
    assert!(
        stats.image_hits >= 1,
        "expected hit after load+set_image_rgb, got {stats:?}"
    );
}

#[test]
fn save_load_text_round_trip_across_contexts() {
    let model = match env_or_skip("SAM3_MODEL_PATH") {
        Some(v) => v,
        None => return,
    };
    let bpe = match env_or_skip("SAM3_BPE_PATH") {
        Some(v) => v,
        None => return,
    };

    let tmp_dir = std::env::temp_dir().join(format!(
        "sam3-txtcache-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&tmp_dir).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _guard = Cleanup(tmp_dir.clone());

    let cache_path = tmp_dir.join("cat.sam3cache");

    let mut a = Ctx::new().unwrap();
    a.load_model(&model).unwrap();
    a.load_bpe(&bpe).unwrap();
    a.precache_text("cat").unwrap();
    a.cache_save_text("cat", &cache_path).unwrap();
    drop(a);

    assert!(cache_path.is_file());

    let mut b = Ctx::new().unwrap();
    b.load_model(&model).unwrap();
    b.load_bpe(&bpe).unwrap();
    b.cache_load_text(&cache_path).unwrap();
}

#[test]
fn init_with_cache_opts_functions_end_to_end() {
    let model = match env_or_skip("SAM3_MODEL_PATH") {
        Some(v) => v,
        None => return,
    };
    let opts = CacheOpts {
        image_slots: Some(4),
        text_slots: Some(8),
    };
    let mut ctx = Ctx::new_with_cache_opts(&opts).unwrap();
    ctx.load_model(&model).unwrap();
    assert_eq!(ctx.cache_stats(), CacheStats::default());
}
```

- [ ] **Step 2: Run the integration tests**

```bash
cd bindings/rust && DYLD_LIBRARY_PATH=../../build cargo test -p sam3 --test cache_integration
```

Expected: without env vars, all tests skip silently. With `SAM3_MODEL_PATH` set, six tests are eligible:

- `cache_stats_after_fresh_load_is_zero` — passes.
- `precache_then_set_image_produces_a_hit` — also needs `SAM3_TEST_IMAGE`, else skips.
- `cache_clear_image_resets_entries` — also needs `SAM3_TEST_IMAGE`, else skips.
- `save_load_image_round_trip_across_contexts` — passes (uses synthetic pixels, only needs `SAM3_MODEL_PATH`).
- `save_load_text_round_trip_across_contexts` — also needs `SAM3_BPE_PATH`, else skips.
- `init_with_cache_opts_functions_end_to_end` — passes.

- [ ] **Step 3: Commit**

```bash
git add bindings/rust/sam3/tests/cache_integration.rs
git commit -m "bindings/rust: integration tests for cache hit/miss and round-trip"
```

---

## Task 3: Final verification

- [ ] **Step 1: Full Python test suite**

```bash
cd bindings/python && pytest -v
```

Expected: all pre-existing tests still pass; new `test_cache.py` and `test_cache_ffi.py` pass or skip.

- [ ] **Step 2: Full Rust test suite**

```bash
cd bindings/rust && DYLD_LIBRARY_PATH=../../build cargo test --workspace
```

Expected: all tests pass or skip.

- [ ] **Step 3: Lint / format (match existing conventions)**

```bash
cd bindings/rust && cargo fmt --check && cargo clippy --workspace -- -D warnings
```

Expected: no warnings.

- [ ] **Step 4: Push a single tag-style commit if desired, or stop here**

The series of small commits above is the intended history. No squash.
