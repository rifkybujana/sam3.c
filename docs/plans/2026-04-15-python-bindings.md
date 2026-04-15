# Python Bindings Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a pip-installable Python package (`sam3`) that wraps libsam3 via cffi ABI mode, providing a Pythonic `Model` class with numpy array I/O.

**Architecture:** cffi `dlopen()` loads `libsam3.{so,dylib}` bundled in the wheel. A manually maintained `cdef` block declares the public C API. The high-level `Model` class wraps context lifecycle, converts between numpy arrays and C pointers, and maps C error codes to Python exceptions.

**Tech Stack:** Python 3.9+, cffi, numpy, setuptools, cmake (build time)

---

### Task 1: Expose utility functions in public header

Two functions the Python bindings need (`sam3_error_str`, `sam3_log_set_level`) are currently internal-only. They must be declared in the public header so the shared library exports them and the cffi cdef can reference them.

**Files:**
- Modify: `include/sam3/sam3.h:153` (add declarations before profiling section)

**Step 1: Add declarations to sam3.h**

Add after `sam3_version()` (line 153), before the profiling section:

```c
/*
 * sam3_error_str - Return a human-readable string for an error code.
 *
 * @err: Error code to describe.
 *
 * Returns a static string. Never returns NULL.
 */
const char *sam3_error_str(enum sam3_error err);

/*
 * sam3_log_set_level - Set the minimum log level.
 *
 * @level: Messages below this level are suppressed.
 *         Default is SAM3_LOG_INFO.
 */
void sam3_log_set_level(int level);
```

Note: Use `int level` instead of `enum sam3_log_level` because the enum is defined in `src/util/log.h` (internal). The int representation is ABI-compatible. Alternatively, move `enum sam3_log_level` into `sam3_types.h` — that's cleaner. Prefer the enum approach:

Add to `include/sam3/sam3_types.h` after line 33 (after `sam3_error` enum):

```c
/* Log severity levels. */
enum sam3_log_level {
	SAM3_LOG_DEBUG,
	SAM3_LOG_INFO,
	SAM3_LOG_WARN,
	SAM3_LOG_ERROR,
};
```

Then in `sam3.h`, use `enum sam3_log_level` directly. Remove the duplicate definition from `src/util/log.h` and replace with `#include "sam3/sam3_types.h"`.

**Step 2: Verify the existing log.h includes still work**

Run: `cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug && make -j$(nproc) 2>&1 | head -20`
Expected: Clean build, no errors.

**Step 3: Run tests**

Run: `cd build && ctest --output-on-failure`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add include/sam3/sam3.h include/sam3/sam3_types.h src/util/log.h
git commit -m "api: expose sam3_error_str and sam3_log_set_level in public headers"
```

---

### Task 2: Add shared library build option to CMakeLists.txt

**Files:**
- Modify: `CMakeLists.txt:119` (change static library to support shared)

**Step 1: Add SAM3_SHARED option and conditional library type**

After the options block (line 17), add:

```cmake
option(SAM3_SHARED "Build shared library" OFF)
```

Replace line 119 (`add_library(sam3 STATIC ${SAM3_SOURCES})`) with:

```cmake
if(SAM3_SHARED)
	add_library(sam3 SHARED ${SAM3_SOURCES})
	set_target_properties(sam3 PROPERTIES
		POSITION_INDEPENDENT_CODE ON
		C_VISIBILITY_PRESET hidden
	)
	# Export only sam3_ prefixed public symbols
	if(APPLE)
		set_target_properties(sam3 PROPERTIES
			LINK_FLAGS "-exported_symbols_list ${CMAKE_SOURCE_DIR}/python/exports.txt"
		)
	endif()
else()
	add_library(sam3 STATIC ${SAM3_SOURCES})
endif()
```

**Step 2: Create exports file for macOS**

Create `python/exports.txt`:

```
_sam3_init
_sam3_free
_sam3_load_model
_sam3_load_bpe
_sam3_set_image
_sam3_set_image_file
_sam3_set_prompt_space
_sam3_set_text
_sam3_segment
_sam3_result_free
_sam3_get_image_size
_sam3_version
_sam3_error_str
_sam3_log_set_level
_sam3_profile_enable
_sam3_profile_disable
_sam3_profile_report
_sam3_profile_reset
_sam3_dump_tensors
```

**Step 3: Test shared library build**

Run: `mkdir -p build-shared && cd build-shared && cmake .. -DSAM3_SHARED=ON -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)`
Expected: Produces `libsam3.dylib` (macOS) or `libsam3.so` (Linux).

**Step 4: Verify symbols are exported**

Run: `nm -gU build-shared/libsam3.dylib | grep sam3_`
Expected: All functions from exports.txt are listed.

**Step 5: Commit**

```bash
git add CMakeLists.txt python/exports.txt
git commit -m "build: add SAM3_SHARED option for shared library builds"
```

---

### Task 3: Create the cffi FFI layer

**Files:**
- Create: `python/sam3/__init__.py`
- Create: `python/sam3/_ffi.py`
- Create: `python/sam3/_lib.py`

**Step 1: Create `python/sam3/_lib.py`**

```python
"""
Locate and load the bundled libsam3 shared library.
"""
import os
import sys

import cffi

ffi = cffi.FFI()


def _find_library():
    pkg_dir = os.path.dirname(os.path.abspath(__file__))

    if sys.platform == "darwin":
        name = "libsam3.dylib"
    elif sys.platform == "win32":
        name = "sam3.dll"
    else:
        name = "libsam3.so"

    path = os.path.join(pkg_dir, name)
    if not os.path.isfile(path):
        raise OSError(
            f"libsam3 not found at {path}. "
            "Reinstall the sam3 package or build with SAM3_SHARED=ON."
        )
    return path


lib = ffi.dlopen(_find_library())
```

**Step 2: Create `python/sam3/_ffi.py`**

```python
"""
cffi type declarations for the sam3 C API.

This is a manually maintained subset of include/sam3/sam3.h and
include/sam3/sam3_types.h. Update this file when the C API changes.
"""
from sam3._lib import ffi

ffi.cdef("""
    /* Error codes */
    enum sam3_error {
        SAM3_OK       =  0,
        SAM3_EINVAL   = -1,
        SAM3_ENOMEM   = -2,
        SAM3_EIO      = -3,
        SAM3_EBACKEND = -4,
        SAM3_EMODEL   = -5,
        SAM3_EDTYPE   = -6,
    };

    /* Prompt types */
    enum sam3_prompt_type {
        SAM3_PROMPT_POINT,
        SAM3_PROMPT_BOX,
        SAM3_PROMPT_MASK,
        SAM3_PROMPT_TEXT,
    };

    /* Log levels */
    enum sam3_log_level {
        SAM3_LOG_DEBUG,
        SAM3_LOG_INFO,
        SAM3_LOG_WARN,
        SAM3_LOG_ERROR,
    };

    /* Structs */
    struct sam3_point {
        float x;
        float y;
        int   label;
    };

    struct sam3_box {
        float x1;
        float y1;
        float x2;
        float y2;
    };

    struct sam3_prompt {
        enum sam3_prompt_type type;
        union {
            struct sam3_point point;
            struct sam3_box   box;
            struct {
                const float *data;
                int          width;
                int          height;
            } mask;
            const char *text;
        };
    };

    struct sam3_result {
        float *masks;
        float *iou_scores;
        int    n_masks;
        int    mask_height;
        int    mask_width;
        int    iou_valid;
        float *boxes;
        int    boxes_valid;
        int    best_mask;
    };

    /* Opaque context */
    typedef struct sam3_ctx sam3_ctx;

    /* Lifecycle */
    sam3_ctx *sam3_init(void);
    void sam3_free(sam3_ctx *ctx);

    /* Model loading */
    enum sam3_error sam3_load_model(sam3_ctx *ctx, const char *path);
    enum sam3_error sam3_load_bpe(sam3_ctx *ctx, const char *path);

    /* Image input */
    enum sam3_error sam3_set_image(sam3_ctx *ctx, const uint8_t *pixels,
                                   int width, int height);
    enum sam3_error sam3_set_image_file(sam3_ctx *ctx, const char *path);
    void sam3_set_prompt_space(sam3_ctx *ctx, int width, int height);

    /* Text prompt (async encoding) */
    enum sam3_error sam3_set_text(sam3_ctx *ctx, const char *text);

    /* Inference */
    enum sam3_error sam3_segment(sam3_ctx *ctx,
                                 const struct sam3_prompt *prompts,
                                 int n_prompts,
                                 struct sam3_result *result);
    void sam3_result_free(struct sam3_result *result);

    /* Queries */
    int sam3_get_image_size(const sam3_ctx *ctx);
    const char *sam3_version(void);

    /* Utilities */
    const char *sam3_error_str(enum sam3_error err);
    void sam3_log_set_level(enum sam3_log_level level);

    /* Profiling (no-op if not compiled with SAM3_HAS_PROFILE) */
    enum sam3_error sam3_profile_enable(sam3_ctx *ctx);
    void sam3_profile_disable(sam3_ctx *ctx);
    void sam3_profile_report(sam3_ctx *ctx);
    void sam3_profile_reset(sam3_ctx *ctx);
""")
```

**Step 3: Create `python/sam3/__init__.py`**

```python
"""SAM3 — Python bindings for the sam3 inference engine."""
from sam3._lib import ffi, lib  # noqa: F401
import sam3._ffi  # noqa: F401  — registers cdef

from sam3.model import Model, Result, Sam3Error  # noqa: F401

__all__ = ["Model", "Result", "Sam3Error"]


def version():
    """Return the sam3 library version string."""
    return ffi.string(lib.sam3_version()).decode()


_LOG_LEVELS = {
    "debug": lib.SAM3_LOG_DEBUG,
    "info": lib.SAM3_LOG_INFO,
    "warn": lib.SAM3_LOG_WARN,
    "error": lib.SAM3_LOG_ERROR,
}


def set_log_level(level):
    """Set the minimum log level ('debug', 'info', 'warn', 'error')."""
    if isinstance(level, str):
        level = _LOG_LEVELS[level.lower()]
    lib.sam3_log_set_level(level)
```

**Step 4: Commit**

```bash
git add python/sam3/__init__.py python/sam3/_ffi.py python/sam3/_lib.py
git commit -m "python: add cffi FFI layer and library loader"
```

---

### Task 4: Create error exception hierarchy

**Files:**
- Create: `python/sam3/errors.py`
- Modify: `python/sam3/__init__.py` (update imports)

**Step 1: Create `python/sam3/errors.py`**

```python
"""Exception hierarchy mapping sam3 C error codes to Python exceptions."""
from sam3._lib import ffi, lib


class Sam3Error(Exception):
    """Base exception for all sam3 errors."""

    def __init__(self, code, message=None):
        self.code = code
        if message is None:
            message = ffi.string(lib.sam3_error_str(code)).decode()
        super().__init__(message)


class InvalidArgumentError(Sam3Error):
    """SAM3_EINVAL: Invalid argument."""
    pass


class OutOfMemoryError(Sam3Error):
    """SAM3_ENOMEM: Out of memory."""
    pass


class IOError(Sam3Error):
    """SAM3_EIO: I/O error."""
    pass


class BackendError(Sam3Error):
    """SAM3_EBACKEND: Backend initialization failed."""
    pass


class ModelError(Sam3Error):
    """SAM3_EMODEL: Model format error."""
    pass


class DtypeError(Sam3Error):
    """SAM3_EDTYPE: Unsupported or mismatched dtype."""
    pass


_ERROR_MAP = {
    -1: InvalidArgumentError,
    -2: OutOfMemoryError,
    -3: IOError,
    -4: BackendError,
    -5: ModelError,
    -6: DtypeError,
}


def check(code):
    """Check a sam3 return code; raise appropriate exception if non-zero."""
    if code != 0:
        cls = _ERROR_MAP.get(code, Sam3Error)
        raise cls(code)
```

**Step 2: Update `__init__.py` imports**

Replace the Sam3Error import line with:

```python
from sam3.errors import (  # noqa: F401
    Sam3Error, InvalidArgumentError, OutOfMemoryError,
    IOError, BackendError, ModelError, DtypeError,
)
from sam3.model import Model, Result  # noqa: F401

__all__ = [
    "Model", "Result", "Sam3Error",
    "InvalidArgumentError", "OutOfMemoryError", "IOError",
    "BackendError", "ModelError", "DtypeError",
    "version", "set_log_level",
]
```

**Step 3: Commit**

```bash
git add python/sam3/errors.py python/sam3/__init__.py
git commit -m "python: add exception hierarchy for sam3 error codes"
```

---

### Task 5: Create the Model class

**Files:**
- Create: `python/sam3/model.py`

**Step 1: Create `python/sam3/model.py`**

```python
"""High-level Pythonic interface to sam3 inference."""
import numpy as np

from sam3._lib import ffi, lib
from sam3.errors import check


class Result:
    """Segmentation result with numpy array views of C-allocated data.

    Attributes:
        masks:      np.ndarray of shape (n_masks, H, W), float32.
        iou_scores: np.ndarray of shape (n_masks,), float32.
                    Only meaningful if iou_valid is True.
        boxes:      np.ndarray of shape (n_masks, 4), float32 (xyxy).
                    Only meaningful if boxes_valid is True.
        best_mask:  int, stability-selected mask index (-1 if N/A).
        iou_valid:  bool, whether iou_scores are model-predicted.
        boxes_valid: bool, whether boxes are computed.
    """

    def __init__(self, c_result):
        self._c = c_result
        n = c_result.n_masks
        h = c_result.mask_height
        w = c_result.mask_width

        # Zero-copy views into C-allocated memory
        buf = ffi.buffer(c_result.masks, n * h * w * 4)
        self.masks = np.frombuffer(buf, dtype=np.float32).reshape(n, h, w)

        buf = ffi.buffer(c_result.iou_scores, n * 4)
        self.iou_scores = np.frombuffer(buf, dtype=np.float32).copy()

        self.iou_valid = bool(c_result.iou_valid)
        self.boxes_valid = bool(c_result.boxes_valid)
        self.best_mask = c_result.best_mask

        if c_result.boxes_valid and c_result.boxes != ffi.NULL:
            buf = ffi.buffer(c_result.boxes, n * 4 * 4)
            self.boxes = np.frombuffer(buf, dtype=np.float32).reshape(n, 4).copy()
        else:
            self.boxes = np.empty((n, 4), dtype=np.float32)
            self.boxes[:] = np.nan

        # Copy masks so we can free the C result immediately
        self.masks = self.masks.copy()

        # Free C memory now that we have numpy copies
        lib.sam3_result_free(self._c)
        self._c = None


class Model:
    """SAM3 segmentation model.

    Usage::

        with sam3.Model("model.sam3") as model:
            model.set_image("photo.jpg")
            result = model.segment(points=[(500, 300, 1)])
            print(result.masks.shape)

    Args:
        model_path: Path to .sam3 model weights file.
        bpe_path:   Optional path to BPE vocabulary file.
    """

    def __init__(self, model_path, bpe_path=None):
        self._ctx = lib.sam3_init()
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

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        if not self._closed:
            self.close()

    def close(self):
        """Release all resources. Safe to call multiple times."""
        if not self._closed:
            lib.sam3_free(self._ctx)
            self._ctx = ffi.NULL
            self._closed = True

    def _check_open(self):
        if self._closed:
            raise ValueError("Model is closed")

    def load_bpe(self, path):
        """Load BPE vocabulary for text prompts."""
        self._check_open()
        check(lib.sam3_load_bpe(self._ctx, path.encode()))

    def set_image(self, image):
        """Set the input image.

        Args:
            image: Either a file path (str) or a numpy array of shape
                   (H, W, 3) with dtype uint8 (RGB).
        """
        self._check_open()
        if isinstance(image, str):
            check(lib.sam3_set_image_file(self._ctx, image.encode()))
        else:
            image = np.ascontiguousarray(image, dtype=np.uint8)
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    f"Expected (H, W, 3) RGB array, got shape {image.shape}"
                )
            h, w = image.shape[:2]
            ptr = ffi.cast("const uint8_t *", image.ctypes.data)
            check(lib.sam3_set_image(self._ctx, ptr, w, h))

    def set_prompt_space(self, width, height):
        """Set the coordinate space for point/box prompts."""
        self._check_open()
        lib.sam3_set_prompt_space(self._ctx, width, height)

    def set_text(self, text):
        """Pre-tokenize and asynchronously encode a text prompt."""
        self._check_open()
        check(lib.sam3_set_text(self._ctx, text.encode()))

    def get_image_size(self):
        """Return the model's expected input image size."""
        self._check_open()
        return lib.sam3_get_image_size(self._ctx)

    def segment(self, *, points=None, boxes=None, masks=None, text=None):
        """Run segmentation with the given prompts.

        Args:
            points: List of (x, y, label) tuples. label=1 foreground, 0 background.
            boxes:  List of (x1, y1, x2, y2) tuples.
            masks:  List of numpy arrays, each shape (H, W) float32.
            text:   A string text prompt.

        Returns:
            Result object with masks, iou_scores, boxes, and best_mask.
        """
        self._check_open()

        prompts = []
        if points:
            for x, y, label in points:
                p = ffi.new("struct sam3_prompt *")
                p.type = lib.SAM3_PROMPT_POINT
                p.point.x = x
                p.point.y = y
                p.point.label = label
                prompts.append(p)

        if boxes:
            for x1, y1, x2, y2 in boxes:
                p = ffi.new("struct sam3_prompt *")
                p.type = lib.SAM3_PROMPT_BOX
                p.box.x1 = x1
                p.box.y1 = y1
                p.box.x2 = x2
                p.box.y2 = y2
                prompts.append(p)

        if masks:
            for mask_arr in masks:
                mask_arr = np.ascontiguousarray(mask_arr, dtype=np.float32)
                if mask_arr.ndim != 2:
                    raise ValueError(
                        f"Mask must be 2D, got {mask_arr.ndim}D"
                    )
                p = ffi.new("struct sam3_prompt *")
                p.type = lib.SAM3_PROMPT_MASK
                p.mask.data = ffi.cast("const float *", mask_arr.ctypes.data)
                p.mask.height = mask_arr.shape[0]
                p.mask.width = mask_arr.shape[1]
                prompts.append(p)

        if text:
            p = ffi.new("struct sam3_prompt *")
            p.type = lib.SAM3_PROMPT_TEXT
            p.text = text.encode()
            prompts.append(p)

        if not prompts:
            raise ValueError("At least one prompt is required")

        # Build contiguous array of sam3_prompt structs
        c_prompts = ffi.new("struct sam3_prompt[]", len(prompts))
        for i, p in enumerate(prompts):
            c_prompts[i] = p[0]

        c_result = ffi.new("struct sam3_result *")
        check(lib.sam3_segment(self._ctx, c_prompts, len(prompts), c_result))

        return Result(c_result)

    # --- Profiling ---

    def profile_enable(self):
        """Enable profiling (requires SAM3_HAS_PROFILE at compile time)."""
        self._check_open()
        check(lib.sam3_profile_enable(self._ctx))

    def profile_disable(self):
        """Disable profiling (data is preserved)."""
        self._check_open()
        lib.sam3_profile_disable(self._ctx)

    def profile_report(self):
        """Print profiling report to stderr."""
        self._check_open()
        lib.sam3_profile_report(self._ctx)

    def profile_reset(self):
        """Clear all collected profiling data."""
        self._check_open()
        lib.sam3_profile_reset(self._ctx)


# Import for Model constructor error
from sam3.errors import Sam3Error  # noqa: E402
```

**Step 2: Commit**

```bash
git add python/sam3/model.py
git commit -m "python: add Model class with segment/set_image/profiling"
```

---

### Task 6: Create build/packaging files

**Files:**
- Create: `python/pyproject.toml`
- Create: `python/setup.py`

**Step 1: Create `python/pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "sam3"
version = "0.1.0"
description = "Python bindings for the SAM3 inference engine"
requires-python = ">=3.9"
license = "MIT"
dependencies = [
    "cffi>=1.15",
    "numpy>=1.21",
]

[project.optional-dependencies]
dev = ["pytest>=7"]
```

**Step 2: Create `python/setup.py`**

```python
"""
Build script that compiles libsam3 as a shared library and bundles it
into the Python package.
"""
import os
import shutil
import subprocess
import sys

from setuptools import setup
from setuptools.command.build_ext import build_ext


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam3")


class CMakeBuild(build_ext):
    def run(self):
        build_dir = os.path.join(ROOT, "build-python")
        os.makedirs(build_dir, exist_ok=True)

        cmake_args = [
            "cmake", ROOT,
            "-DSAM3_SHARED=ON",
            "-DSAM3_TESTS=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        subprocess.check_call(cmake_args, cwd=build_dir)
        subprocess.check_call(
            ["cmake", "--build", ".", "--parallel"],
            cwd=build_dir,
        )

        # Copy shared library into package directory
        if sys.platform == "darwin":
            lib_name = "libsam3.dylib"
        else:
            lib_name = "libsam3.so"

        src = os.path.join(build_dir, lib_name)
        dst = os.path.join(PKG_DIR, lib_name)
        shutil.copy2(src, dst)


setup(
    cmdclass={"build_ext": CMakeBuild},
    packages=["sam3"],
    package_dir={"sam3": "sam3"},
    has_ext_modules=lambda: True,
)
```

**Step 3: Commit**

```bash
git add python/pyproject.toml python/setup.py
git commit -m "python: add build and packaging configuration"
```

---

### Task 7: Write tests

**Files:**
- Create: `python/tests/__init__.py` (empty)
- Create: `python/tests/conftest.py`
- Create: `python/tests/test_errors.py`
- Create: `python/tests/test_model.py`

**Step 1: Create `python/tests/__init__.py`**

Empty file.

**Step 2: Create `python/tests/conftest.py`**

```python
import os
import pytest


@pytest.fixture
def model_path():
    """Path to test model. Set SAM3_MODEL_PATH env var or skip."""
    path = os.environ.get("SAM3_MODEL_PATH")
    if not path or not os.path.isfile(path):
        pytest.skip("SAM3_MODEL_PATH not set or file not found")
    return path


@pytest.fixture
def test_image_path():
    """Path to test image. Set SAM3_TEST_IMAGE env var or skip."""
    path = os.environ.get("SAM3_TEST_IMAGE")
    if not path or not os.path.isfile(path):
        pytest.skip("SAM3_TEST_IMAGE not set or file not found")
    return path
```

**Step 3: Create `python/tests/test_errors.py`**

```python
"""Unit tests for error handling (no model needed)."""
import pytest
import sam3


def test_version_returns_string():
    v = sam3.version()
    assert isinstance(v, str)
    assert len(v) > 0


def test_set_log_level_valid():
    for level in ("debug", "info", "warn", "error"):
        sam3.set_log_level(level)


def test_set_log_level_invalid():
    with pytest.raises(KeyError):
        sam3.set_log_level("invalid")


def test_model_invalid_path():
    with pytest.raises(sam3.Sam3Error):
        sam3.Model("/nonexistent/model.sam3")


def test_model_close_twice():
    """Closing a Model twice should not crash."""
    try:
        m = sam3.Model("/nonexistent/model.sam3")
    except sam3.Sam3Error:
        pass  # Expected — the point is that close() doesn't crash
```

**Step 4: Create `python/tests/test_model.py`**

```python
"""Integration tests requiring a real model and image."""
import numpy as np
import pytest
import sam3


class TestModelSegment:
    def test_set_image_file(self, model_path, test_image_path):
        with sam3.Model(model_path) as m:
            m.set_image(test_image_path)
            assert m.get_image_size() > 0

    def test_set_image_array(self, model_path):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        with sam3.Model(model_path) as m:
            m.set_image(img)

    def test_segment_point(self, model_path, test_image_path):
        with sam3.Model(model_path) as m:
            m.set_image(test_image_path)
            result = m.segment(points=[(100, 100, 1)])
            assert result.masks.ndim == 3
            assert result.masks.shape[0] > 0
            assert result.iou_scores.shape[0] == result.masks.shape[0]

    def test_segment_box(self, model_path, test_image_path):
        with sam3.Model(model_path) as m:
            m.set_image(test_image_path)
            result = m.segment(boxes=[(50, 50, 200, 200)])
            assert result.masks.ndim == 3

    def test_segment_no_prompts(self, model_path, test_image_path):
        with sam3.Model(model_path) as m:
            m.set_image(test_image_path)
            with pytest.raises(ValueError, match="At least one prompt"):
                m.segment()

    def test_invalid_image_shape(self, model_path):
        with sam3.Model(model_path) as m:
            with pytest.raises(ValueError, match="Expected.*RGB"):
                m.set_image(np.zeros((256, 256), dtype=np.uint8))

    def test_closed_model_raises(self, model_path):
        m = sam3.Model(model_path)
        m.close()
        with pytest.raises(ValueError, match="closed"):
            m.set_image("test.jpg")

    def test_context_manager(self, model_path):
        with sam3.Model(model_path) as m:
            assert m.get_image_size() > 0
        with pytest.raises(ValueError, match="closed"):
            m.get_image_size()
```

**Step 5: Run the unit tests (no model needed)**

Run: `cd python && pip install -e ".[dev]" && pytest tests/test_errors.py -v`
Expected: All test_errors tests pass (or skip gracefully).

**Step 6: Commit**

```bash
git add python/tests/
git commit -m "python: add unit and integration test suite"
```

---

### Task 8: End-to-end verification

**Step 1: Build the shared library**

Run: `cd /Users/rbisri/Documents/sam3 && mkdir -p build-shared && cd build-shared && cmake .. -DSAM3_SHARED=ON -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)`

**Step 2: Install the Python package in dev mode**

Run: `cd /Users/rbisri/Documents/sam3/python && pip install -e ".[dev]"`

**Step 3: Run unit tests**

Run: `cd /Users/rbisri/Documents/sam3/python && pytest tests/test_errors.py -v`
Expected: All pass.

**Step 4: Run integration tests (if model available)**

Run: `SAM3_MODEL_PATH=/path/to/model.sam3 SAM3_TEST_IMAGE=/path/to/image.jpg pytest tests/test_model.py -v`
Expected: All pass.

**Step 5: Verify Python API works interactively**

```python
import sam3
print(sam3.version())
sam3.set_log_level("debug")
```

**Step 6: Final commit if any fixups needed**

```bash
git add -A python/
git commit -m "python: fix issues found during end-to-end verification"
```
