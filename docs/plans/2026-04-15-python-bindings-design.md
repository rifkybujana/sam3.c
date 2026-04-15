# Python Bindings Design

## Summary

cffi-based Python bindings for SAM3, providing a Pythonic high-level API.
ABI mode (dlopen) — the shared library ships inside the wheel.

## Package Structure

```
python/
├── sam3/
│   ├── __init__.py          # Public API: Model, Sam3Error, set_log_level
│   ├── _ffi.py              # cffi cdef + dlopen
│   ├── _lib.py              # Locate libsam3.so/.dylib
│   └── model.py             # High-level Model class
├── pyproject.toml
├── setup.py                 # cmake integration for wheel builds
└── tests/
    ├── test_model.py
    └── conftest.py
```

## Python API

```python
import sam3
import numpy as np

with sam3.Model("model.sam3") as model:
    model.set_image("photo.jpg")           # From file
    model.set_image(np.array(...))         # From (H, W, 3) uint8 RGB

    model.load_bpe("vocab.bpe")            # For text prompts

    result = model.segment(points=[(500, 300, 1)])
    result = model.segment(boxes=[(x1, y1, x2, y2)])
    result = model.segment(text="the dog")
    result = model.segment(
        points=[(500, 300, 1), (100, 100, 0)],
        boxes=[(50, 50, 400, 400)],
    )

result.masks        # np.ndarray (n_masks, H, W) float32
result.iou_scores   # np.ndarray (n_masks,) float32
result.boxes        # np.ndarray (n_masks, 4) float32 xyxy
result.best_mask    # int
```

## Error Handling

C error codes map to Python exception hierarchy:

```
Sam3Error
├── InvalidArgumentError   (SAM3_EINVAL)
├── OutOfMemoryError       (SAM3_ENOMEM)
├── IOError                (SAM3_EIO)
├── BackendError           (SAM3_EBACKEND)
├── ModelError             (SAM3_EMODEL)
└── DtypeError             (SAM3_EDTYPE)
```

Every C function return is checked; non-zero raises the matching exception
with the message from `sam3_error_str()`.

## Lifecycle

- `Model` is a context manager. `__exit__` calls `sam3_free`.
- `Model.__del__` calls `sam3_free` as fallback.
- `Result.__del__` calls `sam3_result_free`.
- Using a closed `Model` raises `ValueError`.
- Thread safety: not guaranteed (same as C library), documented.

## FFI Layer

Manually maintained `ffi.cdef()` block with the public API subset.
No header parsing at runtime — explicit contract.

`_lib.py` locates `libsam3.{so,dylib}` bundled in the package directory.

## Build & Packaging

```toml
[project]
name = "sam3"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = ["cffi>=1.15", "numpy>=1.21"]
```

Build flow:
1. `setup.py` invokes cmake with `-DBUILD_SHARED_LIBS=ON`
2. cmake builds `libsam3.{so,dylib}`
3. Library is copied into `python/sam3/`
4. setuptools packages into a wheel

CMake change: add `BUILD_SHARED_LIBS` option (currently static only).

Platform targets:
- macOS arm64 (Metal + CPU)
- macOS x86_64 (CPU)
- Linux x86_64 (CPU)

## Logging

```python
sam3.set_log_level("debug")  # Wraps sam3_log_set_level
```

Direct passthrough, no Python logging integration.
