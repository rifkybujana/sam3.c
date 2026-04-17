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
