# Windows Support Progress Report

Date: 2026-05-10

Branch: `feature/windows-blas`

Scope: upstream `sam3.c` only. The app vendor checkout at
`app.netrart.com/vendor/sam3.c` was not edited directly; the app still needs
to consume these changes by advancing the submodule after upstream verification.

## Summary

Phases 0 through 5 of the Windows support plan are implemented and verified.
The engine now has MSVC-oriented CMake configuration, a cross-platform runtime
platform layer, Windows-safe CPU/threading behavior, vcpkg FFmpeg/OpenBLAS
detection, portable video directory/encoding support, and Rust bindings that
can link against MSVC `sam3.dll`/`sam3.lib` artifacts on Windows.

Recent commit map:

- `d95a19f` — `feat: add Windows platform support`
- `4be68a6` — `feat: add Windows video directory portability`
- `0cb530f` — `fix: make video encoding portable on Windows`
- `c028648` — `fix: make Rust bindings link on Windows`

## Phase-by-phase Work

### Phase 0: Guardrails And Bootstrap

- Confirmed the work happens in upstream `sam3.c`, not the app vendor checkout.
- Recorded the starting vendor submodule SHA for later app integration.
- Added CMake options used throughout the Windows work:
  - `SAM3_VIDEO`
  - `SAM3_REQUIRE_VIDEO`
  - `SAM3_REQUIRE_BLAS`
- Made warning flags compiler-specific so MSVC no longer receives GCC/Clang
  warning options.
- Added real video-off behavior so early Windows builds can avoid Unix-style
  FFmpeg setup before platform portability is in place.

### Phase 1: Platform Layer

- Added `src/util/platform.h` and `src/util/platform.c` as the OS abstraction
  layer.
- Implemented cross-platform helpers for:
  - read-only file mapping and unmapping
  - file prefetch hints
  - regular-file and directory detection
  - path basename handling for both `/` and `\`
  - directory listing
  - CPU count
  - temp directory creation
  - mkdir/rmdir
  - threads, mutexes, and condition variables
- Migrated weight loading and SafeTensors loading away from direct POSIX
  `open`, `fstat`, `mmap`, `madvise`, `munmap`, and `pthread` usage.
- Moved async text worker thread handling to the platform thread helpers,
  preserving the larger stack request via `sam3_thread_create_with_stack`.
- Moved feature-cache temp/spill directory creation to platform helpers.
- Added a Windows `QueryPerformanceCounter` path for high-resolution timing.
- Replaced non-standard `M_PI` usage with a portable constant.
- Added `tests/test_platform_file.c` for file mapping, path handling, and later
  directory listing coverage.

### Phase 2: Threadpool And CPU Portability

- Migrated `src/util/threadpool.c` from direct `pthread_*` and `sysconf` calls
  to `sam3_thread`, `sam3_mutex`, `sam3_cond`, and `sam3_platform_cpu_count`.
- Added MSVC executable stack reserve handling via `sam3_configure_executable`
  because some test executables allocate large graph structures on the stack.
- Fixed MSVC-exposed scalar broadcast bugs in CPU kernels:
  - `cpu_elementwise.c`
  - `cpu_mul_f16.c`
  - `cpu_add_bf16.c`
  - `cpu_mul_bf16.c`
- Adjusted `tests/test_cpu_kernels.c` to avoid MSVC constant divide-by-zero
  compilation failure.
- Made path/env behavior portable in tests that previously assumed POSIX
  paths or functions.

### Phase 3: CMake, FFmpeg, And BLAS

- Added `cmake/Sam3Ffmpeg.cmake`.
- On Windows, FFmpeg detection now tries vcpkg/config/module/pkg-config paths
  and defines the `sam3_ffmpeg` interface target when found.
- On non-Windows, the existing static FFmpeg source-built path is preserved.
- Added `SAM3_HAS_VIDEO` gating so video sources/tests are only compiled when
  video dependencies are actually available.
- Updated video tests to link through `sam3_ffmpeg` instead of the old
  hard-coded `ffmpeg_static` target.
- Updated BLAS detection so Windows prefers `OpenBLAS::OpenBLAS` from vcpkg,
  then falls back to FindBLAS only when it can also resolve `cblas.h`.
- Added `SAM3_REQUIRE_BLAS` failure behavior for explicit BLAS-required builds.
- Installed local vcpkg dependencies for the Windows build:
  - `ffmpeg:x64-windows`
  - `openblas[core,threads,dynamic-arch]:x64-windows`
  - `zlib:x64-windows`
- The `[threads,dynamic-arch]` features on OpenBLAS are mandatory: the default
  `openblas:x64-windows` triplet ships a `SINGLE_THREADED generic` DLL, which
  pins `cblas_sgemm` to one core and disables AVX2/AVX-512 kernels. Because the
  CPU `matmul` and `conv2d` kernels intentionally bypass the SAM3 thread pool
  when `SAM3_HAS_BLAS` is defined (and let BLAS do its own parallelism), the
  default DLL leaves the image encoder running single-threaded on a generic
  scalar kernel. With these features the DLL self-identifies as
  `OpenBLAS … DYNAMIC_ARCH NO_AFFINITY Haswell USE_OPENMP` (or USE_THREAD)
  with `MAX_THREADS` ≥ host core count, and the encoder saturates available
  cores via `OPENBLAS_NUM_THREADS` (auto-detected by default).

### Phase 4: Video Directory And Encoding Portability

- Extended `tests/test_platform_file.c` with directory-listing assertions.
- Reworked `src/util/video.c` to remove direct `dirent.h` and `stat` usage.
- `sam3_frame_dir_list_open` now uses `sam3_dir_list_open` and keeps its
  existing image filtering and filename sorting behavior.
- `sam3_video_detect_type` now uses `sam3_path_is_dir` and
  `sam3_path_is_regular`.
- Replaced `strcasecmp` usage in `src/util/video_encode.c` with a portable
  ASCII case-insensitive compare helper.
- Fixed Windows FFmpeg encoder selection so the software RGB-to-YUV path does
  not pick hardware-backed encoders such as `h264_mf` that reject `yuv420p`.
- Added an MPEG-4 fallback for MP4 outputs when compatible H.264 is unavailable.
- Made `tests/test_video_encode.c` use platform temp directories instead of
  POSIX `/tmp` and `getpid` assumptions.
- Added defensive early returns in the encoder test so encoder-open failures
  report as test failures rather than misleading crashes.

### Phase 5: Rust Windows Linking And Cache Contract

- Pinned the Rust workspace to the installed MSVC host toolchain:
  `1.77.2-x86_64-pc-windows-msvc`.
- Updated `sam3-sys` build artifact discovery so Windows accepts:
  - `build/Release/sam3.dll` plus `sam3.lib`
  - `build/Debug/sam3.dll` plus `sam3.lib`
  - `build/sam3.dll` plus `sam3.lib`
  - Unix `libsam3.dylib` / `libsam3.so`
- Fixed relative `SAM3_BUILD_DIR` handling so the plan's
  `..\..\build-win` works from the Rust workspace and not only from the
  `sam3-sys` package directory.
- Added runtime DLL staging for Windows Cargo tests/binaries by copying
  `sam3.dll` and sibling dependency DLLs into the Cargo profile directory and
  `deps` directory.
- Added checked-in fallback FFI bindings for Windows systems without
  `libclang.dll`, while keeping bindgen as the normal path when libclang is
  available.
- Added safe cache Rust API:
  - `CacheOpts`
  - `CacheKind`
  - `CacheStats`
  - `Ctx::new_with_cache`
  - `Ctx::new_with_cache_opts`
  - `Ctx::cache_clear`
  - `Ctx::cache_stats`
  - `Ctx::precache_image`
  - `Ctx::cache_save_image`
  - `Ctx::cache_load_image`
- Added Rust cache tests covering app-needed slot counts, default cache options,
  fresh stats, and clear-on-fresh-context behavior.

## Validation Completed

## Phase 6 Progress

- Started Phase 6 upstream verification on Windows.
- Current planned gate: configure and build `build-win` as a Release shared
  MSVC build with `SAM3_VIDEO=ON` and `SAM3_BLAS=ON`, then run full Release
  CTest.
- Configure completed for `build-win` with the Visual Studio 17 2022 x64
  generator. CMake found FFmpeg through the vcpkg `FFMPEG` module and found
  OpenBLAS for BLAS support.
- Release shared build produced `build-win/Release/sam3.dll` and
  `build-win/Release/sam3.lib`, but the full build did not complete because
  several test/tool targets still contain MSVC portability issues.
- Build blockers found from the log:
  - POSIX-only headers remain in tests/tools: `unistd.h`, `sys/wait.h`,
    `getopt.h`, and `libgen.h`.
  - Several tests use compile-time floating-point divide-by-zero expressions
    that MSVC rejects as errors.
  - Benchmark timing helpers still call `clock_gettime(CLOCK_MONOTONIC)`.
  - `tools/gen_nhwc_fixtures.c` still depends on POSIX `S_ISDIR`/`mkdir`
    behavior.
- Applied a corrective portability pass after the failed build: added MSVC
  compatibility headers for lightweight POSIX test/CLI conveniences, moved
  benchmark timing to `sam3_time_ns`, made CLI subprocess tests resolve the
  built `sam3_cli` target, replaced MSVC-rejected float constants with runtime
  helper values, and moved `gen_nhwc_fixtures` directory handling onto the
  platform layer.
- Second build attempt cleared those blocker classes and reduced the hard
  failure to `test_cli_common` using POSIX `open_memstream`; the test capture
  path was changed to use a portable temporary file.
- Release shared build now completes successfully. Confirmed key artifacts:
  `build-win/Release/sam3.dll`, `build-win/Release/sam3.lib`,
  `build-win/Release/sam3_cli.exe`, and
  `build-win/Release/test_cli_common.exe`.
- First full Release CTest run executed 82 tests: 76 passed and 6 failed. The
  failing tests all used hard-coded POSIX `/tmp/...` paths on Windows:
  `test_feature_cache_persist`, `test_image`, `test_sam3_1_header`,
  `test_tensor_dump`, `test_weight_q8`, and `test_weight_safetensors`.
- Replaced those active test paths, plus skipped cache persistence smoke paths,
  with local relative test artifacts so they work under Windows CTest without a
  POSIX `/tmp` directory.
- Full Release CTest now passes: 82 tests passed, 0 failed.
- Upstream SAM3 Phase 6 commit step completed on `feature/windows-blas`.

### Platform And Weight Tests

Static MSVC validation was used because Windows Defender blocks the Debug shared
`sam3.dll`.

```powershell
ctest --test-dir build-win-static -C Debug `
  -R "^(test_platform_file|test_weight|test_fixture_compare)$" `
  --output-on-failure
```

Result: `100% tests passed`.

### CPU And Threadpool Tests

```powershell
ctest --test-dir build-win-static -C Debug `
  -R "^(test_cpu_backend|test_cpu_kernels|test_elementwise_f16|test_elementwise_bf16)$" `
  --output-on-failure
```

Result: `100% tests passed`.

### FFmpeg Detection

```powershell
cmake -S . -B build-win -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" `
  -DSAM3_TESTS=ON -DSAM3_SHARED=ON -DSAM3_REQUIRE_VIDEO=ON -DSAM3_VIDEO=ON
```

Result: configure found FFmpeg via the vcpkg `FFMPEG` module.

### BLAS Detection

The Phase 3 BLAS checkpoint was verified with video disabled so Phase 4 video
portability did not gate BLAS detection.

```powershell
cmake -S . -B build-win-blas -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" `
  -DSAM3_SHARED=ON -DSAM3_TESTS=ON -DSAM3_REQUIRE_BLAS=ON -DSAM3_VIDEO=OFF
cmake --build build-win-blas --config Release --target sam3
```

Result: configure reported `BLAS: found OpenBLAS` and produced
`build-win-blas/Release/sam3.dll`.

### Video Portability Tests

```powershell
ctest --test-dir build-win-static -C Debug `
  -R "^(test_platform_file|test_video_io|test_video_encode)$" `
  --output-on-failure
```

Result:

```text
test_platform_file ............... Passed
test_video_encode ................ Passed
test_video_io .................... Passed
100% tests passed, 0 tests failed out of 3
```

### Rust Bindings

```powershell
cd d:\NetraRT\Development\NetraVision\sam3.c\bindings\rust
$env:SAM3_BUILD_DIR = "..\..\build-win"
cargo test
```

Result:

```text
32 unit tests passed
4 cache tests passed
5 integration tests passed
1 sam3-sys version test passed
Doc-tests passed
```

Expected warning on this machine:

```text
sam3-sys: using checked-in fallback bindings because bindgen panicked
(libclang may be missing)
```

## Technical Debt And Known Risks

- **Windows Defender blocks Debug shared `sam3.dll`.** Debug shared validation
  can fail with process/load errors because Defender flags the generated DLL.
  No exclusions or bypasses were added. Current reliable validation uses
  `build-win-static` or Release shared builds.
- **Full Phase 6 shared Release verification is complete.** The full
  `cmake --build build-win --config Release --parallel` plus
  `ctest --test-dir build-win -C Release --output-on-failure` gate passed after
  the remaining Windows test/tool portability fixes.
- **Rust fallback bindings can drift.** `sam3-sys/src/fallback_bindings.rs` is
  a checked-in manual fallback used when `libclang.dll` is absent. It must be
  kept in sync with `include/sam3/sam3.h` and `include/sam3/sam3_types.h` when
  the C API changes.
- **Bindgen still requires LLVM/libclang for the generated path.** The fallback
  makes Windows builds work without LLVM, but machines with `libclang.dll`
  should still use bindgen-generated bindings as the primary path.
- **Build-script helper tests are embedded in `build.rs`.** Cargo does not run
  ordinary `#[cfg(test)]` tests inside build scripts as part of normal package
  tests, so the artifact-detection helper test is mostly documentation unless
  factored into a testable helper crate/module later.
- **FFmpeg encoder quality fallback is pragmatic.** On Windows, hardware H.264
  encoders like `h264_mf` are skipped for compatibility with the current
  software `yuv420p` pipeline. MP4 can fall back to MPEG-4 when compatible H.264
  is unavailable. This favors passing/runtime stability over optimal output
  quality.
- **MSVC warnings remain.** Builds still emit warnings around functions such as
  `fopen`, `strdup`, and `sprintf` from existing code/vendor headers. They are
  not currently treated as errors.
- **vcpkg dependencies are local machine state.** FFmpeg, OpenBLAS, and zlib are
  installed under `.tools/vcpkg`, but this still depends on the local vcpkg root
  being present and passed via `CMAKE_TOOLCHAIN_FILE` on fresh machines.
- **CMake cache reuse can hide toolchain changes.** Reusing build directories
  may print warnings such as manually specified `CMAKE_TOOLCHAIN_FILE` not being
  used. Fresh build directories are safer for final verification.
- **Phase 3 BLAS checkpoint used `SAM3_VIDEO=OFF`.** This was deliberate to keep
  BLAS validation independent from Phase 4 video portability, but Phase 6 should
  verify BLAS and video together in one Release shared build.
- **App integration is not done yet.** `app.netrart.com` has not been updated to
  consume the upstream changes. The vendor submodule bump, app build scripts,
  runtime DLL staging, and final app packaging/self-check are still future
  phases.
- **No direct vendor edits.** This is intentional, but it means the app will not
  see any of these fixes until the upstream branch is committed/pushed and the
  app submodule pointer is advanced.

## Next Step

Proceed to Phase 6: run the full upstream SAM3 Release shared build and test
suite on Windows, then prepare the upstream SAM3 commit/push state needed for
the app submodule bump.
