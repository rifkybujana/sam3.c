# SAM3 — Pure C Inference Engine

## What This Project Is

SAM3 is a pure C11 inference engine for Facebook's Segment Anything Model 3.
Metal backend first, extensible to CUDA/Vulkan. Modeled after ggml/llama.cpp.
Upstream reference: https://github.com/facebookresearch/sam3

### CRITICAL: This is SAM3, not SAM2

- **Always refer to this project as SAM3** (Segment Anything Model 3).
- **Never use "SAM2", "sam2", or "Segment Anything Model 2"** in code,
  comments, commit messages, or conversation. The upstream repo is `sam3`.
- All prefixes are `sam3_`, all macros are `SAM3_`, all paths use `sam3/`.
- If referencing the upstream Python repo, it is `facebookresearch/sam3`.

## Build

    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Debug
    make -j$(nproc)

Run tests:

    cd build && ctest --output-on-failure

## Directory Map

    include/sam3/     Public API headers (sam3.h, sam3_types.h)
    src/core/         Tensor ops, arena allocator, compute graph
    src/backend/      Backend abstraction + Metal/CPU implementations
    src/model/        SAM3 layers (image encoder, prompt encoder, mask decoder)
    src/util/         Logging, error codes
    tools/            CLI binaries (inference, weight conversion)
    tests/            Unit and integration tests
    models/           Model weights (.gitignored)
    docs/             Documentation (format specs, reference material)

## .sam3 Weight Format

Model weights are stored in the `.sam3` binary format. See
[docs/weight-format.md](docs/weight-format.md) for the full specification.

Quick summary:
- **Magic**: `0x334D4153` (ASCII "SAM3", little-endian)
- **Version**: 2
- **Layout**: 48-byte header + tensor descriptors (176 B each) + page-aligned
  data blob
- **Alignment**: data blob at 4096-byte boundary, tensors at 64-byte boundaries
- **Loading**: mmap-based with FNV-1a hash table for O(1) tensor lookup
- **Dtypes**: F32, F16, BF16, I32, I8, Q8_0 (block-quantized int8)
- **Conversion**: `sam3_convert` reads SafeTensors, writes `.sam3`
- **Source**: `src/core/weight.h` (format structs), `src/core/weight.c`
  (loader/writer)

## C Coding Standard

These rules are non-negotiable. Every file must follow them exactly.

### Language

- **C11 only.** No C++ features, no GNU extensions unless guarded by `#ifdef`.
- Compile with: `-std=c11 -Wall -Wextra -Wpedantic`
- Debug builds add: `-Werror -fsanitize=address,undefined`

### Formatting

- **Tabs for indentation, 8 characters wide.** This is the Linux kernel convention.
  It forces you to keep nesting shallow.
- **80-column soft limit, 100 hard limit.** Break long lines at operators or after commas.
- **K&R brace style** for functions (opening brace on its own line).
  Same-line braces for `if`, `for`, `while`, `switch`, `struct`.

```c
/* Function: opening brace on its own line */
static int compute_offset(int row, int col, int stride)
{
	if (row < 0 || col < 0) {
		return -1;
	}

	return row * stride + col;
}
```

### Naming

- **`snake_case` for everything:** functions, variables, types, enum values, macros.
- **Prefix public symbols with `sam3_`.**
  Internal symbols use subsystem prefix: `tensor_`, `metal_`, `graph_`, etc.
- **No Hungarian notation.** No `pFoo`, `m_bar`, `szName`.
- **No typedef hiding pointers.** If it's a pointer, the `*` must be visible.
- Typedefs are acceptable for opaque structs in public API:
  `typedef struct sam3_ctx sam3_ctx;`

### File Documentation Header

**Every `.c` and `.h` file MUST begin with this header.** This is the single most
important convention — it gives an LLM instant context about any file.

```c
/*
 * <relative/path/to/file> - <one-line description>
 *
 * <2-4 sentences explaining purpose, role in the system, and key design
 * decisions. Mention what subsystem this belongs to and how it fits into
 * the larger architecture.>
 *
 * Key types:  <primary structs/enums defined or used here>
 * Depends on: <direct header dependencies, not transitive>
 * Used by:    <files that directly include or call into this>
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */
```

Rules for the header:
- `Key types` lists the 1-3 most important types. Not every type.
- `Depends on` lists headers this file directly includes (not system headers).
- `Used by` lists files that directly depend on this one. Update when adding
  new callers. "Unknown" is acceptable for new files.
- Keep the description factual. No aspirational language.

### Function Documentation

Document non-trivial functions with a comment block above:

```c
/*
 * sam3_tensor_reshape - Change tensor dimensions without copying data.
 *
 * @t:        Tensor to reshape (must not be a view)
 * @new_dims: Array of new dimension sizes
 * @n_dims:   Number of dimensions (1-4)
 *
 * Returns 0 on success, -SAM3_EINVAL if total element count changes.
 * The tensor data pointer is not modified.
 */
int sam3_tensor_reshape(struct sam3_tensor *t, const int *new_dims, int n_dims);
```

Trivial getters, simple wrappers, and static helpers do not need doc comments
unless the behavior is surprising.

### Memory Management

- **Arena allocators for inference.** No `malloc`/`free` in hot paths.
  All allocations go through `sam3_alloc_*` functions.
- **No global mutable state.** All state lives in `sam3_ctx` or is passed
  as function arguments.
- **Ownership is explicit.** If a function allocates, its doc comment says
  who frees. Prefer arena allocation where the arena owns everything.

### Error Handling

- **Return `enum sam3_error` codes.** Never use errno for sam3 errors.
- **`goto cleanup` pattern** for functions that acquire multiple resources:

```c
int sam3_do_thing(struct sam3_ctx *ctx)
{
	struct resource *a = NULL;
	struct resource *b = NULL;
	int err;

	a = acquire_a();
	if (!a) {
		err = SAM3_ENOMEM;
		goto cleanup;
	}

	b = acquire_b();
	if (!b) {
		err = SAM3_ENOMEM;
		goto cleanup;
	}

	err = use_resources(a, b);

cleanup:
	release_b(b);
	release_a(a);
	return err;
}
```

- **Never silently ignore errors.** Log or propagate.

### Logging

All diagnostic output goes through `src/util/log.h`. Never use raw
`printf`/`fprintf` for diagnostics — use the logging macros.

**Levels** (defined in `enum sam3_log_level`):
- `SAM3_LOG_DEBUG` — detailed tracing (suppressed by default)
- `SAM3_LOG_INFO` — operational milestones (model loaded, patches evaluated)
- `SAM3_LOG_WARN` — non-fatal issues (cache miss, map full)
- `SAM3_LOG_ERROR` — failures that affect correctness

**Macros** — use these, not `sam3_log_write()` directly:

```c
sam3_log_debug("block %d/%d evaluated", i + 1, depth);
sam3_log_info("patch embedding evaluated (%d patches)", np);
sam3_log_warn("tensor map full, array not cached");
sam3_log_error("unsupported dtype %d", t->dtype);
```

Each macro captures `__FILE__` and `__LINE__` automatically. Output format:
`[LEVEL] file:line: message` on stderr.

**Configuration:**
- Default level is `SAM3_LOG_INFO`. Call `sam3_log_set_level(SAM3_LOG_DEBUG)`
  to enable debug output (CLI tools do this with `-v`).
- No initialization required — the macros work immediately.

**Rules:**
- Error paths must log before returning an error code.
- Use `sam3_log_error` for failures, not `sam3_log_warn`.
- Keep messages short and include relevant numeric context
  (sizes, counts, indices).
- Do not log in tight loops. One message per operation, not per iteration.

### Backend Abstraction

- Backends implement `struct sam3_backend_ops` (vtable of function pointers).
- Backend selection happens at runtime via `sam3_backend_init()`.
- Never call Metal/CUDA/CPU functions directly from model code — always
  go through the backend vtable.

### Includes

- System headers first (`<stdint.h>`, `<stdlib.h>`), then project headers.
- Use `#include "sam3/header.h"` for public headers.
- Use `#include "local_header.h"` for same-directory private headers.
- Every header has an include guard: `#ifndef SAM3_CORE_TENSOR_H` / `#define` / `#endif`
- No `#pragma once`.

### Testing

- One test file per module: `tests/test_<module>.c`
- Test functions named `test_<module>_<behavior>`
- Tests should be runnable via CTest
- Test assertions use a simple macro from `tests/test_helpers.h`

### Commits

- One logical change per commit.
- Imperative mood: "Add tensor reshape" not "Added tensor reshape"
- Format: `<subsystem>: <description>` — e.g., `core/tensor: add reshape operation`

### Performance Rules

These rules apply to every new feature, hot path, or improvement. They are not
optional micro-optimizations — they are the baseline expectation for code that
runs during inference.

**1. No allocations in hot paths.**
Use stack buffers, arena allocators, or pre-allocated storage. Never call
`malloc`/`free` inside a loop that runs per-token, per-pixel, or per-layer.
If you need a temporary key or buffer, size it on the stack.

**2. Don't compute what you can cache.**
If a value is expensive to produce and reused across iterations, store it.
Track derived quantities (string lengths, hash values, merge ranks) in
parallel arrays rather than recomputing them.

**3. Don't scan data twice.**
If you need both the length and the content, do both in one pass. Avoid
patterns like `strlen(s)` followed by a loop over `s`. Prefer a single loop
with an early-out.

**4. Bulk memory ops over scalar loops.**
`memcpy`/`memset` use SIMD internally and are faster than hand-written byte
loops. For filling with a non-zero pattern, use `memcpy` from a `static const`
array. Reserve scalar loops for cases where per-element logic is required.

**5. Branchless over branchy for simple predicates.**
Replace predictable `if`/`else` on arithmetic conditions with bitwise
operations. Example: `c |= ((unsigned)(c - 'A') < 26u) << 5` instead of
`if (c >= 'A' && c <= 'Z') c += 32`. One instruction beats a branch.

**6. SIMD for byte-level bulk work (always guarded).**
When processing arrays of bytes (lowercase, copy, compare), use NEON/SSE
to handle 16 at a time. Always behind `#ifdef __aarch64__` (or `__SSE2__`)
with a scalar fallback. Use `__attribute__((no_sanitize("address")))` on
SIMD helpers that intentionally over-read within a page boundary.

**7. Cache results of expensive pure functions.**
If a function is deterministic and called repeatedly with the same inputs,
add a direct-mapped or small hash cache. BPE word encoding is a textbook
example: the same words appear repeatedly, and each encode is O(n^2).

**8. Benchmark without sanitizers.**
ASan adds 5-20x overhead per memory access. Always measure performance in
a Release build (`cmake .. -DCMAKE_BUILD_TYPE=Release`). Debug builds with
sanitizers are for correctness, not speed.

### What NOT To Do

- Do not use `typedef` to hide `struct` keywords in internal code.
  Public API typedefs for opaque handles are the exception.
- Do not use variadic macros for control flow.
- Do not `#include` a `.c` file.
- Do not use `alloca()`. Use the arena allocator.
- Do not add features "for later." YAGNI. Build what is needed now.
- Do not write C++. If it doesn't compile with `-std=c11`, it doesn't ship.
