# SAM3 — Pure C Inference Engine

## What This Project Is

SAM3 is a pure C11 inference engine for Facebook's Segment Anything Model 3.
Metal backend first, extensible to CUDA/Vulkan. Modeled after ggml/llama.cpp.

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
 * Copyright (c) 2026
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

### What NOT To Do

- Do not use `typedef` to hide `struct` keywords in internal code.
  Public API typedefs for opaque handles are the exception.
- Do not use variadic macros for control flow.
- Do not `#include` a `.c` file.
- Do not use `alloca()`. Use the arena allocator.
- Do not add features "for later." YAGNI. Build what is needed now.
- Do not write C++. If it doesn't compile with `-std=c11`, it doesn't ship.
