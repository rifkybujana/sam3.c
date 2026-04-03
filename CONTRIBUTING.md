# Contributing to SAM3

Welcome to the SAM3 project! We are building a pure C11 inference engine for Facebook's Segment Anything Model 3. 

To ensure the codebase remains clean, fast, and maintainable, we have strict coding standards. Please review these guidelines before submitting any pull requests.

## Architecture & Directory Map

Before you start, familiarize yourself with the layout:

* `include/sam3/` - Public API headers (`sam3.h`, `sam3_types.h`)
* `src/core/` - Tensor ops, arena allocator, compute graph
* `src/backend/` - Backend abstraction + Metal/CPU implementations
* `src/model/` - SAM3 layers (image encoder, prompt encoder, mask decoder)
* `src/util/` - Logging, error codes
* `tools/` - CLI binaries (inference, weight conversion)
* `tests/` - Unit and integration tests

## C Coding Standard

These rules are non-negotiable. Every file must follow them exactly.

### Language Constraints
* **C11 only.** No C++ features, no GNU extensions unless guarded by `#ifdef`.
* Your code must compile with: `-std=c11 -Wall -Wextra -Wpedantic`
* If it doesn't compile with `-std=c11`, it doesn't ship.

### Formatting
* **Indentation:** Use tabs, 8 characters wide (Linux kernel convention). Keep nesting shallow.
* **Line length:** 80-column soft limit, 100 hard limit. Break long lines at operators or after commas.
* **Braces:** K&R brace style for functions (opening brace on its own line). Same-line braces for `if`, `for`, `while`, `switch`, `struct`.

### Naming Conventions
* **Cases:** `snake_case` for everything (functions, variables, types, enum values, macros).
* **Prefixes:** Prefix public symbols with `sam3_`. Internal symbols use their subsystem prefix (e.g., `tensor_`, `metal_`, `graph_`).
* **Anti-patterns:** No Hungarian notation (`pFoo`, `m_bar`, `szName`). No typedefs hiding pointers (the `*` must be visible). Typedefs are acceptable for opaque structs in the public API (`typedef struct sam3_ctx sam3_ctx;`).

### Memory Management
* **Allocations:** We use arena allocators for inference. No `malloc`/`free` in hot paths. All allocations go through `sam3_alloc_*` functions.
* **State:** No global mutable state. All state lives in `sam3_ctx` or is passed as function arguments. YAGNI: do not use `alloca()`.
* **Ownership:** Ownership is explicit. If a function allocates, its doc comment says who frees. Prefer arena allocation where the arena owns everything.

### Error Handling
* **Return Codes:** Use `enum sam3_error` codes. Never use `errno` for sam3 errors.
* **Cleanup:** Use the `goto cleanup` pattern for functions that acquire multiple resources.
* **Never silently ignore errors.** Log or propagate them.

### File Documentation Header

**Every `.c` and `.h` file MUST begin with this header.** 

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

### Function Documentation

Document non-trivial functions with a comment block above them:

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

### Includes
* Include system headers first (`<stdint.h>`, `<stdlib.h>`), then project headers.
* Use `#include "sam3/header.h"` for public headers, and `#include "local_header.h"` for same-directory private headers.
* Use standard include guards (`#ifndef SAM3_CORE_TENSOR_H`). **Do not use `#pragma once`.**
* Do not `#include` a `.c` file.

### Backend Development
* Backends must implement `struct sam3_backend_ops` (vtable of function pointers).
* Never call Metal/CUDA/CPU functions directly from model code — always go through the backend vtable.

## Performance Checklist

These rules apply to every hot path change. If your code runs during inference
(per-token, per-pixel, per-layer), all eight rules are mandatory.

### 1. No allocations in hot paths

Use stack buffers or arena allocators. Never `malloc`/`free` inside a loop.

```c
/* BAD */
for (int i = 0; i < n; i++) {
	char *key = malloc(len_a + len_b + 2);
	/* ... */
	free(key);
}

/* GOOD */
char key_buf[128];
for (int i = 0; i < n; i++) {
	/* build key in key_buf */
}
```

### 2. Don't compute what you can cache

Track derived quantities in parallel arrays. If you call `strlen()` on the
same string more than once, store the length.

```c
/* BAD: recompute every iteration */
for (int i = 0; i < n - 1; i++) {
	size_t la = strlen(symbols[i]);
	size_t lb = strlen(symbols[i + 1]);
}

/* GOOD: parallel length array, update on mutation */
int sym_len[MAX_SYMBOLS];
```

### 3. Don't scan data twice

If you need the length and the content, do both in one pass.

```c
/* BAD */
int len = (int)strlen(text);
int n = len < limit ? len : limit;
for (int i = 0; i < n; i++) { /* process */ }

/* GOOD */
int i = 0;
while (i < limit && text[i]) { /* process */ i++; }
```

### 4. Bulk memory ops over scalar loops

`memcpy`/`memset` use SIMD internally. For non-zero fill patterns, copy from
a `static const` array.

```c
/* BAD */
for (int i = pos; i < max; i++)
	tokens[i] = EOT_TOKEN;

/* GOOD */
static const int32_t eot_pad[77] = { E_, E_, ... };
memcpy(tokens + pos, eot_pad, (max - pos) * sizeof(int32_t));
```

### 5. Branchless over branchy for simple predicates

Replace predictable branches with arithmetic.

```c
/* BAD */
if (c >= 'A' && c <= 'Z')
	c += 'a' - 'A';

/* GOOD */
c |= (unsigned char)(((unsigned)(c - 'A') < 26u) << 5);
```

### 6. SIMD for byte-level bulk work (always guarded)

Use NEON (ARM64) or SSE (x86) to process 16 bytes at a time. Always provide
a scalar fallback. Mark SIMD helpers that intentionally over-read with
`no_sanitize("address")`.

```c
#ifdef __aarch64__
__attribute__((no_sanitize("address")))
static int neon_process(const uint8_t *src, int32_t *dst, int limit)
{
	/* 16 bytes per iteration, scalar fallback after */
}
#endif
```

### 7. Cache results of expensive pure functions

If a deterministic function is called repeatedly with the same inputs, add a
direct-mapped hash cache.

```c
int slot = fnv1a(word, len) & (CACHE_SIZE - 1);
if (cache[slot].key_len == len && memcmp(...) == 0)
	return cache[slot].result;   /* hit */
/* miss: compute, then store in cache[slot] */
```

### 8. Benchmark in Release mode

ASan adds 5-20x overhead per memory access. Debug builds measure sanitizer
cost, not your code.

```
# Correctness
cd build && ctest --output-on-failure

# Performance
cd build-release && ./bench_tokenizer
```

## Benchmarking

Benchmark files live in `tests/bench_*.c` and are auto-registered via the
`foreach` loop in `CMakeLists.txt`. They are not included in CTest.

To add a new benchmark:

1. Create `tests/bench_<module>.c` following existing patterns
2. Use `clock_gettime(CLOCK_MONOTONIC)` for timing
3. Include warmup iterations before timed iterations
4. Report meaningful metrics (enc/s, GFLOPS, GB/s, ns/op)
5. Run from a Release build: `cd build-release && ./bench_<module>`

## Testing Your Changes
* Write tests for new modules in `tests/test_<module>.c`.
* Name test functions `test_<module>_<behavior>`.
* Ensure tests can be run via CTest. Use the predefined assertions in `tests/test_helpers.h`.

## Committing

* Make one logical change per commit.
* Use the imperative mood for subject lines (e.g., "Add tensor reshape", not "Added tensor reshape").
* Follow the format: `<subsystem>: <description>` (e.g., `core/tensor: add reshape operation`).
* Do not add features "for later" (YAGNI). Build only what is needed now.

Thank you for contributing to SAM3!