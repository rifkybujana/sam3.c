# SAM3 Project Scaffold Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Scaffold a pure C inference engine for SAM3 with Linux kernel-style coding rules, LLM-friendly documentation headers, and Metal-first backend architecture.

**Architecture:** ggml-inspired layout — public headers in `include/sam3/`, implementation split into `core/` (tensor primitives), `backend/` (vtable dispatch with Metal/CPU), `model/` (SAM3 layers), and `util/` (logging, errors). Backend extensibility via function pointer vtable.

**Tech Stack:** C11, CMake, Metal (macOS), POSIX

---

### Task 1: Git Setup and .gitignore

**Files:**
- Create: `.gitignore`

**Step 1: Create .gitignore**

```gitignore
# Build
build/
cmake-build-*/
*.o
*.a
*.so
*.dylib

# Models / weights
models/*.bin
models/*.safetensors
models/*.pt
models/*.onnx

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Generated
compile_commands.json
```

**Step 2: Commit**

```bash
git add .gitignore
git commit -m "Add .gitignore for C project with model weights exclusion"
```

---

### Task 2: CLAUDE.md — Kernel-Style C Coding Rules

**Files:**
- Create: `CLAUDE.md`

**Step 1: Create CLAUDE.md**

```markdown
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
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "Add CLAUDE.md with kernel-style C coding rules"
```

---

### Task 3: Root CMakeLists.txt

**Files:**
- Create: `CMakeLists.txt`

**Step 1: Create root CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.20)
project(sam3 VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(SAM3_METAL "Enable Metal backend" OFF)
option(SAM3_CPU   "Enable CPU backend"   ON)
option(SAM3_TESTS "Build tests"          ON)

# Auto-detect Metal on macOS
if(APPLE AND NOT DEFINED SAM3_METAL)
	set(SAM3_METAL ON)
elseif(APPLE)
	# User explicitly set it, keep their choice
endif()

# Compiler warnings
add_compile_options(
	-Wall -Wextra -Wpedantic
	-Wno-unused-parameter
	-Wstrict-prototypes
	-Wmissing-prototypes
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	add_compile_options(-fsanitize=address,undefined)
	add_link_options(-fsanitize=address,undefined)
endif()

# Include paths
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

# Library sources
file(GLOB_RECURSE SAM3_CORE_SOURCES   "src/core/*.c")
file(GLOB_RECURSE SAM3_MODEL_SOURCES  "src/model/*.c")
file(GLOB_RECURSE SAM3_UTIL_SOURCES   "src/util/*.c")

set(SAM3_SOURCES
	${SAM3_CORE_SOURCES}
	${SAM3_MODEL_SOURCES}
	${SAM3_UTIL_SOURCES}
)

# Backend sources
if(SAM3_CPU)
	file(GLOB_RECURSE SAM3_CPU_SOURCES "src/backend/cpu/*.c")
	list(APPEND SAM3_SOURCES ${SAM3_CPU_SOURCES})
	add_definitions(-DSAM3_HAS_CPU)
endif()

if(SAM3_METAL)
	file(GLOB_RECURSE SAM3_METAL_SOURCES "src/backend/metal/*.c")
	list(APPEND SAM3_SOURCES ${SAM3_METAL_SOURCES})
	add_definitions(-DSAM3_HAS_METAL)
	find_library(METAL_FRAMEWORK Metal REQUIRED)
	find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
endif()

# Static library
add_library(sam3 STATIC ${SAM3_SOURCES})

if(SAM3_METAL)
	target_link_libraries(sam3 ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
endif()

# CLI tools
add_executable(sam3_main tools/sam3_main.c)
target_link_libraries(sam3_main sam3)

add_executable(sam3_convert tools/sam3_convert.c)
target_link_libraries(sam3_convert sam3)

# Tests
if(SAM3_TESTS)
	enable_testing()
	file(GLOB TEST_SOURCES "tests/test_*.c")
	foreach(test_src ${TEST_SOURCES})
		get_filename_component(test_name ${test_src} NAME_WE)
		add_executable(${test_name} ${test_src})
		target_link_libraries(${test_name} sam3)
		add_test(NAME ${test_name} COMMAND ${test_name})
	endforeach()
endif()
```

**Step 2: Commit**

```bash
git add CMakeLists.txt
git commit -m "Add root CMakeLists.txt with Metal/CPU backend options"
```

---

### Task 4: Public API Headers

**Files:**
- Create: `include/sam3/sam3_types.h`
- Create: `include/sam3/sam3.h`

**Step 1: Create sam3_types.h**

```c
/*
 * include/sam3/sam3_types.h - Public type definitions for sam3
 *
 * Defines all public-facing types: error codes, tensor descriptors,
 * configuration structs, and inference results. This header is included
 * by sam3.h and should not be included directly by users.
 *
 * Key types:  sam3_error, sam3_dtype, sam3_tensor_desc, sam3_mask_result
 * Depends on: <stdint.h>, <stddef.h>
 * Used by:    sam3.h, all internal modules
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_TYPES_H
#define SAM3_TYPES_H

#include <stdint.h>
#include <stddef.h>

#define SAM3_MAX_DIMS 4

/* Error codes returned by all sam3 public functions. */
enum sam3_error {
	SAM3_OK       =  0,
	SAM3_EINVAL   = -1,  /* Invalid argument */
	SAM3_ENOMEM   = -2,  /* Out of memory */
	SAM3_EIO      = -3,  /* I/O error (file read, Metal shader) */
	SAM3_EBACKEND = -4,  /* Backend initialization failed */
	SAM3_EMODEL   = -5,  /* Model format error */
};

/* Supported tensor data types. */
enum sam3_dtype {
	SAM3_DTYPE_F32,
	SAM3_DTYPE_F16,
	SAM3_DTYPE_BF16,
	SAM3_DTYPE_I32,
	SAM3_DTYPE_I8,
};

/* Prompt type for segmentation. */
enum sam3_prompt_type {
	SAM3_PROMPT_POINT,
	SAM3_PROMPT_BOX,
	SAM3_PROMPT_MASK,
};

/* A 2D point prompt (x, y) with label (foreground=1, background=0). */
struct sam3_point {
	float x;
	float y;
	int   label;
};

/* A bounding box prompt. */
struct sam3_box {
	float x1;
	float y1;
	float x2;
	float y2;
};

/* Segmentation prompt (union of point, box, or mask input). */
struct sam3_prompt {
	enum sam3_prompt_type type;
	union {
		struct sam3_point point;
		struct sam3_box   box;
		/* Mask input: pointer to H*W float array */
		struct {
			const float *data;
			int          width;
			int          height;
		} mask;
	};
};

/* Result of a segmentation inference. */
struct sam3_result {
	float *masks;        /* Output masks: n_masks * H * W floats */
	float *iou_scores;   /* IoU score per mask */
	int    n_masks;
	int    mask_height;
	int    mask_width;
};

/* Model configuration loaded from weights file. */
struct sam3_model_config {
	int image_size;       /* Input image size (e.g., 1024) */
	int encoder_dim;      /* Image encoder embedding dimension */
	int decoder_dim;      /* Mask decoder dimension */
	int n_encoder_layers;
	int n_decoder_layers;
};

/* Opaque context handle. */
typedef struct sam3_ctx sam3_ctx;

#endif /* SAM3_TYPES_H */
```

**Step 2: Create sam3.h**

```c
/*
 * include/sam3/sam3.h - Main public API for sam3 inference
 *
 * This is the only header users need to include. Provides functions to
 * load a SAM3 model, run segmentation inference with point/box/mask
 * prompts, and free resources. All functions are thread-safe with
 * respect to different sam3_ctx instances.
 *
 * Key types:  sam3_ctx (opaque handle)
 * Depends on: sam3_types.h
 * Used by:    tools/sam3_main.c, user applications
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_H
#define SAM3_H

#include "sam3_types.h"

/*
 * sam3_init - Create and initialize a sam3 context.
 *
 * Initializes the compute backend and prepares for model loading.
 * Returns NULL on failure. Call sam3_free() when done.
 */
sam3_ctx *sam3_init(void);

/*
 * sam3_free - Release all resources held by a sam3 context.
 *
 * @ctx: Context to free (may be NULL).
 */
void sam3_free(sam3_ctx *ctx);

/*
 * sam3_load_model - Load SAM3 model weights from a file.
 *
 * @ctx:  Initialized context
 * @path: Path to model weights file
 *
 * Returns SAM3_OK on success. The context takes ownership of the
 * loaded weights and frees them in sam3_free().
 */
enum sam3_error sam3_load_model(sam3_ctx *ctx, const char *path);

/*
 * sam3_set_image - Set the input image for segmentation.
 *
 * @ctx:    Initialized context with loaded model
 * @pixels: RGB pixel data (H * W * 3 uint8_t values)
 * @width:  Image width in pixels
 * @height: Image height in pixels
 *
 * Runs the image encoder. Subsequent sam3_segment() calls reuse the
 * encoded image until a new image is set.
 */
enum sam3_error sam3_set_image(sam3_ctx *ctx, const uint8_t *pixels,
			       int width, int height);

/*
 * sam3_segment - Run segmentation with the given prompts.
 *
 * @ctx:      Context with image already set
 * @prompts:  Array of prompts (points, boxes, or masks)
 * @n_prompts: Number of prompts
 * @result:   Output result (caller must call sam3_result_free)
 *
 * Returns SAM3_OK on success. The result contains one or more predicted
 * masks with IoU confidence scores.
 */
enum sam3_error sam3_segment(sam3_ctx *ctx, const struct sam3_prompt *prompts,
			     int n_prompts, struct sam3_result *result);

/*
 * sam3_result_free - Free memory allocated in a sam3_result.
 *
 * @result: Result struct to free (fields set to NULL/0 after).
 */
void sam3_result_free(struct sam3_result *result);

/*
 * sam3_version - Return the sam3 version string.
 */
const char *sam3_version(void);

#endif /* SAM3_H */
```

**Step 3: Commit**

```bash
git add include/
git commit -m "Add public API headers (sam3.h, sam3_types.h)"
```

---

### Task 5: Core Module Skeletons

**Files:**
- Create: `src/core/tensor.h`
- Create: `src/core/tensor.c`
- Create: `src/core/alloc.h`
- Create: `src/core/alloc.c`
- Create: `src/core/graph.h`
- Create: `src/core/graph.c`

**Step 1: Create tensor.h**

```c
/*
 * src/core/tensor.h - Multi-dimensional tensor type
 *
 * Defines the core tensor struct used throughout sam3. Tensors are
 * dense, contiguous, row-major arrays with up to SAM3_MAX_DIMS
 * dimensions. They do not own their memory — allocation is handled
 * by the arena allocator in alloc.h.
 *
 * Key types:  sam3_tensor
 * Depends on: sam3/sam3_types.h
 * Used by:    graph.h, all model/ files, all backend/ files
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_CORE_TENSOR_H
#define SAM3_CORE_TENSOR_H

#include "sam3/sam3_types.h"

struct sam3_tensor {
	enum sam3_dtype dtype;
	int             n_dims;
	int             dims[SAM3_MAX_DIMS];
	int             strides[SAM3_MAX_DIMS];
	void           *data;
	size_t          nbytes;
};

/* Return the total number of elements in the tensor. */
int sam3_tensor_nelems(const struct sam3_tensor *t);

/* Return the size in bytes of one element of the given dtype. */
size_t sam3_dtype_size(enum sam3_dtype dtype);

/* Compute strides from dims (row-major). Fills t->strides. */
void sam3_tensor_compute_strides(struct sam3_tensor *t);

#endif /* SAM3_CORE_TENSOR_H */
```

**Step 2: Create tensor.c**

```c
/*
 * src/core/tensor.c - Tensor operations
 *
 * Implements element counting, dtype sizing, and stride computation
 * for the sam3_tensor type. These are low-level utilities used by
 * the arena allocator and compute graph.
 *
 * Key types:  sam3_tensor
 * Depends on: tensor.h
 * Used by:    alloc.c, graph.c, model/ files
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "tensor.h"

int sam3_tensor_nelems(const struct sam3_tensor *t)
{
	int n = 1;

	for (int i = 0; i < t->n_dims; i++)
		n *= t->dims[i];

	return n;
}

size_t sam3_dtype_size(enum sam3_dtype dtype)
{
	switch (dtype) {
	case SAM3_DTYPE_F32: return 4;
	case SAM3_DTYPE_F16: return 2;
	case SAM3_DTYPE_BF16: return 2;
	case SAM3_DTYPE_I32: return 4;
	case SAM3_DTYPE_I8:  return 1;
	}
	return 0;
}

void sam3_tensor_compute_strides(struct sam3_tensor *t)
{
	t->strides[t->n_dims - 1] = 1;
	for (int i = t->n_dims - 2; i >= 0; i--)
		t->strides[i] = t->strides[i + 1] * t->dims[i + 1];
}
```

**Step 3: Create alloc.h**

```c
/*
 * src/core/alloc.h - Arena memory allocator
 *
 * Provides a simple bump allocator for inference-time memory. All
 * tensor data is allocated from an arena, which is freed in one shot
 * when inference completes. This avoids per-tensor malloc/free overhead
 * and prevents memory fragmentation.
 *
 * Key types:  sam3_arena
 * Depends on: <stddef.h>
 * Used by:    tensor.c, graph.c, model/ files
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_CORE_ALLOC_H
#define SAM3_CORE_ALLOC_H

#include <stddef.h>
#include "sam3/sam3_types.h"

struct sam3_arena {
	void   *base;      /* Start of allocated region */
	size_t  size;      /* Total capacity in bytes */
	size_t  offset;    /* Current allocation offset */
};

/* Create an arena with the given capacity. Returns SAM3_OK or SAM3_ENOMEM. */
enum sam3_error sam3_arena_init(struct sam3_arena *arena, size_t capacity);

/* Allocate nbytes from the arena (16-byte aligned). Returns NULL if full. */
void *sam3_arena_alloc(struct sam3_arena *arena, size_t nbytes);

/* Reset the arena (frees all allocations but keeps the backing memory). */
void sam3_arena_reset(struct sam3_arena *arena);

/* Free the arena and its backing memory. */
void sam3_arena_free(struct sam3_arena *arena);

#endif /* SAM3_CORE_ALLOC_H */
```

**Step 4: Create alloc.c**

```c
/*
 * src/core/alloc.c - Arena allocator implementation
 *
 * Simple bump allocator. Allocations are 16-byte aligned for SIMD
 * compatibility. The arena uses a single malloc for its backing store
 * and never calls malloc again until freed.
 *
 * Key types:  sam3_arena
 * Depends on: alloc.h
 * Used by:    graph.c, model/ files
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include <string.h>

#include "alloc.h"

#define SAM3_ARENA_ALIGN 16

enum sam3_error sam3_arena_init(struct sam3_arena *arena, size_t capacity)
{
	arena->base = malloc(capacity);
	if (!arena->base)
		return SAM3_ENOMEM;

	arena->size = capacity;
	arena->offset = 0;
	return SAM3_OK;
}

void *sam3_arena_alloc(struct sam3_arena *arena, size_t nbytes)
{
	size_t aligned = (arena->offset + SAM3_ARENA_ALIGN - 1)
			 & ~(size_t)(SAM3_ARENA_ALIGN - 1);

	if (aligned + nbytes > arena->size)
		return NULL;

	void *ptr = (char *)arena->base + aligned;
	arena->offset = aligned + nbytes;
	memset(ptr, 0, nbytes);
	return ptr;
}

void sam3_arena_reset(struct sam3_arena *arena)
{
	arena->offset = 0;
}

void sam3_arena_free(struct sam3_arena *arena)
{
	free(arena->base);
	arena->base = NULL;
	arena->size = 0;
	arena->offset = 0;
}
```

**Step 5: Create graph.h**

```c
/*
 * src/core/graph.h - Compute graph for inference
 *
 * Represents a DAG of tensor operations. The model builds a compute
 * graph during setup, then the backend evaluates it. This allows
 * backends to optimize execution order, fuse operations, and manage
 * GPU command buffers.
 *
 * Key types:  sam3_graph, sam3_node, sam3_op
 * Depends on: tensor.h
 * Used by:    model/ files, backend/ files
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_CORE_GRAPH_H
#define SAM3_CORE_GRAPH_H

#include "tensor.h"

#define SAM3_GRAPH_MAX_NODES 4096
#define SAM3_NODE_MAX_INPUTS 4

/* Compute operation types. */
enum sam3_op {
	SAM3_OP_NONE,
	SAM3_OP_MATMUL,
	SAM3_OP_ADD,
	SAM3_OP_MUL,
	SAM3_OP_SOFTMAX,
	SAM3_OP_RELU,
	SAM3_OP_GELU,
	SAM3_OP_LAYERNORM,
	SAM3_OP_CONV2D,
	SAM3_OP_RESHAPE,
	SAM3_OP_TRANSPOSE,
};

/* A single node in the compute graph. */
struct sam3_node {
	enum sam3_op         op;
	struct sam3_tensor  *inputs[SAM3_NODE_MAX_INPUTS];
	int                  n_inputs;
	struct sam3_tensor  *output;
};

/* A compute graph: ordered list of nodes. */
struct sam3_graph {
	struct sam3_node nodes[SAM3_GRAPH_MAX_NODES];
	int              n_nodes;
};

/* Initialize an empty compute graph. */
void sam3_graph_init(struct sam3_graph *g);

/* Add a node to the graph. Returns pointer to the output tensor, or NULL. */
struct sam3_tensor *sam3_graph_add_op(struct sam3_graph *g, enum sam3_op op,
				     struct sam3_tensor **inputs, int n_inputs,
				     struct sam3_tensor *output);

#endif /* SAM3_CORE_GRAPH_H */
```

**Step 6: Create graph.c**

```c
/*
 * src/core/graph.c - Compute graph construction
 *
 * Builds the compute graph by appending nodes. The graph is a simple
 * linear array — topological ordering is the responsibility of the
 * caller (model code builds nodes in evaluation order).
 *
 * Key types:  sam3_graph, sam3_node
 * Depends on: graph.h
 * Used by:    model/ files
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <string.h>

#include "graph.h"

void sam3_graph_init(struct sam3_graph *g)
{
	memset(g, 0, sizeof(*g));
}

struct sam3_tensor *sam3_graph_add_op(struct sam3_graph *g, enum sam3_op op,
				     struct sam3_tensor **inputs, int n_inputs,
				     struct sam3_tensor *output)
{
	if (g->n_nodes >= SAM3_GRAPH_MAX_NODES)
		return NULL;

	if (n_inputs > SAM3_NODE_MAX_INPUTS)
		return NULL;

	struct sam3_node *node = &g->nodes[g->n_nodes++];
	node->op = op;
	node->n_inputs = n_inputs;
	node->output = output;

	for (int i = 0; i < n_inputs; i++)
		node->inputs[i] = inputs[i];

	return output;
}
```

**Step 7: Commit**

```bash
git add src/core/
git commit -m "core: add tensor, arena allocator, and compute graph skeletons"
```

---

### Task 6: Backend Abstraction and Stubs

**Files:**
- Create: `src/backend/backend.h`
- Create: `src/backend/cpu/cpu_backend.h`
- Create: `src/backend/cpu/cpu_backend.c`
- Create: `src/backend/metal/metal_backend.h`
- Create: `src/backend/metal/metal_backend.c`

**Step 1: Create backend.h (vtable interface)**

```c
/*
 * src/backend/backend.h - Backend abstraction layer
 *
 * Defines the vtable interface that all compute backends must implement.
 * Model code calls backend operations through this interface, never
 * directly. This allows runtime backend selection and makes it trivial
 * to add new backends (CUDA, Vulkan) without changing model code.
 *
 * Key types:  sam3_backend, sam3_backend_ops
 * Depends on: core/graph.h
 * Used by:    model/ files, tools/
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_BACKEND_H
#define SAM3_BACKEND_H

#include "core/graph.h"

/* Backend type identifiers. */
enum sam3_backend_type {
	SAM3_BACKEND_CPU,
	SAM3_BACKEND_METAL,
};

struct sam3_backend;

/* Operations vtable that every backend must implement. */
struct sam3_backend_ops {
	/* Initialize backend resources. */
	enum sam3_error (*init)(struct sam3_backend *be);

	/* Free backend resources. */
	void (*free)(struct sam3_backend *be);

	/* Allocate a tensor buffer on this backend's memory. */
	enum sam3_error (*alloc_tensor)(struct sam3_backend *be,
				       struct sam3_tensor *t);

	/* Evaluate a compute graph on this backend. */
	enum sam3_error (*graph_eval)(struct sam3_backend *be,
				     struct sam3_graph *g);
};

/* Backend instance. Backends embed this as first member. */
struct sam3_backend {
	enum sam3_backend_type    type;
	const struct sam3_backend_ops *ops;
};

/*
 * sam3_backend_init - Create and initialize a backend.
 *
 * @type: Which backend to create.
 *
 * Returns a heap-allocated backend or NULL on failure.
 * Caller must call sam3_backend_free() when done.
 */
struct sam3_backend *sam3_backend_init(enum sam3_backend_type type);

/* Free a backend created by sam3_backend_init. */
void sam3_backend_free(struct sam3_backend *be);

#endif /* SAM3_BACKEND_H */
```

**Step 2: Create cpu_backend.h**

```c
/*
 * src/backend/cpu/cpu_backend.h - CPU compute backend
 *
 * CPU fallback backend using scalar and SIMD operations. Implements
 * all operations in the sam3_backend_ops vtable. Primarily used for
 * testing and as a reference implementation for validating GPU backends.
 *
 * Key types:  sam3_cpu_backend
 * Depends on: backend/backend.h
 * Used by:    backend.h (via sam3_backend_init)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_BACKEND_CPU_H
#define SAM3_BACKEND_CPU_H

#include "backend/backend.h"

struct sam3_cpu_backend {
	struct sam3_backend base;  /* Must be first member */
};

/* Get the CPU backend ops vtable. */
const struct sam3_backend_ops *sam3_cpu_backend_ops(void);

#endif /* SAM3_BACKEND_CPU_H */
```

**Step 3: Create cpu_backend.c**

```c
/*
 * src/backend/cpu/cpu_backend.c - CPU backend implementation
 *
 * Stub implementation of the CPU compute backend. Each operation
 * will be implemented as needed with scalar code first, then
 * optimized with SIMD (NEON on ARM, AVX2 on x86) later.
 *
 * Key types:  sam3_cpu_backend
 * Depends on: cpu_backend.h
 * Used by:    backend.h (registered at init)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "cpu_backend.h"
#include "util/log.h"

static enum sam3_error cpu_init(struct sam3_backend *be)
{
	(void)be;
	sam3_log_info("CPU backend initialized");
	return SAM3_OK;
}

static void cpu_free(struct sam3_backend *be)
{
	(void)be;
}

static enum sam3_error cpu_alloc_tensor(struct sam3_backend *be,
					struct sam3_tensor *t)
{
	(void)be;
	(void)t;
	/* TODO: allocate from CPU arena */
	return SAM3_OK;
}

static enum sam3_error cpu_graph_eval(struct sam3_backend *be,
				     struct sam3_graph *g)
{
	(void)be;
	(void)g;
	/* TODO: iterate nodes, dispatch to CPU kernels */
	return SAM3_OK;
}

static const struct sam3_backend_ops cpu_ops = {
	.init         = cpu_init,
	.free         = cpu_free,
	.alloc_tensor = cpu_alloc_tensor,
	.graph_eval   = cpu_graph_eval,
};

const struct sam3_backend_ops *sam3_cpu_backend_ops(void)
{
	return &cpu_ops;
}
```

**Step 4: Create metal_backend.h**

```c
/*
 * src/backend/metal/metal_backend.h - Metal compute backend
 *
 * GPU compute backend using Apple's Metal API. Compiles compute
 * shaders at init time, dispatches graph operations as Metal
 * command buffers. This is the primary backend for sam3 on macOS
 * and iOS.
 *
 * Key types:  sam3_metal_backend
 * Depends on: backend/backend.h
 * Used by:    backend.h (via sam3_backend_init)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_BACKEND_METAL_H
#define SAM3_BACKEND_METAL_H

#include "backend/backend.h"

struct sam3_metal_backend {
	struct sam3_backend base;  /* Must be first member */
	void *device;              /* id<MTLDevice> — stored as void* for C compat */
	void *command_queue;       /* id<MTLCommandQueue> */
	void *library;             /* id<MTLLibrary> — compiled shaders */
};

/* Get the Metal backend ops vtable. */
const struct sam3_backend_ops *sam3_metal_backend_ops(void);

#endif /* SAM3_BACKEND_METAL_H */
```

**Step 5: Create metal_backend.c**

```c
/*
 * src/backend/metal/metal_backend.c - Metal backend implementation
 *
 * Stub implementation of the Metal compute backend. Metal API calls
 * require Objective-C, so the actual Metal code will live in .m files.
 * This C file provides the vtable entry points and delegates to the
 * Objective-C implementation.
 *
 * Key types:  sam3_metal_backend
 * Depends on: metal_backend.h
 * Used by:    backend.h (registered at init)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "metal_backend.h"
#include "util/log.h"

#ifdef SAM3_HAS_METAL

static enum sam3_error metal_init(struct sam3_backend *be)
{
	(void)be;
	sam3_log_info("Metal backend initialized (stub)");
	/* TODO: create MTLDevice, compile shader library */
	return SAM3_OK;
}

static void metal_free(struct sam3_backend *be)
{
	(void)be;
	/* TODO: release Metal objects */
}

static enum sam3_error metal_alloc_tensor(struct sam3_backend *be,
					  struct sam3_tensor *t)
{
	(void)be;
	(void)t;
	/* TODO: create MTLBuffer for tensor data */
	return SAM3_OK;
}

static enum sam3_error metal_graph_eval(struct sam3_backend *be,
					struct sam3_graph *g)
{
	(void)be;
	(void)g;
	/* TODO: encode compute commands, commit, wait */
	return SAM3_OK;
}

static const struct sam3_backend_ops metal_ops = {
	.init         = metal_init,
	.free         = metal_free,
	.alloc_tensor = metal_alloc_tensor,
	.graph_eval   = metal_graph_eval,
};

const struct sam3_backend_ops *sam3_metal_backend_ops(void)
{
	return &metal_ops;
}

#else /* !SAM3_HAS_METAL */

const struct sam3_backend_ops *sam3_metal_backend_ops(void)
{
	return NULL;
}

#endif /* SAM3_HAS_METAL */
```

**Step 6: Commit**

```bash
git add src/backend/
git commit -m "backend: add vtable interface with Metal and CPU stubs"
```

---

### Task 7: Utility Modules

**Files:**
- Create: `src/util/log.h`
- Create: `src/util/log.c`
- Create: `src/util/error.h`
- Create: `src/util/error.c`

**Step 1: Create log.h**

```c
/*
 * src/util/log.h - Logging subsystem
 *
 * Simple leveled logging (debug, info, warn, error) to stderr.
 * Log level is set at runtime. All log output includes the source
 * file and line number for easy debugging.
 *
 * Key types:  sam3_log_level
 * Depends on: <stdio.h>
 * Used by:    all modules
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_UTIL_LOG_H
#define SAM3_UTIL_LOG_H

enum sam3_log_level {
	SAM3_LOG_DEBUG,
	SAM3_LOG_INFO,
	SAM3_LOG_WARN,
	SAM3_LOG_ERROR,
};

/* Set the minimum log level. Messages below this level are suppressed. */
void sam3_log_set_level(enum sam3_log_level level);

/* Internal log function — use the macros below instead. */
void sam3_log_write(enum sam3_log_level level, const char *file, int line,
		    const char *fmt, ...);

#define sam3_log_debug(...) \
	sam3_log_write(SAM3_LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define sam3_log_info(...) \
	sam3_log_write(SAM3_LOG_INFO, __FILE__, __LINE__, __VA_ARGS__)
#define sam3_log_warn(...) \
	sam3_log_write(SAM3_LOG_WARN, __FILE__, __LINE__, __VA_ARGS__)
#define sam3_log_error(...) \
	sam3_log_write(SAM3_LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)

#endif /* SAM3_UTIL_LOG_H */
```

**Step 2: Create log.c**

```c
/*
 * src/util/log.c - Logging implementation
 *
 * Writes formatted log messages to stderr with level prefix and
 * source location. Thread-safe via fprintf atomicity guarantee.
 *
 * Key types:  sam3_log_level
 * Depends on: log.h
 * Used by:    all modules
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdarg.h>

#include "log.h"

static enum sam3_log_level g_log_level = SAM3_LOG_INFO;

static const char *level_str(enum sam3_log_level level)
{
	switch (level) {
	case SAM3_LOG_DEBUG: return "DEBUG";
	case SAM3_LOG_INFO:  return "INFO";
	case SAM3_LOG_WARN:  return "WARN";
	case SAM3_LOG_ERROR: return "ERROR";
	}
	return "?";
}

void sam3_log_set_level(enum sam3_log_level level)
{
	g_log_level = level;
}

void sam3_log_write(enum sam3_log_level level, const char *file, int line,
		    const char *fmt, ...)
{
	if (level < g_log_level)
		return;

	va_list ap;
	va_start(ap, fmt);
	fprintf(stderr, "[%s] %s:%d: ", level_str(level), file, line);
	vfprintf(stderr, fmt, ap);
	fprintf(stderr, "\n");
	va_end(ap);
}
```

**Step 3: Create error.h**

```c
/*
 * src/util/error.h - Error code utilities
 *
 * Provides human-readable error messages for sam3_error codes.
 * All sam3 error codes are defined in sam3_types.h; this module
 * only provides string conversion.
 *
 * Key types:  (uses sam3_error from sam3_types.h)
 * Depends on: sam3/sam3_types.h
 * Used by:    tools/, user applications
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_UTIL_ERROR_H
#define SAM3_UTIL_ERROR_H

#include "sam3/sam3_types.h"

/* Return a human-readable string for the given error code. */
const char *sam3_error_str(enum sam3_error err);

#endif /* SAM3_UTIL_ERROR_H */
```

**Step 4: Create error.c**

```c
/*
 * src/util/error.c - Error string conversion
 *
 * Maps sam3_error codes to descriptive strings for logging and
 * user-facing error messages.
 *
 * Key types:  (uses sam3_error from sam3_types.h)
 * Depends on: error.h
 * Used by:    tools/, user applications
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "error.h"

const char *sam3_error_str(enum sam3_error err)
{
	switch (err) {
	case SAM3_OK:       return "success";
	case SAM3_EINVAL:   return "invalid argument";
	case SAM3_ENOMEM:   return "out of memory";
	case SAM3_EIO:      return "I/O error";
	case SAM3_EBACKEND: return "backend initialization failed";
	case SAM3_EMODEL:   return "model format error";
	}
	return "unknown error";
}
```

**Step 5: Commit**

```bash
git add src/util/
git commit -m "util: add logging subsystem and error string conversion"
```

---

### Task 8: Model Layer Skeletons

**Files:**
- Create: `src/model/image_encoder.h` and `.c`
- Create: `src/model/prompt_encoder.h` and `.c`
- Create: `src/model/mask_decoder.h` and `.c`
- Create: `src/model/memory_attn.h` and `.c`

Each file follows the documentation header convention. Implementation is all stubs — just the function signatures with `(void)` casts and `return SAM3_OK`.

**Step 1: Create image_encoder.h**

```c
/*
 * src/model/image_encoder.h - SAM3 image encoder (Hiera backbone)
 *
 * The image encoder is a hierarchical vision transformer (Hiera) that
 * processes the input image into multi-scale feature maps. It runs once
 * per image and its output is reused across multiple prompt/segment calls.
 *
 * Key types:  sam3_image_encoder
 * Depends on: core/tensor.h, core/graph.h, core/alloc.h
 * Used by:    sam3.h (via sam3_set_image)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_IMAGE_ENCODER_H
#define SAM3_MODEL_IMAGE_ENCODER_H

#include "core/tensor.h"
#include "core/graph.h"
#include "core/alloc.h"

struct sam3_image_encoder {
	struct sam3_tensor *patch_embed_weight;
	struct sam3_tensor *pos_embed;
	int                n_layers;
	int                embed_dim;
	/* TODO: per-layer weights */
};

/* Build the image encoder subgraph. */
enum sam3_error sam3_image_encoder_build(struct sam3_image_encoder *enc,
					struct sam3_graph *g,
					struct sam3_tensor *input_image,
					struct sam3_tensor *output_features,
					struct sam3_arena *arena);

#endif /* SAM3_MODEL_IMAGE_ENCODER_H */
```

**Step 2: Create image_encoder.c**

```c
/*
 * src/model/image_encoder.c - Image encoder graph construction
 *
 * Builds the compute graph for the Hiera vision transformer backbone.
 * The encoder produces multi-scale feature maps that are consumed by
 * the prompt encoder and mask decoder.
 *
 * Key types:  sam3_image_encoder
 * Depends on: image_encoder.h
 * Used by:    sam3.c (top-level API)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "image_encoder.h"

enum sam3_error sam3_image_encoder_build(struct sam3_image_encoder *enc,
					struct sam3_graph *g,
					struct sam3_tensor *input_image,
					struct sam3_tensor *output_features,
					struct sam3_arena *arena)
{
	(void)enc;
	(void)g;
	(void)input_image;
	(void)output_features;
	(void)arena;
	/* TODO: patch embedding -> Hiera blocks -> multi-scale output */
	return SAM3_OK;
}
```

**Step 3: Create prompt_encoder.h**

```c
/*
 * src/model/prompt_encoder.h - SAM3 prompt encoder
 *
 * Encodes user prompts (points, boxes, masks) into embeddings that
 * condition the mask decoder. Points and boxes are encoded as learned
 * positional embeddings; mask prompts are downscaled and convolved.
 *
 * Key types:  sam3_prompt_encoder
 * Depends on: core/tensor.h, core/graph.h, sam3/sam3_types.h
 * Used by:    sam3.h (via sam3_segment)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_PROMPT_ENCODER_H
#define SAM3_MODEL_PROMPT_ENCODER_H

#include "core/tensor.h"
#include "core/graph.h"
#include "core/alloc.h"
#include "sam3/sam3_types.h"

struct sam3_prompt_encoder {
	struct sam3_tensor *point_embeddings;
	struct sam3_tensor *not_a_point_embed;
	struct sam3_tensor *mask_downscale_weights;
	int                embed_dim;
};

/* Build the prompt encoder subgraph. */
enum sam3_error sam3_prompt_encoder_build(struct sam3_prompt_encoder *pe,
					 struct sam3_graph *g,
					 const struct sam3_prompt *prompts,
					 int n_prompts,
					 struct sam3_tensor *output_sparse,
					 struct sam3_tensor *output_dense,
					 struct sam3_arena *arena);

#endif /* SAM3_MODEL_PROMPT_ENCODER_H */
```

**Step 4: Create prompt_encoder.c**

```c
/*
 * src/model/prompt_encoder.c - Prompt encoder graph construction
 *
 * Builds the compute graph for encoding user prompts into sparse
 * (point/box) and dense (mask) embeddings.
 *
 * Key types:  sam3_prompt_encoder
 * Depends on: prompt_encoder.h
 * Used by:    sam3.c (top-level API)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "prompt_encoder.h"

enum sam3_error sam3_prompt_encoder_build(struct sam3_prompt_encoder *pe,
					 struct sam3_graph *g,
					 const struct sam3_prompt *prompts,
					 int n_prompts,
					 struct sam3_tensor *output_sparse,
					 struct sam3_tensor *output_dense,
					 struct sam3_arena *arena)
{
	(void)pe;
	(void)g;
	(void)prompts;
	(void)n_prompts;
	(void)output_sparse;
	(void)output_dense;
	(void)arena;
	/* TODO: encode points/boxes as positional embeddings */
	return SAM3_OK;
}
```

**Step 5: Create mask_decoder.h**

```c
/*
 * src/model/mask_decoder.h - SAM3 mask decoder
 *
 * Transformer-based decoder that takes image features and prompt
 * embeddings to predict segmentation masks and IoU scores. Uses
 * two-way cross-attention between prompt tokens and image features.
 *
 * Key types:  sam3_mask_decoder
 * Depends on: core/tensor.h, core/graph.h
 * Used by:    sam3.h (via sam3_segment)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_MASK_DECODER_H
#define SAM3_MODEL_MASK_DECODER_H

#include "core/tensor.h"
#include "core/graph.h"
#include "core/alloc.h"

struct sam3_mask_decoder {
	int n_layers;
	int embed_dim;
	int n_heads;
	/* TODO: per-layer transformer weights, output MLP weights */
};

/* Build the mask decoder subgraph. */
enum sam3_error sam3_mask_decoder_build(struct sam3_mask_decoder *dec,
				       struct sam3_graph *g,
				       struct sam3_tensor *image_features,
				       struct sam3_tensor *sparse_prompts,
				       struct sam3_tensor *dense_prompts,
				       struct sam3_tensor *output_masks,
				       struct sam3_tensor *output_iou,
				       struct sam3_arena *arena);

#endif /* SAM3_MODEL_MASK_DECODER_H */
```

**Step 6: Create mask_decoder.c**

```c
/*
 * src/model/mask_decoder.c - Mask decoder graph construction
 *
 * Builds the compute graph for the transformer mask decoder. Cross-
 * attends prompt tokens to image features, then upscales to produce
 * full-resolution masks.
 *
 * Key types:  sam3_mask_decoder
 * Depends on: mask_decoder.h
 * Used by:    sam3.c (top-level API)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "mask_decoder.h"

enum sam3_error sam3_mask_decoder_build(struct sam3_mask_decoder *dec,
				       struct sam3_graph *g,
				       struct sam3_tensor *image_features,
				       struct sam3_tensor *sparse_prompts,
				       struct sam3_tensor *dense_prompts,
				       struct sam3_tensor *output_masks,
				       struct sam3_tensor *output_iou,
				       struct sam3_arena *arena)
{
	(void)dec;
	(void)g;
	(void)image_features;
	(void)sparse_prompts;
	(void)dense_prompts;
	(void)output_masks;
	(void)output_iou;
	(void)arena;
	/* TODO: two-way transformer -> upscale -> MLP -> masks + IoU */
	return SAM3_OK;
}
```

**Step 7: Create memory_attn.h**

```c
/*
 * src/model/memory_attn.h - Memory attention for video tracking
 *
 * Implements the memory attention mechanism that allows SAM3 to track
 * objects across video frames. Maintains a memory bank of past frame
 * features and attends to them when processing new frames.
 *
 * Key types:  sam3_memory_attn
 * Depends on: core/tensor.h, core/graph.h
 * Used by:    sam3.h (future video API)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_MEMORY_ATTN_H
#define SAM3_MODEL_MEMORY_ATTN_H

#include "core/tensor.h"
#include "core/graph.h"
#include "core/alloc.h"

struct sam3_memory_attn {
	int embed_dim;
	int n_heads;
	int max_memory_frames;
	/* TODO: memory bank tensors, attention weights */
};

/* Build the memory attention subgraph for video tracking. */
enum sam3_error sam3_memory_attn_build(struct sam3_memory_attn *mem,
				      struct sam3_graph *g,
				      struct sam3_tensor *current_features,
				      struct sam3_tensor *memory_bank,
				      struct sam3_tensor *output,
				      struct sam3_arena *arena);

#endif /* SAM3_MODEL_MEMORY_ATTN_H */
```

**Step 8: Create memory_attn.c**

```c
/*
 * src/model/memory_attn.c - Memory attention implementation
 *
 * Builds the compute graph for cross-attending current frame features
 * to a bank of past frame features. Used for video object tracking.
 *
 * Key types:  sam3_memory_attn
 * Depends on: memory_attn.h
 * Used by:    sam3.c (future video API)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "memory_attn.h"

enum sam3_error sam3_memory_attn_build(struct sam3_memory_attn *mem,
				      struct sam3_graph *g,
				      struct sam3_tensor *current_features,
				      struct sam3_tensor *memory_bank,
				      struct sam3_tensor *output,
				      struct sam3_arena *arena)
{
	(void)mem;
	(void)g;
	(void)current_features;
	(void)memory_bank;
	(void)output;
	(void)arena;
	/* TODO: cross-attention with memory bank */
	return SAM3_OK;
}
```

**Step 9: Commit**

```bash
git add src/model/
git commit -m "model: add image encoder, prompt encoder, mask decoder, and memory attention skeletons"
```

---

### Task 9: CLI Tool Skeletons

**Files:**
- Create: `tools/sam3_main.c`
- Create: `tools/sam3_convert.c`

**Step 1: Create sam3_main.c**

```c
/*
 * tools/sam3_main.c - SAM3 inference CLI
 *
 * Main command-line tool for running SAM3 segmentation. Takes an image
 * and prompt coordinates, outputs segmentation masks. This is the
 * primary user-facing binary.
 *
 * Usage: sam3 -m <model> -i <image> -p <x,y,label> [-o <output>]
 *
 * Key types:  (uses sam3_ctx from sam3.h)
 * Depends on: sam3/sam3.h
 * Used by:    end users
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>

#include "sam3/sam3.h"
#include "util/error.h"

int main(int argc, char **argv)
{
	(void)argc;
	(void)argv;

	printf("sam3 inference tool v%s\n", sam3_version());
	printf("usage: sam3 -m <model> -i <image> -p <x,y,label>\n");
	printf("(not yet implemented)\n");

	return 0;
}
```

**Step 2: Create sam3_convert.c**

```c
/*
 * tools/sam3_convert.c - Model weight conversion tool
 *
 * Converts SAM3 model weights from PyTorch/safetensors format to
 * sam3's native binary format. Handles tensor layout transposition,
 * dtype conversion, and weight name mapping.
 *
 * Usage: sam3_convert -i <input.pt> -o <output.sam3>
 *
 * Key types:  (standalone tool)
 * Depends on: sam3/sam3_types.h
 * Used by:    end users (pre-inference step)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>

#include "sam3/sam3_types.h"

int main(int argc, char **argv)
{
	(void)argc;
	(void)argv;

	printf("sam3 weight conversion tool\n");
	printf("usage: sam3_convert -i <input.pt> -o <output.sam3>\n");
	printf("(not yet implemented)\n");

	return 0;
}
```

**Step 3: Commit**

```bash
git add tools/
git commit -m "tools: add inference CLI and weight conversion tool skeletons"
```

---

### Task 10: Test Infrastructure and First Test

**Files:**
- Create: `tests/test_helpers.h`
- Create: `tests/test_tensor.c`

**Step 1: Create test_helpers.h**

```c
/*
 * tests/test_helpers.h - Minimal test assertion macros
 *
 * Provides ASSERT and ASSERT_EQ macros for unit tests. Each test file
 * is a standalone executable with a main() that calls test functions.
 * CTest discovers and runs them.
 *
 * Key types:  (macros only)
 * Depends on: <stdio.h>, <stdlib.h>
 * Used by:    all test_*.c files
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_TEST_HELPERS_H
#define SAM3_TEST_HELPERS_H

#include <stdio.h>
#include <stdlib.h>

static int tests_run = 0;
static int tests_failed = 0;

#define ASSERT(cond) do {						\
	tests_run++;							\
	if (!(cond)) {							\
		fprintf(stderr, "FAIL %s:%d: %s\n",			\
			__FILE__, __LINE__, #cond);			\
		tests_failed++;						\
	}								\
} while (0)

#define ASSERT_EQ(a, b) do {						\
	tests_run++;							\
	if ((a) != (b)) {						\
		fprintf(stderr, "FAIL %s:%d: %s != %s\n",		\
			__FILE__, __LINE__, #a, #b);			\
		tests_failed++;						\
	}								\
} while (0)

#define TEST_REPORT() do {						\
	printf("%d tests, %d failures\n", tests_run, tests_failed);	\
	return tests_failed ? 1 : 0;					\
} while (0)

#endif /* SAM3_TEST_HELPERS_H */
```

**Step 2: Create test_tensor.c**

```c
/*
 * tests/test_tensor.c - Unit tests for core tensor operations
 *
 * Tests tensor element counting, dtype sizing, and stride computation.
 * Run via: ctest --output-on-failure
 *
 * Key types:  sam3_tensor
 * Depends on: core/tensor.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "core/tensor.h"

static void test_tensor_nelems_1d(void)
{
	struct sam3_tensor t = { .n_dims = 1, .dims = {10} };
	ASSERT_EQ(sam3_tensor_nelems(&t), 10);
}

static void test_tensor_nelems_3d(void)
{
	struct sam3_tensor t = { .n_dims = 3, .dims = {3, 224, 224} };
	ASSERT_EQ(sam3_tensor_nelems(&t), 3 * 224 * 224);
}

static void test_dtype_size(void)
{
	ASSERT_EQ(sam3_dtype_size(SAM3_DTYPE_F32), 4);
	ASSERT_EQ(sam3_dtype_size(SAM3_DTYPE_F16), 2);
	ASSERT_EQ(sam3_dtype_size(SAM3_DTYPE_I8), 1);
}

static void test_tensor_strides(void)
{
	struct sam3_tensor t = { .n_dims = 3, .dims = {3, 4, 5} };
	sam3_tensor_compute_strides(&t);

	ASSERT_EQ(t.strides[2], 1);
	ASSERT_EQ(t.strides[1], 5);
	ASSERT_EQ(t.strides[0], 20);
}

int main(void)
{
	test_tensor_nelems_1d();
	test_tensor_nelems_3d();
	test_dtype_size();
	test_tensor_strides();

	TEST_REPORT();
}
```

**Step 3: Commit**

```bash
git add tests/
git commit -m "tests: add test helpers and tensor unit tests"
```

---

### Task 11: Create models/ Directory and Verify Build

**Files:**
- Create: `models/.gitkeep`

**Step 1: Create models directory placeholder**

```bash
mkdir -p models
touch models/.gitkeep
```

**Step 2: Build and run tests**

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(sysctl -n hw.ncpu)
ctest --output-on-failure
```

Expected: Build succeeds, test_tensor passes (4 tests, 0 failures).

**Step 3: Commit**

```bash
git add models/.gitkeep
git commit -m "Add models directory placeholder"
```

---

### Task 12: Final Verification

**Step 1: Run full build from clean state**

```bash
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(sysctl -n hw.ncpu) 2>&1
ctest --output-on-failure
```

Expected: Clean build, all tests pass.

**Step 2: Verify file headers**

Spot-check that every `.c` and `.h` file has the documentation header.

**Step 3: Verify git log**

```bash
git log --oneline
```

Expected: Clean commit history with one commit per logical unit.
