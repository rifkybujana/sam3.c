# SAM3 C Inference Engine — Project Setup Design

**Date**: 2026-03-30
**Status**: Approved

## Overview

Pure C inference engine for Facebook's Segment Anything Model 3 (SAM3).
Modeled after ggml/llama.cpp — kernel-style C with LLM-friendly documentation.

## Goals

- Reimplement SAM3 inference in pure C11
- Metal backend first, extensible architecture for CUDA/Vulkan later
- Linux kernel-grade code quality and documentation
- Every file self-documenting for LLM context

## Project Structure

```
sam3/
├── CLAUDE.md                    # Coding rules (kernel-style C, documentation conventions)
├── CMakeLists.txt               # Root build
├── include/sam3/                 # Public API
│   ├── sam3.h                   # Main inference API
│   └── sam3_types.h             # Public types (tensors, configs, results)
├── src/
│   ├── core/                    # Tensor ops, memory mgmt, compute graph
│   │   ├── tensor.c/.h
│   │   ├── alloc.c/.h
│   │   └── graph.c/.h
│   ├── backend/                 # Backend abstraction layer
│   │   ├── backend.h            # Backend interface (vtable)
│   │   ├── metal/               # Metal implementation
│   │   │   ├── metal_backend.c/.h
│   │   │   └── shaders/         # .metal shader files
│   │   └── cpu/                 # CPU fallback (stub for now)
│   │       └── cpu_backend.c/.h
│   ├── model/                   # SAM3 architecture
│   │   ├── image_encoder.c/.h   # ViT/Hiera encoder
│   │   ├── prompt_encoder.c/.h  # Point/box/mask prompt encoding
│   │   ├── mask_decoder.c/.h    # Transformer mask decoder
│   │   └── memory_attn.c/.h     # Memory attention (video tracking)
│   └── util/                    # Utilities
│       ├── log.c/.h             # Logging subsystem
│       └── error.c/.h           # Error codes and handling
├── tools/                       # CLI tools
│   ├── sam3_main.c              # Main inference CLI
│   └── sam3_convert.c           # Weight conversion tool
├── models/                      # .gitignored, model weights
├── tests/                       # Tests
│   └── test_tensor.c
└── docs/plans/                  # Design docs
```

## Coding Standards (CLAUDE.md)

### Style
- C11 standard, no C++ features
- 8-character tab indentation (kernel style)
- 80-column soft limit, 100 hard limit
- K&R brace style for functions, same-line for blocks
- `snake_case` for everything
- Prefix all public symbols with `sam3_`
- No Hungarian notation, no `typedef` hiding pointers

### File Documentation Header
Every `.c` and `.h` file starts with:
```c
/*
 * <path> - <one-line description>
 *
 * <2-4 sentence explanation of purpose and role in the system>
 *
 * Key types:  <main types defined/used>
 * Depends on: <direct dependencies>
 * Used by:    <known dependents>
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */
```

### Memory Management
- Arena allocators for inference (no malloc/free in hot paths)
- All allocations go through `sam3_alloc` interface
- No global mutable state

### Error Handling
- Return error codes (`enum sam3_error`)
- `goto cleanup` pattern for resource cleanup

### Backend Abstraction
- vtable-based dispatch (`struct sam3_backend_ops`)
- Each backend implements the same ops interface
- Runtime backend selection

## Build System

CMake with:
- `option(SAM3_METAL "Enable Metal backend" ON)` — auto-detected on macOS
- `option(SAM3_CPU "Enable CPU backend" ON)` — always available
- Future: `SAM3_CUDA`, `SAM3_VULKAN`
- C11 standard enforced
- `-Wall -Wextra -Wpedantic -Werror` in debug builds

## Initial Scaffold

Git repo, `.gitignore`, `CLAUDE.md`, `CMakeLists.txt`, and skeleton files
with proper documentation headers (declarations only, no implementations).
