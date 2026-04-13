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
 * Copyright (c) 2026 Rifky Bujana Bisri
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
	int     skip_data; /* When set, gh_alloc_tensor skips data allocation */
};

/* Create an arena with the given capacity. Returns SAM3_OK or SAM3_ENOMEM. */
enum sam3_error sam3_arena_init(struct sam3_arena *arena, size_t capacity);

/* Allocate nbytes from the arena (16-byte aligned, zero-filled). Returns NULL if full. */
void *sam3_arena_alloc(struct sam3_arena *arena, size_t nbytes);

/* Allocate nbytes from the arena (16-byte aligned, uninitialized). Returns NULL if full. */
void *sam3_arena_alloc_raw(struct sam3_arena *arena, size_t nbytes);

/* Reset the arena (frees all allocations but keeps the backing memory). */
void sam3_arena_reset(struct sam3_arena *arena);

/* Free the arena and its backing memory. */
void sam3_arena_free(struct sam3_arena *arena);


struct sam3_profiler;

/* Set the active profiler for memory tracking. NULL to disable. */
void sam3_arena_set_profiler(struct sam3_profiler *p);

#endif /* SAM3_CORE_ALLOC_H */
