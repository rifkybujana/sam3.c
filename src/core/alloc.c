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
