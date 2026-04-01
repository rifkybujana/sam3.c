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
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include <string.h>

#include "alloc.h"
#include "util/profile.h"

#ifdef SAM3_HAS_PROFILE
static struct sam3_profiler *g_alloc_profiler;
#else
#define g_alloc_profiler NULL
#endif

#define SAM3_ARENA_ALIGN 16

enum sam3_error sam3_arena_init(struct sam3_arena *arena, size_t capacity)
{
	arena->base = malloc(capacity);
	if (!arena->base)
		return SAM3_ENOMEM;

	arena->size = capacity;
	arena->offset = 0;
	SAM3_PROF_MEM_ARENA(g_alloc_profiler);
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
	SAM3_PROF_MEM(g_alloc_profiler, nbytes);
	memset(ptr, 0, nbytes);
	return ptr;
}

void sam3_arena_reset(struct sam3_arena *arena)
{
	size_t freed = arena->offset;
	arena->offset = 0;
	SAM3_PROF_MEM_ARENA_RESET(g_alloc_profiler, freed);
	(void)freed;
}

void sam3_arena_free(struct sam3_arena *arena)
{
	free(arena->base);
	arena->base = NULL;
	arena->size = 0;
	arena->offset = 0;
}

void sam3_arena_set_profiler(struct sam3_profiler *p)
{
#ifdef SAM3_HAS_PROFILE
	g_alloc_profiler = p;
#else
	(void)p;
#endif
}
