/*
 * src/backend/cpu/cpu_dispatch.h - CPU kernel dispatch table interface
 *
 * Declares the dispatch entry point that routes graph nodes to the
 * correct dtype-specific kernel via a 2D static table indexed by
 * [op][dtype]. Centralises all dispatch logic so cpu_backend.c
 * reduces to a single call per node.
 *
 * Key types:  (function declaration only)
 * Depends on: core/graph.h, core/alloc.h, sam3/sam3_types.h
 * Used by:    cpu_backend.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_CPU_DISPATCH_H
#define SAM3_CPU_DISPATCH_H

#include "core/graph.h"
#include "core/alloc.h"
#include "sam3/sam3_types.h"

struct sam3_threadpool;

/*
 * cpu_dispatch_node - Dispatch a graph node to the correct dtype-specific kernel.
 *
 * Validates all inputs share the same dtype, looks up the kernel
 * in the [op][dtype] dispatch table, and calls it. Returns SAM3_EDTYPE
 * for dtype mismatches or unimplemented (op, dtype) combinations.
 */
enum sam3_error cpu_dispatch_node(const struct sam3_node *node,
				  struct sam3_arena *scratch,
				  struct sam3_threadpool *pool);

#endif /* SAM3_CPU_DISPATCH_H */
