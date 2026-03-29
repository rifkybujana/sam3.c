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
