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
 * Copyright (c) 2026 Rifky Bujana Bisri
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
