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
