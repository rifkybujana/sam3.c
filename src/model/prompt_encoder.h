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
