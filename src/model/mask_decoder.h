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
