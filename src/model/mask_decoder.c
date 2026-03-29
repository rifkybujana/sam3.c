/*
 * src/model/mask_decoder.c - Mask decoder graph construction
 *
 * Builds the compute graph for the transformer mask decoder. Cross-
 * attends prompt tokens to image features, then upscales to produce
 * full-resolution masks.
 *
 * Key types:  sam3_mask_decoder
 * Depends on: mask_decoder.h
 * Used by:    sam3.c (top-level API)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "mask_decoder.h"

enum sam3_error sam3_mask_decoder_build(struct sam3_mask_decoder *dec,
				       struct sam3_graph *g,
				       struct sam3_tensor *image_features,
				       struct sam3_tensor *sparse_prompts,
				       struct sam3_tensor *dense_prompts,
				       struct sam3_tensor *output_masks,
				       struct sam3_tensor *output_iou,
				       struct sam3_arena *arena)
{
	(void)dec;
	(void)g;
	(void)image_features;
	(void)sparse_prompts;
	(void)dense_prompts;
	(void)output_masks;
	(void)output_iou;
	(void)arena;
	/* TODO: two-way transformer -> upscale -> MLP -> masks + IoU */
	return SAM3_OK;
}
