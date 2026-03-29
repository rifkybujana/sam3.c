/*
 * src/model/image_encoder.c - Image encoder graph construction
 *
 * Builds the compute graph for the Hiera vision transformer backbone.
 * The encoder produces multi-scale feature maps that are consumed by
 * the prompt encoder and mask decoder.
 *
 * Key types:  sam3_image_encoder
 * Depends on: image_encoder.h
 * Used by:    sam3.c (top-level API)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "image_encoder.h"

enum sam3_error sam3_image_encoder_build(struct sam3_image_encoder *enc,
					struct sam3_graph *g,
					struct sam3_tensor *input_image,
					struct sam3_tensor *output_features,
					struct sam3_arena *arena)
{
	(void)enc;
	(void)g;
	(void)input_image;
	(void)output_features;
	(void)arena;
	/* TODO: patch embedding -> Hiera blocks -> multi-scale output */
	return SAM3_OK;
}
