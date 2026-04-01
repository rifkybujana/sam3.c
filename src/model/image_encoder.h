/*
 * src/model/image_encoder.h - SAM3 image encoder (Hiera backbone)
 *
 * The image encoder is a hierarchical vision transformer (Hiera) that
 * processes the input image into multi-scale feature maps. It runs once
 * per image and its output is reused across multiple prompt/segment calls.
 *
 * Key types:  sam3_image_encoder
 * Depends on: core/tensor.h, core/graph.h, core/alloc.h
 * Used by:    sam3.h (via sam3_set_image)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_IMAGE_ENCODER_H
#define SAM3_MODEL_IMAGE_ENCODER_H

#include "core/tensor.h"
#include "core/graph.h"
#include "core/alloc.h"

struct sam3_image_encoder {
	struct sam3_tensor *patch_embed_weight;
	struct sam3_tensor *pos_embed;
	int                n_layers;
	int                embed_dim;
	/* TODO: per-layer weights */
};

/* Build the image encoder subgraph. */
enum sam3_error sam3_image_encoder_build(struct sam3_image_encoder *enc,
					struct sam3_graph *g,
					struct sam3_tensor *input_image,
					struct sam3_tensor *output_features,
					struct sam3_arena *arena);

#endif /* SAM3_MODEL_IMAGE_ENCODER_H */
