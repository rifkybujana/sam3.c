/*
 * src/model/position_encoding.h - 2D sinusoidal position encoding
 *
 * Precomputes 2D sinusoidal position embeddings for the vision transformer.
 * The encoding is computed on the CPU once and added to image patch features
 * during inference. Each spatial position gets a unique embedding vector
 * composed of sin/cos signals at different frequencies for both axes.
 *
 * Key types:  sam3_pos_encoding
 * Depends on: core/tensor.h, core/alloc.h, sam3/sam3_types.h
 * Used by:    model/image_encoder.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_POSITION_ENCODING_H
#define SAM3_MODEL_POSITION_ENCODING_H

#include "core/tensor.h"
#include "core/alloc.h"
#include "sam3/sam3_types.h"

struct sam3_pos_encoding {
	int num_pos_feats;		/* 256 — half of the output channel dim */
	float temperature;		/* 10000.0f */
	struct sam3_tensor *cached;	/* precomputed [H, W, num_pos_feats*2] */
};

/*
 * sam3_pos_encoding_precompute - Generate 2D sinusoidal position encoding.
 *
 * @pe:             Position encoding struct (caller-allocated)
 * @height:         Grid height (e.g. 72 for 1008/14 patches)
 * @width:          Grid width
 * @num_pos_feats:  Number of position features per dimension (256)
 * @arena:          Arena for tensor allocation
 *
 * Precomputes [height, width, num_pos_feats*2] tensor.
 * The first half of the last dim encodes y position,
 * the second half encodes x position.
 *
 * Returns SAM3_OK on success, SAM3_EINVAL for bad arguments,
 * SAM3_ENOMEM if the arena is full.
 */
enum sam3_error sam3_pos_encoding_precompute(
	struct sam3_pos_encoding *pe,
	int height, int width, int num_pos_feats,
	struct sam3_arena *arena);

/* Get the precomputed position encoding tensor. */
struct sam3_tensor *sam3_pos_encoding_get(struct sam3_pos_encoding *pe);

#endif /* SAM3_MODEL_POSITION_ENCODING_H */
