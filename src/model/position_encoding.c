/*
 * src/model/position_encoding.c - 2D sinusoidal position encoding
 *
 * Implements CPU-only precomputation of 2D sinusoidal position embeddings.
 * For each spatial position (y, x) on a grid, normalized coordinates are
 * mapped through sin/cos at geometrically spaced frequencies controlled
 * by the temperature parameter. This is a one-time cost at model load.
 *
 * Key types:  sam3_pos_encoding
 * Depends on: position_encoding.h, model/graph_helpers.h
 * Used by:    model/image_encoder.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "position_encoding.h"
#include "graph_helpers.h"
#include <math.h>

enum sam3_error sam3_pos_encoding_precompute(
	struct sam3_pos_encoding *pe,
	int height, int width, int num_pos_feats,
	struct sam3_arena *arena)
{
	if (!pe || !arena || height <= 0 || width <= 0 ||
	    num_pos_feats <= 0)
		return SAM3_EINVAL;

	pe->num_pos_feats = num_pos_feats;
	pe->temperature = 10000.0f;

	int out_dim = num_pos_feats * 2;
	pe->cached = gh_alloc_tensor(arena, SAM3_DTYPE_F32,
				     3, (int[]){height, width, out_dim});
	if (!pe->cached)
		return SAM3_ENOMEM;

	float *data = (float *)pe->cached->data;

	/*
	 * Python: cumsum([1,2,...,H]) / H * 2π → [(1/H)*2π, ..., 2π]
	 * We match this with (y+1)/H * 2π for y=0..H-1.
	 */
	const float scale = 2.0f * 3.14159265358979323846f;

	for (int y = 0; y < height; y++) {
		float y_pos = ((float)(y + 1)) / (float)height * scale;

		for (int x = 0; x < width; x++) {
			float x_pos = ((float)(x + 1)) / (float)width
				* scale;
			float *row = data + (y * width + x) * out_dim;

			for (int i = 0; i < num_pos_feats; i++) {
				/* Pairs share frequency: floor(i/2) */
				int pair = i / 2;
				float dim_t = powf(
					pe->temperature,
					2.0f * (float)pair /
					(float)num_pos_feats);

				/* Y dimension: first half */
				if (i % 2 == 0)
					row[i] = sinf(y_pos / dim_t);
				else
					row[i] = cosf(y_pos / dim_t);

				/* X dimension: second half */
				if (i % 2 == 0)
					row[num_pos_feats + i] =
						sinf(x_pos / dim_t);
				else
					row[num_pos_feats + i] =
						cosf(x_pos / dim_t);
			}
		}
	}

	return SAM3_OK;
}

struct sam3_tensor *sam3_pos_encoding_get(struct sam3_pos_encoding *pe)
{
	if (!pe)
		return NULL;

	/* Lazy init: compute on first access */
	if (!pe->cached && pe->arena) {
		sam3_pos_encoding_precompute(pe, pe->height, pe->width,
					     pe->num_pos_feats, pe->arena);
	}

	return pe->cached;
}
