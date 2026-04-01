/*
 * src/core/quant.h - Block-quantized INT8 (Q8_0) types and utilities
 *
 * Defines the Q8_0 block format: 32 int8 values with a single f32 scale
 * factor per block. Provides quantize (f32 -> Q8_0) and dequantize
 * (Q8_0 -> f32) functions with NEON SIMD acceleration on ARM.
 *
 * Key types:  sam3_q8_block
 * Depends on: <stdint.h>, <stddef.h>
 * Used by:    cpu_matmul_q8.c, weight.c, sam3_convert.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_CORE_QUANT_H
#define SAM3_CORE_QUANT_H

#include <stdint.h>
#include <stddef.h>

#define SAM3_Q8_BLOCK_SIZE 32

/*
 * Q8_0 block: symmetric per-block quantization.
 * scale = max(abs(values[0..31])) / 127
 * data[i] = round(values[i] / scale), clamped to [-127, 127]
 * Dequant: float_val = data[i] * scale
 */
struct sam3_q8_block {
	float   scale;
	int8_t  data[SAM3_Q8_BLOCK_SIZE];
};

/* Returns number of Q8 blocks needed for nelems f32 elements. */
int sam3_q8_block_count(int nelems);

/* Returns total byte size for nelems elements stored as Q8_0. */
size_t sam3_q8_nbytes(int nelems);

/*
 * sam3_q8_quantize - Quantize f32 array to Q8_0 blocks.
 *
 * @src:    Source f32 values
 * @dst:    Destination Q8 blocks (caller-allocated, sam3_q8_block_count() blocks)
 * @nelems: Number of f32 elements. Tail block is zero-padded.
 */
void sam3_q8_quantize(const float *src, struct sam3_q8_block *dst,
		      int nelems);

/*
 * sam3_q8_dequantize - Dequantize Q8_0 blocks to f32 array.
 *
 * @src:    Source Q8 blocks
 * @dst:    Destination f32 values (caller-allocated, nelems floats)
 * @nelems: Number of f32 elements to produce.
 */
void sam3_q8_dequantize(const struct sam3_q8_block *src, float *dst,
			int nelems);

#endif /* SAM3_CORE_QUANT_H */
