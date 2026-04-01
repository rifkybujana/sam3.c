/*
 * src/core/quant.c - Q8_0 quantization and dequantization
 *
 * Implements symmetric per-block INT8 quantization. Each block of 32
 * float values is quantized with a single scale factor. NEON SIMD paths
 * accelerate both quantize and dequantize on ARM; scalar fallback is
 * provided for other platforms.
 *
 * Key types:  sam3_q8_block
 * Depends on: core/quant.h
 * Used by:    cpu_matmul_q8.c, tools/sam3_convert.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "quant.h"

#include <math.h>
#include <string.h>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define SAM3_Q8_HAS_NEON 1
#else
#define SAM3_Q8_HAS_NEON 0
#endif

int sam3_q8_block_count(int nelems)
{
	return (nelems + SAM3_Q8_BLOCK_SIZE - 1) / SAM3_Q8_BLOCK_SIZE;
}

size_t sam3_q8_nbytes(int nelems)
{
	return (size_t)sam3_q8_block_count(nelems) *
	       sizeof(struct sam3_q8_block);
}

/* -- Quantize ---------------------------------------------------------- */

#if SAM3_Q8_HAS_NEON

static float block_absmax_neon(const float *src, int n)
{
	float32x4_t vmax = vdupq_n_f32(0.0f);
	int i = 0;

	for (; i + 4 <= n; i += 4) {
		float32x4_t v = vld1q_f32(src + i);
		vmax = vmaxq_f32(vmax, vabsq_f32(v));
	}

	float amax = vmaxvq_f32(vmax);

	for (; i < n; i++) {
		float a = fabsf(src[i]);
		if (a > amax)
			amax = a;
	}

	return amax;
}

#endif /* SAM3_Q8_HAS_NEON */

void sam3_q8_quantize(const float *src, struct sam3_q8_block *dst,
		      int nelems)
{
	int nblocks = sam3_q8_block_count(nelems);

	for (int b = 0; b < nblocks; b++) {
		int offset = b * SAM3_Q8_BLOCK_SIZE;
		int count = nelems - offset;
		if (count > SAM3_Q8_BLOCK_SIZE)
			count = SAM3_Q8_BLOCK_SIZE;

		/* Find absmax in this block */
		float amax;
#if SAM3_Q8_HAS_NEON
		amax = block_absmax_neon(src + offset, count);
#else
		amax = 0.0f;
		for (int i = 0; i < count; i++) {
			float a = fabsf(src[offset + i]);
			if (a > amax)
				amax = a;
		}
#endif

		float scale = amax / 127.0f;
		dst[b].scale = scale;

		if (scale == 0.0f) {
			memset(dst[b].data, 0, SAM3_Q8_BLOCK_SIZE);
			continue;
		}

		float inv_scale = 1.0f / scale;
		int i = 0;

#if SAM3_Q8_HAS_NEON
		float32x4_t vinv = vdupq_n_f32(inv_scale);

		for (; i + 4 <= count; i += 4) {
			float32x4_t vf = vld1q_f32(src + offset + i);
			float32x4_t vs = vmulq_f32(vf, vinv);
			int32x4_t vi32 = vcvtnq_s32_f32(vs);
			int16x4_t vi16 = vqmovn_s32(vi32);
			int8x8_t vi8 = vqmovn_s16(
				vcombine_s16(vi16, vi16));
			/* Store lower 4 bytes */
			vst1_lane_s32((int32_t *)&dst[b].data[i],
				      vreinterpret_s32_s8(vi8), 0);
		}
#endif

		for (; i < count; i++) {
			float v = src[offset + i] * inv_scale;
			int q = (int)roundf(v);
			if (q > 127) q = 127;
			if (q < -127) q = -127;
			dst[b].data[i] = (int8_t)q;
		}

		/* Zero-pad tail if block is not full */
		for (int i2 = count; i2 < SAM3_Q8_BLOCK_SIZE; i2++)
			dst[b].data[i2] = 0;
	}
}

/* -- Dequantize -------------------------------------------------------- */

void sam3_q8_dequantize(const struct sam3_q8_block *src, float *dst,
			int nelems)
{
	int nblocks = sam3_q8_block_count(nelems);

	for (int b = 0; b < nblocks; b++) {
		int offset = b * SAM3_Q8_BLOCK_SIZE;
		int count = nelems - offset;
		if (count > SAM3_Q8_BLOCK_SIZE)
			count = SAM3_Q8_BLOCK_SIZE;

		float scale = src[b].scale;
		int i = 0;

#if SAM3_Q8_HAS_NEON
		float32x4_t vs = vdupq_n_f32(scale);

		for (; i + 4 <= count; i += 4) {
			/* Load 4 int8 values, widen to int32, convert to f32 */
			int8x8_t vi8 = vld1_s8(&src[b].data[i]);
			int16x8_t vi16 = vmovl_s8(vi8);
			int32x4_t vi32 = vmovl_s16(vget_low_s16(vi16));
			float32x4_t vf = vcvtq_f32_s32(vi32);
			vst1q_f32(dst + offset + i, vmulq_f32(vf, vs));
		}
#endif

		for (; i < count; i++)
			dst[offset + i] = (float)src[b].data[i] * scale;
	}
}
