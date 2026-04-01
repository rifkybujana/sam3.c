/*
 * src/backend/cpu/kernels/cpu_simd_f16.h - NEON fp16 SIMD helpers
 *
 * Provides float16x8_t helper functions for native fp16 arithmetic
 * on ARMv8.2-A+ (Apple Silicon M1+). Guarded by SAM3_HAS_NEON_FP16
 * which checks __ARM_FEATURE_FP16_VECTOR_ARITHMETIC.
 *
 * Key types:  (inline functions using float16x8_t)
 * Depends on: cpu_simd.h, <arm_neon.h>, <math.h>
 * Used by:    cpu_*_f16.c kernel files
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_CPU_SIMD_F16_H
#define SAM3_CPU_SIMD_F16_H

#include "cpu_simd.h"

#include <math.h>

/* Detect native fp16 vector arithmetic support */
#if SAM3_HAS_NEON && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#define SAM3_HAS_NEON_FP16 1
#endif

#if SAM3_HAS_NEON_FP16

static inline float16x8_t neon_f16_zero(void)
{
	return vdupq_n_f16((_Float16)0.0f);
}

/* Horizontal sum of float16x8_t, returned as f32 for precision. */
static inline float neon_f16_hsum(float16x8_t v)
{
	float32x4_t lo = vcvt_f32_f16(vget_low_f16(v));
	float32x4_t hi = vcvt_f32_f16(vget_high_f16(v));
	return neon_hsum_f32(vaddq_f32(lo, hi));
}

/* Horizontal max of float16x8_t, returned as _Float16. */
static inline _Float16 neon_f16_hmax(float16x8_t v)
{
	float16x4_t lo = vget_low_f16(v);
	float16x4_t hi = vget_high_f16(v);
	float16x4_t mx = vpmax_f16(lo, hi);
	mx = vpmax_f16(mx, mx);
	mx = vpmax_f16(mx, mx);
	return vget_lane_f16(mx, 0);
}

/*
 * neon_f16_gelu_approx - GELU approximation in fp16.
 *
 * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * Since there is no fp16 tanh intrinsic, we upcast to f32 for the
 * tanh call, then convert back. The rest is native fp16 arithmetic.
 */
static inline float16x8_t neon_f16_gelu_approx(float16x8_t x)
{
	float16x8_t half     = vdupq_n_f16((_Float16)0.5f);
	float16x8_t one      = vdupq_n_f16((_Float16)1.0f);
	float16x8_t coeff    = vdupq_n_f16((_Float16)0.044715f);
	float16x8_t sqrt2pi  = vdupq_n_f16((_Float16)0.7978845608f);

	float16x8_t x3 = vmulq_f16(vmulq_f16(x, x), x);
	float16x8_t inner = vmulq_f16(sqrt2pi,
				       vfmaq_f16(x, coeff, x3));

	/* tanh via f32 — no fp16 tanh intrinsic */
	float tmp[8];
	float32x4_t lo = vcvt_f32_f16(vget_low_f16(inner));
	float32x4_t hi = vcvt_f32_f16(vget_high_f16(inner));
	vst1q_f32(tmp, lo);
	vst1q_f32(tmp + 4, hi);
	for (int i = 0; i < 8; i++)
		tmp[i] = tanhf(tmp[i]);
	lo = vld1q_f32(tmp);
	hi = vld1q_f32(tmp + 4);
	float16x8_t tanh_v = vcombine_f16(vcvt_f16_f32(lo),
					    vcvt_f16_f32(hi));

	return vmulq_f16(half, vmulq_f16(x, vaddq_f16(one, tanh_v)));
}

#endif /* SAM3_HAS_NEON_FP16 */

#endif /* SAM3_CPU_SIMD_F16_H */
