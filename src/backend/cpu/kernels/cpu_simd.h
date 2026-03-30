/*
 * src/backend/cpu/kernels/cpu_simd.h - SIMD detection and helpers
 *
 * Compile-time SIMD feature detection and horizontal reduction helpers.
 * Defines SAM3_HAS_NEON on ARM64 and SAM3_HAS_AVX2 on x86_64 when
 * available. Provides horizontal sum/max for both instruction sets.
 *
 * Key types:  (macros only)
 * Depends on: <arm_neon.h> or <immintrin.h> (platform-dependent)
 * Used by:    all cpu_*.c kernel files
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_CPU_SIMD_H
#define SAM3_CPU_SIMD_H

/* --- SIMD feature detection --- */

#if defined(__aarch64__) || defined(_M_ARM64)
#define SAM3_HAS_NEON 1
#include <arm_neon.h>
#endif

#if defined(__AVX2__)
#define SAM3_HAS_AVX2 1
#include <immintrin.h>
#endif

/* --- NEON helpers --- */

#if SAM3_HAS_NEON

/* Horizontal sum of a float32x4 vector. */
static inline float neon_hsum_f32(float32x4_t v)
{
	float32x2_t sum = vadd_f32(vget_low_f32(v), vget_high_f32(v));
	sum = vpadd_f32(sum, sum);
	return vget_lane_f32(sum, 0);
}

/* Horizontal max of a float32x4 vector. */
static inline float neon_hmax_f32(float32x4_t v)
{
	float32x2_t mx = vpmax_f32(vget_low_f32(v), vget_high_f32(v));
	mx = vpmax_f32(mx, mx);
	return vget_lane_f32(mx, 0);
}

#endif /* SAM3_HAS_NEON */

/* --- AVX2 helpers --- */

#if SAM3_HAS_AVX2

/* Horizontal sum of a __m256 (8 floats). */
static inline float avx2_hsum_f32(__m256 v)
{
	__m128 lo = _mm256_castps256_ps128(v);
	__m128 hi = _mm256_extractf128_ps(v, 1);
	lo = _mm_add_ps(lo, hi);
	__m128 shuf = _mm_movehdup_ps(lo);
	__m128 sums = _mm_add_ps(lo, shuf);
	shuf = _mm_movehl_ps(shuf, sums);
	sums = _mm_add_ss(sums, shuf);
	return _mm_cvtss_f32(sums);
}

/* Horizontal max of a __m256 (8 floats). */
static inline float avx2_hmax_f32(__m256 v)
{
	__m128 lo = _mm256_castps256_ps128(v);
	__m128 hi = _mm256_extractf128_ps(v, 1);
	lo = _mm_max_ps(lo, hi);
	__m128 shuf = _mm_movehdup_ps(lo);
	__m128 mx = _mm_max_ps(lo, shuf);
	shuf = _mm_movehl_ps(shuf, mx);
	mx = _mm_max_ss(mx, shuf);
	return _mm_cvtss_f32(mx);
}

#endif /* SAM3_HAS_AVX2 */

#endif /* SAM3_CPU_SIMD_H */
