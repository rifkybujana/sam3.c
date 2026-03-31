/*
 * src/core/half.h - fp16 and bf16 conversion utilities
 *
 * Header-only library providing scalar and SIMD conversions between
 * IEEE 754 single-precision (f32), half-precision (fp16), and bfloat16
 * (bf16). Scalar paths are portable C11. SIMD paths are guarded by
 * SAM3_HAS_NEON / __aarch64__ for ARM NEON and SAM3_HAS_AVX2 / __AVX2__
 * plus __F16C__ for x86 AVX2+F16C. All functions are static inline.
 *
 * Key types:  uint16_t (fp16/bf16 bit patterns), float32x4_t, __m256
 * Depends on: <stdint.h>, <string.h>
 * Used by:    src/core/weight.h, src/backend/cpu/kernels.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_CORE_HALF_H
#define SAM3_CORE_HALF_H

#include <stdint.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Scalar portability helpers                                          */
/* ------------------------------------------------------------------ */

/*
 * u32_as_f32 / f32_as_u32 - type-pun between float and uint32_t.
 *
 * Using memcpy is the only strictly-conforming C11 way to alias
 * between float and integer representations.
 */
static inline float
u32_as_f32(uint32_t u)
{
	float f;
	memcpy(&f, &u, sizeof(f));
	return f;
}

static inline uint32_t
f32_as_u32(float f)
{
	uint32_t u;
	memcpy(&u, &f, sizeof(u));
	return u;
}

/* ------------------------------------------------------------------ */
/* fp16 scalar conversions                                             */
/* ------------------------------------------------------------------ */

/*
 * fp16_to_f32 - Convert an IEEE 754 fp16 bit pattern to f32.
 *
 * @h: 16-bit fp16 bit pattern (sign | exp[5] | mantissa[10])
 *
 * Handles denormals, signed zero, infinity, and NaN correctly.
 * Returns the equivalent f32 value.
 */
static inline float
fp16_to_f32(uint16_t h)
{
	uint32_t sign     = (uint32_t)(h & 0x8000u) << 16;
	uint32_t exp16    = (h >> 10) & 0x1Fu;
	uint32_t mant16   = h & 0x3FFu;
	uint32_t result;

	if (exp16 == 0u) {
		/* Zero or denormal */
		if (mant16 == 0u) {
			/* Signed zero */
			result = sign;
		} else {
			/*
			 * Denormal: normalise by shifting mantissa left until
			 * the leading 1 drops off, adjusting exponent accordingly.
			 */
			uint32_t mant = mant16;
			uint32_t exp32 = 127u - 14u; /* f32 bias - fp16 bias */
			while ((mant & 0x400u) == 0u) {
				mant <<= 1;
				exp32--;
			}
			mant &= 0x3FFu;
			result = sign | (exp32 << 23) | (mant << 13);
		}
	} else if (exp16 == 0x1Fu) {
		/* Inf or NaN: exponent all-ones maps to f32 all-ones exp */
		result = sign | (0xFFu << 23) | (mant16 << 13);
	} else {
		/* Normal: re-bias exponent from 15 to 127 */
		uint32_t exp32 = exp16 + (127u - 15u);
		result = sign | (exp32 << 23) | (mant16 << 13);
	}

	return u32_as_f32(result);
}

/*
 * f32_to_fp16 - Convert an f32 value to an fp16 bit pattern.
 *
 * @f: Single-precision float to convert.
 *
 * Uses round-to-nearest-even. Overflows to infinity. Returns the
 * 16-bit fp16 bit pattern.
 */
static inline uint16_t
f32_to_fp16(float f)
{
	uint32_t u    = f32_as_u32(f);
	uint32_t sign = (u >> 16) & 0x8000u;
	int32_t  exp  = (int32_t)((u >> 23) & 0xFFu) - 127 + 15;
	uint32_t mant = u & 0x7FFFFFu;

	if (exp >= 31) {
		/* Overflow or was already inf/nan */
		if (((u >> 23) & 0xFFu) == 0xFFu) {
			/* Preserve NaN mantissa (shifted), or inf */
			return (uint16_t)(sign | 0x7C00u |
				(mant ? (mant >> 13) | 1u : 0u));
		}
		/* Overflow to infinity */
		return (uint16_t)(sign | 0x7C00u);
	}

	if (exp <= 0) {
		/* Zero or too small: produce zero or denormal */
		if (exp < -10) {
			return (uint16_t)sign;
		}
		/*
		 * Denormal path: the implicit leading 1 of the normal
		 * mantissa must be made explicit, then shifted right.
		 */
		mant = (mant | 0x800000u) >> (1 - exp);
		/* Round-to-nearest-even */
		uint32_t round_bit = mant & 0x1000u;
		uint32_t sticky    = mant & 0x0FFFu;
		mant >>= 13;
		if (round_bit && (sticky || (mant & 1u))) {
			mant++;
		}
		return (uint16_t)(sign | mant);
	}

	/* Normal: round-to-nearest-even on the dropped bits */
	uint32_t round_bit = mant & 0x1000u;
	uint32_t sticky    = mant & 0x0FFFu;
	mant >>= 13;
	if (round_bit && (sticky || (mant & 1u))) {
		mant++;
		if (mant == 0x400u) {
			/* Mantissa overflow: carry into exponent */
			mant = 0u;
			exp++;
			if (exp >= 31) {
				return (uint16_t)(sign | 0x7C00u);
			}
		}
	}

	return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

/* ------------------------------------------------------------------ */
/* bf16 scalar conversions                                             */
/* ------------------------------------------------------------------ */

/*
 * bf16_to_f32 - Convert a bfloat16 bit pattern to f32.
 *
 * @b: 16-bit bf16 bit pattern (top 16 bits of IEEE 754 f32).
 *
 * bf16 is simply the upper 16 bits of an f32; conversion is a zero
 * extension / shift. Returns the equivalent f32 value.
 */
static inline float
bf16_to_f32(uint16_t b)
{
	uint32_t u = (uint32_t)b << 16;
	return u32_as_f32(u);
}

/*
 * f32_to_bf16 - Convert an f32 value to a bf16 bit pattern.
 *
 * @f: Single-precision float to convert.
 *
 * Uses round-to-nearest-even (not truncation). NaN payloads are
 * preserved as quiet NaN. Returns the 16-bit bf16 bit pattern.
 */
static inline uint16_t
f32_to_bf16(float f)
{
	uint32_t u    = f32_as_u32(f);
	uint32_t exp  = (u >> 23) & 0xFFu;
	uint32_t mant = u & 0x7FFFFFu;

	/* NaN: return quiet NaN, preserve sign */
	if (exp == 0xFFu && mant != 0u) {
		return (uint16_t)((u >> 16) | 0x0040u);
	}

	/* Round-to-nearest-even on bit 16 */
	uint32_t round_bit = (u >> 15) & 1u;
	uint32_t sticky    = u & 0x7FFFu;
	uint32_t truncated = u >> 16;

	if (round_bit && (sticky || (truncated & 1u))) {
		truncated++;
	}

	return (uint16_t)truncated;
}

/* ------------------------------------------------------------------ */
/* Validation helpers                                                  */
/* ------------------------------------------------------------------ */

/*
 * fp16_is_nan - Return non-zero if the fp16 bit pattern is NaN.
 *
 * @h: 16-bit fp16 bit pattern.
 *
 * NaN: exponent field is all-ones (0x1F) and mantissa is non-zero.
 */
static inline int
fp16_is_nan(uint16_t h)
{
	return ((h & 0x7C00u) == 0x7C00u) && ((h & 0x03FFu) != 0u);
}

/*
 * fp16_is_inf - Return non-zero if the fp16 bit pattern is infinity.
 *
 * @h: 16-bit fp16 bit pattern.
 *
 * Inf: exponent all-ones, mantissa zero. Sign bit may be 0 or 1.
 */
static inline int
fp16_is_inf(uint16_t h)
{
	return (h & 0x7FFFu) == 0x7C00u;
}

/*
 * bf16_is_nan - Return non-zero if the bf16 bit pattern is NaN.
 *
 * @b: 16-bit bf16 bit pattern.
 *
 * NaN: exponent field (bits 14..7) is all-ones (0xFF) and the
 * 7-bit mantissa field is non-zero.
 */
static inline int
bf16_is_nan(uint16_t b)
{
	return ((b & 0x7F80u) == 0x7F80u) && ((b & 0x007Fu) != 0u);
}

/*
 * bf16_is_inf - Return non-zero if the bf16 bit pattern is infinity.
 *
 * @b: 16-bit bf16 bit pattern.
 *
 * Inf: exponent all-ones, mantissa zero. Sign bit may be 0 or 1.
 */
static inline int
bf16_is_inf(uint16_t b)
{
	return (b & 0x7FFFu) == 0x7F80u;
}

/* ------------------------------------------------------------------ */
/* ARM NEON SIMD paths                                                 */
/* ------------------------------------------------------------------ */

#if defined(SAM3_HAS_NEON) || \
	(defined(__aarch64__) && defined(__ARM_NEON))

#include <arm_neon.h>

/*
 * fp16x4_to_f32x4 - Convert 4 fp16 values to 4 f32 values via NEON.
 *
 * @src: Pointer to 4 consecutive uint16_t fp16 bit patterns.
 *
 * Returns a float32x4_t with the converted values.
 */
static inline float32x4_t
fp16x4_to_f32x4(const uint16_t *src)
{
	float16x4_t h = vld1_f16((const __fp16 *)src);
	return vcvt_f32_f16(h);
}

/*
 * f32x4_to_fp16x4 - Convert 4 f32 values to 4 fp16 values via NEON.
 *
 * @dst: Pointer to storage for 4 uint16_t fp16 bit patterns.
 * @v:   float32x4_t of values to convert.
 */
static inline void
f32x4_to_fp16x4(uint16_t *dst, float32x4_t v)
{
	float16x4_t h = vcvt_f16_f32(v);
	vst1_f16((__fp16 *)dst, h);
}

/*
 * bf16x4_to_f32x4 - Convert 4 bf16 values to 4 f32 values via NEON.
 *
 * @src: Pointer to 4 consecutive uint16_t bf16 bit patterns.
 *
 * bf16 is the upper half of f32, so a left-shift by 16 suffices.
 * vshll_n_u16 shifts a uint16x4_t to uint32x4_t in one instruction.
 */
static inline float32x4_t
bf16x4_to_f32x4(const uint16_t *src)
{
	uint16x4_t u16 = vld1_u16(src);
	uint32x4_t u32 = vshll_n_u16(u16, 16);
	return vreinterpretq_f32_u32(u32);
}

/*
 * f32x4_to_bf16x4 - Convert 4 f32 values to 4 bf16 values via NEON.
 *
 * @dst: Pointer to storage for 4 uint16_t bf16 bit patterns.
 * @v:   float32x4_t of values to convert.
 *
 * Implements round-to-nearest-even: adds 0x7FFF + bit16 before
 * truncating the lower 16 bits.
 */
static inline void
f32x4_to_bf16x4(uint16_t *dst, float32x4_t v)
{
	uint32x4_t u   = vreinterpretq_u32_f32(v);
	/*
	 * Round-to-nearest-even:
	 *   round_inc = 0x7FFF + ((u >> 16) & 1)
	 * Adding this to u and then shifting right 16 gives RNE result
	 * for normal numbers. NaN handling: if exponent==0xFF and
	 * mantissa!=0 we force quiet NaN by ORing 0x0040 into the
	 * top half — handled by the scalar path; here we accept that
	 * NEON will produce a rounded result which may alter the NaN
	 * payload, consistent with hardware bf16 conversion units.
	 */
	uint32x4_t lsb   = vandq_u32(vshrq_n_u32(u, 16),
		vdupq_n_u32(1u));
	uint32x4_t round = vaddq_u32(lsb, vdupq_n_u32(0x7FFFu));
	uint32x4_t r     = vaddq_u32(u, round);
	uint16x4_t hi    = vmovn_u32(vshrq_n_u32(r, 16));
	vst1_u16(dst, hi);
}

/*
 * fp16x8_to_f32x4x2 - Convert 8 fp16 values to two float32x4_t via
 * NEON.
 *
 * @src: Pointer to 8 consecutive uint16_t fp16 bit patterns.
 * @lo:  Output: low 4 values as float32x4_t.
 * @hi:  Output: high 4 values as float32x4_t.
 */
static inline void
fp16x8_to_f32x4x2(const uint16_t *src, float32x4_t *lo, float32x4_t *hi)
{
	float16x8_t h = vld1q_f16((const __fp16 *)src);
	*lo = vcvt_f32_f16(vget_low_f16(h));
	*hi = vcvt_f32_f16(vget_high_f16(h));
}

/*
 * f32x4x2_to_fp16x8 - Convert two float32x4_t to 8 fp16 values via
 * NEON.
 *
 * @dst: Pointer to storage for 8 uint16_t fp16 bit patterns.
 * @lo:  Low 4 f32 values.
 * @hi:  High 4 f32 values.
 */
static inline void
f32x4x2_to_fp16x8(uint16_t *dst, float32x4_t lo, float32x4_t hi)
{
	float16x4_t hlo = vcvt_f16_f32(lo);
	float16x4_t hhi = vcvt_f16_f32(hi);
	float16x8_t h   = vcombine_f16(hlo, hhi);
	vst1q_f16((__fp16 *)dst, h);
}

#endif /* SAM3_HAS_NEON || (__aarch64__ && __ARM_NEON) */

/* ------------------------------------------------------------------ */
/* x86 AVX2 + F16C SIMD paths                                         */
/* ------------------------------------------------------------------ */

#if defined(SAM3_HAS_AVX2) || defined(__AVX2__)
#ifdef __F16C__

#include <immintrin.h>

/*
 * fp16x8_to_f32x8 - Convert 8 fp16 values to 8 f32 values via F16C.
 *
 * @src: Pointer to 8 consecutive uint16_t fp16 bit patterns.
 *
 * Uses _mm256_cvtph_ps which is a single instruction on F16C CPUs.
 * Returns a __m256 with the 8 converted float values.
 */
static inline __m256
fp16x8_to_f32x8(const uint16_t *src)
{
	__m128i h = _mm_loadu_si128((const __m128i *)src);
	return _mm256_cvtph_ps(h);
}

/*
 * f32x8_to_fp16x8 - Convert 8 f32 values to 8 fp16 values via F16C.
 *
 * @dst: Pointer to storage for 8 uint16_t fp16 bit patterns.
 * @v:   __m256 of 8 float values to convert.
 *
 * Uses _mm256_cvtps_ph with round-to-nearest-even (_MM_FROUND_RTE).
 */
static inline void
f32x8_to_fp16x8(uint16_t *dst, __m256 v)
{
	__m128i h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
	_mm_storeu_si128((__m128i *)dst, h);
}

#endif /* __F16C__ */
#endif /* SAM3_HAS_AVX2 || __AVX2__ */

#endif /* SAM3_CORE_HALF_H */
