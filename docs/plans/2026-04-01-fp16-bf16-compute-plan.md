# FP16/BF16 Compute Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add fp16 and bf16 compute paths to the SAM3 CPU backend with native NEON fp16 arithmetic, bf16-to-f32 upcasting, a 2D dispatch table, and compile-time-gated tracing.

**Architecture:** Separate kernel files per dtype dispatched via a 2D `[op][dtype]` function pointer table. fp16 uses native ARMv8.2-A NEON arithmetic with scalar fallback. bf16 always upcasts to f32. A new trace system provides per-kernel dtype logging and numeric diagnostics. All tracing is compile-time gated via `SAM3_HAS_TRACE`.

**Tech Stack:** C11, ARM NEON (float16x8_t), AVX2 F16C, CMake, CTest

**Design doc:** `docs/plans/2026-04-01-fp16-bf16-compute-design.md`

---

### Task 1: Add half.h conversion library

**Files:**
- Create: `src/core/half.h`
- Test: `tests/test_half.c`

**Step 1: Write the failing test**

Create `tests/test_half.c`:

```c
/*
 * tests/test_half.c - Unit tests for fp16/bf16 conversions
 *
 * Tests scalar and SIMD conversion round-trips, special values
 * (NaN, Inf, denormals, zero), and validation helpers.
 *
 * Key types:  (uses half.h conversion functions)
 * Depends on: core/half.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "core/half.h"

static void test_fp16_round_trip(void)
{
	/* Test a range of normal values */
	float vals[] = {0.0f, 1.0f, -1.0f, 0.5f, 65504.0f, 6.1e-5f, 3.14f};
	for (int i = 0; i < 7; i++) {
		uint16_t h = f32_to_fp16(vals[i]);
		float back = fp16_to_f32(h);
		ASSERT_NEAR(back, vals[i], fabsf(vals[i]) * 1e-3f + 1e-7f);
	}
}

static void test_fp16_special_values(void)
{
	/* Positive zero */
	uint16_t pz = f32_to_fp16(0.0f);
	ASSERT(fp16_to_f32(pz) == 0.0f);

	/* Negative zero */
	uint16_t nz = f32_to_fp16(-0.0f);
	ASSERT(fp16_to_f32(nz) == -0.0f);

	/* Infinity */
	uint16_t pinf = f32_to_fp16(1.0f / 0.0f);
	ASSERT(fp16_is_inf(pinf));
	ASSERT(!fp16_is_nan(pinf));

	/* NaN */
	uint16_t nan_h = f32_to_fp16(0.0f / 0.0f);
	ASSERT(fp16_is_nan(nan_h));
	ASSERT(!fp16_is_inf(nan_h));
}

static void test_bf16_round_trip(void)
{
	float vals[] = {0.0f, 1.0f, -1.0f, 3.14f, 1e10f, -1e10f, 1e-20f};
	for (int i = 0; i < 7; i++) {
		uint16_t b = f32_to_bf16(vals[i]);
		float back = bf16_to_f32(b);
		ASSERT_NEAR(back, vals[i], fabsf(vals[i]) * 0.008f + 1e-38f);
	}
}

static void test_bf16_special_values(void)
{
	/* bf16 infinity */
	uint16_t pinf = f32_to_bf16(1.0f / 0.0f);
	ASSERT(bf16_is_inf(pinf));
	ASSERT(!bf16_is_nan(pinf));

	/* bf16 NaN */
	uint16_t nan_b = f32_to_bf16(0.0f / 0.0f);
	ASSERT(bf16_is_nan(nan_b));
	ASSERT(!bf16_is_inf(nan_b));
}

static void test_bf16_round_to_nearest_even(void)
{
	/*
	 * f32 1.5 = 0x3FC00000. Truncation and round-to-nearest-even
	 * should both give bf16 0x3FC0.
	 * f32 1.5009766 = 0x3FC02000. Truncation gives 0x3FC0,
	 * round-to-nearest-even should give 0x3FC0 (ties to even).
	 */
	uint16_t b = f32_to_bf16(1.5f);
	ASSERT_EQ(b, 0x3FC0);
}

int main(void)
{
	test_fp16_round_trip();
	test_fp16_special_values();
	test_bf16_round_trip();
	test_bf16_special_values();
	test_bf16_round_to_nearest_even();

	TEST_REPORT();
}
```

**Step 2: Run test to verify it fails**

Run: `cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug && make test_half 2>&1`
Expected: FAIL — `core/half.h: No such file or directory`

**Step 3: Write half.h implementation**

Create `src/core/half.h`:

```c
/*
 * src/core/half.h - FP16 and BF16 conversion library
 *
 * Header-only library of static inline functions for converting between
 * fp16 (IEEE 754 half-precision), bf16 (bfloat16), and f32. Provides
 * scalar portable C11 conversions plus SIMD-accelerated batch conversions
 * for NEON (4-wide and 8-wide) and AVX2 F16C (8-wide).
 *
 * Key types:  (inline functions only)
 * Depends on: <stdint.h>, <string.h>, cpu_simd.h (for SIMD paths)
 * Used by:    all fp16/bf16 kernel files, trace.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_CORE_HALF_H
#define SAM3_CORE_HALF_H

#include <stdint.h>
#include <string.h>
#include <math.h>

/* ---- Scalar fp16 <-> f32 ---- */

/*
 * fp16_to_f32 - Convert IEEE 754 half-precision to float.
 *
 * Handles normals, denormals, inf, nan, and signed zero.
 */
static inline float fp16_to_f32(uint16_t h)
{
	uint32_t sign = (uint32_t)(h >> 15) << 31;
	uint32_t exp  = (h >> 10) & 0x1F;
	uint32_t mant = h & 0x3FF;
	uint32_t bits;

	if (exp == 0) {
		if (mant == 0) {
			/* Signed zero */
			bits = sign;
		} else {
			/* Denormal: convert to normalized f32 */
			float f = (float)mant / 1024.0f;
			f *= (1.0f / 16384.0f);  /* 2^-14 */
			uint32_t fbits;
			memcpy(&fbits, &f, 4);
			bits = sign | fbits;
		}
	} else if (exp == 31) {
		/* Inf or NaN */
		bits = sign | 0x7F800000 | ((uint32_t)mant << 13);
	} else {
		/* Normal: rebias exponent from 15 to 127 */
		bits = sign | ((uint32_t)(exp + 112) << 23) | ((uint32_t)mant << 13);
	}

	float result;
	memcpy(&result, &bits, 4);
	return result;
}

/*
 * f32_to_fp16 - Convert float to IEEE 754 half-precision.
 *
 * Uses round-to-nearest-even. Handles overflow to inf and
 * underflow to denormal/zero.
 */
static inline uint16_t f32_to_fp16(float f)
{
	uint32_t bits;
	memcpy(&bits, &f, 4);

	uint16_t sign = (uint16_t)((bits >> 16) & 0x8000);
	int32_t  exp  = (int32_t)((bits >> 23) & 0xFF) - 127;
	uint32_t mant = bits & 0x7FFFFF;

	if (exp > 15) {
		/* Overflow or inf */
		if (exp == 128 && mant != 0) {
			/* NaN — preserve some mantissa bits */
			return sign | 0x7C00 | (uint16_t)(mant >> 13) | 1;
		}
		return sign | 0x7C00;  /* Inf */
	}

	if (exp < -14) {
		/* Underflow to denormal or zero */
		if (exp < -24)
			return sign;  /* Too small, flush to zero */

		/* Denormal */
		mant |= 0x800000;  /* Add implicit leading 1 */
		int shift = -1 - exp;  /* How many bits to shift right */
		uint32_t round_bit = 1U << (shift + 12);
		uint32_t sticky = (mant & (round_bit - 1)) ? 1 : 0;
		mant >>= (shift + 13);
		mant += (round_bit >> 13) & (mant | sticky);  /* Round to nearest even */
		return sign | (uint16_t)mant;
	}

	/* Normal value */
	uint16_t hexp  = (uint16_t)((exp + 15) << 10);
	uint16_t hmant = (uint16_t)(mant >> 13);
	/* Round to nearest even */
	uint32_t round_bit = mant & 0x1000;
	uint32_t sticky    = mant & 0xFFF;
	if (round_bit && (sticky || (hmant & 1)))
		hmant++;
	if (hmant > 0x3FF) {
		hmant = 0;
		hexp += 0x0400;
	}
	return sign | hexp | hmant;
}

/* ---- Scalar bf16 <-> f32 ---- */

/*
 * bf16_to_f32 - Convert bfloat16 to float.
 *
 * bf16 is just the upper 16 bits of f32, so conversion is a
 * left shift by 16.
 */
static inline float bf16_to_f32(uint16_t b)
{
	uint32_t bits = (uint32_t)b << 16;
	float result;
	memcpy(&result, &bits, 4);
	return result;
}

/*
 * f32_to_bf16 - Convert float to bfloat16 with round-to-nearest-even.
 *
 * Matches PyTorch's bf16 rounding behavior.
 */
static inline uint16_t f32_to_bf16(float f)
{
	uint32_t bits;
	memcpy(&bits, &f, 4);

	/* Handle NaN: force canonical NaN */
	if ((bits & 0x7F800000) == 0x7F800000 && (bits & 0x7FFFFF) != 0)
		return (uint16_t)((bits >> 16) | 1);

	/* Round to nearest even */
	uint32_t round_bit = 0x8000;  /* bit 15 */
	uint32_t lsb       = 0x10000; /* bit 16 (LSB of bf16) */
	uint32_t remainder  = bits & 0xFFFF;

	if (remainder > round_bit ||
	    (remainder == round_bit && (bits & lsb)))
		bits += lsb;

	return (uint16_t)(bits >> 16);
}

/* ---- Validation helpers ---- */

static inline int fp16_is_nan(uint16_t h)
{
	return ((h & 0x7C00) == 0x7C00) && ((h & 0x03FF) != 0);
}

static inline int fp16_is_inf(uint16_t h)
{
	return ((h & 0x7FFF) == 0x7C00);
}

static inline int bf16_is_nan(uint16_t b)
{
	return ((b & 0x7F80) == 0x7F80) && ((b & 0x007F) != 0);
}

static inline int bf16_is_inf(uint16_t b)
{
	return ((b & 0x7FFF) == 0x7F80);
}

/* ---- SIMD conversions: NEON ---- */

#if defined(SAM3_HAS_NEON) || (defined(__aarch64__) && defined(__ARM_NEON))

#include <arm_neon.h>

/* 4-wide fp16 -> f32 using NEON vcvt */
static inline float32x4_t fp16x4_to_f32x4(const uint16_t *src)
{
	float16x4_t h = vld1_f16((const _Float16 *)src);
	return vcvt_f32_f16(h);
}

/* 4-wide f32 -> fp16 using NEON vcvt */
static inline void f32x4_to_fp16x4(uint16_t *dst, float32x4_t v)
{
	float16x4_t h = vcvt_f16_f32(v);
	vst1_f16((_Float16 *)dst, h);
}

/* 8-wide fp16 -> 2x f32x4 */
static inline void fp16x8_to_f32x4x2(const uint16_t *src,
				       float32x4_t *lo, float32x4_t *hi)
{
	float16x8_t h = vld1q_f16((const _Float16 *)src);
	*lo = vcvt_f32_f16(vget_low_f16(h));
	*hi = vcvt_f32_f16(vget_high_f16(h));
}

/* 2x f32x4 -> 8-wide fp16 */
static inline void f32x4x2_to_fp16x8(uint16_t *dst,
				       float32x4_t lo, float32x4_t hi)
{
	float16x4_t hlo = vcvt_f16_f32(lo);
	float16x4_t hhi = vcvt_f16_f32(hi);
	float16x8_t h = vcombine_f16(hlo, hhi);
	vst1q_f16((_Float16 *)dst, h);
}

/* 4-wide bf16 -> f32 using shift trick */
static inline float32x4_t bf16x4_to_f32x4(const uint16_t *src)
{
	uint16x4_t raw = vld1_u16(src);
	uint32x4_t wide = vshll_n_u16(raw, 16);
	return vreinterpretq_f32_u32(wide);
}

/* 4-wide f32 -> bf16 (truncation; round-to-nearest-even for exact
 * rounding would need more work — matches fast path) */
static inline void f32x4_to_bf16x4(uint16_t *dst, float32x4_t v)
{
	uint32x4_t bits = vreinterpretq_u32_f32(v);
	/* Round to nearest even: add 0x7FFF + bit16 */
	uint32x4_t lsb    = vshrq_n_u32(bits, 16);
	lsb = vandq_u32(lsb, vdupq_n_u32(1));
	uint32x4_t rounding = vaddq_u32(vdupq_n_u32(0x7FFF), lsb);
	bits = vaddq_u32(bits, rounding);
	uint16x4_t narrow = vshrn_n_u32(bits, 16);
	vst1_u16(dst, narrow);
}

#endif /* NEON */

/* ---- SIMD conversions: AVX2 with F16C ---- */

#if defined(SAM3_HAS_AVX2) || defined(__AVX2__)

#include <immintrin.h>

#ifdef __F16C__

/* 8-wide fp16 -> f32 using F16C */
static inline __m256 fp16x8_to_f32x8(const uint16_t *src)
{
	__m128i h = _mm_loadu_si128((const __m128i *)src);
	return _mm256_cvtph_ps(h);
}

/* 8-wide f32 -> fp16 using F16C (round to nearest even) */
static inline void f32x8_to_fp16x8(uint16_t *dst, __m256 v)
{
	__m128i h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
	_mm_storeu_si128((__m128i *)dst, h);
}

#endif /* __F16C__ */

#endif /* AVX2 */

#endif /* SAM3_CORE_HALF_H */
```

**Step 4: Add test to CMake and run**

The test is auto-discovered by the existing `file(GLOB TEST_SOURCES "tests/test_*.c")` pattern.

Run: `cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug && make test_half && ./test_half`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add -f src/core/half.h tests/test_half.c
git commit -m "core/half: add fp16/bf16 conversion library with tests"
```

---

### Task 2: Add SAM3_EDTYPE error code and dtype/op string helpers

**Files:**
- Modify: `include/sam3/sam3_types.h:31` — add SAM3_EDTYPE
- Modify: `src/util/error.c:21` — add case for SAM3_EDTYPE
- Modify: `src/core/tensor.h` — add sam3_dtype_str()
- Modify: `src/core/tensor.c` — implement sam3_dtype_str()
- Modify: `src/core/graph.h` — add SAM3_OP_CAST, SAM3_OP_COUNT, sam3_op_str()
- Modify: `src/core/graph.c` — implement sam3_op_str()

**Step 1: Write the failing test**

Add to `tests/test_tensor.c`:

```c
static void test_dtype_str(void)
{
	ASSERT(strcmp(sam3_dtype_str(SAM3_DTYPE_F32), "F32") == 0);
	ASSERT(strcmp(sam3_dtype_str(SAM3_DTYPE_F16), "F16") == 0);
	ASSERT(strcmp(sam3_dtype_str(SAM3_DTYPE_BF16), "BF16") == 0);
	ASSERT(strcmp(sam3_dtype_str(SAM3_DTYPE_I32), "I32") == 0);
	ASSERT(strcmp(sam3_dtype_str(SAM3_DTYPE_I8), "I8") == 0);
}
```

Add `#include <string.h>` at top of test_tensor.c and call `test_dtype_str()` from main.

**Step 2: Run test to verify it fails**

Run: `cd build && cmake .. && make test_tensor && ./test_tensor`
Expected: FAIL — `sam3_dtype_str` undefined

**Step 3: Implement changes**

In `include/sam3/sam3_types.h`, add after `SAM3_EMODEL`:
```c
	SAM3_EDTYPE   = -6,  /* Unsupported or mismatched dtype */
```

In `src/util/error.c`, add case:
```c
	case SAM3_EDTYPE:   return "unsupported or mismatched dtype";
```

In `src/core/tensor.h`, add declaration:
```c
/* Return a short string name for the dtype ("F32", "F16", etc). */
const char *sam3_dtype_str(enum sam3_dtype dtype);
```

In `src/core/tensor.c`, implement:
```c
const char *sam3_dtype_str(enum sam3_dtype dtype)
{
	switch (dtype) {
	case SAM3_DTYPE_F32:  return "F32";
	case SAM3_DTYPE_F16:  return "F16";
	case SAM3_DTYPE_BF16: return "BF16";
	case SAM3_DTYPE_I32:  return "I32";
	case SAM3_DTYPE_I8:   return "I8";
	}
	return "UNKNOWN";
}
```

In `src/core/graph.h`, add to `enum sam3_op` before the closing brace:
```c
	SAM3_OP_CAST,
	SAM3_OP_COUNT,  /* must be last */
```

Also add declaration:
```c
/* Return a short string name for the op ("MATMUL", "ADD", etc). */
const char *sam3_op_str(enum sam3_op op);
```

In `src/core/graph.c`, implement:
```c
const char *sam3_op_str(enum sam3_op op)
{
	static const char *names[] = {
		[SAM3_OP_NONE]      = "NONE",
		[SAM3_OP_MATMUL]    = "MATMUL",
		[SAM3_OP_ADD]       = "ADD",
		[SAM3_OP_MUL]       = "MUL",
		[SAM3_OP_SOFTMAX]   = "SOFTMAX",
		[SAM3_OP_RELU]      = "RELU",
		[SAM3_OP_GELU]      = "GELU",
		[SAM3_OP_LAYERNORM] = "LAYERNORM",
		[SAM3_OP_CONV2D]    = "CONV2D",
		[SAM3_OP_RESHAPE]   = "RESHAPE",
		[SAM3_OP_TRANSPOSE] = "TRANSPOSE",
		[SAM3_OP_CAST]      = "CAST",
	};
	if (op >= 0 && op < SAM3_OP_COUNT)
		return names[op];
	return "UNKNOWN";
}
```

Update `src/util/profile.h` line 28 to use the new enum:
```c
#define SAM3_OP_COUNT_PROF SAM3_OP_COUNT
```
Actually, `SAM3_OP_COUNT` in profile.h is `#define SAM3_OP_COUNT (SAM3_OP_TRANSPOSE + 1)`. Replace this with:
```c
/* Remove the old #define, SAM3_OP_COUNT is now in the enum in graph.h */
```
Delete line 28 (`#define SAM3_OP_COUNT ...`) from profile.h since the enum now provides it.

**Step 4: Run tests**

Run: `cd build && cmake .. && make -j && ctest --output-on-failure`
Expected: All PASS

**Step 5: Commit**

```bash
git add include/sam3/sam3_types.h src/util/error.c src/core/tensor.h src/core/tensor.c src/core/graph.h src/core/graph.c src/util/profile.h tests/test_tensor.c
git commit -m "core: add SAM3_EDTYPE, SAM3_OP_CAST, dtype/op string helpers"
```

---

### Task 3: Add trace system

**Files:**
- Create: `src/core/trace.h`
- Create: `src/core/trace.c`
- Create: `tests/test_trace.c`
- Modify: `CMakeLists.txt` — add SAM3_TRACE option

**Step 1: Write the failing test**

Create `tests/test_trace.c`:

```c
/*
 * tests/test_trace.c - Unit tests for the trace system
 *
 * Tests numeric diagnostics, tensor comparison, and trace flag control.
 * Must be compiled with -DSAM3_HAS_TRACE for the functions to be active.
 *
 * Key types:  sam3_numeric_stats, sam3_compare_result
 * Depends on: core/trace.h, core/tensor.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"

/* Force trace on for testing */
#ifndef SAM3_HAS_TRACE
#define SAM3_HAS_TRACE
#endif

#include "core/trace.h"
#include "core/tensor.h"

static void test_numeric_stats_f32(void)
{
	float data[] = {1.0f, -2.0f, 3.0f, 0.5f};
	struct sam3_tensor t = {
		.dtype = SAM3_DTYPE_F32,
		.n_dims = 1,
		.dims = {4},
		.data = data,
		.nbytes = sizeof(data),
	};

	struct sam3_numeric_stats stats;
	sam3_trace_compute_stats(&t, &stats);

	ASSERT_NEAR(stats.min, -2.0f, 1e-6f);
	ASSERT_NEAR(stats.max, 3.0f, 1e-6f);
	ASSERT_NEAR(stats.mean, 0.625f, 1e-6f);
	ASSERT_EQ(stats.nan_count, 0);
	ASSERT_EQ(stats.inf_count, 0);
	ASSERT_EQ(stats.total_elems, 4);
}

static void test_numeric_stats_nan_inf(void)
{
	float data[] = {1.0f, 0.0f / 0.0f, 1.0f / 0.0f, -1.0f / 0.0f};
	struct sam3_tensor t = {
		.dtype = SAM3_DTYPE_F32,
		.n_dims = 1,
		.dims = {4},
		.data = data,
		.nbytes = sizeof(data),
	};

	struct sam3_numeric_stats stats;
	sam3_trace_compute_stats(&t, &stats);

	ASSERT_EQ(stats.nan_count, 1);
	ASSERT_EQ(stats.inf_count, 2);
}

static void test_compare_identical(void)
{
	float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
	struct sam3_tensor a = {
		.dtype = SAM3_DTYPE_F32, .n_dims = 1, .dims = {4},
		.data = data, .nbytes = sizeof(data),
	};
	struct sam3_tensor b = a;

	struct sam3_compare_result result;
	sam3_trace_compute_compare(&a, &b, 1e-6f, &result);

	ASSERT_NEAR(result.max_abs_error, 0.0f, 1e-7f);
	ASSERT_EQ(result.mismatches, 0);
}

int main(void)
{
	test_numeric_stats_f32();
	test_numeric_stats_nan_inf();
	test_compare_identical();

	TEST_REPORT();
}
```

**Step 2: Run test to verify it fails**

Run: `cd build && cmake .. -DSAM3_TRACE=ON && make test_trace 2>&1`
Expected: FAIL — `core/trace.h: No such file or directory`

**Step 3: Implement trace.h**

Create `src/core/trace.h`:

```c
/*
 * src/core/trace.h - Dtype tracing and numeric diagnostics
 *
 * Compile-time gated (SAM3_HAS_TRACE) tracing for debugging fp16/bf16
 * precision issues. Provides per-kernel dtype logging, numeric stats
 * (NaN/Inf/denormal detection), tensor comparison, and graph execution
 * tracing. Zero overhead when SAM3_HAS_TRACE is not defined.
 *
 * Key types:  sam3_numeric_stats, sam3_compare_result, sam3_trace_flags
 * Depends on: core/tensor.h, core/graph.h, sam3/sam3_types.h
 * Used by:    cpu_dispatch.c, test_trace.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_CORE_TRACE_H
#define SAM3_CORE_TRACE_H

#include "sam3/sam3_types.h"

struct sam3_tensor;
struct sam3_graph;

/* Trace flag bits for runtime control. */
enum sam3_trace_flags {
	SAM3_TRACE_KERNELS  = 1 << 0,
	SAM3_TRACE_NUMERIC  = 1 << 1,
	SAM3_TRACE_COMPARE  = 1 << 2,
	SAM3_TRACE_GRAPH    = 1 << 3,
	SAM3_TRACE_ALL      = 0xFFFF,
};

/* Numeric statistics for a tensor. */
struct sam3_numeric_stats {
	float min;
	float max;
	float mean;
	int   nan_count;
	int   inf_count;
	int   denormal_count;
	int   total_elems;
};

/* Result of comparing two tensors element-wise. */
struct sam3_compare_result {
	float max_abs_error;
	float max_rel_error;
	float mean_abs_error;
	int   mismatches;
};

#ifdef SAM3_HAS_TRACE

/* Runtime trace flag control */
void sam3_trace_set_flags(unsigned flags);
unsigned sam3_trace_get_flags(void);

/* Kernel dtype tracing */
void sam3_trace_kernel(const char *kernel_name,
		       enum sam3_dtype in_dtype,
		       enum sam3_dtype out_dtype,
		       const char *variant);

/* Numeric diagnostics — compute stats for any tensor dtype */
void sam3_trace_compute_stats(const struct sam3_tensor *t,
			      struct sam3_numeric_stats *out);

/* Log numeric stats to debug output */
void sam3_trace_numeric(const char *label, const struct sam3_tensor *t);

/* Tensor comparison */
void sam3_trace_compute_compare(const struct sam3_tensor *actual,
				const struct sam3_tensor *reference,
				float tolerance,
				struct sam3_compare_result *out);

void sam3_trace_compare(const char *label,
			const struct sam3_tensor *actual,
			const struct sam3_tensor *reference,
			float tolerance);

/* Graph-level tracing */
void sam3_trace_graph_plan(const struct sam3_graph *g);
void sam3_trace_graph_done(const struct sam3_graph *g, double elapsed_ms);

/* Convenience macros */
#define SAM3_TRACE_KERNEL(name, in_dt, out_dt, var) \
	sam3_trace_kernel((name), (in_dt), (out_dt), (var))
#define SAM3_TRACE_NUMERIC(label, tensor) \
	do { if (sam3_trace_get_flags() & SAM3_TRACE_NUMERIC) \
		sam3_trace_numeric((label), (tensor)); } while (0)

#else /* !SAM3_HAS_TRACE */

#define sam3_trace_set_flags(f)                     ((void)0)
#define sam3_trace_get_flags()                      (0U)
#define SAM3_TRACE_KERNEL(name, in_dt, out_dt, var) ((void)0)
#define SAM3_TRACE_NUMERIC(label, tensor)            ((void)0)

/* Still provide compute functions even without trace — tests need them */
void sam3_trace_compute_stats(const struct sam3_tensor *t,
			      struct sam3_numeric_stats *out);
void sam3_trace_compute_compare(const struct sam3_tensor *actual,
				const struct sam3_tensor *reference,
				float tolerance,
				struct sam3_compare_result *out);

#endif /* SAM3_HAS_TRACE */

#endif /* SAM3_CORE_TRACE_H */
```

**Step 4: Implement trace.c**

Create `src/core/trace.c`:

```c
/*
 * src/core/trace.c - Trace system implementation
 *
 * Implements numeric diagnostics, tensor comparison, and trace logging.
 * Converts fp16/bf16 tensors to f32 internally via half.h for analysis.
 * Compute functions are always available; logging functions require
 * SAM3_HAS_TRACE.
 *
 * Key types:  sam3_numeric_stats, sam3_compare_result
 * Depends on: trace.h, half.h, tensor.h, log.h
 * Used by:    cpu_dispatch.c, tests
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include <float.h>

#include "trace.h"
#include "half.h"
#include "tensor.h"
#include "util/log.h"
#include "core/graph.h"

/* Helper: get element as f32, regardless of tensor dtype */
static float get_elem_f32(const struct sam3_tensor *t, int idx)
{
	switch (t->dtype) {
	case SAM3_DTYPE_F32:
		return ((const float *)t->data)[idx];
	case SAM3_DTYPE_F16:
		return fp16_to_f32(((const uint16_t *)t->data)[idx]);
	case SAM3_DTYPE_BF16:
		return bf16_to_f32(((const uint16_t *)t->data)[idx]);
	default:
		return 0.0f;
	}
}

void sam3_trace_compute_stats(const struct sam3_tensor *t,
			      struct sam3_numeric_stats *out)
{
	int n = sam3_tensor_nelems(t);

	out->min = FLT_MAX;
	out->max = -FLT_MAX;
	out->nan_count = 0;
	out->inf_count = 0;
	out->denormal_count = 0;
	out->total_elems = n;

	double sum = 0.0;

	for (int i = 0; i < n; i++) {
		float v = get_elem_f32(t, i);

		if (isnan(v)) {
			out->nan_count++;
			continue;
		}
		if (isinf(v)) {
			out->inf_count++;
			continue;
		}

		/* Check denormal */
		if (v != 0.0f && fabsf(v) < FLT_MIN)
			out->denormal_count++;

		if (v < out->min) out->min = v;
		if (v > out->max) out->max = v;
		sum += (double)v;
	}

	int valid = n - out->nan_count - out->inf_count;
	out->mean = valid > 0 ? (float)(sum / valid) : 0.0f;

	if (valid == 0) {
		out->min = 0.0f;
		out->max = 0.0f;
	}
}

void sam3_trace_compute_compare(const struct sam3_tensor *actual,
				const struct sam3_tensor *reference,
				float tolerance,
				struct sam3_compare_result *out)
{
	int n = sam3_tensor_nelems(actual);

	out->max_abs_error = 0.0f;
	out->max_rel_error = 0.0f;
	out->mean_abs_error = 0.0f;
	out->mismatches = 0;

	double sum_abs = 0.0;

	for (int i = 0; i < n; i++) {
		float a = get_elem_f32(actual, i);
		float r = get_elem_f32(reference, i);
		float abs_err = fabsf(a - r);
		float rel_err = (fabsf(r) > 1e-8f) ? abs_err / fabsf(r) : abs_err;

		if (abs_err > out->max_abs_error)
			out->max_abs_error = abs_err;
		if (rel_err > out->max_rel_error)
			out->max_rel_error = rel_err;

		sum_abs += (double)abs_err;

		if (abs_err > tolerance)
			out->mismatches++;
	}

	out->mean_abs_error = n > 0 ? (float)(sum_abs / n) : 0.0f;
}

#ifdef SAM3_HAS_TRACE

static unsigned g_trace_flags = SAM3_TRACE_ALL;

void sam3_trace_set_flags(unsigned flags)
{
	g_trace_flags = flags;
}

unsigned sam3_trace_get_flags(void)
{
	return g_trace_flags;
}

void sam3_trace_kernel(const char *kernel_name,
		       enum sam3_dtype in_dtype,
		       enum sam3_dtype out_dtype,
		       const char *variant)
{
	if (!(g_trace_flags & SAM3_TRACE_KERNELS))
		return;

	sam3_log_debug("[TRACE] %s: %s -> %s (%s)",
		       kernel_name,
		       sam3_dtype_str(in_dtype),
		       sam3_dtype_str(out_dtype),
		       variant);
}

void sam3_trace_numeric(const char *label, const struct sam3_tensor *t)
{
	struct sam3_numeric_stats stats;
	sam3_trace_compute_stats(t, &stats);

	sam3_log_debug("[TRACE] %s: min=%.4g max=%.4g mean=%.4g "
		       "nan=%d inf=%d denorm=%d n=%d",
		       label, stats.min, stats.max, stats.mean,
		       stats.nan_count, stats.inf_count,
		       stats.denormal_count, stats.total_elems);
}

void sam3_trace_compare(const char *label,
			const struct sam3_tensor *actual,
			const struct sam3_tensor *reference,
			float tolerance)
{
	if (!(g_trace_flags & SAM3_TRACE_COMPARE))
		return;

	struct sam3_compare_result result;
	sam3_trace_compute_compare(actual, reference, tolerance, &result);

	sam3_log_debug("[TRACE] %s: max_abs=%.4g max_rel=%.4g "
		       "mean_abs=%.4g mismatches=%d",
		       label, result.max_abs_error, result.max_rel_error,
		       result.mean_abs_error, result.mismatches);
}

void sam3_trace_graph_plan(const struct sam3_graph *g)
{
	if (!(g_trace_flags & SAM3_TRACE_GRAPH))
		return;

	sam3_log_debug("[TRACE] graph: %d nodes", g->n_nodes);

	for (int i = 0; i < g->n_nodes && i < 20; i++) {
		const struct sam3_node *n = &g->nodes[i];
		const char *dt_str = (n->inputs[0])
			? sam3_dtype_str(n->inputs[0]->dtype) : "?";
		sam3_log_debug("[TRACE]   node[%d]: %s %s",
			       i, sam3_op_str(n->op), dt_str);
	}

	if (g->n_nodes > 20)
		sam3_log_debug("[TRACE]   ... (%d more)", g->n_nodes - 20);
}

void sam3_trace_graph_done(const struct sam3_graph *g, double elapsed_ms)
{
	if (!(g_trace_flags & SAM3_TRACE_GRAPH))
		return;

	sam3_log_debug("[TRACE] graph: completed %d nodes in %.1fms",
		       g->n_nodes, elapsed_ms);
}

#endif /* SAM3_HAS_TRACE */
```

**Step 5: Update CMakeLists.txt**

Add after `option(SAM3_PROFILE ...)`:
```cmake
option(SAM3_TRACE "Enable dtype tracing" OFF)
```

Add after `if(SAM3_PROFILE)`:
```cmake
if(SAM3_TRACE)
	add_definitions(-DSAM3_HAS_TRACE)
endif()
```

For test_trace, it needs SAM3_HAS_TRACE defined. Add after the test foreach loop:
```cmake
	# test_trace needs trace enabled regardless of global setting
	if(TARGET test_trace)
		target_compile_definitions(test_trace PRIVATE SAM3_HAS_TRACE)
	endif()
```

**Step 6: Run tests**

Run: `cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug && make -j && ctest --output-on-failure`
Expected: All PASS

**Step 7: Commit**

```bash
git add -f src/core/trace.h src/core/trace.c tests/test_trace.c CMakeLists.txt
git commit -m "core/trace: add dtype tracing and numeric diagnostics"
```

---

### Task 4: Add dispatch table and refactor cpu_backend.c

**Files:**
- Create: `src/backend/cpu/cpu_dispatch.h`
- Create: `src/backend/cpu/cpu_dispatch.c`
- Modify: `src/backend/cpu/cpu_backend.c:120-175` — replace switch with dispatch call
- Modify: `src/backend/cpu/kernels/cpu_kernels.h` — update header comment, add SAM3_DTYPE_COUNT
- Create: `tests/test_dispatch.c`

**Step 1: Write the failing test**

Create `tests/test_dispatch.c`:

```c
/*
 * tests/test_dispatch.c - Dispatch table tests
 *
 * Verifies dtype mismatch rejection, NULL kernel handling, and
 * successful dispatch for f32 ops.
 *
 * Key types:  sam3_node, sam3_tensor
 * Depends on: test_helpers.h, backend/cpu/cpu_dispatch.h
 * Used by:    CTest
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <string.h>

#include "test_helpers.h"
#include "backend/cpu/cpu_dispatch.h"
#include "backend/cpu/cpu_backend.h"
#include "core/tensor.h"

static struct sam3_cpu_backend g_cpu;

static void setup(void)
{
	memset(&g_cpu, 0, sizeof(g_cpu));
	g_cpu.base.type = SAM3_BACKEND_CPU;
	g_cpu.base.ops = sam3_cpu_backend_ops();
	g_cpu.arena_capacity = 1024 * 1024;
	g_cpu.base.ops->init(&g_cpu.base);
}

static void teardown(void)
{
	g_cpu.base.ops->free(&g_cpu.base);
}

static void test_dispatch_dtype_mismatch(void)
{
	setup();

	struct sam3_tensor a = { .dtype = SAM3_DTYPE_F32, .n_dims = 1, .dims = {4} };
	struct sam3_tensor b = { .dtype = SAM3_DTYPE_F16, .n_dims = 1, .dims = {4} };
	struct sam3_tensor c = { .dtype = SAM3_DTYPE_F32, .n_dims = 1, .dims = {4} };

	g_cpu.base.ops->alloc_tensor(&g_cpu.base, &a);
	g_cpu.base.ops->alloc_tensor(&g_cpu.base, &b);
	g_cpu.base.ops->alloc_tensor(&g_cpu.base, &c);

	struct sam3_tensor *inputs[] = {&a, &b};
	struct sam3_node node = {
		.op = SAM3_OP_ADD,
		.inputs = {inputs[0], inputs[1]},
		.n_inputs = 2,
		.output = &c,
	};

	enum sam3_error err = cpu_dispatch_node(&node, &g_cpu.scratch, g_cpu.pool);
	ASSERT_EQ(err, SAM3_EDTYPE);

	teardown();
}

static void test_dispatch_f32_add(void)
{
	setup();

	int dims[] = {4};
	struct sam3_tensor a = { .dtype = SAM3_DTYPE_F32, .n_dims = 1, .dims = {4} };
	struct sam3_tensor b = { .dtype = SAM3_DTYPE_F32, .n_dims = 1, .dims = {4} };
	struct sam3_tensor c = { .dtype = SAM3_DTYPE_F32, .n_dims = 1, .dims = {4} };

	g_cpu.base.ops->alloc_tensor(&g_cpu.base, &a);
	g_cpu.base.ops->alloc_tensor(&g_cpu.base, &b);
	g_cpu.base.ops->alloc_tensor(&g_cpu.base, &c);

	float *ad = (float *)a.data;
	float *bd = (float *)b.data;
	ad[0] = 1.0f; ad[1] = 2.0f; ad[2] = 3.0f; ad[3] = 4.0f;
	bd[0] = 10.0f; bd[1] = 20.0f; bd[2] = 30.0f; bd[3] = 40.0f;

	struct sam3_node node = {
		.op = SAM3_OP_ADD,
		.inputs = {&a, &b},
		.n_inputs = 2,
		.output = &c,
	};

	enum sam3_error err = cpu_dispatch_node(&node, &g_cpu.scratch, g_cpu.pool);
	ASSERT_EQ(err, SAM3_OK);

	float *cd = (float *)c.data;
	ASSERT_NEAR(cd[0], 11.0f, 1e-6f);
	ASSERT_NEAR(cd[3], 44.0f, 1e-6f);

	teardown();
}

int main(void)
{
	test_dispatch_dtype_mismatch();
	test_dispatch_f32_add();

	TEST_REPORT();
}
```

**Step 2: Run test to verify it fails**

Run: `cd build && cmake .. && make test_dispatch 2>&1`
Expected: FAIL — `cpu_dispatch.h: No such file or directory`

**Step 3: Implement dispatch table**

Add to `include/sam3/sam3_types.h`, after the dtype enum:
```c
#define SAM3_DTYPE_COUNT 5
```

Create `src/backend/cpu/cpu_dispatch.h`:

```c
/*
 * src/backend/cpu/cpu_dispatch.h - CPU kernel dispatch table
 *
 * Provides a 2D dispatch table [op][dtype] -> kernel function for
 * routing compute graph nodes to the correct dtype-specific kernel.
 * Validates dtype consistency across node inputs before dispatch.
 *
 * Key types:  sam3_kernel_fn
 * Depends on: core/graph.h, core/alloc.h, sam3/sam3_types.h
 * Used by:    cpu_backend.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_CPU_DISPATCH_H
#define SAM3_CPU_DISPATCH_H

#include "core/graph.h"
#include "core/alloc.h"
#include "sam3/sam3_types.h"

struct sam3_threadpool;

/*
 * cpu_dispatch_node - Dispatch a graph node to the correct kernel.
 *
 * Validates all inputs share the same dtype, looks up the kernel
 * in the dispatch table, and calls it. Returns SAM3_EDTYPE for
 * dtype mismatches or unimplemented (op, dtype) combinations.
 */
enum sam3_error cpu_dispatch_node(const struct sam3_node *node,
				  struct sam3_arena *scratch,
				  struct sam3_threadpool *pool);

#endif /* SAM3_CPU_DISPATCH_H */
```

Create `src/backend/cpu/cpu_dispatch.c`:

```c
/*
 * src/backend/cpu/cpu_dispatch.c - CPU kernel dispatch implementation
 *
 * 2D dispatch table mapping (op, dtype) to kernel functions. During
 * graph evaluation, each node is routed here. The dispatcher validates
 * dtype consistency, looks up the kernel, runs trace hooks, and calls
 * the kernel. NULL entries in the table mean "not implemented yet."
 *
 * Key types:  sam3_kernel_fn
 * Depends on: cpu_dispatch.h, cpu_kernels.h, core/trace.h, util/log.h
 * Used by:    cpu_backend.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "cpu_dispatch.h"
#include "kernels/cpu_kernels.h"
#include "core/tensor.h"
#include "core/trace.h"
#include "util/log.h"
#include "util/threadpool.h"

/*
 * Kernel wrapper signatures. The dispatch table uses a uniform
 * function pointer type. Each kernel wrapper adapts the uniform
 * signature to the kernel's actual signature.
 */
typedef enum sam3_error (*sam3_kernel_fn)(const struct sam3_node *node,
					  struct sam3_arena *scratch,
					  struct sam3_threadpool *pool);

/* Wrappers for kernels that don't need scratch arena */
static enum sam3_error wrap_matmul(const struct sam3_node *n,
				    struct sam3_arena *s,
				    struct sam3_threadpool *p)
{
	(void)s;
	return cpu_kernel_matmul(n, p);
}

static enum sam3_error wrap_add(const struct sam3_node *n,
				 struct sam3_arena *s,
				 struct sam3_threadpool *p)
{
	(void)s;
	return cpu_kernel_add(n, p);
}

static enum sam3_error wrap_mul(const struct sam3_node *n,
				 struct sam3_arena *s,
				 struct sam3_threadpool *p)
{
	(void)s;
	return cpu_kernel_mul(n, p);
}

static enum sam3_error wrap_softmax(const struct sam3_node *n,
				     struct sam3_arena *s,
				     struct sam3_threadpool *p)
{
	(void)s;
	return cpu_kernel_softmax(n, p);
}

static enum sam3_error wrap_relu(const struct sam3_node *n,
				  struct sam3_arena *s,
				  struct sam3_threadpool *p)
{
	(void)s;
	return cpu_kernel_relu(n, p);
}

static enum sam3_error wrap_gelu(const struct sam3_node *n,
				  struct sam3_arena *s,
				  struct sam3_threadpool *p)
{
	(void)s;
	return cpu_kernel_gelu(n, p);
}

static enum sam3_error wrap_layernorm(const struct sam3_node *n,
				       struct sam3_arena *s,
				       struct sam3_threadpool *p)
{
	(void)s;
	return cpu_kernel_layernorm(n, p);
}

static enum sam3_error wrap_conv2d(const struct sam3_node *n,
				    struct sam3_arena *s,
				    struct sam3_threadpool *p)
{
	return cpu_kernel_conv2d(n, s, p);
}

static enum sam3_error wrap_reshape(const struct sam3_node *n,
				     struct sam3_arena *s,
				     struct sam3_threadpool *p)
{
	(void)s;
	(void)p;
	return cpu_kernel_reshape(n);
}

static enum sam3_error wrap_transpose(const struct sam3_node *n,
				       struct sam3_arena *s,
				       struct sam3_threadpool *p)
{
	(void)s;
	return cpu_kernel_transpose(n, p);
}

/*
 * 2D dispatch table: [op][dtype] -> kernel.
 *
 * NULL entries mean "not yet implemented" and will return SAM3_EDTYPE.
 * reshape and transpose are dtype-agnostic — registered for all dtypes.
 */
static const sam3_kernel_fn
cpu_dispatch_table[SAM3_OP_COUNT][SAM3_DTYPE_COUNT] = {
	[SAM3_OP_MATMUL] = {
		[SAM3_DTYPE_F32] = wrap_matmul,
	},
	[SAM3_OP_ADD] = {
		[SAM3_DTYPE_F32] = wrap_add,
	},
	[SAM3_OP_MUL] = {
		[SAM3_DTYPE_F32] = wrap_mul,
	},
	[SAM3_OP_SOFTMAX] = {
		[SAM3_DTYPE_F32] = wrap_softmax,
	},
	[SAM3_OP_RELU] = {
		[SAM3_DTYPE_F32] = wrap_relu,
	},
	[SAM3_OP_GELU] = {
		[SAM3_DTYPE_F32] = wrap_gelu,
	},
	[SAM3_OP_LAYERNORM] = {
		[SAM3_DTYPE_F32] = wrap_layernorm,
	},
	[SAM3_OP_CONV2D] = {
		[SAM3_DTYPE_F32] = wrap_conv2d,
	},
	[SAM3_OP_RESHAPE] = {
		[SAM3_DTYPE_F32] = wrap_reshape,
		[SAM3_DTYPE_F16] = wrap_reshape,
		[SAM3_DTYPE_BF16] = wrap_reshape,
		[SAM3_DTYPE_I32] = wrap_reshape,
		[SAM3_DTYPE_I8] = wrap_reshape,
	},
	[SAM3_OP_TRANSPOSE] = {
		[SAM3_DTYPE_F32] = wrap_transpose,
	},
};

enum sam3_error cpu_dispatch_node(const struct sam3_node *node,
				  struct sam3_arena *scratch,
				  struct sam3_threadpool *pool)
{
	if (!node || !node->inputs[0]) {
		sam3_log_error("dispatch: NULL node or input");
		return SAM3_EINVAL;
	}

	enum sam3_dtype dt = node->inputs[0]->dtype;

	/* Verify all inputs share the same dtype */
	for (int i = 1; i < node->n_inputs; i++) {
		if (node->inputs[i] &&
		    node->inputs[i]->dtype != dt) {
			sam3_log_error("dispatch: dtype mismatch input[0]=%s "
				       "input[%d]=%s for op %s",
				       sam3_dtype_str(dt), i,
				       sam3_dtype_str(node->inputs[i]->dtype),
				       sam3_op_str(node->op));
			return SAM3_EDTYPE;
		}
	}

	if (node->op < 0 || node->op >= SAM3_OP_COUNT ||
	    dt < 0 || dt >= SAM3_DTYPE_COUNT) {
		sam3_log_error("dispatch: out of range op=%d dtype=%d",
			       node->op, dt);
		return SAM3_EINVAL;
	}

	sam3_kernel_fn fn = cpu_dispatch_table[node->op][dt];
	if (!fn) {
		sam3_log_error("dispatch: no kernel for op=%s dtype=%s",
			       sam3_op_str(node->op), sam3_dtype_str(dt));
		return SAM3_EDTYPE;
	}

	/* Trace hooks */
	SAM3_TRACE_KERNEL(sam3_op_str(node->op), dt,
			  node->output ? node->output->dtype : dt,
			  sam3_dtype_str(dt));

	enum sam3_error err = fn(node, scratch, pool);

	if (err == SAM3_OK && node->output)
		SAM3_TRACE_NUMERIC(sam3_op_str(node->op), node->output);

	return err;
}
```

**Step 4: Refactor cpu_backend.c to use dispatch**

Replace the `switch` statement in `cpu_graph_eval` (lines ~130-175) with:

```c
	/* Replace the entire switch block with: */
	err = cpu_dispatch_node(node, &cpu->scratch, cpu->pool);
```

Remove the `#include "kernels/cpu_kernels.h"` from cpu_backend.c and add `#include "cpu_dispatch.h"`.

Also update `cpu_kernel_reshape` to be dtype-agnostic: remove the `if (in->dtype != SAM3_DTYPE_F32)` check from `cpu_reshape.c`.

**Step 5: Run tests**

Run: `cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug && make -j && ctest --output-on-failure`
Expected: All PASS (existing tests use f32 which is still dispatched correctly)

**Step 6: Commit**

```bash
git add -f src/backend/cpu/cpu_dispatch.h src/backend/cpu/cpu_dispatch.c \
  src/backend/cpu/cpu_backend.c src/backend/cpu/kernels/cpu_reshape.c \
  tests/test_dispatch.c
git commit -m "cpu: add 2D dispatch table, refactor graph_eval to use it"
```

---

### Task 5: Add cpu_simd_f16.h — NEON fp16 helpers

**Files:**
- Create: `src/backend/cpu/kernels/cpu_simd_f16.h`

**Step 1: Create the SIMD helper header**

```c
/*
 * src/backend/cpu/kernels/cpu_simd_f16.h - NEON fp16 SIMD helpers
 *
 * Provides float16x8_t helper functions for native fp16 arithmetic
 * on ARMv8.2-A+ (Apple Silicon M1+). Guarded by SAM3_HAS_NEON_FP16
 * which checks __ARM_FEATURE_FP16_VECTOR_ARITHMETIC.
 *
 * Key types:  (inline functions using float16x8_t)
 * Depends on: <arm_neon.h>
 * Used by:    cpu_*_f16.c kernel files
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_CPU_SIMD_F16_H
#define SAM3_CPU_SIMD_F16_H

#include "cpu_simd.h"

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
```

You need to `#include <math.h>` at the top for `tanhf`.

**Step 2: Commit**

```bash
git add -f src/backend/cpu/kernels/cpu_simd_f16.h
git commit -m "cpu/simd: add NEON fp16 vector helpers"
```

---

### Task 6: Add fp16 elementwise kernels (add, mul, relu)

**Files:**
- Create: `src/backend/cpu/kernels/cpu_add_f16.c`
- Create: `src/backend/cpu/kernels/cpu_mul_f16.c`
- Create: `src/backend/cpu/kernels/cpu_relu_f16.c`
- Create: `tests/test_elementwise_f16.c`
- Modify: `src/backend/cpu/kernels/cpu_kernels.h` — add declarations
- Modify: `src/backend/cpu/cpu_dispatch.c` — register in table

Each fp16 kernel follows this pattern:
1. `SAM3_HAS_NEON_FP16` path: native `float16x8_t` operations
2. Scalar fallback: convert each element via `fp16_to_f32`, compute, `f32_to_fp16`
3. Same threading structure as f32 kernels

**Step 1: Write the failing test**

Create `tests/test_elementwise_f16.c` testing add, mul, relu for fp16 against f32 reference with tolerances from the design doc.

**Step 2: Implement each kernel**

Example: `cpu_add_f16.c` core loop (native NEON path):

```c
#if SAM3_HAS_NEON_FP16
static void add_f16_neon(const _Float16 *a, const _Float16 *b,
			  _Float16 *out, int broadcast_n,
			  int start, int end)
{
	if (broadcast_n <= 0) {
		int i = start;
		for (; i + 8 <= end; i += 8) {
			float16x8_t va = vld1q_f16(a + i);
			float16x8_t vb = vld1q_f16(b + i);
			vst1q_f16(out + i, vaddq_f16(va, vb));
		}
		for (; i < end; i++)
			out[i] = a[i] + b[i];
	} else {
		for (int r = start; r < end; r++) {
			int base = r * broadcast_n;
			int j = 0;
			for (; j + 8 <= broadcast_n; j += 8) {
				float16x8_t va = vld1q_f16(a + base + j);
				float16x8_t vb = vld1q_f16(b + j);
				vst1q_f16(out + base + j, vaddq_f16(va, vb));
			}
			for (; j < broadcast_n; j++)
				out[base + j] = a[base + j] + b[j];
		}
	}
}
#endif
```

Scalar fallback uses `fp16_to_f32` / `f32_to_fp16` from half.h.

Add declarations to `cpu_kernels.h`:
```c
enum sam3_error cpu_kernel_add_f16(const struct sam3_node *node,
				    struct sam3_threadpool *pool);
enum sam3_error cpu_kernel_mul_f16(const struct sam3_node *node,
				    struct sam3_threadpool *pool);
enum sam3_error cpu_kernel_relu_f16(const struct sam3_node *node,
				     struct sam3_threadpool *pool);
```

Register in `cpu_dispatch.c` dispatch table:
```c
[SAM3_OP_ADD] = {
	[SAM3_DTYPE_F32] = wrap_add,
	[SAM3_DTYPE_F16] = wrap_add_f16,
},
```

**Step 3: Run tests**

Run: `cd build && cmake .. && make -j && ctest --output-on-failure`
Expected: All PASS

**Step 4: Commit**

```bash
git add -f src/backend/cpu/kernels/cpu_add_f16.c \
  src/backend/cpu/kernels/cpu_mul_f16.c \
  src/backend/cpu/kernels/cpu_relu_f16.c \
  src/backend/cpu/kernels/cpu_kernels.h \
  src/backend/cpu/cpu_dispatch.c \
  tests/test_elementwise_f16.c
git commit -m "cpu/kernels: add fp16 elementwise kernels (add, mul, relu)"
```

---

### Task 7: Add fp16 gelu kernel

**Files:**
- Create: `src/backend/cpu/kernels/cpu_gelu_f16.c`
- Modify: `src/backend/cpu/kernels/cpu_kernels.h`
- Modify: `src/backend/cpu/cpu_dispatch.c`

Uses `neon_f16_gelu_approx` from `cpu_simd_f16.h` for the NEON path. Scalar fallback converts to f32, calls tanhf, converts back.

Add test to `tests/test_elementwise_f16.c` for GELU.

**Commit:** `cpu/kernels: add fp16 GELU kernel`

---

### Task 8: Add fp16 matmul kernel

**Files:**
- Create: `src/backend/cpu/kernels/cpu_matmul_f16.c`
- Create: `tests/test_matmul_f16.c`
- Modify: `src/backend/cpu/kernels/cpu_kernels.h`
- Modify: `src/backend/cpu/cpu_dispatch.c`

The fp16 matmul uses native NEON `float16x8_t` inner loop with `vfmaq_f16` for the NEON path. Same 8x8x64 tiling as f32. Thread pool over M rows.

```c
#if SAM3_HAS_NEON_FP16
static void matmul_f16_neon(const _Float16 *a, const _Float16 *b,
			     _Float16 *c, int M, int K, int N,
			     int m_start, int m_end)
{
	/* Zero output rows */
	memset(c + (size_t)m_start * N, 0,
	       (size_t)(m_end - m_start) * N * sizeof(_Float16));

	for (int i0 = m_start; i0 < m_end; i0 += TILE_M) {
		int imax = (i0 + TILE_M < m_end) ? i0 + TILE_M : m_end;
		for (int j0 = 0; j0 < N; j0 += TILE_N) {
			int jmax = (j0 + TILE_N < N) ? j0 + TILE_N : N;
			for (int k0 = 0; k0 < K; k0 += TILE_K) {
				int kmax = (k0 + TILE_K < K) ? k0 + TILE_K : K;
				for (int i = i0; i < imax; i++) {
					for (int k = k0; k < kmax; k++) {
						float16x8_t va = vdupq_n_f16(a[i * K + k]);
						int j = j0;
						for (; j + 8 <= jmax; j += 8) {
							float16x8_t vc = vld1q_f16(c + i * N + j);
							float16x8_t vb = vld1q_f16(b + k * N + j);
							vst1q_f16(c + i * N + j,
								   vfmaq_f16(vc, va, vb));
						}
						_Float16 aik = a[i * K + k];
						for (; j < jmax; j++)
							c[i * N + j] += aik * b[k * N + j];
					}
				}
			}
		}
	}
}
#endif
```

Test: `tests/test_matmul_f16.c` computes matmul in both f32 and fp16, compares with tolerance 5e-3.

**Commit:** `cpu/kernels: add fp16 matmul kernel`

---

### Task 9: Add fp16 softmax and layernorm kernels

**Files:**
- Create: `src/backend/cpu/kernels/cpu_softmax_f16.c`
- Create: `src/backend/cpu/kernels/cpu_layernorm_f16.c`
- Modify: `src/backend/cpu/kernels/cpu_kernels.h`
- Modify: `src/backend/cpu/cpu_dispatch.c`

Both use native fp16 arithmetic. Note: even though the user chose "full fp16 accumulation," the `expf`/`tanhf` calls still need f32 (no fp16 transcendental intrinsics) — upcast for those calls only, keep accumulation in fp16.

**Commit:** `cpu/kernels: add fp16 softmax and layernorm kernels`

---

### Task 10: Add fp16 conv2d kernel

**Files:**
- Create: `src/backend/cpu/kernels/cpu_conv2d_f16.c`
- Modify: `src/backend/cpu/kernels/cpu_kernels.h`
- Modify: `src/backend/cpu/cpu_dispatch.c`

Uses fp16 im2col + fp16 matmul. Same scratch arena pattern as f32.

**Commit:** `cpu/kernels: add fp16 conv2d kernel`

---

### Task 11: Make transpose dtype-agnostic

**Files:**
- Modify: `src/backend/cpu/kernels/cpu_transpose.c`
- Modify: `src/backend/cpu/cpu_dispatch.c`

Remove the `in->dtype != SAM3_DTYPE_F32` check. For non-f32 dtypes, use a byte-level transpose with `sam3_dtype_size()` to get element stride. The existing NEON 4x4 block transpose only works for 4-byte elements, so fp16 (2-byte) uses a simple scalar byte copy.

Register in dispatch table for all dtypes.

**Commit:** `cpu/kernels: make transpose dtype-agnostic`

---

### Task 12: Add bf16 elementwise kernels

**Files:**
- Create: `src/backend/cpu/kernels/cpu_add_bf16.c`
- Create: `src/backend/cpu/kernels/cpu_mul_bf16.c`
- Create: `src/backend/cpu/kernels/cpu_relu_bf16.c`
- Modify: `src/backend/cpu/kernels/cpu_kernels.h`
- Modify: `src/backend/cpu/cpu_dispatch.c`

bf16 kernels are thin: load bf16 -> f32 via `bf16x4_to_f32x4`, use existing f32 NEON arithmetic, convert back via `f32x4_to_bf16x4`.

```c
static void add_bf16_neon(const uint16_t *a, const uint16_t *b,
			   uint16_t *out, int broadcast_n,
			   int start, int end)
{
	if (broadcast_n <= 0) {
		int i = start;
		for (; i + 4 <= end; i += 4) {
			float32x4_t va = bf16x4_to_f32x4(a + i);
			float32x4_t vb = bf16x4_to_f32x4(b + i);
			f32x4_to_bf16x4(out + i, vaddq_f32(va, vb));
		}
		for (; i < end; i++)
			out[i] = f32_to_bf16(bf16_to_f32(a[i]) + bf16_to_f32(b[i]));
	}
	/* ... broadcasting path similar ... */
}
```

**Commit:** `cpu/kernels: add bf16 elementwise kernels (add, mul, relu)`

---

### Task 13: Add bf16 gelu kernel

**Files:**
- Create: `src/backend/cpu/kernels/cpu_gelu_bf16.c`
- Modify: `src/backend/cpu/kernels/cpu_kernels.h`
- Modify: `src/backend/cpu/cpu_dispatch.c`

Load bf16 -> f32, apply GELU in f32, store f32 -> bf16.

**Commit:** `cpu/kernels: add bf16 GELU kernel`

---

### Task 14: Add bf16 matmul kernel

**Files:**
- Create: `src/backend/cpu/kernels/cpu_matmul_bf16.c`
- Create: `tests/test_matmul_bf16.c`
- Modify: `src/backend/cpu/kernels/cpu_kernels.h`
- Modify: `src/backend/cpu/cpu_dispatch.c`

Load bf16 input tiles -> f32, use the f32 tiled matmul core, store result f32 -> bf16.

Test: `tests/test_matmul_bf16.c` with tolerance 2e-2.

**Commit:** `cpu/kernels: add bf16 matmul kernel`

---

### Task 15: Add bf16 softmax, layernorm, conv2d kernels

**Files:**
- Create: `src/backend/cpu/kernels/cpu_softmax_bf16.c`
- Create: `src/backend/cpu/kernels/cpu_layernorm_bf16.c`
- Create: `src/backend/cpu/kernels/cpu_conv2d_bf16.c`
- Modify: `src/backend/cpu/kernels/cpu_kernels.h`
- Modify: `src/backend/cpu/cpu_dispatch.c`

All follow the upcast pattern: load bf16 -> f32, compute in f32, store f32 -> bf16.

**Commit:** `cpu/kernels: add bf16 softmax, layernorm, conv2d kernels`

---

### Task 16: Add cast kernel (SAM3_OP_CAST)

**Files:**
- Create: `src/backend/cpu/kernels/cpu_cast.c`
- Modify: `src/backend/cpu/kernels/cpu_kernels.h`
- Modify: `src/backend/cpu/cpu_dispatch.c`

The cast kernel converts between any two dtypes. `node->params[0]` holds the target dtype. The dispatch table registers cast for all source dtypes.

```c
enum sam3_error cpu_kernel_cast(const struct sam3_node *node,
				 struct sam3_threadpool *pool);
```

Uses `half.h` conversion functions. SIMD batch conversion where available.

**Commit:** `cpu/kernels: add dtype cast kernel`

---

### Task 17: Add performance benchmark

**Files:**
- Create: `tests/bench_dtype.c`
- Modify: `CMakeLists.txt` — add bench target (not auto-run by CTest)

Benchmarks matmul and elementwise add at 64x64, 256x256, 1024x1024 for f32, fp16, bf16. Reports GFLOPS. Uses `clock_gettime(CLOCK_MONOTONIC)` for timing.

Add to CMakeLists.txt:
```cmake
# Benchmark (not included in CTest)
add_executable(bench_dtype tests/bench_dtype.c)
target_link_libraries(bench_dtype sam3)
```

**Commit:** `tests: add dtype performance benchmark`

---

### Task 18: Final integration test and cleanup

**Files:**
- Modify: `src/backend/cpu/kernels/cpu_kernels.h` — update header doc
- Run full test suite

**Step 1: Run all tests**

Run: `cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DSAM3_TRACE=ON && make -j && ctest --output-on-failure`
Expected: All PASS

**Step 2: Run benchmark**

Run: `cd build && ./bench_dtype`
Expected: fp16 GFLOPS > f32 GFLOPS on Apple Silicon for matmul

**Step 3: Final commit**

```bash
git add -A
git commit -m "fp16/bf16: complete compute support with tracing and benchmarks"
```
