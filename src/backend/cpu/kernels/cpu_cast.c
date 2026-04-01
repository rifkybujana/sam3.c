/*
 * src/backend/cpu/kernels/cpu_cast.c - Dtype cast kernel
 *
 * Converts between f32, f16, and bf16 tensor data types. The target
 * dtype is read from node->params[0]. Same-dtype casts reduce to
 * memcpy. Cross-dtype casts use scalar conversion functions from
 * core/half.h, with NEON SIMD paths for 4-wide batch conversions
 * on aarch64.
 *
 * Key types:  sam3_node, sam3_tensor, sam3_dtype
 * Depends on: cpu_kernels.h, core/half.h, core/tensor.h, util/log.h
 * Used by:    cpu_dispatch.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "cpu_kernels.h"
#include "core/half.h"
#include "core/tensor.h"
#include "util/log.h"

#include <string.h>

/* ── Static conversion functions ───────────────────────────────────── */

static void cast_f32_to_f16(const void *in, void *out, int n)
{
	const float *src = (const float *)in;
	uint16_t *dst = (uint16_t *)out;

#if defined(SAM3_HAS_NEON) || \
	(defined(__aarch64__) && defined(__ARM_NEON))
	int i = 0;
	for (; i + 4 <= n; i += 4) {
		float32x4_t v = vld1q_f32(src + i);
		f32x4_to_fp16x4(dst + i, v);
	}
	for (; i < n; i++)
		dst[i] = f32_to_fp16(src[i]);
#else
	for (int i = 0; i < n; i++)
		dst[i] = f32_to_fp16(src[i]);
#endif
}

static void cast_f32_to_bf16(const void *in, void *out, int n)
{
	const float *src = (const float *)in;
	uint16_t *dst = (uint16_t *)out;

#if defined(SAM3_HAS_NEON) || \
	(defined(__aarch64__) && defined(__ARM_NEON))
	int i = 0;
	for (; i + 4 <= n; i += 4) {
		float32x4_t v = vld1q_f32(src + i);
		f32x4_to_bf16x4(dst + i, v);
	}
	for (; i < n; i++)
		dst[i] = f32_to_bf16(src[i]);
#else
	for (int i = 0; i < n; i++)
		dst[i] = f32_to_bf16(src[i]);
#endif
}

static void cast_f16_to_f32(const void *in, void *out, int n)
{
	const uint16_t *src = (const uint16_t *)in;
	float *dst = (float *)out;

#if defined(SAM3_HAS_NEON) || \
	(defined(__aarch64__) && defined(__ARM_NEON))
	int i = 0;
	for (; i + 4 <= n; i += 4) {
		float32x4_t v = fp16x4_to_f32x4(src + i);
		vst1q_f32(dst + i, v);
	}
	for (; i < n; i++)
		dst[i] = fp16_to_f32(src[i]);
#else
	for (int i = 0; i < n; i++)
		dst[i] = fp16_to_f32(src[i]);
#endif
}

static void cast_f16_to_bf16(const void *in, void *out, int n)
{
	const uint16_t *src = (const uint16_t *)in;
	uint16_t *dst = (uint16_t *)out;

#if defined(SAM3_HAS_NEON) || \
	(defined(__aarch64__) && defined(__ARM_NEON))
	int i = 0;
	for (; i + 4 <= n; i += 4) {
		float32x4_t v = fp16x4_to_f32x4(src + i);
		f32x4_to_bf16x4(dst + i, v);
	}
	for (; i < n; i++)
		dst[i] = f32_to_bf16(fp16_to_f32(src[i]));
#else
	for (int i = 0; i < n; i++)
		dst[i] = f32_to_bf16(fp16_to_f32(src[i]));
#endif
}

static void cast_bf16_to_f32(const void *in, void *out, int n)
{
	const uint16_t *src = (const uint16_t *)in;
	float *dst = (float *)out;

#if defined(SAM3_HAS_NEON) || \
	(defined(__aarch64__) && defined(__ARM_NEON))
	int i = 0;
	for (; i + 4 <= n; i += 4) {
		float32x4_t v = bf16x4_to_f32x4(src + i);
		vst1q_f32(dst + i, v);
	}
	for (; i < n; i++)
		dst[i] = bf16_to_f32(src[i]);
#else
	for (int i = 0; i < n; i++)
		dst[i] = bf16_to_f32(src[i]);
#endif
}

static void cast_bf16_to_f16(const void *in, void *out, int n)
{
	const uint16_t *src = (const uint16_t *)in;
	uint16_t *dst = (uint16_t *)out;

#if defined(SAM3_HAS_NEON) || \
	(defined(__aarch64__) && defined(__ARM_NEON))
	int i = 0;
	for (; i + 4 <= n; i += 4) {
		float32x4_t v = bf16x4_to_f32x4(src + i);
		f32x4_to_fp16x4(dst + i, v);
	}
	for (; i < n; i++)
		dst[i] = f32_to_fp16(bf16_to_f32(src[i]));
#else
	for (int i = 0; i < n; i++)
		dst[i] = f32_to_fp16(bf16_to_f32(src[i]));
#endif
}

/* ── Public kernel ─────────────────────────────────────────────────── */

/*
 * cpu_kernel_cast - Convert tensor data between dtypes.
 *
 * @node: Node with inputs[0] as source tensor. node->params[0] holds
 *        the target dtype (as int). node->output->dtype must match
 *        node->params[0].
 * @pool: Thread pool (unused; cast is lightweight).
 *
 * Supports all 9 combinations of {f32, f16, bf16} x {f32, f16, bf16}.
 * Same-dtype casts are a memcpy. Returns SAM3_OK on success,
 * SAM3_EINVAL for NULL tensors or dtype mismatch, SAM3_EDTYPE for
 * unsupported conversion pairs.
 */
enum sam3_error cpu_kernel_cast(const struct sam3_node *node,
				struct sam3_threadpool *pool)
{
	(void)pool;

	if (!node->inputs[0] || !node->output) {
		sam3_log_error("cast: NULL tensor");
		return SAM3_EINVAL;
	}

	enum sam3_dtype src_dt = node->inputs[0]->dtype;
	enum sam3_dtype dst_dt = (enum sam3_dtype)node->params[0];
	int n = sam3_tensor_nelems(node->inputs[0]);

	if (dst_dt != node->output->dtype) {
		sam3_log_error("cast: output dtype %d != params[0] %d",
			       node->output->dtype, dst_dt);
		return SAM3_EINVAL;
	}

	/* Same dtype: memcpy */
	if (src_dt == dst_dt) {
		size_t sz = sam3_dtype_size(src_dt);
		memcpy(node->output->data, node->inputs[0]->data,
		       (size_t)n * sz);
		return SAM3_OK;
	}

	const void *in = node->inputs[0]->data;
	void *out = node->output->data;

	if (src_dt == SAM3_DTYPE_F32 && dst_dt == SAM3_DTYPE_F16) {
		cast_f32_to_f16(in, out, n);
	} else if (src_dt == SAM3_DTYPE_F32 && dst_dt == SAM3_DTYPE_BF16) {
		cast_f32_to_bf16(in, out, n);
	} else if (src_dt == SAM3_DTYPE_F16 && dst_dt == SAM3_DTYPE_F32) {
		cast_f16_to_f32(in, out, n);
	} else if (src_dt == SAM3_DTYPE_F16 && dst_dt == SAM3_DTYPE_BF16) {
		cast_f16_to_bf16(in, out, n);
	} else if (src_dt == SAM3_DTYPE_BF16 && dst_dt == SAM3_DTYPE_F32) {
		cast_bf16_to_f32(in, out, n);
	} else if (src_dt == SAM3_DTYPE_BF16 && dst_dt == SAM3_DTYPE_F16) {
		cast_bf16_to_f16(in, out, n);
	} else {
		sam3_log_error("cast: unsupported %d -> %d", src_dt, dst_dt);
		return SAM3_EDTYPE;
	}

	return SAM3_OK;
}
