/*
 * tests/test_cast.c - Unit tests for the dtype cast kernel
 *
 * Tests all 9 conversion paths between {f32, f16, bf16} including
 * same-dtype memcpy, direct conversions, and cross-conversions
 * (f16<->bf16) that go through an f32 intermediate.
 *
 * Key types:  sam3_node, sam3_tensor
 * Depends on: core/half.h, core/tensor.h, core/graph.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "core/half.h"
#include "core/tensor.h"
#include "core/graph.h"
#include "backend/cpu/kernels/cpu_kernels.h"
#include "util/threadpool.h"

#include <stdint.h>
#include <string.h>

/* ── Helpers ──────────────────────────────────────────────────────── */

static void make_f32_tensor(struct sam3_tensor *t, float *data, int n)
{
	t->dtype      = SAM3_DTYPE_F32;
	t->n_dims     = 1;
	t->dims[0]    = n;
	t->dims[1]    = 1;
	t->dims[2]    = 1;
	t->dims[3]    = 1;
	t->strides[0] = 1;
	t->strides[1] = n;
	t->strides[2] = n;
	t->strides[3] = n;
	t->data       = data;
	t->nbytes     = (size_t)n * sizeof(float);
}

static void make_f16_tensor(struct sam3_tensor *t, uint16_t *data, int n)
{
	t->dtype      = SAM3_DTYPE_F16;
	t->n_dims     = 1;
	t->dims[0]    = n;
	t->dims[1]    = 1;
	t->dims[2]    = 1;
	t->dims[3]    = 1;
	t->strides[0] = 1;
	t->strides[1] = n;
	t->strides[2] = n;
	t->strides[3] = n;
	t->data       = data;
	t->nbytes     = (size_t)n * sizeof(uint16_t);
}

static void make_bf16_tensor(struct sam3_tensor *t, uint16_t *data, int n)
{
	t->dtype      = SAM3_DTYPE_BF16;
	t->n_dims     = 1;
	t->dims[0]    = n;
	t->dims[1]    = 1;
	t->dims[2]    = 1;
	t->dims[3]    = 1;
	t->strides[0] = 1;
	t->strides[1] = n;
	t->strides[2] = n;
	t->strides[3] = n;
	t->data       = data;
	t->nbytes     = (size_t)n * sizeof(uint16_t);
}

static struct sam3_node make_cast_node(struct sam3_tensor *in,
				       struct sam3_tensor *out,
				       enum sam3_dtype target_dt)
{
	struct sam3_node node;
	memset(&node, 0, sizeof(node));
	node.op        = SAM3_OP_CAST;
	node.n_inputs  = 1;
	node.inputs[0] = in;
	node.output    = out;
	node.params[0] = (int)target_dt;
	return node;
}

/* ── Tests ────────────────────────────────────────────────────────── */

static void test_cast_f32_to_f16(void)
{
	float in[8] = {1.0f, 2.5f, -3.0f, 0.0f,
		       100.0f, -0.5f, 0.125f, 42.0f};
	uint16_t out[8];
	memset(out, 0, sizeof(out));

	struct sam3_tensor tin, tout;
	make_f32_tensor(&tin, in, 8);
	make_f16_tensor(&tout, out, 8);

	struct sam3_node node = make_cast_node(&tin, &tout, SAM3_DTYPE_F16);

	struct sam3_threadpool *pool = sam3_threadpool_create(1);
	ASSERT(pool != NULL);

	enum sam3_error err = cpu_kernel_cast(&node, pool);
	ASSERT_EQ(err, SAM3_OK);

	for (int i = 0; i < 8; i++) {
		float result = fp16_to_f32(out[i]);
		ASSERT_NEAR(result, in[i], 1e-2f);
	}

	sam3_threadpool_free(pool);
}

static void test_cast_f16_to_f32(void)
{
	float ref[8] = {1.0f, 2.5f, -3.0f, 0.0f,
			100.0f, -0.5f, 0.125f, 42.0f};
	uint16_t in[8];
	float out[8];

	for (int i = 0; i < 8; i++)
		in[i] = f32_to_fp16(ref[i]);
	memset(out, 0, sizeof(out));

	struct sam3_tensor tin, tout;
	make_f16_tensor(&tin, in, 8);
	make_f32_tensor(&tout, out, 8);

	struct sam3_node node = make_cast_node(&tin, &tout, SAM3_DTYPE_F32);

	struct sam3_threadpool *pool = sam3_threadpool_create(1);
	ASSERT(pool != NULL);

	enum sam3_error err = cpu_kernel_cast(&node, pool);
	ASSERT_EQ(err, SAM3_OK);

	for (int i = 0; i < 8; i++)
		ASSERT_NEAR(out[i], ref[i], 1e-3f);

	sam3_threadpool_free(pool);
}

static void test_cast_f32_to_bf16(void)
{
	float in[8] = {1.0f, 2.5f, -3.0f, 0.0f,
		       100.0f, -0.5f, 0.125f, 42.0f};
	uint16_t out[8];
	memset(out, 0, sizeof(out));

	struct sam3_tensor tin, tout;
	make_f32_tensor(&tin, in, 8);
	make_bf16_tensor(&tout, out, 8);

	struct sam3_node node = make_cast_node(&tin, &tout, SAM3_DTYPE_BF16);

	struct sam3_threadpool *pool = sam3_threadpool_create(1);
	ASSERT(pool != NULL);

	enum sam3_error err = cpu_kernel_cast(&node, pool);
	ASSERT_EQ(err, SAM3_OK);

	for (int i = 0; i < 8; i++) {
		float result = bf16_to_f32(out[i]);
		ASSERT_NEAR(result, in[i], 1e-1f);
	}

	sam3_threadpool_free(pool);
}

static void test_cast_bf16_to_f32(void)
{
	float ref[8] = {1.0f, 2.5f, -3.0f, 0.0f,
			100.0f, -0.5f, 0.125f, 42.0f};
	uint16_t in[8];
	float out[8];

	for (int i = 0; i < 8; i++)
		in[i] = f32_to_bf16(ref[i]);
	memset(out, 0, sizeof(out));

	struct sam3_tensor tin, tout;
	make_bf16_tensor(&tin, in, 8);
	make_f32_tensor(&tout, out, 8);

	struct sam3_node node = make_cast_node(&tin, &tout, SAM3_DTYPE_F32);

	struct sam3_threadpool *pool = sam3_threadpool_create(1);
	ASSERT(pool != NULL);

	enum sam3_error err = cpu_kernel_cast(&node, pool);
	ASSERT_EQ(err, SAM3_OK);

	for (int i = 0; i < 8; i++)
		ASSERT_NEAR(out[i], ref[i], 1e-2f);

	sam3_threadpool_free(pool);
}

static void test_cast_f16_to_bf16(void)
{
	float ref[8] = {1.0f, 2.5f, -3.0f, 0.0f,
			100.0f, -0.5f, 0.125f, 42.0f};
	uint16_t in[8], out[8];

	for (int i = 0; i < 8; i++)
		in[i] = f32_to_fp16(ref[i]);
	memset(out, 0, sizeof(out));

	struct sam3_tensor tin, tout;
	make_f16_tensor(&tin, in, 8);
	make_bf16_tensor(&tout, out, 8);

	struct sam3_node node = make_cast_node(&tin, &tout, SAM3_DTYPE_BF16);

	struct sam3_threadpool *pool = sam3_threadpool_create(1);
	ASSERT(pool != NULL);

	enum sam3_error err = cpu_kernel_cast(&node, pool);
	ASSERT_EQ(err, SAM3_OK);

	/* fp16->bf16 goes via f32, so tolerance accounts for both */
	for (int i = 0; i < 8; i++) {
		float result = bf16_to_f32(out[i]);
		ASSERT_NEAR(result, ref[i], 1e-1f);
	}

	sam3_threadpool_free(pool);
}

static void test_cast_bf16_to_f16(void)
{
	float ref[8] = {1.0f, 2.5f, -3.0f, 0.0f,
			100.0f, -0.5f, 0.125f, 42.0f};
	uint16_t in[8], out[8];

	for (int i = 0; i < 8; i++)
		in[i] = f32_to_bf16(ref[i]);
	memset(out, 0, sizeof(out));

	struct sam3_tensor tin, tout;
	make_bf16_tensor(&tin, in, 8);
	make_f16_tensor(&tout, out, 8);

	struct sam3_node node = make_cast_node(&tin, &tout, SAM3_DTYPE_F16);

	struct sam3_threadpool *pool = sam3_threadpool_create(1);
	ASSERT(pool != NULL);

	enum sam3_error err = cpu_kernel_cast(&node, pool);
	ASSERT_EQ(err, SAM3_OK);

	/* bf16->fp16 goes via f32, so tolerance accounts for both */
	for (int i = 0; i < 8; i++) {
		float result = fp16_to_f32(out[i]);
		ASSERT_NEAR(result, ref[i], 1e-1f);
	}

	sam3_threadpool_free(pool);
}

static void test_cast_same_dtype_f32(void)
{
	float in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
	float out[4];
	memset(out, 0, sizeof(out));

	struct sam3_tensor tin, tout;
	make_f32_tensor(&tin, in, 4);
	make_f32_tensor(&tout, out, 4);

	struct sam3_node node = make_cast_node(&tin, &tout, SAM3_DTYPE_F32);

	struct sam3_threadpool *pool = sam3_threadpool_create(1);
	ASSERT(pool != NULL);

	enum sam3_error err = cpu_kernel_cast(&node, pool);
	ASSERT_EQ(err, SAM3_OK);

	/* Same dtype should be exact memcpy */
	for (int i = 0; i < 4; i++)
		ASSERT_NEAR(out[i], in[i], 0.0f);

	sam3_threadpool_free(pool);
}

static void test_cast_same_dtype_f16(void)
{
	uint16_t in[4] = {
		f32_to_fp16(1.0f),
		f32_to_fp16(2.0f),
		f32_to_fp16(3.0f),
		f32_to_fp16(4.0f),
	};
	uint16_t out[4];
	memset(out, 0, sizeof(out));

	struct sam3_tensor tin, tout;
	make_f16_tensor(&tin, in, 4);
	make_f16_tensor(&tout, out, 4);

	struct sam3_node node = make_cast_node(&tin, &tout, SAM3_DTYPE_F16);

	struct sam3_threadpool *pool = sam3_threadpool_create(1);
	ASSERT(pool != NULL);

	enum sam3_error err = cpu_kernel_cast(&node, pool);
	ASSERT_EQ(err, SAM3_OK);

	/* Same dtype: bit-exact copy */
	for (int i = 0; i < 4; i++)
		ASSERT_EQ(out[i], in[i]);

	sam3_threadpool_free(pool);
}

static void test_cast_same_dtype_bf16(void)
{
	uint16_t in[4] = {
		f32_to_bf16(1.0f),
		f32_to_bf16(2.0f),
		f32_to_bf16(3.0f),
		f32_to_bf16(4.0f),
	};
	uint16_t out[4];
	memset(out, 0, sizeof(out));

	struct sam3_tensor tin, tout;
	make_bf16_tensor(&tin, in, 4);
	make_bf16_tensor(&tout, out, 4);

	struct sam3_node node = make_cast_node(&tin, &tout, SAM3_DTYPE_BF16);

	struct sam3_threadpool *pool = sam3_threadpool_create(1);
	ASSERT(pool != NULL);

	enum sam3_error err = cpu_kernel_cast(&node, pool);
	ASSERT_EQ(err, SAM3_OK);

	for (int i = 0; i < 4; i++)
		ASSERT_EQ(out[i], in[i]);

	sam3_threadpool_free(pool);
}

static void test_cast_null_tensor(void)
{
	struct sam3_node node;
	memset(&node, 0, sizeof(node));
	node.op       = SAM3_OP_CAST;
	node.n_inputs = 1;
	/* inputs[0] is NULL */

	struct sam3_threadpool *pool = sam3_threadpool_create(1);
	ASSERT(pool != NULL);

	enum sam3_error err = cpu_kernel_cast(&node, pool);
	ASSERT_EQ(err, SAM3_EINVAL);

	sam3_threadpool_free(pool);
}

static void test_cast_dtype_mismatch(void)
{
	/* output dtype doesn't match params[0] */
	float in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
	float out[4];
	memset(out, 0, sizeof(out));

	struct sam3_tensor tin, tout;
	make_f32_tensor(&tin, in, 4);
	make_f32_tensor(&tout, out, 4);  /* F32 output */

	struct sam3_node node = make_cast_node(&tin, &tout, SAM3_DTYPE_F16);
	/* params[0]=F16 but output dtype=F32: mismatch */

	struct sam3_threadpool *pool = sam3_threadpool_create(1);
	ASSERT(pool != NULL);

	enum sam3_error err = cpu_kernel_cast(&node, pool);
	ASSERT_EQ(err, SAM3_EINVAL);

	sam3_threadpool_free(pool);
}

static void test_cast_f32_to_f16_roundtrip(void)
{
	/* Cast f32->f16->f32 and verify round-trip accuracy */
	float orig[8] = {0.0f, 1.0f, -1.0f, 0.5f,
			 -0.5f, 3.14159f, 65504.0f, -65504.0f};
	uint16_t mid[8];
	float final[8];
	memset(mid, 0, sizeof(mid));
	memset(final, 0, sizeof(final));

	struct sam3_tensor t_f32_in, t_f16, t_f32_out;
	make_f32_tensor(&t_f32_in, orig, 8);
	make_f16_tensor(&t_f16, mid, 8);
	make_f32_tensor(&t_f32_out, final, 8);

	struct sam3_threadpool *pool = sam3_threadpool_create(1);
	ASSERT(pool != NULL);

	/* f32 -> f16 */
	struct sam3_node n1 = make_cast_node(&t_f32_in, &t_f16,
					     SAM3_DTYPE_F16);
	enum sam3_error err = cpu_kernel_cast(&n1, pool);
	ASSERT_EQ(err, SAM3_OK);

	/* f16 -> f32 */
	struct sam3_node n2 = make_cast_node(&t_f16, &t_f32_out,
					     SAM3_DTYPE_F32);
	err = cpu_kernel_cast(&n2, pool);
	ASSERT_EQ(err, SAM3_OK);

	for (int i = 0; i < 8; i++)
		ASSERT_NEAR(final[i], orig[i], 1e-2f);

	sam3_threadpool_free(pool);
}

int main(void)
{
	test_cast_f32_to_f16();
	test_cast_f16_to_f32();
	test_cast_f32_to_bf16();
	test_cast_bf16_to_f32();
	test_cast_f16_to_bf16();
	test_cast_bf16_to_f16();
	test_cast_same_dtype_f32();
	test_cast_same_dtype_f16();
	test_cast_same_dtype_bf16();
	test_cast_null_tensor();
	test_cast_dtype_mismatch();
	test_cast_f32_to_f16_roundtrip();

	TEST_REPORT();
}
