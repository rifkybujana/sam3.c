/*
 * tests/test_elementwise_f16.c - Unit tests for fp16 elementwise kernels
 *
 * Tests fp16 add, mul, relu against f32 reference values with tolerance
 * of 1e-3 (fp16 has ~0.1% relative error for normal values).
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

/* ── Tests ────────────────────────────────────────────────────────── */

static void test_add_f16(void)
{
	/* a = [1,2,3,4,5,6,7,8], b = [0.5,...,0.5], expected = a + b */
	float a_f32[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
	float b_f32[8] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
	uint16_t a_f16[8], b_f16[8], out_f16[8];
	int i;

	for (i = 0; i < 8; i++) {
		a_f16[i] = f32_to_fp16(a_f32[i]);
		b_f16[i] = f32_to_fp16(b_f32[i]);
	}
	memset(out_f16, 0, sizeof(out_f16));

	struct sam3_tensor ta, tb, tout;
	make_f16_tensor(&ta,   a_f16,   8);
	make_f16_tensor(&tb,   b_f16,   8);
	make_f16_tensor(&tout, out_f16, 8);

	struct sam3_node node;
	memset(&node, 0, sizeof(node));
	node.op        = SAM3_OP_ADD;
	node.n_inputs  = 2;
	node.inputs[0] = &ta;
	node.inputs[1] = &tb;
	node.output    = &tout;

	struct sam3_threadpool *pool = sam3_threadpool_create(1);
	ASSERT(pool != NULL);

	enum sam3_error err = cpu_kernel_add_f16(&node, pool);
	ASSERT_EQ(err, SAM3_OK);

	for (i = 0; i < 8; i++) {
		float result   = fp16_to_f32(out_f16[i]);
		float expected = a_f32[i] + b_f32[i];
		ASSERT_NEAR(result, expected, 1e-3f);
	}

	sam3_threadpool_free(pool);
}

static void test_mul_f16(void)
{
	/* a = [1,2,3,4,5,6,7,8], b = [2,2,...,2], expected = a * 2 */
	float a_f32[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
	float b_f32[8] = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
	uint16_t a_f16[8], b_f16[8], out_f16[8];
	int i;

	for (i = 0; i < 8; i++) {
		a_f16[i] = f32_to_fp16(a_f32[i]);
		b_f16[i] = f32_to_fp16(b_f32[i]);
	}
	memset(out_f16, 0, sizeof(out_f16));

	struct sam3_tensor ta, tb, tout;
	make_f16_tensor(&ta,   a_f16,   8);
	make_f16_tensor(&tb,   b_f16,   8);
	make_f16_tensor(&tout, out_f16, 8);

	struct sam3_node node;
	memset(&node, 0, sizeof(node));
	node.op        = SAM3_OP_MUL;
	node.n_inputs  = 2;
	node.inputs[0] = &ta;
	node.inputs[1] = &tb;
	node.output    = &tout;

	struct sam3_threadpool *pool = sam3_threadpool_create(1);
	ASSERT(pool != NULL);

	enum sam3_error err = cpu_kernel_mul_f16(&node, pool);
	ASSERT_EQ(err, SAM3_OK);

	for (i = 0; i < 8; i++) {
		float result   = fp16_to_f32(out_f16[i]);
		float expected = a_f32[i] * b_f32[i];
		ASSERT_NEAR(result, expected, 1e-3f);
	}

	sam3_threadpool_free(pool);
}

static void test_relu_f16(void)
{
	/* in = [-2,-1,0,1,2,-3,4,-0.5], expected = max(0, x) */
	float in_f32[8]  = {-2.0f, -1.0f, 0.0f, 1.0f,
			    2.0f, -3.0f, 4.0f, -0.5f};
	uint16_t in_f16[8], out_f16[8];
	int i;

	for (i = 0; i < 8; i++)
		in_f16[i] = f32_to_fp16(in_f32[i]);
	memset(out_f16, 0, sizeof(out_f16));

	struct sam3_tensor tin, tout;
	make_f16_tensor(&tin,  in_f16,  8);
	make_f16_tensor(&tout, out_f16, 8);

	struct sam3_node node;
	memset(&node, 0, sizeof(node));
	node.op        = SAM3_OP_RELU;
	node.n_inputs  = 1;
	node.inputs[0] = &tin;
	node.output    = &tout;

	struct sam3_threadpool *pool = sam3_threadpool_create(1);
	ASSERT(pool != NULL);

	enum sam3_error err = cpu_kernel_relu_f16(&node, pool);
	ASSERT_EQ(err, SAM3_OK);

	for (i = 0; i < 8; i++) {
		float result   = fp16_to_f32(out_f16[i]);
		float expected = in_f32[i] > 0.0f ? in_f32[i] : 0.0f;
		ASSERT_NEAR(result, expected, 1e-3f);
	}

	sam3_threadpool_free(pool);
}

static void test_add_f16_dtype_reject(void)
{
	/* Passing F32 tensors to the F16 kernel must return SAM3_EINVAL */
	float a_f32[4] = {1.0f, 2.0f, 3.0f, 4.0f};
	float out_f32[4] = {0};

	struct sam3_tensor ta, tout;
	ta.dtype      = SAM3_DTYPE_F32;
	ta.n_dims     = 1;
	ta.dims[0]    = 4;
	ta.dims[1]    = 1;
	ta.dims[2]    = 1;
	ta.dims[3]    = 1;
	ta.strides[0] = 1;
	ta.strides[1] = 4;
	ta.strides[2] = 4;
	ta.strides[3] = 4;
	ta.data       = a_f32;
	ta.nbytes     = 4 * sizeof(float);

	tout = ta;
	tout.data  = out_f32;

	struct sam3_node node;
	memset(&node, 0, sizeof(node));
	node.op        = SAM3_OP_ADD;
	node.n_inputs  = 2;
	node.inputs[0] = &ta;
	node.inputs[1] = &ta;
	node.output    = &tout;

	struct sam3_threadpool *pool = sam3_threadpool_create(1);
	ASSERT(pool != NULL);

	enum sam3_error err = cpu_kernel_add_f16(&node, pool);
	ASSERT_EQ(err, SAM3_EINVAL);

	sam3_threadpool_free(pool);
}

int main(void)
{
	test_add_f16();
	test_mul_f16();
	test_relu_f16();
	test_add_f16_dtype_reject();

	TEST_REPORT();
}
