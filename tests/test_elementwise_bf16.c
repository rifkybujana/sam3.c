/*
 * tests/test_elementwise_bf16.c - Unit tests for bf16 elementwise kernels
 *
 * Tests bf16 add, mul, relu against f32 reference values with tolerance
 * of 1e-2 (bf16 has 8-bit mantissa, ~0.8% relative error for normals).
 *
 * Key types:  sam3_node, sam3_tensor
 * Depends on: core/half.h, core/tensor.h, core/graph.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "core/half.h"
#include "core/tensor.h"
#include "core/graph.h"
#include "backend/cpu/kernels/cpu_kernels.h"
#include "util/threadpool.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

/* ── Helpers ──────────────────────────────────────────────────────── */

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

/* ── Tests ────────────────────────────────────────────────────────── */

static void test_add_bf16(void)
{
	/* a = [1,2,3,4,5,6,7,8], b = [0.5,...,0.5], expected = a + b */
	float a_f32[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
	float b_f32[8] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
	uint16_t a_bf16[8], b_bf16[8], out_bf16[8];
	int i;

	for (i = 0; i < 8; i++) {
		a_bf16[i] = f32_to_bf16(a_f32[i]);
		b_bf16[i] = f32_to_bf16(b_f32[i]);
	}
	memset(out_bf16, 0, sizeof(out_bf16));

	struct sam3_tensor ta, tb, tout;
	make_bf16_tensor(&ta,   a_bf16,   8);
	make_bf16_tensor(&tb,   b_bf16,   8);
	make_bf16_tensor(&tout, out_bf16, 8);

	struct sam3_node node;
	memset(&node, 0, sizeof(node));
	node.op        = SAM3_OP_ADD;
	node.n_inputs  = 2;
	node.inputs[0] = &ta;
	node.inputs[1] = &tb;
	node.output    = &tout;

	struct sam3_threadpool *pool = sam3_threadpool_create(1);
	ASSERT(pool != NULL);

	enum sam3_error err = cpu_kernel_add_bf16(&node, pool);
	ASSERT_EQ(err, SAM3_OK);

	for (i = 0; i < 8; i++) {
		float result   = bf16_to_f32(out_bf16[i]);
		float expected = a_f32[i] + b_f32[i];
		ASSERT_NEAR(result, expected, 1e-2f);
	}

	sam3_threadpool_free(pool);
}

static void test_mul_bf16(void)
{
	/* a = [1,2,3,4,5,6,7,8], b = [2,2,...,2], expected = a * 2 */
	float a_f32[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
	float b_f32[8] = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
	uint16_t a_bf16[8], b_bf16[8], out_bf16[8];
	int i;

	for (i = 0; i < 8; i++) {
		a_bf16[i] = f32_to_bf16(a_f32[i]);
		b_bf16[i] = f32_to_bf16(b_f32[i]);
	}
	memset(out_bf16, 0, sizeof(out_bf16));

	struct sam3_tensor ta, tb, tout;
	make_bf16_tensor(&ta,   a_bf16,   8);
	make_bf16_tensor(&tb,   b_bf16,   8);
	make_bf16_tensor(&tout, out_bf16, 8);

	struct sam3_node node;
	memset(&node, 0, sizeof(node));
	node.op        = SAM3_OP_MUL;
	node.n_inputs  = 2;
	node.inputs[0] = &ta;
	node.inputs[1] = &tb;
	node.output    = &tout;

	struct sam3_threadpool *pool = sam3_threadpool_create(1);
	ASSERT(pool != NULL);

	enum sam3_error err = cpu_kernel_mul_bf16(&node, pool);
	ASSERT_EQ(err, SAM3_OK);

	for (i = 0; i < 8; i++) {
		float result   = bf16_to_f32(out_bf16[i]);
		float expected = a_f32[i] * b_f32[i];
		ASSERT_NEAR(result, expected, 1e-2f);
	}

	sam3_threadpool_free(pool);
}

static void test_relu_bf16(void)
{
	/* in = [-2,-1,0,1,2,-3,4,-0.5], expected = max(0, x) */
	float in_f32[8]  = {-2.0f, -1.0f, 0.0f, 1.0f,
			    2.0f, -3.0f, 4.0f, -0.5f};
	uint16_t in_bf16[8], out_bf16[8];
	int i;

	for (i = 0; i < 8; i++)
		in_bf16[i] = f32_to_bf16(in_f32[i]);
	memset(out_bf16, 0, sizeof(out_bf16));

	struct sam3_tensor tin, tout;
	make_bf16_tensor(&tin,  in_bf16,  8);
	make_bf16_tensor(&tout, out_bf16, 8);

	struct sam3_node node;
	memset(&node, 0, sizeof(node));
	node.op        = SAM3_OP_RELU;
	node.n_inputs  = 1;
	node.inputs[0] = &tin;
	node.output    = &tout;

	struct sam3_threadpool *pool = sam3_threadpool_create(1);
	ASSERT(pool != NULL);

	enum sam3_error err = cpu_kernel_relu_bf16(&node, pool);
	ASSERT_EQ(err, SAM3_OK);

	for (i = 0; i < 8; i++) {
		float result   = bf16_to_f32(out_bf16[i]);
		float expected = in_f32[i] > 0.0f ? in_f32[i] : 0.0f;
		ASSERT_NEAR(result, expected, 1e-2f);
	}

	sam3_threadpool_free(pool);
}

static void test_gelu_bf16(void)
{
	float in_f32[8] = {-2.0f, -1.0f, -0.5f, 0.0f,
			   0.5f, 1.0f, 1.5f, 2.0f};
	uint16_t in_bf16[8], out_bf16[8];
	int i;

	for (i = 0; i < 8; i++)
		in_bf16[i] = f32_to_bf16(in_f32[i]);
	memset(out_bf16, 0, sizeof(out_bf16));

	/* Compute expected GELU in f32 */
	float expected[8];
	for (i = 0; i < 8; i++) {
		float x = in_f32[i];
		float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
		expected[i] = 0.5f * x * (1.0f + tanhf(inner));
	}

	struct sam3_tensor tin, tout;
	make_bf16_tensor(&tin,  in_bf16,  8);
	make_bf16_tensor(&tout, out_bf16, 8);

	struct sam3_node node;
	memset(&node, 0, sizeof(node));
	node.op        = SAM3_OP_GELU;
	node.n_inputs  = 1;
	node.inputs[0] = &tin;
	node.output    = &tout;

	struct sam3_threadpool *pool = sam3_threadpool_create(1);
	ASSERT(pool != NULL);

	enum sam3_error err = cpu_kernel_gelu_bf16(&node, pool);
	ASSERT_EQ(err, SAM3_OK);

	for (i = 0; i < 8; i++) {
		float result = bf16_to_f32(out_bf16[i]);
		ASSERT_NEAR(result, expected[i], 1e-2f);
	}

	sam3_threadpool_free(pool);
}

static void test_softmax_bf16(void)
{
	float in_f32[8] = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f};
	uint16_t in_bf16[8], out_bf16[8];
	int i;

	for (i = 0; i < 8; i++)
		in_bf16[i] = f32_to_bf16(in_f32[i]);
	memset(out_bf16, 0, sizeof(out_bf16));

	/* Two rows of 4: softmax along last dim */
	struct sam3_tensor tin, tout;
	memset(&tin, 0, sizeof(tin));
	memset(&tout, 0, sizeof(tout));
	tin.dtype      = SAM3_DTYPE_BF16;
	tin.n_dims     = 2;
	tin.dims[0]    = 2;
	tin.dims[1]    = 4;
	tin.dims[2]    = 1;
	tin.dims[3]    = 1;
	tin.strides[0] = 4;
	tin.strides[1] = 1;
	tin.strides[2] = 8;
	tin.strides[3] = 8;
	tin.data       = in_bf16;
	tin.nbytes     = sizeof(in_bf16);

	tout = tin;
	tout.data   = out_bf16;
	tout.nbytes = sizeof(out_bf16);

	struct sam3_node node;
	memset(&node, 0, sizeof(node));
	node.op        = SAM3_OP_SOFTMAX;
	node.n_inputs  = 1;
	node.inputs[0] = &tin;
	node.output    = &tout;

	struct sam3_threadpool *pool = sam3_threadpool_create(1);
	ASSERT(pool != NULL);

	enum sam3_error err = cpu_kernel_softmax_bf16(&node, pool);
	ASSERT_EQ(err, SAM3_OK);

	/* Compute f32 reference for each row */
	for (int r = 0; r < 2; r++) {
		float row[4];
		for (i = 0; i < 4; i++)
			row[i] = in_f32[r * 4 + i];

		/* Max */
		float max_val = row[0];
		for (i = 1; i < 4; i++)
			if (row[i] > max_val)
				max_val = row[i];

		/* exp(x - max) and sum */
		float sum = 0.0f;
		for (i = 0; i < 4; i++) {
			row[i] = expf(row[i] - max_val);
			sum += row[i];
		}

		/* Normalize and compare */
		for (i = 0; i < 4; i++) {
			float expected = row[i] / sum;
			float result = bf16_to_f32(out_bf16[r * 4 + i]);
			ASSERT_NEAR(result, expected, 1e-2f);
		}
	}

	sam3_threadpool_free(pool);
}

static void test_layernorm_bf16(void)
{
	float in_f32[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
	uint16_t in_bf16[8], out_bf16[8];
	int i;

	for (i = 0; i < 8; i++)
		in_bf16[i] = f32_to_bf16(in_f32[i]);
	memset(out_bf16, 0, sizeof(out_bf16));

	/* Single row of 8, no gamma/beta */
	struct sam3_tensor tin, tout;
	make_bf16_tensor(&tin,  in_bf16,  8);
	make_bf16_tensor(&tout, out_bf16, 8);

	struct sam3_node node;
	memset(&node, 0, sizeof(node));
	node.op        = SAM3_OP_LAYERNORM;
	node.n_inputs  = 1;
	node.inputs[0] = &tin;
	node.output    = &tout;

	struct sam3_threadpool *pool = sam3_threadpool_create(1);
	ASSERT(pool != NULL);

	enum sam3_error err = cpu_kernel_layernorm_bf16(&node, pool);
	ASSERT_EQ(err, SAM3_OK);

	/* Compute f32 reference: mean, variance, normalize */
	float mean = 0.0f;
	for (i = 0; i < 8; i++)
		mean += in_f32[i];
	mean /= 8.0f;

	float var_sum = 0.0f;
	for (i = 0; i < 8; i++) {
		float d = in_f32[i] - mean;
		var_sum += d * d;
	}
	float inv_std = 1.0f / sqrtf(var_sum / 8.0f + 1e-5f);

	for (i = 0; i < 8; i++) {
		float expected = (in_f32[i] - mean) * inv_std;
		float result = bf16_to_f32(out_bf16[i]);
		ASSERT_NEAR(result, expected, 1e-2f);
	}

	sam3_threadpool_free(pool);
}

static void test_add_bf16_dtype_reject(void)
{
	/* Passing F32 tensors to the BF16 kernel must return SAM3_EINVAL */
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

	enum sam3_error err = cpu_kernel_add_bf16(&node, pool);
	ASSERT_EQ(err, SAM3_EINVAL);

	sam3_threadpool_free(pool);
}

int main(void)
{
	test_add_bf16();
	test_mul_bf16();
	test_relu_bf16();
	test_gelu_bf16();
	test_softmax_bf16();
	test_layernorm_bf16();
	test_add_bf16_dtype_reject();

	TEST_REPORT();
}
