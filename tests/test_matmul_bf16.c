/*
 * tests/test_matmul_bf16.c - Unit tests for bf16 matmul kernel
 *
 * Tests bf16 matrix multiplication against f32 reference with tolerance
 * of 2e-2 (bf16 has only 8 bits of mantissa, so accumulated rounding
 * errors are larger than fp16).
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

#include <string.h>
#include <stdint.h>

/* --- Helpers ────── --- */

static void make_bf16_tensor_2d(struct sam3_tensor *t, uint16_t *data,
				int rows, int cols)
{
	t->dtype      = SAM3_DTYPE_BF16;
	t->n_dims     = 2;
	t->dims[0]    = rows;
	t->dims[1]    = cols;
	t->dims[2]    = 1;
	t->dims[3]    = 1;
	t->strides[0] = cols;
	t->strides[1] = 1;
	t->strides[2] = rows * cols;
	t->strides[3] = rows * cols;
	t->data       = data;
	t->nbytes     = (size_t)rows * cols * sizeof(uint16_t);
}

/* --- Tests ──────── --- */

/*
 * test_matmul_bf16_small - 4x3 @ 3x4 = 4x4 matmul.
 *
 * Fills A and B with simple scaled values, computes f32 reference,
 * converts inputs to bf16, runs the kernel, and checks output against
 * the reference with tolerance 2e-2.
 */
static void test_matmul_bf16_small(void)
{
	int i, j, k;
	const int M = 4, K = 3, N = 4;

	/* Build f32 A and B */
	float a_f32[4 * 3];
	float b_f32[3 * 4];

	for (i = 0; i < M; i++)
		for (k = 0; k < K; k++)
			a_f32[i * K + k] = (float)(i + 1) * (float)(k + 1) * 0.1f;

	for (k = 0; k < K; k++)
		for (j = 0; j < N; j++)
			b_f32[k * N + j] = (float)(k + 1) * (float)(j + 1) * 0.1f;

	/* Compute f32 reference */
	float c_ref[4 * 4];
	memset(c_ref, 0, sizeof(c_ref));
	for (i = 0; i < M; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k < K; k++)
				c_ref[i * N + j] +=
					a_f32[i * K + k] * b_f32[k * N + j];

	/* Convert inputs to bf16 */
	uint16_t a_bf16[4 * 3], b_bf16[3 * 4], c_bf16[4 * 4];

	for (i = 0; i < M * K; i++)
		a_bf16[i] = f32_to_bf16(a_f32[i]);
	for (i = 0; i < K * N; i++)
		b_bf16[i] = f32_to_bf16(b_f32[i]);
	memset(c_bf16, 0, sizeof(c_bf16));

	/* Build tensors and node */
	struct sam3_tensor ta, tb, tc;
	make_bf16_tensor_2d(&ta, a_bf16, M, K);
	make_bf16_tensor_2d(&tb, b_bf16, K, N);
	make_bf16_tensor_2d(&tc, c_bf16, M, N);

	struct sam3_node node;
	memset(&node, 0, sizeof(node));
	node.op        = SAM3_OP_MATMUL;
	node.n_inputs  = 2;
	node.inputs[0] = &ta;
	node.inputs[1] = &tb;
	node.output    = &tc;

	struct sam3_threadpool *pool = sam3_threadpool_create(1);
	ASSERT(pool != NULL);

	enum sam3_error err = cpu_kernel_matmul_bf16(&node, pool);
	ASSERT_EQ(err, SAM3_OK);

	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			float result   = bf16_to_f32(c_bf16[i * N + j]);
			float expected = c_ref[i * N + j];
			ASSERT_NEAR(result, expected, 2e-2f);
		}
	}

	sam3_threadpool_free(pool);
}

/*
 * test_matmul_bf16_identity - 4x4 @ identity 4x4 = A.
 *
 * Multiplying any matrix by the identity should return the original
 * matrix within bf16 precision.
 */
static void test_matmul_bf16_identity(void)
{
	int i, j;
	const int M = 4, N = 4;

	/* A is a simple 4x4 matrix with values (i+1)*(j+1)*0.25 */
	float a_f32[4 * 4];
	for (i = 0; i < M; i++)
		for (j = 0; j < N; j++)
			a_f32[i * N + j] =
				(float)(i + 1) * (float)(j + 1) * 0.25f;

	/* B is 4x4 identity */
	float b_f32[4 * 4];
	memset(b_f32, 0, sizeof(b_f32));
	for (i = 0; i < N; i++)
		b_f32[i * N + i] = 1.0f;

	/* Convert to bf16 */
	uint16_t a_bf16[4 * 4], b_bf16[4 * 4], c_bf16[4 * 4];

	for (i = 0; i < M * N; i++)
		a_bf16[i] = f32_to_bf16(a_f32[i]);
	for (i = 0; i < N * N; i++)
		b_bf16[i] = f32_to_bf16(b_f32[i]);
	memset(c_bf16, 0, sizeof(c_bf16));

	struct sam3_tensor ta, tb, tc;
	make_bf16_tensor_2d(&ta, a_bf16, M, N);
	make_bf16_tensor_2d(&tb, b_bf16, N, N);
	make_bf16_tensor_2d(&tc, c_bf16, M, N);

	struct sam3_node node;
	memset(&node, 0, sizeof(node));
	node.op        = SAM3_OP_MATMUL;
	node.n_inputs  = 2;
	node.inputs[0] = &ta;
	node.inputs[1] = &tb;
	node.output    = &tc;

	struct sam3_threadpool *pool = sam3_threadpool_create(1);
	ASSERT(pool != NULL);

	enum sam3_error err = cpu_kernel_matmul_bf16(&node, pool);
	ASSERT_EQ(err, SAM3_OK);

	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			float result   = bf16_to_f32(c_bf16[i * N + j]);
			float expected = a_f32[i * N + j];
			ASSERT_NEAR(result, expected, 2e-2f);
		}
	}

	sam3_threadpool_free(pool);
}

int main(void)
{
	test_matmul_bf16_small();
	test_matmul_bf16_identity();
	TEST_REPORT();
}
