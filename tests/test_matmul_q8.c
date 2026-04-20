/*
 * tests/test_matmul_q8.c - Unit tests for mixed-dtype Q8_0 matmul
 *
 * Tests F32 @ Q8_0 -> F32 matrix multiplication against a pure f32
 * reference. Tolerance is ~1e-2 due to int8 quantization error.
 *
 * Key types:  sam3_node, sam3_tensor, sam3_q8_block
 * Depends on: core/quant.h, core/tensor.h, core/graph.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "core/quant.h"
#include "core/tensor.h"
#include "core/graph.h"
#include "backend/cpu/kernels/cpu_kernels.h"
#include "util/threadpool.h"

#include <string.h>
#include <math.h>

/* --- Helpers ────── --- */

static void make_f32_tensor_2d(struct sam3_tensor *t, float *data,
			       int rows, int cols)
{
	t->dtype      = SAM3_DTYPE_F32;
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
	t->nbytes     = (size_t)rows * cols * sizeof(float);
}

static void make_q8_tensor_2d(struct sam3_tensor *t, void *data,
			      int rows, int cols)
{
	t->dtype      = SAM3_DTYPE_Q8_0;
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
	t->nbytes     = sam3_q8_nbytes(rows * cols);
}

/* --- Tests ──────── --- */

/*
 * test_matmul_q8_small - 4x64 @ 64x4 (K=64 = 2 Q8 blocks per row)
 */
static void test_matmul_q8_small(void)
{
	const int M = 4, K = 64, N = 4;
	int i, j, k;

	float a_f32[4 * 64];
	float b_f32[64 * 4];

	for (i = 0; i < M * K; i++)
		a_f32[i] = sinf((float)i * 0.1f) * 0.5f;
	for (i = 0; i < K * N; i++)
		b_f32[i] = cosf((float)i * 0.1f) * 0.5f;

	/* Compute f32 reference C = A @ B */
	float c_ref[4 * 4];
	memset(c_ref, 0, sizeof(c_ref));
	for (i = 0; i < M; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k < K; k++)
				c_ref[i * N + j] +=
					a_f32[i * K + k] * b_f32[k * N + j];

	/* Quantize B to Q8_0 */
	int b_nelems = K * N;
	struct sam3_q8_block b_q8[8]; /* 256/32 = 8 blocks */
	sam3_q8_quantize(b_f32, b_q8, b_nelems);

	/* Set up tensors and node */
	struct sam3_tensor ta, tb, tc;
	float c_data[4 * 4];
	memset(c_data, 0, sizeof(c_data));

	make_f32_tensor_2d(&ta, a_f32, M, K);
	make_q8_tensor_2d(&tb, b_q8, K, N);
	make_f32_tensor_2d(&tc, c_data, M, N);

	struct sam3_node node;
	memset(&node, 0, sizeof(node));
	node.op = SAM3_OP_MATMUL;
	node.inputs[0] = &ta;
	node.inputs[1] = &tb;
	node.n_inputs = 2;
	node.output = &tc;

	enum sam3_error err = cpu_kernel_matmul_q8(&node, NULL);
	ASSERT_EQ(err, SAM3_OK);

	/* Check output against f32 reference */
	for (i = 0; i < M * N; i++)
		ASSERT_NEAR(c_data[i], c_ref[i], 0.5f);
}

/*
 * test_matmul_q8_larger - 16x128 @ 128x32 with thread pool
 */
static void test_matmul_q8_larger(void)
{
	const int M = 16, K = 128, N = 32;
	int i, j, k;

	float *a_f32 = malloc(M * K * sizeof(float));
	float *b_f32 = malloc(K * N * sizeof(float));
	float *c_ref = calloc(M * N, sizeof(float));
	float *c_out = calloc(M * N, sizeof(float));
	ASSERT(a_f32 && b_f32 && c_ref && c_out);

	for (i = 0; i < M * K; i++)
		a_f32[i] = sinf((float)i * 0.01f) * 2.0f;
	for (i = 0; i < K * N; i++)
		b_f32[i] = cosf((float)i * 0.01f) * 2.0f;

	for (i = 0; i < M; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k < K; k++)
				c_ref[i * N + j] +=
					a_f32[i * K + k] * b_f32[k * N + j];

	int b_nelems = K * N;
	struct sam3_q8_block *b_q8 = malloc(sam3_q8_nbytes(b_nelems));
	ASSERT(b_q8);
	sam3_q8_quantize(b_f32, b_q8, b_nelems);

	struct sam3_tensor ta, tb, tc;
	make_f32_tensor_2d(&ta, a_f32, M, K);
	make_q8_tensor_2d(&tb, b_q8, K, N);
	make_f32_tensor_2d(&tc, c_out, M, N);

	struct sam3_node node;
	memset(&node, 0, sizeof(node));
	node.op = SAM3_OP_MATMUL;
	node.inputs[0] = &ta;
	node.inputs[1] = &tb;
	node.n_inputs = 2;
	node.output = &tc;

	struct sam3_threadpool *pool = sam3_threadpool_create(4);
	enum sam3_error err = cpu_kernel_matmul_q8(&node, pool);
	ASSERT_EQ(err, SAM3_OK);
	sam3_threadpool_free(pool);

	for (i = 0; i < M * N; i++)
		ASSERT_NEAR(c_out[i], c_ref[i], 1.0f);

	free(a_f32);
	free(b_f32);
	free(c_ref);
	free(c_out);
	free(b_q8);
}

int main(void)
{
	test_matmul_q8_small();
	test_matmul_q8_larger();
	TEST_REPORT();
}
