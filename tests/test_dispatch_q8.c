/*
 * tests/test_dispatch_q8.c - Dispatch integration test for Q8_0 matmul
 *
 * Verifies that the CPU dispatch table correctly routes Q8_0 matmul
 * to the mixed-dtype kernel, relaxing the same-dtype constraint for
 * the F32-activation x Q8_0-weight case.
 *
 * Key types:  sam3_node, sam3_tensor, sam3_q8_block
 * Depends on: core/quant.h, core/tensor.h, core/graph.h,
 *             backend/cpu/cpu_dispatch.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "core/quant.h"
#include "core/tensor.h"
#include "core/graph.h"
#include "backend/cpu/cpu_dispatch.h"
#include "core/alloc.h"
#include "util/threadpool.h"

#include <string.h>
#include <math.h>

static void test_dispatch_matmul_q8(void)
{
	const int M = 4, K = 32, N = 4;

	float a_f32[4 * 32];
	float b_f32[32 * 4];
	float c_data[4 * 4];

	for (int i = 0; i < M * K; i++)
		a_f32[i] = (float)i * 0.01f;
	for (int i = 0; i < K * N; i++)
		b_f32[i] = (float)i * 0.01f;

	/* Quantize B */
	struct sam3_q8_block b_q8[4]; /* 128/32 = 4 blocks */
	sam3_q8_quantize(b_f32, b_q8, K * N);

	struct sam3_tensor ta = {
		.dtype = SAM3_DTYPE_F32, .n_dims = 2,
		.dims = {M, K, 1, 1}, .strides = {K, 1, M*K, M*K},
		.data = a_f32, .nbytes = sizeof(a_f32)
	};
	struct sam3_tensor tb = {
		.dtype = SAM3_DTYPE_Q8_0, .n_dims = 2,
		.dims = {K, N, 1, 1}, .strides = {N, 1, K*N, K*N},
		.data = b_q8, .nbytes = sam3_q8_nbytes(K * N)
	};
	struct sam3_tensor tc = {
		.dtype = SAM3_DTYPE_F32, .n_dims = 2,
		.dims = {M, N, 1, 1}, .strides = {N, 1, M*N, M*N},
		.data = c_data, .nbytes = sizeof(c_data)
	};
	memset(c_data, 0, sizeof(c_data));

	struct sam3_node node;
	memset(&node, 0, sizeof(node));
	node.op = SAM3_OP_MATMUL;
	node.inputs[0] = &ta;
	node.inputs[1] = &tb;
	node.n_inputs = 2;
	node.output = &tc;

	struct sam3_arena scratch;
	memset(&scratch, 0, sizeof(scratch));

	enum sam3_error err = cpu_dispatch_node(&node, &scratch, NULL);
	ASSERT_EQ(err, SAM3_OK);

	/* Verify non-zero output */
	int nonzero = 0;
	for (int i = 0; i < M * N; i++)
		if (c_data[i] != 0.0f) nonzero++;
	ASSERT(nonzero > 0);
}

int main(void)
{
	test_dispatch_matmul_q8();
	TEST_REPORT();
}
