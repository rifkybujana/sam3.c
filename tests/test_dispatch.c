/*
 * tests/test_dispatch.c - Unit tests for the CPU dispatch table
 *
 * Exercises cpu_dispatch_node() for dtype mismatch detection, correct
 * dispatch for F32 ops, rejection of unimplemented (op, dtype) pairs,
 * and dtype-agnostic reshape across F16 and BF16 tensors.
 *
 * Key types:  sam3_node, sam3_tensor, sam3_cpu_backend
 * Depends on: test_helpers.h, backend/cpu/cpu_dispatch.h,
 *             backend/cpu/cpu_backend.h, core/tensor.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <string.h>

#include "test_helpers.h"
#include "backend/cpu/cpu_dispatch.h"
#include "backend/cpu/cpu_backend.h"
#include "core/tensor.h"

/* Helper: create and init a CPU backend with given arena capacity. */
static struct sam3_cpu_backend make_cpu_backend(size_t arena_capacity)
{
	struct sam3_cpu_backend cpu;
	memset(&cpu, 0, sizeof(cpu));
	cpu.base.type = SAM3_BACKEND_CPU;
	cpu.base.ops = sam3_cpu_backend_ops();
	cpu.arena_capacity = arena_capacity;
	cpu.base.ops->init(&cpu.base);
	return cpu;
}

/* Helper: allocate a tensor from the backend arena. */
static struct sam3_tensor *alloc_tensor(struct sam3_cpu_backend *cpu,
					enum sam3_dtype dtype,
					int n_dims, const int *dims)
{
	static struct sam3_tensor buf[32];
	static int idx = 0;
	struct sam3_tensor *t = &buf[idx++ % 32];
	memset(t, 0, sizeof(*t));
	t->dtype  = dtype;
	t->n_dims = n_dims;
	for (int i = 0; i < n_dims; i++)
		t->dims[i] = dims[i];
	cpu->base.ops->alloc_tensor(&cpu->base, t);
	return t;
}

/* Helper: build a 2-input node. */
static struct sam3_node make_binary_node(enum sam3_op op,
					 struct sam3_tensor *a,
					 struct sam3_tensor *b,
					 struct sam3_tensor *out)
{
	struct sam3_node node;
	memset(&node, 0, sizeof(node));
	node.op         = op;
	node.inputs[0]  = a;
	node.inputs[1]  = b;
	node.n_inputs   = 2;
	node.output     = out;
	return node;
}

/* Helper: build a 1-input node. */
static struct sam3_node make_unary_node(enum sam3_op op,
					struct sam3_tensor *in,
					struct sam3_tensor *out)
{
	struct sam3_node node;
	memset(&node, 0, sizeof(node));
	node.op        = op;
	node.inputs[0] = in;
	node.n_inputs  = 1;
	node.output    = out;
	return node;
}

/* --- Tests ───────── --- */

/*
 * Dispatch must return SAM3_EDTYPE when inputs have different dtypes.
 */
static void test_dispatch_dtype_mismatch(void)
{
	struct sam3_cpu_backend cpu = make_cpu_backend(1024 * 1024);
	int dims[] = {4};
	struct sam3_tensor *a   = alloc_tensor(&cpu, SAM3_DTYPE_F32, 1, dims);
	struct sam3_tensor *b   = alloc_tensor(&cpu, SAM3_DTYPE_F16, 1, dims);
	struct sam3_tensor *out = alloc_tensor(&cpu, SAM3_DTYPE_F32, 1, dims);
	struct sam3_node node   = make_binary_node(SAM3_OP_ADD, a, b, out);
	enum sam3_error err;

	err = cpu_dispatch_node(&node, &cpu.scratch, cpu.pool);
	ASSERT_EQ(err, SAM3_EDTYPE);

	cpu.base.ops->free(&cpu.base);
}

/*
 * F32 add: two tensors of known values, dispatch should produce correct sum.
 */
static void test_dispatch_f32_add(void)
{
	struct sam3_cpu_backend cpu = make_cpu_backend(1024 * 1024);
	int dims[] = {4};
	struct sam3_tensor *a   = alloc_tensor(&cpu, SAM3_DTYPE_F32, 1, dims);
	struct sam3_tensor *b   = alloc_tensor(&cpu, SAM3_DTYPE_F32, 1, dims);
	struct sam3_tensor *out = alloc_tensor(&cpu, SAM3_DTYPE_F32, 1, dims);
	float *pa = (float *)a->data;
	float *pb = (float *)b->data;
	float *po = (float *)out->data;
	struct sam3_node node;
	enum sam3_error err;

	pa[0] = 1.0f; pa[1] = 2.0f; pa[2] = 3.0f; pa[3] = 4.0f;
	pb[0] = 10.0f; pb[1] = 20.0f; pb[2] = 30.0f; pb[3] = 40.0f;

	node = make_binary_node(SAM3_OP_ADD, a, b, out);
	err  = cpu_dispatch_node(&node, &cpu.scratch, cpu.pool);

	ASSERT_EQ(err, SAM3_OK);
	ASSERT_NEAR(po[0], 11.0f, 1e-5f);
	ASSERT_NEAR(po[1], 22.0f, 1e-5f);
	ASSERT_NEAR(po[2], 33.0f, 1e-5f);
	ASSERT_NEAR(po[3], 44.0f, 1e-5f);

	cpu.base.ops->free(&cpu.base);
}

/*
 * F16 conv2d is not yet registered; dispatch must return SAM3_EDTYPE.
 */
static void test_dispatch_unimplemented(void)
{
	struct sam3_cpu_backend cpu = make_cpu_backend(1024 * 1024);
	int dims[] = {4};
	struct sam3_tensor *a   = alloc_tensor(&cpu, SAM3_DTYPE_F16, 1, dims);
	struct sam3_tensor *out = alloc_tensor(&cpu, SAM3_DTYPE_F16, 1, dims);
	struct sam3_node node   = make_unary_node(SAM3_OP_CONV2D, a, out);
	enum sam3_error err;

	err = cpu_dispatch_node(&node, &cpu.scratch, cpu.pool);
	ASSERT_EQ(err, SAM3_EDTYPE);

	cpu.base.ops->free(&cpu.base);
}

/*
 * Reshape is dtype-agnostic; it must succeed for F16 and BF16.
 */
static void test_dispatch_reshape_any_dtype(void)
{
	struct sam3_cpu_backend cpu = make_cpu_backend(1024 * 1024);
	int in_dims[]  = {2, 4};
	int out_dims[] = {8};
	struct sam3_tensor *f16_in, *f16_out;
	struct sam3_tensor *bf16_in, *bf16_out;
	struct sam3_node node;
	enum sam3_error err;

	/* F16 reshape */
	f16_in  = alloc_tensor(&cpu, SAM3_DTYPE_F16,  2, in_dims);
	f16_out = alloc_tensor(&cpu, SAM3_DTYPE_F16,  1, out_dims);
	node = make_unary_node(SAM3_OP_RESHAPE, f16_in, f16_out);
	err  = cpu_dispatch_node(&node, &cpu.scratch, cpu.pool);
	ASSERT_EQ(err, SAM3_OK);
	/* Output must alias input data */
	ASSERT(f16_out->data == f16_in->data);

	/* BF16 reshape */
	bf16_in  = alloc_tensor(&cpu, SAM3_DTYPE_BF16, 2, in_dims);
	bf16_out = alloc_tensor(&cpu, SAM3_DTYPE_BF16, 1, out_dims);
	node = make_unary_node(SAM3_OP_RESHAPE, bf16_in, bf16_out);
	err  = cpu_dispatch_node(&node, &cpu.scratch, cpu.pool);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT(bf16_out->data == bf16_in->data);

	cpu.base.ops->free(&cpu.base);
}

int main(void)
{
	test_dispatch_dtype_mismatch();
	test_dispatch_f32_add();
	test_dispatch_unimplemented();
	test_dispatch_reshape_any_dtype();

	TEST_REPORT();
}
