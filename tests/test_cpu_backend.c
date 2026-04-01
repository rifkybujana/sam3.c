/*
 * tests/test_cpu_backend.c - CPU backend tensor allocation tests
 *
 * Tests for cpu_alloc_tensor: arena-based allocation, dtype-specific
 * sizing, stride computation, OOM handling, and input validation.
 *
 * Key types:  sam3_cpu_backend, sam3_tensor
 * Depends on: test_helpers.h, backend/cpu/cpu_backend.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <string.h>

#include "test_helpers.h"
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

/* Helper: create a tensor descriptor (no data yet). */
static struct sam3_tensor make_tensor_desc(enum sam3_dtype dtype,
					   int n_dims, const int *dims)
{
	struct sam3_tensor t;
	memset(&t, 0, sizeof(t));
	t.dtype = dtype;
	t.n_dims = n_dims;
	for (int i = 0; i < n_dims; i++)
		t.dims[i] = dims[i];
	return t;
}

static void test_cpu_init_free(void)
{
	struct sam3_cpu_backend cpu = make_cpu_backend(4096);
	ASSERT(cpu.base.ops != NULL);
	cpu.base.ops->free(&cpu.base);
}

static void test_cpu_alloc_basic_f32(void)
{
	struct sam3_cpu_backend cpu = make_cpu_backend(1024 * 1024);
	int dims[] = {3, 224, 224};
	struct sam3_tensor t = make_tensor_desc(SAM3_DTYPE_F32, 3, dims);
	enum sam3_error err;

	err = cpu.base.ops->alloc_tensor(&cpu.base, &t);

	ASSERT_EQ(err, SAM3_OK);
	ASSERT(t.data != NULL);
	ASSERT_EQ(t.nbytes, (size_t)(3 * 224 * 224 * 4));

	/* Strides should be row-major */
	ASSERT_EQ(t.strides[2], 1);
	ASSERT_EQ(t.strides[1], 224);
	ASSERT_EQ(t.strides[0], 224 * 224);

	cpu.base.ops->free(&cpu.base);
}

static void test_cpu_alloc_1d(void)
{
	struct sam3_cpu_backend cpu = make_cpu_backend(4096);
	int dims[] = {256};
	struct sam3_tensor t = make_tensor_desc(SAM3_DTYPE_F32, 1, dims);
	enum sam3_error err;

	err = cpu.base.ops->alloc_tensor(&cpu.base, &t);

	ASSERT_EQ(err, SAM3_OK);
	ASSERT(t.data != NULL);
	ASSERT_EQ(t.nbytes, (size_t)(256 * 4));
	ASSERT_EQ(t.strides[0], 1);

	cpu.base.ops->free(&cpu.base);
}

static void test_cpu_alloc_dtypes(void)
{
	struct sam3_cpu_backend cpu = make_cpu_backend(1024 * 1024);
	int dims[] = {10, 10};

	/* F32: 10*10*4 = 400 bytes */
	struct sam3_tensor f32 = make_tensor_desc(SAM3_DTYPE_F32, 2, dims);
	ASSERT_EQ(cpu.base.ops->alloc_tensor(&cpu.base, &f32), SAM3_OK);
	ASSERT_EQ(f32.nbytes, (size_t)400);

	/* F16: 10*10*2 = 200 bytes */
	struct sam3_tensor f16 = make_tensor_desc(SAM3_DTYPE_F16, 2, dims);
	ASSERT_EQ(cpu.base.ops->alloc_tensor(&cpu.base, &f16), SAM3_OK);
	ASSERT_EQ(f16.nbytes, (size_t)200);

	/* BF16: 10*10*2 = 200 bytes */
	struct sam3_tensor bf16 = make_tensor_desc(SAM3_DTYPE_BF16, 2, dims);
	ASSERT_EQ(cpu.base.ops->alloc_tensor(&cpu.base, &bf16), SAM3_OK);
	ASSERT_EQ(bf16.nbytes, (size_t)200);

	/* I32: 10*10*4 = 400 bytes */
	struct sam3_tensor i32 = make_tensor_desc(SAM3_DTYPE_I32, 2, dims);
	ASSERT_EQ(cpu.base.ops->alloc_tensor(&cpu.base, &i32), SAM3_OK);
	ASSERT_EQ(i32.nbytes, (size_t)400);

	/* I8: 10*10*1 = 100 bytes */
	struct sam3_tensor i8 = make_tensor_desc(SAM3_DTYPE_I8, 2, dims);
	ASSERT_EQ(cpu.base.ops->alloc_tensor(&cpu.base, &i8), SAM3_OK);
	ASSERT_EQ(i8.nbytes, (size_t)100);

	cpu.base.ops->free(&cpu.base);
}

static void test_cpu_alloc_multi_distinct(void)
{
	struct sam3_cpu_backend cpu = make_cpu_backend(1024 * 1024);
	int dims[] = {64, 64};
	struct sam3_tensor a = make_tensor_desc(SAM3_DTYPE_F32, 2, dims);
	struct sam3_tensor b = make_tensor_desc(SAM3_DTYPE_F32, 2, dims);

	ASSERT_EQ(cpu.base.ops->alloc_tensor(&cpu.base, &a), SAM3_OK);
	ASSERT_EQ(cpu.base.ops->alloc_tensor(&cpu.base, &b), SAM3_OK);

	/* Both have data */
	ASSERT(a.data != NULL);
	ASSERT(b.data != NULL);

	/* Data pointers must be distinct */
	ASSERT(a.data != b.data);

	cpu.base.ops->free(&cpu.base);
}

static void test_cpu_alloc_oom(void)
{
	/* Tiny arena: 256 bytes */
	struct sam3_cpu_backend cpu = make_cpu_backend(256);
	int dims[] = {1024};
	struct sam3_tensor t = make_tensor_desc(SAM3_DTYPE_F32, 1, dims);
	enum sam3_error err;

	/* 1024 * 4 = 4096 bytes, exceeds 256 byte arena */
	err = cpu.base.ops->alloc_tensor(&cpu.base, &t);

	ASSERT_EQ(err, SAM3_ENOMEM);
	ASSERT(t.data == NULL);

	cpu.base.ops->free(&cpu.base);
}

static void test_cpu_alloc_null_tensor(void)
{
	struct sam3_cpu_backend cpu = make_cpu_backend(4096);
	enum sam3_error err;

	err = cpu.base.ops->alloc_tensor(&cpu.base, NULL);
	ASSERT_EQ(err, SAM3_EINVAL);

	cpu.base.ops->free(&cpu.base);
}

static void test_cpu_alloc_zero_dims(void)
{
	struct sam3_cpu_backend cpu = make_cpu_backend(4096);
	struct sam3_tensor t;
	memset(&t, 0, sizeof(t));
	t.dtype = SAM3_DTYPE_F32;
	t.n_dims = 0;
	enum sam3_error err;

	err = cpu.base.ops->alloc_tensor(&cpu.base, &t);
	ASSERT_EQ(err, SAM3_EINVAL);

	cpu.base.ops->free(&cpu.base);
}

static void test_cpu_alloc_too_many_dims(void)
{
	struct sam3_cpu_backend cpu = make_cpu_backend(4096);
	struct sam3_tensor t;
	memset(&t, 0, sizeof(t));
	t.dtype = SAM3_DTYPE_F32;
	t.n_dims = SAM3_MAX_DIMS + 1;
	enum sam3_error err;

	err = cpu.base.ops->alloc_tensor(&cpu.base, &t);
	ASSERT_EQ(err, SAM3_EINVAL);

	cpu.base.ops->free(&cpu.base);
}

static void test_cpu_alloc_data_is_zeroed(void)
{
	struct sam3_cpu_backend cpu = make_cpu_backend(4096);
	int dims[] = {16};
	struct sam3_tensor t = make_tensor_desc(SAM3_DTYPE_F32, 1, dims);

	ASSERT_EQ(cpu.base.ops->alloc_tensor(&cpu.base, &t), SAM3_OK);

	/* Arena allocator zeroes memory */
	float *data = (float *)t.data;
	for (int i = 0; i < 16; i++)
		ASSERT(data[i] == 0.0f);

	cpu.base.ops->free(&cpu.base);
}

static void test_cpu_alloc_4d_strides(void)
{
	struct sam3_cpu_backend cpu = make_cpu_backend(1024 * 1024);
	int dims[] = {2, 3, 4, 5};
	struct sam3_tensor t = make_tensor_desc(SAM3_DTYPE_F32, 4, dims);

	ASSERT_EQ(cpu.base.ops->alloc_tensor(&cpu.base, &t), SAM3_OK);

	/* Row-major strides: [3*4*5, 4*5, 5, 1] = [60, 20, 5, 1] */
	ASSERT_EQ(t.strides[3], 1);
	ASSERT_EQ(t.strides[2], 5);
	ASSERT_EQ(t.strides[1], 20);
	ASSERT_EQ(t.strides[0], 60);

	ASSERT_EQ(t.nbytes, (size_t)(2 * 3 * 4 * 5 * 4));

	cpu.base.ops->free(&cpu.base);
}

static void test_cpu_alloc_default_capacity(void)
{
	/* arena_capacity = 0 should use a sensible default */
	struct sam3_cpu_backend cpu = make_cpu_backend(0);
	int dims[] = {64, 64};
	struct sam3_tensor t = make_tensor_desc(SAM3_DTYPE_F32, 2, dims);

	ASSERT_EQ(cpu.base.ops->alloc_tensor(&cpu.base, &t), SAM3_OK);
	ASSERT(t.data != NULL);

	cpu.base.ops->free(&cpu.base);
}

int main(void)
{
	test_cpu_init_free();
	test_cpu_alloc_basic_f32();
	test_cpu_alloc_1d();
	test_cpu_alloc_dtypes();
	test_cpu_alloc_multi_distinct();
	test_cpu_alloc_oom();
	test_cpu_alloc_null_tensor();
	test_cpu_alloc_zero_dims();
	test_cpu_alloc_too_many_dims();
	test_cpu_alloc_data_is_zeroed();
	test_cpu_alloc_4d_strides();
	test_cpu_alloc_default_capacity();

	TEST_REPORT();
}
