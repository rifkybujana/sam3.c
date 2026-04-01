/*
 * tests/bench_dtype.c - Dtype performance benchmark
 *
 * Benchmarks matmul and elementwise add at multiple sizes for f32, fp16,
 * and bf16 dtypes. Reports GFLOPS and throughput. Not included in CTest;
 * run manually via ./bench_dtype.
 *
 * Key types:  sam3_node, sam3_tensor
 * Depends on: core/half.h, core/tensor.h, core/graph.h,
 *             backend/cpu/kernels/cpu_kernels.h, util/threadpool.h
 * Used by:    manual benchmarking
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "core/half.h"
#include "core/tensor.h"
#include "core/graph.h"
#include "backend/cpu/kernels/cpu_kernels.h"
#include "util/threadpool.h"

/* ── Configuration ─────────────────────────────────────────────────── */

#define WARMUP_ITERS   3
#define TIMED_ITERS   10

static const int matmul_sizes[][3] = {
	{   64,   64,   64 },
	{  256,  256,  256 },
	{ 1024, 1024, 1024 },
};
#define N_MATMUL_SIZES (int)(sizeof(matmul_sizes) / sizeof(matmul_sizes[0]))

static const int add_sizes[] = {
	  64 * 64,
	 256 * 256,
	1024 * 1024,
};
#define N_ADD_SIZES (int)(sizeof(add_sizes) / sizeof(add_sizes[0]))

/* ── Timing helper ─────────────────────────────────────────────────── */

static double get_time_ms(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ── Random fill helpers ───────────────────────────────────────────── */

static void fill_random_f32(float *buf, int n)
{
	for (int i = 0; i < n; i++)
		buf[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

static void fill_random_f16(uint16_t *buf, int n)
{
	for (int i = 0; i < n; i++) {
		float v = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
		buf[i] = f32_to_fp16(v);
	}
}

static void fill_random_bf16(uint16_t *buf, int n)
{
	for (int i = 0; i < n; i++) {
		float v = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
		buf[i] = f32_to_bf16(v);
	}
}

/* ── Tensor setup helpers ──────────────────────────────────────────── */

static void make_tensor_2d(struct sam3_tensor *t, void *data,
			   enum sam3_dtype dtype, int rows, int cols)
{
	memset(t, 0, sizeof(*t));
	t->dtype      = dtype;
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
	t->nbytes     = (size_t)rows * cols * sam3_dtype_size(dtype);
}

static void make_tensor_1d(struct sam3_tensor *t, void *data,
			   enum sam3_dtype dtype, int n)
{
	memset(t, 0, sizeof(*t));
	t->dtype      = dtype;
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
	t->nbytes     = (size_t)n * sam3_dtype_size(dtype);
}

static void make_node(struct sam3_node *node, enum sam3_op op,
		      struct sam3_tensor *a, struct sam3_tensor *b,
		      struct sam3_tensor *out)
{
	memset(node, 0, sizeof(*node));
	node->op        = op;
	node->n_inputs  = 2;
	node->inputs[0] = a;
	node->inputs[1] = b;
	node->output    = out;
}

/* ── Matmul benchmarks ─────────────────────────────────────────────── */

static double bench_matmul_f32(int m, int k, int n,
			       struct sam3_threadpool *pool)
{
	float *a = malloc((size_t)m * k * sizeof(float));
	float *b = malloc((size_t)k * n * sizeof(float));
	float *c = malloc((size_t)m * n * sizeof(float));

	if (!a || !b || !c) {
		fprintf(stderr, "bench_matmul_f32: alloc failed\n");
		free(a); free(b); free(c);
		return 0.0;
	}

	fill_random_f32(a, m * k);
	fill_random_f32(b, k * n);

	struct sam3_tensor ta, tb, tc;
	make_tensor_2d(&ta, a, SAM3_DTYPE_F32, m, k);
	make_tensor_2d(&tb, b, SAM3_DTYPE_F32, k, n);
	make_tensor_2d(&tc, c, SAM3_DTYPE_F32, m, n);

	struct sam3_node node;
	make_node(&node, SAM3_OP_MATMUL, &ta, &tb, &tc);

	/* Warmup */
	for (int i = 0; i < WARMUP_ITERS; i++)
		cpu_kernel_matmul(&node, pool);

	/* Timed */
	double t0 = get_time_ms();
	for (int i = 0; i < TIMED_ITERS; i++)
		cpu_kernel_matmul(&node, pool);
	double t1 = get_time_ms();

	free(a); free(b); free(c);

	double avg_ms = (t1 - t0) / TIMED_ITERS;
	double flops  = 2.0 * m * k * n;
	return (flops / (avg_ms / 1000.0)) / 1e9; /* GFLOPS */
}

static double bench_matmul_f16(int m, int k, int n,
			       struct sam3_threadpool *pool)
{
	uint16_t *a = malloc((size_t)m * k * sizeof(uint16_t));
	uint16_t *b = malloc((size_t)k * n * sizeof(uint16_t));
	uint16_t *c = malloc((size_t)m * n * sizeof(uint16_t));

	if (!a || !b || !c) {
		fprintf(stderr, "bench_matmul_f16: alloc failed\n");
		free(a); free(b); free(c);
		return 0.0;
	}

	fill_random_f16(a, m * k);
	fill_random_f16(b, k * n);

	struct sam3_tensor ta, tb, tc;
	make_tensor_2d(&ta, a, SAM3_DTYPE_F16, m, k);
	make_tensor_2d(&tb, b, SAM3_DTYPE_F16, k, n);
	make_tensor_2d(&tc, c, SAM3_DTYPE_F16, m, n);

	struct sam3_node node;
	make_node(&node, SAM3_OP_MATMUL, &ta, &tb, &tc);

	for (int i = 0; i < WARMUP_ITERS; i++)
		cpu_kernel_matmul_f16(&node, pool);

	double t0 = get_time_ms();
	for (int i = 0; i < TIMED_ITERS; i++)
		cpu_kernel_matmul_f16(&node, pool);
	double t1 = get_time_ms();

	free(a); free(b); free(c);

	double avg_ms = (t1 - t0) / TIMED_ITERS;
	double flops  = 2.0 * m * k * n;
	return (flops / (avg_ms / 1000.0)) / 1e9;
}

static double bench_matmul_bf16(int m, int k, int n,
				struct sam3_threadpool *pool)
{
	uint16_t *a = malloc((size_t)m * k * sizeof(uint16_t));
	uint16_t *b = malloc((size_t)k * n * sizeof(uint16_t));
	uint16_t *c = malloc((size_t)m * n * sizeof(uint16_t));

	if (!a || !b || !c) {
		fprintf(stderr, "bench_matmul_bf16: alloc failed\n");
		free(a); free(b); free(c);
		return 0.0;
	}

	fill_random_bf16(a, m * k);
	fill_random_bf16(b, k * n);

	struct sam3_tensor ta, tb, tc;
	make_tensor_2d(&ta, a, SAM3_DTYPE_BF16, m, k);
	make_tensor_2d(&tb, b, SAM3_DTYPE_BF16, k, n);
	make_tensor_2d(&tc, c, SAM3_DTYPE_BF16, m, n);

	struct sam3_node node;
	make_node(&node, SAM3_OP_MATMUL, &ta, &tb, &tc);

	for (int i = 0; i < WARMUP_ITERS; i++)
		cpu_kernel_matmul_bf16(&node, pool);

	double t0 = get_time_ms();
	for (int i = 0; i < TIMED_ITERS; i++)
		cpu_kernel_matmul_bf16(&node, pool);
	double t1 = get_time_ms();

	free(a); free(b); free(c);

	double avg_ms = (t1 - t0) / TIMED_ITERS;
	double flops  = 2.0 * m * k * n;
	return (flops / (avg_ms / 1000.0)) / 1e9;
}

/* ── Add benchmarks ────────────────────────────────────────────────── */

static double bench_add_f32(int n, struct sam3_threadpool *pool)
{
	float *a = malloc((size_t)n * sizeof(float));
	float *b = malloc((size_t)n * sizeof(float));
	float *c = malloc((size_t)n * sizeof(float));

	if (!a || !b || !c) {
		fprintf(stderr, "bench_add_f32: alloc failed\n");
		free(a); free(b); free(c);
		return 0.0;
	}

	fill_random_f32(a, n);
	fill_random_f32(b, n);

	struct sam3_tensor ta, tb, tc;
	make_tensor_1d(&ta, a, SAM3_DTYPE_F32, n);
	make_tensor_1d(&tb, b, SAM3_DTYPE_F32, n);
	make_tensor_1d(&tc, c, SAM3_DTYPE_F32, n);

	struct sam3_node node;
	make_node(&node, SAM3_OP_ADD, &ta, &tb, &tc);

	for (int i = 0; i < WARMUP_ITERS; i++)
		cpu_kernel_add(&node, pool);

	double t0 = get_time_ms();
	for (int i = 0; i < TIMED_ITERS; i++)
		cpu_kernel_add(&node, pool);
	double t1 = get_time_ms();

	free(a); free(b); free(c);

	double avg_ms = (t1 - t0) / TIMED_ITERS;
	/* Bandwidth: read 2 arrays + write 1 array, each n * sizeof(float) */
	double bytes  = 3.0 * n * sizeof(float);
	return (bytes / (avg_ms / 1000.0)) / 1e9; /* GB/s */
}

static double bench_add_f16(int n, struct sam3_threadpool *pool)
{
	uint16_t *a = malloc((size_t)n * sizeof(uint16_t));
	uint16_t *b = malloc((size_t)n * sizeof(uint16_t));
	uint16_t *c = malloc((size_t)n * sizeof(uint16_t));

	if (!a || !b || !c) {
		fprintf(stderr, "bench_add_f16: alloc failed\n");
		free(a); free(b); free(c);
		return 0.0;
	}

	fill_random_f16(a, n);
	fill_random_f16(b, n);

	struct sam3_tensor ta, tb, tc;
	make_tensor_1d(&ta, a, SAM3_DTYPE_F16, n);
	make_tensor_1d(&tb, b, SAM3_DTYPE_F16, n);
	make_tensor_1d(&tc, c, SAM3_DTYPE_F16, n);

	struct sam3_node node;
	make_node(&node, SAM3_OP_ADD, &ta, &tb, &tc);

	for (int i = 0; i < WARMUP_ITERS; i++)
		cpu_kernel_add_f16(&node, pool);

	double t0 = get_time_ms();
	for (int i = 0; i < TIMED_ITERS; i++)
		cpu_kernel_add_f16(&node, pool);
	double t1 = get_time_ms();

	free(a); free(b); free(c);

	double avg_ms = (t1 - t0) / TIMED_ITERS;
	double bytes  = 3.0 * n * sizeof(uint16_t);
	return (bytes / (avg_ms / 1000.0)) / 1e9;
}

static double bench_add_bf16(int n, struct sam3_threadpool *pool)
{
	uint16_t *a = malloc((size_t)n * sizeof(uint16_t));
	uint16_t *b = malloc((size_t)n * sizeof(uint16_t));
	uint16_t *c = malloc((size_t)n * sizeof(uint16_t));

	if (!a || !b || !c) {
		fprintf(stderr, "bench_add_bf16: alloc failed\n");
		free(a); free(b); free(c);
		return 0.0;
	}

	fill_random_bf16(a, n);
	fill_random_bf16(b, n);

	struct sam3_tensor ta, tb, tc;
	make_tensor_1d(&ta, a, SAM3_DTYPE_BF16, n);
	make_tensor_1d(&tb, b, SAM3_DTYPE_BF16, n);
	make_tensor_1d(&tc, c, SAM3_DTYPE_BF16, n);

	struct sam3_node node;
	make_node(&node, SAM3_OP_ADD, &ta, &tb, &tc);

	for (int i = 0; i < WARMUP_ITERS; i++)
		cpu_kernel_add_bf16(&node, pool);

	double t0 = get_time_ms();
	for (int i = 0; i < TIMED_ITERS; i++)
		cpu_kernel_add_bf16(&node, pool);
	double t1 = get_time_ms();

	free(a); free(b); free(c);

	double avg_ms = (t1 - t0) / TIMED_ITERS;
	double bytes  = 3.0 * n * sizeof(uint16_t);
	return (bytes / (avg_ms / 1000.0)) / 1e9;
}

/* ── Main ──────────────────────────────────────────────────────────── */

int main(void)
{
	struct sam3_threadpool *pool = sam3_threadpool_create(0);
	if (!pool) {
		fprintf(stderr, "Failed to create thread pool\n");
		return 1;
	}

	int n_threads = sam3_threadpool_n_threads(pool);
	srand(42);

	printf("SAM3 Dtype Performance Benchmark\n");
	printf("================================\n");
	printf("Threads: %d | Warmup: %d | Timed: %d\n\n",
	       n_threads, WARMUP_ITERS, TIMED_ITERS);

	/* Matmul benchmarks */
	printf("Matmul (MxKxN):\n");
	printf("  %-14s | %13s | %14s | %14s\n",
	       "Size", "f32 (GFLOPS)", "fp16 (GFLOPS)", "bf16 (GFLOPS)");
	printf("  %-14s-+-%13s-+-%14s-+-%14s\n",
	       "--------------", "-------------",
	       "--------------", "--------------");

	for (int s = 0; s < N_MATMUL_SIZES; s++) {
		int m = matmul_sizes[s][0];
		int k = matmul_sizes[s][1];
		int n = matmul_sizes[s][2];

		double gf_f32  = bench_matmul_f32(m, k, n, pool);
		double gf_f16  = bench_matmul_f16(m, k, n, pool);
		double gf_bf16 = bench_matmul_bf16(m, k, n, pool);

		printf("  %4dx%4dx%-4d | %13.2f | %14.2f | %14.2f\n",
		       m, k, n, gf_f32, gf_f16, gf_bf16);
	}

	printf("\n");

	/* Add benchmarks */
	printf("Add (N elements):\n");
	printf("  %-14s | %13s | %14s | %14s\n",
	       "Size", "f32 (GB/s)", "fp16 (GB/s)", "bf16 (GB/s)");
	printf("  %-14s-+-%13s-+-%14s-+-%14s\n",
	       "--------------", "-------------",
	       "--------------", "--------------");

	for (int s = 0; s < N_ADD_SIZES; s++) {
		int n = add_sizes[s];

		double gb_f32  = bench_add_f32(n, pool);
		double gb_f16  = bench_add_f16(n, pool);
		double gb_bf16 = bench_add_bf16(n, pool);

		printf("  %-14d | %13.2f | %14.2f | %14.2f\n",
		       n, gb_f32, gb_f16, gb_bf16);
	}

	printf("\nDone.\n");

	sam3_threadpool_free(pool);
	return 0;
}
