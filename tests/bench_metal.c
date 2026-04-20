/*
 * tests/bench_metal.c - Metal vs CPU backend benchmark
 *
 * Compares Metal (MLX-C GPU) and CPU backends across matmul, add,
 * softmax, and multi-op graphs at various sizes. Reports wall-clock
 * time, speedup ratios, GFLOPS, and % of theoretical peak. Backends
 * are created once and reused across all benchmarks for accuracy.
 * Not included in CTest; run manually.
 *
 * Key types:  sam3_backend, sam3_graph, sam3_node
 * Depends on: backend/backend.h, core/graph.h, core/tensor.h
 * Used by:    manual benchmarking
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "backend/backend.h"
#include "core/graph.h"
#include "core/tensor.h"
#include "core/half.h"

/* --- Configuration ─ --- */

#define WARMUP_ITERS   5
#define TIMED_ITERS   50

/* Theoretical peak GFLOPS for Apple Silicon (M-series GPU). */
#define PEAK_F32_GFLOPS  3400.0
#define PEAK_F16_GFLOPS  6800.0

/* --- Timing helper ─ --- */

static double get_time_ms(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* --- Random fill  --- */

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

/* --- Tensor setup  --- */

static struct sam3_tensor make_tensor_2d(enum sam3_dtype dtype,
					 int rows, int cols)
{
	struct sam3_tensor t;
	memset(&t, 0, sizeof(t));
	t.dtype      = dtype;
	t.n_dims     = 2;
	t.dims[0]    = rows;
	t.dims[1]    = cols;
	t.dims[2]    = 1;
	t.dims[3]    = 1;
	t.strides[0] = cols;
	t.strides[1] = 1;
	t.strides[2] = rows * cols;
	t.strides[3] = rows * cols;
	t.nbytes     = (size_t)rows * cols * sam3_dtype_size(dtype);
	return t;
}

/* --- Benchmark: matmul F32  --- */

static double bench_matmul(struct sam3_backend *be, int m, int k, int n)
{
	struct sam3_tensor a = make_tensor_2d(SAM3_DTYPE_F32, m, k);
	struct sam3_tensor b = make_tensor_2d(SAM3_DTYPE_F32, k, n);
	struct sam3_tensor c = make_tensor_2d(SAM3_DTYPE_F32, m, n);

	a.data = malloc(a.nbytes);
	b.data = malloc(b.nbytes);
	c.data = malloc(c.nbytes);
	if (!a.data || !b.data || !c.data) {
		free(a.data); free(b.data); free(c.data);
		return -1.0;
	}
	fill_random_f32(a.data, m * k);
	fill_random_f32(b.data, k * n);

	struct sam3_graph g;
	sam3_graph_init(&g);
	g.nodes[0] = (struct sam3_node){
		.op = SAM3_OP_MATMUL, .n_inputs = 2,
		.inputs = {&a, &b}, .output = &c,
	};
	g.n_nodes = 1;

	for (int i = 0; i < WARMUP_ITERS; i++)
		be->ops->graph_eval(be, &g);

	double t0 = get_time_ms();
	for (int i = 0; i < TIMED_ITERS; i++)
		be->ops->graph_eval(be, &g);
	double elapsed = (get_time_ms() - t0) / TIMED_ITERS;

	free(a.data);
	free(b.data);
	free(c.data);
	return elapsed;
}

/* --- Benchmark: matmul F16  --- */

static double bench_matmul_f16(struct sam3_backend *be, int m, int k, int n)
{
	struct sam3_tensor a = make_tensor_2d(SAM3_DTYPE_F16, m, k);
	struct sam3_tensor b = make_tensor_2d(SAM3_DTYPE_F16, k, n);
	struct sam3_tensor c = make_tensor_2d(SAM3_DTYPE_F16, m, n);

	a.data = malloc(a.nbytes);
	b.data = malloc(b.nbytes);
	c.data = malloc(c.nbytes);
	if (!a.data || !b.data || !c.data) {
		free(a.data); free(b.data); free(c.data);
		return -1.0;
	}
	fill_random_f16(a.data, m * k);
	fill_random_f16(b.data, k * n);

	struct sam3_graph g;
	sam3_graph_init(&g);
	g.nodes[0] = (struct sam3_node){
		.op = SAM3_OP_MATMUL, .n_inputs = 2,
		.inputs = {&a, &b}, .output = &c,
	};
	g.n_nodes = 1;

	for (int i = 0; i < WARMUP_ITERS; i++)
		be->ops->graph_eval(be, &g);

	double t0 = get_time_ms();
	for (int i = 0; i < TIMED_ITERS; i++)
		be->ops->graph_eval(be, &g);
	double elapsed = (get_time_ms() - t0) / TIMED_ITERS;

	free(a.data);
	free(b.data);
	free(c.data);
	return elapsed;
}

/* --- Benchmark: elementwise add  --- */

static double bench_add(struct sam3_backend *be, int rows, int cols)
{
	struct sam3_tensor a = make_tensor_2d(SAM3_DTYPE_F32, rows, cols);
	struct sam3_tensor b = make_tensor_2d(SAM3_DTYPE_F32, rows, cols);
	struct sam3_tensor c = make_tensor_2d(SAM3_DTYPE_F32, rows, cols);

	a.data = malloc(a.nbytes);
	b.data = malloc(b.nbytes);
	c.data = malloc(c.nbytes);
	if (!a.data || !b.data || !c.data) {
		free(a.data); free(b.data); free(c.data);
		return -1.0;
	}
	fill_random_f32(a.data, rows * cols);
	fill_random_f32(b.data, rows * cols);

	struct sam3_graph g;
	sam3_graph_init(&g);
	g.nodes[0] = (struct sam3_node){
		.op = SAM3_OP_ADD, .n_inputs = 2,
		.inputs = {&a, &b}, .output = &c,
	};
	g.n_nodes = 1;

	for (int i = 0; i < WARMUP_ITERS; i++)
		be->ops->graph_eval(be, &g);

	double t0 = get_time_ms();
	for (int i = 0; i < TIMED_ITERS; i++)
		be->ops->graph_eval(be, &g);
	double elapsed = (get_time_ms() - t0) / TIMED_ITERS;

	free(a.data);
	free(b.data);
	free(c.data);
	return elapsed;
}

/* --- Benchmark: softmax  --- */

static double bench_softmax(struct sam3_backend *be, int rows, int cols)
{
	struct sam3_tensor a = make_tensor_2d(SAM3_DTYPE_F32, rows, cols);
	struct sam3_tensor c = make_tensor_2d(SAM3_DTYPE_F32, rows, cols);

	a.data = malloc(a.nbytes);
	c.data = malloc(c.nbytes);
	if (!a.data || !c.data) {
		free(a.data); free(c.data);
		return -1.0;
	}
	fill_random_f32(a.data, rows * cols);

	struct sam3_graph g;
	sam3_graph_init(&g);
	g.nodes[0] = (struct sam3_node){
		.op = SAM3_OP_SOFTMAX, .n_inputs = 1,
		.inputs = {&a}, .output = &c,
	};
	g.n_nodes = 1;

	for (int i = 0; i < WARMUP_ITERS; i++)
		be->ops->graph_eval(be, &g);

	double t0 = get_time_ms();
	for (int i = 0; i < TIMED_ITERS; i++)
		be->ops->graph_eval(be, &g);
	double elapsed = (get_time_ms() - t0) / TIMED_ITERS;

	free(a.data);
	free(c.data);
	return elapsed;
}

/* --- Benchmark: matmul + add + softmax pipeline --- */

static double bench_pipeline(struct sam3_backend *be,
			     int m, int k, int n)
{
	struct sam3_tensor a    = make_tensor_2d(SAM3_DTYPE_F32, m, k);
	struct sam3_tensor b    = make_tensor_2d(SAM3_DTYPE_F32, k, n);
	struct sam3_tensor mm   = make_tensor_2d(SAM3_DTYPE_F32, m, n);
	struct sam3_tensor bias = make_tensor_2d(SAM3_DTYPE_F32, m, n);
	struct sam3_tensor add  = make_tensor_2d(SAM3_DTYPE_F32, m, n);
	struct sam3_tensor out  = make_tensor_2d(SAM3_DTYPE_F32, m, n);

	a.data    = malloc(a.nbytes);
	b.data    = malloc(b.nbytes);
	mm.data   = malloc(mm.nbytes);
	bias.data = malloc(bias.nbytes);
	add.data  = malloc(add.nbytes);
	out.data  = malloc(out.nbytes);
	if (!a.data || !b.data || !mm.data ||
	    !bias.data || !add.data || !out.data) {
		free(a.data); free(b.data); free(mm.data);
		free(bias.data); free(add.data); free(out.data);
		return -1.0;
	}
	fill_random_f32(a.data, m * k);
	fill_random_f32(b.data, k * n);
	fill_random_f32(bias.data, m * n);

	struct sam3_graph g;
	sam3_graph_init(&g);
	g.nodes[0] = (struct sam3_node){
		.op = SAM3_OP_MATMUL, .n_inputs = 2,
		.inputs = {&a, &b}, .output = &mm,
	};
	g.nodes[1] = (struct sam3_node){
		.op = SAM3_OP_ADD, .n_inputs = 2,
		.inputs = {&mm, &bias}, .output = &add,
	};
	g.nodes[2] = (struct sam3_node){
		.op = SAM3_OP_SOFTMAX, .n_inputs = 1,
		.inputs = {&add}, .output = &out,
	};
	g.n_nodes = 3;

	for (int i = 0; i < WARMUP_ITERS; i++)
		be->ops->graph_eval(be, &g);

	double t0 = get_time_ms();
	for (int i = 0; i < TIMED_ITERS; i++)
		be->ops->graph_eval(be, &g);
	double elapsed = (get_time_ms() - t0) / TIMED_ITERS;

	free(a.data);
	free(b.data);
	free(mm.data);
	free(bias.data);
	free(add.data);
	free(out.data);
	return elapsed;
}

/* --- Reporting  --- */

static void print_row(const char *label, double cpu_ms, double metal_ms)
{
	if (metal_ms < 0) {
		printf("  %-28s %13s  %13s  %9s\n",
		       label, "---", "FAILED", "---");
		return;
	}
	if (cpu_ms < 0) {
		printf("  %-28s %13s  %10.3f ms  %9s\n",
		       label, "---", metal_ms, "---");
		return;
	}
	double speedup = cpu_ms / metal_ms;
	printf("  %-28s %10.3f ms  %10.3f ms  %8.2fx\n",
	       label, cpu_ms, metal_ms, speedup);
}

static void print_header(void)
{
	printf("  %-28s %13s  %13s  %9s\n",
	       "Operation", "CPU", "Metal", "Speedup");
	printf("  %-28s %13s  %13s  %9s\n",
	       "----------------------------",
	       "-------------", "-------------", "---------");
}

/* --- Main  --- */

int main(void)
{
	struct sam3_backend *cpu = sam3_backend_init(SAM3_BACKEND_CPU);
	struct sam3_backend *metal = sam3_backend_init(SAM3_BACKEND_METAL);
	if (!cpu) {
		fprintf(stderr, "Failed to init CPU backend\n");
		return 1;
	}
	if (!metal) {
		fprintf(stderr, "Failed to init Metal backend\n");
		sam3_backend_free(cpu);
		return 1;
	}

	srand(42);
	double peak_f32 = 0, peak_f16 = 0;

	printf("\n");
	printf("========================================"
	       "========================================\n");
	printf("            Metal vs CPU Backend Benchmark\n");
	printf("            Warmup: %d iters, Timed: %d iters"
	       "\n", WARMUP_ITERS, TIMED_ITERS);
	printf("========================================"
	       "========================================\n\n");

	/* --- Matmul F32  --- */
	printf("  MATMUL F32 (M x K x N)\n");
	print_header();

	static const int mm_sizes[][3] = {
		{   64,   64,   64 },
		{  256,  256,  256 },
		{  512,  512,  512 },
		{ 1024, 1024, 1024 },
		{ 2048, 2048, 2048 },
		{ 4096, 4096, 4096 },
	};
	int n_mm = (int)(sizeof(mm_sizes) / sizeof(mm_sizes[0]));

	for (int i = 0; i < n_mm; i++) {
		int m = mm_sizes[i][0], k = mm_sizes[i][1];
		int n = mm_sizes[i][2];
		char label[64];
		snprintf(label, sizeof(label), "%d x %d x %d", m, k, n);

		/* Skip CPU for large sizes (too slow, not interesting) */
		double tc = (m >= 4096) ? -1.0 : bench_matmul(cpu, m, k, n);
		double tm = bench_matmul(metal, m, k, n);
		print_row(label, tc, tm);

		if (tm > 0) {
			double gf_cpu = (tc > 0)
				? (2.0 * m * k * n) / (tc * 1e6) : 0;
			double gf_metal = (2.0 * m * k * n) / (tm * 1e6);
			if (tc > 0)
				printf("  %-28s %10.1f GFLOPS  %7.1f GFLOPS"
				       "  (%4.1f%%)\n",
				       "", gf_cpu, gf_metal,
				       gf_metal / PEAK_F32_GFLOPS * 100.0);
			else
				printf("  %-28s %10s         %7.1f GFLOPS"
				       "  (%4.1f%%)\n",
				       "", "---", gf_metal,
				       gf_metal / PEAK_F32_GFLOPS * 100.0);
			if (gf_metal > peak_f32)
				peak_f32 = gf_metal;
		}
	}

	/* --- Matmul F16  --- */
	printf("\n  MATMUL F16 (M x K x N)\n");
	print_header();

	for (int i = 0; i < n_mm; i++) {
		int m = mm_sizes[i][0], k = mm_sizes[i][1];
		int n = mm_sizes[i][2];
		char label[64];
		snprintf(label, sizeof(label), "%d x %d x %d", m, k, n);

		double tc = (m >= 4096) ? -1.0
			: bench_matmul_f16(cpu, m, k, n);
		double tm = bench_matmul_f16(metal, m, k, n);
		print_row(label, tc, tm);

		if (tm > 0) {
			double gf_cpu = (tc > 0)
				? (2.0 * m * k * n) / (tc * 1e6) : 0;
			double gf_metal = (2.0 * m * k * n) / (tm * 1e6);
			if (tc > 0)
				printf("  %-28s %10.1f GFLOPS  %7.1f GFLOPS"
				       "  (%4.1f%%)\n",
				       "", gf_cpu, gf_metal,
				       gf_metal / PEAK_F16_GFLOPS * 100.0);
			else
				printf("  %-28s %10s         %7.1f GFLOPS"
				       "  (%4.1f%%)\n",
				       "", "---", gf_metal,
				       gf_metal / PEAK_F16_GFLOPS * 100.0);
			if (gf_metal > peak_f16)
				peak_f16 = gf_metal;
		}
	}

	/* --- Add  --- */
	printf("\n  ELEMENTWISE ADD (rows x cols)\n");
	print_header();

	static const int add_sizes[][2] = {
		{  256,  256 },
		{ 1024, 1024 },
		{ 2048, 2048 },
		{ 4096, 4096 },
	};
	int n_add = (int)(sizeof(add_sizes) / sizeof(add_sizes[0]));

	for (int i = 0; i < n_add; i++) {
		int rows = add_sizes[i][0], cols = add_sizes[i][1];
		char label[64];
		snprintf(label, sizeof(label), "%d x %d", rows, cols);

		double tc = bench_add(cpu, rows, cols);
		double tm = bench_add(metal, rows, cols);
		print_row(label, tc, tm);
	}

	/* --- Softmax  --- */
	printf("\n  SOFTMAX (rows x cols)\n");
	print_header();

	static const int sm_sizes[][2] = {
		{  256,  256 },
		{ 1024, 1024 },
		{ 2048, 2048 },
		{ 4096, 4096 },
	};
	int n_sm = (int)(sizeof(sm_sizes) / sizeof(sm_sizes[0]));

	for (int i = 0; i < n_sm; i++) {
		int rows = sm_sizes[i][0], cols = sm_sizes[i][1];
		char label[64];
		snprintf(label, sizeof(label), "%d x %d", rows, cols);

		double tc = bench_softmax(cpu, rows, cols);
		double tm = bench_softmax(metal, rows, cols);
		print_row(label, tc, tm);
	}

	/* --- Pipeline  --- */
	printf("\n  PIPELINE: matmul + add + softmax (M x K x N)\n");
	print_header();

	static const int pipe_sizes[][3] = {
		{  256,  256,  256 },
		{  512,  512,  512 },
		{ 1024, 1024, 1024 },
		{ 2048, 2048, 2048 },
	};
	int n_pipe = (int)(sizeof(pipe_sizes) / sizeof(pipe_sizes[0]));

	for (int i = 0; i < n_pipe; i++) {
		int m = pipe_sizes[i][0], k = pipe_sizes[i][1];
		int n = pipe_sizes[i][2];
		char label[64];
		snprintf(label, sizeof(label), "%d x %d x %d", m, k, n);

		double tc = bench_pipeline(cpu, m, k, n);
		double tm = bench_pipeline(metal, m, k, n);
		print_row(label, tc, tm);
	}

	/* --- Peak summary  --- */
	printf("\n  ---- Peak Throughput ----\n");
	printf("  Peak F32: %.1f GFLOPS (%.1f%% of 3.4 TFLOPS)\n",
	       peak_f32, peak_f32 / PEAK_F32_GFLOPS * 100.0);
	printf("  Peak F16: %.1f GFLOPS (%.1f%% of 6.8 TFLOPS)\n",
	       peak_f16, peak_f16 / PEAK_F16_GFLOPS * 100.0);

	printf("\n");

	sam3_backend_free(metal);
	sam3_backend_free(cpu);
	return 0;
}
