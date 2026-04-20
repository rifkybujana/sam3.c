/*
 * src/bench/bench_kernels.c - Kernel microbenchmark suite implementation
 *
 * Benchmarks individual compute kernels (matmul, elementwise ops, softmax,
 * layernorm, transpose) across a matrix of sizes and dtypes. Each case
 * allocates tensors from a scratch arena, builds a one-op compute graph,
 * and times execution through the backend vtable via the bench harness.
 *
 * Key types:  kernel_bench_ctx, sam3_bench_config, sam3_bench_result
 * Depends on: bench/bench.h, backend/backend.h, core/graph.h,
 *             core/tensor.h, core/alloc.h, core/half.h, util/log.h
 * Used by:    cli_bench.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "bench/bench_kernels.h"
#include "core/graph.h"
#include "core/tensor.h"
#include "core/alloc.h"
#include "core/half.h"
#include "util/log.h"

/* 256 MB scratch arena for tensor data. */
#define SCRATCH_ARENA_SIZE (256 * 1024 * 1024)

/* Context passed to the benchmark callback. */
struct kernel_bench_ctx {
	struct sam3_backend *be;
	struct sam3_graph    graph;
};

/* Benchmark callback: evaluate the single-op graph. */
static void kernel_bench_fn(void *ctx)
{
	struct kernel_bench_ctx *kctx = ctx;
	kctx->be->ops->graph_eval(kctx->be, &kctx->graph);
}

/*
 * init_tensor - Initialize a tensor header and allocate data from the arena.
 *
 * @arena: Scratch arena for data allocation.
 * @t:     Tensor struct to initialize (must already be allocated).
 * @dtype: Data type.
 * @dims:  Dimension array.
 * @n_dims: Number of dimensions.
 *
 * Returns 0 on success, -1 if arena is full.
 */
static int init_tensor(struct sam3_arena *arena, struct sam3_tensor *t,
		       enum sam3_dtype dtype, const int *dims, int n_dims)
{
	memset(t, 0, sizeof(*t));
	t->dtype = dtype;
	t->n_dims = n_dims;
	for (int i = 0; i < n_dims; i++)
		t->dims[i] = dims[i];
	sam3_tensor_compute_strides(t);

	int nelems = sam3_tensor_nelems(t);
	size_t elem_sz = sam3_dtype_size(dtype);
	size_t nbytes = (size_t)nelems * elem_sz;
	t->nbytes = nbytes;

	t->data = sam3_arena_alloc(arena, nbytes);
	if (!t->data)
		return -1;

	return 0;
}

/*
 * fill_random_f32 - Fill a tensor's data buffer with pseudo-random f32 values.
 *
 * Values are small ([-0.5, 0.5]) to avoid overflow in reduced-precision dtypes.
 * For F16/BF16 tensors, generates f32 values and converts in-place.
 */
static void fill_random(struct sam3_tensor *t, uint32_t *seed)
{
	int nelems = sam3_tensor_nelems(t);

	if (t->dtype == SAM3_DTYPE_F32) {
		float *p = (float *)t->data;
		for (int i = 0; i < nelems; i++) {
			/* xorshift32 */
			*seed ^= *seed << 13;
			*seed ^= *seed >> 17;
			*seed ^= *seed << 5;
			p[i] = ((float)(*seed & 0xFFFFu) / 65536.0f) - 0.5f;
		}
	} else if (t->dtype == SAM3_DTYPE_F16) {
		uint16_t *p = (uint16_t *)t->data;
		for (int i = 0; i < nelems; i++) {
			*seed ^= *seed << 13;
			*seed ^= *seed >> 17;
			*seed ^= *seed << 5;
			float v = ((float)(*seed & 0xFFFFu) / 65536.0f) - 0.5f;
			p[i] = f32_to_fp16(v);
		}
	} else if (t->dtype == SAM3_DTYPE_BF16) {
		uint16_t *p = (uint16_t *)t->data;
		for (int i = 0; i < nelems; i++) {
			*seed ^= *seed << 13;
			*seed ^= *seed >> 17;
			*seed ^= *seed << 5;
			float v = ((float)(*seed & 0xFFFFu) / 65536.0f) - 0.5f;
			p[i] = f32_to_bf16(v);
		}
	}
}

/* Dtype tag strings for benchmark names. */
static const char *dtype_tag(enum sam3_dtype dtype)
{
	switch (dtype) {
	case SAM3_DTYPE_F32:  return "f32";
	case SAM3_DTYPE_F16:  return "f16";
	case SAM3_DTYPE_BF16: return "bf16";
	default:              return "unk";
	}
}

/* --- Matmul benchmarks --- */

static int bench_matmul(const struct sam3_bench_config *cfg,
			struct sam3_backend *be,
			struct sam3_arena *arena,
			struct sam3_bench_result *results,
			int max_results, int count)
{
	static const int sizes[] = { 256, 512, 1024, 2048, 4096 };
	static const enum sam3_dtype dtypes[] = {
		SAM3_DTYPE_F32, SAM3_DTYPE_F16, SAM3_DTYPE_BF16,
	};
	int n_sizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
	int n_dtypes = (int)(sizeof(dtypes) / sizeof(dtypes[0]));
	uint32_t seed = 42;

	for (int si = 0; si < n_sizes; si++) {
		for (int di = 0; di < n_dtypes; di++) {
			if (count >= max_results)
				return count;

			int m = sizes[si];
			char name[128];
			snprintf(name, sizeof(name), "matmul_%s_%dx%dx%d",
				 dtype_tag(dtypes[di]), m, m, m);

			if (!sam3_bench_filter_match(name, cfg->filter))
				continue;

			sam3_arena_reset(arena);
			if (be->ops->arena_reset)
				be->ops->arena_reset(be);

			/* Allocate tensor headers from arena. */
			struct sam3_tensor *a = sam3_arena_alloc(arena,
				sizeof(struct sam3_tensor));
			struct sam3_tensor *b = sam3_arena_alloc(arena,
				sizeof(struct sam3_tensor));
			struct sam3_tensor *c = sam3_arena_alloc(arena,
				sizeof(struct sam3_tensor));
			if (!a || !b || !c) {
				sam3_log_error("matmul bench: arena OOM for headers");
				return -1;
			}

			int dims_ab[2] = { m, m };
			int dims_c[2] = { m, m };
			if (init_tensor(arena, a, dtypes[di], dims_ab, 2) ||
			    init_tensor(arena, b, dtypes[di], dims_ab, 2) ||
			    init_tensor(arena, c, dtypes[di], dims_c, 2)) {
				sam3_log_error("matmul bench: arena OOM for data");
				return -1;
			}

			fill_random(a, &seed);
			fill_random(b, &seed);

			/* Allocate tensors on backend. */
			if (be->ops->alloc_tensor(be, a) != SAM3_OK ||
			    be->ops->alloc_tensor(be, b) != SAM3_OK ||
			    be->ops->alloc_tensor(be, c) != SAM3_OK) {
				sam3_log_error("matmul bench: backend alloc failed");
				return -1;
			}

			struct kernel_bench_ctx kctx;
			kctx.be = be;
			sam3_graph_init(&kctx.graph);

			struct sam3_tensor *inputs[2] = { a, b };
			sam3_graph_add_op(&kctx.graph, SAM3_OP_MATMUL,
					  inputs, 2, c);

			double flops = 2.0 * m * m * m;

			if (sam3_bench_run(cfg, name, "kernel",
					   kernel_bench_fn, &kctx,
					   flops, 0,
					   &results[count]) != 0) {
				sam3_log_error("matmul bench: harness error");
				return -1;
			}
			count++;
		}
	}

	return count;
}

/* --- Elementwise benchmarks (add, mul, gelu, silu) --- */

static int bench_elementwise(const struct sam3_bench_config *cfg,
			     struct sam3_backend *be,
			     struct sam3_arena *arena,
			     struct sam3_bench_result *results,
			     int max_results, int count)
{
	static const int sizes[] = { 1024 * 1024, 4 * 1024 * 1024,
				     16 * 1024 * 1024 };
	static const char *size_tags[] = { "1M", "4M", "16M" };
	static const enum sam3_dtype dtypes[] = {
		SAM3_DTYPE_F32, SAM3_DTYPE_F16,
	};

	struct {
		const char  *tag;
		enum sam3_op op;
		int          n_inputs; /* 2 = binary, 1 = unary */
	} ops[] = {
		{ "add",  SAM3_OP_ADD,  2 },
		{ "mul",  SAM3_OP_MUL,  2 },
		{ "gelu", SAM3_OP_GELU, 1 },
		{ "silu", SAM3_OP_SILU, 1 },
	};
	int n_ops = (int)(sizeof(ops) / sizeof(ops[0]));
	int n_sizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
	int n_dtypes = (int)(sizeof(dtypes) / sizeof(dtypes[0]));
	uint32_t seed = 123;

	for (int oi = 0; oi < n_ops; oi++) {
		for (int si = 0; si < n_sizes; si++) {
			for (int di = 0; di < n_dtypes; di++) {
				if (count >= max_results)
					return count;

				char name[128];
				snprintf(name, sizeof(name), "%s_%s_%s",
					 ops[oi].tag,
					 dtype_tag(dtypes[di]),
					 size_tags[si]);

				if (!sam3_bench_filter_match(name, cfg->filter))
					continue;

				sam3_arena_reset(arena);
				if (be->ops->arena_reset)
					be->ops->arena_reset(be);

				int n = sizes[si];
				int dims[1] = { n };
				size_t elem_sz = sam3_dtype_size(dtypes[di]);
				int is_binary = (ops[oi].n_inputs == 2);

				/* bytes = n * elem_sz * (inputs + output) */
				double bytes = (double)n * elem_sz *
					(is_binary ? 3 : 2);

				/* Allocate tensor headers. */
				struct sam3_tensor *x = sam3_arena_alloc(
					arena, sizeof(struct sam3_tensor));
				struct sam3_tensor *out = sam3_arena_alloc(
					arena, sizeof(struct sam3_tensor));
				struct sam3_tensor *y = NULL;
				if (is_binary) {
					y = sam3_arena_alloc(arena,
						sizeof(struct sam3_tensor));
				}
				if (!x || !out || (is_binary && !y)) {
					sam3_log_error("elem bench: arena OOM");
					return -1;
				}

				if (init_tensor(arena, x, dtypes[di],
						dims, 1) ||
				    init_tensor(arena, out, dtypes[di],
						dims, 1)) {
					sam3_log_error("elem bench: arena OOM");
					return -1;
				}
				fill_random(x, &seed);

				if (be->ops->alloc_tensor(be, x) != SAM3_OK ||
				    be->ops->alloc_tensor(be, out) != SAM3_OK) {
					sam3_log_error("elem bench: backend alloc");
					return -1;
				}

				struct kernel_bench_ctx kctx;
				kctx.be = be;
				sam3_graph_init(&kctx.graph);

				if (is_binary) {
					if (init_tensor(arena, y, dtypes[di],
							dims, 1)) {
						sam3_log_error("elem bench: OOM");
						return -1;
					}
					fill_random(y, &seed);
					if (be->ops->alloc_tensor(be, y) !=
					    SAM3_OK) {
						sam3_log_error("elem bench: "
							"backend alloc");
						return -1;
					}
					struct sam3_tensor *inputs[2] = {
						x, y
					};
					sam3_graph_add_op(&kctx.graph,
							  ops[oi].op,
							  inputs, 2, out);
				} else {
					struct sam3_tensor *inputs[1] = { x };
					sam3_graph_add_op(&kctx.graph,
							  ops[oi].op,
							  inputs, 1, out);
				}

				if (sam3_bench_run(cfg, name, "kernel",
						   kernel_bench_fn, &kctx,
						   0, bytes,
						   &results[count]) != 0) {
					sam3_log_error("elem bench: harness");
					return -1;
				}
				count++;
			}
		}
	}

	return count;
}

/* --- Softmax benchmarks  --- */

static int bench_softmax(const struct sam3_bench_config *cfg,
			 struct sam3_backend *be,
			 struct sam3_arena *arena,
			 struct sam3_bench_result *results,
			 int max_results, int count)
{
	static const int sizes[] = { 1024, 4096, 16384, 65536 };
	static const enum sam3_dtype dtypes[] = {
		SAM3_DTYPE_F32, SAM3_DTYPE_F16,
	};
	int n_sizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
	int n_dtypes = (int)(sizeof(dtypes) / sizeof(dtypes[0]));
	uint32_t seed = 456;

	for (int si = 0; si < n_sizes; si++) {
		for (int di = 0; di < n_dtypes; di++) {
			if (count >= max_results)
				return count;

			char name[128];
			snprintf(name, sizeof(name), "softmax_%s_%d",
				 dtype_tag(dtypes[di]), sizes[si]);

			if (!sam3_bench_filter_match(name, cfg->filter))
				continue;

			sam3_arena_reset(arena);
			if (be->ops->arena_reset)
				be->ops->arena_reset(be);

			int dims[2] = { 1, sizes[si] };

			struct sam3_tensor *x = sam3_arena_alloc(arena,
				sizeof(struct sam3_tensor));
			struct sam3_tensor *out = sam3_arena_alloc(arena,
				sizeof(struct sam3_tensor));
			if (!x || !out) {
				sam3_log_error("softmax bench: arena OOM");
				return -1;
			}

			if (init_tensor(arena, x, dtypes[di], dims, 2) ||
			    init_tensor(arena, out, dtypes[di], dims, 2)) {
				sam3_log_error("softmax bench: arena OOM");
				return -1;
			}
			fill_random(x, &seed);

			if (be->ops->alloc_tensor(be, x) != SAM3_OK ||
			    be->ops->alloc_tensor(be, out) != SAM3_OK) {
				sam3_log_error("softmax bench: backend alloc");
				return -1;
			}

			struct kernel_bench_ctx kctx;
			kctx.be = be;
			sam3_graph_init(&kctx.graph);

			struct sam3_tensor *inputs[1] = { x };
			sam3_graph_add_op(&kctx.graph, SAM3_OP_SOFTMAX,
					  inputs, 1, out);

			if (sam3_bench_run(cfg, name, "kernel",
					   kernel_bench_fn, &kctx,
					   0, 0,
					   &results[count]) != 0) {
				sam3_log_error("softmax bench: harness error");
				return -1;
			}
			count++;
		}
	}

	return count;
}

/* --- LayerNorm benchmarks --- */

static int bench_layernorm(const struct sam3_bench_config *cfg,
			   struct sam3_backend *be,
			   struct sam3_arena *arena,
			   struct sam3_bench_result *results,
			   int max_results, int count)
{
	static const int sizes[] = { 256, 512, 1024, 2048 };
	static const enum sam3_dtype dtypes[] = {
		SAM3_DTYPE_F32, SAM3_DTYPE_F16,
	};
	int n_sizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
	int n_dtypes = (int)(sizeof(dtypes) / sizeof(dtypes[0]));
	uint32_t seed = 789;

	for (int si = 0; si < n_sizes; si++) {
		for (int di = 0; di < n_dtypes; di++) {
			if (count >= max_results)
				return count;

			char name[128];
			snprintf(name, sizeof(name), "layernorm_%s_%d",
				 dtype_tag(dtypes[di]), sizes[si]);

			if (!sam3_bench_filter_match(name, cfg->filter))
				continue;

			sam3_arena_reset(arena);
			if (be->ops->arena_reset)
				be->ops->arena_reset(be);

			int dims_x[2] = { 1, sizes[si] };
			int dims_p[1] = { sizes[si] };

			struct sam3_tensor *x = sam3_arena_alloc(arena,
				sizeof(struct sam3_tensor));
			struct sam3_tensor *gamma = sam3_arena_alloc(arena,
				sizeof(struct sam3_tensor));
			struct sam3_tensor *beta = sam3_arena_alloc(arena,
				sizeof(struct sam3_tensor));
			struct sam3_tensor *out = sam3_arena_alloc(arena,
				sizeof(struct sam3_tensor));
			if (!x || !gamma || !beta || !out) {
				sam3_log_error("layernorm bench: arena OOM");
				return -1;
			}

			if (init_tensor(arena, x, dtypes[di], dims_x, 2) ||
			    init_tensor(arena, gamma, dtypes[di], dims_p, 1) ||
			    init_tensor(arena, beta, dtypes[di], dims_p, 1) ||
			    init_tensor(arena, out, dtypes[di], dims_x, 2)) {
				sam3_log_error("layernorm bench: arena OOM");
				return -1;
			}
			fill_random(x, &seed);
			fill_random(gamma, &seed);
			fill_random(beta, &seed);

			if (be->ops->alloc_tensor(be, x) != SAM3_OK ||
			    be->ops->alloc_tensor(be, gamma) != SAM3_OK ||
			    be->ops->alloc_tensor(be, beta) != SAM3_OK ||
			    be->ops->alloc_tensor(be, out) != SAM3_OK) {
				sam3_log_error("layernorm bench: backend alloc");
				return -1;
			}

			struct kernel_bench_ctx kctx;
			kctx.be = be;
			sam3_graph_init(&kctx.graph);

			struct sam3_tensor *inputs[3] = { x, gamma, beta };
			sam3_graph_add_op(&kctx.graph, SAM3_OP_LAYERNORM,
					  inputs, 3, out);

			if (sam3_bench_run(cfg, name, "kernel",
					   kernel_bench_fn, &kctx,
					   0, 0,
					   &results[count]) != 0) {
				sam3_log_error("layernorm bench: harness error");
				return -1;
			}
			count++;
		}
	}

	return count;
}

/* --- Transpose benchmarks --- */

static int bench_transpose(const struct sam3_bench_config *cfg,
			   struct sam3_backend *be,
			   struct sam3_arena *arena,
			   struct sam3_bench_result *results,
			   int max_results, int count)
{
	static const int sizes[] = { 1024, 2048 };
	static const enum sam3_dtype dtypes[] = {
		SAM3_DTYPE_F32, SAM3_DTYPE_F16,
	};
	int n_sizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
	int n_dtypes = (int)(sizeof(dtypes) / sizeof(dtypes[0]));
	uint32_t seed = 1011;

	for (int si = 0; si < n_sizes; si++) {
		for (int di = 0; di < n_dtypes; di++) {
			if (count >= max_results)
				return count;

			int n = sizes[si];
			char name[128];
			snprintf(name, sizeof(name), "transpose_%s_%dx%d",
				 dtype_tag(dtypes[di]), n, n);

			if (!sam3_bench_filter_match(name, cfg->filter))
				continue;

			sam3_arena_reset(arena);
			if (be->ops->arena_reset)
				be->ops->arena_reset(be);

			int dims[2] = { n, n };

			struct sam3_tensor *x = sam3_arena_alloc(arena,
				sizeof(struct sam3_tensor));
			struct sam3_tensor *out = sam3_arena_alloc(arena,
				sizeof(struct sam3_tensor));
			if (!x || !out) {
				sam3_log_error("transpose bench: arena OOM");
				return -1;
			}

			if (init_tensor(arena, x, dtypes[di], dims, 2) ||
			    init_tensor(arena, out, dtypes[di], dims, 2)) {
				sam3_log_error("transpose bench: arena OOM");
				return -1;
			}
			fill_random(x, &seed);

			if (be->ops->alloc_tensor(be, x) != SAM3_OK ||
			    be->ops->alloc_tensor(be, out) != SAM3_OK) {
				sam3_log_error("transpose bench: backend alloc");
				return -1;
			}

			struct kernel_bench_ctx kctx;
			kctx.be = be;
			sam3_graph_init(&kctx.graph);

			struct sam3_tensor *inputs[1] = { x };
			sam3_graph_add_op(&kctx.graph, SAM3_OP_TRANSPOSE,
					  inputs, 1, out);

			/* bytes = rows * cols * elem_size * 2 (read + write) */
			double bytes = (double)n * n *
				sam3_dtype_size(dtypes[di]) * 2;

			if (sam3_bench_run(cfg, name, "kernel",
					   kernel_bench_fn, &kctx,
					   0, bytes,
					   &results[count]) != 0) {
				sam3_log_error("transpose bench: harness error");
				return -1;
			}
			count++;
		}
	}

	return count;
}

/* --- Public entry point  --- */

int sam3_bench_run_kernels(const struct sam3_bench_config *cfg,
			   struct sam3_backend *be,
			   struct sam3_bench_result *results,
			   int max_results)
{
	if (!cfg || !be || !results || max_results <= 0) {
		sam3_log_error("bench_run_kernels: invalid arguments");
		return -1;
	}

	struct sam3_arena arena;
	if (sam3_arena_init(&arena, SCRATCH_ARENA_SIZE) != SAM3_OK) {
		sam3_log_error("bench_run_kernels: failed to allocate "
			       "scratch arena (%d MB)",
			       SCRATCH_ARENA_SIZE / (1024 * 1024));
		return -1;
	}

	int count = 0;

	count = bench_matmul(cfg, be, &arena, results, max_results, count);
	if (count < 0)
		goto cleanup;

	count = bench_elementwise(cfg, be, &arena, results, max_results,
				  count);
	if (count < 0)
		goto cleanup;

	count = bench_softmax(cfg, be, &arena, results, max_results, count);
	if (count < 0)
		goto cleanup;

	count = bench_layernorm(cfg, be, &arena, results, max_results, count);
	if (count < 0)
		goto cleanup;

	count = bench_transpose(cfg, be, &arena, results, max_results, count);
	if (count < 0)
		goto cleanup;

	sam3_log_info("kernel benchmarks: %d cases completed", count);

cleanup:
	sam3_arena_free(&arena);
	return count;
}
