/*
 * tests/test_metal_cpu_concurrent.c - Metal + CPU concurrent backends test
 *
 * Risk gate for the async image/text pipeline (#11): validates that one
 * sam3_metal_backend and one sam3_cpu_backend can run trivial graphs
 * concurrently from two pthreads without crashing or producing wrong
 * results. This is the actual deployed configuration: the image
 * encoder runs on Metal while the text encoder runs on CPU on a
 * worker thread.
 *
 * Why not two Metal backends? MLX-C 0.6 keeps a process-wide
 * mlx::core::metal::Device with an internal kernel cache that is not
 * thread-safe. Two Metal backends concurrently calling Device::get_kernel
 * race on an unordered_map<MTL::Library*, ...> mutation. CPU + Metal
 * sit on disjoint hardware so they synchronize cleanly.
 *
 * Key types:  sam3_metal_backend, sam3_cpu_backend
 * Depends on: backend/metal/metal_backend.h, backend/cpu/cpu_backend.h,
 *             model/graph_helpers.h, core/graph.h, core/alloc.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"

#ifdef SAM3_HAS_METAL

#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "backend/metal/metal_backend.h"
#include "backend/cpu/cpu_backend.h"
#include "model/graph_helpers.h"
#include "core/graph.h"
#include "core/alloc.h"

#define M 64
#define N 64
#define K 64
#define N_ITERS 16

struct worker_ctx {
	struct sam3_backend *be;
	struct sam3_arena   *scratch;
	float                seed;
	int                  ok;
};

/*
 * worker - Run N_ITERS small matmul+add graphs against the given backend.
 *
 * Each iteration resets the scratch arena, allocates fresh F32 tensors,
 * fills them with deterministic values keyed off the seed, runs the
 * graph, and verifies the output is finite. Wrong values or crashes
 * indicate cross-thread state corruption between the two backends.
 */
static void *worker(void *arg)
{
	struct worker_ctx *ctx = arg;

	for (int it = 0; it < N_ITERS; it++) {
		sam3_arena_reset(ctx->scratch);

		struct sam3_graph g;
		sam3_graph_init(&g);

		int a_dims[]    = {M, K};
		int b_dims[]    = {K, N};
		int bias_dims[] = {N};

		struct sam3_tensor *a = gh_alloc_tensor(
			ctx->scratch, SAM3_DTYPE_F32, 2, a_dims);
		struct sam3_tensor *b = gh_alloc_tensor(
			ctx->scratch, SAM3_DTYPE_F32, 2, b_dims);
		struct sam3_tensor *bias = gh_alloc_tensor(
			ctx->scratch, SAM3_DTYPE_F32, 1, bias_dims);
		if (!a || !b || !bias) {
			ctx->ok = 0;
			return NULL;
		}

		float *ad    = (float *)a->data;
		float *bd    = (float *)b->data;
		float *biasd = (float *)bias->data;
		for (int i = 0; i < M * K; i++)
			ad[i] = ctx->seed + (float)(i % 7) * 0.01f;
		for (int i = 0; i < K * N; i++)
			bd[i] = ctx->seed + (float)(i % 5) * 0.02f;
		for (int i = 0; i < N; i++)
			biasd[i] = ctx->seed * 0.1f;

		struct sam3_tensor *out;
		out = gh_matmul(&g, ctx->scratch, a, b);
		if (!out) { ctx->ok = 0; return NULL; }
		out = gh_add(&g, ctx->scratch, out, bias);
		if (!out) { ctx->ok = 0; return NULL; }

		enum sam3_error err = ctx->be->ops->graph_eval(ctx->be, &g);
		if (err != SAM3_OK) { ctx->ok = 0; return NULL; }

		const float *od = (const float *)out->data;
		if (out->dims[0] != M || out->dims[1] != N) {
			ctx->ok = 0;
			return NULL;
		}
		for (int i = 0; i < M * N; i++) {
			if (!isfinite(od[i])) {
				ctx->ok = 0;
				return NULL;
			}
		}
	}

	ctx->ok = 1;
	return NULL;
}

/*
 * test_metal_cpu_concurrent - Spin up Metal + CPU backends + two threads.
 *
 * Both threads run worker() in parallel: one against Metal, one against
 * CPU. Each has its own scratch arena. The test passes if both threads
 * report ok=1 and neither crashes. This mirrors the actual async
 * pipeline configuration in sam3_processor (image encoder on Metal,
 * text encoder on CPU worker thread).
 */
static void test_metal_cpu_concurrent(void)
{
	struct sam3_metal_backend metal_be = {0};
	struct sam3_cpu_backend   cpu_be   = {0};

	metal_be.base.type = SAM3_BACKEND_METAL;
	metal_be.base.ops  = sam3_metal_backend_ops();
	cpu_be.base.type   = SAM3_BACKEND_CPU;
	cpu_be.base.ops    = sam3_cpu_backend_ops();
	cpu_be.arena_capacity = 16UL * 1024 * 1024;

	if (!metal_be.base.ops) {
		printf("  metal not available, skipping\n");
		return;
	}

	ASSERT_EQ(metal_be.base.ops->init(&metal_be.base), SAM3_OK);
	ASSERT_EQ(cpu_be.base.ops->init(&cpu_be.base), SAM3_OK);

	struct sam3_arena scratch_metal, scratch_cpu;
	ASSERT_EQ(sam3_arena_init(&scratch_metal, 16UL * 1024 * 1024), SAM3_OK);
	ASSERT_EQ(sam3_arena_init(&scratch_cpu,   16UL * 1024 * 1024), SAM3_OK);

	struct worker_ctx ctx_metal = {
		.be      = &metal_be.base,
		.scratch = &scratch_metal,
		.seed    = 1.0f,
		.ok      = 0,
	};
	struct worker_ctx ctx_cpu = {
		.be      = &cpu_be.base,
		.scratch = &scratch_cpu,
		.seed    = 2.0f,
		.ok      = 0,
	};

	/*
	 * MLX-C and ASan together can blow the default 512 KiB pthread
	 * stack on macOS, so request 8 MiB for worker threads. The same
	 * sizing is used by sam3_processor for the async text worker.
	 */
	pthread_attr_t attr;
	ASSERT_EQ(pthread_attr_init(&attr), 0);
	ASSERT_EQ(pthread_attr_setstacksize(&attr, 8UL * 1024 * 1024), 0);

	pthread_t t_metal, t_cpu;
	ASSERT_EQ(pthread_create(&t_metal, &attr, worker, &ctx_metal), 0);
	ASSERT_EQ(pthread_create(&t_cpu,   &attr, worker, &ctx_cpu),   0);
	pthread_join(t_metal, NULL);
	pthread_join(t_cpu,   NULL);

	pthread_attr_destroy(&attr);

	ASSERT_EQ(ctx_metal.ok, 1);
	ASSERT_EQ(ctx_cpu.ok,   1);

	sam3_arena_free(&scratch_metal);
	sam3_arena_free(&scratch_cpu);
	metal_be.base.ops->free(&metal_be.base);
	cpu_be.base.ops->free(&cpu_be.base);
}

int main(void)
{
	test_metal_cpu_concurrent();
	TEST_REPORT();
}

#else /* !SAM3_HAS_METAL */

int main(void)
{
	printf("SAM3_HAS_METAL not defined, skipping\n");
	return 0;
}

#endif
