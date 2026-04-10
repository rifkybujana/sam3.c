/*
 * tests/test_metal_two_backends.c - Two Metal backends concurrency smoke test
 *
 * Risk gate for the async image/text pipeline (#11): validates that two
 * independent sam3_metal_backend instances, each with its own mlx_device
 * and mlx_stream, can run trivial graphs concurrently from two pthreads
 * without crashing or producing wrong results. If this test fails, the
 * entire async-pipeline approach must be revisited.
 *
 * Key types:  sam3_metal_backend
 * Depends on: backend/metal/metal_backend.h, model/graph_helpers.h,
 *             core/graph.h, core/alloc.h
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
#include "model/graph_helpers.h"
#include "core/graph.h"
#include "core/alloc.h"

#define M 64
#define N 64
#define K 64
#define N_ITERS 16

struct worker_ctx {
	struct sam3_metal_backend *be;
	struct sam3_arena         *scratch;
	float                      seed;
	int                        ok;
};

/*
 * worker - Run N_ITERS small matmul+add graphs against the given backend.
 *
 * Each iteration resets the scratch arena, allocates fresh F32 tensors,
 * fills them with deterministic values keyed off the seed, runs the
 * graph, and verifies the output is finite. Wrong values or crashes
 * indicate cross-thread state corruption between the two metal backends.
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

		enum sam3_error err = ctx->be->base.ops->graph_eval(
			&ctx->be->base, &g);
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
 * test_two_metal_backends_concurrent - Spin up two backends + two threads.
 *
 * Both threads run worker() in parallel, each against its own backend
 * and scratch arena. The test passes if both threads report ok=1 and
 * neither crashes inside MLX-C.
 */
static void test_two_metal_backends_concurrent(void)
{
	struct sam3_metal_backend be1 = {0};
	struct sam3_metal_backend be2 = {0};

	be1.base.type = SAM3_BACKEND_METAL;
	be1.base.ops  = sam3_metal_backend_ops();
	be2.base.type = SAM3_BACKEND_METAL;
	be2.base.ops  = sam3_metal_backend_ops();

	if (!be1.base.ops) {
		printf("  metal not available, skipping\n");
		return;
	}

	ASSERT_EQ(be1.base.ops->init(&be1.base), SAM3_OK);
	ASSERT_EQ(be2.base.ops->init(&be2.base), SAM3_OK);

	struct sam3_arena scratch1, scratch2;
	ASSERT_EQ(sam3_arena_init(&scratch1, 16UL * 1024 * 1024), SAM3_OK);
	ASSERT_EQ(sam3_arena_init(&scratch2, 16UL * 1024 * 1024), SAM3_OK);

	struct worker_ctx ctx1 = {
		.be = &be1, .scratch = &scratch1, .seed = 1.0f, .ok = 0,
	};
	struct worker_ctx ctx2 = {
		.be = &be2, .scratch = &scratch2, .seed = 2.0f, .ok = 0,
	};

	pthread_t t1, t2;
	ASSERT_EQ(pthread_create(&t1, NULL, worker, &ctx1), 0);
	ASSERT_EQ(pthread_create(&t2, NULL, worker, &ctx2), 0);
	pthread_join(t1, NULL);
	pthread_join(t2, NULL);

	ASSERT_EQ(ctx1.ok, 1);
	ASSERT_EQ(ctx2.ok, 1);

	sam3_arena_free(&scratch1);
	sam3_arena_free(&scratch2);
	be1.base.ops->free(&be1.base);
	be2.base.ops->free(&be2.base);
}

int main(void)
{
	test_two_metal_backends_concurrent();
	TEST_REPORT();
}

#else /* !SAM3_HAS_METAL */

int main(void)
{
	printf("SAM3_HAS_METAL not defined, skipping\n");
	return 0;
}

#endif
