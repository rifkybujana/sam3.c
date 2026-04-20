/*
 * tests/test_memory_attn_multiplex_forward.c - Memory attention multiplex forward smoke test.
 *
 * Loads the tracker_multiplex memory-attention transformer from
 * models/sam3.1.sam3 and runs one forward pass on a miniature 8x8
 * grid (Nq=64) with synthetic inputs. Verifies that the 4-layer
 * decoupled RoPE transformer compiles a runnable graph, produces the
 * expected [1, Nq, 256] output, and that values are finite and
 * non-trivial. Real numerical parity against the Python reference
 * waits for the full CUDA / BatchedDatapoint reference path to be
 * unblocked (tracked in the sub-project 2 design spec).
 *
 * Key types:  sam3_tracker_multiplex, sam3_multiplex_memory_attn
 * Depends on: model/tracker_multiplex.h, model/graph_helpers.h, backend/cpu,
 *             core/graph.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "sam3/sam3.h"
#include "core/weight.h"
#include "core/alloc.h"
#include "core/graph.h"
#include "model/tracker_multiplex.h"
#include "model/graph_helpers.h"
#include "backend/cpu/cpu_backend.h"
#include "backend/backend.h"
#include "test_helpers.h"

#ifndef SAM3_SOURCE_DIR
#error "SAM3_SOURCE_DIR must be defined by CMake"
#endif

#define MODEL_PATH SAM3_SOURCE_DIR "/models/sam3.1.sam3"

/*
 * Miniature grid. Production uses 72x72 = 5184 and the attention
 * matrix scales as Nq^2 per head (~130 MiB per layer at fp32), which
 * is slow to run in CI. An 8x8 grid exercises the same code paths in
 * under a second.
 */
#define TEST_GRID_W   8
#define TEST_NQ       (TEST_GRID_W * TEST_GRID_W)    /* 64 */
#define TEST_HIDDEN   256
#define TEST_N_MM     2                              /* memory slots */
#define TEST_N_OBJ    4                              /* obj_ptr tokens */
#define TEST_NM       (TEST_NQ * TEST_N_MM + TEST_N_OBJ)  /* 132 */

static void fill_pattern(float *dst, int n, float step, int period)
{
	for (int i = 0; i < n; i++)
		dst[i] = step * (float)((i % period) - (period / 2));
}

int main(void)
{
	if (access(MODEL_PATH, F_OK) != 0) {
		printf("test_memory_attn_multiplex_forward: SKIP (%s missing)\n",
		       MODEL_PATH);
		return 0;
	}

	/* --- 1. Backend + arenas  --- */
	struct sam3_cpu_backend cpu;
	memset(&cpu, 0, sizeof(cpu));
	cpu.base.type = SAM3_BACKEND_CPU;
	cpu.base.ops = sam3_cpu_backend_ops();
	cpu.arena_capacity = 512 * 1024 * 1024; /* 512 MiB scratch */
	ASSERT_EQ(cpu.base.ops->init(&cpu.base), SAM3_OK);

	struct sam3_arena weight_arena;
	memset(&weight_arena, 0, sizeof(weight_arena));
	ASSERT_EQ(sam3_arena_init(&weight_arena, 16 * 1024 * 1024), SAM3_OK);

	/* --- 2. Load tracker_multiplex weights  --- */
	struct sam3_weight_file wf;
	memset(&wf, 0, sizeof(wf));
	ASSERT_EQ(sam3_weight_open(&wf, MODEL_PATH), SAM3_OK);

	struct sam3_tracker_multiplex trk;
	ASSERT_EQ(sam3_tracker_multiplex_init(&trk), SAM3_OK);
	ASSERT_EQ(sam3_tracker_multiplex_load(&trk, &wf, &weight_arena), SAM3_OK);

	/* --- 3. Build synthetic inputs  --- */
	int nq_dims[] = {1, TEST_NQ, TEST_HIDDEN};
	int nm_dims[] = {1, TEST_NM, TEST_HIDDEN};

	struct sam3_tensor *tgt          = gh_alloc_tensor(&cpu.arena,
			SAM3_DTYPE_F32, 3, nq_dims);
	struct sam3_tensor *image        = gh_alloc_tensor(&cpu.arena,
			SAM3_DTYPE_F32, 3, nq_dims);
	struct sam3_tensor *memory       = gh_alloc_tensor(&cpu.arena,
			SAM3_DTYPE_F32, 3, nm_dims);
	struct sam3_tensor *memory_image = gh_alloc_tensor(&cpu.arena,
			SAM3_DTYPE_F32, 3, nm_dims);
	struct sam3_tensor *memory_image_pos = gh_alloc_tensor(&cpu.arena,
			SAM3_DTYPE_F32, 3, nm_dims);
	ASSERT(tgt && image && memory
	       && memory_image && memory_image_pos);

	fill_pattern((float *)tgt->data, TEST_NQ * TEST_HIDDEN,
		     0.03f, 23);
	fill_pattern((float *)image->data, TEST_NQ * TEST_HIDDEN,
		     0.025f, 29);
	fill_pattern((float *)memory->data, TEST_NM * TEST_HIDDEN,
		     0.04f, 31);
	fill_pattern((float *)memory_image->data, TEST_NM * TEST_HIDDEN,
		     0.035f, 37);
	fill_pattern((float *)memory_image_pos->data,
		     TEST_NM * TEST_HIDDEN, 0.01f, 41);

	/* --- 4. Two-pass forward: memory_image_pos populated, then NULL --- */
	struct sam3_tensor *pass_mem_ip[2]   = {memory_image_pos, NULL};
	const char *labels[2] = {"with_mem_pos", "no_mem_pos"};

	for (int pass = 0; pass < 2; pass++) {
		struct sam3_graph graph;
		sam3_graph_init(&graph);

		struct sam3_tensor *out = sam3_multiplex_memory_attn_forward(
				&graph, &cpu.arena, &trk.transformer,
				tgt, image,
				memory, memory_image, pass_mem_ip[pass],
				TEST_GRID_W, TEST_N_OBJ);
		ASSERT(out != NULL);

		ASSERT_EQ(out->n_dims, 3);
		ASSERT_EQ(out->dims[0], 1);
		ASSERT_EQ(out->dims[1], TEST_NQ);
		ASSERT_EQ(out->dims[2], TEST_HIDDEN);

		enum sam3_error err = cpu.base.ops->graph_eval(
				&cpu.base, &graph);
		ASSERT_EQ(err, SAM3_OK);

		/* --- 5. Verify finite, non-trivial output --- */
		const float *od = (const float *)out->data;
		int n_out = TEST_NQ * TEST_HIDDEN;
		int any_nan = 0;
		float abs_max = 0.0f;
		double sum_sq = 0.0;
		for (int i = 0; i < n_out; i++) {
			float v = od[i];
			if (v != v) { any_nan = 1; break; }
			float a = v < 0 ? -v : v;
			if (a > abs_max) abs_max = a;
			sum_sq += (double)v * (double)v;
		}
		ASSERT(!any_nan);
		ASSERT(abs_max > 0.0f);
		/* The output is the final LayerNorm of the 4-layer
		 * encoder; per-token L2 norm should be comparable to
		 * sqrt(256). Require a very generous lower bound to
		 * catch accidental all-zero paths. */
		double mean_sq = sum_sq / (double)n_out;
		ASSERT(mean_sq > 1e-4);

		printf("test_memory_attn_multiplex_forward[%s]: PASS "
		       "(out [%d,%d,%d] abs_max=%.4f mean_sq=%.4f)\n",
		       labels[pass],
		       out->dims[0], out->dims[1], out->dims[2],
		       (double)abs_max, mean_sq);
	}

	sam3_weight_close(&wf);
	sam3_arena_free(&weight_arena);
	cpu.base.ops->free(&cpu.base);
	return 0;
}
