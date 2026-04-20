/*
 * tests/test_maskmem_multiplex_forward.c - Maskmem multiplex forward smoke test.
 *
 * Loads the tracker_multiplex maskmem backbone from models/sam3.1.sam3 and
 * runs it on synthetic 128x128 mask input (instead of production
 * 1152x1152) to verify the graph builder compiles a runnable forward
 * that produces a finite [1, 8, 8, 256] output. Real numerical parity
 * waits for the full Python reference (deferred).
 *
 * Key types:  sam3_tracker_multiplex, sam3_multiplex_maskmem
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

/* Use a miniature input (128x128 masks, 8x8 pix_feat) so the test runs
 * in a few seconds instead of several minutes. The downsampler's
 * per-stage convs are size-agnostic. */
#define TEST_MASK_HW  128
#define TEST_PIX_HW   (TEST_MASK_HW / 16)  /* 8 */

int main(void)
{
	if (access(MODEL_PATH, F_OK) != 0) {
		printf("test_maskmem_multiplex_forward: SKIP (%s missing)\n",
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
	ASSERT_EQ(sam3_arena_init(&weight_arena, 8 * 1024 * 1024), SAM3_OK);

	/* --- 2. Load maskmem weights  --- */
	struct sam3_weight_file wf;
	memset(&wf, 0, sizeof(wf));
	ASSERT_EQ(sam3_weight_open(&wf, MODEL_PATH), SAM3_OK);

	struct sam3_tracker_multiplex trk;
	ASSERT_EQ(sam3_tracker_multiplex_init(&trk), SAM3_OK);
	ASSERT_EQ(sam3_tracker_multiplex_load(&trk, &wf, &weight_arena), SAM3_OK);

	/* --- 3. Build synthetic input tensors --- */
	int mask_dims[] = {1, TEST_MASK_HW, TEST_MASK_HW,
			   SAM3_MULTIPLEX_IN_CHANNELS};
	int pix_dims[]  = {1, TEST_PIX_HW, TEST_PIX_HW, SAM3_MULTIPLEX_HIDDEN_DIM};

	struct sam3_tensor *masks = gh_alloc_tensor(&cpu.arena,
			SAM3_DTYPE_F32, 4, mask_dims);
	ASSERT(masks != NULL);

	struct sam3_tensor *pix = gh_alloc_tensor(&cpu.arena,
			SAM3_DTYPE_F32, 4, pix_dims);
	ASSERT(pix != NULL);

	float *md = (float *)masks->data;
	int n_mask = 1;
	for (int i = 0; i < masks->n_dims; i++) n_mask *= masks->dims[i];
	for (int i = 0; i < n_mask; i++)
		md[i] = 0.1f * (float)((i % 31) - 15);  /* small non-zero */

	float *pd = (float *)pix->data;
	int n_pix = 1;
	for (int i = 0; i < pix->n_dims; i++) n_pix *= pix->dims[i];
	for (int i = 0; i < n_pix; i++)
		pd[i] = 0.05f * (float)((i % 17) - 8);

	/* --- 4. Build + evaluate the forward graph --- */
	struct sam3_graph graph;
	sam3_graph_init(&graph);

	struct sam3_tensor *out = sam3_multiplex_maskmem_forward(
			&graph, &cpu.arena, &trk.maskmem, pix, masks, 0);
	ASSERT(out != NULL);

	ASSERT_EQ(out->n_dims, 4);
	ASSERT_EQ(out->dims[0], 1);
	ASSERT_EQ(out->dims[1], TEST_PIX_HW);
	ASSERT_EQ(out->dims[2], TEST_PIX_HW);
	ASSERT_EQ(out->dims[3], SAM3_MULTIPLEX_HIDDEN_DIM);

	enum sam3_error err = cpu.base.ops->graph_eval(&cpu.base, &graph);
	ASSERT_EQ(err, SAM3_OK);

	/* --- 5. Verify output is finite and non-trivial --- */
	const float *od = (const float *)out->data;
	int n_out = 1;
	for (int i = 0; i < out->n_dims; i++) n_out *= out->dims[i];
	int any_nan = 0;
	float abs_max = 0.0f;
	for (int i = 0; i < n_out; i++) {
		if (od[i] != od[i]) { any_nan = 1; break; }
		float a = od[i] < 0 ? -od[i] : od[i];
		if (a > abs_max) abs_max = a;
	}
	ASSERT(!any_nan);
	ASSERT(abs_max > 0.0f);  /* non-trivial response */

	printf("test_maskmem_multiplex_forward: PASS "
	       "(out [%d,%d,%d,%d] abs_max=%.4f)\n",
	       out->dims[0], out->dims[1], out->dims[2], out->dims[3],
	       (double)abs_max);

	sam3_weight_close(&wf);
	sam3_arena_free(&weight_arena);
	cpu.base.ops->free(&cpu.base);
	return 0;
}
