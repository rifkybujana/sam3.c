/*
 * tests/test_mask_decoder_multiplex_forward.c - SAM 3.1 mask decoder forward smoke test.
 *
 * Loads the tracker_multiplex sam_mask_decoder from models/sam3.1.sam3 and runs
 * one forward pass on a miniature 8x8 grid with synthetic inputs. Verifies
 * that the 2-layer two-way transformer + upscaling + hypernetwork + heads
 * compile a runnable graph, produce the expected multiplex-shaped output
 * tensors, and that values are finite and non-trivial. Real numerical
 * parity against the Python reference waits on the full BatchedDatapoint
 * reference path (tracked in the sub-project 2 TODO's "deferred polish"
 * section).
 *
 * Key types:  sam3_tracker_multiplex, sam3_multiplex_mask_decoder
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
 * Miniature grid. Production uses 72x72 image features which would balloon
 * the intermediate [32, 4H*4W] matmul well past 512 MiB. An 8x8 grid
 * exercises the same code paths end-to-end at tractable scratch sizes
 * (peak ~30 MiB) and runs in under a second.
 */
#define TEST_H         8
#define TEST_W         8
#define TEST_C         256
#define TEST_HW        (TEST_H * TEST_W)
#define TEST_MUX       SAM3_MULTIPLEX_COUNT    /* 16 */
#define TEST_NMASK     3                          /* num_multimask_outputs */

static void fill_pattern(float *dst, int n, float step, int period)
{
	for (int i = 0; i < n; i++)
		dst[i] = step * (float)((i % period) - (period / 2));
}

static void check_finite_nontrivial(const struct sam3_tensor *t,
				    const char *label)
{
	int n = sam3_tensor_nelems(t);
	const float *d = (const float *)t->data;
	int any_nan = 0;
	float abs_max = 0.0f;
	double sum_sq = 0.0;
	for (int i = 0; i < n; i++) {
		float v = d[i];
		if (v != v) { any_nan = 1; break; }
		float a = v < 0 ? -v : v;
		if (a > abs_max) abs_max = a;
		sum_sq += (double)v * (double)v;
	}
	ASSERT(!any_nan);
	ASSERT(abs_max > 0.0f);
	ASSERT(sum_sq / (double)n > 1e-8);
	printf("  %s: nelems=%d abs_max=%.4f mean_sq=%.4f\n",
	       label, n, (double)abs_max, sum_sq / (double)n);
}

int main(void)
{
	if (access(MODEL_PATH, F_OK) != 0) {
		printf("test_mask_decoder_multiplex_forward: SKIP (%s missing)\n",
		       MODEL_PATH);
		return 0;
	}

	/* --- 1. Backend + arenas ──────────── --- */
	struct sam3_cpu_backend cpu;
	memset(&cpu, 0, sizeof(cpu));
	cpu.base.type = SAM3_BACKEND_CPU;
	cpu.base.ops = sam3_cpu_backend_ops();
	cpu.arena_capacity = 512 * 1024 * 1024; /* 512 MiB scratch */
	ASSERT_EQ(cpu.base.ops->init(&cpu.base), SAM3_OK);

	struct sam3_arena weight_arena;
	memset(&weight_arena, 0, sizeof(weight_arena));
	ASSERT_EQ(sam3_arena_init(&weight_arena, 16 * 1024 * 1024), SAM3_OK);

	/* --- 2. Load tracker_multiplex weights ─────── --- */
	struct sam3_weight_file wf;
	memset(&wf, 0, sizeof(wf));
	ASSERT_EQ(sam3_weight_open(&wf, MODEL_PATH), SAM3_OK);

	struct sam3_tracker_multiplex trk;
	ASSERT_EQ(sam3_tracker_multiplex_init(&trk), SAM3_OK);
	ASSERT_EQ(sam3_tracker_multiplex_load(&trk, &wf, &weight_arena), SAM3_OK);

	/* --- 3. Build synthetic inputs ────── --- */
	int img_dims[]   = {1, TEST_H, TEST_W, TEST_C};
	int pe_dims[]    = {TEST_HW, TEST_C};
	int s1_dims[]    = {1, TEST_H * 2, TEST_W * 2, TEST_C};
	int s0_dims[]    = {1, TEST_H * 4, TEST_W * 4, TEST_C};
	int extra_dims[] = {TEST_MUX, TEST_C};

	struct sam3_tensor *image_embed = gh_alloc_tensor(&cpu.arena,
			SAM3_DTYPE_F32, 4, img_dims);
	struct sam3_tensor *image_pe    = gh_alloc_tensor(&cpu.arena,
			SAM3_DTYPE_F32, 2, pe_dims);
	struct sam3_tensor *feat_s1     = gh_alloc_tensor(&cpu.arena,
			SAM3_DTYPE_F32, 4, s1_dims);
	struct sam3_tensor *feat_s0     = gh_alloc_tensor(&cpu.arena,
			SAM3_DTYPE_F32, 4, s0_dims);
	struct sam3_tensor *extra_emb   = gh_alloc_tensor(&cpu.arena,
			SAM3_DTYPE_F32, 2, extra_dims);
	ASSERT(image_embed && image_pe && feat_s1 && feat_s0 && extra_emb);

	fill_pattern((float *)image_embed->data,
		     TEST_HW * TEST_C, 0.03f, 23);
	fill_pattern((float *)image_pe->data,
		     TEST_HW * TEST_C, 0.02f, 19);
	fill_pattern((float *)feat_s1->data,
		     TEST_H * 2 * TEST_W * 2 * TEST_C, 0.025f, 29);
	fill_pattern((float *)feat_s0->data,
		     TEST_H * 4 * TEST_W * 4 * TEST_C, 0.018f, 31);
	fill_pattern((float *)extra_emb->data,
		     TEST_MUX * TEST_C, 0.01f, 13);

	/* --- 4. Two passes: with extra_per_object, and without --- */
	struct sam3_tensor *pass_extra[2] = {extra_emb, NULL};
	const char *labels[2] = {"with_extra", "no_extra"};

	for (int pass = 0; pass < 2; pass++) {
		struct sam3_graph graph;
		sam3_graph_init(&graph);

		struct sam3_tensor *masks     = NULL;
		struct sam3_tensor *ious      = NULL;
		struct sam3_tensor *obj_logits = NULL;
		struct sam3_tensor *sam_tok   = NULL;

		enum sam3_error err = sam3_multiplex_mask_decoder_forward(
				&graph, &cpu.arena, &trk.sam_mask_decoder,
				image_embed, image_pe, feat_s1, feat_s0,
				pass_extra[pass],
				&masks, &ious, &obj_logits, &sam_tok);
		ASSERT_EQ(err, SAM3_OK);
		ASSERT(masks && ious && obj_logits && sam_tok);

		/* Shape checks */
		ASSERT_EQ(masks->n_dims, 4);
		ASSERT_EQ(masks->dims[0], TEST_MUX);
		ASSERT_EQ(masks->dims[1], TEST_NMASK);
		ASSERT_EQ(masks->dims[2], TEST_H * 4);
		ASSERT_EQ(masks->dims[3], TEST_W * 4);

		ASSERT_EQ(ious->n_dims, 2);
		ASSERT_EQ(ious->dims[0], TEST_MUX);
		ASSERT_EQ(ious->dims[1], TEST_NMASK);

		ASSERT_EQ(obj_logits->n_dims, 2);
		ASSERT_EQ(obj_logits->dims[0], TEST_MUX);
		ASSERT_EQ(obj_logits->dims[1], 1);

		ASSERT_EQ(sam_tok->n_dims, 3);
		ASSERT_EQ(sam_tok->dims[0], TEST_MUX);
		ASSERT_EQ(sam_tok->dims[1], TEST_NMASK);
		ASSERT_EQ(sam_tok->dims[2], TEST_C);

		err = cpu.base.ops->graph_eval(&cpu.base, &graph);
		ASSERT_EQ(err, SAM3_OK);

		printf("test_mask_decoder_multiplex_forward[%s]:\n", labels[pass]);
		check_finite_nontrivial(masks, "masks");
		check_finite_nontrivial(ious, "iou_scores");
		check_finite_nontrivial(obj_logits, "obj_score_logits");
		check_finite_nontrivial(sam_tok, "sam_tokens");
		printf("test_mask_decoder_multiplex_forward[%s]: PASS\n",
		       labels[pass]);
	}

	sam3_weight_close(&wf);
	sam3_arena_free(&weight_arena);
	cpu.base.ops->free(&cpu.base);
	return 0;
}
