/*
 * tests/test_memory_encoder.c - Memory encoder unit tests
 *
 * Tests initialization, weight loading (zero-init path), and graph
 * construction for the SimpleMaskEncoder memory encoder. Uses the
 * CPU backend to evaluate the built graph and verify output shapes.
 *
 * Key types:  sam3_memory_encoder
 * Depends on: test_helpers.h, model/memory_encoder.h,
 *             backend/cpu/cpu_backend.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <string.h>

#include "test_helpers.h"
#include "model/memory_encoder.h"
#include "model/graph_helpers.h"
#include "backend/cpu/cpu_backend.h"
#include "backend/backend.h"
#include "core/graph.h"

/* --- test infrastructure --- */

static struct sam3_cpu_backend g_cpu;

static void setup(void)
{
	memset(&g_cpu, 0, sizeof(g_cpu));
	g_cpu.base.type = SAM3_BACKEND_CPU;
	g_cpu.base.ops = sam3_cpu_backend_ops();
	g_cpu.arena_capacity = 256 * 1024 * 1024; /* 256 MiB */
	g_cpu.base.ops->init(&g_cpu.base);
}

static void teardown(void)
{
	g_cpu.base.ops->free(&g_cpu.base);
}

/* --- test_mem_encoder_init --- */

static void test_mem_encoder_init(void)
{
	struct sam3_memory_encoder enc;
	enum sam3_error err = sam3_memory_encoder_init(&enc, 256, 64);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT_EQ(enc.in_dim, 256);
	ASSERT_EQ(enc.out_dim, 64);
	ASSERT_EQ(enc.interpol_h, 1152);
	ASSERT_EQ(enc.interpol_w, 1152);
}

static void test_mem_encoder_init_bad_args(void)
{
	struct sam3_memory_encoder enc;
	ASSERT_EQ(sam3_memory_encoder_init(NULL, 256, 64), SAM3_EINVAL);
	ASSERT_EQ(sam3_memory_encoder_init(&enc, 0, 64), SAM3_EINVAL);
	ASSERT_EQ(sam3_memory_encoder_init(&enc, 256, -1), SAM3_EINVAL);
}

/* --- test_mem_encoder_load --- */

static void test_mem_encoder_load_null_wf(void)
{
	struct sam3_memory_encoder enc;
	sam3_memory_encoder_init(&enc, 256, 64);

	enum sam3_error err = sam3_memory_encoder_load(
		&enc, NULL, &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	/* Verify weight tensors are allocated (non-NULL) */
	ASSERT(enc.ds[0].conv_w != NULL);
	ASSERT(enc.ds[0].conv_b != NULL);
	ASSERT(enc.ds[0].ln_w != NULL);
	ASSERT(enc.ds[0].ln_b != NULL);
	ASSERT(enc.ds[3].conv_w != NULL);
	ASSERT(enc.ds_proj_w != NULL);
	ASSERT(enc.ds_proj_b != NULL);
	ASSERT(enc.pix_proj_w != NULL);
	ASSERT(enc.pix_proj_b != NULL);
	ASSERT(enc.fuser[0].dwconv_w != NULL);
	ASSERT(enc.fuser[0].gamma != NULL);
	ASSERT(enc.fuser[1].pwconv2_w != NULL);
	ASSERT(enc.out_proj_w != NULL);
	ASSERT(enc.out_proj_b != NULL);

	/* Verify downsampler conv shapes: OHWI */
	ASSERT_EQ(enc.ds[0].conv_w->dims[0], 4);    /* out_ch = 4 */
	ASSERT_EQ(enc.ds[0].conv_w->dims[1], 3);    /* KH */
	ASSERT_EQ(enc.ds[0].conv_w->dims[2], 3);    /* KW */
	ASSERT_EQ(enc.ds[0].conv_w->dims[3], 1);    /* in_ch = 1 */

	ASSERT_EQ(enc.ds[1].conv_w->dims[0], 16);   /* out_ch */
	ASSERT_EQ(enc.ds[1].conv_w->dims[3], 4);    /* in_ch */

	ASSERT_EQ(enc.ds[2].conv_w->dims[0], 64);   /* out_ch */
	ASSERT_EQ(enc.ds[2].conv_w->dims[3], 16);   /* in_ch */

	ASSERT_EQ(enc.ds[3].conv_w->dims[0], 256);  /* out_ch */
	ASSERT_EQ(enc.ds[3].conv_w->dims[3], 64);   /* in_ch */

	/* Final projection: [256, 1, 1, 256] */
	ASSERT_EQ(enc.ds_proj_w->dims[0], 256);
	ASSERT_EQ(enc.ds_proj_w->dims[1], 1);
	ASSERT_EQ(enc.ds_proj_w->dims[3], 256);

	/* Fuser CXBlock depthwise: [256, 7, 7, 1] */
	ASSERT_EQ(enc.fuser[0].dwconv_w->dims[0], 256);
	ASSERT_EQ(enc.fuser[0].dwconv_w->dims[1], 7);
	ASSERT_EQ(enc.fuser[0].dwconv_w->dims[3], 1);

	/* Fuser CXBlock pwconv1: [1024, 256] */
	ASSERT_EQ(enc.fuser[0].pwconv1_w->dims[0], 1024);
	ASSERT_EQ(enc.fuser[0].pwconv1_w->dims[1], 256);

	/* Output projection: [64, 1, 1, 256] */
	ASSERT_EQ(enc.out_proj_w->dims[0], 64);
	ASSERT_EQ(enc.out_proj_w->dims[3], 256);

	/* Position encoding should be precomputed at 72x72 */
	struct sam3_tensor *pos = sam3_pos_encoding_get(&enc.pos_enc);
	ASSERT(pos != NULL);
	ASSERT_EQ(pos->dims[0], 72);
	ASSERT_EQ(pos->dims[1], 72);
	ASSERT_EQ(pos->dims[2], 128); /* out_dim * 2 = 64 * 2 */
}

/* --- test_mem_encoder_build --- */

static void test_mem_encoder_build_shapes(void)
{
	struct sam3_memory_encoder enc;
	sam3_memory_encoder_init(&enc, 256, 64);
	sam3_memory_encoder_load(&enc, NULL, &g_cpu.arena);

	struct sam3_graph graph;
	sam3_graph_init(&graph);

	/*
	 * For the build test we need:
	 * - pix_feat: [1, 72, 72, 256] NHWC
	 * - mask: [1, 1152, 1152, 1] NHWC (at interpol_size)
	 *
	 * 1152x1152 is too large for a unit test. Instead, test with a
	 * smaller mask that maintains the 16x downsample ratio.
	 * Use 128x128 mask -> 8x8 after downsample (4 stride-2 layers).
	 * pix_feat at 8x8 to match the downsampled mask.
	 * 8x8 is large enough for the 7x7 depthwise conv in CXBlock.
	 */
	int pf_dims[] = {1, 8, 8, 256};
	struct sam3_tensor *pix_feat = gh_alloc_tensor(
		&g_cpu.arena, SAM3_DTYPE_F32, 4, pf_dims);
	ASSERT(pix_feat != NULL);

	int mk_dims[] = {1, 128, 128, 1};
	struct sam3_tensor *mask = gh_alloc_tensor(
		&g_cpu.arena, SAM3_DTYPE_F32, 4, mk_dims);
	ASSERT(mask != NULL);

	struct sam3_tensor *out_feat = NULL;
	struct sam3_tensor *out_pos = NULL;

	enum sam3_error err = sam3_memory_encoder_build(
		&enc, &graph, pix_feat, mask, &g_cpu.arena,
		&out_feat, &out_pos);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT(out_feat != NULL);
	ASSERT(out_pos != NULL);

	/* Output feature shape: [1, 8, 8, 64] */
	ASSERT_EQ(out_feat->n_dims, 4);
	ASSERT_EQ(out_feat->dims[0], 1);
	ASSERT_EQ(out_feat->dims[1], 8);
	ASSERT_EQ(out_feat->dims[2], 8);
	ASSERT_EQ(out_feat->dims[3], 64);

	/* Position encoding: [72, 72, 128] (precomputed) */
	ASSERT_EQ(out_pos->n_dims, 3);
	ASSERT_EQ(out_pos->dims[0], 72);
	ASSERT_EQ(out_pos->dims[1], 72);
	ASSERT_EQ(out_pos->dims[2], 128);

	/* Evaluate graph to verify no crashes */
	err = g_cpu.base.ops->graph_eval(&g_cpu.base, &graph);
	ASSERT_EQ(err, SAM3_OK);

	/* Verify output is finite */
	float *feat_data = (float *)out_feat->data;
	int feat_n = out_feat->dims[0] * out_feat->dims[1] *
		     out_feat->dims[2] * out_feat->dims[3];
	for (int i = 0; i < feat_n; i++) {
		ASSERT(feat_data[i] == feat_data[i]); /* Not NaN */
	}
}

static void test_mem_encoder_build_null_args(void)
{
	struct sam3_memory_encoder enc;
	sam3_memory_encoder_init(&enc, 256, 64);

	struct sam3_graph graph;
	sam3_graph_init(&graph);

	struct sam3_tensor dummy;
	memset(&dummy, 0, sizeof(dummy));

	struct sam3_tensor *of = NULL, *op = NULL;

	ASSERT_EQ(sam3_memory_encoder_build(
		NULL, &graph, &dummy, &dummy, &g_cpu.arena,
		&of, &op), SAM3_EINVAL);
	ASSERT_EQ(sam3_memory_encoder_build(
		&enc, NULL, &dummy, &dummy, &g_cpu.arena,
		&of, &op), SAM3_EINVAL);
	ASSERT_EQ(sam3_memory_encoder_build(
		&enc, &graph, NULL, &dummy, &g_cpu.arena,
		&of, &op), SAM3_EINVAL);
	ASSERT_EQ(sam3_memory_encoder_build(
		&enc, &graph, &dummy, NULL, &g_cpu.arena,
		&of, &op), SAM3_EINVAL);
	ASSERT_EQ(sam3_memory_encoder_build(
		&enc, &graph, &dummy, &dummy, NULL,
		&of, &op), SAM3_EINVAL);
	ASSERT_EQ(sam3_memory_encoder_build(
		&enc, &graph, &dummy, &dummy, &g_cpu.arena,
		NULL, &op), SAM3_EINVAL);
	ASSERT_EQ(sam3_memory_encoder_build(
		&enc, &graph, &dummy, &dummy, &g_cpu.arena,
		&of, NULL), SAM3_EINVAL);
}

/* --- main --- */

int main(void)
{
	test_mem_encoder_init();
	test_mem_encoder_init_bad_args();

	setup();

	test_mem_encoder_load_null_wf();
	test_mem_encoder_build_shapes();
	test_mem_encoder_build_null_args();

	teardown();

	TEST_REPORT();
}
