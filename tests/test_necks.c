/*
 * tests/test_necks.c - Feature pyramid neck unit tests
 *
 * Tests the multi-scale FPN neck with small dimensions to verify
 * initialization, weight loading, graph construction, and NHWC
 * output shapes. Uses zeroed weights (no weight file) and a small
 * config: d_model=16, backbone_dim=32, grid_size=4, 4 scales.
 *
 * After the NHWC migration (Task 8) the neck emits
 * [1, H, W, d_model] feature maps and conv weights are stored in
 * OHWI [OC, KH, KW, IC] order — both the shape assertions below
 * reflect the new layout.
 *
 * Key types:  sam3_neck, sam3_graph, sam3_cpu_backend
 * Depends on: test_helpers.h, model/necks.h,
 *             backend/cpu/cpu_backend.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "model/necks.h"
#include "model/graph_helpers.h"
#include "backend/cpu/cpu_backend.h"
#include "backend/backend.h"
#include "core/graph.h"
#include "core/tensor.h"

#include <string.h>

#define TEST_D_MODEL      16
#define TEST_BACKBONE_DIM 32
#define TEST_GRID_SIZE    4
#define TEST_N_PATCHES    (TEST_GRID_SIZE * TEST_GRID_SIZE)  /* 16 */
#define TEST_N_SCALES     4

static const float test_scales[] = {4.0f, 2.0f, 1.0f, 0.5f};

static struct sam3_cpu_backend g_cpu;

static void setup(void)
{
	memset(&g_cpu, 0, sizeof(g_cpu));
	g_cpu.base.type = SAM3_BACKEND_CPU;
	g_cpu.base.ops = sam3_cpu_backend_ops();
	g_cpu.arena_capacity = 128 * 1024 * 1024; /* 128 MiB */
	g_cpu.base.ops->init(&g_cpu.base);
}

static void teardown(void)
{
	g_cpu.base.ops->free(&g_cpu.base);
}

/*
 * test_neck_init - Verify neck initialization sets all fields.
 */
static void test_neck_init(void)
{
	struct sam3_neck neck;
	enum sam3_error err;

	err = sam3_neck_init(&neck, TEST_D_MODEL, TEST_BACKBONE_DIM,
			      TEST_GRID_SIZE, TEST_N_SCALES, test_scales);
	ASSERT_EQ(err, SAM3_OK);

	ASSERT_EQ(neck.d_model, TEST_D_MODEL);
	ASSERT_EQ(neck.backbone_dim, TEST_BACKBONE_DIM);
	ASSERT_EQ(neck.grid_size, TEST_GRID_SIZE);
	ASSERT_EQ(neck.n_scales, TEST_N_SCALES);

	ASSERT_NEAR(neck.stages[0].scale_factor, 4.0f, 1e-6f);
	ASSERT_NEAR(neck.stages[1].scale_factor, 2.0f, 1e-6f);
	ASSERT_NEAR(neck.stages[2].scale_factor, 1.0f, 1e-6f);
	ASSERT_NEAR(neck.stages[3].scale_factor, 0.5f, 1e-6f);

	/* Verify stage metadata */
	ASSERT_EQ(neck.stages[0].n_convs, 4);   /* 4x: 2 ConvT + 1x1 + 3x3 */
	ASSERT_EQ(neck.stages[0].has_maxpool, 0);
	ASSERT_EQ(neck.stages[0].is_transpose[0], 1);
	ASSERT_EQ(neck.stages[0].gelu_after[0], 1);

	ASSERT_EQ(neck.stages[1].n_convs, 3);   /* 2x: ConvT + 1x1 + 3x3 */
	ASSERT_EQ(neck.stages[2].n_convs, 2);   /* 1x: 1x1 + 3x3 */
	ASSERT_EQ(neck.stages[3].n_convs, 2);   /* 0.5x: 1x1 + 3x3 */
	ASSERT_EQ(neck.stages[3].has_maxpool, 1);
}

/*
 * test_neck_init_invalid - Verify init rejects bad n_scales.
 */
static void test_neck_init_invalid(void)
{
	struct sam3_neck neck;
	enum sam3_error err;

	err = sam3_neck_init(&neck, TEST_D_MODEL, TEST_BACKBONE_DIM,
			      TEST_GRID_SIZE, 0, test_scales);
	ASSERT_EQ(err, SAM3_EINVAL);

	err = sam3_neck_init(&neck, TEST_D_MODEL, TEST_BACKBONE_DIM,
			      TEST_GRID_SIZE, 5, test_scales);
	ASSERT_EQ(err, SAM3_EINVAL);
}

/*
 * test_neck_load - Verify weight loading with NULL wf.
 */
static void test_neck_load(void)
{
	struct sam3_neck neck;
	enum sam3_error err;

	err = sam3_neck_init(&neck, TEST_D_MODEL, TEST_BACKBONE_DIM,
			      TEST_GRID_SIZE, TEST_N_SCALES, test_scales);
	ASSERT_EQ(err, SAM3_OK);

	err = sam3_neck_load(&neck, NULL, &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	/* Check all conv layers have been allocated */
	for (int i = 0; i < TEST_N_SCALES; i++) {
		for (int j = 0; j < neck.stages[i].n_convs; j++) {
			ASSERT(neck.stages[i].conv_w[j] != NULL);
			ASSERT_EQ(neck.stages[i].conv_w[j]->n_dims, 4);
			ASSERT(neck.stages[i].conv_b[j] != NULL);
			ASSERT_EQ(neck.stages[i].conv_b[j]->n_dims, 1);
		}
	}

	/*
	 * Verify scale=4.0 stage weight shapes after the NHWC
	 * migration. All conv weights (Conv2d and ConvTranspose2d)
	 * are stored in OHWI [OC, KH, KW, IC] after permutation at
	 * load time.
	 *
	 * conv[0]: ConvT dim->dim/2, k=2 -> OHWI [16, 2, 2, 32]
	 * conv[1]: ConvT dim/2->dim/4, k=2 -> OHWI [8, 2, 2, 16]
	 * conv[2]: Conv1x1 dim/4->d_model  -> OHWI [16, 1, 1, 8]
	 * conv[3]: Conv3x3 d_model->d_model -> OHWI [16, 3, 3, 16]
	 */
	ASSERT_EQ(neck.stages[0].conv_w[0]->dims[0],
		  TEST_BACKBONE_DIM / 2);
	ASSERT_EQ(neck.stages[0].conv_w[0]->dims[1], 2);
	ASSERT_EQ(neck.stages[0].conv_w[0]->dims[2], 2);
	ASSERT_EQ(neck.stages[0].conv_w[0]->dims[3], TEST_BACKBONE_DIM);

	ASSERT_EQ(neck.stages[0].conv_w[2]->dims[0], TEST_D_MODEL);
	ASSERT_EQ(neck.stages[0].conv_w[2]->dims[1], 1);
	ASSERT_EQ(neck.stages[0].conv_w[2]->dims[2], 1);
	ASSERT_EQ(neck.stages[0].conv_w[2]->dims[3],
		  TEST_BACKBONE_DIM / 4);

	ASSERT_EQ(neck.stages[0].conv_w[3]->dims[0], TEST_D_MODEL);
	ASSERT_EQ(neck.stages[0].conv_w[3]->dims[1], 3);
	ASSERT_EQ(neck.stages[0].conv_w[3]->dims[2], 3);
	ASSERT_EQ(neck.stages[0].conv_w[3]->dims[3], TEST_D_MODEL);

	/*
	 * Verify scale=0.5 stage has 2 convs (maxpool is not a conv).
	 * conv[0]: Conv1x1 dim->d_model  -> OHWI [16, 1, 1, 32]
	 * conv[1]: Conv3x3 d_model->d_model -> OHWI [16, 3, 3, 16]
	 */
	ASSERT_EQ(neck.stages[3].conv_w[0]->dims[0], TEST_D_MODEL);
	ASSERT_EQ(neck.stages[3].conv_w[0]->dims[1], 1);
	ASSERT_EQ(neck.stages[3].conv_w[0]->dims[2], 1);
	ASSERT_EQ(neck.stages[3].conv_w[0]->dims[3], TEST_BACKBONE_DIM);
}

/*
 * test_neck_build_shapes - Verify output tensor shapes.
 */
static void test_neck_build_shapes(void)
{
	struct sam3_neck neck;
	enum sam3_error err;

	err = sam3_neck_init(&neck, TEST_D_MODEL, TEST_BACKBONE_DIM,
			      TEST_GRID_SIZE, TEST_N_SCALES, test_scales);
	ASSERT_EQ(err, SAM3_OK);

	err = sam3_neck_load(&neck, NULL, &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	/* Create dummy ViT output [n_patches, backbone_dim] */
	int vit_dims[] = {TEST_N_PATCHES, TEST_BACKBONE_DIM};
	struct sam3_tensor *vit_out;
	vit_out = gh_alloc_tensor(&g_cpu.arena, SAM3_DTYPE_F32,
				   2, vit_dims);
	ASSERT(vit_out != NULL);

	/* Build graph */
	struct sam3_graph graph;
	sam3_graph_init(&graph);

	struct sam3_tensor *features[SAM3_NECK_MAX_SCALES];
	err = sam3_neck_build(&neck, &graph, vit_out, features,
			       &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	/*
	 * Expected output shapes (NHWC [1, H, W, d_model]):
	 * scale=4.0: [1, 16, 16, d_model]  (4*4 = 16)
	 * scale=2.0: [1,  8,  8, d_model]  (4*2 = 8)
	 * scale=1.0: [1,  4,  4, d_model]  (no change)
	 * scale=0.5: [1,  2,  2, d_model]  (4/2 = 2)
	 */
	ASSERT(features[0] != NULL);
	ASSERT_EQ(features[0]->n_dims, 4);
	ASSERT_EQ(features[0]->dims[0], 1);
	ASSERT_EQ(features[0]->dims[1], TEST_GRID_SIZE * 4);
	ASSERT_EQ(features[0]->dims[2], TEST_GRID_SIZE * 4);
	ASSERT_EQ(features[0]->dims[3], TEST_D_MODEL);

	ASSERT(features[1] != NULL);
	ASSERT_EQ(features[1]->n_dims, 4);
	ASSERT_EQ(features[1]->dims[0], 1);
	ASSERT_EQ(features[1]->dims[1], TEST_GRID_SIZE * 2);
	ASSERT_EQ(features[1]->dims[2], TEST_GRID_SIZE * 2);
	ASSERT_EQ(features[1]->dims[3], TEST_D_MODEL);

	ASSERT(features[2] != NULL);
	ASSERT_EQ(features[2]->n_dims, 4);
	ASSERT_EQ(features[2]->dims[0], 1);
	ASSERT_EQ(features[2]->dims[1], TEST_GRID_SIZE);
	ASSERT_EQ(features[2]->dims[2], TEST_GRID_SIZE);
	ASSERT_EQ(features[2]->dims[3], TEST_D_MODEL);

	ASSERT(features[3] != NULL);
	ASSERT_EQ(features[3]->n_dims, 4);
	ASSERT_EQ(features[3]->dims[0], 1);
	ASSERT_EQ(features[3]->dims[1], TEST_GRID_SIZE / 2);
	ASSERT_EQ(features[3]->dims[2], TEST_GRID_SIZE / 2);
	ASSERT_EQ(features[3]->dims[3], TEST_D_MODEL);
}

/*
 * test_neck_eval - Evaluate the neck graph on CPU.
 *
 * Builds and evaluates with small nonzero conv weights.
 * Verifies that output values are finite.
 */
static void test_neck_eval(void)
{
	struct sam3_neck neck;
	enum sam3_error err;

	err = sam3_neck_init(&neck, TEST_D_MODEL, TEST_BACKBONE_DIM,
			      TEST_GRID_SIZE, TEST_N_SCALES, test_scales);
	ASSERT_EQ(err, SAM3_OK);

	err = sam3_neck_load(&neck, NULL, &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	/*
	 * Set conv weights to small nonzero values so the pipeline
	 * produces finite output.
	 */
	for (int i = 0; i < TEST_N_SCALES; i++) {
		for (int j = 0; j < neck.stages[i].n_convs; j++) {
			int nelems = sam3_tensor_nelems(
				neck.stages[i].conv_w[j]);
			float *w = (float *)neck.stages[i].conv_w[j]->data;
			for (int k = 0; k < nelems; k++)
				w[k] = 0.01f;
		}
	}

	/* Create dummy ViT output with small values */
	int vit_dims[] = {TEST_N_PATCHES, TEST_BACKBONE_DIM};
	struct sam3_tensor *vit_out;
	vit_out = gh_alloc_tensor(&g_cpu.arena, SAM3_DTYPE_F32,
				   2, vit_dims);
	ASSERT(vit_out != NULL);

	float *vd = (float *)vit_out->data;
	int n_vit = TEST_N_PATCHES * TEST_BACKBONE_DIM;
	for (int i = 0; i < n_vit; i++)
		vd[i] = 0.01f * (float)(i % 13);

	/* Build graph */
	struct sam3_graph graph;
	sam3_graph_init(&graph);

	struct sam3_tensor *features[SAM3_NECK_MAX_SCALES];
	err = sam3_neck_build(&neck, &graph, vit_out, features,
			       &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	/* Evaluate on CPU */
	err = g_cpu.base.ops->graph_eval(&g_cpu.base, &graph);
	ASSERT_EQ(err, SAM3_OK);

	/* Verify all outputs are finite */
	for (int s = 0; s < TEST_N_SCALES; s++) {
		int nelems = sam3_tensor_nelems(features[s]);
		float *data = (float *)features[s]->data;
		for (int i = 0; i < nelems; i++) {
			ASSERT(data[i] == data[i]);    /* Not NaN */
			ASSERT(data[i] < 1e10f);      /* Not huge */
			ASSERT(data[i] > -1e10f);
		}
	}
}

int main(void)
{
	setup();

	test_neck_init();
	test_neck_init_invalid();
	test_neck_load();
	test_neck_build_shapes();
	test_neck_eval();

	teardown();

	TEST_REPORT();
}
