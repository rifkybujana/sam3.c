/*
 * tests/test_necks.c - Feature pyramid neck unit tests
 *
 * Tests the multi-scale neck with small dimensions to verify
 * initialization, weight loading, graph construction, and output
 * shapes. Uses zeroed weights (no weight file) and a small config:
 * d_model=16, backbone_dim=32, grid_size=4, 3 scales.
 *
 * Key types:  sam3_neck, sam3_graph, sam3_cpu_backend
 * Depends on: test_helpers.h, model/necks.h,
 *             backend/cpu/cpu_backend.h
 * Used by:    CTest
 *
 * Copyright (c) 2026
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
#define TEST_N_SCALES     3

static const float test_scales[] = {2.0f, 1.0f, 0.5f};

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

	ASSERT_NEAR(neck.stages[0].scale_factor, 2.0f, 1e-6f);
	ASSERT_NEAR(neck.stages[1].scale_factor, 1.0f, 1e-6f);
	ASSERT_NEAR(neck.stages[2].scale_factor, 0.5f, 1e-6f);
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

	for (int i = 0; i < TEST_N_SCALES; i++) {
		/* Projection weights */
		ASSERT(neck.stages[i].proj_w != NULL);
		ASSERT_EQ(neck.stages[i].proj_w->n_dims, 2);
		ASSERT_EQ(neck.stages[i].proj_w->dims[0], TEST_D_MODEL);
		ASSERT_EQ(neck.stages[i].proj_w->dims[1],
			  TEST_BACKBONE_DIM);

		ASSERT(neck.stages[i].proj_b != NULL);
		ASSERT_EQ(neck.stages[i].proj_b->dims[0], TEST_D_MODEL);

		/* Layer norm */
		ASSERT(neck.stages[i].ln_w != NULL);
		ASSERT_EQ(neck.stages[i].ln_w->dims[0], TEST_D_MODEL);
		ASSERT(neck.stages[i].ln_b != NULL);
		ASSERT_EQ(neck.stages[i].ln_b->dims[0], TEST_D_MODEL);
	}

	/* Downsample conv should exist only for scale < 1.0 */
	ASSERT(neck.stages[0].down_w == NULL); /* scale=2.0 */
	ASSERT(neck.stages[1].down_w == NULL); /* scale=1.0 */
	ASSERT(neck.stages[2].down_w != NULL); /* scale=0.5 */

	ASSERT_EQ(neck.stages[2].down_w->n_dims, 4);
	ASSERT_EQ(neck.stages[2].down_w->dims[0], TEST_D_MODEL);
	ASSERT_EQ(neck.stages[2].down_w->dims[1], TEST_D_MODEL);
	ASSERT_EQ(neck.stages[2].down_w->dims[2], 1);
	ASSERT_EQ(neck.stages[2].down_w->dims[3], 1);
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
	 * Expected output shapes (NCHW):
	 * scale=2.0: [1, d_model, 8, 8]   (4*2 = 8)
	 * scale=1.0: [1, d_model, 4, 4]   (no change)
	 * scale=0.5: [1, d_model, 2, 2]   (4/2 = 2)
	 */
	ASSERT(features[0] != NULL);
	ASSERT_EQ(features[0]->n_dims, 4);
	ASSERT_EQ(features[0]->dims[0], 1);
	ASSERT_EQ(features[0]->dims[1], TEST_D_MODEL);
	ASSERT_EQ(features[0]->dims[2], TEST_GRID_SIZE * 2);
	ASSERT_EQ(features[0]->dims[3], TEST_GRID_SIZE * 2);

	ASSERT(features[1] != NULL);
	ASSERT_EQ(features[1]->n_dims, 4);
	ASSERT_EQ(features[1]->dims[0], 1);
	ASSERT_EQ(features[1]->dims[1], TEST_D_MODEL);
	ASSERT_EQ(features[1]->dims[2], TEST_GRID_SIZE);
	ASSERT_EQ(features[1]->dims[3], TEST_GRID_SIZE);

	ASSERT(features[2] != NULL);
	ASSERT_EQ(features[2]->n_dims, 4);
	ASSERT_EQ(features[2]->dims[0], 1);
	ASSERT_EQ(features[2]->dims[1], TEST_D_MODEL);
	ASSERT_EQ(features[2]->dims[2], TEST_GRID_SIZE / 2);
	ASSERT_EQ(features[2]->dims[3], TEST_GRID_SIZE / 2);
}

/*
 * test_neck_eval - Evaluate the neck graph on CPU.
 *
 * Builds and evaluates with zeroed weights (except layernorm gamma=1).
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
	 * Set layer norm gamma to 1.0 so layernorm produces finite
	 * output even with zero input (avoids 0/0 = NaN).
	 */
	for (int i = 0; i < TEST_N_SCALES; i++) {
		float *w = (float *)neck.stages[i].ln_w->data;
		for (int j = 0; j < TEST_D_MODEL; j++)
			w[j] = 1.0f;
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
