/*
 * tests/test_vit.c - ViT image encoder unit tests
 *
 * Tests the ViT backbone with small dimensions to verify
 * initialization, weight loading, graph construction, output shapes,
 * and numerical stability. Uses zeroed weights (no weight file) so
 * the encoder can be tested standalone.
 *
 * Key types:  sam3_vit, sam3_graph, sam3_cpu_backend
 * Depends on: test_helpers.h, model/image_encoder.h,
 *             backend/cpu/cpu_backend.h
 * Used by:    CTest
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "model/image_encoder.h"
#include "model/graph_helpers.h"
#include "backend/cpu/cpu_backend.h"
#include "backend/backend.h"
#include "core/alloc.h"
#include "core/graph.h"
#include "core/tensor.h"

#include <string.h>

/* Small ViT config for fast testing */
#define TEST_IMG_SIZE     28
#define TEST_PATCH_SIZE   14
#define TEST_EMBED_DIM    32
#define TEST_DEPTH        2
#define TEST_N_HEADS      4
#define TEST_WINDOW_SIZE  2
#define TEST_MLP_DIM      64
#define TEST_GRID_SIZE    (TEST_IMG_SIZE / TEST_PATCH_SIZE)   /* 2 */
#define TEST_N_PATCHES    (TEST_GRID_SIZE * TEST_GRID_SIZE)   /* 4 */

static struct sam3_cpu_backend g_cpu;
static struct sam3_arena g_scratch;
static struct sam3_arena g_persist;

static void setup(void)
{
	memset(&g_cpu, 0, sizeof(g_cpu));
	g_cpu.base.type = SAM3_BACKEND_CPU;
	g_cpu.base.ops = sam3_cpu_backend_ops();
	g_cpu.arena_capacity = 128 * 1024 * 1024; /* 128 MiB */
	g_cpu.base.ops->init(&g_cpu.base);

	sam3_arena_init(&g_scratch, 64 * 1024 * 1024); /* 64 MiB */
	sam3_arena_init(&g_persist, 16 * 1024 * 1024); /* 16 MiB */
}

static void teardown(void)
{
	g_cpu.base.ops->free(&g_cpu.base);
	sam3_arena_free(&g_scratch);
	sam3_arena_free(&g_persist);
}

/*
 * test_vit_init - Verify ViT initialization sets all fields correctly.
 */
static void test_vit_init(void)
{
	struct sam3_vit vit;
	enum sam3_error err;

	err = sam3_vit_init(&vit, TEST_IMG_SIZE, TEST_PATCH_SIZE,
			     TEST_EMBED_DIM, TEST_DEPTH, TEST_N_HEADS,
			     TEST_WINDOW_SIZE, TEST_MLP_DIM,
			     &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	ASSERT_EQ(vit.img_size, TEST_IMG_SIZE);
	ASSERT_EQ(vit.patch_size, TEST_PATCH_SIZE);
	ASSERT_EQ(vit.embed_dim, TEST_EMBED_DIM);
	ASSERT_EQ(vit.depth, TEST_DEPTH);
	ASSERT_EQ(vit.n_heads, TEST_N_HEADS);
	ASSERT_EQ(vit.window_size, TEST_WINDOW_SIZE);
	ASSERT_EQ(vit.mlp_dim, TEST_MLP_DIM);
	ASSERT_EQ(vit.grid_size, TEST_GRID_SIZE);
	ASSERT_EQ(vit.n_patches, TEST_N_PATCHES);

	/*
	 * RoPE tables, window mask, and pos_embed are lazily computed
	 * on first sam3_vit_build() call, so they should be NULL here.
	 */
	ASSERT(vit.rope_win_cos == NULL);
	ASSERT(vit.rope_win_sin == NULL);
	ASSERT(vit.rope_glo_cos == NULL);
	ASSERT(vit.rope_glo_sin == NULL);
	ASSERT(vit.window_mask == NULL);
	ASSERT(vit.precomputed == 0);
	ASSERT(vit.model_arena == &g_cpu.arena);

	/* Check is_global flags: depth=2, so no global blocks */
	for (int i = 0; i < TEST_DEPTH; i++)
		ASSERT_EQ(vit.layers[i].is_global, 0);
}

/*
 * test_vit_load - Verify weight loading with NULL wf.
 *
 * All weights should be allocated as zero-initialized tensors
 * with correct shapes.
 */
static void test_vit_load(void)
{
	struct sam3_vit vit;
	enum sam3_error err;

	err = sam3_vit_init(&vit, TEST_IMG_SIZE, TEST_PATCH_SIZE,
			     TEST_EMBED_DIM, TEST_DEPTH, TEST_N_HEADS,
			     TEST_WINDOW_SIZE, TEST_MLP_DIM,
			     &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	err = sam3_vit_load(&vit, NULL, &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	/* Patch embedding weight */
	ASSERT(vit.patch_embed_w != NULL);
	ASSERT_EQ(vit.patch_embed_w->n_dims, 4);
	ASSERT_EQ(vit.patch_embed_w->dims[0], TEST_EMBED_DIM);
	ASSERT_EQ(vit.patch_embed_w->dims[1], 3);
	ASSERT_EQ(vit.patch_embed_w->dims[2], TEST_PATCH_SIZE);
	ASSERT_EQ(vit.patch_embed_w->dims[3], TEST_PATCH_SIZE);

	/* Patch embedding bias */
	ASSERT(vit.patch_embed_b != NULL);
	ASSERT_EQ(vit.patch_embed_b->n_dims, 1);
	ASSERT_EQ(vit.patch_embed_b->dims[0], TEST_EMBED_DIM);

	/* Per-layer weights */
	for (int i = 0; i < TEST_DEPTH; i++) {
		ASSERT(vit.layers[i].ln1_w != NULL);
		ASSERT(vit.layers[i].ln1_b != NULL);
		ASSERT_EQ(vit.layers[i].ln1_w->dims[0], TEST_EMBED_DIM);

		ASSERT(vit.layers[i].qkv_w != NULL);
		ASSERT_EQ(vit.layers[i].qkv_w->dims[0],
			  3 * TEST_EMBED_DIM);
		ASSERT_EQ(vit.layers[i].qkv_w->dims[1], TEST_EMBED_DIM);
		ASSERT(vit.layers[i].qkv_b != NULL);
		ASSERT_EQ(vit.layers[i].qkv_b->dims[0],
			  3 * TEST_EMBED_DIM);

		ASSERT(vit.layers[i].proj_w != NULL);
		ASSERT_EQ(vit.layers[i].proj_w->dims[0], TEST_EMBED_DIM);
		ASSERT_EQ(vit.layers[i].proj_w->dims[1], TEST_EMBED_DIM);

		ASSERT(vit.layers[i].proj_b != NULL);
		ASSERT_EQ(vit.layers[i].proj_b->dims[0], TEST_EMBED_DIM);

		ASSERT(vit.layers[i].ln2_w != NULL);
		ASSERT(vit.layers[i].ln2_b != NULL);

		ASSERT(vit.layers[i].mlp_fc1_w != NULL);
		ASSERT_EQ(vit.layers[i].mlp_fc1_w->dims[0], TEST_MLP_DIM);
		ASSERT_EQ(vit.layers[i].mlp_fc1_w->dims[1],
			  TEST_EMBED_DIM);

		ASSERT(vit.layers[i].mlp_fc1_b != NULL);
		ASSERT_EQ(vit.layers[i].mlp_fc1_b->dims[0], TEST_MLP_DIM);

		ASSERT(vit.layers[i].mlp_fc2_w != NULL);
		ASSERT_EQ(vit.layers[i].mlp_fc2_w->dims[0],
			  TEST_EMBED_DIM);
		ASSERT_EQ(vit.layers[i].mlp_fc2_w->dims[1], TEST_MLP_DIM);

		ASSERT(vit.layers[i].mlp_fc2_b != NULL);
		ASSERT_EQ(vit.layers[i].mlp_fc2_b->dims[0],
			  TEST_EMBED_DIM);
	}
}

/*
 * test_vit_build_shapes - Verify output tensor shapes.
 *
 * Builds the graph with a dummy image and checks that the output
 * is [n_patches, embed_dim].
 */
static void test_vit_build_shapes(void)
{
	struct sam3_vit vit;
	enum sam3_error err;

	err = sam3_vit_init(&vit, TEST_IMG_SIZE, TEST_PATCH_SIZE,
			     TEST_EMBED_DIM, TEST_DEPTH, TEST_N_HEADS,
			     TEST_WINDOW_SIZE, TEST_MLP_DIM,
			     &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	err = sam3_vit_load(&vit, NULL, &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	/* Set layernorm gamma to 1.0 for numerical stability */
	{
		float *w;
		w = (float *)vit.ln_pre_w->data;
		for (int i = 0; i < TEST_EMBED_DIM; i++)
			w[i] = 1.0f;
	}
	for (int l = 0; l < TEST_DEPTH; l++) {
		float *w;
		w = (float *)vit.layers[l].ln1_w->data;
		for (int i = 0; i < TEST_EMBED_DIM; i++)
			w[i] = 1.0f;
		w = (float *)vit.layers[l].ln2_w->data;
		for (int i = 0; i < TEST_EMBED_DIM; i++)
			w[i] = 1.0f;
	}

	/* Create dummy input image [3, img_size, img_size] */
	int img_dims[] = {3, TEST_IMG_SIZE, TEST_IMG_SIZE};
	struct sam3_tensor *image;
	image = gh_alloc_tensor(&g_cpu.arena, SAM3_DTYPE_F32,
				 3, img_dims);
	ASSERT(image != NULL);

	/* Per-block eval: evaluates internally, returns output */
	sam3_arena_reset(&g_scratch);
	sam3_arena_reset(&g_persist);

	struct sam3_tensor *out;
	out = sam3_vit_build(&vit, &g_cpu.base, image,
			      &g_scratch, &g_persist, NULL);
	ASSERT(out != NULL);

	/* Output shape: [n_patches, embed_dim] */
	ASSERT_EQ(out->n_dims, 2);
	ASSERT_EQ(out->dims[0], TEST_N_PATCHES);
	ASSERT_EQ(out->dims[1], TEST_EMBED_DIM);
}

/*
 * test_vit_eval - Evaluate the graph on CPU.
 *
 * Builds and evaluates with zeroed weights (except layernorm gamma=1).
 * Verifies that the output values are finite (no NaN or Inf).
 */
static void test_vit_eval(void)
{
	struct sam3_vit vit;
	enum sam3_error err;

	err = sam3_vit_init(&vit, TEST_IMG_SIZE, TEST_PATCH_SIZE,
			     TEST_EMBED_DIM, TEST_DEPTH, TEST_N_HEADS,
			     TEST_WINDOW_SIZE, TEST_MLP_DIM,
			     &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	err = sam3_vit_load(&vit, NULL, &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	/*
	 * Set layer norm gamma to 1.0 so that layernorm produces
	 * finite output even with zero input (avoids 0/0 = NaN).
	 */
	for (int l = 0; l < TEST_DEPTH; l++) {
		float *w;
		w = (float *)vit.layers[l].ln1_w->data;
		for (int i = 0; i < TEST_EMBED_DIM; i++)
			w[i] = 1.0f;
		w = (float *)vit.layers[l].ln2_w->data;
		for (int i = 0; i < TEST_EMBED_DIM; i++)
			w[i] = 1.0f;
	}

	/* Create dummy input image [3, img_size, img_size] */
	int img_dims[] = {3, TEST_IMG_SIZE, TEST_IMG_SIZE};
	struct sam3_tensor *image;
	image = gh_alloc_tensor(&g_cpu.arena, SAM3_DTYPE_F32,
				 3, img_dims);
	ASSERT(image != NULL);

	/* Fill with small values to avoid numerical issues */
	float *img_data = (float *)image->data;
	int img_elems = 3 * TEST_IMG_SIZE * TEST_IMG_SIZE;
	for (int i = 0; i < img_elems; i++)
		img_data[i] = 0.01f * (float)(i % 17);

	/* Per-block eval: evaluates internally */
	sam3_arena_reset(&g_scratch);
	sam3_arena_reset(&g_persist);

	struct sam3_tensor *out;
	out = sam3_vit_build(&vit, &g_cpu.base, image,
			      &g_scratch, &g_persist, NULL);
	ASSERT(out != NULL);

	/* Verify output is finite (already evaluated by vit_build) */
	int n_out = TEST_N_PATCHES * TEST_EMBED_DIM;
	float *od = (float *)out->data;
	for (int i = 0; i < n_out; i++) {
		ASSERT(od[i] == od[i]);   /* Not NaN */
		ASSERT(od[i] < 1e10f);   /* Not huge */
		ASSERT(od[i] > -1e10f);
	}
}

/*
 * test_vit_windowed_attention - Verify windowed attention mask.
 *
 * Uses a larger config (grid_size=4, window_size=2) so that the
 * 16x16 mask has non-trivial window structure. Layer 0 is windowed,
 * layer 1 is global. Verifies mask shape, mask values at known
 * positions, and that the graph builds and evaluates without error.
 */
static void test_vit_windowed_attention(void)
{
	/* Config: grid_size=4 (56/14), window_size=2, depth=2 */
	int img_size = 56;
	int patch_size = 14;
	int embed_dim = 32;
	int depth = 2;
	int n_heads = 4;
	int window_size = 2;
	int mlp_dim = 64;
	int grid_size = img_size / patch_size;   /* 4 */
	int n_patches = grid_size * grid_size;   /* 16 */

	struct sam3_vit vit;
	enum sam3_error err;

	err = sam3_vit_init(&vit, img_size, patch_size,
			     embed_dim, depth, n_heads,
			     window_size, mlp_dim,
			     &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	/*
	 * Override is_global: layer 0 windowed, layer 1 global.
	 * (Default from init: both are non-global since depth=2
	 *  and global blocks are 7,15,23,31.)
	 */
	vit.layers[0].is_global = 0;
	vit.layers[1].is_global = 1;

	/* Window mask is lazily computed on first build */
	ASSERT(vit.window_mask == NULL);

	/* Load weights (zero-init) and build graph */
	err = sam3_vit_load(&vit, NULL, &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	/* Set layernorm gamma to 1.0 for numerical stability */
	{
		float *w;
		w = (float *)vit.ln_pre_w->data;
		for (int i = 0; i < embed_dim; i++)
			w[i] = 1.0f;
	}
	for (int l = 0; l < depth; l++) {
		float *w;
		w = (float *)vit.layers[l].ln1_w->data;
		for (int i = 0; i < embed_dim; i++)
			w[i] = 1.0f;
		w = (float *)vit.layers[l].ln2_w->data;
		for (int i = 0; i < embed_dim; i++)
			w[i] = 1.0f;
	}

	/* Create dummy input image [3, img_size, img_size] */
	int img_dims[] = {3, img_size, img_size};
	struct sam3_tensor *image;
	image = gh_alloc_tensor(&g_cpu.arena, SAM3_DTYPE_F32,
				 3, img_dims);
	ASSERT(image != NULL);

	/* Fill with small values */
	float *img_data = (float *)image->data;
	int img_elems = 3 * img_size * img_size;
	for (int i = 0; i < img_elems; i++)
		img_data[i] = 0.01f * (float)(i % 17);

	/* Per-block eval: evaluates internally */
	sam3_arena_reset(&g_scratch);
	sam3_arena_reset(&g_persist);

	struct sam3_tensor *out;
	out = sam3_vit_build(&vit, &g_cpu.base, image,
			      &g_scratch, &g_persist, NULL);
	ASSERT(out != NULL);

	/*
	 * Verify window_mask (lazily computed by first build).
	 *
	 * Grid layout (4x4), window_size=2:
	 *   Patch index = row * grid_size + col
	 *   Window = (row / ws, col / ws)
	 *
	 * Patch (0,0) = index 0:  row=0, col=0 -> window (0,0)
	 * Patch (0,1) = index 1:  row=0, col=1 -> window (0,0)
	 * Patch (0,2) = index 2:  row=0, col=2 -> window (0,1)
	 * Patch (1,0) = index 4:  row=1, col=0 -> window (0,0)
	 * Patch (1,1) = index 5:  row=1, col=1 -> window (0,0)
	 */
	ASSERT(vit.window_mask != NULL);
	ASSERT_EQ(vit.window_mask->n_dims, 2);
	ASSERT_EQ(vit.window_mask->dims[0], n_patches);
	ASSERT_EQ(vit.window_mask->dims[1], n_patches);

	float *mask_data = (float *)vit.window_mask->data;

	/* Patches 0 and 1: same window (0,0) -> 0.0f */
	ASSERT_NEAR(mask_data[0 * n_patches + 1], 0.0f, 1e-6f);

	/* Patches 0 and 4: same window (0,0) -> 0.0f */
	ASSERT_NEAR(mask_data[0 * n_patches + 4], 0.0f, 1e-6f);

	/* Patches 0 and 5: same window (0,0) -> 0.0f */
	ASSERT_NEAR(mask_data[0 * n_patches + 5], 0.0f, 1e-6f);

	/* Patches 0 and 2: different windows (0,0) vs (0,1) -> -1e9f */
	ASSERT_NEAR(mask_data[0 * n_patches + 2], -1e9f, 1.0f);

	/* Patches 0 and 8: different windows (0,0) vs (1,0) -> -1e9f */
	ASSERT_NEAR(mask_data[0 * n_patches + 8], -1e9f, 1.0f);

	/* Diagonal: always same window -> 0.0f */
	for (int i = 0; i < n_patches; i++)
		ASSERT_NEAR(mask_data[i * n_patches + i], 0.0f, 1e-6f);

	/* Output shape: [n_patches, embed_dim] */
	ASSERT_EQ(out->n_dims, 2);
	ASSERT_EQ(out->dims[0], n_patches);
	ASSERT_EQ(out->dims[1], embed_dim);

	/* Verify output is finite (already evaluated by vit_build) */
	int n_out = n_patches * embed_dim;
	float *od = (float *)out->data;
	for (int i = 0; i < n_out; i++) {
		ASSERT(od[i] == od[i]);   /* Not NaN */
		ASSERT(od[i] < 1e10f);   /* Not huge */
		ASSERT(od[i] > -1e10f);
	}
}

int main(void)
{
	setup();

	test_vit_init();
	test_vit_load();
	test_vit_build_shapes();
	test_vit_eval();
	test_vit_windowed_attention();

	teardown();

	TEST_REPORT();
}
