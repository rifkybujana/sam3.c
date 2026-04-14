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
	 * RoPE tables and pos_embed are lazily computed on first
	 * sam3_vit_build() call, so they should be NULL here.
	 */
	ASSERT(vit.rope_glo_cos == NULL);
	ASSERT(vit.rope_glo_sin == NULL);
	ASSERT(vit.rope_win_local_cos == NULL);
	ASSERT(vit.rope_win_local_sin == NULL);
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

	/*
	 * Patch embedding weight. The loader permutes checkpoint
	 * OIHW [oc, 3, ps, ps] into OHWI [oc, ps, ps, 3] so the conv
	 * can use the NHWC path.
	 */
	ASSERT(vit.patch_embed_w != NULL);
	ASSERT_EQ(vit.patch_embed_w->n_dims, 4);
	ASSERT_EQ(vit.patch_embed_w->dims[0], TEST_EMBED_DIM);
	ASSERT_EQ(vit.patch_embed_w->dims[1], TEST_PATCH_SIZE);
	ASSERT_EQ(vit.patch_embed_w->dims[2], TEST_PATCH_SIZE);
	ASSERT_EQ(vit.patch_embed_w->dims[3], 3);

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
 * test_vit_window_partition - Verify partition/unpartition correctness.
 *
 * Uses grid=4, ws=2, e=3 so the result is small enough to inspect
 * by hand. Patch (py, px) carries value py*100 + px*10 + c.
 */
static void test_vit_window_partition(void)
{
	int gs = 4;
	int ws = 2;
	int e = 3;
	int np = gs * gs;	/* 16 */
	int n_win = (gs / ws) * (gs / ws); /* 4 */
	int win_pos = ws * ws;	/* 4 */

	sam3_arena_reset(&g_scratch);

	int x_dims[] = {np, e};
	struct sam3_tensor *x;
	x = gh_alloc_tensor(&g_scratch, SAM3_DTYPE_F32, 2, x_dims);
	ASSERT(x != NULL);

	float *xd = (float *)x->data;
	for (int py = 0; py < gs; py++) {
		for (int px = 0; px < gs; px++) {
			for (int c = 0; c < e; c++) {
				xd[(py * gs + px) * e + c] =
					(float)(py * 100 + px * 10 + c);
			}
		}
	}

	struct sam3_graph g;
	sam3_graph_init(&g);

	struct sam3_tensor *part;
	part = gh_window_partition(&g, &g_scratch, x, ws, gs);
	ASSERT(part != NULL);
	ASSERT_EQ(part->n_dims, 3);
	ASSERT_EQ(part->dims[0], n_win);
	ASSERT_EQ(part->dims[1], win_pos);
	ASSERT_EQ(part->dims[2], e);

	struct sam3_tensor *back;
	back = gh_window_unpartition(&g, &g_scratch, part, ws, gs);
	ASSERT(back != NULL);
	ASSERT_EQ(back->n_dims, 2);
	ASSERT_EQ(back->dims[0], np);
	ASSERT_EQ(back->dims[1], e);

	enum sam3_error err = g_cpu.base.ops->graph_eval(&g_cpu.base, &g);
	ASSERT_EQ(err, SAM3_OK);

	/*
	 * After eval, "part" should hold windows in (wy, wx, cy, cx, e)
	 * order. Window (0, 0) holds patches (0,0), (0,1), (1,0), (1,1).
	 */
	float *pd = (float *)part->data;

	/* Window 0, position 0 = patch (0, 0) */
	for (int c = 0; c < e; c++)
		ASSERT_NEAR(pd[(0 * win_pos + 0) * e + c],
			    (float)(0 * 100 + 0 * 10 + c), 1e-6f);

	/* Window 0, position 1 = patch (0, 1) */
	for (int c = 0; c < e; c++)
		ASSERT_NEAR(pd[(0 * win_pos + 1) * e + c],
			    (float)(0 * 100 + 1 * 10 + c), 1e-6f);

	/* Window 0, position 2 = patch (1, 0) */
	for (int c = 0; c < e; c++)
		ASSERT_NEAR(pd[(0 * win_pos + 2) * e + c],
			    (float)(1 * 100 + 0 * 10 + c), 1e-6f);

	/* Window 0, position 3 = patch (1, 1) */
	for (int c = 0; c < e; c++)
		ASSERT_NEAR(pd[(0 * win_pos + 3) * e + c],
			    (float)(1 * 100 + 1 * 10 + c), 1e-6f);

	/* Window 3 = (wy=1, wx=1), position 3 = patch (3, 3) */
	for (int c = 0; c < e; c++)
		ASSERT_NEAR(pd[(3 * win_pos + 3) * e + c],
			    (float)(3 * 100 + 3 * 10 + c), 1e-6f);

	/*
	 * After unpartition, every position should equal the original.
	 */
	float *bd = (float *)back->data;
	for (int i = 0; i < np * e; i++)
		ASSERT_NEAR(bd[i], xd[i], 1e-6f);
}

/*
 * test_vit_windowed_attention - Verify mask-free windowed attention.
 *
 * Uses a larger config (grid_size=4, window_size=2) so layer 0 is
 * windowed and layer 1 is global. Verifies that the small window-
 * local RoPE table is lazily built with the right shape, that its
 * first row corresponds to position 0 (cos=1, sin=0), and that the
 * graph builds and evaluates without error.
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
	int head_dim = embed_dim / n_heads;      /* 8 */

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

	/* Window-local RoPE is lazily computed on first build */
	ASSERT(vit.rope_win_local_cos == NULL);
	ASSERT(vit.rope_win_local_sin == NULL);

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
	 * Verify window-local RoPE table (lazily computed by first
	 * build). Shape is [ws*ws, head_dim/2] with positions 0..ws-1
	 * in each axis. Row 0 corresponds to position (0, 0), so all
	 * angles are zero -> cos=1, sin=0.
	 */
	ASSERT(vit.rope_win_local_cos != NULL);
	ASSERT(vit.rope_win_local_sin != NULL);
	ASSERT_EQ(vit.rope_win_local_cos->n_dims, 2);
	ASSERT_EQ(vit.rope_win_local_cos->dims[0],
		  window_size * window_size);
	ASSERT_EQ(vit.rope_win_local_cos->dims[1], head_dim / 2);
	ASSERT_EQ(vit.rope_win_local_sin->n_dims, 2);
	ASSERT_EQ(vit.rope_win_local_sin->dims[0],
		  window_size * window_size);
	ASSERT_EQ(vit.rope_win_local_sin->dims[1], head_dim / 2);

	/* Row 0 (position (0,0)) has angle 0 -> cos=1, sin=0 */
	float *rcos = (float *)vit.rope_win_local_cos->data;
	float *rsin = (float *)vit.rope_win_local_sin->data;
	for (int i = 0; i < head_dim / 2; i++) {
		ASSERT_NEAR(rcos[i], 1.0f, 1e-6f);
		ASSERT_NEAR(rsin[i], 0.0f, 1e-6f);
	}

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

/*
 * test_mha_rope_batch2 - Smoke test that MHA helper accepts batch>1.
 *
 * Uses zero-init weights so output is deterministic (zero) but
 * exercises the full reshape/permute/SDPA path with batch=2.
 */
static void test_mha_rope_batch2(void)
{
	int batch = 2;
	int seq = 4;
	int e = 8;
	int n_heads = 2;
	int hd = e / n_heads;

	sam3_arena_reset(&g_scratch);

	int x_dims[] = {batch, seq, e};
	struct sam3_tensor *x = gh_alloc_tensor(&g_scratch,
		SAM3_DTYPE_F32, 3, x_dims);
	ASSERT(x != NULL);

	int qkv_w_dims[] = {3 * e, e};
	int qkv_b_dims[] = {3 * e};
	int o_w_dims[] = {e, e};
	int o_b_dims[] = {e};

	struct sam3_tensor *qkv_w = gh_alloc_tensor(&g_scratch,
		SAM3_DTYPE_F32, 2, qkv_w_dims);
	struct sam3_tensor *qkv_b = gh_alloc_tensor(&g_scratch,
		SAM3_DTYPE_F32, 1, qkv_b_dims);
	struct sam3_tensor *o_w = gh_alloc_tensor(&g_scratch,
		SAM3_DTYPE_F32, 2, o_w_dims);
	struct sam3_tensor *o_b = gh_alloc_tensor(&g_scratch,
		SAM3_DTYPE_F32, 1, o_b_dims);
	ASSERT(qkv_w && qkv_b && o_w && o_b);

	int rope_dims[] = {seq, hd / 2};
	struct sam3_tensor *rcos = gh_alloc_tensor(&g_scratch,
		SAM3_DTYPE_F32, 2, rope_dims);
	struct sam3_tensor *rsin = gh_alloc_tensor(&g_scratch,
		SAM3_DTYPE_F32, 2, rope_dims);
	ASSERT(rcos && rsin);

	/* cos = 1.0, sin = 0.0 -> RoPE is identity */
	float *cd = (float *)rcos->data;
	for (int i = 0; i < seq * hd / 2; i++)
		cd[i] = 1.0f;

	struct sam3_graph g;
	sam3_graph_init(&g);

	struct sam3_tensor *out;
	out = gh_multihead_attention_rope(&g, &g_scratch, x,
					  NULL, NULL,
					  qkv_w, qkv_b, o_w, o_b,
					  n_heads, rcos, rsin, NULL,
					  0, 0.0f);
	ASSERT(out != NULL);
	/* Output is [batch*seq, e] */
	ASSERT_EQ(out->n_dims, 2);
	ASSERT_EQ(out->dims[0], batch * seq);
	ASSERT_EQ(out->dims[1], e);

	enum sam3_error err = g_cpu.base.ops->graph_eval(&g_cpu.base, &g);
	ASSERT_EQ(err, SAM3_OK);

	/* All-zero weights -> all-zero output. */
	float *od = (float *)out->data;
	for (int i = 0; i < batch * seq * e; i++)
		ASSERT_NEAR(od[i], 0.0f, 1e-6f);
}

int main(void)
{
	setup();

	test_vit_init();
	test_vit_load();
	test_vit_build_shapes();
	test_vit_eval();
	test_vit_window_partition();
	test_vit_windowed_attention();
	test_mha_rope_batch2();

	teardown();

	TEST_REPORT();
}
