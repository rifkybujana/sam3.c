/*
 * tests/test_model_smoke.c - End-to-end model pipeline smoke tests
 *
 * Validates that all SAM3 model modules (encoder fusion, decoder,
 * geometry encoder, segmentation head) wire together correctly and
 * produce finite outputs. Uses small dimensions and zeroed weights
 * for fast execution -- tests graph construction and numerical
 * stability, not model accuracy.
 *
 * Key types:  sam3_encoder_fusion, sam3_decoder, sam3_geometry_encoder,
 *             sam3_seg_head
 * Depends on: test_helpers.h, model headers, backend/cpu/cpu_backend.h
 * Used by:    CTest
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "model/encoder.h"
#include "model/decoder.h"
#include "model/prompt_encoder.h"
#include "model/segmentation.h"
#include "model/graph_helpers.h"
#include "backend/cpu/cpu_backend.h"
#include "backend/backend.h"
#include "core/graph.h"
#include "core/tensor.h"

#include <string.h>

/* Small config for fast testing */
#define SMOKE_D_MODEL     32
#define SMOKE_N_HEADS     4
#define SMOKE_D_FFN       64
#define SMOKE_N_PIXELS    16  /* 4x4 grid */
#define SMOKE_SEQ_LEN     8
#define SMOKE_N_QUERIES   4
#define SMOKE_N_LAYERS    2
#define SMOKE_GRID_H      4
#define SMOKE_GRID_W      4
#define SMOKE_N_PROMPTS   3
#define SMOKE_GEOM_HEADS  1   /* d_model/32 = 32/32 = 1 */

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

/*
 * set_ones - Set all elements of a 1D tensor to 1.0f.
 *
 * Used to initialize layernorm gamma weights so that layernorm
 * produces finite output even with zero inputs (avoids 0/0 = NaN).
 */
static void set_ones(struct sam3_tensor *t)
{
	float *d = (float *)t->data;
	int n = t->dims[0];
	for (int i = 0; i < n; i++)
		d[i] = 1.0f;
}

/*
 * fill_small_values - Fill a tensor with small deterministic values.
 *
 * Populates the tensor data with small nonzero floats (0.01 * (i%p))
 * to provide a numerically stable test input.
 */
static void fill_small_values(struct sam3_tensor *t, int prime)
{
	float *d = (float *)t->data;
	int n = sam3_tensor_nelems(t);
	for (int i = 0; i < n; i++)
		d[i] = 0.01f * (float)(i % prime);
}

/*
 * check_finite - Verify all elements of a tensor are finite.
 *
 * Checks that no element is NaN and none exceeds 1e10 in magnitude.
 */
static void check_finite(struct sam3_tensor *t, const char *label)
{
	float *d = (float *)t->data;
	int n = sam3_tensor_nelems(t);
	for (int i = 0; i < n; i++) {
		if (d[i] != d[i]) { /* NaN check */
			fprintf(stderr, "NaN in %s at index %d\n",
				label, i);
			ASSERT(0);
			return;
		}
		if (d[i] > 1e10f || d[i] < -1e10f) {
			fprintf(stderr, "Inf/huge in %s at index %d: %g\n",
				label, i, (double)d[i]);
			ASSERT(0);
			return;
		}
	}
}

/* ------------------------------------------------------------------ */
/* test_encoder_decoder_pipeline                                      */
/*                                                                    */
/* Tests encoder fusion + decoder wired together end-to-end.          */
/* ------------------------------------------------------------------ */
static void test_encoder_decoder_pipeline(void)
{
	enum sam3_error err;

	/* Init encoder fusion */
	struct sam3_encoder_fusion enc;
	err = sam3_encoder_fusion_init(&enc, SMOKE_D_MODEL,
					SMOKE_N_HEADS, SMOKE_N_LAYERS,
					SMOKE_D_FFN);
	ASSERT_EQ(err, SAM3_OK);

	err = sam3_encoder_fusion_load(&enc, NULL, &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	/* Init decoder */
	struct sam3_decoder dec;
	err = sam3_decoder_init(&dec, SMOKE_D_MODEL, SMOKE_N_HEADS,
				 SMOKE_N_LAYERS, SMOKE_D_FFN,
				 SMOKE_N_QUERIES);
	ASSERT_EQ(err, SAM3_OK);

	err = sam3_decoder_load(&dec, NULL, &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	/* Set all layernorm gamma weights to 1.0 */
	for (int i = 0; i < SMOKE_N_LAYERS; i++) {
		set_ones(enc.layers[i].sa_ln_w);
		set_ones(enc.layers[i].ca_ln_w);
		set_ones(enc.layers[i].ffn_ln_w);

		set_ones(dec.layers[i].sa_ln_w);
		set_ones(dec.layers[i].ca_ln_w);
		set_ones(dec.layers[i].tca_ln_w);
		set_ones(dec.layers[i].ffn_ln_w);
	}

	/* Create dummy inputs */
	int img_dims[] = {SMOKE_N_PIXELS, SMOKE_D_MODEL};
	struct sam3_tensor *image_features;
	image_features = gh_alloc_tensor(&g_cpu.arena, SAM3_DTYPE_F32,
					  2, img_dims);
	ASSERT(image_features != NULL);
	fill_small_values(image_features, 17);

	int txt_dims[] = {SMOKE_SEQ_LEN, SMOKE_D_MODEL};
	struct sam3_tensor *text_features;
	text_features = gh_alloc_tensor(&g_cpu.arena, SAM3_DTYPE_F32,
					 2, txt_dims);
	ASSERT(text_features != NULL);
	fill_small_values(text_features, 13);

	/* Build encoder fusion graph */
	struct sam3_graph graph;
	sam3_graph_init(&graph);

	struct sam3_tensor *fused;
	fused = sam3_encoder_fusion_build(&enc, &graph,
					    image_features,
					    text_features,
					    &g_cpu.arena);
	ASSERT(fused != NULL);

	/* Verify encoder output shape */
	ASSERT_EQ(fused->n_dims, 2);
	ASSERT_EQ(fused->dims[0], SMOKE_N_PIXELS);
	ASSERT_EQ(fused->dims[1], SMOKE_D_MODEL);

	/* Build decoder graph on top of encoder output */
	struct sam3_tensor *boxes = NULL;
	struct sam3_tensor *queries;
	queries = sam3_decoder_build(&dec, &graph, fused,
				      text_features, &boxes,
				      &g_cpu.arena);
	ASSERT(queries != NULL);
	ASSERT(boxes != NULL);

	/* Verify decoder output shapes */
	ASSERT_EQ(queries->n_dims, 2);
	ASSERT_EQ(queries->dims[0], SMOKE_N_QUERIES);
	ASSERT_EQ(queries->dims[1], SMOKE_D_MODEL);
	ASSERT_EQ(boxes->n_dims, 2);
	ASSERT_EQ(boxes->dims[0], SMOKE_N_QUERIES);
	ASSERT_EQ(boxes->dims[1], 4);

	/* Evaluate on CPU */
	err = g_cpu.base.ops->graph_eval(&g_cpu.base, &graph);
	ASSERT_EQ(err, SAM3_OK);

	/* Verify finite outputs */
	check_finite(fused, "encoder_fused");
	check_finite(queries, "decoder_queries");
	check_finite(boxes, "decoder_boxes");
}

/* ------------------------------------------------------------------ */
/* test_geom_enc_pipeline                                             */
/*                                                                    */
/* Tests the geometry encoder: prompt_tokens + image_features ->      */
/* [N+1, d_model] output.                                             */
/* ------------------------------------------------------------------ */
static void test_geom_enc_pipeline(void)
{
	enum sam3_error err;

	struct sam3_geometry_encoder genc;
	err = sam3_geometry_encoder_init(&genc, SMOKE_D_MODEL,
					  SMOKE_N_LAYERS);
	ASSERT_EQ(err, SAM3_OK);

	err = sam3_geometry_encoder_load(&genc, NULL, &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	/* Set layernorm gamma to 1.0 for all layers */
	for (int i = 0; i < SMOKE_N_LAYERS; i++)
		set_ones(genc.layers[i].ca_ln_w);

	/* Create dummy prompt tokens [N_PROMPTS, d_model] */
	int prompt_dims[] = {SMOKE_N_PROMPTS, SMOKE_D_MODEL};
	struct sam3_tensor *prompt_tokens;
	prompt_tokens = gh_alloc_tensor(&g_cpu.arena, SAM3_DTYPE_F32,
					 2, prompt_dims);
	ASSERT(prompt_tokens != NULL);
	fill_small_values(prompt_tokens, 11);

	/* Create dummy image features [N_PIXELS, d_model] */
	int img_dims[] = {SMOKE_N_PIXELS, SMOKE_D_MODEL};
	struct sam3_tensor *image_features;
	image_features = gh_alloc_tensor(&g_cpu.arena, SAM3_DTYPE_F32,
					  2, img_dims);
	ASSERT(image_features != NULL);
	fill_small_values(image_features, 17);

	/* Build geometry encoder graph */
	struct sam3_graph graph;
	sam3_graph_init(&graph);

	struct sam3_tensor *geom_out;
	geom_out = sam3_geometry_encoder_build(&genc, &graph,
						prompt_tokens,
						image_features,
						NULL,
						&g_cpu.arena);
	ASSERT(geom_out != NULL);

	/* Output shape: [N_PROMPTS+1, d_model] (CLS token prepended) */
	ASSERT_EQ(geom_out->n_dims, 2);
	ASSERT_EQ(geom_out->dims[0], SMOKE_N_PROMPTS + 1);
	ASSERT_EQ(geom_out->dims[1], SMOKE_D_MODEL);

	/* Evaluate on CPU */
	err = g_cpu.base.ops->graph_eval(&g_cpu.base, &graph);
	ASSERT_EQ(err, SAM3_OK);

	/* Verify finite values */
	check_finite(geom_out, "geom_encoder_out");
}

/* ------------------------------------------------------------------ */
/* test_seg_head_pipeline                                             */
/*                                                                    */
/* Tests the segmentation head: query_embed + pixel_features ->       */
/* mask logits [n_queries, grid_h*8 * grid_w*8].                      */
/* Use grid 2x2 so output is [4, 256] = manageable size.             */
/* ------------------------------------------------------------------ */
static void test_seg_head_pipeline(void)
{
	enum sam3_error err;

	/*
	 * Use small spatial dims: enc=2×2 (0.5x), then FPN stages
	 * produce 4×4, 8×8, 16×16.
	 */
	int seg_enc_h = 4;  /* encoder output at 1× (72×72 in real model) */
	int seg_enc_w = 4;
	int seg_enc_pixels = seg_enc_h * seg_enc_w;

	struct sam3_seg_head head;
	err = sam3_seg_head_init(&head, SMOKE_D_MODEL, SMOKE_N_HEADS);
	ASSERT_EQ(err, SAM3_OK);

	err = sam3_seg_head_load(&head, NULL, &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	/* Set GroupNorm gamma to 1.0 for numerical stability */
	for (int i = 0; i < SAM3_SEG_FPN_STAGES; i++)
		set_ones(head.fpn[i].gn_w);

	/* Set prompt cross-attn norm gamma to 1.0 */
	set_ones(head.pxattn_norm_w);

	/* Create dummy query embeddings [n_queries, d_model] */
	int q_dims[] = {SMOKE_N_QUERIES, SMOKE_D_MODEL};
	struct sam3_tensor *query_embed;
	query_embed = gh_alloc_tensor(&g_cpu.arena, SAM3_DTYPE_F32,
				       2, q_dims);
	ASSERT(query_embed != NULL);
	fill_small_values(query_embed, 7);

	/* Create dummy encoder states [enc_pixels, d_model] */
	int enc_dims[] = {seg_enc_pixels, SMOKE_D_MODEL};
	struct sam3_tensor *enc_states;
	enc_states = gh_alloc_tensor(&g_cpu.arena, SAM3_DTYPE_F32,
				       2, enc_dims);
	ASSERT(enc_states != NULL);
	fill_small_values(enc_states, 13);

	/* Create backbone features at each scale (NHWC).
	 * FPN only uses feat_2x and feat_4x (2 stages). */
	int d = SMOKE_D_MODEL;
	int f2_dims[] = {1, seg_enc_h * 2, seg_enc_w * 2, d};  /* 2x */
	int f4_dims[] = {1, seg_enc_h * 4, seg_enc_w * 4, d};  /* 4x */

	struct sam3_tensor *feat_2x, *feat_4x;
	feat_2x = gh_alloc_tensor(&g_cpu.arena, SAM3_DTYPE_F32,
				    4, f2_dims);
	feat_4x = gh_alloc_tensor(&g_cpu.arena, SAM3_DTYPE_F32,
				    4, f4_dims);
	ASSERT(feat_2x != NULL);
	ASSERT(feat_4x != NULL);
	fill_small_values(feat_2x, 19);
	fill_small_values(feat_4x, 23);

	/* Build seg head graph */
	struct sam3_graph graph;
	sam3_graph_init(&graph);

	struct sam3_tensor *masks;
	masks = sam3_seg_head_build(&head, &graph, query_embed,
				     enc_states, feat_2x,
				     feat_4x, seg_enc_h, seg_enc_w,
				     &g_cpu.arena);
	ASSERT(masks != NULL);

	/* Output shape: [n_queries, feat_4x_h, feat_4x_w] */
	ASSERT_EQ(masks->n_dims, 3);
	ASSERT_EQ(masks->dims[0], SMOKE_N_QUERIES);
	ASSERT_EQ(masks->dims[1], seg_enc_h * 4);
	ASSERT_EQ(masks->dims[2], seg_enc_w * 4);

	/* Evaluate on CPU */
	err = g_cpu.base.ops->graph_eval(&g_cpu.base, &graph);
	ASSERT_EQ(err, SAM3_OK);

	/* Verify finite values (raw logits, not sigmoid) */
	check_finite(masks, "seg_head_masks");
}

/* ------------------------------------------------------------------ */
/* test_full_pipeline                                                 */
/*                                                                    */
/* Wires geometry_encoder + encoder_fusion + decoder + seg_head       */
/* together into a single graph and evaluates it end-to-end.          */
/* ------------------------------------------------------------------ */
static void test_full_pipeline(void)
{
	enum sam3_error err;

	/* Use 2x2 grid for manageable seg head output */
	int full_grid_h = 2;
	int full_grid_w = 2;
	int full_n_pixels = full_grid_h * full_grid_w; /* 4 */

	/* --- Initialize all modules --- */

	struct sam3_geometry_encoder genc;
	err = sam3_geometry_encoder_init(&genc, SMOKE_D_MODEL,
					  SMOKE_N_LAYERS);
	ASSERT_EQ(err, SAM3_OK);
	err = sam3_geometry_encoder_load(&genc, NULL, &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	struct sam3_encoder_fusion enc;
	err = sam3_encoder_fusion_init(&enc, SMOKE_D_MODEL,
					SMOKE_N_HEADS, SMOKE_N_LAYERS,
					SMOKE_D_FFN);
	ASSERT_EQ(err, SAM3_OK);
	err = sam3_encoder_fusion_load(&enc, NULL, &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	struct sam3_decoder dec;
	err = sam3_decoder_init(&dec, SMOKE_D_MODEL, SMOKE_N_HEADS,
				 SMOKE_N_LAYERS, SMOKE_D_FFN,
				 SMOKE_N_QUERIES);
	ASSERT_EQ(err, SAM3_OK);
	err = sam3_decoder_load(&dec, NULL, &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	struct sam3_seg_head seg;
	err = sam3_seg_head_init(&seg, SMOKE_D_MODEL, SMOKE_N_HEADS);
	ASSERT_EQ(err, SAM3_OK);
	err = sam3_seg_head_load(&seg, NULL, &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	/* --- Set layernorm/groupnorm gamma to 1.0 everywhere --- */

	for (int i = 0; i < SMOKE_N_LAYERS; i++) {
		set_ones(genc.layers[i].ca_ln_w);

		set_ones(enc.layers[i].sa_ln_w);
		set_ones(enc.layers[i].ca_ln_w);
		set_ones(enc.layers[i].ffn_ln_w);

		set_ones(dec.layers[i].sa_ln_w);
		set_ones(dec.layers[i].ca_ln_w);
		set_ones(dec.layers[i].tca_ln_w);
		set_ones(dec.layers[i].ffn_ln_w);
	}

	for (int i = 0; i < SAM3_SEG_FPN_STAGES; i++)
		set_ones(seg.fpn[i].gn_w);
	set_ones(seg.pxattn_norm_w);

	/* --- Create dummy inputs --- */

	int prompt_dims[] = {SMOKE_N_PROMPTS, SMOKE_D_MODEL};
	struct sam3_tensor *prompt_tokens;
	prompt_tokens = gh_alloc_tensor(&g_cpu.arena, SAM3_DTYPE_F32,
					 2, prompt_dims);
	ASSERT(prompt_tokens != NULL);
	fill_small_values(prompt_tokens, 11);

	int img_dims[] = {full_n_pixels, SMOKE_D_MODEL};
	struct sam3_tensor *image_features;
	image_features = gh_alloc_tensor(&g_cpu.arena, SAM3_DTYPE_F32,
					  2, img_dims);
	ASSERT(image_features != NULL);
	fill_small_values(image_features, 17);

	int txt_dims[] = {SMOKE_SEQ_LEN, SMOKE_D_MODEL};
	struct sam3_tensor *text_features;
	text_features = gh_alloc_tensor(&g_cpu.arena, SAM3_DTYPE_F32,
					 2, txt_dims);
	ASSERT(text_features != NULL);
	fill_small_values(text_features, 13);

	/* Backbone features (NHWC) at 2x and 4x scales */
	int d = SMOKE_D_MODEL;
	int f2_dims[] = {1, full_grid_h * 2, full_grid_w * 2, d};  /* 2x */
	int f4_dims[] = {1, full_grid_h * 4, full_grid_w * 4, d};  /* 4x */

	struct sam3_tensor *feat_2x, *feat_4x;
	feat_2x = gh_alloc_tensor(&g_cpu.arena, SAM3_DTYPE_F32,
				    4, f2_dims);
	feat_4x = gh_alloc_tensor(&g_cpu.arena, SAM3_DTYPE_F32,
				    4, f4_dims);
	ASSERT(feat_2x != NULL);
	ASSERT(feat_4x != NULL);
	fill_small_values(feat_2x, 31);
	fill_small_values(feat_4x, 37);

	/* --- Build full pipeline graph --- */

	struct sam3_graph graph;
	sam3_graph_init(&graph);

	/*
	 * Stage 1: Geometry encoder.
	 * prompt_tokens + image_features -> geom_out [N+1, d_model]
	 */
	struct sam3_tensor *geom_out;
	geom_out = sam3_geometry_encoder_build(&genc, &graph,
						prompt_tokens,
						image_features,
						NULL,
						&g_cpu.arena);
	ASSERT(geom_out != NULL);
	ASSERT_EQ(geom_out->dims[0], SMOKE_N_PROMPTS + 1);
	ASSERT_EQ(geom_out->dims[1], SMOKE_D_MODEL);

	/*
	 * Stage 2: Encoder fusion.
	 * image_features + geom_out -> fused [n_pixels, d_model]
	 *
	 * The encoder fusion takes (image_features, text_features).
	 * In the full pipeline, we pass geom_out as the "text" input
	 * since it encodes the prompt context.
	 */
	struct sam3_tensor *fused;
	fused = sam3_encoder_fusion_build(&enc, &graph,
					    image_features,
					    geom_out,
					    &g_cpu.arena);
	ASSERT(fused != NULL);
	ASSERT_EQ(fused->dims[0], full_n_pixels);
	ASSERT_EQ(fused->dims[1], SMOKE_D_MODEL);

	/*
	 * Stage 3: Decoder.
	 * fused + text_features -> queries [n_queries, d_model]
	 *                       -> boxes   [n_queries, 4]
	 */
	struct sam3_tensor *boxes = NULL;
	struct sam3_tensor *queries;
	queries = sam3_decoder_build(&dec, &graph, fused,
				      text_features, &boxes,
				      &g_cpu.arena);
	ASSERT(queries != NULL);
	ASSERT(boxes != NULL);
	ASSERT_EQ(queries->dims[0], SMOKE_N_QUERIES);
	ASSERT_EQ(queries->dims[1], SMOKE_D_MODEL);
	ASSERT_EQ(boxes->dims[0], SMOKE_N_QUERIES);
	ASSERT_EQ(boxes->dims[1], 4);

	/*
	 * Stage 4: Segmentation head.
	 * queries + fused + backbone features -> masks [n_q, H, W]
	 */
	struct sam3_tensor *masks;
	masks = sam3_seg_head_build(&seg, &graph, queries,
				     fused, feat_2x, feat_4x,
				     full_grid_h, full_grid_w,
				     &g_cpu.arena);
	ASSERT(masks != NULL);

	ASSERT_EQ(masks->n_dims, 3);
	ASSERT_EQ(masks->dims[0], SMOKE_N_QUERIES);
	ASSERT_EQ(masks->dims[1], full_grid_h * 4);
	ASSERT_EQ(masks->dims[2], full_grid_w * 4);

	/* --- Evaluate entire pipeline on CPU --- */

	err = g_cpu.base.ops->graph_eval(&g_cpu.base, &graph);
	ASSERT_EQ(err, SAM3_OK);

	/* --- Verify all outputs are finite --- */

	check_finite(geom_out, "full_geom_out");
	check_finite(fused, "full_fused");
	check_finite(queries, "full_queries");
	check_finite(boxes, "full_boxes");
	check_finite(masks, "full_masks");

	/* Box outputs should be in [0, 1] (sigmoid) */
	float *bd = (float *)boxes->data;
	int nb = sam3_tensor_nelems(boxes);
	for (int i = 0; i < nb; i++) {
		ASSERT(bd[i] >= 0.0f);
		ASSERT(bd[i] <= 1.0f);
	}
}

int main(void)
{
	setup();

	test_encoder_decoder_pipeline();
	test_geom_enc_pipeline();
	test_seg_head_pipeline();
	test_full_pipeline();

	teardown();

	TEST_REPORT();
}
