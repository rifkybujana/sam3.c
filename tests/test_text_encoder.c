/*
 * tests/test_text_encoder.c - Text encoder unit tests
 *
 * Tests the CLIP text encoder with small dimensions to verify graph
 * construction, output shapes, and numerical stability. Uses zeroed
 * weights (no weight file) so the encoder can be tested standalone.
 *
 * Key types:  sam3_text_encoder, sam3_graph, sam3_cpu_backend
 * Depends on: test_helpers.h, model/text_encoder.h,
 *             backend/cpu/cpu_backend.h
 * Used by:    CTest
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "model/text_encoder.h"
#include "model/graph_helpers.h"
#include "backend/cpu/cpu_backend.h"
#include "backend/backend.h"
#include "core/graph.h"
#include "core/tensor.h"

#include <string.h>

/* Small encoder dimensions for fast testing */
#define TEST_WIDTH      32
#define TEST_D_MODEL    16
#define TEST_N_HEADS    4
#define TEST_N_LAYERS   2
#define TEST_VOCAB_SIZE 100
#define TEST_CTX_LEN    8
#define TEST_SEQ_LEN    5

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
 * test_text_encoder_load - Verify weight loading with NULL wf.
 *
 * All weights should be allocated as zero-initialized tensors
 * with correct shapes.
 */
static void test_text_encoder_load(void)
{
	struct sam3_text_encoder te;
	memset(&te, 0, sizeof(te));
	te.d_model = TEST_D_MODEL;
	te.width = TEST_WIDTH;
	te.n_heads = TEST_N_HEADS;
	te.n_layers = TEST_N_LAYERS;
	te.context_len = TEST_CTX_LEN;
	te.vocab_size = TEST_VOCAB_SIZE;

	enum sam3_error err;
	err = sam3_text_encoder_load(&te, NULL, &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	/* Check embedding shapes */
	ASSERT(te.token_embedding != NULL);
	ASSERT_EQ(te.token_embedding->n_dims, 2);
	ASSERT_EQ(te.token_embedding->dims[0], TEST_VOCAB_SIZE);
	ASSERT_EQ(te.token_embedding->dims[1], TEST_WIDTH);

	ASSERT(te.pos_embedding != NULL);
	ASSERT_EQ(te.pos_embedding->n_dims, 2);
	ASSERT_EQ(te.pos_embedding->dims[0], TEST_CTX_LEN);
	ASSERT_EQ(te.pos_embedding->dims[1], TEST_WIDTH);

	/* Check final layer norm */
	ASSERT(te.ln_final_w != NULL);
	ASSERT_EQ(te.ln_final_w->dims[0], TEST_WIDTH);
	ASSERT(te.ln_final_b != NULL);
	ASSERT_EQ(te.ln_final_b->dims[0], TEST_WIDTH);

	/* Check text projection */
	ASSERT(te.text_projection != NULL);
	ASSERT_EQ(te.text_projection->n_dims, 2);
	ASSERT_EQ(te.text_projection->dims[0], TEST_WIDTH);
	ASSERT_EQ(te.text_projection->dims[1], TEST_D_MODEL);

	/* Check per-layer weights */
	for (int i = 0; i < TEST_N_LAYERS; i++) {
		ASSERT(te.layers[i].ln1_w != NULL);
		ASSERT(te.layers[i].ln1_b != NULL);
		ASSERT(te.layers[i].attn_q_w != NULL);
		ASSERT_EQ(te.layers[i].attn_q_w->dims[0], TEST_WIDTH);
		ASSERT_EQ(te.layers[i].attn_q_w->dims[1], TEST_WIDTH);
		ASSERT(te.layers[i].attn_q_b != NULL);
		ASSERT(te.layers[i].attn_k_w != NULL);
		ASSERT(te.layers[i].attn_k_b != NULL);
		ASSERT(te.layers[i].attn_v_w != NULL);
		ASSERT(te.layers[i].attn_v_b != NULL);
		ASSERT(te.layers[i].attn_out_w != NULL);
		ASSERT(te.layers[i].attn_out_b != NULL);
		ASSERT(te.layers[i].ln2_w != NULL);
		ASSERT(te.layers[i].ln2_b != NULL);
		ASSERT(te.layers[i].mlp_fc1_w != NULL);
		ASSERT_EQ(te.layers[i].mlp_fc1_w->dims[0],
			  4 * TEST_WIDTH);
		ASSERT(te.layers[i].mlp_fc1_b != NULL);
		ASSERT(te.layers[i].mlp_fc2_w != NULL);
		ASSERT_EQ(te.layers[i].mlp_fc2_w->dims[0], TEST_WIDTH);
		ASSERT(te.layers[i].mlp_fc2_b != NULL);
	}
}

/*
 * test_text_encoder_build_shapes - Verify output tensor shapes.
 *
 * Builds the graph with dummy token IDs and checks that the
 * per-token output is [seq_len, d_model] and pooled output
 * is [d_model].
 */
static void test_text_encoder_build_shapes(void)
{
	struct sam3_text_encoder te;
	memset(&te, 0, sizeof(te));
	te.d_model = TEST_D_MODEL;
	te.width = TEST_WIDTH;
	te.n_heads = TEST_N_HEADS;
	te.n_layers = TEST_N_LAYERS;
	te.context_len = TEST_CTX_LEN;
	te.vocab_size = TEST_VOCAB_SIZE;

	enum sam3_error err;
	err = sam3_text_encoder_load(&te, NULL, &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	/* Create dummy token IDs */
	int tok_dims[] = {TEST_SEQ_LEN};
	struct sam3_tensor *token_ids;
	token_ids = gh_alloc_tensor(&g_cpu.arena, SAM3_DTYPE_I32,
				     1, tok_dims);
	ASSERT(token_ids != NULL);

	/* Fill with valid token IDs [0..vocab_size-1] */
	int32_t *toks = (int32_t *)token_ids->data;
	for (int i = 0; i < TEST_SEQ_LEN; i++)
		toks[i] = i % TEST_VOCAB_SIZE;

	/* Build graph */
	struct sam3_graph graph;
	sam3_graph_init(&graph);

	struct sam3_tensor *pooled = NULL;
	struct sam3_tensor *out;
	out = sam3_text_encoder_build(&te, &graph, token_ids,
				       &pooled, &g_cpu.arena);
	ASSERT(out != NULL);

	/* Per-token output: [seq_len, d_model] */
	ASSERT_EQ(out->n_dims, 2);
	ASSERT_EQ(out->dims[0], TEST_SEQ_LEN);
	ASSERT_EQ(out->dims[1], TEST_D_MODEL);

	/* Pooled output: [d_model] */
	ASSERT(pooled != NULL);
	ASSERT_EQ(pooled->n_dims, 1);
	ASSERT_EQ(pooled->dims[0], TEST_D_MODEL);
}

/*
 * test_text_encoder_eval - Evaluate the graph on CPU.
 *
 * Builds and evaluates with zeroed weights. Verifies that the
 * output values are finite (no NaN or Inf).
 */
static void test_text_encoder_eval(void)
{
	struct sam3_text_encoder te;
	memset(&te, 0, sizeof(te));
	te.d_model = TEST_D_MODEL;
	te.width = TEST_WIDTH;
	te.n_heads = TEST_N_HEADS;
	te.n_layers = TEST_N_LAYERS;
	te.context_len = TEST_CTX_LEN;
	te.vocab_size = TEST_VOCAB_SIZE;

	enum sam3_error err;
	err = sam3_text_encoder_load(&te, NULL, &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	/*
	 * Set layer norm gamma to 1.0 so that layernorm produces
	 * finite output even with zero input (avoids 0/0 = NaN).
	 */
	float *w;
	w = (float *)te.ln_final_w->data;
	for (int i = 0; i < TEST_WIDTH; i++)
		w[i] = 1.0f;

	for (int l = 0; l < TEST_N_LAYERS; l++) {
		w = (float *)te.layers[l].ln1_w->data;
		for (int i = 0; i < TEST_WIDTH; i++)
			w[i] = 1.0f;
		w = (float *)te.layers[l].ln2_w->data;
		for (int i = 0; i < TEST_WIDTH; i++)
			w[i] = 1.0f;
	}

	/* Create token IDs */
	int tok_dims[] = {TEST_SEQ_LEN};
	struct sam3_tensor *token_ids;
	token_ids = gh_alloc_tensor(&g_cpu.arena, SAM3_DTYPE_I32,
				     1, tok_dims);
	ASSERT(token_ids != NULL);

	int32_t *toks = (int32_t *)token_ids->data;
	for (int i = 0; i < TEST_SEQ_LEN; i++)
		toks[i] = i;

	/* Build graph */
	struct sam3_graph graph;
	sam3_graph_init(&graph);

	struct sam3_tensor *pooled = NULL;
	struct sam3_tensor *out;
	out = sam3_text_encoder_build(&te, &graph, token_ids,
				       &pooled, &g_cpu.arena);
	ASSERT(out != NULL);
	ASSERT(pooled != NULL);

	/* Evaluate on CPU backend */
	err = g_cpu.base.ops->graph_eval(&g_cpu.base, &graph);
	ASSERT_EQ(err, SAM3_OK);

	/* Verify per-token output is finite */
	int n_out = TEST_SEQ_LEN * TEST_D_MODEL;
	float *od = (float *)out->data;
	for (int i = 0; i < n_out; i++) {
		ASSERT(od[i] == od[i]);          /* Not NaN */
		ASSERT(od[i] < 1e10f);          /* Not huge */
		ASSERT(od[i] > -1e10f);
	}

	/* Verify pooled output is finite */
	float *pd = (float *)pooled->data;
	for (int i = 0; i < TEST_D_MODEL; i++) {
		ASSERT(pd[i] == pd[i]);          /* Not NaN */
		ASSERT(pd[i] < 1e10f);
		ASSERT(pd[i] > -1e10f);
	}
}

/*
 * test_text_encoder_no_pooled - Build without requesting pooled output.
 */
static void test_text_encoder_no_pooled(void)
{
	struct sam3_text_encoder te;
	memset(&te, 0, sizeof(te));
	te.d_model = TEST_D_MODEL;
	te.width = TEST_WIDTH;
	te.n_heads = TEST_N_HEADS;
	te.n_layers = TEST_N_LAYERS;
	te.context_len = TEST_CTX_LEN;
	te.vocab_size = TEST_VOCAB_SIZE;

	enum sam3_error err;
	err = sam3_text_encoder_load(&te, NULL, &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	int tok_dims[] = {TEST_SEQ_LEN};
	struct sam3_tensor *token_ids;
	token_ids = gh_alloc_tensor(&g_cpu.arena, SAM3_DTYPE_I32,
				     1, tok_dims);
	ASSERT(token_ids != NULL);
	int32_t *toks = (int32_t *)token_ids->data;
	for (int i = 0; i < TEST_SEQ_LEN; i++)
		toks[i] = i;

	struct sam3_graph graph;
	sam3_graph_init(&graph);

	/* Pass NULL for pooled_out */
	struct sam3_tensor *out;
	out = sam3_text_encoder_build(&te, &graph, token_ids,
				       NULL, &g_cpu.arena);
	ASSERT(out != NULL);
	ASSERT_EQ(out->n_dims, 2);
	ASSERT_EQ(out->dims[0], TEST_SEQ_LEN);
	ASSERT_EQ(out->dims[1], TEST_D_MODEL);
}

int main(void)
{
	setup();

	test_text_encoder_load();
	test_text_encoder_build_shapes();
	test_text_encoder_eval();
	test_text_encoder_no_pooled();

	teardown();

	TEST_REPORT();
}
