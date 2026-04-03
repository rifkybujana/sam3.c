/*
 * tests/test_graph_helpers.c - Graph helper unit tests
 *
 * Tests the graph builder helper functions: tensor allocation,
 * linear layers, multi-head attention, and MLP construction.
 * Uses the CPU backend to evaluate built graphs and verify
 * output shapes and values.
 *
 * Key types:  sam3_graph, sam3_tensor, sam3_cpu_backend
 * Depends on: test_helpers.h, model/graph_helpers.h, backend/cpu/cpu_backend.h
 * Used by:    CTest
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "model/graph_helpers.h"
#include "backend/cpu/cpu_backend.h"
#include "backend/backend.h"
#include "core/graph.h"
#include "core/tensor.h"

#include <string.h>
#include <math.h>

#define EPS 1e-4f

/* --- Test infrastructure --- */

static struct sam3_cpu_backend g_cpu;

static void setup(void)
{
	memset(&g_cpu, 0, sizeof(g_cpu));
	g_cpu.base.type = SAM3_BACKEND_CPU;
	g_cpu.base.ops = sam3_cpu_backend_ops();
	g_cpu.arena_capacity = 64 * 1024 * 1024; /* 64 MiB for tests */
	g_cpu.base.ops->init(&g_cpu.base);
}

static void teardown(void)
{
	g_cpu.base.ops->free(&g_cpu.base);
}

static struct sam3_tensor *make_tensor(int n_dims, const int *dims)
{
	struct sam3_tensor *t = (struct sam3_tensor *)
		sam3_arena_alloc(&g_cpu.arena, sizeof(struct sam3_tensor));
	memset(t, 0, sizeof(*t));
	t->dtype = SAM3_DTYPE_F32;
	t->n_dims = n_dims;
	for (int i = 0; i < n_dims; i++)
		t->dims[i] = dims[i];
	g_cpu.base.ops->alloc_tensor(&g_cpu.base, t);
	return t;
}

static void fill_data(struct sam3_tensor *t, const float *data)
{
	memcpy(t->data, data, t->nbytes);
}

/* --- test_gh_alloc_tensor --- */

static void test_gh_alloc_tensor(void)
{
	int dims[] = {2, 3};
	struct sam3_tensor *t = gh_alloc_tensor(&g_cpu.arena,
						 SAM3_DTYPE_F32, 2, dims);

	ASSERT(t != NULL);
	ASSERT_EQ(t->n_dims, 2);
	ASSERT_EQ(t->dims[0], 2);
	ASSERT_EQ(t->dims[1], 3);
	ASSERT_EQ(t->strides[0], 3);
	ASSERT_EQ(t->strides[1], 1);
	ASSERT_EQ((int)t->nbytes, 2 * 3 * 4); /* 6 floats * 4 bytes */
	ASSERT(t->data != NULL);
}

/* --- test_gh_linear --- */

static void test_gh_linear(void)
{
	/*
	 * input: [3, 4] — 3 vectors of dim 4
	 * weight: [5, 4] — output dim 5, input dim 4
	 * bias: [5]
	 *
	 * linear(input, weight, bias) = input @ weight^T + bias
	 * output shape: [3, 5]
	 */
	struct sam3_graph graph;
	sam3_graph_init(&graph);

	int in_dims[] = {3, 4};
	int w_dims[] = {5, 4};
	int b_dims[] = {5};

	struct sam3_tensor *input = make_tensor(2, in_dims);
	struct sam3_tensor *weight = make_tensor(2, w_dims);
	struct sam3_tensor *bias = make_tensor(1, b_dims);

	/* Fill input with simple values */
	float in_data[12];
	for (int i = 0; i < 12; i++)
		in_data[i] = (float)(i + 1) * 0.1f;
	fill_data(input, in_data);

	/* Fill weight: identity-ish (first 4 cols of 5x4) */
	float w_data[20];
	for (int i = 0; i < 20; i++)
		w_data[i] = (i % 4 == i / 4) ? 1.0f : 0.0f;
	/* Row 4 (5th output neuron): all 0.5 */
	w_data[16] = 0.5f;
	w_data[17] = 0.5f;
	w_data[18] = 0.5f;
	w_data[19] = 0.5f;
	fill_data(weight, w_data);

	/* Bias: all zeros except last */
	float b_data[] = {0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
	fill_data(bias, b_data);

	struct sam3_tensor *out = gh_linear(&graph, &g_cpu.arena,
					     input, weight, bias);
	ASSERT(out != NULL);

	/* Verify output shape */
	ASSERT_EQ(out->n_dims, 2);
	ASSERT_EQ(out->dims[0], 3);
	ASSERT_EQ(out->dims[1], 5);

	/* Evaluate graph */
	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base, &graph), SAM3_OK);

	float *o = (float *)out->data;

	/*
	 * Row 0 of input: [0.1, 0.2, 0.3, 0.4]
	 * Weight rows 0-3 are identity, so outputs 0-3 copy input.
	 * Weight row 4: [0.5, 0.5, 0.5, 0.5],
	 *   dot with [0.1,0.2,0.3,0.4] = 0.5
	 * Plus bias [0,0,0,0,1]:
	 *   out[0..3] = [0.1, 0.2, 0.3, 0.4]
	 *   out[4] = 0.5 + 1.0 = 1.5
	 */
	ASSERT_NEAR(o[0], 0.1f, EPS);
	ASSERT_NEAR(o[1], 0.2f, EPS);
	ASSERT_NEAR(o[2], 0.3f, EPS);
	ASSERT_NEAR(o[3], 0.4f, EPS);
	ASSERT_NEAR(o[4], 1.5f, EPS);
}

/* --- test_gh_multihead_attention --- */

static void test_gh_multihead_attention(void)
{
	/*
	 * Small test: batch=1, seq=2, d_model=4, n_heads=2
	 * head_dim = 4/2 = 2
	 *
	 * We use identity weights so the output is predictable.
	 * The MHA flattens to 2D internally, so the output shape
	 * is [batch*seq, d_model] = [2, 4].
	 */
	struct sam3_graph graph;
	sam3_graph_init(&graph);

	int q_dims[] = {1, 2, 4};     /* [batch, seq, d_model] */
	int qkv_w_dims[] = {12, 4};   /* [3*d_model, d_model] */
	int qkv_b_dims[] = {12};      /* [3*d_model] */
	int out_w_dims[] = {4, 4};    /* [d_model, d_model] */
	int out_b_dims[] = {4};       /* [d_model] */

	struct sam3_tensor *q = make_tensor(3, q_dims);
	struct sam3_tensor *qkv_w = make_tensor(2, qkv_w_dims);
	struct sam3_tensor *qkv_b = make_tensor(1, qkv_b_dims);
	struct sam3_tensor *out_w = make_tensor(2, out_w_dims);
	struct sam3_tensor *out_b = make_tensor(1, out_b_dims);

	/* Fill Q with simple values */
	float q_data[] = {1.0f, 0.0f, 0.0f, 0.0f,
			  0.0f, 1.0f, 0.0f, 0.0f};
	fill_data(q, q_data);

	/* QKV weight: identity block for each of Q, K, V projections */
	float qkv_w_data[48];
	memset(qkv_w_data, 0, sizeof(qkv_w_data));
	for (int block = 0; block < 3; block++) {
		for (int i = 0; i < 4; i++)
			qkv_w_data[(block * 4 + i) * 4 + i] = 1.0f;
	}
	fill_data(qkv_w, qkv_w_data);

	/* QKV bias: zeros */
	float qkv_b_data[12];
	memset(qkv_b_data, 0, sizeof(qkv_b_data));
	fill_data(qkv_b, qkv_b_data);

	/* Output projection: identity */
	float out_w_data[16];
	memset(out_w_data, 0, sizeof(out_w_data));
	for (int i = 0; i < 4; i++)
		out_w_data[i * 4 + i] = 1.0f;
	fill_data(out_w, out_w_data);

	/* Output bias: zeros */
	float out_b_data[] = {0.0f, 0.0f, 0.0f, 0.0f};
	fill_data(out_b, out_b_data);

	struct sam3_tensor *out = gh_multihead_attention(
		&graph, &g_cpu.arena,
		q, q, q, /* k and v are unused (packed QKV) */
		qkv_w, qkv_b,
		out_w, out_b,
		2);

	ASSERT(out != NULL);

	/*
	 * Output is 2D [batch*seq, d_model] = [2, 4].
	 * Total elements must equal batch * seq * d_model.
	 */
	ASSERT_EQ(out->n_dims, 2);
	ASSERT_EQ(out->dims[0], 2);
	ASSERT_EQ(out->dims[1], 4);

	/* Evaluate to verify no crashes */
	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base, &graph), SAM3_OK);

	/* Output should be finite */
	float *o = (float *)out->data;
	for (int i = 0; i < 8; i++) {
		ASSERT(o[i] == o[i]);         /* Not NaN */
		ASSERT(o[i] < 1.0f / 0.0f);  /* Not +Inf */
	}
}

/* --- test_gh_mlp --- */

static void test_gh_mlp(void)
{
	/*
	 * input: [2, 4]
	 * fc1: [8, 4] (expand to hidden dim 8)
	 * fc2: [4, 8] (project back to dim 4)
	 * activation: GELU
	 * output shape: [2, 4]
	 */
	struct sam3_graph graph;
	sam3_graph_init(&graph);

	int in_dims[] = {2, 4};
	int fc1_w_dims[] = {8, 4};
	int fc1_b_dims[] = {8};
	int fc2_w_dims[] = {4, 8};
	int fc2_b_dims[] = {4};

	struct sam3_tensor *input = make_tensor(2, in_dims);
	struct sam3_tensor *fc1_w = make_tensor(2, fc1_w_dims);
	struct sam3_tensor *fc1_b = make_tensor(1, fc1_b_dims);
	struct sam3_tensor *fc2_w = make_tensor(2, fc2_w_dims);
	struct sam3_tensor *fc2_b = make_tensor(1, fc2_b_dims);

	/* Fill with small values */
	float in_data[] = {0.1f, 0.2f, 0.3f, 0.4f,
			   0.5f, 0.6f, 0.7f, 0.8f};
	fill_data(input, in_data);

	float *fc1_w_data = (float *)fc1_w->data;
	for (int i = 0; i < 32; i++)
		fc1_w_data[i] = (float)(i % 5) * 0.1f;

	float fc1_b_data[] = {0.0f, 0.0f, 0.0f, 0.0f,
			      0.0f, 0.0f, 0.0f, 0.0f};
	fill_data(fc1_b, fc1_b_data);

	float *fc2_w_data = (float *)fc2_w->data;
	for (int i = 0; i < 32; i++)
		fc2_w_data[i] = (float)(i % 3) * 0.1f;

	float fc2_b_data[] = {0.0f, 0.0f, 0.0f, 0.0f};
	fill_data(fc2_b, fc2_b_data);

	struct sam3_tensor *out = gh_mlp(&graph, &g_cpu.arena, input,
					  fc1_w, fc1_b, fc2_w, fc2_b,
					  SAM3_OP_GELU);
	ASSERT(out != NULL);

	/* Verify output shape: [2, 4] */
	ASSERT_EQ(out->n_dims, 2);
	ASSERT_EQ(out->dims[0], 2);
	ASSERT_EQ(out->dims[1], 4);

	/* Evaluate to verify numerical correctness */
	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base, &graph), SAM3_OK);

	/* Output should be finite (not NaN or Inf) */
	float *o = (float *)out->data;
	for (int i = 0; i < 8; i++) {
		ASSERT(o[i] == o[i]);           /* Not NaN */
		ASSERT(o[i] < 1.0f / 0.0f);    /* Not +Inf */
		ASSERT(o[i] > -1.0f / 0.0f);   /* Not -Inf */
	}
}

/* --- Main --- */

int main(void)
{
	setup();

	test_gh_alloc_tensor();
	test_gh_linear();
	test_gh_multihead_attention();
	test_gh_mlp();

	teardown();

	TEST_REPORT();
}
