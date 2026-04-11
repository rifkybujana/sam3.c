/*
 * tests/test_cpu_kernels.c - CPU kernel unit tests
 *
 * Tests all CPU compute kernels: matmul, add, mul, relu, gelu,
 * softmax, layernorm, conv2d, reshape, transpose. Uses the backend
 * arena for tensor allocation and ASSERT_NEAR for float comparisons.
 *
 * Key types:  sam3_node, sam3_tensor, sam3_cpu_backend
 * Depends on: test_helpers.h, cpu_backend.h, core/graph.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "backend/cpu/cpu_backend.h"
#include "backend/backend.h"
#include "core/graph.h"
#include "core/tensor.h"
#ifdef SAM3_HAS_PROFILE
#include "util/profile.h"
#endif

#include <string.h>

#define EPS 1e-4f

/* --- Helpers --- */

static struct sam3_cpu_backend g_cpu;

static void setup(void)
{
	memset(&g_cpu, 0, sizeof(g_cpu));
	g_cpu.base.type = SAM3_BACKEND_CPU;
	g_cpu.base.ops = sam3_cpu_backend_ops();
	g_cpu.arena_capacity = 16 * 1024 * 1024; /* 16 MiB for tests */
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

static struct sam3_node make_node(enum sam3_op op,
				  struct sam3_tensor **inputs, int n_inputs,
				  struct sam3_tensor *output)
{
	struct sam3_node node;
	memset(&node, 0, sizeof(node));
	node.op = op;
	node.n_inputs = n_inputs;
	for (int i = 0; i < n_inputs; i++)
		node.inputs[i] = inputs[i];
	node.output = output;
	return node;
}

/* --- Add tests --- */

static void test_add_basic(void)
{
	int dims[] = {4};
	struct sam3_tensor *a = make_tensor(1, dims);
	struct sam3_tensor *b = make_tensor(1, dims);
	struct sam3_tensor *c = make_tensor(1, dims);

	float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
	float b_data[] = {10.0f, 20.0f, 30.0f, 40.0f};
	fill_data(a, a_data);
	fill_data(b, b_data);

	struct sam3_tensor *inputs[] = {a, b};
	struct sam3_node node = make_node(SAM3_OP_ADD, inputs, 2, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	float *out = (float *)c->data;
	ASSERT_NEAR(out[0], 11.0f, EPS);
	ASSERT_NEAR(out[1], 22.0f, EPS);
	ASSERT_NEAR(out[2], 33.0f, EPS);
	ASSERT_NEAR(out[3], 44.0f, EPS);
}

static void test_add_broadcast(void)
{
	int a_dims[] = {2, 3};
	int b_dims[] = {3};
	int c_dims[] = {2, 3};
	struct sam3_tensor *a = make_tensor(2, a_dims);
	struct sam3_tensor *b = make_tensor(1, b_dims);
	struct sam3_tensor *c = make_tensor(2, c_dims);

	float a_data[] = {1, 2, 3, 4, 5, 6};
	float b_data[] = {10, 20, 30};
	fill_data(a, a_data);
	fill_data(b, b_data);

	struct sam3_tensor *inputs[] = {a, b};
	struct sam3_node node = make_node(SAM3_OP_ADD, inputs, 2, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	float *out = (float *)c->data;
	ASSERT_NEAR(out[0], 11.0f, EPS);
	ASSERT_NEAR(out[1], 22.0f, EPS);
	ASSERT_NEAR(out[2], 33.0f, EPS);
	ASSERT_NEAR(out[3], 14.0f, EPS);
	ASSERT_NEAR(out[4], 25.0f, EPS);
	ASSERT_NEAR(out[5], 36.0f, EPS);
}

static void test_add_large(void)
{
	int dims[] = {256};
	struct sam3_tensor *a = make_tensor(1, dims);
	struct sam3_tensor *b = make_tensor(1, dims);
	struct sam3_tensor *c = make_tensor(1, dims);

	float *ad = (float *)a->data;
	float *bd = (float *)b->data;
	for (int i = 0; i < 256; i++) {
		ad[i] = (float)i;
		bd[i] = (float)(256 - i);
	}

	struct sam3_tensor *inputs[] = {a, b};
	struct sam3_node node = make_node(SAM3_OP_ADD, inputs, 2, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	float *out = (float *)c->data;
	for (int i = 0; i < 256; i++)
		ASSERT_NEAR(out[i], 256.0f, EPS);
}

/* --- Mul tests --- */

static void test_mul_basic(void)
{
	int dims[] = {4};
	struct sam3_tensor *a = make_tensor(1, dims);
	struct sam3_tensor *b = make_tensor(1, dims);
	struct sam3_tensor *c = make_tensor(1, dims);

	float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
	float b_data[] = {2.0f, 3.0f, 4.0f, 5.0f};
	fill_data(a, a_data);
	fill_data(b, b_data);

	struct sam3_tensor *inputs[] = {a, b};
	struct sam3_node node = make_node(SAM3_OP_MUL, inputs, 2, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	float *out = (float *)c->data;
	ASSERT_NEAR(out[0], 2.0f, EPS);
	ASSERT_NEAR(out[1], 6.0f, EPS);
	ASSERT_NEAR(out[2], 12.0f, EPS);
	ASSERT_NEAR(out[3], 20.0f, EPS);
}

static void test_mul_broadcast(void)
{
	int a_dims[] = {2, 3};
	int b_dims[] = {3};
	int c_dims[] = {2, 3};
	struct sam3_tensor *a = make_tensor(2, a_dims);
	struct sam3_tensor *b = make_tensor(1, b_dims);
	struct sam3_tensor *c = make_tensor(2, c_dims);

	float a_data[] = {1, 2, 3, 4, 5, 6};
	float b_data[] = {2, 3, 4};
	fill_data(a, a_data);
	fill_data(b, b_data);

	struct sam3_tensor *inputs[] = {a, b};
	struct sam3_node node = make_node(SAM3_OP_MUL, inputs, 2, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	float *out = (float *)c->data;
	ASSERT_NEAR(out[0], 2.0f, EPS);
	ASSERT_NEAR(out[1], 6.0f, EPS);
	ASSERT_NEAR(out[2], 12.0f, EPS);
	ASSERT_NEAR(out[3], 8.0f, EPS);
	ASSERT_NEAR(out[4], 15.0f, EPS);
	ASSERT_NEAR(out[5], 24.0f, EPS);
}

/* --- ReLU tests --- */

static void test_relu_basic(void)
{
	int dims[] = {6};
	struct sam3_tensor *a = make_tensor(1, dims);
	struct sam3_tensor *c = make_tensor(1, dims);

	float a_data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
	fill_data(a, a_data);

	struct sam3_tensor *inputs[] = {a};
	struct sam3_node node = make_node(SAM3_OP_RELU, inputs, 1, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	float *out = (float *)c->data;
	ASSERT_NEAR(out[0], 0.0f, EPS);
	ASSERT_NEAR(out[1], 0.0f, EPS);
	ASSERT_NEAR(out[2], 0.0f, EPS);
	ASSERT_NEAR(out[3], 1.0f, EPS);
	ASSERT_NEAR(out[4], 2.0f, EPS);
	ASSERT_NEAR(out[5], 3.0f, EPS);
}

static void test_relu_all_negative(void)
{
	int dims[] = {4};
	struct sam3_tensor *a = make_tensor(1, dims);
	struct sam3_tensor *c = make_tensor(1, dims);

	float a_data[] = {-5.0f, -3.0f, -1.0f, -0.1f};
	fill_data(a, a_data);

	struct sam3_tensor *inputs[] = {a};
	struct sam3_node node = make_node(SAM3_OP_RELU, inputs, 1, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	float *out = (float *)c->data;
	for (int i = 0; i < 4; i++)
		ASSERT_NEAR(out[i], 0.0f, EPS);
}

/* --- GELU tests --- */

static void test_gelu_basic(void)
{
	int dims[] = {4};
	struct sam3_tensor *a = make_tensor(1, dims);
	struct sam3_tensor *c = make_tensor(1, dims);

	float a_data[] = {0.0f, 1.0f, -1.0f, 2.0f};
	fill_data(a, a_data);

	struct sam3_tensor *inputs[] = {a};
	struct sam3_node node = make_node(SAM3_OP_GELU, inputs, 1, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	float *out = (float *)c->data;
	/* GELU(0) = 0 */
	ASSERT_NEAR(out[0], 0.0f, EPS);
	/* GELU(1) ~ 0.8412 */
	ASSERT_NEAR(out[1], 0.8412f, 1e-3f);
	/* GELU(-1) ~ -0.1588 */
	ASSERT_NEAR(out[2], -0.1588f, 1e-3f);
	/* GELU(2) ~ 1.9545 */
	ASSERT_NEAR(out[3], 1.9545f, 1e-3f);
}

static void test_gelu_zero(void)
{
	int dims[] = {1};
	struct sam3_tensor *a = make_tensor(1, dims);
	struct sam3_tensor *c = make_tensor(1, dims);

	float a_data[] = {0.0f};
	fill_data(a, a_data);

	struct sam3_tensor *inputs[] = {a};
	struct sam3_node node = make_node(SAM3_OP_GELU, inputs, 1, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	ASSERT_NEAR(((float *)c->data)[0], 0.0f, EPS);
}

/* --- Softmax tests --- */

static void test_softmax_basic(void)
{
	int dims[] = {4};
	struct sam3_tensor *a = make_tensor(1, dims);
	struct sam3_tensor *c = make_tensor(1, dims);

	float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
	fill_data(a, a_data);

	struct sam3_tensor *inputs[] = {a};
	struct sam3_node node = make_node(SAM3_OP_SOFTMAX, inputs, 1, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	float *out = (float *)c->data;

	/* Verify sum = 1 */
	float sum = 0.0f;
	for (int i = 0; i < 4; i++)
		sum += out[i];
	ASSERT_NEAR(sum, 1.0f, EPS);

	/* Values should be increasing */
	ASSERT(out[0] < out[1]);
	ASSERT(out[1] < out[2]);
	ASSERT(out[2] < out[3]);
}

static void test_softmax_2d(void)
{
	int dims[] = {2, 3};
	struct sam3_tensor *a = make_tensor(2, dims);
	struct sam3_tensor *c = make_tensor(2, dims);

	float a_data[] = {1, 2, 3, 4, 5, 6};
	fill_data(a, a_data);

	struct sam3_tensor *inputs[] = {a};
	struct sam3_node node = make_node(SAM3_OP_SOFTMAX, inputs, 1, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	float *out = (float *)c->data;

	/* Each row should sum to 1 */
	float sum0 = out[0] + out[1] + out[2];
	float sum1 = out[3] + out[4] + out[5];
	ASSERT_NEAR(sum0, 1.0f, EPS);
	ASSERT_NEAR(sum1, 1.0f, EPS);
}

static void test_softmax_uniform(void)
{
	int dims[] = {4};
	struct sam3_tensor *a = make_tensor(1, dims);
	struct sam3_tensor *c = make_tensor(1, dims);

	float a_data[] = {1.0f, 1.0f, 1.0f, 1.0f};
	fill_data(a, a_data);

	struct sam3_tensor *inputs[] = {a};
	struct sam3_node node = make_node(SAM3_OP_SOFTMAX, inputs, 1, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	float *out = (float *)c->data;
	for (int i = 0; i < 4; i++)
		ASSERT_NEAR(out[i], 0.25f, EPS);
}

/* --- LayerNorm tests --- */

static void test_layernorm_no_affine(void)
{
	int dims[] = {4};
	struct sam3_tensor *a = make_tensor(1, dims);
	struct sam3_tensor *c = make_tensor(1, dims);

	float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
	fill_data(a, a_data);

	struct sam3_tensor *inputs[] = {a};
	struct sam3_node node = make_node(SAM3_OP_LAYERNORM, inputs, 1, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	float *out = (float *)c->data;

	/* Mean should be ~0 */
	float mean = 0.0f;
	for (int i = 0; i < 4; i++)
		mean += out[i];
	mean /= 4.0f;
	ASSERT_NEAR(mean, 0.0f, EPS);

	/* Variance should be ~1 */
	float var = 0.0f;
	for (int i = 0; i < 4; i++)
		var += out[i] * out[i];
	var /= 4.0f;
	ASSERT_NEAR(var, 1.0f, 1e-3f);
}

static void test_layernorm_with_affine(void)
{
	int dims[] = {4};
	struct sam3_tensor *a = make_tensor(1, dims);
	struct sam3_tensor *gamma = make_tensor(1, dims);
	struct sam3_tensor *beta = make_tensor(1, dims);
	struct sam3_tensor *c = make_tensor(1, dims);

	float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
	float g_data[] = {2.0f, 2.0f, 2.0f, 2.0f};
	float b_data[] = {1.0f, 1.0f, 1.0f, 1.0f};
	fill_data(a, a_data);
	fill_data(gamma, g_data);
	fill_data(beta, b_data);

	struct sam3_tensor *inputs[] = {a, gamma, beta};
	struct sam3_node node = make_node(SAM3_OP_LAYERNORM, inputs, 3, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	float *out = (float *)c->data;

	/* Mean of output should be ~1 (beta) since norm has mean 0 */
	float mean = 0.0f;
	for (int i = 0; i < 4; i++)
		mean += out[i];
	mean /= 4.0f;
	ASSERT_NEAR(mean, 1.0f, 1e-3f);
}

static void test_layernorm_2d(void)
{
	int dims[] = {2, 4};
	struct sam3_tensor *a = make_tensor(2, dims);
	struct sam3_tensor *c = make_tensor(2, dims);

	float a_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
	fill_data(a, a_data);

	struct sam3_tensor *inputs[] = {a};
	struct sam3_node node = make_node(SAM3_OP_LAYERNORM, inputs, 1, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	float *out = (float *)c->data;

	/* Each row should have mean ~0 */
	float mean0 = (out[0] + out[1] + out[2] + out[3]) / 4.0f;
	float mean1 = (out[4] + out[5] + out[6] + out[7]) / 4.0f;
	ASSERT_NEAR(mean0, 0.0f, EPS);
	ASSERT_NEAR(mean1, 0.0f, EPS);
}

/* --- Matmul tests --- */

static void test_matmul_2x3_3x2(void)
{
	int a_dims[] = {2, 3};
	int b_dims[] = {3, 2};
	int c_dims[] = {2, 2};
	struct sam3_tensor *a = make_tensor(2, a_dims);
	struct sam3_tensor *b = make_tensor(2, b_dims);
	struct sam3_tensor *c = make_tensor(2, c_dims);

	float a_data[] = {1, 2, 3, 4, 5, 6};
	float b_data[] = {7, 8, 9, 10, 11, 12};
	fill_data(a, a_data);
	fill_data(b, b_data);

	struct sam3_tensor *inputs[] = {a, b};
	struct sam3_node node = make_node(SAM3_OP_MATMUL, inputs, 2, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	float *out = (float *)c->data;
	/* [1,2,3]@[7,9,11;8,10,12] = [58,64;139,154] */
	ASSERT_NEAR(out[0], 58.0f, EPS);
	ASSERT_NEAR(out[1], 64.0f, EPS);
	ASSERT_NEAR(out[2], 139.0f, EPS);
	ASSERT_NEAR(out[3], 154.0f, EPS);
}

static void test_matmul_identity(void)
{
	int dims[] = {3, 3};
	struct sam3_tensor *a = make_tensor(2, dims);
	struct sam3_tensor *eye = make_tensor(2, dims);
	struct sam3_tensor *c = make_tensor(2, dims);

	float a_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	float eye_data[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
	fill_data(a, a_data);
	fill_data(eye, eye_data);

	struct sam3_tensor *inputs[] = {a, eye};
	struct sam3_node node = make_node(SAM3_OP_MATMUL, inputs, 2, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	float *out = (float *)c->data;
	for (int i = 0; i < 9; i++)
		ASSERT_NEAR(out[i], a_data[i], EPS);
}

static void test_matmul_large(void)
{
	int a_dims[] = {16, 32};
	int b_dims[] = {32, 16};
	int c_dims[] = {16, 16};
	struct sam3_tensor *a = make_tensor(2, a_dims);
	struct sam3_tensor *b = make_tensor(2, b_dims);
	struct sam3_tensor *c = make_tensor(2, c_dims);

	/* Fill with known pattern */
	float *ad = (float *)a->data;
	float *bd = (float *)b->data;
	for (int i = 0; i < 16 * 32; i++)
		ad[i] = (float)(i % 7) * 0.1f;
	for (int i = 0; i < 32 * 16; i++)
		bd[i] = (float)(i % 5) * 0.1f;

	struct sam3_tensor *inputs[] = {a, b};
	struct sam3_node node = make_node(SAM3_OP_MATMUL, inputs, 2, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	/* Verify first element manually: row 0 of a dot col 0 of b */
	float expected = 0.0f;
	for (int k = 0; k < 32; k++)
		expected += ad[k] * bd[k * 16];
	ASSERT_NEAR(((float *)c->data)[0], expected, 1e-3f);
}

/* --- Reshape tests --- */

static void test_reshape_basic(void)
{
	int in_dims[] = {2, 3};
	int out_dims[] = {6};
	struct sam3_tensor *a = make_tensor(2, in_dims);
	struct sam3_tensor *c = make_tensor(1, out_dims);

	float a_data[] = {1, 2, 3, 4, 5, 6};
	fill_data(a, a_data);

	struct sam3_tensor *inputs[] = {a};
	struct sam3_node node = make_node(SAM3_OP_RESHAPE, inputs, 1, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	/* Zero-copy: data pointer should be same */
	ASSERT(c->data == a->data);

	float *out = (float *)c->data;
	for (int i = 0; i < 6; i++)
		ASSERT_NEAR(out[i], a_data[i], EPS);
}

static void test_reshape_3d_to_2d(void)
{
	int in_dims[] = {2, 3, 4};
	int out_dims[] = {6, 4};
	struct sam3_tensor *a = make_tensor(3, in_dims);
	struct sam3_tensor *c = make_tensor(2, out_dims);

	/* Just verify it succeeds and data pointer is aliased */
	struct sam3_tensor *inputs[] = {a};
	struct sam3_node node = make_node(SAM3_OP_RESHAPE, inputs, 1, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);
	ASSERT(c->data == a->data);
}

static void test_reshape_size_mismatch(void)
{
	int in_dims[] = {2, 3};
	int out_dims[] = {5};
	struct sam3_tensor *a = make_tensor(2, in_dims);
	struct sam3_tensor *c = make_tensor(1, out_dims);

	struct sam3_tensor *inputs[] = {a};
	struct sam3_node node = make_node(SAM3_OP_RESHAPE, inputs, 1, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_EINVAL);
}

/* --- Transpose tests --- */

static void test_transpose_basic(void)
{
	int in_dims[] = {2, 3};
	int out_dims[] = {3, 2};
	struct sam3_tensor *a = make_tensor(2, in_dims);
	struct sam3_tensor *c = make_tensor(2, out_dims);

	float a_data[] = {1, 2, 3, 4, 5, 6};
	fill_data(a, a_data);

	struct sam3_tensor *inputs[] = {a};
	struct sam3_node node = make_node(SAM3_OP_TRANSPOSE, inputs, 1, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	float *out = (float *)c->data;
	/* Transposed: [[1,4],[2,5],[3,6]] */
	ASSERT_NEAR(out[0], 1.0f, EPS);
	ASSERT_NEAR(out[1], 4.0f, EPS);
	ASSERT_NEAR(out[2], 2.0f, EPS);
	ASSERT_NEAR(out[3], 5.0f, EPS);
	ASSERT_NEAR(out[4], 3.0f, EPS);
	ASSERT_NEAR(out[5], 6.0f, EPS);
}

static void test_transpose_square(void)
{
	int dims[] = {4, 4};
	struct sam3_tensor *a = make_tensor(2, dims);
	struct sam3_tensor *c = make_tensor(2, dims);

	float a_data[16];
	for (int i = 0; i < 16; i++)
		a_data[i] = (float)i;
	fill_data(a, a_data);

	struct sam3_tensor *inputs[] = {a};
	struct sam3_node node = make_node(SAM3_OP_TRANSPOSE, inputs, 1, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	float *out = (float *)c->data;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			ASSERT_NEAR(out[j * 4 + i], a_data[i * 4 + j], EPS);
}

static void test_transpose_large(void)
{
	int in_dims[] = {8, 16};
	int out_dims[] = {16, 8};
	struct sam3_tensor *a = make_tensor(2, in_dims);
	struct sam3_tensor *c = make_tensor(2, out_dims);

	float *ad = (float *)a->data;
	for (int i = 0; i < 128; i++)
		ad[i] = (float)i;

	struct sam3_tensor *inputs[] = {a};
	struct sam3_node node = make_node(SAM3_OP_TRANSPOSE, inputs, 1, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	float *out = (float *)c->data;
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 16; j++)
			ASSERT_NEAR(out[j * 8 + i], ad[i * 16 + j], EPS);
}

/* --- Conv2D tests --- */

static void test_conv2d_basic(void)
{
	/*
	 * NHWC input [N,H,W,C] = [1,4,4,1],
	 * OHWI weight [OC,KH,KW,IC] = [1,2,2,1],
	 * NHWC output = [1,3,3,1]. Stride=1, pad=0.
	 */
	int in_dims[] = {1, 4, 4, 1};
	int w_dims[] = {1, 2, 2, 1};
	int out_dims[] = {1, 3, 3, 1};

	struct sam3_tensor *input = make_tensor(4, in_dims);
	struct sam3_tensor *weight = make_tensor(4, w_dims);
	struct sam3_tensor *output = make_tensor(4, out_dims);

	float in_data[16];
	for (int i = 0; i < 16; i++)
		in_data[i] = (float)(i + 1);
	fill_data(input, in_data);

	float w_data[] = {1, 0, 0, 1};
	fill_data(weight, w_data);

	struct sam3_tensor *inputs[] = {input, weight};
	struct sam3_node node = make_node(SAM3_OP_CONV2D, inputs, 2, output);
	node.params[0] = 1; /* stride */
	node.params[1] = 0; /* padding */

	struct sam3_graph g;
	memset(&g, 0, sizeof(g));
	g.nodes[0] = node;
	g.n_nodes = 1;

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base, &g), SAM3_OK);

	float *out = (float *)output->data;
	/* With kernel [1,0;0,1]: out[0,0] = in[0,0]*1 + in[0,1]*0 + in[1,0]*0 + in[1,1]*1 */
	/* = 1 + 6 = 7 */
	ASSERT_NEAR(out[0], 7.0f, EPS);
}

static void test_conv2d_with_padding(void)
{
	/*
	 * NHWC input [1,3,3,1], OHWI weight [1,3,3,1],
	 * NHWC output [1,3,3,1]. Stride=1, pad=1.
	 */
	int in_dims[] = {1, 3, 3, 1};
	int w_dims[] = {1, 3, 3, 1};
	int out_dims[] = {1, 3, 3, 1};

	struct sam3_tensor *input = make_tensor(4, in_dims);
	struct sam3_tensor *weight = make_tensor(4, w_dims);
	struct sam3_tensor *output = make_tensor(4, out_dims);

	float in_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	fill_data(input, in_data);

	/* All-ones kernel */
	float w_data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
	fill_data(weight, w_data);

	struct sam3_tensor *inputs[] = {input, weight};
	struct sam3_node node = make_node(SAM3_OP_CONV2D, inputs, 2, output);
	node.params[0] = 1;
	node.params[1] = 1;

	struct sam3_graph g;
	memset(&g, 0, sizeof(g));
	g.nodes[0] = node;
	g.n_nodes = 1;

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base, &g), SAM3_OK);

	float *out = (float *)output->data;
	/* Center element: sum of all = 45 */
	ASSERT_NEAR(out[4], 45.0f, EPS);
	/* Corner (0,0): 1+2+4+5 = 12 */
	ASSERT_NEAR(out[0], 12.0f, EPS);
}

static void test_conv2d_stride2(void)
{
	/*
	 * NHWC input [1,4,4,1], OHWI weight [1,2,2,1],
	 * NHWC output [1,2,2,1]. Stride=2, pad=0.
	 */
	int in_dims[] = {1, 4, 4, 1};
	int w_dims[] = {1, 2, 2, 1};
	int out_dims[] = {1, 2, 2, 1};

	struct sam3_tensor *input = make_tensor(4, in_dims);
	struct sam3_tensor *weight = make_tensor(4, w_dims);
	struct sam3_tensor *output = make_tensor(4, out_dims);

	float in_data[16];
	for (int i = 0; i < 16; i++)
		in_data[i] = 1.0f;
	fill_data(input, in_data);

	float w_data[] = {1, 1, 1, 1};
	fill_data(weight, w_data);

	struct sam3_tensor *inputs[] = {input, weight};
	struct sam3_node node = make_node(SAM3_OP_CONV2D, inputs, 2, output);
	node.params[0] = 2;
	node.params[1] = 0;

	struct sam3_graph g;
	memset(&g, 0, sizeof(g));
	g.nodes[0] = node;
	g.n_nodes = 1;

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base, &g), SAM3_OK);

	float *out = (float *)output->data;
	/* All-ones input with all-ones 2x2 kernel = 4 everywhere */
	for (int i = 0; i < 4; i++)
		ASSERT_NEAR(out[i], 4.0f, EPS);
}

/* --- Graph eval multi-node test --- */

static void test_graph_multi_node(void)
{
	/* Test: a + b, then relu the result */
	int dims[] = {4};
	struct sam3_tensor *a = make_tensor(1, dims);
	struct sam3_tensor *b = make_tensor(1, dims);
	struct sam3_tensor *sum = make_tensor(1, dims);
	struct sam3_tensor *out = make_tensor(1, dims);

	float a_data[] = {-3.0f, -1.0f, 1.0f, 3.0f};
	float b_data[] = {1.0f, -1.0f, -1.0f, 1.0f};
	fill_data(a, a_data);
	fill_data(b, b_data);

	struct sam3_graph g;
	memset(&g, 0, sizeof(g));

	struct sam3_tensor *add_inputs[] = {a, b};
	g.nodes[0] = make_node(SAM3_OP_ADD, add_inputs, 2, sum);

	struct sam3_tensor *relu_inputs[] = {sum};
	g.nodes[1] = make_node(SAM3_OP_RELU, relu_inputs, 1, out);
	g.n_nodes = 2;

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base, &g), SAM3_OK);

	float *o = (float *)out->data;
	/* a+b = [-2, -2, 0, 4], relu = [0, 0, 0, 4] */
	ASSERT_NEAR(o[0], 0.0f, EPS);
	ASSERT_NEAR(o[1], 0.0f, EPS);
	ASSERT_NEAR(o[2], 0.0f, EPS);
	ASSERT_NEAR(o[3], 4.0f, EPS);
}

/* --- Error handling tests --- */

static void test_error_null_tensor(void)
{
	struct sam3_node node;
	memset(&node, 0, sizeof(node));
	node.op = SAM3_OP_ADD;
	node.n_inputs = 2;
	/* inputs are NULL */

	struct sam3_graph g = {.n_nodes = 1};
	g.nodes[0] = node;

	ASSERT(g_cpu.base.ops->graph_eval(&g_cpu.base, &g) != SAM3_OK);
}

static void test_error_dtype(void)
{
	int dims[] = {4};
	struct sam3_tensor *a = make_tensor(1, dims);
	struct sam3_tensor *c = make_tensor(1, dims);

	a->dtype = SAM3_DTYPE_I32; /* Not F32 */

	struct sam3_tensor *inputs[] = {a};
	struct sam3_node node = make_node(SAM3_OP_RELU, inputs, 1, c);

	struct sam3_graph g = {.n_nodes = 1};
	g.nodes[0] = node;

	ASSERT(g_cpu.base.ops->graph_eval(&g_cpu.base, &g) != SAM3_OK);
}

/* --- Mul SIMD test --- */

static void test_mul_large(void)
{
	int dims[] = {256};
	struct sam3_tensor *a = make_tensor(1, dims);
	struct sam3_tensor *b = make_tensor(1, dims);
	struct sam3_tensor *c = make_tensor(1, dims);

	float *ad = (float *)a->data;
	float *bd = (float *)b->data;
	for (int i = 0; i < 256; i++) {
		ad[i] = (float)(i + 1);
		bd[i] = 2.0f;
	}

	struct sam3_tensor *inputs[] = {a, b};
	struct sam3_node node = make_node(SAM3_OP_MUL, inputs, 2, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	float *out = (float *)c->data;
	for (int i = 0; i < 256; i++)
		ASSERT_NEAR(out[i], (float)((i + 1) * 2), EPS);
}

/* --- ReLU SIMD test --- */

static void test_relu_large(void)
{
	int dims[] = {256};
	struct sam3_tensor *a = make_tensor(1, dims);
	struct sam3_tensor *c = make_tensor(1, dims);

	float *ad = (float *)a->data;
	for (int i = 0; i < 256; i++)
		ad[i] = (float)(i - 128);

	struct sam3_tensor *inputs[] = {a};
	struct sam3_node node = make_node(SAM3_OP_RELU, inputs, 1, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	float *out = (float *)c->data;
	for (int i = 0; i < 256; i++) {
		float expected = (float)(i - 128);
		if (expected < 0.0f)
			expected = 0.0f;
		ASSERT_NEAR(out[i], expected, EPS);
	}
}

/* --- GELU SIMD test --- */

static float ref_gelu(float x)
{
	/* GELU(x) = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) */
	float c = 0.7978845608f; /* sqrt(2/pi) */
	return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

static void test_gelu_large(void)
{
	int dims[] = {256};
	struct sam3_tensor *a = make_tensor(1, dims);
	struct sam3_tensor *c = make_tensor(1, dims);

	float *ad = (float *)a->data;
	for (int i = 0; i < 256; i++)
		ad[i] = (float)(i - 128) * 0.05f;

	struct sam3_tensor *inputs[] = {a};
	struct sam3_node node = make_node(SAM3_OP_GELU, inputs, 1, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	float *out = (float *)c->data;
	for (int i = 0; i < 256; i++)
		ASSERT_NEAR(out[i], ref_gelu(ad[i]), 1e-3f);
}

/* --- Softmax edge cases --- */

static void test_softmax_large_logits(void)
{
	int dims[] = {3};
	struct sam3_tensor *a = make_tensor(1, dims);
	struct sam3_tensor *c = make_tensor(1, dims);

	float a_data[] = {1000.0f, 1001.0f, 1002.0f};
	fill_data(a, a_data);

	struct sam3_tensor *inputs[] = {a};
	struct sam3_node node = make_node(SAM3_OP_SOFTMAX, inputs, 1, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	float *out = (float *)c->data;

	/* Sum must be 1.0 even with large logits (numerical stability) */
	float sum = out[0] + out[1] + out[2];
	ASSERT_NEAR(sum, 1.0f, EPS);

	/* Values should still be ordered */
	ASSERT(out[0] < out[1]);
	ASSERT(out[1] < out[2]);

	/* No NaN or Inf */
	for (int i = 0; i < 3; i++) {
		ASSERT(out[i] == out[i]); /* NaN check */
		ASSERT(out[i] < 1.0f / 0.0f); /* Inf check */
	}
}

static void test_softmax_single(void)
{
	int dims[] = {1};
	struct sam3_tensor *a = make_tensor(1, dims);
	struct sam3_tensor *c = make_tensor(1, dims);

	float a_data[] = {42.0f};
	fill_data(a, a_data);

	struct sam3_tensor *inputs[] = {a};
	struct sam3_node node = make_node(SAM3_OP_SOFTMAX, inputs, 1, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	ASSERT_NEAR(((float *)c->data)[0], 1.0f, EPS);
}

/* --- LayerNorm gamma-only test --- */

static void test_layernorm_gamma_only(void)
{
	int dims[] = {4};
	struct sam3_tensor *a = make_tensor(1, dims);
	struct sam3_tensor *gamma = make_tensor(1, dims);
	struct sam3_tensor *c = make_tensor(1, dims);

	float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
	float g_data[] = {2.0f, 2.0f, 2.0f, 2.0f};
	fill_data(a, a_data);
	fill_data(gamma, g_data);

	/* n_inputs=2: gamma but no beta (inputs[2] = NULL) */
	struct sam3_tensor *inputs[] = {a, gamma};
	struct sam3_node node = make_node(SAM3_OP_LAYERNORM, inputs, 2, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	float *out = (float *)c->data;

	/* Output = normalized * 2.0. Mean of normalized is 0, so mean of
	 * output should be 0 * 2 = 0. Variance should be scaled by 4. */
	float mean = 0.0f;
	for (int i = 0; i < 4; i++)
		mean += out[i];
	mean /= 4.0f;
	ASSERT_NEAR(mean, 0.0f, 1e-3f);

	/* Each element should be 2x the no-affine result */
	float var = 0.0f;
	for (int i = 0; i < 4; i++)
		var += out[i] * out[i];
	var /= 4.0f;
	/* Variance of 2*normalized = 4 * 1.0 = 4.0 */
	ASSERT_NEAR(var, 4.0f, 1e-2f);
}

/* --- Matmul non-tile-aligned test --- */

static void test_matmul_non_tile_aligned(void)
{
	int a_dims[] = {7, 13};
	int b_dims[] = {13, 5};
	int c_dims[] = {7, 5};
	struct sam3_tensor *a = make_tensor(2, a_dims);
	struct sam3_tensor *b = make_tensor(2, b_dims);
	struct sam3_tensor *c = make_tensor(2, c_dims);

	float *ad = (float *)a->data;
	float *bd = (float *)b->data;
	for (int i = 0; i < 7 * 13; i++)
		ad[i] = (float)(i % 11) * 0.1f;
	for (int i = 0; i < 13 * 5; i++)
		bd[i] = (float)(i % 7) * 0.1f;

	struct sam3_tensor *inputs[] = {a, b};
	struct sam3_node node = make_node(SAM3_OP_MATMUL, inputs, 2, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	/* Verify all 7*5=35 elements against naive computation */
	float *out = (float *)c->data;
	for (int i = 0; i < 7; i++) {
		for (int j = 0; j < 5; j++) {
			float expected = 0.0f;
			for (int k = 0; k < 13; k++)
				expected += ad[i * 13 + k] * bd[k * 5 + j];
			ASSERT_NEAR(out[i * 5 + j], expected, 1e-3f);
		}
	}
}

/* --- Matmul full verify test --- */

static void test_matmul_full_verify(void)
{
	int a_dims[] = {16, 16};
	int b_dims[] = {16, 16};
	int c_dims[] = {16, 16};
	struct sam3_tensor *a = make_tensor(2, a_dims);
	struct sam3_tensor *b = make_tensor(2, b_dims);
	struct sam3_tensor *c = make_tensor(2, c_dims);

	float *ad = (float *)a->data;
	float *bd = (float *)b->data;
	for (int i = 0; i < 256; i++) {
		ad[i] = (float)(i % 7) * 0.1f;
		bd[i] = (float)(i % 5) * 0.1f;
	}

	struct sam3_tensor *inputs[] = {a, b};
	struct sam3_node node = make_node(SAM3_OP_MATMUL, inputs, 2, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	/* Verify ALL 256 output elements */
	float *out = (float *)c->data;
	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 16; j++) {
			float expected = 0.0f;
			for (int k = 0; k < 16; k++)
				expected += ad[i * 16 + k] * bd[k * 16 + j];
			ASSERT_NEAR(out[i * 16 + j], expected, 1e-3f);
		}
	}
}

/* --- Conv2D multi-channel test --- */

static void test_conv2d_multi_channel(void)
{
	/*
	 * NHWC input [1,3,3,2] (C_in=2, interleaved: ch0=1.0,
	 * ch1=2.0 at every pixel),
	 * OHWI weight [2,2,2,2] (OC=2, KH=2, KW=2, IC=2) all-ones,
	 * NHWC output [1,2,2,2] stride=1 pad=0.
	 */
	int in_dims[] = {1, 3, 3, 2};
	int w_dims[] = {2, 2, 2, 2};
	int out_dims[] = {1, 2, 2, 2};

	struct sam3_tensor *input = make_tensor(4, in_dims);
	struct sam3_tensor *weight = make_tensor(4, w_dims);
	struct sam3_tensor *output = make_tensor(4, out_dims);

	/* NHWC fill: each pixel has [c0=1.0, c1=2.0] interleaved */
	float in_data[18];
	for (int p = 0; p < 9; p++) {
		in_data[p * 2 + 0] = 1.0f;
		in_data[p * 2 + 1] = 2.0f;
	}
	fill_data(input, in_data);

	/* All-ones kernel: each filter sums 2x2 patch across 2 inputs */
	float w_data[16];
	for (int i = 0; i < 16; i++)
		w_data[i] = 1.0f;
	fill_data(weight, w_data);

	struct sam3_tensor *inputs[] = {input, weight};
	struct sam3_node node = make_node(SAM3_OP_CONV2D, inputs, 2, output);
	node.params[0] = 1;
	node.params[1] = 0;

	struct sam3_graph g;
	memset(&g, 0, sizeof(g));
	g.nodes[0] = node;
	g.n_nodes = 1;

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base, &g), SAM3_OK);

	float *out = (float *)output->data;
	/*
	 * Per output pixel per filter: sum of 2x2 patch over 2 channels
	 *   = 4 * 1.0 + 4 * 2.0 = 12
	 * NHWC output [1,2,2,2] has 8 elements, all 12.0.
	 */
	for (int i = 0; i < 8; i++)
		ASSERT_NEAR(out[i], 12.0f, EPS);
}

/* --- Conv2D multi-batch test --- */

static void test_conv2d_multi_batch(void)
{
	/*
	 * NHWC input [2,3,3,1] (C_in=1, batch 0 all 1.0, batch 1
	 * all 3.0), OHWI weight [1,2,2,1], NHWC output [2,2,2,1].
	 * Stride=1 pad=0.
	 */
	int in_dims[] = {2, 3, 3, 1};
	int w_dims[] = {1, 2, 2, 1};
	int out_dims[] = {2, 2, 2, 1};

	struct sam3_tensor *input = make_tensor(4, in_dims);
	struct sam3_tensor *weight = make_tensor(4, w_dims);
	struct sam3_tensor *output = make_tensor(4, out_dims);

	/* Batch 0: all 1s, Batch 1: all 3s */
	float in_data[18];
	for (int i = 0; i < 9; i++)
		in_data[i] = 1.0f;
	for (int i = 9; i < 18; i++)
		in_data[i] = 3.0f;
	fill_data(input, in_data);

	/* All-ones kernel */
	float w_data[] = {1.0f, 1.0f, 1.0f, 1.0f};
	fill_data(weight, w_data);

	struct sam3_tensor *inputs[] = {input, weight};
	struct sam3_node node = make_node(SAM3_OP_CONV2D, inputs, 2, output);
	node.params[0] = 1;
	node.params[1] = 0;

	struct sam3_graph g;
	memset(&g, 0, sizeof(g));
	g.nodes[0] = node;
	g.n_nodes = 1;

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base, &g), SAM3_OK);

	float *out = (float *)output->data;
	/* Batch 0: each 2x2 sum of 1s = 4 */
	for (int i = 0; i < 4; i++)
		ASSERT_NEAR(out[i], 4.0f, EPS);
	/* Batch 1: each 2x2 sum of 3s = 12 */
	for (int i = 4; i < 8; i++)
		ASSERT_NEAR(out[i], 12.0f, EPS);
}

/* --- Transpose non-square test --- */

static void test_transpose_non_square(void)
{
	int in_dims[] = {5, 3};
	int out_dims[] = {3, 5};
	struct sam3_tensor *a = make_tensor(2, in_dims);
	struct sam3_tensor *c = make_tensor(2, out_dims);

	float a_data[15];
	for (int i = 0; i < 15; i++)
		a_data[i] = (float)(i + 1);
	fill_data(a, a_data);

	struct sam3_tensor *inputs[] = {a};
	struct sam3_node node = make_node(SAM3_OP_TRANSPOSE, inputs, 1, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}), SAM3_OK);

	float *out = (float *)c->data;
	for (int i = 0; i < 5; i++)
		for (int j = 0; j < 3; j++)
			ASSERT_NEAR(out[j * 5 + i], a_data[i * 3 + j], EPS);
}

/* --- Graph error mid-chain test --- */

static void test_graph_error_mid_chain(void)
{
	/* 3-node graph: add -> relu -> relu
	 * Middle node has wrong dtype to force an error. */
	int dims[] = {4};
	struct sam3_tensor *a = make_tensor(1, dims);
	struct sam3_tensor *b = make_tensor(1, dims);
	struct sam3_tensor *sum = make_tensor(1, dims);
	struct sam3_tensor *mid = make_tensor(1, dims);
	struct sam3_tensor *out = make_tensor(1, dims);

	float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
	float b_data[] = {1.0f, 1.0f, 1.0f, 1.0f};
	fill_data(a, a_data);
	fill_data(b, b_data);

	struct sam3_graph g;
	memset(&g, 0, sizeof(g));

	/* Node 0: add (OK) */
	struct sam3_tensor *add_inputs[] = {a, b};
	g.nodes[0] = make_node(SAM3_OP_ADD, add_inputs, 2, sum);

	/* Node 1: relu with wrong dtype input -> should fail */
	sum->dtype = SAM3_DTYPE_I32; /* Force dtype error */
	struct sam3_tensor *relu_inputs[] = {sum};
	g.nodes[1] = make_node(SAM3_OP_RELU, relu_inputs, 1, mid);

	/* Node 2: relu (should never run) */
	struct sam3_tensor *relu2_inputs[] = {mid};
	g.nodes[2] = make_node(SAM3_OP_RELU, relu2_inputs, 1, out);

	g.n_nodes = 3;

	/* Graph should fail at node 1 */
	ASSERT(g_cpu.base.ops->graph_eval(&g_cpu.base, &g) != SAM3_OK);
}

/* --- Profiler op counting test --- */

#ifdef SAM3_HAS_PROFILE
static void test_profiler_op_counting(void)
{
	struct sam3_profiler *prof = sam3_profiler_create();
	sam3_profiler_enable(prof);
	g_cpu.profiler = prof;

	int dims[] = {4};
	struct sam3_tensor *a = make_tensor(1, dims);
	struct sam3_tensor *b = make_tensor(1, dims);
	struct sam3_tensor *sum = make_tensor(1, dims);
	struct sam3_tensor *out = make_tensor(1, dims);

	float a_data[] = {-3.0f, -1.0f, 1.0f, 3.0f};
	float b_data[] = {1.0f, 1.0f, 1.0f, 1.0f};
	fill_data(a, a_data);
	fill_data(b, b_data);

	struct sam3_graph g;
	memset(&g, 0, sizeof(g));

	struct sam3_tensor *add_inputs[] = {a, b};
	g.nodes[0] = make_node(SAM3_OP_ADD, add_inputs, 2, sum);

	struct sam3_tensor *relu_inputs[] = {sum};
	g.nodes[1] = make_node(SAM3_OP_RELU, relu_inputs, 1, out);
	g.n_nodes = 2;

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base, &g), SAM3_OK);

	ASSERT_EQ(prof->op_stats[SAM3_OP_ADD].calls, 1);
	ASSERT_EQ(prof->op_stats[SAM3_OP_RELU].calls, 1);
	ASSERT_EQ(prof->op_stats[SAM3_OP_MATMUL].calls, 0);

	g_cpu.profiler = NULL;
	sam3_profiler_free(prof);
}
#endif /* SAM3_HAS_PROFILE */

/* --- Main --- */

int main(void)
{
	setup();

	/* Add */
	test_add_basic();
	test_add_broadcast();
	test_add_large();

	/* Mul */
	test_mul_basic();
	test_mul_broadcast();
	test_mul_large();

	/* ReLU */
	test_relu_basic();
	test_relu_all_negative();
	test_relu_large();

	/* GELU */
	test_gelu_basic();
	test_gelu_zero();
	test_gelu_large();

	/* Softmax */
	test_softmax_basic();
	test_softmax_2d();
	test_softmax_uniform();
	test_softmax_large_logits();
	test_softmax_single();

	/* LayerNorm */
	test_layernorm_no_affine();
	test_layernorm_with_affine();
	test_layernorm_2d();
	test_layernorm_gamma_only();

	/* Matmul */
	test_matmul_2x3_3x2();
	test_matmul_identity();
	test_matmul_large();
	test_matmul_non_tile_aligned();
	test_matmul_full_verify();

	/* Reshape */
	test_reshape_basic();
	test_reshape_3d_to_2d();
	test_reshape_size_mismatch();

	/* Transpose */
	test_transpose_basic();
	test_transpose_square();
	test_transpose_large();
	test_transpose_non_square();

	/* Conv2D */
	test_conv2d_basic();
	test_conv2d_with_padding();
	test_conv2d_stride2();
	test_conv2d_multi_channel();
	test_conv2d_multi_batch();

	/* Multi-node graph */
	test_graph_multi_node();

	/* Error handling */
	test_error_null_tensor();
	test_error_dtype();
	test_graph_error_mid_chain();

	/* Profiler */
#ifdef SAM3_HAS_PROFILE
	test_profiler_op_counting();
#endif

	teardown();

	TEST_REPORT();
}
