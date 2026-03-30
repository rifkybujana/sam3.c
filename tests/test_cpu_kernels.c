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
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "backend/cpu/cpu_backend.h"
#include "backend/backend.h"
#include "core/graph.h"
#include "core/tensor.h"

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
	/* Input: [1,1,4,4], Weight: [1,1,2,2], stride=1, pad=0 -> [1,1,3,3] */
	int in_dims[] = {1, 1, 4, 4};
	int w_dims[] = {1, 1, 2, 2};
	int out_dims[] = {1, 1, 3, 3};

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
	/* Input: [1,1,3,3], Weight: [1,1,3,3], stride=1, pad=1 -> [1,1,3,3] */
	int in_dims[] = {1, 1, 3, 3};
	int w_dims[] = {1, 1, 3, 3};
	int out_dims[] = {1, 1, 3, 3};

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
	/* Input: [1,1,4,4], Weight: [1,1,2,2], stride=2, pad=0 -> [1,1,2,2] */
	int in_dims[] = {1, 1, 4, 4};
	int w_dims[] = {1, 1, 2, 2};
	int out_dims[] = {1, 1, 2, 2};

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

	/* ReLU */
	test_relu_basic();
	test_relu_all_negative();

	/* GELU */
	test_gelu_basic();
	test_gelu_zero();

	/* Softmax */
	test_softmax_basic();
	test_softmax_2d();
	test_softmax_uniform();

	/* LayerNorm */
	test_layernorm_no_affine();
	test_layernorm_with_affine();
	test_layernorm_2d();

	/* Matmul */
	test_matmul_2x3_3x2();
	test_matmul_identity();
	test_matmul_large();

	/* Reshape */
	test_reshape_basic();
	test_reshape_3d_to_2d();
	test_reshape_size_mismatch();

	/* Transpose */
	test_transpose_basic();
	test_transpose_square();
	test_transpose_large();

	/* Conv2D */
	test_conv2d_basic();
	test_conv2d_with_padding();
	test_conv2d_stride2();

	/* Multi-node graph */
	test_graph_multi_node();

	/* Error handling */
	test_error_null_tensor();
	test_error_dtype();

	teardown();

	TEST_REPORT();
}
