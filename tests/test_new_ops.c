/*
 * tests/test_new_ops.c - Unit tests for sigmoid, silu, embed, concat, slice, upsample
 *
 * Tests element-wise sigmoid, element-wise SiLU (Swish), embedding
 * table lookup, tensor concatenation, tensor slicing, and nearest-
 * neighbor upsampling kernels.  Uses the CPU backend with arena
 * allocation and graph_eval for execution.
 *
 * Key types:  sam3_node, sam3_tensor, sam3_cpu_backend
 * Depends on: test_helpers.h, cpu_backend.h, core/graph.h, core/tensor.h
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

#include <math.h>
#include <string.h>

#define EPS 1e-4f

/* --- Helpers --- */

static struct sam3_cpu_backend g_cpu;

static void setup(void)
{
	memset(&g_cpu, 0, sizeof(g_cpu));
	g_cpu.base.type = SAM3_BACKEND_CPU;
	g_cpu.base.ops = sam3_cpu_backend_ops();
	g_cpu.arena_capacity = 16 * 1024 * 1024;
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

static struct sam3_tensor *make_i32_tensor(int n_dims, const int *dims)
{
	struct sam3_tensor *t = (struct sam3_tensor *)
		sam3_arena_alloc(&g_cpu.arena, sizeof(struct sam3_tensor));
	memset(t, 0, sizeof(*t));
	t->dtype = SAM3_DTYPE_I32;
	t->n_dims = n_dims;
	for (int i = 0; i < n_dims; i++)
		t->dims[i] = dims[i];
	g_cpu.base.ops->alloc_tensor(&g_cpu.base, t);
	return t;
}

static void fill_data(struct sam3_tensor *t, const void *data)
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

/* Reference functions */

static float ref_sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

static float ref_silu(float x)
{
	return x / (1.0f + expf(-x));
}

/* --- Sigmoid tests --- */

static void test_sigmoid_basic(void)
{
	int dims[] = {5};
	struct sam3_tensor *a = make_tensor(1, dims);
	struct sam3_tensor *c = make_tensor(1, dims);

	float a_data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
	fill_data(a, a_data);

	struct sam3_tensor *inputs[] = {a};
	struct sam3_node node = make_node(SAM3_OP_SIGMOID, inputs, 1, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}),
		  SAM3_OK);

	float *out = (float *)c->data;
	for (int i = 0; i < 5; i++)
		ASSERT_NEAR(out[i], ref_sigmoid(a_data[i]), EPS);
}

static void test_sigmoid_large(void)
{
	int dims[] = {256};
	struct sam3_tensor *a = make_tensor(1, dims);
	struct sam3_tensor *c = make_tensor(1, dims);

	float *ad = (float *)a->data;
	for (int i = 0; i < 256; i++)
		ad[i] = (float)(i - 128) * 0.05f;

	struct sam3_tensor *inputs[] = {a};
	struct sam3_node node = make_node(SAM3_OP_SIGMOID, inputs, 1, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}),
		  SAM3_OK);

	float *out = (float *)c->data;
	for (int i = 0; i < 256; i++)
		ASSERT_NEAR(out[i], ref_sigmoid(ad[i]), EPS);
}

/* --- SiLU tests --- */

static void test_silu_basic(void)
{
	int dims[] = {5};
	struct sam3_tensor *a = make_tensor(1, dims);
	struct sam3_tensor *c = make_tensor(1, dims);

	float a_data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
	fill_data(a, a_data);

	struct sam3_tensor *inputs[] = {a};
	struct sam3_node node = make_node(SAM3_OP_SILU, inputs, 1, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}),
		  SAM3_OK);

	float *out = (float *)c->data;
	for (int i = 0; i < 5; i++)
		ASSERT_NEAR(out[i], ref_silu(a_data[i]), EPS);
}

static void test_silu_zero(void)
{
	int dims[] = {1};
	struct sam3_tensor *a = make_tensor(1, dims);
	struct sam3_tensor *c = make_tensor(1, dims);

	float a_data[] = {0.0f};
	fill_data(a, a_data);

	struct sam3_tensor *inputs[] = {a};
	struct sam3_node node = make_node(SAM3_OP_SILU, inputs, 1, c);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}),
		  SAM3_OK);

	ASSERT_NEAR(((float *)c->data)[0], 0.0f, EPS);
}

/* --- Embed tests --- */

static void test_embed_basic(void)
{
	/* 4x3 embedding table, 3 token indices */
	int table_dims[] = {4, 3};
	int idx_dims[] = {3};
	int out_dims[] = {3, 3};

	struct sam3_tensor *table = make_tensor(2, table_dims);
	struct sam3_tensor *indices = make_i32_tensor(1, idx_dims);
	struct sam3_tensor *output = make_tensor(2, out_dims);

	/* Table rows:
	 *   row 0: {1, 2, 3}
	 *   row 1: {4, 5, 6}
	 *   row 2: {7, 8, 9}
	 *   row 3: {10, 11, 12}
	 */
	float table_data[] = {
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f,
		7.0f, 8.0f, 9.0f,
		10.0f, 11.0f, 12.0f,
	};
	fill_data(table, table_data);

	int32_t idx_data[] = {0, 2, 1};
	fill_data(indices, idx_data);

	struct sam3_tensor *inputs[] = {table, indices};
	struct sam3_node node = make_node(SAM3_OP_EMBED, inputs, 2, output);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}),
		  SAM3_OK);

	float *out = (float *)output->data;

	/* output[0,:] = table[0,:] = {1, 2, 3} */
	ASSERT_NEAR(out[0], 1.0f, EPS);
	ASSERT_NEAR(out[1], 2.0f, EPS);
	ASSERT_NEAR(out[2], 3.0f, EPS);

	/* output[1,:] = table[2,:] = {7, 8, 9} */
	ASSERT_NEAR(out[3], 7.0f, EPS);
	ASSERT_NEAR(out[4], 8.0f, EPS);
	ASSERT_NEAR(out[5], 9.0f, EPS);

	/* output[2,:] = table[1,:] = {4, 5, 6} */
	ASSERT_NEAR(out[6], 4.0f, EPS);
	ASSERT_NEAR(out[7], 5.0f, EPS);
	ASSERT_NEAR(out[8], 6.0f, EPS);
}

static void test_embed_single(void)
{
	/* Single token lookup from a 4x3 table */
	int table_dims[] = {4, 3};
	int idx_dims[] = {1};
	int out_dims[] = {1, 3};

	struct sam3_tensor *table = make_tensor(2, table_dims);
	struct sam3_tensor *indices = make_i32_tensor(1, idx_dims);
	struct sam3_tensor *output = make_tensor(2, out_dims);

	float table_data[] = {
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f,
		7.0f, 8.0f, 9.0f,
		10.0f, 11.0f, 12.0f,
	};
	fill_data(table, table_data);

	int32_t idx_data[] = {3};
	fill_data(indices, idx_data);

	struct sam3_tensor *inputs[] = {table, indices};
	struct sam3_node node = make_node(SAM3_OP_EMBED, inputs, 2, output);

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}),
		  SAM3_OK);

	float *out = (float *)output->data;

	/* output[0,:] = table[3,:] = {10, 11, 12} */
	ASSERT_NEAR(out[0], 10.0f, EPS);
	ASSERT_NEAR(out[1], 11.0f, EPS);
	ASSERT_NEAR(out[2], 12.0f, EPS);
}

/* --- Concat tests --- */

static void test_concat_axis0(void)
{
	/* concat([2,3], [1,3], axis=0) -> [3,3] */
	struct sam3_tensor *a = make_tensor(2, (int[]){2, 3});
	struct sam3_tensor *b = make_tensor(2, (int[]){1, 3});
	struct sam3_tensor *out = make_tensor(2, (int[]){3, 3});

	float a_data[] = {1, 2, 3, 4, 5, 6};
	float b_data[] = {7, 8, 9};
	fill_data(a, a_data);
	fill_data(b, b_data);

	struct sam3_node node;
	memset(&node, 0, sizeof(node));
	node.op = SAM3_OP_CONCAT;
	node.inputs[0] = a;
	node.inputs[1] = b;
	node.n_inputs = 2;
	node.output = out;
	node.params[0] = 0; /* axis */

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}),
		  SAM3_OK);

	float *o = (float *)out->data;
	float expect[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	for (int i = 0; i < 9; i++)
		ASSERT_NEAR(o[i], expect[i], EPS);
}

static void test_concat_axis1(void)
{
	/* concat([2,2], [2,3], axis=1) -> [2,5] */
	struct sam3_tensor *a = make_tensor(2, (int[]){2, 2});
	struct sam3_tensor *b = make_tensor(2, (int[]){2, 3});
	struct sam3_tensor *out = make_tensor(2, (int[]){2, 5});

	float a_data[] = {1, 2, 3, 4};
	float b_data[] = {5, 6, 7, 8, 9, 10};
	fill_data(a, a_data);
	fill_data(b, b_data);

	struct sam3_node node;
	memset(&node, 0, sizeof(node));
	node.op = SAM3_OP_CONCAT;
	node.inputs[0] = a;
	node.inputs[1] = b;
	node.n_inputs = 2;
	node.output = out;
	node.params[0] = 1; /* axis */

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}),
		  SAM3_OK);

	float *o = (float *)out->data;
	/* row 0: a[0,0], a[0,1], b[0,0], b[0,1], b[0,2] = 1,2,5,6,7 */
	/* row 1: a[1,0], a[1,1], b[1,0], b[1,1], b[1,2] = 3,4,8,9,10 */
	float expect[] = {1, 2, 5, 6, 7, 3, 4, 8, 9, 10};
	for (int i = 0; i < 10; i++)
		ASSERT_NEAR(o[i], expect[i], EPS);
}

static void test_concat_three(void)
{
	/* concat three [2,2] tensors along axis 0 -> [6,2] */
	struct sam3_tensor *a = make_tensor(2, (int[]){2, 2});
	struct sam3_tensor *b = make_tensor(2, (int[]){2, 2});
	struct sam3_tensor *c = make_tensor(2, (int[]){2, 2});
	struct sam3_tensor *out = make_tensor(2, (int[]){6, 2});

	float a_data[] = {1, 2, 3, 4};
	float b_data[] = {5, 6, 7, 8};
	float c_data[] = {9, 10, 11, 12};
	fill_data(a, a_data);
	fill_data(b, b_data);
	fill_data(c, c_data);

	struct sam3_node node;
	memset(&node, 0, sizeof(node));
	node.op = SAM3_OP_CONCAT;
	node.inputs[0] = a;
	node.inputs[1] = b;
	node.inputs[2] = c;
	node.n_inputs = 3;
	node.output = out;
	node.params[0] = 0; /* axis */

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}),
		  SAM3_OK);

	float *o = (float *)out->data;
	float expect[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	for (int i = 0; i < 12; i++)
		ASSERT_NEAR(o[i], expect[i], EPS);
}

/* --- Slice tests --- */

static void test_slice_axis0(void)
{
	/* slice([4,3], axis=0, start=1, end=3) -> [2,3] */
	struct sam3_tensor *inp = make_tensor(2, (int[]){4, 3});
	struct sam3_tensor *out = make_tensor(2, (int[]){2, 3});

	float inp_data[] = {
		1, 2, 3,     /* row 0 */
		4, 5, 6,     /* row 1 */
		7, 8, 9,     /* row 2 */
		10, 11, 12,  /* row 3 */
	};
	fill_data(inp, inp_data);

	struct sam3_node node;
	memset(&node, 0, sizeof(node));
	node.op = SAM3_OP_SLICE;
	node.inputs[0] = inp;
	node.n_inputs = 1;
	node.output = out;
	node.params[0] = 0; /* axis */
	node.params[1] = 1; /* start */
	node.params[2] = 3; /* end */

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}),
		  SAM3_OK);

	float *o = (float *)out->data;
	/* rows 1 and 2: {4,5,6, 7,8,9} */
	float expect[] = {4, 5, 6, 7, 8, 9};
	for (int i = 0; i < 6; i++)
		ASSERT_NEAR(o[i], expect[i], EPS);
}

static void test_slice_axis1(void)
{
	/* slice([2,5], axis=1, start=1, end=4) -> [2,3] */
	struct sam3_tensor *inp = make_tensor(2, (int[]){2, 5});
	struct sam3_tensor *out = make_tensor(2, (int[]){2, 3});

	float inp_data[] = {
		1, 2, 3, 4, 5,     /* row 0 */
		6, 7, 8, 9, 10,    /* row 1 */
	};
	fill_data(inp, inp_data);

	struct sam3_node node;
	memset(&node, 0, sizeof(node));
	node.op = SAM3_OP_SLICE;
	node.inputs[0] = inp;
	node.n_inputs = 1;
	node.output = out;
	node.params[0] = 1; /* axis */
	node.params[1] = 1; /* start */
	node.params[2] = 4; /* end */

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}),
		  SAM3_OK);

	float *o = (float *)out->data;
	/* row 0 cols 1..3: {2,3,4}, row 1 cols 1..3: {7,8,9} */
	float expect[] = {2, 3, 4, 7, 8, 9};
	for (int i = 0; i < 6; i++)
		ASSERT_NEAR(o[i], expect[i], EPS);
}

/* --- Upsample tests --- */

static void test_upsample_2x(void)
{
	/* [1,1,2,2] with scale=2 -> [1,1,4,4] */
	struct sam3_tensor *inp = make_tensor(4, (int[]){1, 1, 2, 2});
	struct sam3_tensor *out = make_tensor(4, (int[]){1, 1, 4, 4});

	float inp_data[] = {1, 2, 3, 4};
	fill_data(inp, inp_data);

	struct sam3_node node;
	memset(&node, 0, sizeof(node));
	node.op = SAM3_OP_UPSAMPLE;
	node.inputs[0] = inp;
	node.n_inputs = 1;
	node.output = out;
	node.params[0] = 2; /* scale */

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}),
		  SAM3_OK);

	float *o = (float *)out->data;
	/*
	 * Input:     Output (nearest-neighbor 2x):
	 *  1 2        1 1 2 2
	 *  3 4        1 1 2 2
	 *             3 3 4 4
	 *             3 3 4 4
	 */
	float expect[] = {
		1, 1, 2, 2,
		1, 1, 2, 2,
		3, 3, 4, 4,
		3, 3, 4, 4,
	};
	for (int i = 0; i < 16; i++)
		ASSERT_NEAR(o[i], expect[i], EPS);
}

static void test_upsample_3x(void)
{
	/* [1,1,2,3] with scale=3 -> [1,1,6,9] */
	struct sam3_tensor *inp = make_tensor(4, (int[]){1, 1, 2, 3});
	struct sam3_tensor *out = make_tensor(4, (int[]){1, 1, 6, 9});

	float inp_data[] = {1, 2, 3, 4, 5, 6};
	fill_data(inp, inp_data);

	struct sam3_node node;
	memset(&node, 0, sizeof(node));
	node.op = SAM3_OP_UPSAMPLE;
	node.inputs[0] = inp;
	node.n_inputs = 1;
	node.output = out;
	node.params[0] = 3; /* scale */

	ASSERT_EQ(g_cpu.base.ops->graph_eval(&g_cpu.base,
		  &(struct sam3_graph){.nodes = {node}, .n_nodes = 1}),
		  SAM3_OK);

	float *o = (float *)out->data;
	/*
	 * Input:        Output (nearest-neighbor 3x):
	 *  1 2 3         1 1 1 2 2 2 3 3 3    (rows 0-2)
	 *  4 5 6         1 1 1 2 2 2 3 3 3
	 *                1 1 1 2 2 2 3 3 3
	 *                4 4 4 5 5 5 6 6 6    (rows 3-5)
	 *                4 4 4 5 5 5 6 6 6
	 *                4 4 4 5 5 5 6 6 6
	 */
	for (int y = 0; y < 6; y++) {
		int sy = y / 3;
		for (int x = 0; x < 9; x++) {
			int sx = x / 3;
			float expected = inp_data[sy * 3 + sx];
			ASSERT_NEAR(o[y * 9 + x], expected, EPS);
		}
	}
}

/* --- Main --- */

int main(void)
{
	setup();

	/* Sigmoid */
	test_sigmoid_basic();
	test_sigmoid_large();

	/* SiLU */
	test_silu_basic();
	test_silu_zero();

	/* Embed */
	test_embed_basic();
	test_embed_single();

	/* Concat */
	test_concat_axis0();
	test_concat_axis1();
	test_concat_three();

	/* Slice */
	test_slice_axis0();
	test_slice_axis1();

	/* Upsample */
	test_upsample_2x();
	test_upsample_3x();

	teardown();

	TEST_REPORT();
}
