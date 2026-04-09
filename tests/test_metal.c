/*
 * tests/test_metal.c - Metal backend tests
 *
 * Tests the MLX-C Metal backend: init/free, tensor allocation, dtype
 * mapping, and per-op correctness. Each op test builds a small graph,
 * evaluates on Metal, and compares against CPU results. Skips
 * gracefully if Metal is unavailable.
 *
 * Key types:  sam3_metal_backend
 * Depends on: test_helpers.h, backend/backend.h, core/tensor.h,
 *             core/graph.h, core/quant.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <string.h>
#include <math.h>

#include "test_helpers.h"
#include "backend/backend.h"
#include "core/tensor.h"
#include "core/graph.h"
#include "core/quant.h"

#ifdef SAM3_HAS_METAL
#include "backend/metal/metal_backend.h"
#endif
#ifdef SAM3_HAS_CPU
#include "backend/cpu/cpu_backend.h"
#endif

#ifdef SAM3_HAS_METAL

/* ── Helpers ──────────────────────────────────────────────────────── */

static struct sam3_tensor make_tensor(enum sam3_dtype dtype,
				      int n_dims, const int *dims)
{
	struct sam3_tensor t;
	memset(&t, 0, sizeof(t));
	t.dtype = dtype;
	t.n_dims = n_dims;
	for (int i = 0; i < n_dims; i++)
		t.dims[i] = dims[i];
	return t;
}

static int float_arrays_match(const float *a, const float *b,
			      int n, float rel_tol)
{
	for (int i = 0; i < n; i++) {
		float diff = fabsf(a[i] - b[i]);
		float mag = fmaxf(fabsf(a[i]), fabsf(b[i]));
		float tol = fmaxf(rel_tol * mag, 1e-6f);
		if (diff > tol)
			return 0;
	}
	return 1;
}

/* ── Test: init and free ─────────────────────────────────────────── */

static void test_metal_init_free(void)
{
	struct sam3_backend *be = sam3_backend_init(SAM3_BACKEND_METAL);
	ASSERT(be != NULL);
	if (!be)
		return;
	ASSERT_EQ(be->type, SAM3_BACKEND_METAL);
	ASSERT(be->ops != NULL);
	sam3_backend_free(be);
}

/* ── Test: tensor allocation ─────────────────────────────────────── */

static void test_metal_alloc_tensor(void)
{
	struct sam3_backend *be = sam3_backend_init(SAM3_BACKEND_METAL);
	ASSERT(be != NULL);
	if (!be)
		return;

	int dims[] = {4, 8};
	struct sam3_tensor t = make_tensor(SAM3_DTYPE_F32, 2, dims);

	enum sam3_error err = be->ops->alloc_tensor(be, &t);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT(t.data != NULL);
	ASSERT_EQ(t.nbytes, (size_t)(4 * 8 * sizeof(float)));
	ASSERT_EQ(t.strides[1], 1);
	ASSERT_EQ(t.strides[0], 8);

	sam3_backend_free(be);
}

/* ── Helper: run a single-op graph on a backend ──────────────────── */

#ifdef SAM3_HAS_CPU

static enum sam3_error eval_single_op(
	struct sam3_backend *be,
	enum sam3_op op,
	struct sam3_tensor *inputs[], int n_inputs,
	const void *input_data[],
	struct sam3_tensor *output,
	const int *params)
{
	struct sam3_graph g;
	sam3_graph_init(&g);

	for (int i = 0; i < n_inputs; i++) {
		enum sam3_error err = be->ops->alloc_tensor(be, inputs[i]);
		if (err != SAM3_OK)
			return err;
		memcpy(inputs[i]->data, input_data[i], inputs[i]->nbytes);
	}

	enum sam3_error err = be->ops->alloc_tensor(be, output);
	if (err != SAM3_OK)
		return err;

	struct sam3_node *node = &g.nodes[0];
	node->op = op;
	node->n_inputs = n_inputs;
	for (int i = 0; i < n_inputs; i++)
		node->inputs[i] = inputs[i];
	node->output = output;
	if (params)
		memcpy(node->params, params, sizeof(node->params));
	g.n_nodes = 1;

	return be->ops->graph_eval(be, &g);
}

/* ── Test: matmul ────────────────────────────────────────────────── */

static void test_metal_matmul(void)
{
	struct sam3_backend *metal = sam3_backend_init(SAM3_BACKEND_METAL);
	struct sam3_backend *cpu = sam3_backend_init(SAM3_BACKEND_CPU);
	ASSERT(metal && cpu);
	if (!metal || !cpu) {
		sam3_backend_free(metal);
		sam3_backend_free(cpu);
		return;
	}

	int dims_a[] = {2, 3};
	int dims_b[] = {3, 4};
	int dims_c[] = {2, 4};

	float data_a[] = {1, 2, 3, 4, 5, 6};
	float data_b[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

	struct sam3_tensor ma = make_tensor(SAM3_DTYPE_F32, 2, dims_a);
	struct sam3_tensor mb = make_tensor(SAM3_DTYPE_F32, 2, dims_b);
	struct sam3_tensor mc = make_tensor(SAM3_DTYPE_F32, 2, dims_c);
	struct sam3_tensor *m_inputs[] = {&ma, &mb};
	const void *m_data[] = {data_a, data_b};
	ASSERT_EQ(eval_single_op(metal, SAM3_OP_MATMUL, m_inputs, 2,
				 m_data, &mc, NULL), SAM3_OK);

	struct sam3_tensor ca = make_tensor(SAM3_DTYPE_F32, 2, dims_a);
	struct sam3_tensor cb = make_tensor(SAM3_DTYPE_F32, 2, dims_b);
	struct sam3_tensor cc = make_tensor(SAM3_DTYPE_F32, 2, dims_c);
	struct sam3_tensor *c_inputs[] = {&ca, &cb};
	const void *c_data[] = {data_a, data_b};
	ASSERT_EQ(eval_single_op(cpu, SAM3_OP_MATMUL, c_inputs, 2,
				 c_data, &cc, NULL), SAM3_OK);

	ASSERT(float_arrays_match((float *)mc.data, (float *)cc.data,
				  2 * 4, 1e-5f));

	sam3_backend_free(metal);
	sam3_backend_free(cpu);
}

/* ── Test: add ───────────────────────────────────────────────────── */

static void test_metal_add(void)
{
	struct sam3_backend *metal = sam3_backend_init(SAM3_BACKEND_METAL);
	struct sam3_backend *cpu = sam3_backend_init(SAM3_BACKEND_CPU);
	ASSERT(metal && cpu);
	if (!metal || !cpu) {
		sam3_backend_free(metal);
		sam3_backend_free(cpu);
		return;
	}

	int dims[] = {2, 3};
	float data_a[] = {1, 2, 3, 4, 5, 6};
	float data_b[] = {10, 20, 30, 40, 50, 60};

	struct sam3_tensor ma = make_tensor(SAM3_DTYPE_F32, 2, dims);
	struct sam3_tensor mb = make_tensor(SAM3_DTYPE_F32, 2, dims);
	struct sam3_tensor mc = make_tensor(SAM3_DTYPE_F32, 2, dims);
	struct sam3_tensor *m_inputs[] = {&ma, &mb};
	const void *m_data[] = {data_a, data_b};
	ASSERT_EQ(eval_single_op(metal, SAM3_OP_ADD, m_inputs, 2,
				 m_data, &mc, NULL), SAM3_OK);

	struct sam3_tensor ca = make_tensor(SAM3_DTYPE_F32, 2, dims);
	struct sam3_tensor cb = make_tensor(SAM3_DTYPE_F32, 2, dims);
	struct sam3_tensor cc = make_tensor(SAM3_DTYPE_F32, 2, dims);
	struct sam3_tensor *c_inputs[] = {&ca, &cb};
	const void *c_data[] = {data_a, data_b};
	ASSERT_EQ(eval_single_op(cpu, SAM3_OP_ADD, c_inputs, 2,
				 c_data, &cc, NULL), SAM3_OK);

	ASSERT(float_arrays_match((float *)mc.data, (float *)cc.data,
				  6, 1e-5f));

	sam3_backend_free(metal);
	sam3_backend_free(cpu);
}

/* ── Test: softmax ───────────────────────────────────────────────── */

static void test_metal_softmax(void)
{
	struct sam3_backend *metal = sam3_backend_init(SAM3_BACKEND_METAL);
	struct sam3_backend *cpu = sam3_backend_init(SAM3_BACKEND_CPU);
	ASSERT(metal && cpu);
	if (!metal || !cpu) {
		sam3_backend_free(metal);
		sam3_backend_free(cpu);
		return;
	}

	int dims[] = {2, 4};
	float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

	struct sam3_tensor ma = make_tensor(SAM3_DTYPE_F32, 2, dims);
	struct sam3_tensor mc = make_tensor(SAM3_DTYPE_F32, 2, dims);
	struct sam3_tensor *m_inputs[] = {&ma};
	const void *m_data[] = {data};
	ASSERT_EQ(eval_single_op(metal, SAM3_OP_SOFTMAX, m_inputs, 1,
				 m_data, &mc, NULL), SAM3_OK);

	struct sam3_tensor ca = make_tensor(SAM3_DTYPE_F32, 2, dims);
	struct sam3_tensor cc = make_tensor(SAM3_DTYPE_F32, 2, dims);
	struct sam3_tensor *c_inputs[] = {&ca};
	const void *c_data[] = {data};
	ASSERT_EQ(eval_single_op(cpu, SAM3_OP_SOFTMAX, c_inputs, 1,
				 c_data, &cc, NULL), SAM3_OK);

	ASSERT(float_arrays_match((float *)mc.data, (float *)cc.data,
				  8, 1e-3f));

	sam3_backend_free(metal);
	sam3_backend_free(cpu);
}

/* ── Test: reshape ───────────────────────────────────────────────── */

static void test_metal_reshape(void)
{
	struct sam3_backend *metal = sam3_backend_init(SAM3_BACKEND_METAL);
	ASSERT(metal != NULL);
	if (!metal)
		return;

	int in_dims[] = {2, 6};
	int out_dims[] = {3, 4};
	float data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

	struct sam3_tensor in = make_tensor(SAM3_DTYPE_F32, 2, in_dims);
	struct sam3_tensor out = make_tensor(SAM3_DTYPE_F32, 2, out_dims);
	struct sam3_tensor *inputs[] = {&in};
	const void *in_data[] = {data};
	ASSERT_EQ(eval_single_op(metal, SAM3_OP_RESHAPE, inputs, 1,
				 in_data, &out, NULL), SAM3_OK);

	ASSERT(float_arrays_match((float *)out.data, data, 12, 0.0f));

	sam3_backend_free(metal);
}

/* ── Test: multi-node graph (matmul -> add -> softmax) ────────────── */

static void test_metal_multi_node(void)
{
	struct sam3_backend *metal = sam3_backend_init(SAM3_BACKEND_METAL);
	struct sam3_backend *cpu = sam3_backend_init(SAM3_BACKEND_CPU);
	ASSERT(metal && cpu);
	if (!metal || !cpu) {
		sam3_backend_free(metal);
		sam3_backend_free(cpu);
		return;
	}

	/* C = softmax(A @ B + bias) */
	int dims_a[] = {2, 3};
	int dims_b[] = {3, 2};
	int dims_c[] = {2, 2};

	float data_a[] = {1, 2, 3, 4, 5, 6};
	float data_b[] = {1, 2, 3, 4, 5, 6};
	float data_bias[] = {10, 20, 30, 40};

	/* --- Metal path --- */
	struct sam3_tensor ma = make_tensor(SAM3_DTYPE_F32, 2, dims_a);
	struct sam3_tensor mb = make_tensor(SAM3_DTYPE_F32, 2, dims_b);
	struct sam3_tensor m_mm = make_tensor(SAM3_DTYPE_F32, 2, dims_c);
	struct sam3_tensor m_bias = make_tensor(SAM3_DTYPE_F32, 2, dims_c);
	struct sam3_tensor m_add = make_tensor(SAM3_DTYPE_F32, 2, dims_c);
	struct sam3_tensor m_out = make_tensor(SAM3_DTYPE_F32, 2, dims_c);

	metal->ops->alloc_tensor(metal, &ma);
	metal->ops->alloc_tensor(metal, &mb);
	metal->ops->alloc_tensor(metal, &m_mm);
	metal->ops->alloc_tensor(metal, &m_bias);
	metal->ops->alloc_tensor(metal, &m_add);
	metal->ops->alloc_tensor(metal, &m_out);
	memcpy(ma.data, data_a, ma.nbytes);
	memcpy(mb.data, data_b, mb.nbytes);
	memcpy(m_bias.data, data_bias, m_bias.nbytes);

	struct sam3_graph mg;
	sam3_graph_init(&mg);
	mg.nodes[0] = (struct sam3_node){
		.op = SAM3_OP_MATMUL, .n_inputs = 2,
		.inputs = {&ma, &mb}, .output = &m_mm,
	};
	mg.nodes[1] = (struct sam3_node){
		.op = SAM3_OP_ADD, .n_inputs = 2,
		.inputs = {&m_mm, &m_bias}, .output = &m_add,
	};
	mg.nodes[2] = (struct sam3_node){
		.op = SAM3_OP_SOFTMAX, .n_inputs = 1,
		.inputs = {&m_add}, .output = &m_out,
	};
	mg.n_nodes = 3;

	ASSERT_EQ(metal->ops->graph_eval(metal, &mg), SAM3_OK);

	/* --- CPU path --- */
	struct sam3_tensor ca = make_tensor(SAM3_DTYPE_F32, 2, dims_a);
	struct sam3_tensor cb = make_tensor(SAM3_DTYPE_F32, 2, dims_b);
	struct sam3_tensor c_mm = make_tensor(SAM3_DTYPE_F32, 2, dims_c);
	struct sam3_tensor c_bias = make_tensor(SAM3_DTYPE_F32, 2, dims_c);
	struct sam3_tensor c_add = make_tensor(SAM3_DTYPE_F32, 2, dims_c);
	struct sam3_tensor c_out = make_tensor(SAM3_DTYPE_F32, 2, dims_c);

	cpu->ops->alloc_tensor(cpu, &ca);
	cpu->ops->alloc_tensor(cpu, &cb);
	cpu->ops->alloc_tensor(cpu, &c_mm);
	cpu->ops->alloc_tensor(cpu, &c_bias);
	cpu->ops->alloc_tensor(cpu, &c_add);
	cpu->ops->alloc_tensor(cpu, &c_out);
	memcpy(ca.data, data_a, ca.nbytes);
	memcpy(cb.data, data_b, cb.nbytes);
	memcpy(c_bias.data, data_bias, c_bias.nbytes);

	struct sam3_graph cg;
	sam3_graph_init(&cg);
	cg.nodes[0] = (struct sam3_node){
		.op = SAM3_OP_MATMUL, .n_inputs = 2,
		.inputs = {&ca, &cb}, .output = &c_mm,
	};
	cg.nodes[1] = (struct sam3_node){
		.op = SAM3_OP_ADD, .n_inputs = 2,
		.inputs = {&c_mm, &c_bias}, .output = &c_add,
	};
	cg.nodes[2] = (struct sam3_node){
		.op = SAM3_OP_SOFTMAX, .n_inputs = 1,
		.inputs = {&c_add}, .output = &c_out,
	};
	cg.n_nodes = 3;

	ASSERT_EQ(cpu->ops->graph_eval(cpu, &cg), SAM3_OK);

	ASSERT(float_arrays_match((float *)m_out.data, (float *)c_out.data,
				  4, 1e-2f));

	sam3_backend_free(metal);
	sam3_backend_free(cpu);
}

#endif /* SAM3_HAS_CPU */

/* ── Test: backend factory ───────────────────────────────────────── */

static void test_backend_factory_metal(void)
{
	struct sam3_backend *be = sam3_backend_init(SAM3_BACKEND_METAL);
	ASSERT(be != NULL);
	if (be) {
		ASSERT_EQ(be->type, SAM3_BACKEND_METAL);
		sam3_backend_free(be);
	}
}

#ifdef SAM3_HAS_CPU
static void test_backend_factory_cpu(void)
{
	struct sam3_backend *be = sam3_backend_init(SAM3_BACKEND_CPU);
	ASSERT(be != NULL);
	if (be) {
		ASSERT_EQ(be->type, SAM3_BACKEND_CPU);
		sam3_backend_free(be);
	}
}
#endif

#endif /* SAM3_HAS_METAL */

/* ── Main ────────────────────────────────────────────────────────── */

int main(void)
{
#ifndef SAM3_HAS_METAL
	printf("Metal not available, skipping Metal tests\n");
	printf("0 tests, 0 failures\n");
	return 0;
#else
#ifdef SAM3_HAS_CPU
	test_backend_factory_cpu();
#endif
	test_backend_factory_metal();
	test_metal_init_free();
	test_metal_alloc_tensor();

#ifdef SAM3_HAS_CPU
	test_metal_matmul();
	test_metal_add();
	test_metal_softmax();
	test_metal_reshape();
	test_metal_multi_node();
#endif

	TEST_REPORT();
#endif
}
