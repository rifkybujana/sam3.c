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

/* ── Test: 2D axial RoPE (Metal fast path vs CPU manual) ──────────── */

/*
 * precompute_rope_2d - Build cos/sin tables for a 2D grid.
 *
 * Matches image_encoder.c precompute_rope_table() exactly.
 */
static void precompute_rope_2d(float *cos_out, float *sin_out,
				int grid_w, int grid_h, int head_dim,
				float scale, float theta)
{
	int half = head_dim / 2;
	int quarter = head_dim / 4;

	float freqs[16]; /* quarter <= 16 for our tests */
	for (int i = 0; i < quarter; i++)
		freqs[i] = 1.0f / powf(theta,
				(float)(i * 4) / (float)head_dim);

	for (int py = 0; py < grid_h; py++) {
		for (int px = 0; px < grid_w; px++) {
			int pos = py * grid_w + px;
			float sx = (float)px * scale;
			float sy = (float)py * scale;

			for (int i = 0; i < quarter; i++) {
				float ax = sx * freqs[i];
				cos_out[pos * half + i] = cosf(ax);
				sin_out[pos * half + i] = sinf(ax);
			}
			for (int i = 0; i < quarter; i++) {
				float ay = sy * freqs[i];
				cos_out[pos * half + quarter + i] = cosf(ay);
				sin_out[pos * half + quarter + i] = sinf(ay);
			}
		}
	}
}

static void test_metal_rope_axial(void)
{
	struct sam3_backend *metal = sam3_backend_init(SAM3_BACKEND_METAL);
	struct sam3_backend *cpu = sam3_backend_init(SAM3_BACKEND_CPU);
	ASSERT(metal && cpu);
	if (!metal || !cpu) {
		sam3_backend_free(metal);
		sam3_backend_free(cpu);
		return;
	}

	/*
	 * Small grid: 3x2, head_dim=8, 1 head, batch=1.
	 * seq = 6 positions, half_dim = 4, quarter = 2.
	 */
	int grid_w = 3, grid_h = 2;
	int seq = grid_w * grid_h;	/* 6 */
	int heads = 1;
	int head_dim = 8;
	int half_dim = head_dim / 2;
	int batch = 1;
	float pos_scale = 1.0f;

	/* Build cos/sin tables */
	float cos_tbl[6 * 4], sin_tbl[6 * 4]; /* [seq, half_dim] */
	precompute_rope_2d(cos_tbl, sin_tbl, grid_w, grid_h,
			    head_dim, pos_scale, 10000.0f);

	/* Random-ish input data */
	float x_data[6 * 8];
	for (int i = 0; i < 6 * 8; i++)
		x_data[i] = (float)(i + 1) * 0.1f;

	/* --- CPU path (legacy, grid_w=0) --- */
	int x_dims[] = {batch, seq, heads, head_dim};
	int cs_dims[] = {seq, half_dim};
	int out_dims[] = {batch, seq, heads, head_dim};
	int cpu_params[4] = {head_dim, 0, 0, 0};

	struct sam3_tensor cx = make_tensor(SAM3_DTYPE_F32, 4, x_dims);
	struct sam3_tensor cc = make_tensor(SAM3_DTYPE_F32, 2, cs_dims);
	struct sam3_tensor cs = make_tensor(SAM3_DTYPE_F32, 2, cs_dims);
	struct sam3_tensor co = make_tensor(SAM3_DTYPE_F32, 4, out_dims);
	struct sam3_tensor *ci[] = {&cx, &cc, &cs};
	const void *cd[] = {x_data, cos_tbl, sin_tbl};
	ASSERT_EQ(eval_single_op(cpu, SAM3_OP_ROPE, ci, 3,
				  cd, &co, cpu_params), SAM3_OK);

	/* --- Metal path (fast axial, grid_w>0) --- */
	int metal_params[4];
	metal_params[0] = head_dim;
	metal_params[1] = grid_w;
	memcpy(&metal_params[2], &pos_scale, sizeof(float));
	metal_params[3] = 0;

	struct sam3_tensor mx = make_tensor(SAM3_DTYPE_F32, 4, x_dims);
	struct sam3_tensor mc = make_tensor(SAM3_DTYPE_F32, 2, cs_dims);
	struct sam3_tensor ms = make_tensor(SAM3_DTYPE_F32, 2, cs_dims);
	struct sam3_tensor mo = make_tensor(SAM3_DTYPE_F32, 4, out_dims);
	struct sam3_tensor *mi[] = {&mx, &mc, &ms};
	const void *md[] = {x_data, cos_tbl, sin_tbl};
	ASSERT_EQ(eval_single_op(metal, SAM3_OP_ROPE, mi, 3,
				  md, &mo, metal_params), SAM3_OK);

	/* Compare outputs (1e-2 tolerance: Metal runs F16 internally) */
	int nelems = batch * seq * heads * head_dim;
	ASSERT(float_arrays_match((float *)mo.data, (float *)co.data,
				   nelems, 1e-2f));

	sam3_backend_free(metal);
	sam3_backend_free(cpu);
}

static void test_metal_rope_axial_scaled(void)
{
	struct sam3_backend *metal = sam3_backend_init(SAM3_BACKEND_METAL);
	struct sam3_backend *cpu = sam3_backend_init(SAM3_BACKEND_CPU);
	ASSERT(metal && cpu);
	if (!metal || !cpu) {
		sam3_backend_free(metal);
		sam3_backend_free(cpu);
		return;
	}

	/*
	 * 6x6 grid with scale=0.5 (simulating global RoPE with
	 * window_size/grid_size = 0.5).
	 */
	int grid_w = 6, grid_h = 6;
	int seq = grid_w * grid_h;	/* 36 */
	int heads = 2;
	int head_dim = 8;
	int half_dim = head_dim / 2;
	int batch = 1;
	float pos_scale = 0.5f;

	float cos_tbl[36 * 4], sin_tbl[36 * 4];
	precompute_rope_2d(cos_tbl, sin_tbl, grid_w, grid_h,
			    head_dim, pos_scale, 10000.0f);

	float x_data[36 * 2 * 8]; /* seq * heads * head_dim */
	for (int i = 0; i < 36 * 2 * 8; i++)
		x_data[i] = sinf((float)i * 0.37f);

	int x_dims[] = {batch, seq, heads, head_dim};
	int cs_dims[] = {seq, half_dim};
	int out_dims[] = {batch, seq, heads, head_dim};
	int cpu_params[4] = {head_dim, 0, 0, 0};

	struct sam3_tensor cx = make_tensor(SAM3_DTYPE_F32, 4, x_dims);
	struct sam3_tensor cc = make_tensor(SAM3_DTYPE_F32, 2, cs_dims);
	struct sam3_tensor cs = make_tensor(SAM3_DTYPE_F32, 2, cs_dims);
	struct sam3_tensor co = make_tensor(SAM3_DTYPE_F32, 4, out_dims);
	struct sam3_tensor *ci[] = {&cx, &cc, &cs};
	const void *cd[] = {x_data, cos_tbl, sin_tbl};
	ASSERT_EQ(eval_single_op(cpu, SAM3_OP_ROPE, ci, 3,
				  cd, &co, cpu_params), SAM3_OK);

	int metal_params[4];
	metal_params[0] = head_dim;
	metal_params[1] = grid_w;
	memcpy(&metal_params[2], &pos_scale, sizeof(float));
	metal_params[3] = 0;

	struct sam3_tensor mx = make_tensor(SAM3_DTYPE_F32, 4, x_dims);
	struct sam3_tensor mc = make_tensor(SAM3_DTYPE_F32, 2, cs_dims);
	struct sam3_tensor ms = make_tensor(SAM3_DTYPE_F32, 2, cs_dims);
	struct sam3_tensor mo = make_tensor(SAM3_DTYPE_F32, 4, out_dims);
	struct sam3_tensor *mi[] = {&mx, &mc, &ms};
	const void *md[] = {x_data, cos_tbl, sin_tbl};
	ASSERT_EQ(eval_single_op(metal, SAM3_OP_ROPE, mi, 3,
				  md, &mo, metal_params), SAM3_OK);

	/* F16 precision: use 1e-3 absolute floor for near-zero values */
	int nelems = batch * seq * heads * head_dim;
	{
		float *ma = (float *)mo.data;
		float *ca = (float *)co.data;
		int ok = 1;
		for (int i = 0; i < nelems; i++) {
			float diff = fabsf(ma[i] - ca[i]);
			float mag = fmaxf(fabsf(ma[i]), fabsf(ca[i]));
			float tol = fmaxf(1e-2f * mag, 1e-3f);
			if (diff > tol)
				ok = 0;
		}
		ASSERT(ok);
	}

	sam3_backend_free(metal);
	sam3_backend_free(cpu);
}

static void test_metal_rope_axial_batched(void)
{
	struct sam3_backend *metal = sam3_backend_init(SAM3_BACKEND_METAL);
	struct sam3_backend *cpu = sam3_backend_init(SAM3_BACKEND_CPU);
	ASSERT(metal && cpu);
	if (!metal || !cpu) {
		sam3_backend_free(metal);
		sam3_backend_free(cpu);
		return;
	}

	/*
	 * Simulate windowed attention: batch=4 (4 windows),
	 * seq=4 (2x2 window), 1 head, head_dim=8.
	 */
	int grid_w = 2, grid_h = 2;
	int seq = grid_w * grid_h;	/* 4 */
	int heads = 1;
	int head_dim = 8;
	int half_dim = head_dim / 2;
	int batch = 4;
	float pos_scale = 1.0f;

	float cos_tbl[4 * 4], sin_tbl[4 * 4];
	precompute_rope_2d(cos_tbl, sin_tbl, grid_w, grid_h,
			    head_dim, pos_scale, 10000.0f);

	float x_data[4 * 4 * 1 * 8]; /* batch * seq * heads * hd */
	for (int i = 0; i < 4 * 4 * 8; i++)
		x_data[i] = cosf((float)i * 0.13f);

	int x_dims[] = {batch, seq, heads, head_dim};
	int cs_dims[] = {seq, half_dim};
	int out_dims[] = {batch, seq, heads, head_dim};
	int cpu_params[4] = {head_dim, 0, 0, 0};

	struct sam3_tensor cx = make_tensor(SAM3_DTYPE_F32, 4, x_dims);
	struct sam3_tensor cc = make_tensor(SAM3_DTYPE_F32, 2, cs_dims);
	struct sam3_tensor cs = make_tensor(SAM3_DTYPE_F32, 2, cs_dims);
	struct sam3_tensor co = make_tensor(SAM3_DTYPE_F32, 4, out_dims);
	struct sam3_tensor *ci[] = {&cx, &cc, &cs};
	const void *cd[] = {x_data, cos_tbl, sin_tbl};
	ASSERT_EQ(eval_single_op(cpu, SAM3_OP_ROPE, ci, 3,
				  cd, &co, cpu_params), SAM3_OK);

	int metal_params[4];
	metal_params[0] = head_dim;
	metal_params[1] = grid_w;
	memcpy(&metal_params[2], &pos_scale, sizeof(float));
	metal_params[3] = 0;

	struct sam3_tensor mx = make_tensor(SAM3_DTYPE_F32, 4, x_dims);
	struct sam3_tensor mc = make_tensor(SAM3_DTYPE_F32, 2, cs_dims);
	struct sam3_tensor ms = make_tensor(SAM3_DTYPE_F32, 2, cs_dims);
	struct sam3_tensor mo = make_tensor(SAM3_DTYPE_F32, 4, out_dims);
	struct sam3_tensor *mi[] = {&mx, &mc, &ms};
	const void *md[] = {x_data, cos_tbl, sin_tbl};
	ASSERT_EQ(eval_single_op(metal, SAM3_OP_ROPE, mi, 3,
				  md, &mo, metal_params), SAM3_OK);

	int nelems = batch * seq * heads * head_dim;
	ASSERT(float_arrays_match((float *)mo.data, (float *)co.data,
				   nelems, 1e-2f));

	sam3_backend_free(metal);
	sam3_backend_free(cpu);
}

/*
 * Test SDPA with a mask, run twice with the same mask tensor to
 * exercise the mask reshape cache. Compares Metal vs CPU.
 */
static void test_metal_sdpa_mask_cache(void)
{
	struct sam3_backend *metal = sam3_backend_init(SAM3_BACKEND_METAL);
	struct sam3_backend *cpu = sam3_backend_init(SAM3_BACKEND_CPU);
	ASSERT(metal && cpu);
	if (!metal || !cpu) {
		sam3_backend_free(metal);
		sam3_backend_free(cpu);
		return;
	}

	/*
	 * Two SDPA nodes sharing the same mask, simulating two
	 * text encoder layers with a shared causal mask.
	 *
	 * Q, K, V: [2, 4] (seq=2, hd=4)
	 * mask:    [2, 2] (seq_q=2, seq_kv=2) — lower-triangular causal
	 */
	int qkv_dims[] = {2, 4};
	int mask_dims[] = {2, 2};

	float q_data[] = {1, 0, 0, 0, 0, 1, 0, 0};
	float k_data[] = {1, 0, 0, 0, 0, 1, 0, 0};
	float v_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
	float mask_data[] = {0.0f, -1e9f, 0.0f, 0.0f};

	/* --- Metal path: 2-node graph, same mask --- */
	struct sam3_tensor mq1 = make_tensor(SAM3_DTYPE_F32, 2, qkv_dims);
	struct sam3_tensor mk1 = make_tensor(SAM3_DTYPE_F32, 2, qkv_dims);
	struct sam3_tensor mv1 = make_tensor(SAM3_DTYPE_F32, 2, qkv_dims);
	struct sam3_tensor mmask = make_tensor(SAM3_DTYPE_F32, 2, mask_dims);
	struct sam3_tensor mout1 = make_tensor(SAM3_DTYPE_F32, 2, qkv_dims);

	struct sam3_tensor mq2 = make_tensor(SAM3_DTYPE_F32, 2, qkv_dims);
	struct sam3_tensor mk2 = make_tensor(SAM3_DTYPE_F32, 2, qkv_dims);
	struct sam3_tensor mv2 = make_tensor(SAM3_DTYPE_F32, 2, qkv_dims);
	struct sam3_tensor mout2 = make_tensor(SAM3_DTYPE_F32, 2, qkv_dims);

	metal->ops->alloc_tensor(metal, &mq1);
	metal->ops->alloc_tensor(metal, &mk1);
	metal->ops->alloc_tensor(metal, &mv1);
	metal->ops->alloc_tensor(metal, &mmask);
	metal->ops->alloc_tensor(metal, &mout1);
	metal->ops->alloc_tensor(metal, &mq2);
	metal->ops->alloc_tensor(metal, &mk2);
	metal->ops->alloc_tensor(metal, &mv2);
	metal->ops->alloc_tensor(metal, &mout2);

	memcpy(mq1.data, q_data, mq1.nbytes);
	memcpy(mk1.data, k_data, mk1.nbytes);
	memcpy(mv1.data, v_data, mv1.nbytes);
	memcpy(mmask.data, mask_data, mmask.nbytes);
	memcpy(mq2.data, q_data, mq2.nbytes);
	memcpy(mk2.data, k_data, mk2.nbytes);
	memcpy(mv2.data, v_data, mv2.nbytes);

	struct sam3_graph mg;
	sam3_graph_init(&mg);
	mg.nodes[0] = (struct sam3_node){
		.op = SAM3_OP_SDPA, .n_inputs = 4,
		.inputs = {&mq1, &mk1, &mv1, &mmask},
		.output = &mout1,
		.params = {4},  /* head_dim */
	};
	mg.nodes[1] = (struct sam3_node){
		.op = SAM3_OP_SDPA, .n_inputs = 4,
		.inputs = {&mq2, &mk2, &mv2, &mmask},
		.output = &mout2,
		.params = {4},
	};
	mg.n_nodes = 2;

	ASSERT_EQ(metal->ops->graph_eval(metal, &mg), SAM3_OK);

	/* --- CPU path: same 2-node graph --- */
	struct sam3_tensor cq1 = make_tensor(SAM3_DTYPE_F32, 2, qkv_dims);
	struct sam3_tensor ck1 = make_tensor(SAM3_DTYPE_F32, 2, qkv_dims);
	struct sam3_tensor cv1 = make_tensor(SAM3_DTYPE_F32, 2, qkv_dims);
	struct sam3_tensor cmask = make_tensor(SAM3_DTYPE_F32, 2, mask_dims);
	struct sam3_tensor cout1 = make_tensor(SAM3_DTYPE_F32, 2, qkv_dims);

	struct sam3_tensor cq2 = make_tensor(SAM3_DTYPE_F32, 2, qkv_dims);
	struct sam3_tensor ck2 = make_tensor(SAM3_DTYPE_F32, 2, qkv_dims);
	struct sam3_tensor cv2 = make_tensor(SAM3_DTYPE_F32, 2, qkv_dims);
	struct sam3_tensor cout2 = make_tensor(SAM3_DTYPE_F32, 2, qkv_dims);

	cpu->ops->alloc_tensor(cpu, &cq1);
	cpu->ops->alloc_tensor(cpu, &ck1);
	cpu->ops->alloc_tensor(cpu, &cv1);
	cpu->ops->alloc_tensor(cpu, &cmask);
	cpu->ops->alloc_tensor(cpu, &cout1);
	cpu->ops->alloc_tensor(cpu, &cq2);
	cpu->ops->alloc_tensor(cpu, &ck2);
	cpu->ops->alloc_tensor(cpu, &cv2);
	cpu->ops->alloc_tensor(cpu, &cout2);

	memcpy(cq1.data, q_data, cq1.nbytes);
	memcpy(ck1.data, k_data, ck1.nbytes);
	memcpy(cv1.data, v_data, cv1.nbytes);
	memcpy(cmask.data, mask_data, cmask.nbytes);
	memcpy(cq2.data, q_data, cq2.nbytes);
	memcpy(ck2.data, k_data, ck2.nbytes);
	memcpy(cv2.data, v_data, cv2.nbytes);

	struct sam3_graph cg;
	sam3_graph_init(&cg);
	cg.nodes[0] = (struct sam3_node){
		.op = SAM3_OP_SDPA, .n_inputs = 4,
		.inputs = {&cq1, &ck1, &cv1, &cmask},
		.output = &cout1,
		.params = {4},
	};
	cg.nodes[1] = (struct sam3_node){
		.op = SAM3_OP_SDPA, .n_inputs = 4,
		.inputs = {&cq2, &ck2, &cv2, &cmask},
		.output = &cout2,
		.params = {4},
	};
	cg.n_nodes = 2;

	ASSERT_EQ(cpu->ops->graph_eval(cpu, &cg), SAM3_OK);

	/* Both SDPA outputs should match */
	ASSERT(float_arrays_match((float *)mout1.data, (float *)cout1.data,
				  8, 1e-2f));
	ASSERT(float_arrays_match((float *)mout2.data, (float *)cout2.data,
				  8, 1e-2f));

	/* Both nodes got same input → same output */
	ASSERT(float_arrays_match((float *)mout1.data, (float *)mout2.data,
				  8, 1e-6f));

	sam3_backend_free(metal);
	sam3_backend_free(cpu);
}

/* ── Test: SiLU fused kernel ─────────────────────────────────────── */

static void test_metal_silu(void)
{
	struct sam3_backend *metal = sam3_backend_init(SAM3_BACKEND_METAL);
	struct sam3_backend *cpu = sam3_backend_init(SAM3_BACKEND_CPU);
	ASSERT(metal && cpu);
	if (!metal || !cpu) {
		sam3_backend_free(metal);
		sam3_backend_free(cpu);
		return;
	}

	/* 8 values spanning negative, zero, and positive range */
	int dims[] = {8};
	float data[] = {-3.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 5.0f};

	struct sam3_tensor ma = make_tensor(SAM3_DTYPE_F32, 1, dims);
	struct sam3_tensor mc = make_tensor(SAM3_DTYPE_F32, 1, dims);
	struct sam3_tensor *m_inputs[] = {&ma};
	const void *m_data[] = {data};
	ASSERT_EQ(eval_single_op(metal, SAM3_OP_SILU, m_inputs, 1,
				 m_data, &mc, NULL), SAM3_OK);

	struct sam3_tensor ca = make_tensor(SAM3_DTYPE_F32, 1, dims);
	struct sam3_tensor cc = make_tensor(SAM3_DTYPE_F32, 1, dims);
	struct sam3_tensor *c_inputs[] = {&ca};
	const void *c_data[] = {data};
	ASSERT_EQ(eval_single_op(cpu, SAM3_OP_SILU, c_inputs, 1,
				 c_data, &cc, NULL), SAM3_OK);

	ASSERT(float_arrays_match((float *)mc.data, (float *)cc.data,
				  8, 1e-3f));

	sam3_backend_free(metal);
	sam3_backend_free(cpu);
}

static void test_metal_silu_large(void)
{
	struct sam3_backend *metal = sam3_backend_init(SAM3_BACKEND_METAL);
	struct sam3_backend *cpu = sam3_backend_init(SAM3_BACKEND_CPU);
	ASSERT(metal && cpu);
	if (!metal || !cpu) {
		sam3_backend_free(metal);
		sam3_backend_free(cpu);
		return;
	}

	/* 1024 elements: realistic FFN hidden-dim size */
	int n = 1024;
	int dims[] = {n};
	float data[1024];
	for (int i = 0; i < n; i++)
		data[i] = (float)(i - 512) * 0.01f;  /* [-5.12, +5.11] */

	struct sam3_tensor ma = make_tensor(SAM3_DTYPE_F32, 1, dims);
	struct sam3_tensor mc = make_tensor(SAM3_DTYPE_F32, 1, dims);
	struct sam3_tensor *m_inputs[] = {&ma};
	const void *m_data[] = {data};
	ASSERT_EQ(eval_single_op(metal, SAM3_OP_SILU, m_inputs, 1,
				 m_data, &mc, NULL), SAM3_OK);

	struct sam3_tensor ca = make_tensor(SAM3_DTYPE_F32, 1, dims);
	struct sam3_tensor cc = make_tensor(SAM3_DTYPE_F32, 1, dims);
	struct sam3_tensor *c_inputs[] = {&ca};
	const void *c_data[] = {data};
	ASSERT_EQ(eval_single_op(cpu, SAM3_OP_SILU, c_inputs, 1,
				 c_data, &cc, NULL), SAM3_OK);

	ASSERT(float_arrays_match((float *)mc.data, (float *)cc.data,
				  n, 1e-2f));

	sam3_backend_free(metal);
	sam3_backend_free(cpu);
}

/* ── Test: Q8_0 GPU dequantization ────────────────────────────────── */

/*
 * Validate that the GPU Q8_0->F16 dequant kernel produces correct
 * output by constructing 2 full Q8 blocks with known values, running
 * them through Metal (which dequantizes to F16 on-GPU), casting to
 * F32 for readback, and comparing against CPU-computed expected values.
 */
static void test_metal_dequant_q8_gpu(void)
{
	struct sam3_backend *metal = sam3_backend_init(SAM3_BACKEND_METAL);
	ASSERT(metal != NULL);
	if (!metal)
		return;

	/*
	 * 64 elements = 2 full Q8 blocks.
	 * Block 0: scale=0.5,  data[i] = i     (0..31)
	 * Block 1: scale=0.25, data[i] = -(i+1) (-1..-32)
	 */
	struct sam3_q8_block blocks[2];
	memset(&blocks, 0, sizeof(blocks));

	blocks[0].scale = 0.5f;
	for (int i = 0; i < 32; i++)
		blocks[0].data[i] = (int8_t)i;

	blocks[1].scale = 0.25f;
	for (int i = 0; i < 32; i++)
		blocks[1].data[i] = (int8_t)(-(i + 1));

	/* Expected F32 values */
	float expected[64];
	for (int i = 0; i < 32; i++)
		expected[i] = (float)i * 0.5f;
	for (int i = 0; i < 32; i++)
		expected[32 + i] = (float)(-(i + 1)) * 0.25f;

	/* Build Q8 input tensor manually */
	int in_dims[] = {64};
	struct sam3_tensor q8_in = make_tensor(SAM3_DTYPE_Q8_0, 1, in_dims);
	q8_in.data = blocks;
	q8_in.nbytes = sizeof(blocks);
	sam3_tensor_compute_strides(&q8_in);

	/* Allocate F32 output tensor via backend */
	int out_dims[] = {64};
	struct sam3_tensor f32_out = make_tensor(SAM3_DTYPE_F32, 1, out_dims);
	enum sam3_error err = metal->ops->alloc_tensor(metal, &f32_out);
	ASSERT_EQ(err, SAM3_OK);

	/* Build graph: CAST Q8->F32 (wraps Q8 as F16, then casts to F32) */
	struct sam3_graph g;
	sam3_graph_init(&g);
	g.nodes[0] = (struct sam3_node){
		.op = SAM3_OP_CAST, .n_inputs = 1,
		.inputs = {&q8_in}, .output = &f32_out,
	};
	g.n_nodes = 1;

	ASSERT_EQ(metal->ops->graph_eval(metal, &g), SAM3_OK);

	/* Compare Metal output against expected values */
	ASSERT(float_arrays_match((float *)f32_out.data, expected,
				  64, 1e-2f));

	sam3_backend_free(metal);
}

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
	test_metal_rope_axial();
	test_metal_rope_axial_scaled();
	test_metal_rope_axial_batched();
	test_metal_sdpa_mask_cache();
	test_metal_silu();
	test_metal_silu_large();
	test_metal_dequant_q8_gpu();
#endif

	TEST_REPORT();
#endif
}
