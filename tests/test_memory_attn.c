/*
 * tests/test_memory_attn.c - Memory attention unit tests
 *
 * Tests initialization, weight loading (zero-init path), and graph
 * construction for the 4-layer RoPE cross-attention transformer.
 * Uses the CPU backend to evaluate the built graph and verify output
 * shapes and basic numerical properties.
 *
 * Key types:  sam3_memory_attn
 * Depends on: test_helpers.h, model/memory_attn.h,
 *             backend/cpu/cpu_backend.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <string.h>

#include "test_helpers.h"
#include "model/memory_attn.h"
#include "model/graph_helpers.h"
#include "backend/cpu/cpu_backend.h"
#include "backend/backend.h"
#include "core/graph.h"

/* --- test infrastructure --- */

static struct sam3_cpu_backend g_cpu;

static void setup(void)
{
	memset(&g_cpu, 0, sizeof(g_cpu));
	g_cpu.base.type = SAM3_BACKEND_CPU;
	g_cpu.base.ops = sam3_cpu_backend_ops();
	g_cpu.arena_capacity = 512 * 1024 * 1024; /* 512 MiB */
	g_cpu.base.ops->init(&g_cpu.base);
}

static void teardown(void)
{
	g_cpu.base.ops->free(&g_cpu.base);
}

/* --- test_mem_attn_init --- */

static void test_mem_attn_init(void)
{
	struct sam3_memory_attn attn;
	enum sam3_error err = sam3_memory_attn_init(
		&attn, 256, 64, 4, 1, 72, 72);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT_EQ(attn.d_model, 256);
	ASSERT_EQ(attn.mem_dim, 64);
	ASSERT_EQ(attn.n_layers, 4);
	ASSERT_EQ(attn.n_heads, 1);
	ASSERT_EQ(attn.feat_h, 72);
	ASSERT_EQ(attn.feat_w, 72);
}

/* --- test_mem_attn_init_bad_args --- */

static void test_mem_attn_init_bad_args(void)
{
	struct sam3_memory_attn attn;

	ASSERT_EQ(sam3_memory_attn_init(NULL, 256, 64, 4, 1, 72, 72),
		  SAM3_EINVAL);
	ASSERT_EQ(sam3_memory_attn_init(&attn, 0, 64, 4, 1, 72, 72),
		  SAM3_EINVAL);
	ASSERT_EQ(sam3_memory_attn_init(&attn, 256, -1, 4, 1, 72, 72),
		  SAM3_EINVAL);
	ASSERT_EQ(sam3_memory_attn_init(&attn, 256, 64, 0, 1, 72, 72),
		  SAM3_EINVAL);
	ASSERT_EQ(sam3_memory_attn_init(&attn, 256, 64, 4, 0, 72, 72),
		  SAM3_EINVAL);
	ASSERT_EQ(sam3_memory_attn_init(&attn, 256, 64, 4, 1, 0, 72),
		  SAM3_EINVAL);
	ASSERT_EQ(sam3_memory_attn_init(&attn, 256, 64, 4, 1, 72, 0),
		  SAM3_EINVAL);
}

/* --- test_mem_attn_load_null_wf --- */

static void test_mem_attn_load_null_wf(void)
{
	struct sam3_memory_attn attn;
	sam3_memory_attn_init(&attn, 256, 64, 4, 1, 72, 72);

	enum sam3_error err = sam3_memory_attn_load(
		&attn, NULL, &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	/* Verify all layer 0 weights are allocated */
	struct sam3_memattn_layer *l0 = &attn.layers[0];
	ASSERT(l0->sa_q_w != NULL);
	ASSERT(l0->sa_q_b != NULL);
	ASSERT(l0->sa_k_w != NULL);
	ASSERT(l0->sa_k_b != NULL);
	ASSERT(l0->sa_v_w != NULL);
	ASSERT(l0->sa_v_b != NULL);
	ASSERT(l0->sa_out_w != NULL);
	ASSERT(l0->sa_out_b != NULL);
	ASSERT(l0->ca_q_w != NULL);
	ASSERT(l0->ca_q_b != NULL);
	ASSERT(l0->ca_k_w != NULL);
	ASSERT(l0->ca_k_b != NULL);
	ASSERT(l0->ca_v_w != NULL);
	ASSERT(l0->ca_v_b != NULL);
	ASSERT(l0->ca_out_w != NULL);
	ASSERT(l0->ca_out_b != NULL);
	ASSERT(l0->norm1_w != NULL);
	ASSERT(l0->norm1_b != NULL);
	ASSERT(l0->norm2_w != NULL);
	ASSERT(l0->norm2_b != NULL);
	ASSERT(l0->norm3_w != NULL);
	ASSERT(l0->norm3_b != NULL);
	ASSERT(l0->ffn_fc1_w != NULL);
	ASSERT(l0->ffn_fc1_b != NULL);
	ASSERT(l0->ffn_fc2_w != NULL);
	ASSERT(l0->ffn_fc2_b != NULL);

	/* Verify last layer weights too */
	struct sam3_memattn_layer *l3 = &attn.layers[3];
	ASSERT(l3->sa_q_w != NULL);
	ASSERT(l3->ca_k_w != NULL);
	ASSERT(l3->ffn_fc2_b != NULL);

	/* Verify self-attention weight shapes: [d_model, d_model] */
	ASSERT_EQ(l0->sa_q_w->n_dims, 2);
	ASSERT_EQ(l0->sa_q_w->dims[0], 256);
	ASSERT_EQ(l0->sa_q_w->dims[1], 256);

	/* Verify cross-attention K/V shapes: [d_model, mem_dim] */
	ASSERT_EQ(l0->ca_k_w->n_dims, 2);
	ASSERT_EQ(l0->ca_k_w->dims[0], 256);
	ASSERT_EQ(l0->ca_k_w->dims[1], 64);

	ASSERT_EQ(l0->ca_v_w->n_dims, 2);
	ASSERT_EQ(l0->ca_v_w->dims[0], 256);
	ASSERT_EQ(l0->ca_v_w->dims[1], 64);

	/* Cross-attention Q is [d_model, d_model] */
	ASSERT_EQ(l0->ca_q_w->dims[0], 256);
	ASSERT_EQ(l0->ca_q_w->dims[1], 256);

	/* FFN shapes: fc1 [2048, 256], fc2 [256, 2048] */
	ASSERT_EQ(l0->ffn_fc1_w->dims[0], 2048);
	ASSERT_EQ(l0->ffn_fc1_w->dims[1], 256);
	ASSERT_EQ(l0->ffn_fc2_w->dims[0], 256);
	ASSERT_EQ(l0->ffn_fc2_w->dims[1], 2048);

	/* Final LayerNorm */
	ASSERT(attn.final_norm_w != NULL);
	ASSERT(attn.final_norm_b != NULL);
	ASSERT_EQ(attn.final_norm_w->dims[0], 256);

	/* RoPE tables: [5184, 128] */
	ASSERT(attn.rope_cos != NULL);
	ASSERT(attn.rope_sin != NULL);
	ASSERT_EQ(attn.rope_cos->n_dims, 2);
	ASSERT_EQ(attn.rope_cos->dims[0], 72 * 72); /* 5184 */
	ASSERT_EQ(attn.rope_cos->dims[1], 128);      /* head_dim/2 = 256/2 */
	ASSERT_EQ(attn.rope_sin->dims[0], 72 * 72);
	ASSERT_EQ(attn.rope_sin->dims[1], 128);
}

/* --- test_mem_attn_build_shapes --- */

static void test_mem_attn_build_shapes(void)
{
	struct sam3_memory_attn attn;
	sam3_memory_attn_init(&attn, 256, 64, 4, 1, 72, 72);
	sam3_memory_attn_load(&attn, NULL, &g_cpu.arena);

	struct sam3_graph graph;
	sam3_graph_init(&graph);

	/*
	 * current features: [seq, d_model] = [5184, 256]
	 * Full 72x72 grid is large but necessary for RoPE tables.
	 * Use a smaller grid (16x16=256 tokens) to keep tests fast.
	 * Re-init with smaller feat size for the build test.
	 */
	struct sam3_memory_attn attn_small;
	sam3_memory_attn_init(&attn_small, 256, 64, 4, 1, 16, 16);
	sam3_memory_attn_load(&attn_small, NULL, &g_cpu.arena);

	int seq = 16 * 16; /* 256 */
	int d_model = 256;
	int mem_dim = 64;
	int n_mem = 32; /* number of memory tokens */

	int cur_dims[] = {seq, d_model};
	struct sam3_tensor *current = gh_alloc_tensor(
		&g_cpu.arena, SAM3_DTYPE_F32, 2, cur_dims);
	ASSERT(current != NULL);

	int mem_dims[] = {n_mem, mem_dim};
	struct sam3_tensor *memory = gh_alloc_tensor(
		&g_cpu.arena, SAM3_DTYPE_F32, 2, mem_dims);
	ASSERT(memory != NULL);

	struct sam3_tensor *out = NULL;

	enum sam3_error err = sam3_memory_attn_build_full(
		&attn_small, &graph, current, memory, NULL,
		&g_cpu.arena, &out);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT(out != NULL);

	/* Output shape: [seq, d_model] = [256, 256] */
	ASSERT_EQ(out->n_dims, 2);
	ASSERT_EQ(out->dims[0], seq);
	ASSERT_EQ(out->dims[1], d_model);

	/* Evaluate graph to verify no crashes */
	err = g_cpu.base.ops->graph_eval(&g_cpu.base, &graph);
	ASSERT_EQ(err, SAM3_OK);

	/* Verify output is finite (all zeros with zero-init weights) */
	float *data = (float *)out->data;
	int n = out->dims[0] * out->dims[1];
	for (int i = 0; i < n; i++) {
		ASSERT(data[i] == data[i]); /* not NaN */
	}
}

/* --- test_mem_attn_build_null_args --- */

static void test_mem_attn_build_null_args(void)
{
	struct sam3_memory_attn attn;
	sam3_memory_attn_init(&attn, 256, 64, 4, 1, 72, 72);

	struct sam3_graph graph;
	sam3_graph_init(&graph);

	struct sam3_tensor dummy;
	memset(&dummy, 0, sizeof(dummy));

	struct sam3_tensor *out = NULL;

	ASSERT_EQ(sam3_memory_attn_build_full(
		NULL, &graph, &dummy, &dummy, NULL,
		&g_cpu.arena, &out), SAM3_EINVAL);
	ASSERT_EQ(sam3_memory_attn_build_full(
		&attn, NULL, &dummy, &dummy, NULL,
		&g_cpu.arena, &out), SAM3_EINVAL);
	ASSERT_EQ(sam3_memory_attn_build_full(
		&attn, &graph, NULL, &dummy, NULL,
		&g_cpu.arena, &out), SAM3_EINVAL);
	ASSERT_EQ(sam3_memory_attn_build_full(
		&attn, &graph, &dummy, NULL, NULL,
		&g_cpu.arena, &out), SAM3_EINVAL);
	ASSERT_EQ(sam3_memory_attn_build_full(
		&attn, &graph, &dummy, &dummy, NULL,
		NULL, &out), SAM3_EINVAL);
	ASSERT_EQ(sam3_memory_attn_build_full(
		&attn, &graph, &dummy, &dummy, NULL,
		&g_cpu.arena, NULL), SAM3_EINVAL);
}

/* --- main --- */

int main(void)
{
	test_mem_attn_init();
	test_mem_attn_init_bad_args();

	setup();

	test_mem_attn_load_null_wf();
	test_mem_attn_build_shapes();
	test_mem_attn_build_null_args();

	teardown();

	TEST_REPORT();
}
