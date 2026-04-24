/*
 * tests/test_batched_ops.c - Unit tests for batched building blocks.
 *
 * Each batched op (scorer, seg head, decoder substeps, CPU helpers)
 * is verified here with small synthetic tensors so the tests run
 * without loading a model. Every test loops over both the CPU and
 * Metal backends via run_both_backends(): CPU first (tight tolerance,
 * easier failure localization), Metal second (loose tolerance, the
 * deployment backend). A Metal-only failure indicates backend
 * divergence worth escalating.
 *
 * Key types:  sam3_backend, sam3_arena, sam3_graph, sam3_tensor
 * Depends on: test_helpers.h, backend/backend.h, core/alloc.h,
 *             core/graph.h, model/graph_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "backend/backend.h"
#include "core/alloc.h"
#include "core/graph.h"
#include "model/graph_helpers.h"
#include "model/model_misc.h"
#include "model/segmentation.h"

/*
 * Helper: run a backend-parameterized test case on both CPU and Metal.
 * The callback receives the backend handle and a short human name
 * ("CPU" / "Metal") so failure messages can identify which backend
 * diverged.
 */
typedef void (*backend_test_fn)(struct sam3_backend *be, const char *name);

static void run_both_backends(backend_test_fn fn)
{
	{
		struct sam3_backend *cpu = sam3_backend_init(SAM3_BACKEND_CPU);
		ASSERT_NOT_NULL(cpu);
		fn(cpu, "CPU");
		sam3_backend_free(cpu);
	}
	{
		struct sam3_backend *mtl = sam3_backend_init(SAM3_BACKEND_METAL);
		if (!mtl) {
			printf("  skip Metal: backend unavailable\n");
			return;
		}
		fn(mtl, "Metal");
		sam3_backend_free(mtl);
	}
}

/*
 * scorer_batched_case - Verify batched dot scorer matches per-slot 2D.
 *
 * Runs sam3_dot_scorer_build_batched on B=2 synthetic inputs, then
 * rebuilds the 2D scorer per slot and compares element-wise with a
 * backend-dependent tolerance.
 */
static void scorer_batched_case(struct sam3_backend *be, const char *name)
{
	const int d = 16;
	const int seq = 3;
	const int nq = 8;
	const int B = 2;
	const int d_ffn = 32;
	const float rtol = (be->type == SAM3_BACKEND_METAL) ? 1e-4f : 1e-6f;
	const float atol = (be->type == SAM3_BACKEND_METAL) ? 1e-5f : 1e-6f;

	struct sam3_arena ar;
	sam3_arena_init(&ar, 1 << 22);

	struct sam3_dot_scorer sc = {0};
	sc.d_model = d;
	sc.d_proj = d;
	sc.d_ffn = d_ffn;
	ASSERT_EQ(sam3_dot_scorer_alloc_synthetic(&sc, &ar), 0);

	int qb_dims[] = {B, nq, d};
	int pb_dims[] = {B, seq, d};
	struct sam3_tensor *qb = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 3,
						 qb_dims);
	struct sam3_tensor *pb = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 3,
						 pb_dims);
	ASSERT_NOT_NULL(qb);
	ASSERT_NOT_NULL(pb);
	for (int i = 0; i < B * nq * d; i++)
		((float *)qb->data)[i] = (float)(i % 13) * 0.1f;
	for (int i = 0; i < B * seq * d; i++)
		((float *)pb->data)[i] = (float)(i % 17) * 0.2f;

	struct sam3_graph g;
	sam3_graph_init(&g);
	struct sam3_tensor *sb = sam3_dot_scorer_build_batched(
		&sc, &g, qb, pb, &ar);
	ASSERT_NOT_NULL(sb);
	ASSERT_EQ(be->ops->graph_eval(be, &g), SAM3_OK);
	ASSERT_EQ(sb->n_dims, 3);
	ASSERT_EQ(sb->dims[0], B);
	ASSERT_EQ(sb->dims[1], nq);
	ASSERT_EQ(sb->dims[2], 1);

	for (int b = 0; b < B; b++) {
		int q1_dims[] = {nq, d};
		int p1_dims[] = {seq, d};
		struct sam3_tensor *q1 = gh_alloc_tensor(
			&ar, SAM3_DTYPE_F32, 2, q1_dims);
		struct sam3_tensor *p1 = gh_alloc_tensor(
			&ar, SAM3_DTYPE_F32, 2, p1_dims);
		ASSERT_NOT_NULL(q1);
		ASSERT_NOT_NULL(p1);
		memcpy(q1->data,
		       (char *)qb->data +
			       (size_t)b * nq * d * sizeof(float),
		       (size_t)nq * d * sizeof(float));
		memcpy(p1->data,
		       (char *)pb->data +
			       (size_t)b * seq * d * sizeof(float),
		       (size_t)seq * d * sizeof(float));

		struct sam3_graph g1;
		sam3_graph_init(&g1);
		struct sam3_tensor *s1 = sam3_dot_scorer_build(
			&sc, &g1, q1, p1, &ar);
		ASSERT_NOT_NULL(s1);
		ASSERT_EQ(be->ops->graph_eval(be, &g1), SAM3_OK);
		ASSERT_EQ(s1->dims[0], nq);
		ASSERT_EQ(s1->dims[1], 1);

		const float *bptr = (const float *)sb->data +
			(size_t)b * nq;
		const float *sptr = (const float *)s1->data;
		for (int i = 0; i < nq; i++) {
			tests_run++;
			float expected = sptr[i];
			float actual = bptr[i];
			float tol = atol + rtol * fabsf(expected);
			if (fabsf(actual - expected) > tol) {
				fprintf(stderr,
					"FAIL [%s] b=%d q=%d: "
					"batched=%.6f ref=%.6f tol=%g\n",
					name, b, i, actual, expected,
					tol);
				tests_failed++;
			}
		}
	}

	sam3_arena_free(&ar);
}

static void test_scorer_batched_equals_per_slot(void)
{
	run_both_backends(scorer_batched_case);
}

/*
 * sdpa_4d_batch_case - Derisk the Metal 4D SDPA batching path.
 *
 * The batched decoder (Tasks 11-13) reshapes each per-head attention
 * slice from [nq, hd] to [B, 1, nq, hd] to reuse the existing 4D SDPA
 * path. This smoke test validates that both CPU and Metal SDPA accept
 * B > 1 with a leading-1 head dim and produce per-slot-equivalent
 * output. A failure here would force a decoder-batching strategy pivot
 * before sinking time into Tasks 11-13.
 */
static void sdpa_4d_batch_case(struct sam3_backend *be, const char *name)
{
	const int B = 2;
	const int nq = 4;
	const int nkv = 5;
	const int hd = 8;
	const float rtol = (be->type == SAM3_BACKEND_METAL) ? 1e-4f : 1e-6f;
	const float atol = (be->type == SAM3_BACKEND_METAL) ? 1e-5f : 1e-6f;

	struct sam3_arena ar;
	sam3_arena_init(&ar, 1 << 22);

	int q_dims[] = {B, 1, nq, hd};
	int k_dims[] = {B, 1, nkv, hd};
	struct sam3_tensor *Qb = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 4, q_dims);
	struct sam3_tensor *Kb = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 4, k_dims);
	struct sam3_tensor *Vb = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 4, k_dims);
	ASSERT_NOT_NULL(Qb);
	ASSERT_NOT_NULL(Kb);
	ASSERT_NOT_NULL(Vb);
	for (int i = 0; i < B * nq * hd; i++)
		((float *)Qb->data)[i] = (float)((i * 7 + 3) % 23) * 0.13f;
	for (int i = 0; i < B * nkv * hd; i++) {
		((float *)Kb->data)[i] = (float)((i * 11 + 5) % 19) * 0.09f;
		((float *)Vb->data)[i] = (float)((i * 13 + 7) % 17) * 0.11f;
	}

	struct sam3_graph g;
	sam3_graph_init(&g);
	struct sam3_tensor *outb = gh_sdpa(&g, &ar, Qb, Kb, Vb, NULL, hd);
	ASSERT_NOT_NULL(outb);
	ASSERT_EQ(be->ops->graph_eval(be, &g), SAM3_OK);
	ASSERT_EQ(outb->n_dims, 4);
	ASSERT_EQ(outb->dims[0], B);
	ASSERT_EQ(outb->dims[1], 1);
	ASSERT_EQ(outb->dims[2], nq);
	ASSERT_EQ(outb->dims[3], hd);

	for (int b = 0; b < B; b++) {
		int q1_dims[] = {1, 1, nq, hd};
		int k1_dims[] = {1, 1, nkv, hd};
		struct sam3_tensor *Q1 = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 4, q1_dims);
		struct sam3_tensor *K1 = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 4, k1_dims);
		struct sam3_tensor *V1 = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 4, k1_dims);
		ASSERT_NOT_NULL(Q1);
		ASSERT_NOT_NULL(K1);
		ASSERT_NOT_NULL(V1);
		memcpy(Q1->data,
		       (char *)Qb->data + (size_t)b * nq * hd * sizeof(float),
		       (size_t)nq * hd * sizeof(float));
		memcpy(K1->data,
		       (char *)Kb->data + (size_t)b * nkv * hd * sizeof(float),
		       (size_t)nkv * hd * sizeof(float));
		memcpy(V1->data,
		       (char *)Vb->data + (size_t)b * nkv * hd * sizeof(float),
		       (size_t)nkv * hd * sizeof(float));

		struct sam3_graph g1;
		sam3_graph_init(&g1);
		struct sam3_tensor *out1 = gh_sdpa(&g1, &ar, Q1, K1, V1, NULL, hd);
		ASSERT_NOT_NULL(out1);
		ASSERT_EQ(be->ops->graph_eval(be, &g1), SAM3_OK);

		const float *bptr = (const float *)outb->data +
			(size_t)b * nq * hd;
		const float *sptr = (const float *)out1->data;
		for (int i = 0; i < nq * hd; i++) {
			tests_run++;
			float expected = sptr[i];
			float actual = bptr[i];
			float tol = atol + rtol * fabsf(expected);
			if (fabsf(actual - expected) > tol) {
				fprintf(stderr, "FAIL [%s sdpa] b=%d i=%d: "
					"batched=%.6f ref=%.6f tol=%g\n",
					name, b, i, actual, expected, tol);
				tests_failed++;
			}
		}
	}

	sam3_arena_free(&ar);
}

static void test_sdpa_4d_batch_equals_per_slot(void)
{
	run_both_backends(sdpa_4d_batch_case);
}

/*
 * fill_sin_pattern - Deterministic non-degenerate fill for test tensors.
 *
 * Each call uses a unique offset so swapping two tensors (e.g. a Q
 * weight vs a K weight) would change the output and trip the parity
 * check. sin/cos keep the magnitudes bounded.
 */
static void fill_sin_pattern(struct sam3_tensor *t, float offset)
{
	int n = 1;
	for (int i = 0; i < t->n_dims; i++)
		n *= t->dims[i];
	float *data = (float *)t->data;
	for (int i = 0; i < n; i++)
		data[i] = 0.1f * sinf(offset + (float)i * 0.137f);
}

/*
 * cross_attn_batched_case - Verify batched seg cross-attn matches per-slot 2D.
 *
 * Builds a synthetic seg head (only the prompt cross-attn weights are
 * populated) and checks that the batched builder produces the same
 * output as calling the 2D builder once per batch slot.
 */
static void cross_attn_batched_case(struct sam3_backend *be, const char *name)
{
	const int B = 2;
	const int nq = 9;
	const int ntxt = 4;
	const int d = 16;
	const int n_heads = 2;
	const float rtol = (be->type == SAM3_BACKEND_METAL) ? 1e-4f : 1e-6f;
	const float atol = (be->type == SAM3_BACKEND_METAL) ? 1e-5f : 1e-6f;

	struct sam3_arena ar;
	sam3_arena_init(&ar, 1 << 22);

	struct sam3_seg_head head = {0};
	head.d_model = d;
	head.n_attn_heads = n_heads;

	int d_dims[] = {d};
	int dd_dims[] = {d, d};

	head.pxattn_norm_w = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 1, d_dims);
	head.pxattn_norm_b = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 1, d_dims);
	head.pxattn_q_w    = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 2, dd_dims);
	head.pxattn_q_b    = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 1, d_dims);
	head.pxattn_k_w    = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 2, dd_dims);
	head.pxattn_k_b    = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 1, d_dims);
	head.pxattn_v_w    = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 2, dd_dims);
	head.pxattn_v_b    = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 1, d_dims);
	head.pxattn_o_w    = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 2, dd_dims);
	head.pxattn_o_b    = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 1, d_dims);
	ASSERT_NOT_NULL(head.pxattn_norm_w);
	ASSERT_NOT_NULL(head.pxattn_norm_b);
	ASSERT_NOT_NULL(head.pxattn_q_w);
	ASSERT_NOT_NULL(head.pxattn_q_b);
	ASSERT_NOT_NULL(head.pxattn_k_w);
	ASSERT_NOT_NULL(head.pxattn_k_b);
	ASSERT_NOT_NULL(head.pxattn_v_w);
	ASSERT_NOT_NULL(head.pxattn_v_b);
	ASSERT_NOT_NULL(head.pxattn_o_w);
	ASSERT_NOT_NULL(head.pxattn_o_b);

	/*
	 * Norm weights kept near-identity so the LN step does something
	 * sensible; biases small. Other tensors get unique sin offsets so
	 * weight/bias mix-ups would surface as numeric differences.
	 */
	float *nw = (float *)head.pxattn_norm_w->data;
	float *nb = (float *)head.pxattn_norm_b->data;
	for (int i = 0; i < d; i++) {
		nw[i] = 1.0f + 0.01f * sinf((float)i);
		nb[i] = 0.01f * cosf((float)i);
	}
	fill_sin_pattern(head.pxattn_q_w, 1.0f);
	fill_sin_pattern(head.pxattn_q_b, 2.0f);
	fill_sin_pattern(head.pxattn_k_w, 3.0f);
	fill_sin_pattern(head.pxattn_k_b, 4.0f);
	fill_sin_pattern(head.pxattn_v_w, 5.0f);
	fill_sin_pattern(head.pxattn_v_b, 6.0f);
	fill_sin_pattern(head.pxattn_o_w, 7.0f);
	fill_sin_pattern(head.pxattn_o_b, 8.0f);

	int x_dims[] = {B, nq, d};
	int t_dims[] = {B, ntxt, d};
	struct sam3_tensor *xb = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 3,
						 x_dims);
	struct sam3_tensor *tb = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 3,
						 t_dims);
	ASSERT_NOT_NULL(xb);
	ASSERT_NOT_NULL(tb);
	for (int i = 0; i < B * nq * d; i++)
		((float *)xb->data)[i] = (float)((i * 7 + 3) % 23) * 0.13f;
	for (int i = 0; i < B * ntxt * d; i++)
		((float *)tb->data)[i] = (float)((i * 11 + 5) % 19) * 0.09f;

	struct sam3_graph g;
	sam3_graph_init(&g);
	struct sam3_tensor *outb = sam3_seg_head_build_cross_attn_batched(
		&head, &g, xb, tb, &ar);
	ASSERT_NOT_NULL(outb);
	ASSERT_EQ(be->ops->graph_eval(be, &g), SAM3_OK);
	ASSERT_EQ(outb->n_dims, 3);
	ASSERT_EQ(outb->dims[0], B);
	ASSERT_EQ(outb->dims[1], nq);
	ASSERT_EQ(outb->dims[2], d);

	for (int b = 0; b < B; b++) {
		int x1_dims[] = {nq, d};
		int t1_dims[] = {ntxt, d};
		struct sam3_tensor *x1 = gh_alloc_tensor(
			&ar, SAM3_DTYPE_F32, 2, x1_dims);
		struct sam3_tensor *t1 = gh_alloc_tensor(
			&ar, SAM3_DTYPE_F32, 2, t1_dims);
		ASSERT_NOT_NULL(x1);
		ASSERT_NOT_NULL(t1);
		memcpy(x1->data,
		       (char *)xb->data +
			       (size_t)b * nq * d * sizeof(float),
		       (size_t)nq * d * sizeof(float));
		memcpy(t1->data,
		       (char *)tb->data +
			       (size_t)b * ntxt * d * sizeof(float),
		       (size_t)ntxt * d * sizeof(float));

		struct sam3_graph g1;
		sam3_graph_init(&g1);
		struct sam3_tensor *out1 = sam3_seg_head_build_cross_attn(
			&head, &g1, x1, t1, &ar);
		ASSERT_NOT_NULL(out1);
		ASSERT_EQ(be->ops->graph_eval(be, &g1), SAM3_OK);
		ASSERT_EQ(out1->n_dims, 2);
		ASSERT_EQ(out1->dims[0], nq);
		ASSERT_EQ(out1->dims[1], d);

		const float *bptr = (const float *)outb->data +
			(size_t)b * nq * d;
		const float *sptr = (const float *)out1->data;
		for (int i = 0; i < nq * d; i++) {
			tests_run++;
			float expected = sptr[i];
			float actual = bptr[i];
			float tol = atol + rtol * fabsf(expected);
			if (fabsf(actual - expected) > tol) {
				fprintf(stderr,
					"FAIL [%s xattn] b=%d i=%d: "
					"batched=%.6f ref=%.6f tol=%g\n",
					name, b, i, actual, expected, tol);
				tests_failed++;
			}
		}
	}

	sam3_arena_free(&ar);
}

static void test_cross_attn_batched_equals_per_slot(void)
{
	run_both_backends(cross_attn_batched_case);
}

int main(void)
{
	test_scorer_batched_equals_per_slot();
	test_sdpa_4d_batch_equals_per_slot();
	test_cross_attn_batched_equals_per_slot();
	TEST_REPORT();
}
