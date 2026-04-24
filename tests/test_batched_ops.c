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

/*
 * broadcast_batch_case - Verify gh_broadcast_batch tiles correctly.
 *
 * Pure CPU op (no graph node, just an arena memcpy loop), so the
 * backend handle is unused — we run the case once standalone instead
 * of through run_both_backends to keep the comment honest.
 */
static void broadcast_batch_case(struct sam3_backend *be, const char *name)
{
	(void)be; /* CPU-only op; backend irrelevant but symmetry with other cases */
	(void)name;

	struct sam3_arena ar;
	sam3_arena_init(&ar, 1 << 20);

	/* Shape [1, 3, 4, 2] with a deterministic fill. */
	int in_dims[] = {1, 3, 4, 2};
	struct sam3_tensor *in = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 4, in_dims);
	ASSERT_NOT_NULL(in);
	int n_in = 1 * 3 * 4 * 2;
	for (int i = 0; i < n_in; i++)
		((float *)in->data)[i] = (float)(i * 3 + 1) * 0.25f;

	struct sam3_tensor *out = gh_broadcast_batch(&ar, in, 3);
	ASSERT_NOT_NULL(out);
	ASSERT_EQ(out->n_dims, 4);
	ASSERT_EQ(out->dims[0], 3);
	ASSERT_EQ(out->dims[1], 3);
	ASSERT_EQ(out->dims[2], 4);
	ASSERT_EQ(out->dims[3], 2);

	/* Each batch slot must contain the full input payload. */
	for (int b = 0; b < 3; b++) {
		const float *slot = (const float *)out->data +
			(size_t)b * n_in;
		const float *src  = (const float *)in->data;
		ASSERT_EQ(memcmp(slot, src,
				 (size_t)n_in * sizeof(float)), 0);
	}

	/* Also test rank without a leading-1 collapse:
	 * [H, W, C] -> [B, H, W, C]. */
	int raw_dims[] = {2, 3, 2};
	struct sam3_tensor *raw = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 3,
						  raw_dims);
	ASSERT_NOT_NULL(raw);
	int n_raw = 2 * 3 * 2;
	for (int i = 0; i < n_raw; i++)
		((float *)raw->data)[i] = (float)i + 0.5f;

	struct sam3_tensor *raw_out = gh_broadcast_batch(&ar, raw, 4);
	ASSERT_NOT_NULL(raw_out);
	ASSERT_EQ(raw_out->n_dims, 4);
	ASSERT_EQ(raw_out->dims[0], 4);
	ASSERT_EQ(raw_out->dims[1], 2);
	ASSERT_EQ(raw_out->dims[2], 3);
	ASSERT_EQ(raw_out->dims[3], 2);

	for (int b = 0; b < 4; b++) {
		const float *slot = (const float *)raw_out->data +
			(size_t)b * n_raw;
		const float *src  = (const float *)raw->data;
		ASSERT_EQ(memcmp(slot, src,
				 (size_t)n_raw * sizeof(float)), 0);
	}

	sam3_arena_free(&ar);
}

static void test_broadcast_batch(void)
{
	/* Pure CPU op — no backend needed, but call once just for count. */
	broadcast_batch_case(NULL, "CPU-direct");
}

/*
 * fpn_batched_case - Verify batched FPN+inst_proj matches per-slot.
 *
 * Builds a synthetic seg head with FPN + instance projection weights
 * filled deterministically, then checks that
 * sam3_seg_head_build_fpn_batched on B=2 inputs matches calling the
 * 2D builder once per slot. Spatial dims are kept small because CPU
 * conv2d is slow at 36×36×16 channels.
 */
static void fpn_batched_case(struct sam3_backend *be, const char *name)
{
	const int B = 2;
	const int d = 16;
	const int eh = 9, ew = 9;
	const int h2 = 18, w2 = 18;
	const int h4 = 36, w4 = 36;
	const float rtol = (be->type == SAM3_BACKEND_METAL) ? 1e-4f : 1e-6f;
	const float atol = (be->type == SAM3_BACKEND_METAL) ? 1e-5f : 1e-6f;

	struct sam3_arena ar;
	sam3_arena_init(&ar, 1 << 25);  /* 32 MiB for 36x36 FPN scratch */

	struct sam3_seg_head head = {0};
	head.d_model = d;
	head.n_attn_heads = 1;  /* unused in FPN path */

	/*
	 * FPN weights: 3 stages of 3x3 OHWI conv [d, 3, 3, d] + bias [d]
	 * plus GroupNorm scale/shift [d]. Inst proj is a 1x1 OHWI conv
	 * [d, 1, 1, d] + bias [d]. Use unique sin offsets per tensor so a
	 * conv/norm mix-up would surface as a numeric difference.
	 */
	int conv_w_dims[] = {d, 3, 3, d};
	int proj_w_dims[] = {d, 1, 1, d};
	int d_dims[] = {d};
	float off = 0.5f;
	for (int i = 0; i < SAM3_SEG_FPN_STAGES; i++) {
		head.fpn[i].conv_w = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 4,
						     conv_w_dims);
		head.fpn[i].conv_b = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 1,
						     d_dims);
		head.fpn[i].gn_w   = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 1,
						     d_dims);
		head.fpn[i].gn_b   = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 1,
						     d_dims);
		ASSERT_NOT_NULL(head.fpn[i].conv_w);
		ASSERT_NOT_NULL(head.fpn[i].conv_b);
		ASSERT_NOT_NULL(head.fpn[i].gn_w);
		ASSERT_NOT_NULL(head.fpn[i].gn_b);
		fill_sin_pattern(head.fpn[i].conv_w, off);    off += 1.0f;
		fill_sin_pattern(head.fpn[i].conv_b, off);    off += 1.0f;
		/*
		 * GroupNorm scale near identity so the FPN output stays
		 * numerically tame; bias small.
		 */
		float *gw = (float *)head.fpn[i].gn_w->data;
		float *gb = (float *)head.fpn[i].gn_b->data;
		for (int c = 0; c < d; c++) {
			gw[c] = 1.0f + 0.01f * sinf(off + (float)c);
			gb[c] = 0.01f * cosf(off + (float)c);
		}
		off += 1.0f;
	}
	head.inst_proj_w = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 4, proj_w_dims);
	head.inst_proj_b = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 1, d_dims);
	ASSERT_NOT_NULL(head.inst_proj_w);
	ASSERT_NOT_NULL(head.inst_proj_b);
	fill_sin_pattern(head.inst_proj_w, off);    off += 1.0f;
	fill_sin_pattern(head.inst_proj_b, off);    off += 1.0f;

	/* Batched inputs. Per-slot-different fill so slot aliasing surfaces. */
	int eb_dims[] = {B, eh, ew, d};
	int f2b_dims[] = {B, h2, w2, d};
	int f4b_dims[] = {B, h4, w4, d};
	struct sam3_tensor *eb  = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 4, eb_dims);
	struct sam3_tensor *f2b = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 4, f2b_dims);
	struct sam3_tensor *f4b = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 4, f4b_dims);
	ASSERT_NOT_NULL(eb);
	ASSERT_NOT_NULL(f2b);
	ASSERT_NOT_NULL(f4b);
	for (int i = 0; i < B * eh * ew * d; i++)
		((float *)eb->data)[i] = sinf(0.031f * (float)i + 0.1f);
	for (int i = 0; i < B * h2 * w2 * d; i++)
		((float *)f2b->data)[i] = sinf(0.017f * (float)i + 0.3f);
	for (int i = 0; i < B * h4 * w4 * d; i++)
		((float *)f4b->data)[i] = sinf(0.013f * (float)i + 0.5f);

	struct sam3_graph g;
	sam3_graph_init(&g);
	struct sam3_tensor *outb = sam3_seg_head_build_fpn_batched(
		&head, &g, eb, f2b, f4b, &ar);
	ASSERT_NOT_NULL(outb);
	ASSERT_EQ(be->ops->graph_eval(be, &g), SAM3_OK);
	ASSERT_EQ(outb->n_dims, 4);
	ASSERT_EQ(outb->dims[0], B);
	ASSERT_EQ(outb->dims[1], h4);
	ASSERT_EQ(outb->dims[2], w4);
	ASSERT_EQ(outb->dims[3], d);

	/* Per-slot reference: N=1 FPN on each batch slice. */
	for (int b = 0; b < B; b++) {
		int e1_dims[]  = {1, eh, ew, d};
		int f2_1_dims[] = {1, h2, w2, d};
		int f4_1_dims[] = {1, h4, w4, d};
		struct sam3_tensor *e1   = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 4, e1_dims);
		struct sam3_tensor *f2_1 = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 4, f2_1_dims);
		struct sam3_tensor *f4_1 = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 4, f4_1_dims);
		ASSERT_NOT_NULL(e1);
		ASSERT_NOT_NULL(f2_1);
		ASSERT_NOT_NULL(f4_1);
		memcpy(e1->data,
		       (char *)eb->data +
			       (size_t)b * eh * ew * d * sizeof(float),
		       (size_t)eh * ew * d * sizeof(float));
		memcpy(f2_1->data,
		       (char *)f2b->data +
			       (size_t)b * h2 * w2 * d * sizeof(float),
		       (size_t)h2 * w2 * d * sizeof(float));
		memcpy(f4_1->data,
		       (char *)f4b->data +
			       (size_t)b * h4 * w4 * d * sizeof(float),
		       (size_t)h4 * w4 * d * sizeof(float));

		struct sam3_graph g1;
		sam3_graph_init(&g1);
		struct sam3_tensor *out1 = sam3_seg_head_build_fpn(
			&head, &g1, e1, f2_1, f4_1, &ar);
		ASSERT_NOT_NULL(out1);
		ASSERT_EQ(be->ops->graph_eval(be, &g1), SAM3_OK);

		const float *bptr = (const float *)outb->data +
			(size_t)b * h4 * w4 * d;
		const float *sptr = (const float *)out1->data;
		int n = h4 * w4 * d;
		int slot_fails = 0;
		for (int i = 0; i < n; i++) {
			tests_run++;
			float expected = sptr[i];
			float actual = bptr[i];
			float tol = atol + rtol * fabsf(expected);
			if (fabsf(actual - expected) > tol) {
				if (slot_fails < 10) {
					fprintf(stderr,
						"FAIL [%s fpn] b=%d i=%d: "
						"batched=%.6f ref=%.6f tol=%g\n",
						name, b, i, actual,
						expected, tol);
				}
				tests_failed++;
				slot_fails++;
			}
		}
	}

	sam3_arena_free(&ar);
}

static void test_fpn_batched_equals_per_slot(void)
{
	run_both_backends(fpn_batched_case);
}

int main(void)
{
	test_scorer_batched_equals_per_slot();
	test_sdpa_4d_batch_equals_per_slot();
	test_cross_attn_batched_equals_per_slot();
	test_broadcast_batch();
	test_fpn_batched_equals_per_slot();
	TEST_REPORT();
}
