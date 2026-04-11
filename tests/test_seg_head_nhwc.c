/*
 * tests/test_seg_head_nhwc.c - NHWC seg_head vs pre-migration fixtures
 *
 * Rebuilds the small synthetic seg_head used by tools/gen_nhwc_fixtures
 * (d_model=32, n_heads=4, enc grid 8x8, feat_2x 16x16, feat_4x 32x32,
 * 4 queries) on the CPU backend after the NHWC migration and compares
 * the final mask logits against the pre-migration NCHW reference blob
 * captured in tests/fixtures/nhwc_migration/seg_head_masks.bin.
 *
 * Because mask output shape is [n_queries, H, W] (no channel axis) it
 * is layout-invariant, so a direct element-wise compare works provided
 * the seg_head sees the same numeric inputs. This test permutes the
 * original NCHW PRNG byte sequence into NHWC feat_2x / feat_4x tensors
 * so the internal math matches the fixture.
 *
 * Skips gracefully when the fixture directory is missing so the test
 * is safe to ship without the blobs checked into every clone.
 *
 * Key types:  sam3_seg_head, sam3_graph, sam3_cpu_backend, sam3_tensor
 * Depends on: test_helpers.h, model/segmentation.h, model/graph_helpers.h,
 *             backend/cpu/cpu_backend.h, backend/backend.h,
 *             core/graph.h, core/tensor.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "backend/backend.h"
#include "backend/cpu/cpu_backend.h"
#include "core/graph.h"
#include "core/tensor.h"
#include "model/graph_helpers.h"
#include "model/segmentation.h"

#ifndef SAM3_SOURCE_DIR
#error "SAM3_SOURCE_DIR must be defined by CMake"
#endif

#define FIXTURE_DIR SAM3_SOURCE_DIR "/tests/fixtures/nhwc_migration"

/* Mirror the constants used by tools/gen_nhwc_fixtures.c. */
#define SEG_D_MODEL   32
#define SEG_N_HEADS    4
#define SEG_ENC_H      8
#define SEG_ENC_W      8
#define SEG_SEQ       (SEG_ENC_H * SEG_ENC_W)
#define SEG_FEAT2_H   (SEG_ENC_H * 2)
#define SEG_FEAT2_W   (SEG_ENC_W * 2)
#define SEG_FEAT4_H   (SEG_ENC_H * 4)
#define SEG_FEAT4_W   (SEG_ENC_W * 4)
#define SEG_N_QUERIES  4

/* Must match tools/gen_nhwc_fixtures.c: rng = GEN_NHWC_SEED ^ 0x11111111. */
#define GEN_NHWC_SEED 0x5A3ABCDEu
#define SEG_FIXTURE_SEED (GEN_NHWC_SEED ^ 0x11111111u)

static struct sam3_cpu_backend g_cpu;

static void setup(void)
{
	memset(&g_cpu, 0, sizeof(g_cpu));
	g_cpu.base.type = SAM3_BACKEND_CPU;
	g_cpu.base.ops = sam3_cpu_backend_ops();
	g_cpu.arena_capacity = 128UL * 1024 * 1024;
	g_cpu.base.ops->init(&g_cpu.base);
}

static void teardown(void)
{
	g_cpu.base.ops->free(&g_cpu.base);
}

/*
 * fixtures_available - Probe the seg_head_masks.bin fixture before
 * running the comparison so the test degrades to a skip (not a fail)
 * when the NHWC fixture directory is missing from the working copy.
 */
static int fixtures_available(void)
{
	return access(FIXTURE_DIR "/seg_head_masks.bin", F_OK) == 0;
}

/*
 * prng_next / fill_prng - Deterministic LCG identical to the one in
 * tools/gen_nhwc_fixtures.c. Re-implemented inline so the test links
 * against libsam3 alone and does not depend on the fixture tool.
 */
static float prng_next(uint32_t *state)
{
	*state = (*state) * 1664525u + 1013904223u;
	uint32_t bits = (*state) >> 8;
	float unit = (float)bits / (float)(1u << 24);
	return unit - 0.5f;
}

static void fill_prng(float *dst, int n, float amp, uint32_t *state)
{
	for (int i = 0; i < n; i++)
		dst[i] = amp * prng_next(state);
}

/*
 * load_bin - Read a [u32 n_elems][float payload] fixture blob.
 */
static float *load_bin(const char *path, int expected_n, int *out_n)
{
	FILE *fp = fopen(path, "rb");
	if (!fp) {
		fprintf(stderr, "load_bin: open %s failed\n", path);
		return NULL;
	}

	uint32_t hdr = 0;
	if (fread(&hdr, sizeof(hdr), 1, fp) != 1) {
		fprintf(stderr, "load_bin: header read %s failed\n", path);
		fclose(fp);
		return NULL;
	}

	if ((int)hdr != expected_n) {
		fprintf(stderr, "load_bin: %s n=%u expected %d\n",
			path, hdr, expected_n);
		fclose(fp);
		return NULL;
	}

	float *buf = (float *)malloc((size_t)expected_n * sizeof(float));
	if (!buf) {
		fclose(fp);
		return NULL;
	}

	if (fread(buf, sizeof(float), (size_t)expected_n, fp)
	    != (size_t)expected_n) {
		fprintf(stderr, "load_bin: payload read %s failed\n", path);
		free(buf);
		fclose(fp);
		return NULL;
	}

	fclose(fp);
	if (out_n)
		*out_n = expected_n;
	return buf;
}

/*
 * permute_to_ohwi - Copy a Conv2d OIHW flat buffer into an OHWI
 * tensor. Seg head only has forward convs, so is_transpose is not
 * needed.
 */
static void permute_oihw_to_ohwi(const float *src, float *dst,
				  int oc, int ic, int kh, int kw)
{
	for (int o = 0; o < oc; o++) {
		for (int y = 0; y < kh; y++) {
			for (int x = 0; x < kw; x++) {
				for (int c = 0; c < ic; c++) {
					int s = ((o * ic + c) * kh + y)
						* kw + x;
					int d = ((o * kh + y) * kw + x)
						* ic + c;
					dst[d] = src[s];
				}
			}
		}
	}
}

/*
 * nchw_to_nhwc_copy - Permute [1, C, H, W] f32 -> [1, H, W, C] f32.
 *
 * The fixtures were generated with NCHW feat_2x / feat_4x tensors
 * populated via fill_prng in NCHW byte order. To reproduce the same
 * numeric result on the NHWC seg_head, we fill a temporary NCHW buffer
 * with the identical LCG state, then transpose into the actual NHWC
 * tensor data.
 */
static void nchw_to_nhwc_copy(const float *src, float *dst,
			       int N, int C, int H, int W)
{
	for (int n = 0; n < N; n++) {
		for (int h = 0; h < H; h++) {
			for (int w = 0; w < W; w++) {
				for (int c = 0; c < C; c++) {
					int s = ((n * C + c) * H + h) * W
						+ w;
					int d = ((n * H + h) * W + w) * C
						+ c;
					dst[d] = src[s];
				}
			}
		}
	}
}

/*
 * fill_weights - Reproduce fill_weight_tensors_seg() from
 * tools/gen_nhwc_fixtures.c. Conv weights are now OHWI-permuted at
 * load time, so we fill a temporary OIHW buffer with the PRNG and
 * permute into the actual tensors. GN weights, biases, and the mask
 * MLP layers are filled directly because they are 1-D.
 */
static void fill_weights(struct sam3_seg_head *head, uint32_t *state)
{
	int d = head->d_model;

	for (int i = 0; i < SAM3_SEG_FPN_STAGES; i++) {
		struct sam3_tensor *w = head->fpn[i].conv_w;
		struct sam3_tensor *b = head->fpn[i].conv_b;
		if (w) {
			/* OHWI dims [d, 3, 3, d]; fill matching OIHW. */
			int oc = w->dims[0];
			int kh = w->dims[1];
			int kw = w->dims[2];
			int ic = w->dims[3];
			int n = oc * ic * kh * kw;
			float *raw = (float *)malloc(
				(size_t)n * sizeof(float));
			if (raw) {
				fill_prng(raw, n, 0.1f, state);
				permute_oihw_to_ohwi(raw,
					(float *)w->data,
					oc, ic, kh, kw);
				free(raw);
			}
		}
		if (b)
			fill_prng((float *)b->data,
				  sam3_tensor_nelems(b),
				  0.05f, state);

		/* GroupNorm gamma: PRNG + 1.0 */
		if (head->fpn[i].gn_w) {
			float *gw = (float *)head->fpn[i].gn_w->data;
			int gn = sam3_tensor_nelems(head->fpn[i].gn_w);
			fill_prng(gw, gn, 0.1f, state);
			for (int k = 0; k < gn; k++)
				gw[k] += 1.0f;
		}
		if (head->fpn[i].gn_b)
			fill_prng((float *)head->fpn[i].gn_b->data,
				  sam3_tensor_nelems(head->fpn[i].gn_b),
				  0.05f, state);
	}

	if (head->inst_proj_w) {
		struct sam3_tensor *w = head->inst_proj_w;
		int oc = w->dims[0];
		int kh = w->dims[1];
		int kw = w->dims[2];
		int ic = w->dims[3];
		int n = oc * ic * kh * kw;
		float *raw = (float *)malloc((size_t)n * sizeof(float));
		if (raw) {
			fill_prng(raw, n, 0.1f, state);
			permute_oihw_to_ohwi(raw, (float *)w->data,
					      oc, ic, kh, kw);
			free(raw);
		}
	}
	if (head->inst_proj_b)
		fill_prng((float *)head->inst_proj_b->data,
			  sam3_tensor_nelems(head->inst_proj_b),
			  0.05f, state);

	for (int i = 0; i < SAM3_SEG_MASK_MLP_LAYERS; i++) {
		if (head->mask_mlp[i].w)
			fill_prng((float *)head->mask_mlp[i].w->data,
				  sam3_tensor_nelems(head->mask_mlp[i].w),
				  0.1f, state);
		if (head->mask_mlp[i].b)
			fill_prng((float *)head->mask_mlp[i].b->data,
				  sam3_tensor_nelems(head->mask_mlp[i].b),
				  0.05f, state);
	}

	(void)d;
}

/*
 * fill_feat_nhwc - Populate an NHWC feat tensor whose bytes equal the
 * original NCHW fixture byte order after being transposed.
 *
 * The fixture tool called fill_prng on a raw NCHW tensor. We call the
 * same PRNG to fill a temporary NCHW buffer (same element count, same
 * LCG state) and then permute it into the caller-provided NHWC
 * tensor storage.
 */
static void fill_feat_nhwc(struct sam3_tensor *feat_nhwc,
			    int N, int C, int H, int W,
			    uint32_t *state)
{
	int n = N * C * H * W;
	float *raw = (float *)malloc((size_t)n * sizeof(float));
	if (!raw)
		return;

	fill_prng(raw, n, 0.3f, state);
	nchw_to_nhwc_copy(raw, (float *)feat_nhwc->data, N, C, H, W);
	free(raw);
}

/*
 * test_seg_head_nhwc_fixture - Run the migrated seg head and compare
 * the final mask logits against the pre-migration NCHW reference.
 */
static void test_seg_head_nhwc_fixture(void)
{
	if (!fixtures_available()) {
		printf("SKIP: %s not found\n", FIXTURE_DIR);
		return;
	}

	uint32_t rng = SEG_FIXTURE_SEED;
	struct sam3_seg_head head;
	enum sam3_error err;

	err = sam3_seg_head_init(&head, SEG_D_MODEL, SEG_N_HEADS);
	ASSERT_EQ(err, SAM3_OK);

	err = sam3_seg_head_load(&head, NULL, &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	/* Step 1: weights — same LCG order as the fixture tool. */
	fill_weights(&head, &rng);

	/* Step 2: inputs. enc is 2D [SEQ, d] and its byte layout is
	 * the same as NHWC [1, H, W, d]; feat_2x / feat_4x need the
	 * NCHW -> NHWC byte permute to match the original math. */
	int enc_dims[] = {SEG_SEQ, SEG_D_MODEL};
	int f2_dims[]  = {1, SEG_FEAT2_H, SEG_FEAT2_W, SEG_D_MODEL};
	int f4_dims[]  = {1, SEG_FEAT4_H, SEG_FEAT4_W, SEG_D_MODEL};
	int q_dims[]   = {SEG_N_QUERIES, SEG_D_MODEL};

	struct sam3_tensor *enc = gh_alloc_tensor(&g_cpu.arena,
		SAM3_DTYPE_F32, 2, enc_dims);
	struct sam3_tensor *feat_2x = gh_alloc_tensor(&g_cpu.arena,
		SAM3_DTYPE_F32, 4, f2_dims);
	struct sam3_tensor *feat_4x = gh_alloc_tensor(&g_cpu.arena,
		SAM3_DTYPE_F32, 4, f4_dims);
	struct sam3_tensor *queries = gh_alloc_tensor(&g_cpu.arena,
		SAM3_DTYPE_F32, 2, q_dims);
	ASSERT(enc != NULL);
	ASSERT(feat_2x != NULL);
	ASSERT(feat_4x != NULL);
	ASSERT(queries != NULL);

	/* enc: direct fill (SEQ*d == H*W*d in row order) */
	fill_prng((float *)enc->data, SEG_SEQ * SEG_D_MODEL, 0.3f, &rng);

	fill_feat_nhwc(feat_2x, 1, SEG_D_MODEL, SEG_FEAT2_H, SEG_FEAT2_W,
		       &rng);
	fill_feat_nhwc(feat_4x, 1, SEG_D_MODEL, SEG_FEAT4_H, SEG_FEAT4_W,
		       &rng);

	fill_prng((float *)queries->data,
		  SEG_N_QUERIES * SEG_D_MODEL, 0.3f, &rng);

	/* Build the NHWC seg_head graph and evaluate on CPU. */
	struct sam3_graph graph;
	sam3_graph_init(&graph);

	struct sam3_tensor *masks = sam3_seg_head_build(
		&head, &graph, queries, enc, feat_2x, feat_4x,
		SEG_ENC_H, SEG_ENC_W, &g_cpu.arena);
	ASSERT(masks != NULL);

	err = g_cpu.base.ops->graph_eval(&g_cpu.base, &graph);
	ASSERT_EQ(err, SAM3_OK);

	/* masks shape: [n_queries, feat_4x_h, feat_4x_w] */
	ASSERT_EQ(masks->n_dims, 3);
	ASSERT_EQ(masks->dims[0], SEG_N_QUERIES);
	ASSERT_EQ(masks->dims[1], SEG_FEAT4_H);
	ASSERT_EQ(masks->dims[2], SEG_FEAT4_W);

	int nelems = SEG_N_QUERIES * SEG_FEAT4_H * SEG_FEAT4_W;
	float *ref = load_bin(FIXTURE_DIR "/seg_head_masks.bin",
			      nelems, NULL);
	ASSERT(ref != NULL);
	if (!ref)
		return;

	const float *got = (const float *)masks->data;
	for (int i = 0; i < nelems; i++)
		ASSERT_NEAR(got[i], ref[i], 1e-4f);

	free(ref);
}

int main(void)
{
	setup();

	test_seg_head_nhwc_fixture();

	teardown();

	TEST_REPORT();
}
