/*
 * tests/test_mask_decoder_nhwc.c - NHWC mask decoder vs pre-migration fixture
 *
 * Rebuilds the small synthetic mask decoder used by
 * tools/gen_nhwc_fixtures (d_model=256, grid 4x4, feat_s1 8x8,
 * feat_s0 16x16) on the CPU backend after the NHWC migration and
 * compares the final mask logits against the pre-migration NCHW
 * reference blob captured in
 * tests/fixtures/nhwc_migration/mask_dec_output.bin.
 *
 * The mask output shape is [n_masks, H, W] (no channel axis), so the
 * blob is layout-invariant and a direct element-wise compare works
 * as long as the mask decoder sees numerically identical inputs. The
 * pre-migration fixture was produced with NCHW feat_s0 / feat_s1 and
 * OIHW/IOHW conv weight buffers fed to fill_prng; to reproduce the
 * same math on the NHWC pipeline this test fills a temporary buffer
 * with the identical LCG state and then permutes into the real tensor
 * storage (NHWC for features, OHWI for conv weights).
 *
 * Skips gracefully when the fixture directory is missing so the test
 * is safe to ship without the blobs checked into every clone.
 *
 * Key types:  sam3_mask_decoder, sam3_graph, sam3_cpu_backend,
 *             sam3_tensor
 * Depends on: test_helpers.h, model/mask_decoder.h,
 *             model/graph_helpers.h, backend/cpu/cpu_backend.h,
 *             backend/backend.h, core/graph.h, core/tensor.h
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
#include "model/mask_decoder.h"

#ifndef SAM3_SOURCE_DIR
#error "SAM3_SOURCE_DIR must be defined by CMake"
#endif

#define FIXTURE_DIR SAM3_SOURCE_DIR "/tests/fixtures/nhwc_migration"

/* Mirror the constants used by tools/gen_nhwc_fixtures.c. */
#define MD_GRID_H    4
#define MD_GRID_W    4
#define MD_N_PIX     (MD_GRID_H * MD_GRID_W)
#define MD_D_MODEL   256
#define MD_FEAT_S1_H (MD_GRID_H * 2)
#define MD_FEAT_S1_W (MD_GRID_W * 2)
#define MD_FEAT_S0_H (MD_GRID_H * 4)
#define MD_FEAT_S0_W (MD_GRID_W * 4)
#define MD_FINAL_H   (MD_GRID_H * 4)
#define MD_FINAL_W   (MD_GRID_W * 4)
#define MD_N_MASKS   4

/* Must match tools/gen_nhwc_fixtures.c: rng = GEN_NHWC_SEED ^ 0x22222222. */
#define GEN_NHWC_SEED 0x5A3ABCDEu
#define MD_FIXTURE_SEED (GEN_NHWC_SEED ^ 0x22222222u)

static struct sam3_cpu_backend g_cpu;

static void setup(void)
{
	memset(&g_cpu, 0, sizeof(g_cpu));
	g_cpu.base.type = SAM3_BACKEND_CPU;
	g_cpu.base.ops = sam3_cpu_backend_ops();
	g_cpu.arena_capacity = 256UL * 1024 * 1024;
	g_cpu.base.ops->init(&g_cpu.base);
}

static void teardown(void)
{
	g_cpu.base.ops->free(&g_cpu.base);
}

/*
 * fixtures_available - Probe the mask_dec_output.bin fixture before
 * running the comparison so the test degrades to a skip (not a fail)
 * when the NHWC fixture directory is missing from the working copy.
 */
static int fixtures_available(void)
{
	return access(FIXTURE_DIR "/mask_dec_output.bin", F_OK) == 0;
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
 * permute_oihw_to_ohwi - Copy a Conv2d OIHW flat buffer into an OHWI
 * tensor. Used for conv_s0 / conv_s1 (forward convs).
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
 * permute_iohw_to_ohwi - Copy a ConvTranspose2d IOHW flat buffer into
 * an OHWI tensor. Used for up_conv1 / up_conv2. Source layout is
 * [IC, OC, KH, KW] — the in/out channels are swapped relative to
 * Conv2d.
 */
static void permute_iohw_to_ohwi(const float *src, float *dst,
				  int oc, int ic, int kh, int kw)
{
	for (int o = 0; o < oc; o++) {
		for (int y = 0; y < kh; y++) {
			for (int x = 0; x < kw; x++) {
				for (int c = 0; c < ic; c++) {
					int s = ((c * oc + o) * kh + y)
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
 * nchw_to_nhwc_copy - Permute [N, C, H, W] f32 -> [N, H, W, C] f32.
 *
 * The fixture tool populated feat_s0 / feat_s1 via fill_prng in NCHW
 * byte order. To reproduce the same numeric result on the NHWC mask
 * decoder, this test fills a temporary NCHW buffer with the identical
 * LCG state and then transposes into the actual NHWC tensor data.
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
 * fill_conv_transpose_weight - Fill a ConvTranspose2d OHWI weight
 * tensor with the same byte sequence that fill_prng would have
 * produced on the original IOHW raw buffer.
 */
static void fill_conv_transpose_weight(struct sam3_tensor *w,
					float amp, uint32_t *state)
{
	int oc = w->dims[0];
	int kh = w->dims[1];
	int kw = w->dims[2];
	int ic = w->dims[3];
	int n = oc * ic * kh * kw;
	float *raw = (float *)malloc((size_t)n * sizeof(float));
	if (!raw)
		return;
	fill_prng(raw, n, amp, state);
	permute_iohw_to_ohwi(raw, (float *)w->data, oc, ic, kh, kw);
	free(raw);
}

/*
 * fill_conv_weight - Fill a Conv2d OHWI weight tensor with the same
 * byte sequence that fill_prng would have produced on the original
 * OIHW raw buffer.
 */
static void fill_conv_weight(struct sam3_tensor *w,
			      float amp, uint32_t *state)
{
	int oc = w->dims[0];
	int kh = w->dims[1];
	int kw = w->dims[2];
	int ic = w->dims[3];
	int n = oc * ic * kh * kw;
	float *raw = (float *)malloc((size_t)n * sizeof(float));
	if (!raw)
		return;
	fill_prng(raw, n, amp, state);
	permute_oihw_to_ohwi(raw, (float *)w->data, oc, ic, kh, kw);
	free(raw);
}

/*
 * fill_weights - Reproduce fill_weight_tensors_mask_dec() from
 * tools/gen_nhwc_fixtures.c. The sequence of fill_prng calls must
 * match exactly: any drift in PRNG state invalidates the downstream
 * pixel decoder tensors and the fixture compare will fail.
 */
static void fill_weights(struct sam3_mask_decoder *dec, uint32_t *state)
{
	/* Learned tokens. */
	fill_prng((float *)dec->mask_tokens->data,
		  sam3_tensor_nelems(dec->mask_tokens), 0.1f, state);
	fill_prng((float *)dec->iou_token->data,
		  sam3_tensor_nelems(dec->iou_token), 0.1f, state);
	fill_prng((float *)dec->obj_score_token->data,
		  sam3_tensor_nelems(dec->obj_score_token), 0.1f, state);

	for (int l = 0; l < SAM3_MASK_DEC_LAYERS; l++) {
		struct sam3_tensor *t[] = {
			dec->layers[l].sa_qkv_w,  dec->layers[l].sa_qkv_b,
			dec->layers[l].sa_out_w,  dec->layers[l].sa_out_b,
			dec->layers[l].ca_ti_q_w, dec->layers[l].ca_ti_q_b,
			dec->layers[l].ca_ti_k_w, dec->layers[l].ca_ti_k_b,
			dec->layers[l].ca_ti_v_w, dec->layers[l].ca_ti_v_b,
			dec->layers[l].ca_ti_out_w,
			dec->layers[l].ca_ti_out_b,
			dec->layers[l].mlp_fc1_w, dec->layers[l].mlp_fc1_b,
			dec->layers[l].mlp_fc2_w, dec->layers[l].mlp_fc2_b,
			dec->layers[l].ca_it_q_w, dec->layers[l].ca_it_q_b,
			dec->layers[l].ca_it_k_w, dec->layers[l].ca_it_k_b,
			dec->layers[l].ca_it_v_w, dec->layers[l].ca_it_v_b,
			dec->layers[l].ca_it_out_w,
			dec->layers[l].ca_it_out_b,
		};
		int nt = (int)(sizeof(t) / sizeof(t[0]));
		for (int i = 0; i < nt; i++) {
			if (!t[i])
				continue;
			fill_prng((float *)t[i]->data,
				  sam3_tensor_nelems(t[i]), 0.05f, state);
		}

		struct sam3_tensor *lnw[] = {
			dec->layers[l].ln1_w, dec->layers[l].ln2_w,
			dec->layers[l].ln3_w, dec->layers[l].ln4_w,
		};
		struct sam3_tensor *lnb[] = {
			dec->layers[l].ln1_b, dec->layers[l].ln2_b,
			dec->layers[l].ln3_b, dec->layers[l].ln4_b,
		};
		for (int i = 0; i < 4; i++) {
			if (lnw[i]) {
				float *w = (float *)lnw[i]->data;
				int n = sam3_tensor_nelems(lnw[i]);
				fill_prng(w, n, 0.05f, state);
				for (int k = 0; k < n; k++)
					w[k] += 1.0f;
			}
			if (lnb[i])
				fill_prng((float *)lnb[i]->data,
					  sam3_tensor_nelems(lnb[i]),
					  0.02f, state);
		}
	}

	struct sam3_tensor *final_t[] = {
		dec->final_q_w, dec->final_q_b,
		dec->final_k_w, dec->final_k_b,
		dec->final_v_w, dec->final_v_b,
		dec->final_out_w, dec->final_out_b,
	};
	for (int i = 0; i < (int)(sizeof(final_t) / sizeof(final_t[0])); i++) {
		if (!final_t[i])
			continue;
		fill_prng((float *)final_t[i]->data,
			  sam3_tensor_nelems(final_t[i]), 0.05f, state);
	}
	if (dec->final_ln_w) {
		float *w = (float *)dec->final_ln_w->data;
		int n = sam3_tensor_nelems(dec->final_ln_w);
		fill_prng(w, n, 0.05f, state);
		for (int k = 0; k < n; k++)
			w[k] += 1.0f;
	}
	if (dec->final_ln_b)
		fill_prng((float *)dec->final_ln_b->data,
			  sam3_tensor_nelems(dec->final_ln_b),
			  0.02f, state);

	/* Pixel decoder conv weights: OHWI after Task 10. The PRNG
	 * sequence the fixture tool used walked the original IOHW /
	 * OIHW buffers; we reproduce that via an OIHW/IOHW scratch
	 * fill then permute into OHWI. Biases are 1-D, no permute. */
	fill_conv_transpose_weight(dec->up_conv1_w, 0.1f, state);
	if (dec->up_conv1_b)
		fill_prng((float *)dec->up_conv1_b->data,
			  sam3_tensor_nelems(dec->up_conv1_b),
			  0.1f, state);
	fill_conv_transpose_weight(dec->up_conv2_w, 0.1f, state);
	if (dec->up_conv2_b)
		fill_prng((float *)dec->up_conv2_b->data,
			  sam3_tensor_nelems(dec->up_conv2_b),
			  0.1f, state);
	fill_conv_weight(dec->conv_s0_w, 0.1f, state);
	if (dec->conv_s0_b)
		fill_prng((float *)dec->conv_s0_b->data,
			  sam3_tensor_nelems(dec->conv_s0_b),
			  0.1f, state);
	fill_conv_weight(dec->conv_s1_w, 0.1f, state);
	if (dec->conv_s1_b)
		fill_prng((float *)dec->conv_s1_b->data,
			  sam3_tensor_nelems(dec->conv_s1_b),
			  0.1f, state);

	if (dec->up_ln_w) {
		float *w = (float *)dec->up_ln_w->data;
		int n = sam3_tensor_nelems(dec->up_ln_w);
		fill_prng(w, n, 0.05f, state);
		for (int k = 0; k < n; k++)
			w[k] += 1.0f;
	}
	if (dec->up_ln_b)
		fill_prng((float *)dec->up_ln_b->data,
			  sam3_tensor_nelems(dec->up_ln_b),
			  0.02f, state);

	/* Hypernetwork and IoU MLPs. */
	for (int i = 0; i < SAM3_MASK_DEC_MASKS; i++) {
		struct sam3_tensor *h[] = {
			dec->hyper[i].proj_in_w, dec->hyper[i].proj_in_b,
			dec->hyper[i].hidden_w, dec->hyper[i].hidden_b,
			dec->hyper[i].proj_out_w, dec->hyper[i].proj_out_b,
		};
		for (int k = 0; k < (int)(sizeof(h) / sizeof(h[0])); k++) {
			if (!h[k])
				continue;
			fill_prng((float *)h[k]->data,
				  sam3_tensor_nelems(h[k]), 0.05f, state);
		}
	}

	struct sam3_tensor *iou_t[] = {
		dec->iou_proj_in_w, dec->iou_proj_in_b,
		dec->iou_hidden_w, dec->iou_hidden_b,
		dec->iou_proj_out_w, dec->iou_proj_out_b,
	};
	for (int i = 0; i < (int)(sizeof(iou_t) / sizeof(iou_t[0])); i++) {
		if (!iou_t[i])
			continue;
		fill_prng((float *)iou_t[i]->data,
			  sam3_tensor_nelems(iou_t[i]), 0.05f, state);
	}

	if (dec->no_mask_embed)
		fill_prng((float *)dec->no_mask_embed->data,
			  sam3_tensor_nelems(dec->no_mask_embed),
			  0.05f, state);
	if (dec->pe_gaussian)
		fill_prng((float *)dec->pe_gaussian->data,
			  sam3_tensor_nelems(dec->pe_gaussian),
			  0.5f, state);
}

/*
 * fill_feat_nhwc - Populate an NHWC feat tensor whose bytes match the
 * original NCHW fixture byte order after being transposed. The tool
 * used fill_prng directly on an NCHW buffer, so we fill a temporary
 * NCHW buffer with the same LCG state then permute into the NHWC
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
 * test_mask_decoder_nhwc_fixture - Run the migrated mask decoder and
 * compare the final mask logits against the pre-migration NCHW
 * reference. The mask shape is [n_masks, H, W], layout-invariant, so
 * a direct element-wise compare is sufficient.
 */
static void test_mask_decoder_nhwc_fixture(void)
{
	if (!fixtures_available()) {
		printf("SKIP: %s not found\n", FIXTURE_DIR);
		return;
	}

	uint32_t rng = MD_FIXTURE_SEED;
	struct sam3_mask_decoder dec;
	enum sam3_error err;

	err = sam3_mask_decoder_init(&dec);
	ASSERT_EQ(err, SAM3_OK);

	err = sam3_mask_decoder_load(&dec, NULL, &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	/* Step 1: weights — same LCG order as the fixture tool. */
	fill_weights(&dec, &rng);

	/* Step 2: inputs. keys is [N_PIX, d_model] and its byte layout
	 * already matches NHWC [1, H, W, d] (channels innermost), so a
	 * direct fill_prng reproduces the fixture. feat_s0 / feat_s1
	 * were NCHW in the tool, so we use the nchw_to_nhwc_copy
	 * transpose to preserve the byte-equivalent math. */
	int keys_dims[] = {MD_N_PIX, MD_D_MODEL};
	int s1_dims[] = {1, MD_FEAT_S1_H, MD_FEAT_S1_W, MD_D_MODEL};
	int s0_dims[] = {1, MD_FEAT_S0_H, MD_FEAT_S0_W, MD_D_MODEL};

	struct sam3_tensor *keys = gh_alloc_tensor(&g_cpu.arena,
		SAM3_DTYPE_F32, 2, keys_dims);
	struct sam3_tensor *feat_s1 = gh_alloc_tensor(&g_cpu.arena,
		SAM3_DTYPE_F32, 4, s1_dims);
	struct sam3_tensor *feat_s0 = gh_alloc_tensor(&g_cpu.arena,
		SAM3_DTYPE_F32, 4, s0_dims);
	ASSERT(keys != NULL);
	ASSERT(feat_s1 != NULL);
	ASSERT(feat_s0 != NULL);
	if (!keys || !feat_s1 || !feat_s0)
		return;

	fill_prng((float *)keys->data,
		  MD_N_PIX * MD_D_MODEL, 0.3f, &rng);
	fill_feat_nhwc(feat_s0, 1, MD_D_MODEL,
		       MD_FEAT_S0_H, MD_FEAT_S0_W, &rng);
	fill_feat_nhwc(feat_s1, 1, MD_D_MODEL,
		       MD_FEAT_S1_H, MD_FEAT_S1_W, &rng);

	/* Build the NHWC mask decoder graph and evaluate on CPU. */
	struct sam3_graph graph;
	sam3_graph_init(&graph);

	struct sam3_tensor *masks = NULL;
	struct sam3_tensor *iou = NULL;
	err = sam3_mask_decoder_build(&dec, &graph, keys,
				       MD_GRID_H, MD_GRID_W,
				       NULL, feat_s0, feat_s1,
				       &g_cpu.arena, &masks, &iou);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT(masks != NULL);
	if (!masks)
		return;

	err = g_cpu.base.ops->graph_eval(&g_cpu.base, &graph);
	ASSERT_EQ(err, SAM3_OK);

	/* masks shape: [n_masks, final_h, final_w] */
	ASSERT_EQ(masks->n_dims, 3);
	ASSERT_EQ(masks->dims[0], MD_N_MASKS);
	ASSERT_EQ(masks->dims[1], MD_FINAL_H);
	ASSERT_EQ(masks->dims[2], MD_FINAL_W);

	int nelems = MD_N_MASKS * MD_FINAL_H * MD_FINAL_W;
	float *ref = load_bin(FIXTURE_DIR "/mask_dec_output.bin",
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

	test_mask_decoder_nhwc_fixture();

	teardown();

	TEST_REPORT();
}
