/*
 * tests/test_neck_nhwc.c - NHWC FPN neck vs pre-migration fixtures
 *
 * Rebuilds the small synthetic neck used by tools/gen_nhwc_fixtures
 * (d_model=16, backbone_dim=32, grid_size=8, 4 scales) on the CPU
 * backend after the NHWC migration and compares each stage output
 * against the pre-migration NCHW reference blobs captured in
 * tests/fixtures/nhwc_migration/. Output tensors are in NHWC layout
 * [1, H, W, d_model]; they are transposed to NCHW in scratch before
 * the per-element comparison with ASSERT_NEAR(1e-4).
 *
 * Skips gracefully when the fixture directory is missing so the test
 * is safe to ship without the blobs checked into every clone.
 *
 * Key types:  sam3_neck, sam3_graph, sam3_cpu_backend, sam3_tensor
 * Depends on: test_helpers.h, model/necks.h, model/graph_helpers.h,
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
#include "model/necks.h"

#ifndef SAM3_SOURCE_DIR
#error "SAM3_SOURCE_DIR must be defined by CMake"
#endif

#define FIXTURE_DIR SAM3_SOURCE_DIR "/tests/fixtures/nhwc_migration"

/* Mirror the constants used by tools/gen_nhwc_fixtures.c. */
#define NECK_D_MODEL      16
#define NECK_BACKBONE_DIM 32
#define NECK_GRID_SIZE    8
#define NECK_N_SCALES     4
#define NECK_N_PATCHES    (NECK_GRID_SIZE * NECK_GRID_SIZE)

/* Must match the LCG in tools/gen_nhwc_fixtures.c exactly. */
#define GEN_NHWC_SEED 0x5A3ABCDEu

static const float neck_scales[NECK_N_SCALES] = {4.0f, 2.0f, 1.0f, 0.5f};

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
 * fixtures_available - Probe the first fixture blob before running
 * the comparison so the test degrades to a skip (not a fail) when
 * the NHWC fixture directory is missing from the working copy.
 */
static int fixtures_available(void)
{
	return access(FIXTURE_DIR "/neck_input.bin", F_OK) == 0;
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
 *
 * Returns a malloc'd float buffer (caller frees) and writes the
 * element count to *out_n. Returns NULL on I/O error or size
 * mismatch versus @expected_n.
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
 * nhwc_to_nchw_copy - Permute [1, H, W, C] f32 -> [1, C, H, W] f32.
 *
 * The neck output lives in NHWC after Task 8, but the reference
 * fixtures were captured pre-migration as NCHW, so this helper
 * brings the actual tensor into NCHW order for an elementwise
 * compare with the reference.
 */
static void nhwc_to_nchw_copy(const float *src, float *dst,
			      int N, int H, int W, int C)
{
	for (int n = 0; n < N; n++) {
		for (int c = 0; c < C; c++) {
			for (int h = 0; h < H; h++) {
				for (int w = 0; w < W; w++) {
					int s = ((n * H + h) * W + w) * C
						+ c;
					int d = ((n * C + c) * H + h) * W
						+ w;
					dst[d] = src[s];
				}
			}
		}
	}
}

/*
 * permute_to_ohwi - Copy a checkpoint-layout (OIHW for Conv2d, IOHW
 * for ConvTranspose2d) flat buffer into an OHWI tensor.
 *
 * Task 8 permutes every neck conv weight at load time, so the stored
 * conv_w tensors are already OHWI. The reference fixtures were
 * generated with NCHW weights in OIHW/IOHW flat order fed by the
 * same PRNG. This helper applies the same permute the loader does,
 * letting the test reproduce the reference math.
 */
static void permute_to_ohwi(const float *src, float *dst,
			    int oc, int ic, int kh, int kw,
			    int is_transpose)
{
	for (int o = 0; o < oc; o++) {
		for (int y = 0; y < kh; y++) {
			for (int x = 0; x < kw; x++) {
				for (int c = 0; c < ic; c++) {
					int s;
					if (is_transpose) {
						s = ((c * oc + o) * kh
						     + y) * kw + x;
					} else {
						s = ((o * ic + c) * kh
						     + y) * kw + x;
					}
					int d = ((o * kh + y) * kw + x)
						* ic + c;
					dst[d] = src[s];
				}
			}
		}
	}
}

/*
 * fill_weights - Reproduce fill_weight_tensors_neck() from
 * tools/gen_nhwc_fixtures.c. The fixture tool filled pre-migration
 * NCHW weights (OIHW / IOHW) with a single LCG; this test now has
 * OHWI-permuted storage. We fill a temporary flat buffer in the
 * original layout, then permute_to_ohwi into the actual tensor so
 * the LCG state evolves identically to the fixture run.
 */
static void fill_weights(struct sam3_neck *neck, uint32_t *state)
{
	for (int s = 0; s < neck->n_scales; s++) {
		for (int j = 0; j < neck->stages[s].n_convs; j++) {
			struct sam3_tensor *w = neck->stages[s].conv_w[j];
			struct sam3_tensor *b = neck->stages[s].conv_b[j];
			if (w) {
				/*
				 * w is OHWI [OC, KH, KW, IC]. Build a
				 * matching flat OIHW/IOHW buffer with
				 * the PRNG and permute it into w.
				 */
				int oc = w->dims[0];
				int kh = w->dims[1];
				int kw = w->dims[2];
				int ic = w->dims[3];
				int n = oc * ic * kh * kw;
				float *raw = (float *)malloc(
					(size_t)n * sizeof(float));
				if (raw) {
					fill_prng(raw, n, 0.1f, state);
					permute_to_ohwi(raw,
						(float *)w->data,
						oc, ic, kh, kw,
						neck->stages[s]
							.is_transpose[j]);
					free(raw);
				}
			}
			if (b)
				fill_prng((float *)b->data,
					  sam3_tensor_nelems(b),
					  0.05f, state);
		}
	}
}

/*
 * test_neck_nhwc_fixture - Run the migrated neck and compare each
 * stage output against the pre-migration NCHW reference blob.
 */
static void test_neck_nhwc_fixture(void)
{
	if (!fixtures_available()) {
		printf("SKIP: %s not found\n", FIXTURE_DIR);
		return;
	}

	uint32_t rng = GEN_NHWC_SEED;
	struct sam3_neck neck;
	enum sam3_error err;

	err = sam3_neck_init(&neck, NECK_D_MODEL, NECK_BACKBONE_DIM,
			     NECK_GRID_SIZE, NECK_N_SCALES, neck_scales);
	ASSERT_EQ(err, SAM3_OK);

	err = sam3_neck_load(&neck, NULL, &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	/*
	 * WARNING: weight fill must happen BEFORE fill_prng advances
	 * the LCG for the vit_out tensor. The order here mirrors
	 * tools/gen_nhwc_fixtures.c::dump_neck_fixture exactly.
	 */
	fill_weights(&neck, &rng);

	/* Build a dummy ViT output [n_patches, backbone_dim]. */
	int vit_dims[] = {NECK_N_PATCHES, NECK_BACKBONE_DIM};
	struct sam3_tensor *vit_out = gh_alloc_tensor(
		&g_cpu.arena, SAM3_DTYPE_F32, 2, vit_dims);
	ASSERT(vit_out != NULL);

	fill_prng((float *)vit_out->data,
		  sam3_tensor_nelems(vit_out), 0.5f, &rng);

	/*
	 * Sanity check: the fixture's neck_input.bin must match our
	 * in-memory copy byte-for-byte.
	 */
	int in_n = NECK_N_PATCHES * NECK_BACKBONE_DIM;
	float *ref_in = load_bin(FIXTURE_DIR "/neck_input.bin",
				 in_n, NULL);
	ASSERT(ref_in != NULL);
	if (ref_in) {
		const float *got = (const float *)vit_out->data;
		for (int i = 0; i < in_n; i++)
			ASSERT_NEAR(got[i], ref_in[i], 1e-6f);
		free(ref_in);
	}

	/* Build the NHWC neck graph and evaluate on CPU. */
	struct sam3_graph graph;
	sam3_graph_init(&graph);

	struct sam3_tensor *features[SAM3_NECK_MAX_SCALES];
	err = sam3_neck_build(&neck, &graph, vit_out, features,
			      &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	err = g_cpu.base.ops->graph_eval(&g_cpu.base, &graph);
	ASSERT_EQ(err, SAM3_OK);

	static const char *ref_names[NECK_N_SCALES] = {
		"/neck_features_s0.bin",
		"/neck_features_s1.bin",
		"/neck_features_s2.bin",
		"/neck_features_s3.bin",
	};

	/*
	 * For each stage, transpose [1,H,W,d] -> [1,d,H,W] in a
	 * malloc buffer and compare element-wise with the NCHW
	 * reference. Tolerance is 1e-4 per the plan.
	 */
	for (int s = 0; s < NECK_N_SCALES; s++) {
		struct sam3_tensor *feat = features[s];
		ASSERT(feat != NULL);
		if (!feat)
			continue;

		ASSERT_EQ(feat->n_dims, 4);
		int N = feat->dims[0];
		int H = feat->dims[1];
		int W = feat->dims[2];
		int C = feat->dims[3];
		int nelems = N * C * H * W;

		char path[512];
		snprintf(path, sizeof(path), "%s%s",
			 FIXTURE_DIR, ref_names[s]);
		float *ref = load_bin(path, nelems, NULL);
		ASSERT(ref != NULL);
		if (!ref)
			continue;

		float *got_nchw = (float *)malloc((size_t)nelems *
						  sizeof(float));
		ASSERT(got_nchw != NULL);
		if (!got_nchw) {
			free(ref);
			continue;
		}

		nhwc_to_nchw_copy((const float *)feat->data,
				  got_nchw, N, H, W, C);

		for (int i = 0; i < nelems; i++)
			ASSERT_NEAR(got_nchw[i], ref[i], 1e-4f);

		free(got_nchw);
		free(ref);
	}
}

int main(void)
{
	setup();

	test_neck_nhwc_fixture();

	teardown();

	TEST_REPORT();
}
