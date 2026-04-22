/*
 * tests/test_feature_cache_persist.c - .sam3cache persistence tests
 *
 * Round-trip tests for sam3_image_bundle_save/load,
 * sam3_text_bundle_save/load, and the in-process uncompressed spill
 * helpers (sam3_image_bundle_write/read_uncompressed). Asserts
 * value-level equality of restored tensors, NULL-tensor handling,
 * and header field propagation. The on-disk format is intentionally
 * uncompressed; compression was removed in version 3 after benchmarks
 * showed zlib ran at ~12-50 MB/s on the bundle and dominated every
 * demote/save call.
 *
 * Key types:  sam3_cache_persist_sig, sam3_image_bundle, sam3_text_bundle
 * Depends on: test_helpers.h, model/feature_cache_persist.h,
 *             model/feature_cache.h, model/graph_helpers.h, core/alloc.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "core/tensor.h"
#include "core/alloc.h"
#include "model/graph_helpers.h"
#include "model/feature_cache.h"
#include "model/feature_cache_persist.h"

static struct sam3_cache_persist_sig make_sig(void)
{
	struct sam3_cache_persist_sig s = {0};
	s.image_size    = 1024;
	s.encoder_dim   = 256;
	s.decoder_dim   = 256;
	s.backbone_type = 1;
	s.variant       = 2;
	s.n_fpn_scales  = 4;
	s.text_backbone = 3;
	return s;
}

static struct sam3_tensor *
make_f32_fill(struct sam3_arena *a, int n, float fill)
{
	int dims[1] = { n };
	struct sam3_tensor *t = gh_alloc_tensor(a, SAM3_DTYPE_F32, 1, dims);
	if (!t)
		return NULL;
	float *p = (float *)t->data;
	for (int i = 0; i < n; i++)
		p[i] = fill;
	return t;
}

static struct sam3_tensor *
make_f32_seq(struct sam3_arena *a, int n)
{
	int dims[1] = { n };
	struct sam3_tensor *t = gh_alloc_tensor(a, SAM3_DTYPE_F32, 1, dims);
	if (!t)
		return NULL;
	float *p = (float *)t->data;
	for (int i = 0; i < n; i++)
		p[i] = (float)i * 0.0123f - 1.5f;
	return t;
}

/* --- test_persist_image_roundtrip --- */

static void test_persist_image_roundtrip(void)
{
	struct sam3_arena wa;
	ASSERT_EQ(sam3_arena_init(&wa, 4 * 1024 * 1024), SAM3_OK);

	struct sam3_image_bundle b = {0};
	b.image_features = make_f32_seq(&wa, 512);
	b.feat_s0_nhwc   = make_f32_seq(&wa, 256);
	/* feat_s1_nhwc and onward intentionally NULL -> must round-trip NULL. */
	b.prompt_w = 640;
	b.prompt_h = 480;
	b.width    = 1280;
	b.height   = 720;

	struct sam3_cache_persist_sig sig = make_sig();
	uint8_t prefix[6] = { 0xDE, 0xAD, 0xBE, 0xEF, 0x42, 0x01 };
	const char *path = "/tmp/sam3_persist_image_roundtrip.sam3cache";

	ASSERT_EQ(sam3_image_bundle_save(path, &sig, 0xFEEDFACECAFEBEEFULL,
					 prefix, sizeof(prefix), &b),
		  SAM3_OK);

	struct sam3_arena ra;
	ASSERT_EQ(sam3_arena_init(&ra, 4 * 1024 * 1024), SAM3_OK);

	struct sam3_image_bundle out = {0};
	uint64_t out_hash = 0;
	uint8_t  out_prefix[SAM3_CACHE_PREFIX_BYTES];
	size_t   out_plen = 0;

	ASSERT_EQ(sam3_image_bundle_load(path, &sig, &ra, &out_hash,
					 out_prefix, &out_plen, &out),
		  SAM3_OK);

	ASSERT_EQ(out_hash, 0xFEEDFACECAFEBEEFULL);
	ASSERT_EQ(out_plen, (size_t)sizeof(prefix));
	ASSERT_EQ(memcmp(out_prefix, prefix, sizeof(prefix)), 0);
	ASSERT_EQ(out.prompt_w, 640);
	ASSERT_EQ(out.prompt_h, 480);
	ASSERT_EQ(out.width, 1280);
	ASSERT_EQ(out.height, 720);

	ASSERT_TENSOR_CLOSE(out.image_features, b.image_features, 0.0f, 0.0f);
	ASSERT_TENSOR_CLOSE(out.feat_s0_nhwc,   b.feat_s0_nhwc,   0.0f, 0.0f);

	ASSERT(out.feat_s1_nhwc  == NULL);
	ASSERT(out.feat_4x_nhwc  == NULL);
	ASSERT(out.sam2_05x_nhwc == NULL);
	ASSERT(out.sam2_1x_nhwc  == NULL);
	ASSERT(out.sam2_2x_nhwc  == NULL);
	ASSERT(out.sam2_4x_nhwc  == NULL);

	remove(path);
	sam3_arena_free(&ra);
	sam3_arena_free(&wa);
}

/* --- test_persist_text_roundtrip --- */

static void test_persist_text_roundtrip(void)
{
	struct sam3_arena wa;
	ASSERT_EQ(sam3_arena_init(&wa, 4 * 1024 * 1024), SAM3_OK);

	struct sam3_text_bundle b = {0};
	b.n_tokens = 11;
	b.features = make_f32_seq(&wa, 11 * 128);

	struct sam3_cache_persist_sig sig = make_sig();
	int32_t toks[4] = { 101, 202, 303, 404 };
	const char *path = "/tmp/sam3_persist_text_roundtrip.sam3cache";

	ASSERT_EQ(sam3_text_bundle_save(path, &sig, 0x1122334455667788ULL,
					toks, 4, &b),
		  SAM3_OK);

	struct sam3_arena ra;
	ASSERT_EQ(sam3_arena_init(&ra, 4 * 1024 * 1024), SAM3_OK);

	struct sam3_text_bundle out = {0};
	uint64_t out_hash = 0;
	int32_t  out_toks[SAM3_CACHE_PREFIX_BYTES / 4];
	int      out_plen = 0;

	ASSERT_EQ(sam3_text_bundle_load(path, &sig, &ra, &out_hash,
					out_toks, &out_plen, &out),
		  SAM3_OK);

	ASSERT_EQ(out_hash, 0x1122334455667788ULL);
	ASSERT_EQ(out_plen, 4);
	ASSERT_EQ(out_toks[0], 101);
	ASSERT_EQ(out_toks[1], 202);
	ASSERT_EQ(out_toks[2], 303);
	ASSERT_EQ(out_toks[3], 404);
	ASSERT_EQ(out.n_tokens, 11);
	ASSERT_TENSOR_CLOSE(out.features, b.features, 0.0f, 0.0f);

	remove(path);
	sam3_arena_free(&ra);
	sam3_arena_free(&wa);
}

/* --- test_bundle_uncompressed_roundtrip --- */

static void test_bundle_uncompressed_roundtrip(void)
{
	struct sam3_arena wa;
	ASSERT_EQ(sam3_arena_init(&wa, 8 * 1024 * 1024), SAM3_OK);

	struct sam3_image_bundle b = {0};
	b.image_features = make_f32_seq(&wa, 256);
	b.feat_s0_nhwc   = make_f32_seq(&wa, 512);
	b.sam2_2x_nhwc   = make_f32_fill(&wa, 128, 3.14f);
	b.prompt_w = 1920;
	b.prompt_h = 1080;
	b.width    = 1920;
	b.height   = 1080;

	const char *path = "/tmp/sam3_bundle_raw_rt.bin";
	ASSERT_EQ(sam3_image_bundle_write_uncompressed(path, &b), SAM3_OK);

	struct sam3_arena ra;
	ASSERT_EQ(sam3_arena_init(&ra, 8 * 1024 * 1024), SAM3_OK);
	struct sam3_image_bundle out = {0};
	ASSERT_EQ(sam3_image_bundle_read_uncompressed(path, &ra, &out),
		  SAM3_OK);

	ASSERT_EQ(out.prompt_w, 1920);
	ASSERT_EQ(out.width,    1920);
	ASSERT_TENSOR_CLOSE(out.image_features, b.image_features, 0.0f, 0.0f);
	ASSERT_TENSOR_CLOSE(out.feat_s0_nhwc,   b.feat_s0_nhwc,   0.0f, 0.0f);
	ASSERT_TENSOR_CLOSE(out.sam2_2x_nhwc,   b.sam2_2x_nhwc,   0.0f, 0.0f);
	ASSERT(out.feat_s1_nhwc == NULL);

	remove(path);
	sam3_arena_free(&ra);
	sam3_arena_free(&wa);
}

int main(void)
{
	test_persist_image_roundtrip();
	test_persist_text_roundtrip();
	test_bundle_uncompressed_roundtrip();
	TEST_REPORT();
}
