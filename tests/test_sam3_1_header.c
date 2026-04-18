/*
 * tests/test_sam3_1_header.c - Round-trip the SAM 3.1 header slots.
 *
 * Writes a minimal valid .sam3 file with a single 1-element tensor,
 * setting variant=SAM3_VARIANT_SAM3_1 and n_fpn_scales=3. Re-opens it
 * and asserts the fields round-trip. Also asserts that the legacy path
 * (reserved[1..2] zero) yields round-trip zeros — the loader-side
 * fallback is exercised by the integration suite.
 *
 * Key types:  (uses sam3_weight_header)
 * Depends on: sam3/sam3_types.h, core/weight.h, test_helpers.h
 * Used by:    CTest registration in CMakeLists.txt
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sam3/sam3.h"
#include "sam3/sam3_types.h"
#include "core/weight.h"
#include "test_helpers.h"

struct tiny_reader_state {
	float datum;
};

static enum sam3_error tr_open(struct weight_reader *r, const char *p)
{
	(void)r; (void)p;
	return SAM3_OK;
}
static int tr_n_tensors(struct weight_reader *r) { (void)r; return 1; }
static enum sam3_error tr_get_tensor_info(struct weight_reader *r, int idx,
					  struct weight_tensor_info *info)
{
	(void)r; (void)idx;
	info->name   = "dummy";
	info->dtype  = SAM3_DTYPE_F32;
	info->n_dims = 1;
	info->dims[0] = 1;
	info->nbytes  = sizeof(float);
	return SAM3_OK;
}
static enum sam3_error tr_read_tensor_data(struct weight_reader *r, int idx,
					   void *dst, size_t dst_size)
{
	struct tiny_reader_state *s = r->impl;
	(void)idx;
	if (dst_size < sizeof(float)) return SAM3_EINVAL;
	memcpy(dst, &s->datum, sizeof(float));
	return SAM3_OK;
}
static void tr_close(struct weight_reader *r) { (void)r; }

static const struct weight_reader_ops tr_ops = {
	.open             = tr_open,
	.n_tensors        = tr_n_tensors,
	.get_tensor_info  = tr_get_tensor_info,
	.read_tensor_data = tr_read_tensor_data,
	.close            = tr_close,
};

static void test_sam3_1_roundtrip(void)
{
	const char *path = "/tmp/sam3_test_variant.sam3";
	struct sam3_model_config cfg = {
		.image_size       = 1008,
		.encoder_dim      = 1024,
		.decoder_dim      = 256,
		.n_encoder_layers = 32,
		.n_decoder_layers = 2,
		.backbone_type    = SAM3_BACKBONE_HIERA,
		.n_fpn_scales     = 3,
		.variant          = SAM3_VARIANT_SAM3_1,
	};
	struct tiny_reader_state s = { .datum = 1.5f };
	struct weight_reader r = { .ops = &tr_ops, .impl = &s };

	ASSERT(sam3_weight_write(path, &cfg, &r) == SAM3_OK);

	struct sam3_weight_file wf;
	memset(&wf, 0, sizeof(wf));
	ASSERT(sam3_weight_open(&wf, path) == SAM3_OK);
	ASSERT_EQ(wf.header->reserved[1], (uint32_t)SAM3_VARIANT_SAM3_1);
	ASSERT_EQ(wf.header->reserved[2], 3);
	sam3_weight_close(&wf);

	unlink(path);
}

static void test_sam3_legacy_defaults(void)
{
	const char *path = "/tmp/sam3_test_legacy.sam3";
	struct sam3_model_config cfg = {
		.image_size       = 1008,
		.encoder_dim      = 1024,
		.decoder_dim      = 256,
		.n_encoder_layers = 32,
		.n_decoder_layers = 2,
		.backbone_type    = SAM3_BACKBONE_HIERA,
		/* variant and n_fpn_scales both zero */
	};
	struct tiny_reader_state s = { .datum = 0.0f };
	struct weight_reader r = { .ops = &tr_ops, .impl = &s };

	ASSERT(sam3_weight_write(path, &cfg, &r) == SAM3_OK);

	struct sam3_weight_file wf;
	memset(&wf, 0, sizeof(wf));
	ASSERT(sam3_weight_open(&wf, path) == SAM3_OK);
	ASSERT_EQ(wf.header->reserved[1], 0);
	ASSERT_EQ(wf.header->reserved[2], 0);
	sam3_weight_close(&wf);

	unlink(path);
}

int main(void)
{
	test_sam3_1_roundtrip();
	test_sam3_legacy_defaults();
	printf("test_sam3_1_header: PASS\n");
	return 0;
}
