/*
 * tests/test_weight_q8.c - Q8_0 weight format test
 *
 * Tests that Q8_0 tensors can be written to and read from .sam3 files.
 * Writes a small Q8_0 weight file, reads it back, and verifies the
 * tensor data matches.
 *
 * Key types:  sam3_weight_file, sam3_q8_block
 * Depends on: core/weight.h, core/quant.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "core/weight.h"
#include "core/quant.h"

#include <string.h>
#include <unistd.h>

/* Minimal weight_reader that serves one Q8_0 tensor */

struct q8_test_state {
	struct sam3_q8_block blocks[4]; /* 128 elements = 4 blocks */
	int nelems;
};

static enum sam3_error q8r_open(struct weight_reader *r, const char *path)
{
	(void)path;
	struct q8_test_state *s = r->impl;

	float src[128];
	for (int i = 0; i < 128; i++)
		src[i] = (float)(i - 64) * 0.1f;
	sam3_q8_quantize(src, s->blocks, 128);
	s->nelems = 128;

	return SAM3_OK;
}

static int q8r_n_tensors(struct weight_reader *r)
{
	(void)r;
	return 1;
}

static enum sam3_error q8r_get_info(struct weight_reader *r, int idx,
				    struct weight_tensor_info *info)
{
	struct q8_test_state *s = r->impl;
	(void)idx;

	info->name   = "test_weight";
	info->dtype  = SAM3_DTYPE_Q8_0;
	info->n_dims = 2;
	info->dims[0] = 4;
	info->dims[1] = 32;
	info->dims[2] = 0;
	info->dims[3] = 0;
	info->nbytes = sam3_q8_nbytes(s->nelems);

	return SAM3_OK;
}

static enum sam3_error q8r_read_data(struct weight_reader *r, int idx,
				     void *dst, size_t dst_size)
{
	struct q8_test_state *s = r->impl;
	(void)idx;

	size_t nb = sam3_q8_nbytes(s->nelems);
	if (dst_size < nb)
		return SAM3_EINVAL;

	memcpy(dst, s->blocks, nb);
	return SAM3_OK;
}

static void q8r_close(struct weight_reader *r)
{
	(void)r;
}

static const struct weight_reader_ops q8_reader_ops = {
	.open             = q8r_open,
	.n_tensors        = q8r_n_tensors,
	.get_tensor_info  = q8r_get_info,
	.read_tensor_data = q8r_read_data,
	.close            = q8r_close,
};

static void test_weight_q8_round_trip(void)
{
	const char *path = "/tmp/test_q8_weight.sam3";
	struct q8_test_state state;
	struct weight_reader reader = {
		.ops  = &q8_reader_ops,
		.impl = &state,
	};

	/* Write */
	reader.ops->open(&reader, NULL);
	struct sam3_model_config cfg = {
		.image_size       = 1024,
		.encoder_dim      = 1280,
		.decoder_dim      = 256,
		.n_encoder_layers = 32,
		.n_decoder_layers = 2,
	};
	enum sam3_error err = sam3_weight_write(path, &cfg, &reader);
	ASSERT_EQ(err, SAM3_OK);

	/* Read back */
	struct sam3_weight_file wf;
	memset(&wf, 0, sizeof(wf));
	err = sam3_weight_open(&wf, path);
	ASSERT_EQ(err, SAM3_OK);

	const struct sam3_weight_tensor_desc *desc =
		sam3_weight_find(&wf, "test_weight");
	ASSERT(desc != NULL);
	ASSERT_EQ((int)desc->dtype, (int)SAM3_DTYPE_Q8_0);

	struct sam3_tensor t;
	err = sam3_weight_to_tensor(&wf, desc, &t);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT_EQ((int)t.dtype, (int)SAM3_DTYPE_Q8_0);
	ASSERT_EQ((int)t.nbytes, (int)sam3_q8_nbytes(128));

	/* Q8_0 strides should be zeroed (per-element strides are meaningless) */
	for (int i = 0; i < SAM3_MAX_DIMS; i++)
		ASSERT_EQ(t.strides[i], 0);

	/* Verify data matches what we wrote */
	const struct sam3_q8_block *loaded =
		(const struct sam3_q8_block *)t.data;
	for (int b = 0; b < 4; b++) {
		ASSERT_NEAR(loaded[b].scale, state.blocks[b].scale, 1e-6f);
		for (int i = 0; i < SAM3_Q8_BLOCK_SIZE; i++)
			ASSERT_EQ(loaded[b].data[i], state.blocks[b].data[i]);
	}

	sam3_weight_close(&wf);
	unlink(path);
}

int main(void)
{
	test_weight_q8_round_trip();
	TEST_REPORT();
}
