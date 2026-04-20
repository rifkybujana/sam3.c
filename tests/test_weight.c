/*
 * tests/test_weight.c - Unit tests for .sam3 weight format
 *
 * Tests the native weight file writer and mmap loader. Uses an
 * in-memory reader to create test .sam3 files, then verifies
 * round-trip correctness, hash table lookup, and error handling.
 *
 * Key types:  sam3_weight_file
 * Depends on: core/weight.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <string.h>
#include <math.h>
#include "test_helpers.h"
#include "core/weight.h"

#define TEST_FILE "/tmp/test_sam3_weights.sam3"

/* --- Minimal in-memory reader for testing --- */

struct mem_tensor {
	const char     *name;
	enum sam3_dtype dtype;
	int             n_dims;
	int             dims[SAM3_MAX_DIMS];
	const void     *data;
	size_t          nbytes;
};

struct mem_reader_state {
	struct mem_tensor *tensors;
	int                n_tensors;
};

static enum sam3_error mem_open(struct weight_reader *r, const char *path)
{
	(void)path;
	(void)r;
	return SAM3_OK;
}

static int mem_n_tensors(struct weight_reader *r)
{
	struct mem_reader_state *s = r->impl;
	return s->n_tensors;
}

static enum sam3_error mem_get_tensor_info(struct weight_reader *r, int idx,
					   struct weight_tensor_info *info)
{
	struct mem_reader_state *s = r->impl;
	if (idx < 0 || idx >= s->n_tensors)
		return SAM3_EINVAL;
	struct mem_tensor *t = &s->tensors[idx];
	info->name   = t->name;
	info->dtype  = t->dtype;
	info->n_dims = t->n_dims;
	memcpy(info->dims, t->dims, sizeof(info->dims));
	info->nbytes = t->nbytes;
	return SAM3_OK;
}

static enum sam3_error mem_read_tensor_data(struct weight_reader *r, int idx,
					    void *dst, size_t dst_size)
{
	struct mem_reader_state *s = r->impl;
	if (idx < 0 || idx >= s->n_tensors)
		return SAM3_EINVAL;
	struct mem_tensor *t = &s->tensors[idx];
	if (dst_size < t->nbytes)
		return SAM3_EINVAL;
	memcpy(dst, t->data, t->nbytes);
	return SAM3_OK;
}

static void mem_close(struct weight_reader *r)
{
	(void)r;
}

static const struct weight_reader_ops mem_reader_ops = {
	.open             = mem_open,
	.n_tensors        = mem_n_tensors,
	.get_tensor_info  = mem_get_tensor_info,
	.read_tensor_data = mem_read_tensor_data,
	.close            = mem_close,
};

/* --- Helper: create a test reader with known data --- */

static float test_data_a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
static float test_data_b[] = {7.0f, 8.0f, 9.0f, 10.0f};

static struct mem_tensor test_tensors[] = {
	{
		.name   = "encoder.weight",
		.dtype  = SAM3_DTYPE_F32,
		.n_dims = 2,
		.dims   = {2, 3, 0, 0},
		.data   = test_data_a,
		.nbytes = sizeof(test_data_a),
	},
	{
		.name   = "decoder.bias",
		.dtype  = SAM3_DTYPE_F32,
		.n_dims = 1,
		.dims   = {4, 0, 0, 0},
		.data   = test_data_b,
		.nbytes = sizeof(test_data_b),
	},
};

static struct sam3_model_config test_config = {
	.image_size       = 1024,
	.encoder_dim      = 768,
	.decoder_dim      = 256,
	.n_encoder_layers = 12,
	.n_decoder_layers = 2,
};

static enum sam3_error write_test_file(void)
{
	struct mem_reader_state state = {
		.tensors   = test_tensors,
		.n_tensors = 2,
	};
	struct weight_reader reader = {
		.ops  = &mem_reader_ops,
		.impl = &state,
	};
	return sam3_weight_write(TEST_FILE, &test_config, &reader);
}

/* --- Tests ───────── --- */

static void test_weight_roundtrip(void)
{
	enum sam3_error err;
	struct sam3_weight_file wf;

	err = write_test_file();
	ASSERT_EQ(err, SAM3_OK);

	memset(&wf, 0, sizeof(wf));
	err = sam3_weight_open(&wf, TEST_FILE);
	ASSERT_EQ(err, SAM3_OK);

	/* Verify header */
	ASSERT_EQ((int)wf.header->n_tensors, 2);
	ASSERT_EQ(wf.header->image_size, 1024);
	ASSERT_EQ(wf.header->encoder_dim, 768);
	ASSERT_EQ(wf.header->decoder_dim, 256);
	ASSERT_EQ(wf.header->n_encoder_layers, 12);
	ASSERT_EQ(wf.header->n_decoder_layers, 2);

	/* Find tensors by name */
	const struct sam3_weight_tensor_desc *da;
	da = sam3_weight_find(&wf, "encoder.weight");
	ASSERT(da != NULL);
	ASSERT_EQ((int)da->dtype, (int)SAM3_DTYPE_F32);
	ASSERT_EQ((int)da->n_dims, 2);
	ASSERT_EQ(da->dims[0], 2);
	ASSERT_EQ(da->dims[1], 3);

	const struct sam3_weight_tensor_desc *db;
	db = sam3_weight_find(&wf, "decoder.bias");
	ASSERT(db != NULL);
	ASSERT_EQ((int)db->dtype, (int)SAM3_DTYPE_F32);
	ASSERT_EQ((int)db->n_dims, 1);
	ASSERT_EQ(db->dims[0], 4);

	/* Verify data contents */
	const float *pa = sam3_weight_tensor_data(&wf, da);
	ASSERT(pa != NULL);
	for (int i = 0; i < 6; i++)
		ASSERT(fabsf(pa[i] - test_data_a[i]) < 1e-6f);

	const float *pb = sam3_weight_tensor_data(&wf, db);
	ASSERT(pb != NULL);
	for (int i = 0; i < 4; i++)
		ASSERT(fabsf(pb[i] - test_data_b[i]) < 1e-6f);

	/* Missing tensor returns NULL */
	ASSERT(sam3_weight_find(&wf, "nonexistent") == NULL);

	sam3_weight_close(&wf);
}

static void test_weight_to_tensor(void)
{
	enum sam3_error err;
	struct sam3_weight_file wf;
	struct sam3_tensor t;

	err = write_test_file();
	ASSERT_EQ(err, SAM3_OK);

	memset(&wf, 0, sizeof(wf));
	err = sam3_weight_open(&wf, TEST_FILE);
	ASSERT_EQ(err, SAM3_OK);

	const struct sam3_weight_tensor_desc *d;
	d = sam3_weight_find(&wf, "encoder.weight");
	ASSERT(d != NULL);

	err = sam3_weight_to_tensor(&wf, d, &t);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT_EQ((int)t.dtype, (int)SAM3_DTYPE_F32);
	ASSERT_EQ(t.n_dims, 2);
	ASSERT_EQ(t.dims[0], 2);
	ASSERT_EQ(t.dims[1], 3);
	ASSERT_EQ(t.strides[0], 3);
	ASSERT_EQ(t.strides[1], 1);
	ASSERT(t.data != NULL);
	ASSERT_EQ((int)t.nbytes, (int)sizeof(test_data_a));

	sam3_weight_close(&wf);
}

static void test_weight_bad_magic(void)
{
	enum sam3_error err;
	struct sam3_weight_file wf;

	/* Write a valid file first */
	err = write_test_file();
	ASSERT_EQ(err, SAM3_OK);

	/* Corrupt the magic bytes */
	FILE *f = fopen(TEST_FILE, "r+b");
	ASSERT(f != NULL);
	uint32_t bad_magic = 0xDEADBEEF;
	fwrite(&bad_magic, 4, 1, f);
	fclose(f);

	memset(&wf, 0, sizeof(wf));
	err = sam3_weight_open(&wf, TEST_FILE);
	ASSERT_EQ(err, SAM3_EMODEL);
}

static void test_weight_null_args(void)
{
	enum sam3_error err;
	struct sam3_weight_file wf;

	memset(&wf, 0, sizeof(wf));
	err = sam3_weight_open(&wf, NULL);
	ASSERT_EQ(err, SAM3_EINVAL);

	err = sam3_weight_open(NULL, "some_path");
	ASSERT_EQ(err, SAM3_EINVAL);
}

static void test_weight_file_not_found(void)
{
	enum sam3_error err;
	struct sam3_weight_file wf;

	memset(&wf, 0, sizeof(wf));
	err = sam3_weight_open(&wf, "/tmp/nonexistent_sam3_file.sam3");
	ASSERT_EQ(err, SAM3_EIO);
}

int main(void)
{
	test_weight_roundtrip();
	test_weight_to_tensor();
	test_weight_bad_magic();
	test_weight_null_args();
	test_weight_file_not_found();

	/* Clean up test file */
	remove(TEST_FILE);

	TEST_REPORT();
}
