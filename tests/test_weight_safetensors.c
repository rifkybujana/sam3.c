/*
 * tests/test_weight_safetensors.c - Unit tests for SafeTensors reader
 *
 * Creates minimal SafeTensors files in-test and verifies the reader
 * correctly parses headers and returns tensor data.
 *
 * Key types:  weight_reader
 * Depends on: core/weight.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <string.h>
#include <stdint.h>
#include <math.h>
#include "test_helpers.h"
#include "core/weight.h"

#define TEST_ST_FILE  "/tmp/test_sam3.safetensors"
#define TEST_SAM3_FILE "/tmp/test_sam3_from_st.sam3"

/* --- Helper: write a raw SafeTensors file --- */

static int write_safetensors(const char *path, const char *json,
			     const void *data, size_t data_size)
{
	FILE *fp = fopen(path, "wb");
	if (!fp)
		return -1;

	uint64_t header_size = (uint64_t)strlen(json);

	if (fwrite(&header_size, sizeof(header_size), 1, fp) != 1)
		goto fail;
	if (fwrite(json, 1, (size_t)header_size, fp) != (size_t)header_size)
		goto fail;
	if (data_size > 0 &&
	    fwrite(data, 1, data_size, fp) != data_size)
		goto fail;

	fclose(fp);
	return 0;

fail:
	fclose(fp);
	return -1;
}

/* --- Test 1: single tensor read ───────── --- */

static void test_safetensors_read(void)
{
	const char *json =
		"{\"test.weight\":{\"dtype\":\"F32\",\"shape\":[2,3],"
		"\"data_offsets\":[0,24]}}";
	float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

	int rc = write_safetensors(TEST_ST_FILE, json, data, sizeof(data));
	ASSERT_EQ(rc, 0);

	struct weight_reader reader;
	weight_reader_safetensors_init(&reader);

	enum sam3_error err = reader.ops->open(&reader, TEST_ST_FILE);
	ASSERT_EQ(err, SAM3_OK);

	/* Verify tensor count */
	ASSERT_EQ(reader.ops->n_tensors(&reader), 1);

	/* Verify tensor info */
	struct weight_tensor_info info;
	err = reader.ops->get_tensor_info(&reader, 0, &info);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT(strcmp(info.name, "test.weight") == 0);
	ASSERT_EQ((int)info.dtype, (int)SAM3_DTYPE_F32);
	ASSERT_EQ(info.n_dims, 2);
	ASSERT_EQ(info.dims[0], 2);
	ASSERT_EQ(info.dims[1], 3);
	ASSERT_EQ((int)info.nbytes, 24);

	/* Verify tensor data */
	float buf[6];
	err = reader.ops->read_tensor_data(&reader, 0, buf, sizeof(buf));
	ASSERT_EQ(err, SAM3_OK);
	for (int i = 0; i < 6; i++)
		ASSERT(fabsf(buf[i] - data[i]) < 1e-6f);

	reader.ops->close(&reader);
}

/* --- Test 2: multi-tensor ─────────────── --- */

static void test_safetensors_multi_tensor(void)
{
	const char *json =
		"{\"a\":{\"dtype\":\"I32\",\"shape\":[2],"
		"\"data_offsets\":[0,8]},"
		"\"b\":{\"dtype\":\"F32\",\"shape\":[3],"
		"\"data_offsets\":[8,20]}}";

	/* Build combined data: int32[2] then float[3] */
	uint8_t data[20];
	int32_t ints[] = {10, 20};
	float floats[] = {1.0f, 2.0f, 3.0f};
	memcpy(data, ints, 8);
	memcpy(data + 8, floats, 12);

	int rc = write_safetensors(TEST_ST_FILE, json, data, sizeof(data));
	ASSERT_EQ(rc, 0);

	struct weight_reader reader;
	weight_reader_safetensors_init(&reader);

	enum sam3_error err = reader.ops->open(&reader, TEST_ST_FILE);
	ASSERT_EQ(err, SAM3_OK);

	ASSERT_EQ(reader.ops->n_tensors(&reader), 2);

	/* Verify tensor "a" (I32, shape [2]) */
	struct weight_tensor_info info;
	err = reader.ops->get_tensor_info(&reader, 0, &info);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT(strcmp(info.name, "a") == 0);
	ASSERT_EQ((int)info.dtype, (int)SAM3_DTYPE_I32);
	ASSERT_EQ(info.n_dims, 1);
	ASSERT_EQ(info.dims[0], 2);
	ASSERT_EQ((int)info.nbytes, 8);

	int32_t ibuf[2];
	err = reader.ops->read_tensor_data(&reader, 0, ibuf, sizeof(ibuf));
	ASSERT_EQ(err, SAM3_OK);
	ASSERT_EQ(ibuf[0], 10);
	ASSERT_EQ(ibuf[1], 20);

	/* Verify tensor "b" (F32, shape [3]) */
	err = reader.ops->get_tensor_info(&reader, 1, &info);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT(strcmp(info.name, "b") == 0);
	ASSERT_EQ((int)info.dtype, (int)SAM3_DTYPE_F32);
	ASSERT_EQ(info.n_dims, 1);
	ASSERT_EQ(info.dims[0], 3);
	ASSERT_EQ((int)info.nbytes, 12);

	float fbuf[3];
	err = reader.ops->read_tensor_data(&reader, 1, fbuf, sizeof(fbuf));
	ASSERT_EQ(err, SAM3_OK);
	for (int i = 0; i < 3; i++)
		ASSERT(fabsf(fbuf[i] - floats[i]) < 1e-6f);

	reader.ops->close(&reader);
}

/* --- Test 3: bad file ─ --- */

static void test_safetensors_bad_file(void)
{
	struct weight_reader reader;
	weight_reader_safetensors_init(&reader);

	enum sam3_error err = reader.ops->open(
		&reader, "/tmp/nonexistent_safetensors_file.safetensors");
	ASSERT_EQ(err, SAM3_EIO);
}

/* --- Test 4: end-to-end SafeTensors -> .sam3 -> load --- */

static void test_safetensors_end_to_end(void)
{
	/* Write a SafeTensors file */
	const char *json =
		"{\"my.tensor\":{\"dtype\":\"F32\",\"shape\":[2,3],"
		"\"data_offsets\":[0,24]}}";
	float orig_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

	int rc = write_safetensors(TEST_ST_FILE, json,
				   orig_data, sizeof(orig_data));
	ASSERT_EQ(rc, 0);

	/* Open via SafeTensors reader */
	struct weight_reader reader;
	weight_reader_safetensors_init(&reader);

	enum sam3_error err = reader.ops->open(&reader, TEST_ST_FILE);
	ASSERT_EQ(err, SAM3_OK);

	/* Convert to .sam3 via writer */
	struct sam3_model_config config = {
		.image_size       = 512,
		.encoder_dim      = 384,
		.decoder_dim      = 128,
		.n_encoder_layers = 6,
		.n_decoder_layers = 1,
	};

	err = sam3_weight_write(TEST_SAM3_FILE, &config, &reader);
	ASSERT_EQ(err, SAM3_OK);

	reader.ops->close(&reader);

	/* Open the .sam3 file with the native loader */
	struct sam3_weight_file wf;
	memset(&wf, 0, sizeof(wf));
	err = sam3_weight_open(&wf, TEST_SAM3_FILE);
	ASSERT_EQ(err, SAM3_OK);

	/* Find the tensor */
	const struct sam3_weight_tensor_desc *desc;
	desc = sam3_weight_find(&wf, "my.tensor");
	ASSERT(desc != NULL);
	ASSERT_EQ((int)desc->dtype, (int)SAM3_DTYPE_F32);
	ASSERT_EQ((int)desc->n_dims, 2);
	ASSERT_EQ(desc->dims[0], 2);
	ASSERT_EQ(desc->dims[1], 3);
	ASSERT_EQ((int)desc->data_size, (int)sizeof(orig_data));

	/* Verify data matches original */
	const float *loaded = sam3_weight_tensor_data(&wf, desc);
	ASSERT(loaded != NULL);
	for (int i = 0; i < 6; i++)
		ASSERT(fabsf(loaded[i] - orig_data[i]) < 1e-6f);

	sam3_weight_close(&wf);
}

/* --- Main ────────── --- */

int main(void)
{
	test_safetensors_read();
	test_safetensors_multi_tensor();
	test_safetensors_bad_file();
	test_safetensors_end_to_end();

	/* Clean up temp files */
	remove(TEST_ST_FILE);
	remove(TEST_SAM3_FILE);

	TEST_REPORT();
}
