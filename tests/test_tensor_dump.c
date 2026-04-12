/*
 * tests/test_tensor_dump.c - Unit tests for tensor dump utility.
 *
 * Verifies that sam3_tensor_dump writes correct header and data,
 * and rejects invalid inputs.
 *
 * Key types:  none
 * Depends on: sam3/internal/tensor_dump.h, core/tensor.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "test_helpers.h"
#include "sam3/internal/tensor_dump.h"
#include "core/tensor.h"

static void test_dump_2d_tensor(void)
{
	float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	struct sam3_tensor t = {0};
	t.dtype = SAM3_DTYPE_F32;
	t.n_dims = 2;
	t.dims[0] = 2;
	t.dims[1] = 3;
	t.data = data;
	t.nbytes = sizeof(data);

	const char *path = "/tmp/test_tensor_dump_2d.bin";
	ASSERT_EQ(sam3_tensor_dump(path, &t), 0);

	FILE *f = fopen(path, "rb");
	ASSERT(f != NULL);

	int32_t hdr[3];
	ASSERT_EQ(fread(hdr, sizeof(int32_t), 3, f), (size_t)3);
	ASSERT_EQ(hdr[0], 2);  /* n_dims */
	ASSERT_EQ(hdr[1], 2);  /* dim 0 */
	ASSERT_EQ(hdr[2], 3);  /* dim 1 */

	float read_data[6];
	ASSERT_EQ(fread(read_data, sizeof(float), 6, f), (size_t)6);
	for (int i = 0; i < 6; i++)
		ASSERT_NEAR(read_data[i], data[i], 1e-9);

	fclose(f);
	remove(path);
}

static void test_dump_rejects_null(void)
{
	struct sam3_tensor t = {0};
	t.dtype = SAM3_DTYPE_F32;
	t.n_dims = 1;
	t.dims[0] = 1;
	float v = 1.0f;
	t.data = &v;

	ASSERT_EQ(sam3_tensor_dump(NULL, &t), -1);
	ASSERT_EQ(sam3_tensor_dump("/tmp/x.bin", NULL), -1);
}

static void test_dump_rejects_non_f32(void)
{
	float data[] = {1.0f};
	struct sam3_tensor t = {0};
	t.dtype = SAM3_DTYPE_F16;
	t.n_dims = 1;
	t.dims[0] = 1;
	t.data = data;

	ASSERT_EQ(sam3_tensor_dump("/tmp/x.bin", &t), -1);
}

int main(void)
{
	test_dump_2d_tensor();
	test_dump_rejects_null();
	test_dump_rejects_non_f32();
	TEST_REPORT();
}
