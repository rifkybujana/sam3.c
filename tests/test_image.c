/*
 * tests/test_image.c - Unit tests for image loading and resizing
 *
 * Tests sam3_image_load, sam3_image_resize, sam3_image_letterbox, and
 * sam3_image_free. Generates test BMP files programmatically.
 *
 * Key types:  sam3_image
 * Depends on: util/image.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <string.h>
#include "test_helpers.h"
#include "util/image.h"

/* Write a minimal 24-bit BMP file with solid color. */
static void write_test_bmp(const char *path, int w, int h,
			    uint8_t r, uint8_t g, uint8_t b)
{
	int row_bytes = (w * 3 + 3) & ~3;  /* Rows padded to 4 bytes */
	int pixel_size = row_bytes * h;
	int file_size = 54 + pixel_size;

	uint8_t header[54];
	memset(header, 0, 54);
	/* BM magic */
	header[0] = 'B'; header[1] = 'M';
	/* File size */
	header[2] = file_size & 0xFF;
	header[3] = (file_size >> 8) & 0xFF;
	header[4] = (file_size >> 16) & 0xFF;
	header[5] = (file_size >> 24) & 0xFF;
	/* Pixel data offset */
	header[10] = 54;
	/* DIB header size */
	header[14] = 40;
	/* Width */
	header[18] = w & 0xFF;
	header[19] = (w >> 8) & 0xFF;
	/* Height (positive = bottom-up) */
	header[22] = h & 0xFF;
	header[23] = (h >> 8) & 0xFF;
	/* Planes */
	header[26] = 1;
	/* Bits per pixel */
	header[28] = 24;
	/* Image size */
	header[34] = pixel_size & 0xFF;
	header[35] = (pixel_size >> 8) & 0xFF;
	header[36] = (pixel_size >> 16) & 0xFF;
	header[37] = (pixel_size >> 24) & 0xFF;

	FILE *f = fopen(path, "wb");
	if (!f) {
		fprintf(stderr, "FAIL: cannot create test BMP '%s'\n", path);
		abort();
	}
	fwrite(header, 1, 54, f);

	uint8_t *row = calloc(1, row_bytes);
	for (int x = 0; x < w; x++) {
		row[x * 3 + 0] = b;  /* BMP is BGR */
		row[x * 3 + 1] = g;
		row[x * 3 + 2] = r;
	}
	for (int y = 0; y < h; y++)
		fwrite(row, 1, row_bytes, f);
	free(row);
	fclose(f);
}

static void test_image_load_basic(void)
{
	const char *path = "/tmp/sam3_test_load.bmp";
	write_test_bmp(path, 8, 6, 255, 0, 0);

	struct sam3_image img = {0};
	enum sam3_error err = sam3_image_load(path, &img);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT_EQ(img.width, 8);
	ASSERT_EQ(img.height, 6);
	ASSERT(img.pixels != NULL);

	/* First pixel should be red (RGB) */
	ASSERT_EQ(img.pixels[0], 255);
	ASSERT_EQ(img.pixels[1], 0);
	ASSERT_EQ(img.pixels[2], 0);

	sam3_image_free(&img);
	ASSERT(img.pixels == NULL);
	remove(path);
}

static void test_image_load_null_args(void)
{
	struct sam3_image img = {0};
	ASSERT_EQ(sam3_image_load(NULL, &img), SAM3_EINVAL);
	ASSERT_EQ(sam3_image_load("/tmp/sam3_test.bmp", NULL), SAM3_EINVAL);
}

static void test_image_load_nonexistent(void)
{
	struct sam3_image img = {0};
	ASSERT_EQ(sam3_image_load("/tmp/sam3_no_such_file.bmp", &img), SAM3_EIO);
}

static void test_image_resize(void)
{
	const char *path = "/tmp/sam3_test_resize.bmp";
	write_test_bmp(path, 16, 8, 0, 255, 0);

	struct sam3_image src = {0};
	sam3_image_load(path, &src);

	struct sam3_image dst = {0};
	enum sam3_error err = sam3_image_resize(&src, &dst, 4, 2);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT_EQ(dst.width, 4);
	ASSERT_EQ(dst.height, 2);
	ASSERT(dst.pixels != NULL);

	/* Downscaled solid green should still be green */
	ASSERT_EQ(dst.pixels[0], 0);
	ASSERT_EQ(dst.pixels[1], 255);
	ASSERT_EQ(dst.pixels[2], 0);

	sam3_image_free(&dst);
	sam3_image_free(&src);
	remove(path);
}

static void test_image_letterbox_landscape(void)
{
	/* 200x100 landscape -> letterbox to 100x100 */
	const char *path = "/tmp/sam3_test_letterbox.bmp";
	write_test_bmp(path, 200, 100, 0, 0, 255);

	struct sam3_image src = {0};
	sam3_image_load(path, &src);

	struct sam3_image dst = {0};
	enum sam3_error err = sam3_image_letterbox(&src, &dst, 100);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT_EQ(dst.width, 100);
	ASSERT_EQ(dst.height, 100);

	/* Top-left of image area should be blue */
	/* Image is scaled to 100x50, centered vertically, so row 25 col 0 */
	int pad_top = 25;
	int idx = (pad_top * 100 + 0) * 3;
	ASSERT_EQ(dst.pixels[idx + 0], 0);
	ASSERT_EQ(dst.pixels[idx + 1], 0);
	ASSERT_EQ(dst.pixels[idx + 2], 255);

	/* Top-left padding row should be zero (black) */
	ASSERT_EQ(dst.pixels[0], 0);
	ASSERT_EQ(dst.pixels[1], 0);
	ASSERT_EQ(dst.pixels[2], 0);

	sam3_image_free(&dst);
	sam3_image_free(&src);
	remove(path);
}

static void test_image_letterbox_portrait(void)
{
	/* 100x200 portrait -> letterbox to 100x100 */
	const char *path = "/tmp/sam3_test_letterbox_p.bmp";
	write_test_bmp(path, 100, 200, 128, 64, 32);

	struct sam3_image src = {0};
	sam3_image_load(path, &src);

	struct sam3_image dst = {0};
	enum sam3_error err = sam3_image_letterbox(&src, &dst, 100);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT_EQ(dst.width, 100);
	ASSERT_EQ(dst.height, 100);

	/* Left padding column should be zero */
	ASSERT_EQ(dst.pixels[0], 0);
	ASSERT_EQ(dst.pixels[1], 0);
	ASSERT_EQ(dst.pixels[2], 0);

	/* Image area starts at col 25 */
	int pad_left = 25;
	int idx = pad_left * 3;
	ASSERT_EQ(dst.pixels[idx + 0], 128);
	ASSERT_EQ(dst.pixels[idx + 1], 64);
	ASSERT_EQ(dst.pixels[idx + 2], 32);

	sam3_image_free(&dst);
	sam3_image_free(&src);
	remove(path);
}

static void test_image_free_null(void)
{
	/* Should not crash */
	sam3_image_free(NULL);

	struct sam3_image img = {0};
	sam3_image_free(&img);
}

int main(void)
{
	test_image_load_basic();
	test_image_load_null_args();
	test_image_load_nonexistent();
	test_image_resize();
	test_image_letterbox_landscape();
	test_image_letterbox_portrait();
	test_image_free_null();
	TEST_REPORT();
}
