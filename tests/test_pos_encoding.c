/*
 * tests/test_pos_encoding.c - Unit tests for 2D sinusoidal position encoding
 *
 * Tests precomputation of position encodings: output shape, value range,
 * uniqueness across spatial positions, and full-size (72x72) allocation.
 * Uses arena allocation directly (no backend needed since this is CPU-only).
 *
 * Key types:  sam3_pos_encoding
 * Depends on: test_helpers.h, model/position_encoding.h, core/alloc.h
 * Used by:    CTest
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "model/position_encoding.h"
#include "core/alloc.h"

/* --- test_pos_encoding_shape --- */

static void test_pos_encoding_shape(void)
{
	struct sam3_arena arena;
	sam3_arena_init(&arena, 64 * 1024 * 1024);

	struct sam3_pos_encoding pe;
	enum sam3_error err = sam3_pos_encoding_precompute(
		&pe, 4, 4, 8, &arena);

	ASSERT_EQ(err, SAM3_OK);
	ASSERT(pe.cached != NULL);
	ASSERT_EQ(pe.cached->n_dims, 3);
	ASSERT_EQ(pe.cached->dims[0], 4);
	ASSERT_EQ(pe.cached->dims[1], 4);
	ASSERT_EQ(pe.cached->dims[2], 16);  /* num_pos_feats * 2 */
	ASSERT_EQ(pe.num_pos_feats, 8);

	struct sam3_tensor *t = sam3_pos_encoding_get(&pe);
	ASSERT(t == pe.cached);

	sam3_arena_free(&arena);
}

/* --- test_pos_encoding_range --- */

static void test_pos_encoding_range(void)
{
	struct sam3_arena arena;
	sam3_arena_init(&arena, 64 * 1024 * 1024);

	struct sam3_pos_encoding pe;
	enum sam3_error err = sam3_pos_encoding_precompute(
		&pe, 4, 4, 8, &arena);

	ASSERT_EQ(err, SAM3_OK);

	float *data = (float *)pe.cached->data;
	int n = sam3_tensor_nelems(pe.cached);

	for (int i = 0; i < n; i++) {
		ASSERT(data[i] >= -1.0f);
		ASSERT(data[i] <= 1.0f);
		ASSERT(data[i] == data[i]);  /* Not NaN */
	}

	sam3_arena_free(&arena);
}

/* --- test_pos_encoding_different_positions --- */

static void test_pos_encoding_different_positions(void)
{
	struct sam3_arena arena;
	sam3_arena_init(&arena, 64 * 1024 * 1024);

	struct sam3_pos_encoding pe;
	enum sam3_error err = sam3_pos_encoding_precompute(
		&pe, 4, 4, 8, &arena);

	ASSERT_EQ(err, SAM3_OK);

	float *data = (float *)pe.cached->data;
	int out_dim = pe.num_pos_feats * 2;  /* 16 */

	/* Position [0,0] vs [1,1] — must differ in at least one element */
	float *pos_0_0 = data;
	float *pos_1_1 = data + (1 * 4 + 1) * out_dim;

	int differs = 0;
	for (int i = 0; i < out_dim; i++) {
		if (fabs((double)pos_0_0[i] - (double)pos_1_1[i]) > 1e-6)
			differs = 1;
	}
	ASSERT(differs);

	/* Position [0,0] vs [0,1] — x differs, y same */
	float *pos_0_1 = data + (0 * 4 + 1) * out_dim;

	/* Y-encoding half (first num_pos_feats) should be same */
	int y_same = 1;
	for (int i = 0; i < pe.num_pos_feats; i++) {
		if (fabs((double)pos_0_0[i] - (double)pos_0_1[i]) > 1e-6)
			y_same = 0;
	}
	ASSERT(y_same);

	/* X-encoding half (second num_pos_feats) should differ */
	int x_differs = 0;
	for (int i = pe.num_pos_feats; i < out_dim; i++) {
		if (fabs((double)pos_0_0[i] - (double)pos_0_1[i]) > 1e-6)
			x_differs = 1;
	}
	ASSERT(x_differs);

	sam3_arena_free(&arena);
}

/* --- test_pos_encoding_full_size --- */

static void test_pos_encoding_full_size(void)
{
	struct sam3_arena arena;
	sam3_arena_init(&arena, 64 * 1024 * 1024);

	struct sam3_pos_encoding pe;
	enum sam3_error err = sam3_pos_encoding_precompute(
		&pe, 72, 72, 256, &arena);

	ASSERT_EQ(err, SAM3_OK);
	ASSERT(pe.cached != NULL);
	ASSERT_EQ(pe.cached->n_dims, 3);
	ASSERT_EQ(pe.cached->dims[0], 72);
	ASSERT_EQ(pe.cached->dims[1], 72);
	ASSERT_EQ(pe.cached->dims[2], 512);  /* 256 * 2 */

	/* Spot-check a few values are in [-1, 1] */
	float *data = (float *)pe.cached->data;
	int total = 72 * 72 * 512;
	for (int i = 0; i < total; i += 1000) {
		ASSERT(data[i] >= -1.0f);
		ASSERT(data[i] <= 1.0f);
	}

	sam3_arena_free(&arena);
}

/* --- test_pos_encoding_invalid_args --- */

static void test_pos_encoding_invalid_args(void)
{
	struct sam3_arena arena;
	sam3_arena_init(&arena, 1024 * 1024);

	struct sam3_pos_encoding pe;

	ASSERT_EQ(sam3_pos_encoding_precompute(NULL, 4, 4, 8, &arena),
		  SAM3_EINVAL);
	ASSERT_EQ(sam3_pos_encoding_precompute(&pe, 0, 4, 8, &arena),
		  SAM3_EINVAL);
	ASSERT_EQ(sam3_pos_encoding_precompute(&pe, 4, -1, 8, &arena),
		  SAM3_EINVAL);
	ASSERT_EQ(sam3_pos_encoding_precompute(&pe, 4, 4, 0, &arena),
		  SAM3_EINVAL);

	/* NULL arena */
	ASSERT_EQ(sam3_pos_encoding_precompute(&pe, 4, 4, 8, NULL),
		  SAM3_EINVAL);

	/* sam3_pos_encoding_get with NULL */
	ASSERT(sam3_pos_encoding_get(NULL) == NULL);

	sam3_arena_free(&arena);
}

/* --- Main --- */

int main(void)
{
	test_pos_encoding_shape();
	test_pos_encoding_range();
	test_pos_encoding_different_positions();
	test_pos_encoding_full_size();
	test_pos_encoding_invalid_args();

	TEST_REPORT();
}
