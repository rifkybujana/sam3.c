/*
 * tests/test_cli_common.c - Tests for CLI common definitions
 *
 * Validates exit code enum values and the sam3_error-to-exit mapping
 * function from tools/cli_common.h. Ensures the mapping stays in sync
 * with the error codes defined in sam3_types.h.
 *
 * Key types:  sam3_exit, sam3_error
 * Depends on: cli_common.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <string.h>

#include "cli_common.h"
#include "test_helpers.h"

static void test_exit_code_values(void)
{
	ASSERT_EQ(SAM3_EXIT_OK,       0);
	ASSERT_EQ(SAM3_EXIT_USAGE,    1);
	ASSERT_EQ(SAM3_EXIT_IO,       2);
	ASSERT_EQ(SAM3_EXIT_MODEL,    3);
	ASSERT_EQ(SAM3_EXIT_RUNTIME,  4);
	ASSERT_EQ(SAM3_EXIT_INTERNAL, 5);
}

static void test_error_to_exit_ok(void)
{
	ASSERT_EQ(sam3_error_to_exit(SAM3_OK), SAM3_EXIT_OK);
}

static void test_error_to_exit_einval(void)
{
	ASSERT_EQ(sam3_error_to_exit(SAM3_EINVAL), SAM3_EXIT_USAGE);
}

static void test_error_to_exit_enomem(void)
{
	ASSERT_EQ(sam3_error_to_exit(SAM3_ENOMEM), SAM3_EXIT_INTERNAL);
}

static void test_error_to_exit_eio(void)
{
	ASSERT_EQ(sam3_error_to_exit(SAM3_EIO), SAM3_EXIT_IO);
}

static void test_error_to_exit_ebackend(void)
{
	ASSERT_EQ(sam3_error_to_exit(SAM3_EBACKEND), SAM3_EXIT_INTERNAL);
}

static void test_error_to_exit_emodel(void)
{
	ASSERT_EQ(sam3_error_to_exit(SAM3_EMODEL), SAM3_EXIT_MODEL);
}

static void test_error_to_exit_edtype(void)
{
	ASSERT_EQ(sam3_error_to_exit(SAM3_EDTYPE), SAM3_EXIT_RUNTIME);
}

static void test_error_to_exit_unknown(void)
{
	/* An unrecognized error code should map to INTERNAL. */
	ASSERT_EQ(sam3_error_to_exit((enum sam3_error)-99), SAM3_EXIT_INTERNAL);
}

static void test_json_result(void)
{
	struct sam3_result r = {0};
	float scores[2] = {0.95f, 0.72f};
	float boxes[8] = {10, 20, 200, 300, 15, 25, 195, 295};
	r.n_masks = 2;
	r.mask_width = 256;
	r.mask_height = 256;
	r.best_mask = 0;
	r.iou_valid = 1;
	r.iou_scores = scores;
	r.boxes_valid = 1;
	r.boxes = boxes;

	char *buf = NULL;
	size_t len = 0;
	FILE *fp = open_memstream(&buf, &len);
	ASSERT(fp != NULL);
	cli_json_result(fp, &r);
	fclose(fp);

	ASSERT(strstr(buf, "\"n_masks\": 2") != NULL);
	ASSERT(strstr(buf, "\"mask_width\": 256") != NULL);
	ASSERT(strstr(buf, "\"best_mask\": 0") != NULL);
	ASSERT(strstr(buf, "\"iou_score\":") != NULL);
	free(buf);
}

int main(void)
{
	test_exit_code_values();
	test_error_to_exit_ok();
	test_error_to_exit_einval();
	test_error_to_exit_enomem();
	test_error_to_exit_eio();
	test_error_to_exit_ebackend();
	test_error_to_exit_emodel();
	test_error_to_exit_edtype();
	test_error_to_exit_unknown();
	test_json_result();

	TEST_REPORT();
}
