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

	TEST_REPORT();
}
