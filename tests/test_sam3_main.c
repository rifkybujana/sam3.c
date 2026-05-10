/*
 * tests/test_sam3_main.c - CLI argument parsing tests for sam3_main
 *
 * Tests the sam3_main binary's argument validation by invoking it as
 * a subprocess and checking exit codes. Does not test inference (stubs).
 *
 * Key types:  (standalone test)
 * Depends on: test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"

#ifndef SAM3_CLI_PATH
#define SAM3_CLI_PATH "./sam3_cli"
#endif

#ifdef _WIN32
#define SAM3_NULL_DEVICE "NUL"
#define SAM3_EXIT_STATUS(rc) (rc)
#else
#include <sys/wait.h>
#define SAM3_NULL_DEVICE "/dev/null"
#define SAM3_EXIT_STATUS(rc) WEXITSTATUS(rc)
#endif

/* Path to sam3 CLI binary (CTest runs from build directory) */
#define RUN_SAM3(args) system(SAM3_CLI_PATH " " args " > " SAM3_NULL_DEVICE " 2>&1")

static void test_help_returns_zero(void)
{
	int rc = RUN_SAM3("-h");
	ASSERT_EQ(SAM3_EXIT_STATUS(rc), 0);
}

static void test_no_args_returns_error(void)
{
	int rc = RUN_SAM3("");
	ASSERT(SAM3_EXIT_STATUS(rc) != 0);
}

static void test_missing_model_returns_error(void)
{
	int rc = RUN_SAM3("-i foo.png -p 1,2,1");
	ASSERT(SAM3_EXIT_STATUS(rc) != 0);
}

static void test_missing_image_returns_error(void)
{
	int rc = RUN_SAM3("-m foo.sam3 -p 1,2,1");
	ASSERT(SAM3_EXIT_STATUS(rc) != 0);
}

static void test_missing_prompts_returns_error(void)
{
	int rc = RUN_SAM3("-m foo.sam3 -i bar.png");
	ASSERT(SAM3_EXIT_STATUS(rc) != 0);
}

static void test_unknown_option_returns_error(void)
{
	int rc = RUN_SAM3("--bogus");
	ASSERT(SAM3_EXIT_STATUS(rc) != 0);
}

static void test_bad_point_format_returns_error(void)
{
	int rc = RUN_SAM3("-m f.sam3 -i i.png -p notapoint");
	ASSERT(SAM3_EXIT_STATUS(rc) != 0);
}

static void test_bad_box_format_returns_error(void)
{
	int rc = RUN_SAM3("-m f.sam3 -i i.png -b 1,2");
	ASSERT(SAM3_EXIT_STATUS(rc) != 0);
}

int main(void)
{
	test_help_returns_zero();
	test_no_args_returns_error();
	test_missing_model_returns_error();
	test_missing_image_returns_error();
	test_missing_prompts_returns_error();
	test_unknown_option_returns_error();
	test_bad_point_format_returns_error();
	test_bad_box_format_returns_error();

	TEST_REPORT();
}
