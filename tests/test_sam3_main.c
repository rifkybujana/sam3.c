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

#include <sys/wait.h>
#include "test_helpers.h"

/* Path to sam3 CLI binary (CTest runs from build directory) */
#define SAM3_MAIN "./sam3_cli"

static void test_help_returns_zero(void)
{
	int rc = system(SAM3_MAIN " -h > /dev/null 2>&1");
	ASSERT_EQ(WEXITSTATUS(rc), 0);
}

static void test_no_args_returns_error(void)
{
	int rc = system(SAM3_MAIN " > /dev/null 2>&1");
	ASSERT(WEXITSTATUS(rc) != 0);
}

static void test_missing_model_returns_error(void)
{
	int rc = system(SAM3_MAIN " -i foo.png -p 1,2,1"
			" > /dev/null 2>&1");
	ASSERT(WEXITSTATUS(rc) != 0);
}

static void test_missing_image_returns_error(void)
{
	int rc = system(SAM3_MAIN " -m foo.sam3 -p 1,2,1"
			" > /dev/null 2>&1");
	ASSERT(WEXITSTATUS(rc) != 0);
}

static void test_missing_prompts_returns_error(void)
{
	int rc = system(SAM3_MAIN " -m foo.sam3 -i bar.png"
			" > /dev/null 2>&1");
	ASSERT(WEXITSTATUS(rc) != 0);
}

static void test_unknown_option_returns_error(void)
{
	int rc = system(SAM3_MAIN " --bogus > /dev/null 2>&1");
	ASSERT(WEXITSTATUS(rc) != 0);
}

static void test_bad_point_format_returns_error(void)
{
	int rc = system(SAM3_MAIN " -m f.sam3 -i i.png"
			" -p notapoint > /dev/null 2>&1");
	ASSERT(WEXITSTATUS(rc) != 0);
}

static void test_bad_box_format_returns_error(void)
{
	int rc = system(SAM3_MAIN " -m f.sam3 -i i.png"
			" -b 1,2 > /dev/null 2>&1");
	ASSERT(WEXITSTATUS(rc) != 0);
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
