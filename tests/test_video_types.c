/*
 * tests/test_video_types.c - Verify video API types compile and have correct values
 *
 * Validates that the video tracking public types, error codes, and
 * constants are correctly defined and have the expected values.
 *
 * Key types:  sam3_propagate_dir, sam3_video_session
 * Depends on: sam3/sam3.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "sam3/sam3.h"

static void test_propagate_dir_values(void)
{
	ASSERT_EQ(SAM3_PROPAGATE_BOTH, 0);
	ASSERT_EQ(SAM3_PROPAGATE_FORWARD, 1);
	ASSERT_EQ(SAM3_PROPAGATE_BACKWARD, 2);
}

static void test_error_code(void)
{
	ASSERT_EQ(SAM3_EVIDEO, -7);
}

static void test_max_objects(void)
{
	ASSERT(SAM3_MAX_OBJECTS >= 16);
}

static void test_max_memory_frames(void)
{
	ASSERT(SAM3_MAX_MEMORY_FRAMES >= 16);
}

int main(void)
{
	test_propagate_dir_values();
	test_error_code();
	test_max_objects();
	test_max_memory_frames();
	TEST_REPORT();
}
