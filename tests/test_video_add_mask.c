/*
 * tests/test_video_add_mask.c - sam3_video_add_mask input validation
 *
 * Unit-level checks that do not require a real image encoder or weight file.
 * Verifies that NULL arguments and zero/negative mask dimensions are rejected
 * before any session state is touched. A real end-to-end test that exercises
 * the memory encoder belongs in test_video_e2e (Task 5.4).
 *
 * Key types:  sam3_video_session
 * Depends on: sam3/sam3.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "test_helpers.h"
#include "sam3/sam3.h"

/*
 * test_add_mask_null_session - NULL session rejected before any deref.
 */
static void
test_add_mask_null_session(void)
{
	uint8_t buf[4] = {1, 0, 1, 0};
	struct sam3_video_frame_result r;

	memset(&r, 0, sizeof(r));
	ASSERT_EQ(sam3_video_add_mask(NULL, 0, 1, buf, 2, 2, &r),
		  SAM3_EINVAL);
}

/*
 * test_add_mask_null_mask - NULL mask pointer rejected.
 */
static void
test_add_mask_null_mask(void)
{
	struct sam3_video_frame_result r;
	/*
	 * We need a non-NULL session pointer that won't be dereferenced.
	 * The null-mask check fires before session->ctx is read, so casting
	 * a dummy int to a pointer is safe here.
	 */
	sam3_video_session *fake = (sam3_video_session *)(uintptr_t)1;

	memset(&r, 0, sizeof(r));
	ASSERT_EQ(sam3_video_add_mask(fake, 0, 1, NULL, 2, 2, &r),
		  SAM3_EINVAL);
}

/*
 * test_add_mask_null_result - NULL result pointer rejected.
 */
static void
test_add_mask_null_result(void)
{
	uint8_t buf[4] = {1, 0, 1, 0};
	sam3_video_session *fake = (sam3_video_session *)(uintptr_t)1;

	ASSERT_EQ(sam3_video_add_mask(fake, 0, 1, buf, 2, 2, NULL),
		  SAM3_EINVAL);
}

/*
 * test_add_mask_zero_dims - Zero or negative mask dimensions rejected.
 *
 * These checks fire before session->ctx is accessed, so a fake pointer
 * is still safe here.
 */
static void
test_add_mask_zero_dims(void)
{
	uint8_t buf[4] = {1};
	sam3_video_session *fake = (sam3_video_session *)(uintptr_t)1;
	struct sam3_video_frame_result r;

	memset(&r, 0, sizeof(r));
	ASSERT_EQ(sam3_video_add_mask(fake, 0, 1, buf, 0, 2, &r),
		  SAM3_EINVAL);
	ASSERT_EQ(sam3_video_add_mask(fake, 0, 1, buf, 2, 0, &r),
		  SAM3_EINVAL);
	ASSERT_EQ(sam3_video_add_mask(fake, 0, 1, buf, -1, 2, &r),
		  SAM3_EINVAL);
	ASSERT_EQ(sam3_video_add_mask(fake, 0, 1, buf, 2, -1, &r),
		  SAM3_EINVAL);
}

int
main(void)
{
	test_add_mask_null_session();
	test_add_mask_null_mask();
	test_add_mask_null_result();
	test_add_mask_zero_dims();

	TEST_REPORT();
	return tests_failed ? 1 : 0;
}
