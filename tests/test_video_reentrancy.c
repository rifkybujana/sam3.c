/*
 * tests/test_video_reentrancy.c - in_propagate guard
 *
 * Unit-level check that remove_object and reset reject calls made
 * while session->in_propagate is set. Doesn't run the actual
 * propagate sweep — simulates the guard by setting the flag directly.
 *
 * Key types: sam3_video_session
 * Depends on: sam3/sam3.h, model/video_session.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include "test_helpers.h"
#include "sam3/sam3.h"
#include "model/video_session.h"

static void test_remove_rejected_during_propagate(void)
{
	struct sam3_video_session session = {0};
	session.frames.n_frames = 2;
	sam3_session_get_or_add_obj(&session, 42);

	session.in_propagate = 1;
	enum sam3_error err = sam3_video_remove_object(&session, 42);
	session.in_propagate = 0;

	ASSERT_EQ(err, SAM3_EINVAL);
	ASSERT_EQ(session.n_objects, 1); /* still there */

	/* Clean up: remove outside the guard. */
	ASSERT_EQ(sam3_video_remove_object(&session, 42), SAM3_OK);
	ASSERT_EQ(session.n_objects, 0);
}

static void test_reset_rejected_during_propagate(void)
{
	struct sam3_video_session session = {0};
	session.frames.n_frames = 2;
	sam3_session_get_or_add_obj(&session, 1);

	session.in_propagate = 1;
	enum sam3_error err = sam3_video_reset(&session);
	session.in_propagate = 0;

	ASSERT_EQ(err, SAM3_EINVAL);
	ASSERT_EQ(session.n_objects, 1);

	sam3_session_remove_obj(&session, 1);
}

int main(void)
{
	test_remove_rejected_during_propagate();
	test_reset_rejected_during_propagate();
	TEST_REPORT();
}
