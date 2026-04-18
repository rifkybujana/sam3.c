/*
 * tests/test_video_session.c - Video session lifecycle tests
 *
 * Tests object ID mapping (add, lookup, remove) and edge cases
 * (duplicate add, remove nonexistent, capacity limit, null args).
 * Also exercises the prompt list + prompted-frame bitmap helpers
 * using a stack-installed bitmap/buffer so the helpers can be
 * verified without an arena-initialized session.
 *
 * Key types:  sam3_video_session, sam3_video_prompt
 * Depends on: test_helpers.h, model/video_session.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "test_helpers.h"
#include "model/video_session.h"

static void test_session_object_management(void)
{
	struct sam3_video_session session;
	memset(&session, 0, sizeof(session));

	/* Add first object */
	int idx = sam3_session_get_or_add_obj(&session, 42);
	ASSERT_EQ(idx, 0);
	ASSERT_EQ(session.n_objects, 1);
	ASSERT_EQ(session.objects[0].obj_id, 42);

	/* Same obj_id returns same index */
	int idx2 = sam3_session_get_or_add_obj(&session, 42);
	ASSERT_EQ(idx2, 0);
	ASSERT_EQ(session.n_objects, 1);

	/* Different obj_id gets new index */
	int idx3 = sam3_session_get_or_add_obj(&session, 99);
	ASSERT_EQ(idx3, 1);
	ASSERT_EQ(session.n_objects, 2);
}

static void test_session_remove_object(void)
{
	struct sam3_video_session session;
	memset(&session, 0, sizeof(session));

	sam3_session_get_or_add_obj(&session, 10);
	sam3_session_get_or_add_obj(&session, 20);
	sam3_session_get_or_add_obj(&session, 30);
	ASSERT_EQ(session.n_objects, 3);

	/* Remove middle element */
	int err = sam3_session_remove_obj(&session, 20);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT_EQ(session.n_objects, 2);
	ASSERT_EQ(session.objects[0].obj_id, 10);
	ASSERT_EQ(session.objects[1].obj_id, 30);

	/* Remove first element */
	err = sam3_session_remove_obj(&session, 10);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT_EQ(session.n_objects, 1);
	ASSERT_EQ(session.objects[0].obj_id, 30);

	/* Remove last remaining */
	err = sam3_session_remove_obj(&session, 30);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT_EQ(session.n_objects, 0);
}

static void test_session_remove_nonexistent(void)
{
	struct sam3_video_session session;
	memset(&session, 0, sizeof(session));

	int err = sam3_session_remove_obj(&session, 999);
	ASSERT_EQ(err, SAM3_EINVAL);
}

static void test_session_capacity_limit(void)
{
	struct sam3_video_session session;
	memset(&session, 0, sizeof(session));

	/* Fill to capacity */
	for (int i = 0; i < SAM3_MAX_OBJECTS; i++) {
		int idx = sam3_session_get_or_add_obj(&session, i + 100);
		ASSERT_EQ(idx, i);
	}
	ASSERT_EQ(session.n_objects, SAM3_MAX_OBJECTS);

	/* Next add should fail */
	int idx = sam3_session_get_or_add_obj(&session, 9999);
	ASSERT_EQ(idx, -1);
	ASSERT_EQ(session.n_objects, SAM3_MAX_OBJECTS);

	/* But looking up an existing one should still work */
	int idx2 = sam3_session_get_or_add_obj(&session, 100);
	ASSERT_EQ(idx2, 0);
}

static void test_session_null_args(void)
{
	int idx = sam3_session_get_or_add_obj(NULL, 1);
	ASSERT_EQ(idx, -1);

	int err = sam3_session_remove_obj(NULL, 1);
	ASSERT_EQ(err, SAM3_EINVAL);
}

static void test_session_prompts_append_and_clear(void)
{
	struct sam3_video_session session;
	struct sam3_video_prompt buf[8];
	uint8_t bitmap[16];

	memset(&session, 0, sizeof(session));
	memset(buf, 0, sizeof(buf));
	memset(bitmap, 0, sizeof(bitmap));

	/* Install stack-backed storage so helpers can run without arena */
	session.prompts         = buf;
	session.cap_prompts     = (int)(sizeof(buf) / sizeof(buf[0]));
	session.prompted_frames = bitmap;
	session.frames.n_frames = (int)sizeof(bitmap);

	struct sam3_video_prompt p;
	memset(&p, 0, sizeof(p));
	p.frame_idx        = 3;
	p.obj_internal_idx = 0;
	p.kind             = SAM3_PROMPT_POINTS;
	p.data.points.n         = 1;
	p.data.points.xys[0]    = 504.f;
	p.data.points.xys[1]    = 504.f;
	p.data.points.labels[0] = 1;

	ASSERT_EQ(sam3_session_add_prompt(&session, &p), SAM3_OK);
	ASSERT_EQ(session.n_prompts, 1);
	ASSERT_EQ(sam3_session_is_prompted(&session, 3), 1);
	ASSERT_EQ(sam3_session_is_prompted(&session, 0), 0);

	sam3_session_clear_prompts(&session);
	ASSERT_EQ(session.n_prompts, 0);
	ASSERT_EQ(sam3_session_is_prompted(&session, 3), 0);
}

static void test_session_per_object_bank_init(void)
{
	struct sam3_video_session s;
	memset(&s, 0, sizeof(s));
	s.frames.n_frames = 4;

	int idx1 = sam3_session_get_or_add_obj(&s, 101);
	int idx2 = sam3_session_get_or_add_obj(&s, 202);
	ASSERT_EQ(idx1, 0);
	ASSERT_EQ(idx2, 1);

	/* Per-object banks initialized */
	ASSERT_EQ(s.objects[0].obj_id, 101);
	ASSERT_EQ(s.objects[0].bank.capacity, 7);
	ASSERT_EQ(s.objects[1].obj_id, 202);
	ASSERT_EQ(s.objects[1].bank.n_cond, 0);
	ASSERT(s.objects[0].prompted_frames == NULL);
	ASSERT_EQ(s.objects[0].prev_mask_frame, -1);
	ASSERT(s.objects[0].prev_mask_logits == NULL);

	/* Per-object prompted bitmap */
	ASSERT_EQ(sam3_session_obj_is_prompted(&s, 0, 2), 0);
	ASSERT_EQ(sam3_session_obj_mark_prompted(&s, 0, 2), SAM3_OK);
	ASSERT_EQ(sam3_session_obj_is_prompted(&s, 0, 2), 1);
	ASSERT_EQ(sam3_session_obj_is_prompted(&s, 1, 2), 0); /* obj 2 untouched */

	/* Cleanup: free the bitmap */
	free(s.objects[0].prompted_frames);
}

int main(void)
{
	test_session_object_management();
	test_session_remove_object();
	test_session_remove_nonexistent();
	test_session_capacity_limit();
	test_session_null_args();
	test_session_prompts_append_and_clear();
	test_session_per_object_bank_init();
	TEST_REPORT();
}
