/*
 * src/model/video_session.c - Video tracking session state management
 *
 * Implements object ID mapping and stored-prompt helpers for the
 * video tracking session. The session maps user-facing object IDs to
 * contiguous internal indices used by the tracker and memory bank.
 * Prompt helpers append into a preallocated arena-owned list and
 * maintain a byte-per-frame "prompted" bitmap so propagate can skip
 * frames that already have prompts.
 *
 * Key types:  sam3_video_session, sam3_video_prompt
 * Depends on: model/video_session.h, util/log.h
 * Used by:    model/sam3_video.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include <string.h>

#include "model/video_session.h"
#include "util/log.h"

int sam3_session_get_or_add_obj(struct sam3_video_session *session, int obj_id)
{
	if (!session) {
		sam3_log_error("get_or_add_obj: NULL session");
		return -1;
	}

	/* Linear scan for existing obj_id */
	for (int i = 0; i < session->n_objects; i++) {
		if (session->objects[i].obj_id == obj_id)
			return i;
	}

	/* Not found — try to add */
	if (session->n_objects >= SAM3_MAX_OBJECTS) {
		sam3_log_error("get_or_add_obj: session full (%d objects)",
			       SAM3_MAX_OBJECTS);
		return -1;
	}

	int idx = session->n_objects;
	session->objects[idx].obj_id           = obj_id;
	session->objects[idx].prompted_frames  = NULL;
	session->objects[idx].prev_mask_logits = NULL;
	session->objects[idx].prev_mask_frame  = -1;
	sam3_memory_bank_init(&session->objects[idx].bank,
			      /*capacity=*/7,
			      /*max_cond_in_attn=*/4,
			      /*temporal_stride=*/1,
			      /*mf_threshold=*/0.01f);
	session->n_objects++;
	return idx;
}

enum sam3_error sam3_session_remove_obj(struct sam3_video_session *session,
					int obj_id)
{
	if (!session) {
		sam3_log_error("remove_obj: NULL session");
		return SAM3_EINVAL;
	}

	/* Linear scan for obj_id */
	for (int i = 0; i < session->n_objects; i++) {
		if (session->objects[i].obj_id == obj_id) {
			/* Free malloc'd bitmap before compaction */
			free(session->objects[i].prompted_frames);
			/* bank and prev_mask_logits live in arena — null is enough */

			if (i < session->n_objects - 1) {
				memmove(&session->objects[i],
					&session->objects[i + 1],
					(size_t)(session->n_objects - 1 - i) *
					sizeof(session->objects[0]));
			}
			session->n_objects--;
			/* Zero the now-unused trailing slot for cleanliness */
			memset(&session->objects[session->n_objects], 0,
			       sizeof(session->objects[0]));
			return SAM3_OK;
		}
	}

	sam3_log_error("remove_obj: obj_id %d not found", obj_id);
	return SAM3_EINVAL;
}

int sam3_session_add_prompt(struct sam3_video_session *s,
			    const struct sam3_video_prompt *p)
{
	if (!s || !p) {
		sam3_log_error("add_prompt: NULL argument");
		return SAM3_EINVAL;
	}

	if (!s->prompts || s->cap_prompts <= 0) {
		sam3_log_error("add_prompt: prompt list not allocated");
		return SAM3_EINVAL;
	}

	if (p->frame_idx < 0 || p->frame_idx >= s->frames.n_frames) {
		sam3_log_error("add_prompt: frame_idx %d out of range [0,%d)",
			       p->frame_idx, s->frames.n_frames);
		return SAM3_EINVAL;
	}

	if (s->n_prompts >= s->cap_prompts) {
		sam3_log_error("add_prompt: capacity reached (%d)",
			       s->cap_prompts);
		return SAM3_ENOMEM;
	}

	s->prompts[s->n_prompts] = *p;
	s->n_prompts++;

	if (s->prompted_frames) {
		s->prompted_frames[p->frame_idx] = 1;
	}

	return SAM3_OK;
}

void sam3_session_clear_prompts(struct sam3_video_session *s)
{
	if (!s) {
		return;
	}

	s->n_prompts = 0;

	if (s->prompted_frames && s->frames.n_frames > 0) {
		memset(s->prompted_frames, 0,
		       (size_t)s->frames.n_frames);
	}
}

int sam3_session_is_prompted(const struct sam3_video_session *s,
			     int frame_idx)
{
	if (!s) {
		return 0;
	}

	if (frame_idx < 0 || frame_idx >= s->frames.n_frames) {
		return 0;
	}

	return s->prompted_frames ? s->prompted_frames[frame_idx] : 0;
}

int sam3_session_obj_is_prompted(const struct sam3_video_session *s,
				 int obj_idx, int frame_idx)
{
	if (!s || obj_idx < 0 || obj_idx >= s->n_objects)
		return 0;
	if (frame_idx < 0 || frame_idx >= s->frames.n_frames)
		return 0;

	const uint8_t *bm = s->objects[obj_idx].prompted_frames;
	if (!bm)
		return 0;
	return (bm[frame_idx >> 3] >> (frame_idx & 7)) & 1;
}

int sam3_session_obj_mark_prompted(struct sam3_video_session *s,
				   int obj_idx, int frame_idx)
{
	if (!s || obj_idx < 0 || obj_idx >= s->n_objects)
		return SAM3_EINVAL;
	if (frame_idx < 0 || frame_idx >= s->frames.n_frames)
		return SAM3_EINVAL;

	if (!s->objects[obj_idx].prompted_frames) {
		size_t bytes = (size_t)((s->frames.n_frames + 7) / 8);
		s->objects[obj_idx].prompted_frames = calloc(bytes, 1);
		if (!s->objects[obj_idx].prompted_frames)
			return SAM3_ENOMEM;
	}
	s->objects[obj_idx].prompted_frames[frame_idx >> 3] |=
		(uint8_t)(1u << (frame_idx & 7));
	return SAM3_OK;
}
