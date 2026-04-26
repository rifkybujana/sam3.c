/*
 * src/util/video_stubs.c - SAM3_NO_VIDEO public API stubs
 *
 * When the video subsystem is disabled at build time
 * (`-DSAM3_VIDEO=OFF`) the public sam3_video_* functions still need
 * to link because consumers like the Rust binding compile against
 * the full sam3.h header. Each stub returns SAM3_EVIDEO so any
 * caller that actually uses the video API gets a clean runtime error.
 *
 * NetraRT, sam3_cli's segment subcommand, and any other image-only
 * consumer never call these symbols, so they are unaffected.
 *
 * Key types:  sam3_video_session (opaque, never instantiated here)
 * Depends on: sam3/sam3.h
 * Used by:    nothing (link-time only)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifdef SAM3_NO_VIDEO

#include "sam3/sam3.h"

enum sam3_error sam3_video_start(sam3_ctx *ctx,
				 const char *resource_path,
				 sam3_video_session **out_session)
{
	(void)ctx; (void)resource_path;
	if (out_session)
		*out_session = NULL;
	return SAM3_EVIDEO;
}

enum sam3_error sam3_video_start_ex(sam3_ctx *ctx,
				    const char *resource_path,
				    const struct sam3_video_start_opts *opts,
				    sam3_video_session **out_session)
{
	(void)ctx; (void)resource_path; (void)opts;
	if (out_session)
		*out_session = NULL;
	return SAM3_EVIDEO;
}

enum sam3_error sam3_video_add_points(sam3_video_session *session,
				      int frame_idx, int obj_id,
				      const struct sam3_point *points,
				      int n_points,
				      struct sam3_video_frame_result *result)
{
	(void)session; (void)frame_idx; (void)obj_id;
	(void)points; (void)n_points; (void)result;
	return SAM3_EVIDEO;
}

enum sam3_error sam3_video_add_box(sam3_video_session *session,
				   int frame_idx, int obj_id,
				   const struct sam3_box *box,
				   struct sam3_video_frame_result *result)
{
	(void)session; (void)frame_idx; (void)obj_id;
	(void)box; (void)result;
	return SAM3_EVIDEO;
}

enum sam3_error sam3_video_add_mask(sam3_video_session *session,
				    int frame_idx, int obj_id,
				    const uint8_t *mask,
				    int mask_h, int mask_w,
				    struct sam3_video_frame_result *result)
{
	(void)session; (void)frame_idx; (void)obj_id;
	(void)mask; (void)mask_h; (void)mask_w; (void)result;
	return SAM3_EVIDEO;
}

enum sam3_error sam3_video_propagate(sam3_video_session *session,
				     int direction,
				     sam3_video_frame_cb callback,
				     void *user_data)
{
	(void)session; (void)direction; (void)callback; (void)user_data;
	return SAM3_EVIDEO;
}

enum sam3_error sam3_video_remove_object(sam3_video_session *session,
					 int obj_id)
{
	(void)session; (void)obj_id;
	return SAM3_EVIDEO;
}

enum sam3_error sam3_video_reset(sam3_video_session *session)
{
	(void)session;
	return SAM3_EVIDEO;
}

void sam3_video_end(sam3_video_session *session)
{
	(void)session;
}

int sam3_video_frame_count(const sam3_video_session *session)
{
	(void)session;
	return 0;
}

void sam3_video_frame_result_free(struct sam3_video_frame_result *r)
{
	(void)r;
}

#endif /* SAM3_NO_VIDEO */
