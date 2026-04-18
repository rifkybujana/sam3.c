/*
 * tools/cli_track.c - SAM3 track subcommand
 *
 * Runs video-object tracking: loads a model, opens a video session,
 * applies point/box prompts on a chosen frame, propagates masks across
 * frames, and writes a grayscale PNG per frame into the output directory.
 * Mirrors the cli_segment.c parse/run/dispatch shape and replaces the
 * legacy --video block from tools/sam3_main.c.
 *
 * Key types:  track_args, track_prompt_entry
 * Depends on: cli_common.h, cli_track.h, sam3/sam3.h, util/log.h,
 *             util/error.h
 * Used by:    tools/sam3_cli.c, tests/test_cli_track.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cli_common.h"
#include "cli_track.h"

#include "sam3/sam3.h"
#include "util/log.h"
#include "util/error.h"
#include "util/profile.h"

/* Suppress warnings in vendored stb header (declarations only) */
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wstrict-prototypes"
#pragma clang diagnostic ignored "-Wdouble-promotion"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wimplicit-fallthrough"
#pragma clang diagnostic ignored "-Wcomma"
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
#pragma clang diagnostic ignored "-Wcast-align"
#pragma clang diagnostic ignored "-Wcast-qual"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wmissing-prototypes"
#pragma GCC diagnostic ignored "-Wstrict-prototypes"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif

#include "util/vendor/stb_image_write.h"

#ifdef __clang__
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

/* --- argv parsing helpers --- */

/*
 * parse_int_arg - Strict integer parse from argv.
 *
 * Returns 0 on success and writes *out. Returns -1 on any malformed
 * input (non-numeric, trailing garbage, empty, overflow) after
 * logging a descriptive error that names the flag.
 */
static int parse_int_arg(const char *flag, const char *s, int *out)
{
	char *end;
	errno = 0;
	long v = strtol(s, &end, 10);
	if (s[0] == '\0' || *end != '\0' || errno == ERANGE ||
	    v < INT_MIN || v > INT_MAX) {
		sam3_log_error("%s requires an integer (got '%s')",
			       flag, s);
		return -1;
	}
	*out = (int)v;
	return 0;
}

/*
 * parse_point - Parse a "x,y,label" string into a point prompt.
 *
 * @str:    Input string in format "x,y,label".
 * @prompt: Output prompt (populated with SAM3_PROMPT_POINT).
 *
 * Returns 0 on success, 1 on parse error.
 */
static int parse_point(const char *str, struct sam3_prompt *prompt)
{
	float x, y;
	int label;

	if (sscanf(str, "%f,%f,%d", &x, &y, &label) != 3)
		return 1;

	prompt->type = SAM3_PROMPT_POINT;
	prompt->point.x = x;
	prompt->point.y = y;
	prompt->point.label = label;
	return 0;
}

/*
 * parse_box - Parse a "x1,y1,x2,y2" string into a box prompt.
 *
 * @str:    Input string in format "x1,y1,x2,y2".
 * @prompt: Output prompt (populated with SAM3_PROMPT_BOX).
 *
 * Returns 0 on success, 1 on parse error.
 */
static int parse_box(const char *str, struct sam3_prompt *prompt)
{
	float x1, y1, x2, y2;

	if (sscanf(str, "%f,%f,%f,%f", &x1, &y1, &x2, &y2) != 4)
		return 1;

	prompt->type = SAM3_PROMPT_BOX;
	prompt->box.x1 = x1;
	prompt->box.y1 = y1;
	prompt->box.x2 = x2;
	prompt->box.y2 = y2;
	return 0;
}

static void print_usage(const char *prog)
{
	printf("sam3 track - video object tracking v%s\n\n",
	       sam3_version());
	printf("Usage: %s --model <path> --video <path> "
	       "--output <dir> [prompts] [options]\n\n", prog);
	printf("Required:\n");
	printf("  --model <path>        Model weights file (.sam3)\n");
	printf("  --video <path>        Video file or frame "
	       "directory\n");
	printf("  --output <dir>        Output directory for "
	       "per-frame PNG masks\n");
	printf("\nPrompts (at least one required):\n");
	printf("  --point x,y,label     Point prompt (repeatable, "
	       "label: 1=fg, 0=bg)\n");
	printf("  --box x1,y1,x2,y2     Box prompt (repeatable)\n");
	printf("  --obj-id <id>         Object id for the next "
	       "prompt(s) (default: 0)\n");
	printf("\nOptions:\n");
	printf("  --frame <idx>         Frame index where prompts "
	       "apply (default: 0)\n");
	printf("  --propagate <dir>     Propagation direction: "
	       "none | forward |\n");
	printf("                        backward | both "
	       "(default: both)\n");
	printf("  -v, --verbose         Enable debug logging\n");
	printf("  --profile             Print stage/op timing report at end\n");
	printf("  -h, --help            Show this help\n");
}

int cli_track_parse(int argc, char **argv, struct track_args *out)
{
	int cur_obj_id = 0;

	out->model_path = NULL;
	out->video_path = NULL;
	out->output_dir = NULL;
	out->n_prompts  = 0;
	out->frame_idx  = 0;
	out->propagate  = TRACK_PROPAGATE_BOTH;
	out->verbose    = 0;
	out->profile    = 0;

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-h") == 0 ||
		    strcmp(argv[i], "--help") == 0) {
			return 1;
		} else if (strcmp(argv[i], "--model") == 0) {
			if (++i >= argc) {
				sam3_log_error(
					"--model requires a path");
				return -1;
			}
			out->model_path = argv[i];
		} else if (strcmp(argv[i], "--video") == 0) {
			if (++i >= argc) {
				sam3_log_error(
					"--video requires a path");
				return -1;
			}
			out->video_path = argv[i];
		} else if (strcmp(argv[i], "--output") == 0) {
			if (++i >= argc) {
				sam3_log_error(
					"--output requires a path");
				return -1;
			}
			out->output_dir = argv[i];
		} else if (strcmp(argv[i], "--obj-id") == 0) {
			if (++i >= argc) {
				sam3_log_error(
					"--obj-id requires a value");
				return -1;
			}
			int tmp;
			if (parse_int_arg("--obj-id", argv[i], &tmp))
				return -1;
			if (tmp < 0 || tmp >= SAM3_MAX_OBJECTS) {
				sam3_log_error(
					"--obj-id %d out of range "
					"[0,%d)", tmp,
					SAM3_MAX_OBJECTS);
				return -1;
			}
			cur_obj_id = tmp;
		} else if (strcmp(argv[i], "--point") == 0) {
			if (++i >= argc) {
				sam3_log_error(
					"--point requires x,y,label");
				return -1;
			}
			if (out->n_prompts >=
			    SAM3_CLI_TRACK_MAX_PROMPTS) {
				sam3_log_error(
					"too many prompts (max %d)",
					SAM3_CLI_TRACK_MAX_PROMPTS);
				return -1;
			}
			struct track_prompt_entry *e =
				&out->prompts[out->n_prompts];
			if (parse_point(argv[i], &e->prompt)) {
				sam3_log_error(
					"invalid point '%s' "
					"(expected x,y,label)",
					argv[i]);
				return -1;
			}
			e->obj_id = cur_obj_id;
			out->n_prompts++;
		} else if (strcmp(argv[i], "--box") == 0) {
			if (++i >= argc) {
				sam3_log_error(
					"--box requires x1,y1,x2,y2");
				return -1;
			}
			if (out->n_prompts >=
			    SAM3_CLI_TRACK_MAX_PROMPTS) {
				sam3_log_error(
					"too many prompts (max %d)",
					SAM3_CLI_TRACK_MAX_PROMPTS);
				return -1;
			}
			struct track_prompt_entry *e =
				&out->prompts[out->n_prompts];
			if (parse_box(argv[i], &e->prompt)) {
				sam3_log_error(
					"invalid box '%s' "
					"(expected x1,y1,x2,y2)",
					argv[i]);
				return -1;
			}
			e->obj_id = cur_obj_id;
			out->n_prompts++;
		} else if (strcmp(argv[i], "--frame") == 0) {
			if (++i >= argc) {
				sam3_log_error(
					"--frame requires an index");
				return -1;
			}
			int tmp;
			if (parse_int_arg("--frame", argv[i], &tmp))
				return -1;
			if (tmp < 0) {
				sam3_log_error(
					"--frame %d must be >= 0",
					tmp);
				return -1;
			}
			out->frame_idx = tmp;
		} else if (strcmp(argv[i], "--propagate") == 0) {
			if (++i >= argc) {
				sam3_log_error(
					"--propagate requires a "
					"direction");
				return -1;
			}
			if (strcmp(argv[i], "none") == 0) {
				out->propagate = TRACK_PROPAGATE_NONE;
			} else if (strcmp(argv[i], "forward") == 0) {
				out->propagate =
					TRACK_PROPAGATE_FORWARD;
			} else if (strcmp(argv[i], "backward") == 0) {
				out->propagate =
					TRACK_PROPAGATE_BACKWARD;
			} else if (strcmp(argv[i], "both") == 0) {
				out->propagate = TRACK_PROPAGATE_BOTH;
			} else {
				sam3_log_error(
					"--propagate must be none, "
					"forward, backward, or "
					"both (got '%s')", argv[i]);
				return -1;
			}
		} else if (strcmp(argv[i], "-v") == 0 ||
			   strcmp(argv[i], "--verbose") == 0) {
			out->verbose = 1;
		} else if (strcmp(argv[i], "--profile") == 0) {
			out->profile = 1;
		} else {
			sam3_log_error("unknown option '%s'", argv[i]);
			return -1;
		}
	}

	if (!out->model_path) {
		sam3_log_error("--model <path> is required");
		return -1;
	}
	if (!out->video_path) {
		sam3_log_error("--video <path> is required");
		return -1;
	}
	if (!out->output_dir) {
		sam3_log_error("--output <dir> is required");
		return -1;
	}
	if (out->n_prompts == 0) {
		sam3_log_error(
			"at least one --point or --box prompt is "
			"required");
		return -1;
	}

	return 0;
}

/* --- Per-frame mask writer --- */

/*
 * track_frame_ctx - User-data carried through the propagate callback.
 *
 * Carries the output directory plus a reusable scratch buffer for the
 * grayscale PNG conversion. The buffer grows on demand (never shrinks)
 * and is freed once by cli_track_run at session end, keeping the
 * per-frame hot path allocation-free (Performance Rule #1).
 */
struct track_frame_ctx {
	const char           *output_dir;
	uint8_t              *gray_buf;
	size_t                gray_cap;
	struct sam3_profiler *profiler; /* NULL when --profile is off */
};

/*
 * write_frame_mask_png - Write a thresholded grayscale PNG for one frame.
 *
 * Pixels >= 0 become white (255), below become black. Uses the
 * caller-owned scratch buffer in @fc (growing it if needed) instead of
 * allocating per call.
 *
 * Returns 0 on success, non-zero on error.
 */
static int write_frame_mask_png(struct track_frame_ctx *fc,
				const char *path, const float *data,
				int w, int h)
{
	size_t npix = (size_t)w * h;

	if (npix > fc->gray_cap) {
		uint8_t *grown = realloc(fc->gray_buf, npix);
		if (!grown) {
			sam3_log_error(
				"out of memory for frame PNG (%d x %d)",
				w, h);
			return 1;
		}
		fc->gray_buf = grown;
		fc->gray_cap = npix;
	}

	uint8_t *gray = fc->gray_buf;
	for (size_t i = 0; i < npix; i++)
		gray[i] = data[i] >= 0.0f ? 255 : 0;

	int ok = stbi_write_png(path, w, h, 1, gray, w);
	if (!ok) {
		sam3_log_error("failed to write frame PNG '%s'", path);
		return 1;
	}
	return 0;
}

static int track_frame_callback(const struct sam3_video_frame_result *result,
				void *user_data)
{
	struct track_frame_ctx *fc = user_data;
	char path_buf[512];
	int frame_idx;

	if (!result)
		return 0;

	SAM3_PROF_BEGIN(fc->profiler, "frame_output");

	frame_idx = result->frame_idx;
	sam3_log_info("frame %d: %d objects", frame_idx, result->n_objects);

	/*
	 * Write the first object's mask as a grayscale PNG. Multi-object
	 * overlay support can be added in a later pass.
	 */
	if (result->n_objects <= 0 || !result->objects ||
	    !result->objects[0].mask) {
		SAM3_PROF_END(fc->profiler, "frame_output");
		return 0;
	}

	const float *mask = result->objects[0].mask;
	int mask_w = result->objects[0].mask_w;
	int mask_h = result->objects[0].mask_h;

	int n = snprintf(path_buf, sizeof(path_buf),
			 "%s/frame_%05d.png", fc->output_dir, frame_idx);
	if (n < 0 || (size_t)n >= sizeof(path_buf)) {
		sam3_log_error("frame %d: output path too long", frame_idx);
		SAM3_PROF_END(fc->profiler, "frame_output");
		return 0;
	}

	if (write_frame_mask_png(fc, path_buf, mask, mask_w, mask_h)) {
		/* Error already logged. Continue propagation. */
		SAM3_PROF_END(fc->profiler, "frame_output");
		return 0;
	}

	SAM3_PROF_END(fc->profiler, "frame_output");
	return 0;
}

/* --- Run path --- */

/*
 * track_propagate_to_sam3_dir - Map our CLI propagate code onto the
 * public SAM3_PROPAGATE_* enum. Returns -1 if the caller asked for
 * "none" (no propagation at all).
 */
static int track_propagate_to_sam3_dir(int propagate)
{
	switch (propagate) {
	case TRACK_PROPAGATE_NONE:     return -1;
	case TRACK_PROPAGATE_FORWARD:  return SAM3_PROPAGATE_FORWARD;
	case TRACK_PROPAGATE_BACKWARD: return SAM3_PROPAGATE_BACKWARD;
	case TRACK_PROPAGATE_BOTH:     return SAM3_PROPAGATE_BOTH;
	default:                       return SAM3_PROPAGATE_BOTH;
	}
}

static const char *dir_name(enum sam3_propagate_dir dir)
{
	switch (dir) {
	case SAM3_PROPAGATE_FORWARD:  return "forward";
	case SAM3_PROPAGATE_BACKWARD: return "backward";
	case SAM3_PROPAGATE_BOTH:     return "both";
	}
	return "unknown";
}

static int cli_track_run(const struct track_args *a)
{
	sam3_ctx              *ctx     = NULL;
	sam3_video_session    *session = NULL;
	enum sam3_error        err;
	int                    ret     = SAM3_EXIT_INTERNAL;
	struct track_frame_ctx fc      = {
		.output_dir = a->output_dir,
		.gray_buf   = NULL,
		.gray_cap   = 0,
		.profiler   = NULL,
	};

	if (a->verbose)
		sam3_log_set_level(SAM3_LOG_DEBUG);

	ctx = sam3_init();
	if (!ctx) {
		sam3_log_error("failed to initialize sam3 context");
		return SAM3_EXIT_INTERNAL;
	}

	if (a->profile) {
		err = sam3_profile_enable(ctx);
		if (err != SAM3_OK) {
			sam3_log_warn("profiling not available: %s",
				      sam3_error_str(err));
		}
		fc.profiler = sam3_profile_get(ctx);
	}

	sam3_log_info("loading model: %s", a->model_path);
	err = sam3_load_model(ctx, a->model_path);
	if (err != SAM3_OK) {
		sam3_log_error("failed to load model '%s': %s",
			       a->model_path, sam3_error_str(err));
		ret = (int)sam3_error_to_exit(err);
		goto cleanup;
	}

	sam3_log_info("starting video session: %s", a->video_path);
	err = sam3_video_start(ctx, a->video_path, &session);
	if (err != SAM3_OK) {
		sam3_log_error("video start failed: %s",
			       sam3_error_str(err));
		ret = (int)sam3_error_to_exit(err);
		goto cleanup;
	}
	sam3_log_info("video: %d frames loaded",
		      sam3_video_frame_count(session));

	for (int i = 0; i < a->n_prompts; i++) {
		const struct track_prompt_entry *e = &a->prompts[i];
		struct sam3_video_frame_result prompt_result = {0};

		if (e->prompt.type == SAM3_PROMPT_POINT) {
			err = sam3_video_add_points(
				session, a->frame_idx, e->obj_id,
				&e->prompt.point, 1,
				&prompt_result);
		} else if (e->prompt.type == SAM3_PROMPT_BOX) {
			err = sam3_video_add_box(
				session, a->frame_idx, e->obj_id,
				&e->prompt.box, &prompt_result);
		} else {
			sam3_log_warn(
				"prompt %d: unsupported type for "
				"track mode, skipping", i);
			continue;
		}

		if (err != SAM3_OK) {
			sam3_log_error("prompt %d failed: %s",
				       i, sam3_error_str(err));
			sam3_video_frame_result_free(&prompt_result);
			ret = (int)sam3_error_to_exit(err);
			goto cleanup;
		}
		sam3_video_frame_result_free(&prompt_result);
	}

	int sam3_dir = track_propagate_to_sam3_dir(a->propagate);
	if (sam3_dir < 0) {
		sam3_log_info("propagation disabled (--propagate none)");
	} else {
		sam3_log_info("propagating masks (%s)",
			      dir_name((enum sam3_propagate_dir)sam3_dir));

		err = sam3_video_propagate(session, sam3_dir,
					   track_frame_callback, &fc);
		if (err != SAM3_OK) {
			sam3_log_error("propagation failed: %s",
				       sam3_error_str(err));
			ret = (int)sam3_error_to_exit(err);
			goto cleanup;
		}
	}

	ret = SAM3_EXIT_OK;

cleanup:
	if (a->profile)
		sam3_profile_report(ctx);
	if (session)
		sam3_video_end(session);
	sam3_free(ctx);
	free(fc.gray_buf);
	return ret;
}

int cli_track(int argc, char **argv)
{
	struct track_args args;
	int rc = cli_track_parse(argc, argv, &args);

	if (rc > 0) {
		print_usage(argv[0]);
		return SAM3_EXIT_OK;
	}
	if (rc < 0) {
		fprintf(stderr,
			"Run '%s --help' for usage.\n", argv[0]);
		return SAM3_EXIT_USAGE;
	}

	return cli_track_run(&args);
}
