/*
 * tests/test_cli_track.c - Unit tests for the track subcommand parser
 *
 * Exercises cli_track_parse() with synthetic argv arrays. No model,
 * no video, no inference -- these tests only validate argument parsing,
 * propagate-direction mapping, and the required-argument contract.
 *
 * Key types:  track_args
 * Depends on: cli_track.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <string.h>

#include "cli_track.h"
#include "test_helpers.h"

/*
 * ARGV - Build an argv-style (char **) from a compile-time list of
 * const char* literals. cli_track_parse treats argv as read-only, so
 * the const_cast via (char **) is safe for test purposes.
 */
#define ARGV(...) ((char **)(const char *[]){ __VA_ARGS__ })
#define ARGC(...) ((int)(sizeof((const char *[]){ __VA_ARGS__ }) / \
			  sizeof(const char *)))

static void test_track_parse_minimal(void)
{
	struct track_args a = {0};
	char **argv = ARGV("track",
			   "--model", "model.sam3",
			   "--video", "video.mp4",
			   "--output", "out/",
			   "--point", "100,200,1");
	int argc = ARGC("track",
			"--model", "model.sam3",
			"--video", "video.mp4",
			"--output", "out/",
			"--point", "100,200,1");

	ASSERT_EQ(cli_track_parse(argc, argv, &a), 0);
	ASSERT(strcmp(a.model_path, "model.sam3") == 0);
	ASSERT(strcmp(a.video_path, "video.mp4") == 0);
	ASSERT(strcmp(a.output_dir, "out/") == 0);
	ASSERT_EQ(a.n_prompts, 1);
	ASSERT_EQ(a.prompts[0].prompt.type, SAM3_PROMPT_POINT);
	ASSERT_EQ(a.prompts[0].obj_id, 0);
	ASSERT_EQ(a.frame_idx, 0);
	ASSERT_EQ(a.propagate, TRACK_PROPAGATE_BOTH);
}

static void test_track_parse_propagate_directions(void)
{
	const char *dirs[] = { "none", "forward", "backward", "both" };
	const int expect[] = {
		TRACK_PROPAGATE_NONE,
		TRACK_PROPAGATE_FORWARD,
		TRACK_PROPAGATE_BACKWARD,
		TRACK_PROPAGATE_BOTH,
	};

	for (int i = 0; i < 4; i++) {
		struct track_args a = {0};
		char **argv = ARGV("track",
				   "--model", "m.sam3",
				   "--video", "v.mp4",
				   "--output", "o/",
				   "--point", "1,2,1",
				   "--propagate", dirs[i]);
		int argc = ARGC("track",
				"--model", "m.sam3",
				"--video", "v.mp4",
				"--output", "o/",
				"--point", "1,2,1",
				"--propagate", dirs[i]);

		ASSERT_EQ(cli_track_parse(argc, argv, &a), 0);
		ASSERT_EQ(a.propagate, expect[i]);
	}
}

static void test_track_parse_missing_required(void)
{
	struct track_args a = {0};
	/* No --model */
	char **argv = ARGV("track",
			   "--video", "v.mp4",
			   "--output", "o/",
			   "--point", "1,2,1");
	int argc = ARGC("track",
			"--video", "v.mp4",
			"--output", "o/",
			"--point", "1,2,1");

	ASSERT_EQ(cli_track_parse(argc, argv, &a), -1);
}

static void test_track_parse_help(void)
{
	struct track_args a = {0};
	char **argv = ARGV("track", "--help");
	int argc = ARGC("track", "--help");

	ASSERT_EQ(cli_track_parse(argc, argv, &a), 1);
}

static void test_track_parse_bad_point(void)
{
	struct track_args a = {0};
	/* Missing label component */
	char **argv = ARGV("track",
			   "--model", "m.sam3",
			   "--video", "v.mp4",
			   "--output", "o/",
			   "--point", "100,200");
	int argc = ARGC("track",
			"--model", "m.sam3",
			"--video", "v.mp4",
			"--output", "o/",
			"--point", "100,200");

	ASSERT_EQ(cli_track_parse(argc, argv, &a), -1);
}

static void test_track_parse_bad_propagate(void)
{
	struct track_args a = {0};
	char **argv = ARGV("track",
			   "--model", "m.sam3",
			   "--video", "v.mp4",
			   "--output", "o/",
			   "--point", "1,2,1",
			   "--propagate", "sideways");
	int argc = ARGC("track",
			"--model", "m.sam3",
			"--video", "v.mp4",
			"--output", "o/",
			"--point", "1,2,1",
			"--propagate", "sideways");

	ASSERT_EQ(cli_track_parse(argc, argv, &a), -1);
}

static void test_track_parse_box_and_obj_id(void)
{
	struct track_args a = {0};
	char **argv = ARGV("track",
			   "--model", "m.sam3",
			   "--video", "v.mp4",
			   "--output", "o/",
			   "--obj-id", "2",
			   "--box", "10,20,30,40",
			   "--frame", "5");
	int argc = ARGC("track",
			"--model", "m.sam3",
			"--video", "v.mp4",
			"--output", "o/",
			"--obj-id", "2",
			"--box", "10,20,30,40",
			"--frame", "5");

	ASSERT_EQ(cli_track_parse(argc, argv, &a), 0);
	ASSERT_EQ(a.n_prompts, 1);
	ASSERT_EQ(a.prompts[0].prompt.type, SAM3_PROMPT_BOX);
	ASSERT_EQ(a.prompts[0].obj_id, 2);
	ASSERT_EQ(a.frame_idx, 5);
}

static void test_track_parse_alpha_valid(void)
{
	struct track_args a = {0};
	char **argv = ARGV("track",
			   "--model", "m.sam3", "--video", "v.mp4",
			   "--output", "out/",
			   "--point", "1,2,1",
			   "--alpha", "0.75");
	int argc = ARGC("track",
			"--model", "m.sam3", "--video", "v.mp4",
			"--output", "out/",
			"--point", "1,2,1",
			"--alpha", "0.75");

	ASSERT_EQ(cli_track_parse(argc, argv, &a), 0);
	ASSERT(a.alpha > 0.74f && a.alpha < 0.76f);
}

static void test_track_parse_alpha_out_of_range(void)
{
	struct track_args a = {0};
	char **argv = ARGV("track",
			   "--model", "m.sam3", "--video", "v.mp4",
			   "--output", "out/",
			   "--point", "1,2,1",
			   "--alpha", "2.0");
	int argc = ARGC("track",
			"--model", "m.sam3", "--video", "v.mp4",
			"--output", "out/",
			"--point", "1,2,1",
			"--alpha", "2.0");

	ASSERT_EQ(cli_track_parse(argc, argv, &a), -1);
}

static void test_track_parse_fps_valid(void)
{
	struct track_args a = {0};
	char **argv = ARGV("track",
			   "--model", "m.sam3", "--video", "v.mp4",
			   "--output", "out.mp4",
			   "--point", "1,2,1",
			   "--fps", "30");
	int argc = ARGC("track",
			"--model", "m.sam3", "--video", "v.mp4",
			"--output", "out.mp4",
			"--point", "1,2,1",
			"--fps", "30");

	ASSERT_EQ(cli_track_parse(argc, argv, &a), 0);
	ASSERT_EQ(a.fps, 30);
	ASSERT_EQ(a.output_mode, TRACK_OUTPUT_VIDEO);
}

static void test_track_output_mode_video_extensions(void)
{
	const char *paths[] = {
		"out.mp4", "OUT.MOV", "x.mkv", "seg.webm"
	};
	for (size_t i = 0; i < sizeof(paths) / sizeof(paths[0]); i++) {
		struct track_args a = {0};
		char **argv = ARGV("track",
				   "--model", "m.sam3", "--video", "v.mp4",
				   "--output", paths[i],
				   "--point", "1,2,1");
		int argc = ARGC("track",
				"--model", "m.sam3", "--video", "v.mp4",
				"--output", paths[i],
				"--point", "1,2,1");
		ASSERT_EQ(cli_track_parse(argc, argv, &a), 0);
		ASSERT_EQ(a.output_mode, TRACK_OUTPUT_VIDEO);
	}
}

static void test_track_output_mode_dir_default(void)
{
	const char *paths[] = { "out/", "results", "./x.png" };
	for (size_t i = 0; i < sizeof(paths) / sizeof(paths[0]); i++) {
		struct track_args a = {0};
		char **argv = ARGV("track",
				   "--model", "m.sam3", "--video", "v.mp4",
				   "--output", paths[i],
				   "--point", "1,2,1");
		int argc = ARGC("track",
				"--model", "m.sam3", "--video", "v.mp4",
				"--output", paths[i],
				"--point", "1,2,1");
		ASSERT_EQ(cli_track_parse(argc, argv, &a), 0);
		ASSERT_EQ(a.output_mode, TRACK_OUTPUT_DIR);
	}
}

static void test_track_default_alpha(void)
{
	struct track_args a = {0};
	char **argv = ARGV("track",
			   "--model", "m.sam3", "--video", "v.mp4",
			   "--output", "out/",
			   "--point", "1,2,1");
	int argc = ARGC("track",
			"--model", "m.sam3", "--video", "v.mp4",
			"--output", "out/",
			"--point", "1,2,1");
	ASSERT_EQ(cli_track_parse(argc, argv, &a), 0);
	ASSERT(a.alpha > 0.49f && a.alpha < 0.51f);
}

int main(void)
{
	test_track_parse_minimal();
	test_track_parse_propagate_directions();
	test_track_parse_missing_required();
	test_track_parse_help();
	test_track_parse_bad_point();
	test_track_parse_bad_propagate();
	test_track_parse_box_and_obj_id();
	test_track_parse_alpha_valid();
	test_track_parse_alpha_out_of_range();
	test_track_parse_fps_valid();
	test_track_output_mode_video_extensions();
	test_track_output_mode_dir_default();
	test_track_default_alpha();

	TEST_REPORT();
}
