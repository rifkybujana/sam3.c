/*
 * tools/cli_track.h - Track subcommand declaration and test surface
 *
 * Public entry for the `sam3 track` subcommand, plus the internal
 * argument struct and parser exposed for unit testing. The parser is
 * surfaced here (rather than hidden static) so test_cli_track.c can
 * exercise argv handling without any model or video on disk.
 *
 * Key types:  track_args
 * Depends on: sam3/sam3_types.h
 * Used by:    tools/sam3_cli.c, tools/cli_track.c, tests/test_cli_track.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_CLI_TRACK_H
#define SAM3_CLI_TRACK_H

#include "sam3/sam3_types.h"

/*
 * Maximum number of point/box prompts the track subcommand can accept
 * from the command line. Matches the segment subcommand limit.
 */
#define SAM3_CLI_TRACK_MAX_PROMPTS 64

/*
 * Propagation direction values used by `struct track_args`. These are
 * CLI-layer constants (not the public SAM3_PROPAGATE_* enum); cli_track
 * translates them before calling sam3_video_propagate.
 */
enum track_propagate {
	TRACK_PROPAGATE_NONE     = 0,
	TRACK_PROPAGATE_FORWARD  = 1,
	TRACK_PROPAGATE_BACKWARD = 2,
	TRACK_PROPAGATE_BOTH     = 3,
};

/*
 * A single CLI prompt entry. Each prompt carries the object id the
 * caller associated with it via `--obj-id`. Points and boxes share the
 * array so that order of `--point`/`--box` on the command line is
 * preserved when they are replayed against the video session.
 */
struct track_prompt_entry {
	struct sam3_prompt prompt; /* SAM3_PROMPT_POINT or SAM3_PROMPT_BOX */
	int obj_id;                /* Object identifier (>= 0) */
};

/*
 * track_args - Parsed command-line state for `sam3 track`.
 *
 * Internal, test surface. Not part of the public sam3 ABI. Always use
 * cli_track() from application code; this struct is exposed only so
 * that tests/test_cli_track.c can unit-test cli_track_parse().
 */
struct track_args {
	const char               *model_path;
	const char               *video_path;
	const char               *output_dir;
	struct track_prompt_entry prompts[SAM3_CLI_TRACK_MAX_PROMPTS];
	int                       n_prompts;
	int                       frame_idx;
	int                       propagate;  /* enum track_propagate */
	int                       verbose;
	int                       profile;    /* 1 if --profile was passed */
};

/*
 * cli_track_parse - Parse argv into a track_args struct.
 *
 * @argc: Argument count (as received from main).
 * @argv: Argument vector (argv[0] is the subcommand name).
 * @out:  Receives parsed arguments on success.
 *
 * Returns:
 *   0 on success,
 *   1 if the user asked for --help/-h (caller should exit 0),
 *  -1 on parse error (caller should exit non-zero).
 */
int cli_track_parse(int argc, char **argv, struct track_args *out);

/*
 * cli_track - Entry point for the `sam3 track` subcommand.
 *
 * Returns a sam3_exit code.
 */
int cli_track(int argc, char **argv);

#endif /* SAM3_CLI_TRACK_H */
