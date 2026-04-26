/*
 * tools/cli_track_stub.c - SAM3_NO_VIDEO replacement for cli_track.c
 *
 * Provides cli_track() and cli_track_parse() stubs that print a
 * helpful error and return non-zero. Compiled instead of cli_track.c
 * when the video subsystem is disabled (-DSAM3_VIDEO=OFF), since
 * cli_track.c link-references the FFmpeg-backed RGB iterator and
 * video encoder that aren't built in that configuration.
 *
 * Key types:  none
 * Depends on: cli_track.h
 * Used by:    sam3_cli (build-time only)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>

#include "cli_track.h"
#include "cli_common.h"

int cli_track_parse(int argc, char **argv, struct track_args *out)
{
	(void)argc; (void)argv;
	if (out)
		memset(out, 0, sizeof(*out));
	return SAM3_EXIT_USAGE;
}

int cli_track(int argc, char **argv)
{
	(void)argc; (void)argv;
	fprintf(stderr,
		"sam3: 'track' subcommand is unavailable in this build "
		"(compiled with -DSAM3_VIDEO=OFF).\n");
	return SAM3_EXIT_USAGE;
}
