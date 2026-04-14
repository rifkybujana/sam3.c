/*
 * tools/cli_segment.c - SAM3 segment subcommand
 *
 * Runs segmentation inference with point/box/text prompts. Supports
 * stdin image input (-i -), stdout mask output (-o -), JSON metadata
 * (--json), and quiet mode (-q).
 *
 * Key types:  segment_args
 * Depends on: cli_common.h, sam3/sam3.h, util/image.h
 * Used by:    tools/sam3_cli.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include "cli_common.h"
#include "cli_segment.h"

int cli_segment(int argc, char **argv)
{
	(void)argc; (void)argv;
	fprintf(stderr, "error: segment not yet implemented\n");
	return SAM3_EXIT_USAGE;
}
