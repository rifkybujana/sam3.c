/*
 * tools/cli_info.c - SAM3 info subcommand
 *
 * Opens a .sam3 weight file and prints metadata (model config,
 * tensor count, file size). Supports --json output.
 *
 * Key types:  (uses sam3_weight_header)
 * Depends on: cli_common.h, core/weight.h
 * Used by:    tools/sam3_cli.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include "cli_common.h"
#include "cli_info.h"

int cli_info(int argc, char **argv)
{
	(void)argc; (void)argv;
	fprintf(stderr, "error: info not yet implemented\n");
	return SAM3_EXIT_USAGE;
}
