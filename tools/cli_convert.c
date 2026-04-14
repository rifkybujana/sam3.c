/*
 * tools/cli_convert.c - SAM3 convert subcommand
 *
 * Converts model weights from SafeTensors to .sam3 format. Refactored
 * from the standalone sam3_convert tool.
 *
 * Key types:  convert_args
 * Depends on: cli_common.h, core/weight.h
 * Used by:    tools/sam3_cli.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include "cli_common.h"
#include "cli_convert.h"

int cli_convert(int argc, char **argv)
{
	(void)argc; (void)argv;
	fprintf(stderr, "error: convert not yet implemented\n");
	return SAM3_EXIT_USAGE;
}
