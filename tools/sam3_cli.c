/*
 * tools/sam3_cli.c - SAM3 CLI entry point and subcommand dispatcher
 *
 * Routes to segment, convert, or info subcommands based on argv[1].
 * Handles --version and --help at the top level. Falls back to
 * segment if argv[1] starts with '-' (backward compatibility).
 *
 * Key types:  (none)
 * Depends on: cli_common.h, cli_segment.h, cli_convert.h, cli_info.h
 * Used by:    end users
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>

#include "sam3/sam3.h"
#include "cli_common.h"
#include "cli_segment.h"
#include "cli_convert.h"
#include "cli_info.h"
#ifdef SAM3_HAS_BENCH
#include "cli_bench.h"
#endif

static void print_usage(const char *prog)
{
	fprintf(stderr, "sam3 v%s\n\n", sam3_version());
	fprintf(stderr, "Usage: %s <command> [options]\n\n", prog);
	fprintf(stderr, "Commands:\n");
	fprintf(stderr, "  segment    Run segmentation inference\n");
	fprintf(stderr, "  convert    Convert model weights to .sam3\n");
	fprintf(stderr, "  info       Print model file metadata\n");
#ifdef SAM3_HAS_BENCH
	fprintf(stderr, "  bench      Run performance benchmarks\n");
#endif
	fprintf(stderr, "\nOptions:\n");
	fprintf(stderr, "  --version  Print version and exit\n");
	fprintf(stderr, "  --help     Show this help\n");
	fprintf(stderr, "\nRun '%s <command> --help' for "
		"command-specific help.\n", prog);
}

int main(int argc, char **argv)
{
	if (argc < 2) {
		print_usage(argv[0]);
		return SAM3_EXIT_USAGE;
	}

	const char *cmd = argv[1];

	/* Top-level flags */
	if (strcmp(cmd, "--version") == 0) {
		printf("sam3 %s\n", sam3_version());
		return SAM3_EXIT_OK;
	}
	if (strcmp(cmd, "--help") == 0 || strcmp(cmd, "-h") == 0) {
		print_usage(argv[0]);
		return SAM3_EXIT_OK;
	}

	/* Subcommand dispatch */
	if (strcmp(cmd, "segment") == 0)
		return cli_segment(argc - 1, argv + 1);
	if (strcmp(cmd, "convert") == 0)
		return cli_convert(argc - 1, argv + 1);
	if (strcmp(cmd, "info") == 0)
		return cli_info(argc - 1, argv + 1);
#ifdef SAM3_HAS_BENCH
	if (strcmp(cmd, "bench") == 0)
		return cli_bench(argc - 1, argv + 1);
#endif

	/*
	 * Backward compat: if first arg starts with '-', assume
	 * segment (old usage: sam3 -m model -i image ...)
	 */
	if (cmd[0] == '-')
		return cli_segment(argc, argv);

	fprintf(stderr, "error: unknown command '%s'\n", cmd);
	fprintf(stderr, "Run '%s --help' for usage.\n", argv[0]);
	return SAM3_EXIT_USAGE;
}
