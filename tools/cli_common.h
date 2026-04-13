/*
 * tools/cli_common.h - Shared definitions for SAM3 CLI tools
 *
 * Provides standardized exit codes, error-to-exit mapping, and output
 * helpers used by all CLI tool binaries. Exit codes follow a simple
 * scheme: 0 = success, 1 = usage error, 2 = I/O, 3 = model, 4 = runtime,
 * 5 = internal.
 *
 * Key types:  sam3_exit
 * Depends on: sam3/sam3_types.h, <stdio.h>, <unistd.h>
 * Used by:    tools/sam3_main.c, tools/sam3_convert.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_CLI_COMMON_H
#define SAM3_CLI_COMMON_H

#include "sam3/sam3_types.h"

#include <stdio.h>
#include <unistd.h>

/* Process exit codes for CLI tools. */
enum sam3_exit {
	SAM3_EXIT_OK       = 0,
	SAM3_EXIT_USAGE    = 1,
	SAM3_EXIT_IO       = 2,
	SAM3_EXIT_MODEL    = 3,
	SAM3_EXIT_RUNTIME  = 4,
	SAM3_EXIT_INTERNAL = 5,
};

/*
 * sam3_error_to_exit - Map a sam3_error code to a CLI exit code.
 *
 * @err: Error code from any sam3 API function.
 *
 * Returns the appropriate sam3_exit value. Unknown error codes map to
 * SAM3_EXIT_INTERNAL as a safe default.
 */
static inline enum sam3_exit sam3_error_to_exit(enum sam3_error err)
{
	switch (err) {
	case SAM3_OK:       return SAM3_EXIT_OK;
	case SAM3_EINVAL:   return SAM3_EXIT_USAGE;
	case SAM3_ENOMEM:   return SAM3_EXIT_INTERNAL;
	case SAM3_EIO:      return SAM3_EXIT_IO;
	case SAM3_EBACKEND: return SAM3_EXIT_INTERNAL;
	case SAM3_EMODEL:   return SAM3_EXIT_MODEL;
	case SAM3_EDTYPE:   return SAM3_EXIT_RUNTIME;
	default:            return SAM3_EXIT_INTERNAL;
	}
}

/* Print a progress message to stderr (never pollutes stdout). */
#define cli_progress(...) fprintf(stderr, __VA_ARGS__)

/* Return true (non-zero) when stdout is piped / redirected. */
static inline int cli_is_pipe(void)
{
	return !isatty(STDOUT_FILENO);
}

#endif /* SAM3_CLI_COMMON_H */
