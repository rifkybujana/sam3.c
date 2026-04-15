/*
 * src/util/log.h - Logging subsystem
 *
 * Simple leveled logging (debug, info, warn, error) to stderr.
 * Log level is set at runtime. All log output includes the source
 * file and line number for easy debugging.
 *
 * Key types:  sam3_log_level
 * Depends on: <stdio.h>
 * Used by:    all modules
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_UTIL_LOG_H
#define SAM3_UTIL_LOG_H

#include "sam3/sam3_types.h"

/* Set the minimum log level. Messages below this level are suppressed. */
void sam3_log_set_level(enum sam3_log_level level);

/* Internal log function — use the macros below instead. */
void sam3_log_write(enum sam3_log_level level, const char *file, int line,
		    const char *fmt, ...);

#define sam3_log_debug(...) \
	sam3_log_write(SAM3_LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define sam3_log_info(...) \
	sam3_log_write(SAM3_LOG_INFO, __FILE__, __LINE__, __VA_ARGS__)
#define sam3_log_warn(...) \
	sam3_log_write(SAM3_LOG_WARN, __FILE__, __LINE__, __VA_ARGS__)
#define sam3_log_error(...) \
	sam3_log_write(SAM3_LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)

#endif /* SAM3_UTIL_LOG_H */
