/*
 * src/util/log.c - Logging implementation
 *
 * Writes formatted log messages to stderr with level prefix and
 * source location. Thread-safe via fprintf atomicity guarantee.
 *
 * Key types:  sam3_log_level
 * Depends on: log.h
 * Used by:    all modules
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdarg.h>

#include "log.h"

static enum sam3_log_level g_log_level = SAM3_LOG_INFO;

static const char *level_str(enum sam3_log_level level)
{
	switch (level) {
	case SAM3_LOG_DEBUG: return "DEBUG";
	case SAM3_LOG_INFO:  return "INFO";
	case SAM3_LOG_WARN:  return "WARN";
	case SAM3_LOG_ERROR: return "ERROR";
	}
	return "?";
}

void sam3_log_set_level(enum sam3_log_level level)
{
	g_log_level = level;
}

void sam3_log_write(enum sam3_log_level level, const char *file, int line,
		    const char *fmt, ...)
{
	if (level < g_log_level)
		return;

	va_list ap;
	va_start(ap, fmt);
	fprintf(stderr, "[%s] %s:%d: ", level_str(level), file, line);
	vfprintf(stderr, fmt, ap);
	fprintf(stderr, "\n");
	va_end(ap);
}
