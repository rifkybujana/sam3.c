/*
 * src/util/time.h - High-resolution nanosecond clock
 *
 * Provides a platform-independent nanosecond timestamp function
 * for profiling. Uses mach_absolute_time() on macOS and
 * clock_gettime(CLOCK_MONOTONIC) on Linux.
 *
 * Key types:  (none — returns uint64_t)
 * Depends on: <stdint.h>
 * Used by:    util/profile.h
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_UTIL_TIME_H
#define SAM3_UTIL_TIME_H

#include <stdint.h>

/* Return current monotonic time in nanoseconds. */
uint64_t sam3_time_ns(void);

#endif /* SAM3_UTIL_TIME_H */
