/*
 * src/util/time.c - Nanosecond clock implementation
 *
 * Platform-specific high-resolution clock. On macOS, uses
 * mach_absolute_time() which returns ticks that must be converted
 * to nanoseconds via mach_timebase_info. On Linux, uses
 * clock_gettime(CLOCK_MONOTONIC).
 *
 * Key types:  (none)
 * Depends on: time.h
 * Used by:    util/profile.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "time.h"

#ifdef __APPLE__

#include <mach/mach_time.h>

uint64_t sam3_time_ns(void)
{
	static mach_timebase_info_data_t info;
	if (info.denom == 0)
		mach_timebase_info(&info);

	uint64_t ticks = mach_absolute_time();
	return ticks * info.numer / info.denom;
}

#else /* Linux / POSIX */

#include <time.h>

uint64_t sam3_time_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

#endif
