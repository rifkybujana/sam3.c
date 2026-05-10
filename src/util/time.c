/*
 * src/util/time.c - Nanosecond clock implementation
 *
 * Platform-specific high-resolution clock. On macOS, uses
 * mach_absolute_time() which returns ticks that must be converted
 * to nanoseconds via mach_timebase_info. On Windows, uses
 * QueryPerformanceCounter. On Linux, uses clock_gettime(CLOCK_MONOTONIC).
 *
 * Key types:  (none)
 * Depends on: time.h
 * Used by:    util/profile.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "time.h"

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

uint64_t sam3_time_ns(void)
{
	static LARGE_INTEGER freq;
	LARGE_INTEGER counter;

	if (freq.QuadPart == 0)
		QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&counter);
	return (uint64_t)((counter.QuadPart * 1000000000ULL) /
			  (uint64_t)freq.QuadPart);
}

#elif defined(__APPLE__)

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
