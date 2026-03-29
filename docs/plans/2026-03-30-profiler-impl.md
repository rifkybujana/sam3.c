# SAM3 Profiler Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a comprehensive profiler to sam3 with stage timing, per-op timing, memory tracking, and perf-style text reporting — all gated by compile-time and runtime flags for zero overhead in production.

**Architecture:** New `src/util/profile.h/.c` module. Profiler context embedded in `sam3_ctx`. Macros compile to `((void)0)` without `SAM3_HAS_PROFILE`. When compiled in, runtime toggle via `sam3_profile_enable()`. Platform-specific nanosecond clock (`mach_absolute_time` on macOS, `clock_gettime` on Linux).

**Tech Stack:** C11, CMake, macOS `mach/mach_time.h`, POSIX `time.h`

---

### Task 1: Platform Clock Utility

**Files:**
- Create: `src/util/time.h`
- Create: `src/util/time.c`
- Test: `tests/test_time.c`

**Step 1: Write the failing test**

Create `tests/test_time.c`:

```c
/*
 * tests/test_time.c - Unit tests for nanosecond clock utility
 *
 * Tests that sam3_time_ns returns monotonically increasing values
 * and that elapsed time is non-negative.
 *
 * Key types:  (none)
 * Depends on: util/time.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "util/time.h"

static void test_time_ns_returns_nonzero(void)
{
	uint64_t t = sam3_time_ns();
	ASSERT(t > 0);
}

static void test_time_ns_monotonic(void)
{
	uint64_t t1 = sam3_time_ns();
	/* Burn some CPU cycles */
	volatile int x = 0;
	for (int i = 0; i < 10000; i++)
		x += i;
	(void)x;
	uint64_t t2 = sam3_time_ns();
	ASSERT(t2 >= t1);
}

static void test_time_elapsed_positive(void)
{
	uint64_t start = sam3_time_ns();
	volatile int x = 0;
	for (int i = 0; i < 100000; i++)
		x += i;
	(void)x;
	uint64_t end = sam3_time_ns();
	ASSERT(end - start > 0);
}

int main(void)
{
	test_time_ns_returns_nonzero();
	test_time_ns_monotonic();
	test_time_elapsed_positive();

	TEST_REPORT();
}
```

**Step 2: Run test to verify it fails**

Run: `cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DSAM3_METAL=OFF && make -j$(sysctl -n hw.ncpu) 2>&1 && ctest -R test_time --output-on-failure`
Expected: FAIL — `util/time.h` not found

**Step 3: Write minimal implementation**

Create `src/util/time.h`:

```c
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
```

Create `src/util/time.c`:

```c
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
```

**Step 4: Run test to verify it passes**

Run: `cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DSAM3_METAL=OFF && make -j$(sysctl -n hw.ncpu) 2>&1 && ctest -R test_time --output-on-failure`
Expected: PASS — 3 tests, 0 failures

**Step 5: Commit**

```bash
git add src/util/time.h src/util/time.c tests/test_time.c
git commit -m "util/time: add platform-independent nanosecond clock"
```

---

### Task 2: Profiler Core Types and Lifecycle

**Files:**
- Create: `src/util/profile.h`
- Create: `src/util/profile.c`
- Test: `tests/test_profile.c`

**Step 1: Write the failing test**

Create `tests/test_profile.c`:

```c
/*
 * tests/test_profile.c - Unit tests for the profiler
 *
 * Tests profiler lifecycle, stage timing, op timing, memory tracking,
 * and report output. Built only when SAM3_PROFILE is enabled.
 *
 * Key types:  sam3_profiler
 * Depends on: util/profile.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "util/profile.h"

static void test_profiler_create_free(void)
{
	struct sam3_profiler *p = sam3_profiler_create();
	ASSERT(p != NULL);
	sam3_profiler_free(p);
}

static void test_profiler_enable_disable(void)
{
	struct sam3_profiler *p = sam3_profiler_create();
	ASSERT_EQ(sam3_profiler_is_enabled(p), 0);
	sam3_profiler_enable(p);
	ASSERT_EQ(sam3_profiler_is_enabled(p), 1);
	sam3_profiler_disable(p);
	ASSERT_EQ(sam3_profiler_is_enabled(p), 0);
	sam3_profiler_free(p);
}

static void test_profiler_reset(void)
{
	struct sam3_profiler *p = sam3_profiler_create();
	sam3_profiler_enable(p);
	sam3_prof_stage_begin(p, "test_stage");
	sam3_prof_stage_end(p, "test_stage");
	sam3_profiler_reset(p);
	/* After reset, should still be enabled but no data */
	ASSERT_EQ(sam3_profiler_is_enabled(p), 1);
	sam3_profiler_free(p);
}

int main(void)
{
	test_profiler_create_free();
	test_profiler_enable_disable();
	test_profiler_reset();

	TEST_REPORT();
}
```

**Step 2: Run test to verify it fails**

Run: `cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DSAM3_METAL=OFF -DSAM3_PROFILE=ON && make -j$(sysctl -n hw.ncpu) 2>&1 && ctest -R test_profile --output-on-failure`
Expected: FAIL — `util/profile.h` not found

**Step 3: Write minimal implementation**

Create `src/util/profile.h`:

```c
/*
 * src/util/profile.h - Profiler for sam3 inference
 *
 * Instrumentation-based profiler with compile-time and runtime gating.
 * Collects stage timing (pipeline-level), per-op timing (graph node-level),
 * and memory statistics. Produces perf-style text reports to stderr.
 *
 * When SAM3_HAS_PROFILE is not defined, all SAM3_PROF_* macros expand
 * to ((void)0) for zero overhead.
 *
 * Key types:  sam3_profiler, sam3_prof_stage, sam3_prof_op_stats, sam3_prof_mem
 * Depends on: util/time.h, core/graph.h
 * Used by:    sam3.c, core/alloc.c, backend/ files
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_UTIL_PROFILE_H
#define SAM3_UTIL_PROFILE_H

#include <stdint.h>
#include <stddef.h>

#include "core/graph.h"

#define SAM3_PROF_MAX_STAGES 32
#define SAM3_OP_COUNT        (SAM3_OP_TRANSPOSE + 1)

/* Per-stage timing record. */
struct sam3_prof_stage {
	const char *name;
	uint64_t    start_ns;
	uint64_t    total_ns;
	int         calls;
};

/* Per-op aggregate timing. */
struct sam3_prof_op_stats {
	uint64_t total_ns;
	uint64_t start_ns;    /* Scratch for current measurement */
	int      calls;
};

/* Memory usage statistics. */
struct sam3_prof_mem {
	size_t peak_bytes;
	size_t current_bytes;
	int    alloc_count;
	int    arena_count;
};

/* Profiler instance. */
struct sam3_profiler {
	int                       enabled;
	struct sam3_prof_stage     stages[SAM3_PROF_MAX_STAGES];
	int                       n_stages;
	struct sam3_prof_op_stats  op_stats[SAM3_OP_COUNT];
	struct sam3_prof_mem       mem;
};

/* Lifecycle */
struct sam3_profiler *sam3_profiler_create(void);
void sam3_profiler_free(struct sam3_profiler *p);
void sam3_profiler_enable(struct sam3_profiler *p);
void sam3_profiler_disable(struct sam3_profiler *p);
int  sam3_profiler_is_enabled(const struct sam3_profiler *p);
void sam3_profiler_reset(struct sam3_profiler *p);

/* Stage timing */
void sam3_prof_stage_begin(struct sam3_profiler *p, const char *name);
void sam3_prof_stage_end(struct sam3_profiler *p, const char *name);

/* Op timing */
void sam3_prof_op_begin(struct sam3_profiler *p, enum sam3_op op);
void sam3_prof_op_end(struct sam3_profiler *p, enum sam3_op op);

/* Memory tracking */
void sam3_prof_mem_alloc(struct sam3_profiler *p, size_t nbytes);
void sam3_prof_mem_arena(struct sam3_profiler *p);
void sam3_prof_mem_arena_reset(struct sam3_profiler *p, size_t freed_bytes);

/* Report */
void sam3_profiler_report(const struct sam3_profiler *p);

/* Convenience macros — compile to nothing without SAM3_HAS_PROFILE */
#ifdef SAM3_HAS_PROFILE

#define SAM3_PROF_BEGIN(prof, name) \
	do { if (prof) sam3_prof_stage_begin((prof), (name)); } while (0)
#define SAM3_PROF_END(prof, name) \
	do { if (prof) sam3_prof_stage_end((prof), (name)); } while (0)
#define SAM3_PROF_OP_BEGIN(prof, op) \
	do { if (prof) sam3_prof_op_begin((prof), (op)); } while (0)
#define SAM3_PROF_OP_END(prof, op) \
	do { if (prof) sam3_prof_op_end((prof), (op)); } while (0)
#define SAM3_PROF_MEM(prof, nbytes) \
	do { if (prof) sam3_prof_mem_alloc((prof), (nbytes)); } while (0)
#define SAM3_PROF_MEM_ARENA(prof) \
	do { if (prof) sam3_prof_mem_arena((prof)); } while (0)
#define SAM3_PROF_MEM_ARENA_RESET(prof, freed) \
	do { if (prof) sam3_prof_mem_arena_reset((prof), (freed)); } while (0)

#else /* !SAM3_HAS_PROFILE */

#define SAM3_PROF_BEGIN(prof, name)           ((void)0)
#define SAM3_PROF_END(prof, name)             ((void)0)
#define SAM3_PROF_OP_BEGIN(prof, op)          ((void)0)
#define SAM3_PROF_OP_END(prof, op)            ((void)0)
#define SAM3_PROF_MEM(prof, nbytes)           ((void)0)
#define SAM3_PROF_MEM_ARENA(prof)             ((void)0)
#define SAM3_PROF_MEM_ARENA_RESET(prof, freed) ((void)0)

#endif /* SAM3_HAS_PROFILE */

#endif /* SAM3_UTIL_PROFILE_H */
```

Create `src/util/profile.c`:

```c
/*
 * src/util/profile.c - Profiler implementation
 *
 * Implements profiler lifecycle, stage timing, op timing, memory
 * tracking, and perf-style text report. All timing uses sam3_time_ns()
 * for nanosecond resolution.
 *
 * Key types:  sam3_profiler
 * Depends on: profile.h, util/time.h
 * Used by:    sam3.c, core/alloc.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "profile.h"
#include "time.h"

/* Op name lookup table — must match enum sam3_op order in graph.h */
static const char *op_names[] = {
	"NONE",
	"MATMUL",
	"ADD",
	"MUL",
	"SOFTMAX",
	"RELU",
	"GELU",
	"LAYERNORM",
	"CONV2D",
	"RESHAPE",
	"TRANSPOSE",
};

struct sam3_profiler *sam3_profiler_create(void)
{
	struct sam3_profiler *p = calloc(1, sizeof(*p));
	return p;
}

void sam3_profiler_free(struct sam3_profiler *p)
{
	free(p);
}

void sam3_profiler_enable(struct sam3_profiler *p)
{
	if (p)
		p->enabled = 1;
}

void sam3_profiler_disable(struct sam3_profiler *p)
{
	if (p)
		p->enabled = 0;
}

int sam3_profiler_is_enabled(const struct sam3_profiler *p)
{
	return p ? p->enabled : 0;
}

void sam3_profiler_reset(struct sam3_profiler *p)
{
	if (!p)
		return;
	int was_enabled = p->enabled;
	memset(p->stages, 0, sizeof(p->stages));
	p->n_stages = 0;
	memset(p->op_stats, 0, sizeof(p->op_stats));
	memset(&p->mem, 0, sizeof(p->mem));
	p->enabled = was_enabled;
}

/*
 * find_or_create_stage - Find an existing stage by name, or create a new one.
 *
 * Returns NULL if the stage table is full.
 */
static struct sam3_prof_stage *find_or_create_stage(struct sam3_profiler *p,
						    const char *name)
{
	for (int i = 0; i < p->n_stages; i++) {
		if (p->stages[i].name == name)
			return &p->stages[i];
	}

	if (p->n_stages >= SAM3_PROF_MAX_STAGES)
		return NULL;

	struct sam3_prof_stage *s = &p->stages[p->n_stages++];
	s->name = name;
	s->total_ns = 0;
	s->calls = 0;
	return s;
}

void sam3_prof_stage_begin(struct sam3_profiler *p, const char *name)
{
	if (!p || !p->enabled)
		return;

	struct sam3_prof_stage *s = find_or_create_stage(p, name);
	if (s)
		s->start_ns = sam3_time_ns();
}

void sam3_prof_stage_end(struct sam3_profiler *p, const char *name)
{
	if (!p || !p->enabled)
		return;

	uint64_t end = sam3_time_ns();
	struct sam3_prof_stage *s = find_or_create_stage(p, name);
	if (s && s->start_ns > 0) {
		s->total_ns += end - s->start_ns;
		s->calls++;
		s->start_ns = 0;
	}
}

void sam3_prof_op_begin(struct sam3_profiler *p, enum sam3_op op)
{
	if (!p || !p->enabled)
		return;
	if (op < 0 || op >= SAM3_OP_COUNT)
		return;

	p->op_stats[op].start_ns = sam3_time_ns();
}

void sam3_prof_op_end(struct sam3_profiler *p, enum sam3_op op)
{
	if (!p || !p->enabled)
		return;
	if (op < 0 || op >= SAM3_OP_COUNT)
		return;

	uint64_t end = sam3_time_ns();
	struct sam3_prof_op_stats *os = &p->op_stats[op];
	if (os->start_ns > 0) {
		os->total_ns += end - os->start_ns;
		os->calls++;
		os->start_ns = 0;
	}
}

void sam3_prof_mem_alloc(struct sam3_profiler *p, size_t nbytes)
{
	if (!p || !p->enabled)
		return;

	p->mem.current_bytes += nbytes;
	p->mem.alloc_count++;
	if (p->mem.current_bytes > p->mem.peak_bytes)
		p->mem.peak_bytes = p->mem.current_bytes;
}

void sam3_prof_mem_arena(struct sam3_profiler *p)
{
	if (!p || !p->enabled)
		return;

	p->mem.arena_count++;
}

void sam3_prof_mem_arena_reset(struct sam3_profiler *p, size_t freed_bytes)
{
	if (!p || !p->enabled)
		return;

	if (freed_bytes > p->mem.current_bytes)
		p->mem.current_bytes = 0;
	else
		p->mem.current_bytes -= freed_bytes;
}

/* Compare op stats by total_ns descending (for qsort). */
struct op_sort_entry {
	int      op_index;
	uint64_t total_ns;
	int      calls;
};

static int cmp_op_desc(const void *a, const void *b)
{
	const struct op_sort_entry *ea = a;
	const struct op_sort_entry *eb = b;
	if (ea->total_ns > eb->total_ns) return -1;
	if (ea->total_ns < eb->total_ns) return 1;
	return 0;
}

void sam3_profiler_report(const struct sam3_profiler *p)
{
	if (!p)
		return;

	/* Compute total stage time */
	uint64_t total_stage_ns = 0;
	for (int i = 0; i < p->n_stages; i++)
		total_stage_ns += p->stages[i].total_ns;

	/* Compute total op time */
	uint64_t total_op_ns = 0;
	for (int i = 0; i < SAM3_OP_COUNT; i++)
		total_op_ns += p->op_stats[i].total_ns;

	fprintf(stderr,
		"\n"
		"======================================================\n"
		" sam3 profile report\n"
		"======================================================\n");

	/* Stage table */
	if (p->n_stages > 0) {
		fprintf(stderr,
			" %-20s %5s %11s %9s %5s\n",
			"Stage", "Calls", "Total(ms)", "Avg(ms)", "%");
		fprintf(stderr,
			"------------------------------------------------------\n");

		for (int i = 0; i < p->n_stages; i++) {
			const struct sam3_prof_stage *s = &p->stages[i];
			if (s->calls == 0)
				continue;

			double total_ms = (double)s->total_ns / 1e6;
			double avg_ms = total_ms / s->calls;
			double pct = total_stage_ns > 0
				? 100.0 * s->total_ns / total_stage_ns
				: 0.0;

			fprintf(stderr, " %-20s %5d %11.2f %9.2f %5.1f\n",
				s->name, s->calls, total_ms, avg_ms, pct);
		}
		fprintf(stderr,
			"------------------------------------------------------\n\n");
	}

	/* Op breakdown — sort by total time descending */
	struct op_sort_entry entries[SAM3_OP_COUNT];
	int n_active = 0;
	for (int i = 0; i < SAM3_OP_COUNT; i++) {
		if (p->op_stats[i].calls > 0) {
			entries[n_active].op_index = i;
			entries[n_active].total_ns = p->op_stats[i].total_ns;
			entries[n_active].calls = p->op_stats[i].calls;
			n_active++;
		}
	}

	if (n_active > 0) {
		qsort(entries, n_active, sizeof(entries[0]), cmp_op_desc);

		fprintf(stderr,
			" %-20s %5s %11s %9s %5s\n",
			"Op Breakdown", "Calls", "Total(ms)", "Avg(ms)", "%");
		fprintf(stderr,
			"------------------------------------------------------\n");

		for (int i = 0; i < n_active; i++) {
			int idx = entries[i].op_index;
			const char *name = (idx < (int)(sizeof(op_names) / sizeof(op_names[0])))
				? op_names[idx] : "UNKNOWN";
			double total_ms = (double)entries[i].total_ns / 1e6;
			double avg_ms = total_ms / entries[i].calls;
			double pct = total_op_ns > 0
				? 100.0 * entries[i].total_ns / total_op_ns
				: 0.0;

			fprintf(stderr, " %-20s %5d %11.2f %9.2f %5.1f\n",
				name, entries[i].calls, total_ms, avg_ms, pct);
		}
		fprintf(stderr,
			"------------------------------------------------------\n\n");
	}

	/* Memory stats */
	if (p->mem.alloc_count > 0 || p->mem.arena_count > 0) {
		fprintf(stderr,
			" %-20s %5s %11s %9s\n",
			"Memory", "Arenas", "Peak(MB)", "Allocs");
		fprintf(stderr,
			"------------------------------------------------------\n");

		double peak_mb = (double)p->mem.peak_bytes / (1024.0 * 1024.0);
		fprintf(stderr, " %-20s %5d %11.2f %9d\n",
			"inference", p->mem.arena_count, peak_mb,
			p->mem.alloc_count);

		fprintf(stderr,
			"------------------------------------------------------\n\n");
	}

	/* Summary line */
	uint64_t total_ns = total_stage_ns > 0 ? total_stage_ns : total_op_ns;
	double total_ms = (double)total_ns / 1e6;
	double peak_mb = (double)p->mem.peak_bytes / (1024.0 * 1024.0);

	fprintf(stderr, " Total: %.2fms | Peak mem: %.2fMB | %d allocs\n",
		total_ms, peak_mb, p->mem.alloc_count);
	fprintf(stderr,
		"======================================================\n\n");
}
```

**Step 4: Run test to verify it passes**

Run: `cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DSAM3_METAL=OFF -DSAM3_PROFILE=ON && make -j$(sysctl -n hw.ncpu) 2>&1 && ctest -R test_profile --output-on-failure`
Expected: PASS — 3 tests, 0 failures

**Step 5: Commit**

```bash
git add src/util/profile.h src/util/profile.c tests/test_profile.c
git commit -m "util/profile: add profiler core types, lifecycle, and report"
```

---

### Task 3: Stage Timing Tests

**Files:**
- Modify: `tests/test_profile.c`

**Step 1: Add stage timing tests**

Append before `main()` in `tests/test_profile.c`:

```c
static void test_stage_timing(void)
{
	struct sam3_profiler *p = sam3_profiler_create();
	sam3_profiler_enable(p);

	sam3_prof_stage_begin(p, "test_stage");
	/* Burn some time */
	volatile int x = 0;
	for (int i = 0; i < 100000; i++)
		x += i;
	(void)x;
	sam3_prof_stage_end(p, "test_stage");

	ASSERT_EQ(p->n_stages, 1);
	ASSERT_EQ(p->stages[0].calls, 1);
	ASSERT(p->stages[0].total_ns > 0);

	sam3_profiler_free(p);
}

static void test_stage_accumulates(void)
{
	struct sam3_profiler *p = sam3_profiler_create();
	sam3_profiler_enable(p);

	sam3_prof_stage_begin(p, "accum");
	sam3_prof_stage_end(p, "accum");
	sam3_prof_stage_begin(p, "accum");
	sam3_prof_stage_end(p, "accum");

	ASSERT_EQ(p->stages[0].calls, 2);

	sam3_profiler_free(p);
}

static void test_stage_disabled_noop(void)
{
	struct sam3_profiler *p = sam3_profiler_create();
	/* NOT enabled */
	sam3_prof_stage_begin(p, "noop");
	sam3_prof_stage_end(p, "noop");

	ASSERT_EQ(p->n_stages, 0);

	sam3_profiler_free(p);
}
```

Add calls in `main()`:

```c
	test_stage_timing();
	test_stage_accumulates();
	test_stage_disabled_noop();
```

**Step 2: Run test to verify it passes**

Run: `cd build && make -j$(sysctl -n hw.ncpu) 2>&1 && ctest -R test_profile --output-on-failure`
Expected: PASS — 6 tests, 0 failures

**Step 3: Commit**

```bash
git add tests/test_profile.c
git commit -m "tests: add stage timing tests for profiler"
```

---

### Task 4: Op Timing and Memory Tracking Tests

**Files:**
- Modify: `tests/test_profile.c`

**Step 1: Add op timing and memory tests**

Append before `main()` in `tests/test_profile.c`:

```c
static void test_op_timing(void)
{
	struct sam3_profiler *p = sam3_profiler_create();
	sam3_profiler_enable(p);

	sam3_prof_op_begin(p, SAM3_OP_MATMUL);
	volatile int x = 0;
	for (int i = 0; i < 10000; i++)
		x += i;
	(void)x;
	sam3_prof_op_end(p, SAM3_OP_MATMUL);

	ASSERT_EQ(p->op_stats[SAM3_OP_MATMUL].calls, 1);
	ASSERT(p->op_stats[SAM3_OP_MATMUL].total_ns > 0);
	/* Other ops should be untouched */
	ASSERT_EQ(p->op_stats[SAM3_OP_ADD].calls, 0);

	sam3_profiler_free(p);
}

static void test_mem_tracking(void)
{
	struct sam3_profiler *p = sam3_profiler_create();
	sam3_profiler_enable(p);

	sam3_prof_mem_arena(p);
	ASSERT_EQ(p->mem.arena_count, 1);

	sam3_prof_mem_alloc(p, 1024);
	ASSERT_EQ(p->mem.alloc_count, 1);
	ASSERT_EQ((int)p->mem.current_bytes, 1024);
	ASSERT_EQ((int)p->mem.peak_bytes, 1024);

	sam3_prof_mem_alloc(p, 2048);
	ASSERT_EQ(p->mem.alloc_count, 2);
	ASSERT_EQ((int)p->mem.current_bytes, 3072);
	ASSERT_EQ((int)p->mem.peak_bytes, 3072);

	/* Simulate arena reset */
	sam3_prof_mem_arena_reset(p, 3072);
	ASSERT_EQ((int)p->mem.current_bytes, 0);
	/* Peak should not change */
	ASSERT_EQ((int)p->mem.peak_bytes, 3072);

	sam3_profiler_free(p);
}

static void test_report_no_crash(void)
{
	struct sam3_profiler *p = sam3_profiler_create();
	sam3_profiler_enable(p);

	sam3_prof_stage_begin(p, "encode");
	sam3_prof_stage_end(p, "encode");
	sam3_prof_op_begin(p, SAM3_OP_MATMUL);
	sam3_prof_op_end(p, SAM3_OP_MATMUL);
	sam3_prof_mem_arena(p);
	sam3_prof_mem_alloc(p, 4096);

	/* Should not crash */
	sam3_profiler_report(p);

	ASSERT(1); /* If we got here, no crash */
	sam3_profiler_free(p);
}
```

Add calls in `main()`:

```c
	test_op_timing();
	test_mem_tracking();
	test_report_no_crash();
```

**Step 2: Run test to verify it passes**

Run: `cd build && make -j$(sysctl -n hw.ncpu) 2>&1 && ctest -R test_profile --output-on-failure`
Expected: PASS — 9 tests, 0 failures (note: test_report_no_crash counts multiple ASSERTs from test_mem_tracking, actual count will be higher)

**Step 3: Commit**

```bash
git add tests/test_profile.c
git commit -m "tests: add op timing, memory tracking, and report tests"
```

---

### Task 5: CMake Profile Option

**Files:**
- Modify: `CMakeLists.txt`

**Step 1: Add SAM3_PROFILE option**

After line 12 (`option(SAM3_TESTS ...)`), add:

```cmake
option(SAM3_PROFILE "Enable profiler" OFF)
```

After line 32 (end of Debug sanitizer block), add:

```cmake
if(SAM3_PROFILE)
	add_definitions(-DSAM3_HAS_PROFILE)
endif()
```

**Step 2: Verify build with profile ON and OFF**

Run: `cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DSAM3_METAL=OFF -DSAM3_PROFILE=ON && make -j$(sysctl -n hw.ncpu) 2>&1 && ctest --output-on-failure`
Expected: All tests pass (including test_profile and test_time)

Run: `cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DSAM3_METAL=OFF -DSAM3_PROFILE=OFF && make -j$(sysctl -n hw.ncpu) 2>&1 && ctest --output-on-failure`
Expected: Build succeeds. test_profile and test_time still build and pass (they don't use the macros, they call functions directly).

**Step 3: Commit**

```bash
git add CMakeLists.txt
git commit -m "cmake: add SAM3_PROFILE option for compile-time profiler gating"
```

---

### Task 6: Integrate Profiler into sam3_ctx

**Files:**
- Modify: `include/sam3/sam3.h`
- Modify: `src/sam3.c`

**Step 1: Add public profiling API to sam3.h**

Before `#endif /* SAM3_H */` at the end of `include/sam3/sam3.h`, add:

```c

/* --- Profiling API --- */

/*
 * sam3_profile_enable - Enable profiling on this context.
 *
 * @ctx: Initialized context.
 *
 * Allocates profiler state if not already present. Profiling data
 * accumulates until reset or context is freed.
 * Requires SAM3_HAS_PROFILE at compile time; otherwise a no-op.
 */
enum sam3_error sam3_profile_enable(sam3_ctx *ctx);

/*
 * sam3_profile_disable - Disable profiling (data is preserved).
 *
 * @ctx: Initialized context.
 */
void sam3_profile_disable(sam3_ctx *ctx);

/*
 * sam3_profile_report - Print profiling report to stderr.
 *
 * @ctx: Initialized context.
 */
void sam3_profile_report(sam3_ctx *ctx);

/*
 * sam3_profile_reset - Clear all collected profiling data.
 *
 * @ctx: Initialized context.
 */
void sam3_profile_reset(sam3_ctx *ctx);
```

**Step 2: Add profiler field to sam3_ctx and implement API**

In `src/sam3.c`, add include after `#include "sam3/sam3.h"`:

```c
#ifdef SAM3_HAS_PROFILE
#include "util/profile.h"
#endif
```

Add `profiler` field to `struct sam3_ctx`:

```c
struct sam3_ctx {
	struct sam3_model_config config;
	int loaded;
#ifdef SAM3_HAS_PROFILE
	struct sam3_profiler *profiler;
#endif
};
```

Update `sam3_free` to clean up profiler:

```c
void sam3_free(sam3_ctx *ctx)
{
	if (!ctx)
		return;
#ifdef SAM3_HAS_PROFILE
	sam3_profiler_free(ctx->profiler);
#endif
	free(ctx);
}
```

Add the four new functions at the end of `src/sam3.c`:

```c
enum sam3_error sam3_profile_enable(sam3_ctx *ctx)
{
#ifdef SAM3_HAS_PROFILE
	if (!ctx)
		return SAM3_EINVAL;
	if (!ctx->profiler) {
		ctx->profiler = sam3_profiler_create();
		if (!ctx->profiler)
			return SAM3_ENOMEM;
	}
	sam3_profiler_enable(ctx->profiler);
	return SAM3_OK;
#else
	(void)ctx;
	return SAM3_OK;
#endif
}

void sam3_profile_disable(sam3_ctx *ctx)
{
#ifdef SAM3_HAS_PROFILE
	if (ctx && ctx->profiler)
		sam3_profiler_disable(ctx->profiler);
#else
	(void)ctx;
#endif
}

void sam3_profile_report(sam3_ctx *ctx)
{
#ifdef SAM3_HAS_PROFILE
	if (ctx && ctx->profiler)
		sam3_profiler_report(ctx->profiler);
#else
	(void)ctx;
#endif
}

void sam3_profile_reset(sam3_ctx *ctx)
{
#ifdef SAM3_HAS_PROFILE
	if (ctx && ctx->profiler)
		sam3_profiler_reset(ctx->profiler);
#else
	(void)ctx;
#endif
}
```

**Step 3: Verify build**

Run: `cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DSAM3_METAL=OFF -DSAM3_PROFILE=ON && make -j$(sysctl -n hw.ncpu) 2>&1 && ctest --output-on-failure`
Expected: All tests pass.

Run: `cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DSAM3_METAL=OFF -DSAM3_PROFILE=OFF && make -j$(sysctl -n hw.ncpu) 2>&1 && ctest --output-on-failure`
Expected: Build succeeds, all tests pass (profiler API compiles to no-ops).

**Step 4: Commit**

```bash
git add include/sam3/sam3.h src/sam3.c
git commit -m "sam3: integrate profiler into context with public enable/disable/report/reset API"
```

---

### Task 7: Hook Profiler into Arena Allocator

**Files:**
- Modify: `src/core/alloc.h` — add profiler pointer parameter
- Modify: `src/core/alloc.c` — call SAM3_PROF_MEM on allocation

Note: This task takes a different approach than adding a profiler parameter to alloc. Instead, we use a thread-local (file-scope static) profiler pointer that can be set before arena operations. This avoids changing the alloc API signatures which would cascade through the entire codebase.

**Step 1: Add profiler hook to alloc.h**

After the existing function declarations in `src/core/alloc.h`, before `#endif`, add:

```c

struct sam3_profiler;

/* Set the active profiler for memory tracking. NULL to disable. */
void sam3_arena_set_profiler(struct sam3_profiler *p);
```

**Step 2: Implement hook in alloc.c**

Add at the top of `src/core/alloc.c`, after the includes:

```c
#ifdef SAM3_HAS_PROFILE
#include "util/profile.h"
static struct sam3_profiler *g_alloc_profiler;
#endif
```

Add the setter function:

```c
void sam3_arena_set_profiler(struct sam3_profiler *p)
{
#ifdef SAM3_HAS_PROFILE
	g_alloc_profiler = p;
#else
	(void)p;
#endif
}
```

In `sam3_arena_init`, after `arena->offset = 0;` add:

```c
	SAM3_PROF_MEM_ARENA(g_alloc_profiler);
```

In `sam3_arena_alloc`, after `arena->offset = aligned + nbytes;` add:

```c
	SAM3_PROF_MEM(g_alloc_profiler, nbytes);
```

In `sam3_arena_reset`, after `arena->offset = 0;` add:

```c
	SAM3_PROF_MEM_ARENA_RESET(g_alloc_profiler, arena->offset);
```

Wait — the reset zeroes offset before we can read it. Fix: capture offset before reset:

Replace `sam3_arena_reset` with:

```c
void sam3_arena_reset(struct sam3_arena *arena)
{
	size_t freed = arena->offset;
	arena->offset = 0;
	SAM3_PROF_MEM_ARENA_RESET(g_alloc_profiler, freed);
}
```

Also need to add the profile.h include guard for the macros. At the top of alloc.c after the existing includes, add:

```c
#ifdef SAM3_HAS_PROFILE
#include "util/profile.h"
static struct sam3_profiler *g_alloc_profiler;
#else
#define SAM3_PROF_MEM(p, n)              ((void)0)
#define SAM3_PROF_MEM_ARENA(p)           ((void)0)
#define SAM3_PROF_MEM_ARENA_RESET(p, n)  ((void)0)
#endif
```

Actually, the macros are already defined in profile.h with the `#else` fallback. But alloc.c doesn't include profile.h when SAM3_HAS_PROFILE is not defined. So we need the local fallback macros in alloc.c for when profile.h is not included. Simpler: always include profile.h from alloc.c (it has its own guards).

Replace the ifdef block with:

```c
#include "util/profile.h"

#ifdef SAM3_HAS_PROFILE
static struct sam3_profiler *g_alloc_profiler;
#else
static struct sam3_profiler *g_alloc_profiler;
#endif
```

Wait, that's redundant. Simplest approach: always include profile.h (it already handles the `#ifndef SAM3_HAS_PROFILE` case for macros), and always have the static pointer:

After existing includes in alloc.c, add:

```c
#include "util/profile.h"

static struct sam3_profiler *g_alloc_profiler;
```

This works because:
- profile.h defines macros as `((void)0)` when `SAM3_HAS_PROFILE` is not defined
- The static pointer is unused but harmless (compiler may warn; suppress with `(void)g_alloc_profiler` or just accept it)

Actually, cleaner: only declare the pointer when profiling:

```c
#include "util/profile.h"

#ifdef SAM3_HAS_PROFILE
static struct sam3_profiler *g_alloc_profiler;
#else
#define g_alloc_profiler NULL
#endif
```

This way the macros expand to `((void)0)` and the NULL pointer is never used.

**Step 3: Verify build**

Run: `cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DSAM3_METAL=OFF -DSAM3_PROFILE=ON && make -j$(sysctl -n hw.ncpu) 2>&1 && ctest --output-on-failure`
Expected: All tests pass.

Run: `cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DSAM3_METAL=OFF -DSAM3_PROFILE=OFF && make -j$(sysctl -n hw.ncpu) 2>&1 && ctest --output-on-failure`
Expected: Build succeeds, all tests pass.

**Step 4: Commit**

```bash
git add src/core/alloc.h src/core/alloc.c
git commit -m "core/alloc: hook arena allocator into profiler for memory tracking"
```

---

### Task 8: Integration Test — Full Profiler Round Trip

**Files:**
- Modify: `tests/test_profile.c`

**Step 1: Add integration test**

Append before `main()` in `tests/test_profile.c`:

```c
static void test_full_round_trip(void)
{
	struct sam3_profiler *p = sam3_profiler_create();
	sam3_profiler_enable(p);

	/* Simulate a mini inference run */
	sam3_prof_stage_begin(p, "image_encoder");

	sam3_prof_mem_arena(p);
	sam3_prof_mem_alloc(p, 1024 * 1024);  /* 1MB */

	sam3_prof_op_begin(p, SAM3_OP_CONV2D);
	volatile int x = 0;
	for (int i = 0; i < 50000; i++)
		x += i;
	(void)x;
	sam3_prof_op_end(p, SAM3_OP_CONV2D);

	sam3_prof_op_begin(p, SAM3_OP_MATMUL);
	for (int i = 0; i < 50000; i++)
		x += i;
	(void)x;
	sam3_prof_op_end(p, SAM3_OP_MATMUL);

	sam3_prof_stage_end(p, "image_encoder");

	sam3_prof_stage_begin(p, "mask_decoder");
	sam3_prof_op_begin(p, SAM3_OP_SOFTMAX);
	for (int i = 0; i < 10000; i++)
		x += i;
	(void)x;
	sam3_prof_op_end(p, SAM3_OP_SOFTMAX);
	sam3_prof_stage_end(p, "mask_decoder");

	/* Verify data collected */
	ASSERT_EQ(p->n_stages, 2);
	ASSERT(p->stages[0].total_ns > 0);
	ASSERT(p->stages[1].total_ns > 0);
	ASSERT_EQ(p->op_stats[SAM3_OP_CONV2D].calls, 1);
	ASSERT_EQ(p->op_stats[SAM3_OP_MATMUL].calls, 1);
	ASSERT_EQ(p->op_stats[SAM3_OP_SOFTMAX].calls, 1);
	ASSERT_EQ(p->mem.alloc_count, 1);
	ASSERT_EQ((int)p->mem.peak_bytes, 1024 * 1024);

	/* Print report */
	sam3_profiler_report(p);

	sam3_profiler_free(p);
}
```

Add call in `main()`:

```c
	test_full_round_trip();
```

**Step 2: Run test to verify it passes**

Run: `cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DSAM3_METAL=OFF -DSAM3_PROFILE=ON && make -j$(sysctl -n hw.ncpu) 2>&1 && ctest -R test_profile --output-on-failure`
Expected: PASS, and you should see a profile report printed to stderr during test output

**Step 3: Commit**

```bash
git add tests/test_profile.c
git commit -m "tests: add full round-trip integration test for profiler"
```

---

### Task 9: Final Verification

**Step 1: Clean build with profile ON**

```bash
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DSAM3_METAL=OFF -DSAM3_PROFILE=ON
make -j$(sysctl -n hw.ncpu) 2>&1
ctest --output-on-failure
```

Expected: Clean build, all tests pass.

**Step 2: Clean build with profile OFF**

```bash
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DSAM3_METAL=OFF -DSAM3_PROFILE=OFF
make -j$(sysctl -n hw.ncpu) 2>&1
ctest --output-on-failure
```

Expected: Clean build, all tests pass. Profiler code is not compiled into libsam3.a (the macros expand to void).

**Step 3: Verify file headers**

Spot-check that every new `.c` and `.h` file has the documentation header.

**Step 4: Verify git log**

```bash
git log --oneline
```

Expected: Clean commit history with profiler commits following scaffold commits.
