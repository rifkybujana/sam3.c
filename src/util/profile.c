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
