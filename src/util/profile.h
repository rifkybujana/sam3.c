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
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_UTIL_PROFILE_H
#define SAM3_UTIL_PROFILE_H

#include <stdint.h>
#include <stddef.h>

#include "core/graph.h"

#define SAM3_PROF_MAX_STAGES 32

/* Per-stage timing record. */
struct sam3_prof_stage {
	const char *name;
	uint64_t    start_ns;
	uint64_t    total_ns;
	int         calls;
	int         depth;        /* Nesting depth when this stage began */
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
	int                        enabled;
	struct sam3_prof_stage     stages[SAM3_PROF_MAX_STAGES];
	int                        n_stages;
	int                        depth;      /* Current nesting depth */
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

#define SAM3_PROF_BEGIN(prof, name)            ((void)0)
#define SAM3_PROF_END(prof, name)              ((void)0)
#define SAM3_PROF_OP_BEGIN(prof, op)           ((void)0)
#define SAM3_PROF_OP_END(prof, op)             ((void)0)
#define SAM3_PROF_MEM(prof, nbytes)            ((void)0)
#define SAM3_PROF_MEM_ARENA(prof)              ((void)0)
#define SAM3_PROF_MEM_ARENA_RESET(prof, freed) ((void)0)

#endif /* SAM3_HAS_PROFILE */

#endif /* SAM3_UTIL_PROFILE_H */
