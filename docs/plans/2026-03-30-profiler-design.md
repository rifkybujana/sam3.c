# SAM3 Profiler Design

**Date**: 2026-03-30
**Status**: Approved

## Overview

Comprehensive profiler for the sam3 inference engine covering op-level timing,
pipeline stage timing, and memory tracking. Instrumentation-based (not sampling).
Zero overhead when compiled out.

## Architecture

The profiler is a `src/util/profile.h/.c` module. It collects timing and memory
events during inference, then produces a perf-style summary report on demand.

The profiler context is embedded in `sam3_ctx` as a `struct sam3_profiler *`,
allocated only when profiling is enabled.

```
User / model code
  SAM3_PROF_BEGIN / SAM3_PROF_END (stage timing)
  SAM3_PROF_OP_BEGIN / SAM3_PROF_OP_END (op timing)
        │
        ▼
sam3_profiler (fixed-size event buffer)
  - stage_events[SAM3_PROF_MAX_STAGES]
  - op_stats[SAM3_OP_COUNT] (per-op aggregates)
  - mem_stats (peak, current, alloc count)
  - enabled flag (runtime toggle)
        │
        ▼
sam3_profile_report() → stderr text output
```

## Timing

### Stage Timing

High-level pipeline stages (image encoding, prompt encoding, mask decoding):

```c
struct sam3_prof_stage {
    const char *name;
    uint64_t    start_ns;
    uint64_t    total_ns;
    int         calls;
};
```

### Op Timing

Per-operation type aggregates indexed by `enum sam3_op`:

```c
struct sam3_prof_op_stats {
    uint64_t total_ns;
    int      calls;
};
```

### Clock Source

`clock_gettime(CLOCK_MONOTONIC)` on Linux, `mach_absolute_time()` on macOS.
Wrapped in `sam3_time_ns()`.

## Memory Tracking

Hooks into the arena allocator:

```c
struct sam3_prof_mem {
    size_t peak_bytes;
    size_t current_bytes;
    int    alloc_count;
    int    arena_count;
};
```

Each `sam3_arena_alloc()` call updates counters when profiling is active.

## Compile/Runtime Gating

### Compile-Time

CMake option `SAM3_PROFILE` (default OFF). When OFF, all macros expand to
`((void)0)` — zero overhead.

```c
#ifdef SAM3_HAS_PROFILE
#define SAM3_PROF_BEGIN(ctx, name) sam3_prof_stage_begin((ctx)->profiler, (name))
#define SAM3_PROF_END(ctx, name)   sam3_prof_stage_end((ctx)->profiler, (name))
#define SAM3_PROF_OP_BEGIN(prof, op) sam3_prof_op_begin((prof), (op))
#define SAM3_PROF_OP_END(prof, op)   sam3_prof_op_end((prof), (op))
#define SAM3_PROF_MEM(prof, nbytes)  sam3_prof_mem_alloc((prof), (nbytes))
#else
#define SAM3_PROF_BEGIN(ctx, name)   ((void)0)
#define SAM3_PROF_END(ctx, name)     ((void)0)
#define SAM3_PROF_OP_BEGIN(prof, op) ((void)0)
#define SAM3_PROF_OP_END(prof, op)   ((void)0)
#define SAM3_PROF_MEM(prof, nbytes)  ((void)0)
#endif
```

### Runtime

`sam3_profile_enable(ctx)` / `sam3_profile_disable(ctx)`. Even when compiled in,
the profiler is dormant until enabled. The enabled flag is checked by inline
functions (branch prediction favors disabled path).

## Report Output

`sam3_profile_report(ctx)` dumps to stderr:

```
═══════════════════════════════════════════════════════
 sam3 profile report
═══════════════════════════════════════════════════════
 Stage              Calls    Total(ms)   Avg(ms)     %
───────────────────────────────────────────────────────
 image_encoder          1     142.30     142.30   68.2
 prompt_encoder         1       0.82       0.82    0.4
 mask_decoder           1      65.40      65.40   31.4
───────────────────────────────────────────────────────

 Op Breakdown       Calls    Total(ms)   Avg(ms)     %
───────────────────────────────────────────────────────
 MATMUL               384     120.50       0.31   57.8
 CONV2D                16      28.30       1.77   13.6
 SOFTMAX               48      22.10       0.46   10.6
 LAYERNORM             96       8.20       0.09    3.9
 GELU                  48       6.80       0.14    3.3
 ADD                  192       5.40       0.03    2.6
 RESHAPE              128       1.20       0.01    0.6
 TRANSPOSE             64       0.80       0.01    0.4
───────────────────────────────────────────────────────

 Memory              Arenas   Peak(MB)    Allocs
───────────────────────────────────────────────────────
 inference               2     512.40       847
───────────────────────────────────────────────────────

 Total: 208.52ms | Peak mem: 512.40MB | 847 allocs
═══════════════════════════════════════════════════════
```

Ops sorted by total time descending. Only ops with calls > 0 shown.

## Files

### New
- `src/util/profile.h` — profiler types, macros, function declarations
- `src/util/profile.c` — timing, memory tracking, report implementation
- `tests/test_profile.c` — profiler lifecycle, timing, memory, report tests

### Modified
- `src/sam3.c` — add `struct sam3_profiler *profiler` to `sam3_ctx`
- `src/core/alloc.c` — add `SAM3_PROF_MEM()` in `sam3_arena_alloc()`
- `include/sam3/sam3.h` — add public profiling API functions
- `CMakeLists.txt` — add `option(SAM3_PROFILE ...)`

### Not Modified (Yet)
- Backend `graph_eval` — op timing hooks added when backends are implemented
- Model build functions — stage hooks added when model code is real
