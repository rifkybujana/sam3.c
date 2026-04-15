# Benchmarking System Design

**Date:** 2026-04-15
**Status:** Approved

## Goal

A unified benchmarking system for SAM3 that supports regression detection,
optimization guidance, and hardware comparison. Two granularity levels: kernel
microbenchmarks and end-to-end pipeline benchmarks. Results stored as JSON
baselines in the repo for automated comparison.

## Approach

Harness library in `src/bench/` integrated into the main `sam3` CLI via
`--bench` mode. Replaces ad-hoc `bench_*.c` files with a shared execution
framework. JSON baseline files checked into `benchmarks/baselines/`.

## Architecture

### Bench Harness Library (`src/bench/`)

Core execution framework. All benchmarks delegate timing, stats, and reporting
to the harness — no benchmark writes its own timing loop.

**Key types:**

```c
struct sam3_bench_config {
    int           warmup_iters;    /* default: 5 */
    int           timed_iters;     /* default: 50 */
    bool          statistical;     /* compute stddev, CI */
    double        threshold_pct;   /* regression threshold, default 5.0 */
    const char   *output_path;     /* JSON output file, NULL = stdout */
    const char   *baseline_path;   /* JSON baseline to compare against */
    enum sam3_backend_type backend; /* CPU or Metal */
};

struct sam3_bench_result {
    const char   *name;            /* e.g., "matmul_f32_1024x1024" */
    const char   *suite;           /* "kernel" or "pipeline" */
    double        mean_ms;
    double        min_ms;
    double        max_ms;
    double        stddev_ms;       /* 0 if !statistical */
    double        gflops;          /* 0 if not applicable */
    double        throughput_mbs;  /* bytes/sec metric, 0 if N/A */
    int           iterations;
};

struct sam3_bench_env {
    char          chip[64];        /* e.g., "Apple M2 Pro" */
    char          os[64];          /* e.g., "Darwin 24.6.0" */
    int           cpu_cores;
    int           gpu_cores;       /* 0 if CPU-only */
    char          backend[16];     /* "cpu" or "metal" */
    char          commit[12];      /* short git SHA */
    char          timestamp[32];   /* ISO 8601 */
};
```

**Core API:**

```c
/* Run a single benchmark case */
int sam3_bench_run(const struct sam3_bench_config *cfg,
                   const char *name, const char *suite,
                   void (*fn)(void *ctx), void *ctx,
                   struct sam3_bench_result *out);

/* Collect hardware/environment metadata */
void sam3_bench_env_detect(struct sam3_bench_env *env);

/* Write results + env to JSON */
int sam3_bench_write_json(const char *path,
                          const struct sam3_bench_env *env,
                          const struct sam3_bench_result *results, int n);

/* Load baseline, compare, print report, return nonzero if regression */
int sam3_bench_compare(const char *baseline_path,
                       const struct sam3_bench_result *current, int n,
                       double threshold_pct, bool statistical);
```

### Kernel Microbenchmark Suite

Tests individual operations at controlled sizes/dtypes.

| Operation | Sizes | Dtypes | Metric |
|-----------|-------|--------|--------|
| matmul | 256-4096² | F32, F16, BF16 | GFLOPS |
| conv2d | 3×3, 5×5 @ 64/128/256ch | F32, F16 | GFLOPS |
| softmax | 1K-64K seq | F32, F16 | ms |
| layernorm | 256-2048 hidden | F32, F16 | ms |
| sdpa | 8/16/32 heads, 256-4096 seq | F32, F16 | ms |
| gelu/silu | 1M-16M elements | F32, F16 | GB/s |
| add/mul | 1M-16M elements | F32, F16, BF16 | GB/s |
| transpose | 1024-2048² | F32, F16 | GB/s |
| upsample | 32→64, 64→256 | F32 | ms |

Each benchmark allocates tensors from a scratch arena, builds a one-op graph,
and calls `sam3_backend_compute()`. The harness times the compute call.

**Registration:**

```c
int sam3_bench_register_kernels(const struct sam3_bench_config *cfg,
                                struct sam3_backend *backend,
                                struct sam3_bench_result *results,
                                int max_results);
```

### Pipeline Benchmark Suite

End-to-end inference benchmarks using real model weights.

| Case | Description |
|------|-------------|
| `image_encode` | Vision backbone only (1024×1024 input) |
| `text_encode` | Text encoder only (max-length prompt) |
| `prompt_encode` | Point/box embedding |
| `mask_decode` | Full mask decoder (single frame) |
| `full_pipeline_point` | End-to-end with point prompt |
| `full_pipeline_box` | End-to-end with box prompt |
| `full_pipeline_text` | End-to-end with text prompt |

Uses synthetic test images (solid color/gradient) for deterministic timing.
Automatically adapts to whichever backbone is loaded (Hiera, EfficientViT,
TinyViT). Model variant recorded in JSON metadata.

**Registration:**

```c
int sam3_bench_register_pipeline(const struct sam3_bench_config *cfg,
                                 struct sam3_ctx *ctx,
                                 struct sam3_bench_result *results,
                                 int max_results);
```

### JSON Format

```json
{
  "version": 1,
  "env": {
    "chip": "Apple M2 Pro",
    "os": "Darwin 24.6.0",
    "cpu_cores": 12,
    "gpu_cores": 19,
    "backend": "metal",
    "commit": "a1b2c3d",
    "timestamp": "2026-04-15T10:30:00Z",
    "model_variant": "hiera_large"
  },
  "config": {
    "warmup_iters": 5,
    "timed_iters": 50,
    "statistical": false,
    "threshold_pct": 5.0
  },
  "results": [
    {
      "name": "matmul_f32_1024x1024",
      "suite": "kernel",
      "mean_ms": 1.23,
      "min_ms": 1.10,
      "max_ms": 1.45,
      "stddev_ms": 0.0,
      "gflops": 1748.0,
      "throughput_mbs": 0.0,
      "iterations": 50
    }
  ]
}
```

### Baseline Comparison

**Workflow:**
1. Run benchmarks, save: `sam3 --bench all --output benchmarks/baselines/m2pro_metal.json`
2. Baselines checked into `benchmarks/baselines/`
3. Compare: `sam3 --bench all --compare benchmarks/baselines/m2pro_metal.json`

**Comparison modes:**
- **Percentage-based** (default): Flag if current > baseline × (1 + threshold/100)
- **Statistical** (`--statistical`): Run 200+ iterations, flag if
  current_mean > baseline_mean + 2 × baseline_stddev

**Exit code:** 0 = no regressions, 1 = regression detected (CI-friendly).

**Report format:**

```
Benchmark Comparison (baseline: m2pro_metal.json, threshold: 5.0%)
─────────────────────────────────────────────────────────────────
  matmul_f32_1024x1024    1.23ms → 1.25ms   (+1.6%)  OK
  matmul_f16_1024x1024    0.65ms → 0.71ms   (+9.2%)  REGRESSION
  full_pipeline_point    45.20ms → 44.80ms  (-0.9%)  OK
─────────────────────────────────────────────────────────────────
  1 regression detected (threshold: 5.0%)
```

### CLI Integration

Integrated into `sam3` as `--bench` mode with early-exit path in
`tools/sam3_main.c`. No changes to inference codepath.

```
sam3 --bench [SUITE] [OPTIONS] --model MODEL

Suites:
  kernels    Kernel microbenchmarks only
  pipeline   Pipeline benchmarks only (requires --model)
  all        Both suites

Options:
  --model PATH         .sam3 model weights (required for pipeline)
  --backend cpu|metal  Backend (default: auto-detect)
  --output PATH        Write JSON results to file
  --compare PATH       Compare against baseline JSON
  --threshold PCT      Regression threshold (default: 5.0%)
  --statistical        Statistical comparison mode
  --warmup N           Warmup iterations (default: 5)
  --iters N            Timed iterations (default: 50)
  --filter PATTERN     Run only matching benchmarks
  -v                   Verbose (per-iteration timings)
```

## File Layout

```
src/bench/
  bench.h              Harness API
  bench.c              Harness implementation (timing, stats)
  bench_kernels.h      Kernel suite registration
  bench_kernels.c      All kernel microbenchmarks
  bench_pipeline.h     Pipeline suite registration
  bench_pipeline.c     Pipeline benchmarks
  bench_json.c         JSON serializer/parser (uses vendored cJSON)
  bench_compare.c      Baseline comparison + regression detection

benchmarks/
  baselines/           Checked-in baseline JSON files
  .gitkeep
```

## Build Integration

- New CMake option: `-DSAM3_BENCH=ON` (default OFF)
- `src/bench/` compiled as static library `sam3_bench`
- Linked into main `sam3` binary behind `#ifdef SAM3_HAS_BENCH`
- No new external dependencies (uses vendored cJSON)

## Migration

Existing `tests/bench_dtype.c`, `tests/bench_tokenizer.c`, and
`tests/bench_metal.c` remain as-is during initial implementation. Once the
new system is validated, they get deprecated.
