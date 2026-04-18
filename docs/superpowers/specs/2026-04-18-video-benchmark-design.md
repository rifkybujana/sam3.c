# Video Benchmark Expansion — Design

**Date:** 2026-04-18
**Scope:** Extend the existing SAM3 benchmark suite with comprehensive video
tracker coverage, add committed baseline JSON files for regression detection,
and document the new cases in `BENCHMARK.md`.

## Background

Two video benchmarks already exist in `src/bench/`:

- `bench_video_frame.c` — times `sam3_video_reset → add_points →
  propagate(FORWARD)` on a 2-frame synthetic clip, isolating per-frame
  tracking cost.
- `bench_video_end_to_end.c` — times the full `video_start → add_points →
  propagate → end` cycle on an 8-frame clip.

Both are wired into `tools/cli_bench.c` and run as part of the `pipeline` /
`all` suites. The benchmark harness (`src/bench/bench.{h,c}`), JSON writer
(`bench_json.c`), and compare-by-name regression flow (`bench_compare.c`) are
generic and already handle arbitrary named cases.

## Gaps

1. **Documentation.** `BENCHMARK.md` currently reports kernel, image-encoding,
   and segmentation results. It has no video section at all.
2. **Coverage.** The existing two cases are narrow: one object, one prompt
   type (point), FORWARD only, tiny clips (2 and 8 frames), synthetic moving
   square. There is no signal on the dimensions that matter in practice:
   memory-bank saturation as clip length grows, multi-object scaling,
   BOTH-vs-FORWARD propagation cost.
3. **Regression tracking.** `benchmarks/baselines/` contains only a
   `.gitkeep`. No committed baseline JSON files, so `--compare` has nothing
   to diff against.

## Goals

- Replace the single-case video benchmarks with a parameterised case-table
  that sweeps (clip length × object count × direction).
- Commit per-model baseline JSON files under `benchmarks/baselines/` and
  provide a script to refresh them.
- Add two new sections to `BENCHMARK.md`: "Video: Per-Frame Tracking Cost"
  and "Video: End-to-End Clip Latency", each with focused tables and brief
  commentary.

## Non-Goals

- Real video footage ingestion — synthetic clips only.
- Text-prompted video tracking (no public API for this today).
- CI perf gating — regression detection stays developer-driven.
- Non-Mac baselines — matches existing doc; platform expansion is a separate
  spec.

## Design

### Code layout

**Modified files:**

- `src/bench/bench_video.h`
  - Add `#define SAM3_BENCH_VIDEO_CLIP_MAX_FRAMES 128`.
  - Declare `struct bench_video_case { int n_frames; int n_objects;
    int seed_frame; int direction; const char *label; }`. `direction` holds
    an `enum sam3_propagate_direction` value (`SAM3_PROPAGATE_FORWARD` or
    `SAM3_PROPAGATE_BOTH`; `SAM3_PROPAGATE_BACKWARD` is not needed — BOTH
    covers the reverse pass).
- `src/bench/bench_video_frame.c`
  - Replace the single invocation with a case-table driver iterating over
    an internal array of `bench_video_case` entries.
  - Fix `sam3_bench_generate_clip` so the moving square bounces off canvas
    edges instead of writing out of bounds (required for clips >19 frames
    on the 256² canvas).
  - Add `seed_n_objects(session, n, frame_idx)` helper that places `n`
    distinct object IDs inside the frame-0 square at spread positions.
- `src/bench/bench_video_end_to_end.c`
  - Same case-table pattern, different case matrix.
- `tools/cli_bench.c` — no change. Still calls the same two entry points.

**New files:**

- `benchmarks/baselines/efficient.json` — committed baseline for
  `efficient.sam3`.
- `benchmarks/baselines/tinyvit.json` — committed baseline for
  `tinyvit_l.sam3`.
- `benchmarks/baselines/hiera.json` — committed baseline for `sam3.sam3`.
- `scripts/refresh_baselines.sh` — rebuild and regenerate all three JSON
  files. Skips models that are absent.
- `tests/test_bench_video.c` — smoke tests for the new helpers, guarded by
  `SAM3_BENCH`.

**Documentation:**

- `BENCHMARK.md` — two new sections inserted after "End-to-End Segmentation"
  and before "Peak Throughput Summary".

### Clip-generator bounce fix

The existing `sam3_bench_generate_clip` walks the square diagonally from
`(100, 100)` with step 8 on a 256² canvas. It writes OOB after ~19 frames
because `x0 = 100 + i*8` exceeds `256 - 32 = 224`.

Fix with a triangle-wave reflection so the square stays in bounds at any
frame count:

```c
int max_pos = SAM3_BENCH_VIDEO_IMG_SIZE - SAM3_BENCH_VIDEO_SQUARE_SIZE; /* 224 */
int period  = 2 * max_pos;                                              /* 448 */
int raw     = i * SAM3_BENCH_VIDEO_SQUARE_STEP;
int t       = raw % period;
int x0      = (t <= max_pos) ? t : (period - t);
int y0      = x0;  /* same reflection for y */
```

The frame-0 seed point (and `add_points` coordinates) are unchanged because
they target the square at frame 0, which still starts near `(100, 100)`.
Deterministic LCG noise background is preserved.

### Bench case matrix

**Per-frame cost cases** (`bench_video_frame.c`):

| # | name                                  | n_frames | n_objects | seed frame | direction |
|---|---------------------------------------|---------:|----------:|-----------:|-----------|
| 1 | `video_per_frame_8f_1obj_fwd`         | 8        | 1         | 0          | FORWARD   |
| 2 | `video_per_frame_32f_1obj_fwd`        | 32       | 1         | 0          | FORWARD   |
| 3 | `video_per_frame_64f_1obj_fwd`        | 64       | 1         | 0          | FORWARD   |
| 4 | `video_per_frame_32f_2obj_fwd`        | 32       | 2         | 0          | FORWARD   |
| 5 | `video_per_frame_32f_4obj_fwd`        | 32       | 4         | 0          | FORWARD   |
| 6 | `video_per_frame_32f_8obj_fwd`        | 32       | 8         | 0          | FORWARD   |
| 7 | `video_per_frame_32f_1obj_both`       | 32       | 1         | 16 (middle)| BOTH      |

Semantics: timed function = `reset + seed(n_objects) + propagate(direction)`.
`sam3_bench_result.mean_ms` is for that whole sequence. For the "ms per
frame" numbers in `BENCHMARK.md`, divide `mean_ms` by the number of tracked
frames: `(n_frames - 1)` for FORWARD cases, `(n_frames - 1)` for BOTH
(frames 0..15 backward + 16..31 forward, minus seed = 31). Reset/seed
overhead is amortised — for 32+ frame cases it is <5% of `mean_ms`, so the
approximation is fine and is called out in the table commentary.

For BOTH direction the seed is placed on the middle frame so forward and
backward passes do equal work. The seed-point coordinates use the
bounced-square position at the seed frame, not frame 0.

Memory bank saturates after ~6–8 tracked frames so 32f and 64f both measure
steady-state cost. The 8f case sits below saturation and is kept as a
sanity-check datapoint, not a steady-state measurement.

**End-to-end cases** (`bench_video_end_to_end.c`):

| # | name                            | n_frames | n_objects | direction |
|---|---------------------------------|---------:|----------:|-----------|
| 1 | `video_e2e_8f_1obj_fwd`         | 8        | 1         | FORWARD   |
| 2 | `video_e2e_32f_1obj_fwd`        | 32       | 1         | FORWARD   |
| 3 | `video_e2e_64f_1obj_fwd`        | 64       | 1         | FORWARD   |
| 4 | `video_e2e_32f_4obj_fwd`        | 32       | 4         | FORWARD   |

Semantics: full user-facing latency including session init, feature caching,
and session teardown.

Total: 11 new cases per model variant. Across efficient / tinyvit / hiera
that is 33 benchmark rows in a full `bench all` run. Runtime scales with
model: ≈10 min for EfficientViT, ≈30 min for Hiera. Runs are opt-in so
this is acceptable.

### Case-table driver

One static descriptor array per file, one driver loop:

```c
struct bench_video_case {
    int         n_frames;
    int         n_objects;
    int         seed_frame;     /* frame index where add_points is called */
    int         direction;      /* SAM3_PROPAGATE_FORWARD | _BOTH */
    const char *label;
};

static const struct bench_video_case per_frame_cases[] = {
    { 8,  1, 0,  SAM3_PROPAGATE_FORWARD, "8f_1obj_fwd"   },
    { 32, 1, 0,  SAM3_PROPAGATE_FORWARD, "32f_1obj_fwd"  },
    { 64, 1, 0,  SAM3_PROPAGATE_FORWARD, "64f_1obj_fwd"  },
    { 32, 2, 0,  SAM3_PROPAGATE_FORWARD, "32f_2obj_fwd"  },
    { 32, 4, 0,  SAM3_PROPAGATE_FORWARD, "32f_4obj_fwd"  },
    { 32, 8, 0,  SAM3_PROPAGATE_FORWARD, "32f_8obj_fwd"  },
    { 32, 1, 16, SAM3_PROPAGATE_BOTH,    "32f_1obj_both" },
};
```

Driver generates a 64-frame clip once (max across cases). Each case reuses
that directory, calling `sam3_video_start` → `seed_n_objects` → `propagate`
via `sam3_bench_run` with name `"video_per_frame_" + label`. Filter-match
is evaluated per case so `--filter "video_*"` or `--filter "*_4obj_*"`
behave as expected.

### Multi-object seeding helper

```c
static int seed_n_objects(sam3_video_session *s, int n, int frame_idx);
```

Places `n` points inside the frame-0 square at spread positions
(`center ± k · (SQUARE_SIZE / (2*n))` for k = 0..n-1) using distinct
`obj_id = k`. All points stay within the 32x32 white square at frame 0
so the tracker has a consistent target per object. Returns SAM3_OK or the
first error from `sam3_video_add_points`. Frees the interim result struct
after each call to avoid leaks.

### JSON schema

No change. `sam3_bench_result` already carries everything needed. New
cases drop in with `suite = "pipeline"`. `sam3_bench_env.model_variant`
populates from the loaded model and is used to key baseline files.

### Baseline workflow

`scripts/refresh_baselines.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
BUILD=${BUILD:-build-release}
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DSAM3_BENCH=ON
cmake --build "$BUILD" -j
for variant in efficient tinyvit hiera; do
    model="models/${variant}.sam3"
    if [ ! -f "$model" ]; then
        echo "skip: $model not found"
        continue
    fi
    "$BUILD/sam3_cli" bench all \
        --model "$model" --backend metal \
        --output "benchmarks/baselines/${variant}.json"
done
```

Developer regression flow:

```bash
./build-release/sam3_cli bench all \
    --model models/efficient.sam3 --backend metal \
    --compare benchmarks/baselines/efficient.json --threshold 5.0
```

Exits non-zero on any regression beyond 5%. No CI integration; the script
is run manually before and after perf-sensitive changes.

### BENCHMARK.md additions

Two new top-level sections inserted after "End-to-End Segmentation":

**`## Video: Per-Frame Tracking Cost`** — with three tables:

1. Clip-length scaling (1 object, FORWARD): model × {8f, 32f, 64f}.
2. Multi-object scaling (32 frames, FORWARD): model × {1, 2, 4, 8} objects.
3. Propagation direction (32 frames, 1 object): model × {FORWARD,
   BOTH (fwd+bwd)}. The BOTH case seeds on the middle frame so each
   direction does equal work.

Commentary: memory-bank saturation validation, linear-in-N-objects scaling,
BOTH-cost expectation (~1.5–2× forward, since encoded features are reused
across both passes).

**`## Video: End-to-End Clip Latency`** — two tables:

1. End-to-end latency (1 object, FORWARD): model × {8f, 32f, 64f} + an
   Effective-FPS column at 64 frames.
2. Multi-object end-to-end (32 frames, FORWARD): model × {1 obj, 4 obj}.

Commentary: user-facing-FPS for interactive annotation workflows.

Actual numbers are populated during implementation by running the bench
against each model variant and copy-pasting the results table. Until then
the doc uses placeholder `_tbd_` entries.

Existing "Running the Benchmark" section picks up a new example:

```bash
./sam3_cli bench all --model ../models/efficient.sam3 --backend metal \
    --filter "video_*" --output video.json
```

## Testing

**New test file `tests/test_bench_video.c`** (guarded by `SAM3_BENCH`):

- `test_bench_video_generate_clip_bounce` — generate a 128-frame clip into
  a tmpdir, verify every `frame_XXXX.png` exists and is ≥ 100 bytes.
- `test_bench_video_generate_clip_in_bounds` — decode each frame, assert
  every white pixel (value ≥ 250) falls within `[0, IMG_SIZE)` — regression
  test for the bounce fix.
- `test_bench_video_case_table_filter` — assert
  `sam3_bench_filter_match("video_per_frame_32f_4obj_fwd", "*_4obj_*")`
  returns true and a few negative cases return false. Pure filter-glob
  check, no model needed.

**CMake wiring:** register `test_bench_video.c` under `add_executable` in
`CMakeLists.txt` inside the existing `if(SAM3_BENCH)` block so the test
only builds when bench is enabled.

**Not tested automatically:**

- Exact timing numbers — inherently flaky, covered by manual bench runs.
- End-to-end regression against real baselines in CI — out of scope.

**Manual verification** before committing baselines: run
`sam3_cli bench all --model models/efficient.sam3 --backend metal` at least
twice, confirm all new `video_*` rows appear with sane numbers (nonzero,
non-NaN, stddev < 20% of mean).

## Risks

- **Bench runtime.** A full `bench all` with Hiera-L ≈30 min. Mitigation:
  `--filter` is already supported; the refresh script can be run per-model
  overnight.
- **Baseline staleness.** Committed JSON files drift from reality on hardware
  variance or model updates. Mitigation: `refresh_baselines.sh` is one
  command; baseline commits include the machine and commit in the JSON `env`
  section already.
- **Multi-object collision.** If the seed points are too close, the tracker
  may merge them into one mask. Mitigation: the seeding helper spreads
  points across the 32×32 square; the eight-object case packs them tightly
  but each distinct `obj_id` guarantees separate tracks regardless of
  mask-overlap.

## Out-of-Scope / Follow-Ups

- Real video (MP4/MOV) ingestion benchmarks.
- Text-prompted video (needs public API).
- CI perf gating.
- Windows/Linux baselines.
- Per-stage video breakdown (encoder vs memory-attention vs mask-decoder
  split). Interesting for optimisation work but not needed for regression
  tracking.
