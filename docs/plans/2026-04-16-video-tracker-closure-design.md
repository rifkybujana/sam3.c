# Video Tracker Closure — Design

Date: 2026-04-16
Branch: feature/video-tracker
Supersedes/extends: `2026-04-16-video-tracker-design.md`,
`2026-04-16-video-tracker-plan.md`

## Context

The video-tracker branch scaffolded every module the pipeline needs
(memory bank, memory encoder, memory attention, tracker, video session,
public API types, weight conversion) but left the integration layer as
stubs. An end-to-end review identified nine gap clusters. This design
closes all of them.

The architecture is dictated by the upstream Python reference
(facebookresearch/sam3) and the existing still-image path in this
codebase. Nothing new is being invented; this is strictly wiring and
guardrails.

## Goals

1. `sam3_video_add_points`, `sam3_video_add_box`, and
   `sam3_video_propagate` produce real masks that track objects across
   frames.
2. The `sam3_cli` binary exposes video tracking to end users.
3. Tests actually validate numerical correctness against a Python
   reference, not shapes.
4. The feature is documented and reachable from Python.

## Non-goals

- Live-stream / frame-at-a-time APIs. The batched design stands.
- Multi-threaded inference on a single session. Single-thread per
  session remains the contract.
- Full IoU parity against the upstream Python reference on real video.
  Centroid tracking on a synthetic clip is sufficient for CI.

## Section 1 — Core inference wiring

### 1.1 Backbone feature caching

**Decision:** Eager. In `sam3_video_start`, after loading frames, run
the image encoder on every frame and store the neck outputs in
`session->cached_features[i]`. Persist-arena allocation; lives for the
session lifetime.

Rationale: the batched I/O design already materializes every frame up
front; propagate always visits every frame; eager caching removes
cache-miss branching from the hot path.

Memory budget: ~5 MB per frame at 72×72×256 float32. Persist arena is
1 GiB → practical cap ~200 frames. Acceptable for the clip sizes the
reference targets.

### 1.2 Prompt storage

**Decision:** Flat array inside the session.

```c
struct sam3_video_prompt {
	int                 frame_idx;
	int                 obj_internal_idx;  /* index into obj_ids[] */
	enum { POINTS, BOX } kind;
	union {
		struct { float xys[SAM3_MAX_POINTS_PER_OBJ * 2];
		         int   labels[SAM3_MAX_POINTS_PER_OBJ];
		         int   n; } points;
		struct sam3_box  box;
	};
};

struct sam3_video_session {
	/* ... existing fields ... */
	struct sam3_video_prompt *prompts;   /* capacity = SAM3_MAX_OBJECTS * n_frames */
	int                       n_prompts;
	uint8_t                  *prompted_frames;   /* bitmap, n_frames bits */
};
```

`add_points` / `add_box` append a prompt and set the bit in
`prompted_frames`. `propagate` reads the bitmap to decide `is_cond`.

### 1.3 `add_points` / `add_box` flow

```
validate args
→ get_or_add_obj(user_obj_id) → obj_internal_idx
→ append prompt to session->prompts[]
→ set prompted_frames[frame_idx]
→ prompt_encoder(cached_features[frame_idx], prompt) → sparse + dense embeddings
→ mask_decoder(cached_features[frame_idx], embeddings) → masks + iou + obj_token
→ memory_encoder(cached_features[frame_idx], best_mask) → memory features
→ obj_ptr = obj_ptr_proj(obj_token)
→ memory_bank.add(frame_idx, spatial_features, obj_ptr, is_cond=true, score=iou)
→ copy best_mask into result
→ return SAM3_OK
```

Best-of-4 mask selection uses the highest IoU score, matching upstream.

### 1.4 `propagate` flow

```
for each frame f in direction order:
    if prompted_frames[f]:
        re-run the add_points/add_box flow for every prompt with
        frame_idx == f, merging results into one mask per object
    else:
        tracker_track_frame(cached_features[f], memory_bank)
            → masks + iou + obj_token
        memory_encoder + memory_bank.add(is_cond=false)
    invoke user callback with per-object masks
    if callback returns nonzero, stop
```

Conditioning frames are still added to the bank on the propagate pass
(to match upstream's memory selection heuristic). The bank's existing
FIFO + SAM3-Long selection applies unchanged to non-conditioning
frames.

BOTH direction: forward pass first, then backward from the max
prompted frame going down. This matches upstream.

### 1.5 Object pointer extraction

**Decision:** Extend `sam3_mask_decoder_build` with an optional
out-parameter `out_obj_token`. Current signature becomes:

```c
int sam3_mask_decoder_build(struct sam3_graph *g, struct sam3_arena *arena,
                            const struct sam3_mask_decoder *md,
                            struct sam3_tensor *features,
                            struct sam3_tensor *sparse_embeds,
                            struct sam3_tensor *dense_embeds,
                            struct sam3_tensor *pos_enc,
                            struct sam3_tensor **out_masks,
                            struct sam3_tensor **out_iou,
                            struct sam3_tensor **out_obj_token);  /* NEW, NULLable */
```

Passing NULL preserves existing behavior for the still-image path. The
tracker passes a non-NULL pointer and projects the token through
`obj_ptr_proj_w/b`.

### 1.6 Session reset

Add to `sam3_video_reset`:
- clear `prompted_frames` bitmap
- zero `n_prompts`
- `sam3_arena_reset(&scratch)`
- clear `obj_ids[0..SAM3_MAX_OBJECTS]` for hygiene (cheap)

## Section 2 — CLI, tests, fixtures

### 2.1 CLI

**Decision:** New `tools/cli_track.c` subcommand registered in
`sam3_cli.c` and `CMakeLists.txt`. Delete the orphaned
`tools/sam3_main.c`.

```
sam3_cli track -m <model> -v <video> [options]
  -m, --model    <path>      .sam3 model file
  -v, --video    <path>      video file or frame directory
  -p, --point    x,y,label   add point prompt (repeatable)
  -b, --box      x,y,x,y     add box prompt (repeatable)
      --frame    N           frame index for prompts (default 0)
      --obj-id   N           object id for prompts (default 0)
      --direction fwd|bwd|both   (default fwd)
  -o, --output   <dir>       output directory for per-frame overlays
      --no-overlay           write raw mask PNGs only
  -v, --verbose              debug logging
```

Output: `<outdir>/frame_NNNNN.png` per frame (overlay or raw). Exit
non-zero if propagate returns an error.

### 2.2 Fixture tests

Two changes to `tests/test_tracker_fixtures.c`:

1. Remove the silent-skip. Introduce a CMake option
   `SAM3_FIXTURE_TESTS` (default OFF, CI turns ON). When OFF, the
   test is not registered with CTest. When ON, absence of the
   fixture directory is a **test failure**, not a skip.
2. Replace shape-only checks with tolerance-based comparison. Add
   `assert_tensor_close(actual, expected_path, rtol, atol)` helper in
   `tests/test_helpers.h`. Default tolerances: `rtol=1e-3, atol=1e-4`
   for linear/attention outputs; `rtol=1e-2, atol=1e-3` for mask
   decoder outputs.

### 2.3 Fixture realism

Add a second clip to `scripts/gen_tracker_fixtures.py`: an 8-frame
synthetic "moving square" — 32×32 white square on noisy gray
background, translating diagonally by 8 px/frame. Deterministic
(`torch.manual_seed(0)` + NumPy seeded RNG for the noise). Point
prompt at frame 0 on the square center. Fixture size target ≤ 5 MB.

### 2.4 Integration test

New `tests/test_video_e2e.c`:

1. Load a 2-layer zero-initialized model (shape-only, for arena
   sizing).
2. Start session on the 8-frame synthetic clip (generated in-process,
   not loaded from fixtures).
3. `add_points` on frame 0 at square center.
4. `propagate(FORWARD)` collecting per-frame masks.
5. Assert: mask is non-empty on every frame; centroid is within ±8 px
   of the expected square center.
6. `reset` + `propagate` → callback gets empty results.
7. `end` cleanly; ASan must report no leaks.

This runs in the default test suite (no fixture dependency).

## Section 3 — Polish

### 3.1 Python bindings

Add `VideoSession` to the CFFI package:

```python
class VideoSession:
    def __init__(self, model: Model, video_path: str): ...
    def add_points(self, frame: int, obj_id: int,
                   points: list[tuple[float, float, int]]) -> np.ndarray: ...
    def add_box(self, frame: int, obj_id: int,
                box: tuple[float, float, float, float]) -> np.ndarray: ...
    def propagate(self, direction: str = "forward"
                  ) -> Iterator[tuple[int, list[np.ndarray]]]: ...
    def reset(self) -> None: ...
    def remove_object(self, obj_id: int) -> None: ...
    def close(self) -> None: ...
    # context manager: __enter__ / __exit__
```

`propagate` uses a CFFI callback bridge that yields per-frame results
to the Python iterator. Pytest: same synthetic-clip assertions as the
C integration test.

### 3.2 Documentation

- `README.md`: add "Video object tracking" to Features list; add a
  Quick Start block that mirrors the image path.
- `docs/architecture.md`: add Section "Video Tracker" with
  - data flow diagram (frame → backbone cache → memory bank →
    track_frame → mask)
  - memory bank policy (conditioning FIFO + SAM3-Long score gate)
  - limits (4096 frames max, batched-only, single-threaded)
- Remove the stale "not wired up" sentence at architecture.md:1901.

### 3.3 Benchmarks

Add to `src/bench/`:

- `bench_video_frame` — per-frame track_frame latency on the synthetic
  8-frame clip (pre-populated memory bank, excludes backbone).
- `bench_video_end_to_end` — start + add_points + propagate + end on
  the synthetic clip.

Both registered in the existing bench harness; no Python-reference
comparison.

## Summary of file changes

| File | Change |
|---|---|
| `include/sam3/sam3.h` | Unchanged public API signatures |
| `include/sam3/sam3_types.h` | Add `SAM3_MAX_POINTS_PER_OBJ` |
| `src/model/video_session.h` | Add prompt storage + bitmap |
| `src/model/video_session.c` | Allocators for new fields |
| `src/model/sam3_video.c` | Implement add_points/add_box/propagate |
| `src/model/tracker.c` | Populate memory bank; extract obj_ptr; full memory collection |
| `src/model/mask_decoder.h` | Add `out_obj_token` out-param |
| `src/model/mask_decoder.c` | Emit object-score token |
| `tests/test_video_e2e.c` | NEW |
| `tests/test_helpers.h` | `assert_tensor_close` |
| `tests/test_tracker_fixtures.c` | Real numerical asserts; no silent skip |
| `scripts/gen_tracker_fixtures.py` | Moving-square clip |
| `tools/cli_track.c` | NEW |
| `tools/sam3_cli.c` | Register `track` subcommand |
| `tools/sam3_main.c` | DELETE |
| `tools/cli_common.h` | Remove stale `Used by` reference |
| `CMakeLists.txt` | Add cli_track.c; `SAM3_FIXTURE_TESTS` option; bench entries |
| `python/sam3/video.py` | NEW — `VideoSession` |
| `python/tests/test_video.py` | NEW |
| `README.md` | Features + Quick Start |
| `docs/architecture.md` | Video Tracker section |
| `src/bench/bench_video_frame.c` | NEW |
| `src/bench/bench_video_end_to_end.c` | NEW |

## Risks

1. **Arena sizing.** Backbone cache + per-frame memory entries may
   exceed the current 1 GiB persist arena for long clips. Mitigation:
   log memory use at `video_start` and return `SAM3_ENOMEM` cleanly if
   over budget; document the practical frame cap.
2. **obj_token extension ripples.** Every caller of
   `sam3_mask_decoder_build` must pass NULL or a real pointer. The
   still-image callers (sam3_image.c, test_mask_decoder.c) must be
   updated to pass NULL.
3. **Memory attention with populated bank** was fixed numerically
   (commit 6972b08) but has never been exercised with real memory
   tokens. First real run may surface additional bugs. The integration
   test is the safety net.
4. **Fixture generator on macOS** already works around Triton and bf16
   fused ops. The moving-square addition should keep to the same CPU
   path; no new platform gymnastics.

## Success criteria

- `ctest --output-on-failure` passes including `test_video_e2e`.
- `sam3_cli track -m model.sam3 -v sample.mp4 -p cx,cy,1` produces
  per-frame overlays matching the prompted object.
- `pytest python/tests/test_video.py` passes.
- When `SAM3_FIXTURE_TESTS=ON`, `test_tracker_fixtures` passes with
  real numerical tolerances, not shape-only.
- README + architecture.md describe the feature accurately.
