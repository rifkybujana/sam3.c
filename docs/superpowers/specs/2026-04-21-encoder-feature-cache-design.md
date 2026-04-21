# Encoder Feature Cache — Design

**Date:** 2026-04-21
**Status:** Approved for implementation planning
**Scope:** In-memory LRU caching of image-encoder and text-encoder outputs in
the single-image (non-video) pipeline.

## Motivation

An image-editing application built on SAM3 typically loads one image and then
runs many segmentation prompts against it. Today:

- The image encoder output is cached **only as a single slot** inside
  `proc->model_arena`. Loading a different image and flipping back re-runs the
  full encoder (hundreds of milliseconds on an M-series GPU).
- The text encoder output is **one-shot**: `sam3_set_text("cat")` spawns a
  worker whose result is consumed and cleared by the next `sam3_segment()`.
  Re-typing `"cat"` after `"dog"` re-runs the encoder.

Both are wasted work for an interactive editor. This design adds a small
in-memory LRU cache in front of each encoder, keyed by content hash.

## Non-Goals

- **No persistence.** The cache lives only for the process lifetime. Reopening
  the app re-encodes. (Disk caching can be layered on later without changing
  the in-memory design.)
- **No video frame cache changes.** The tiered frame cache in
  `src/model/frame_cache.{c,h}` is unrelated and untouched.
- **No implicit invalidation on weight changes.** `sam3_load_model()` flushes
  the caches; everything else assumes weights are stable.

## Architecture

### New module: `src/model/feature_cache.{h,c}`

A self-contained LRU keyed by 64-bit content hash. Owns its own arenas and is
owned by `sam3_processor`. Never touches `sam3_image_model` internals beyond
reading/writing the `cached_*` tensor pointers.

### Image cache

```c
struct sam3_image_bundle {
	/* Mirrors sam3_image_model.cached_* — 8 tensor pointers. */
	struct sam3_tensor *image_features;   /* 0.5x / neck */
	struct sam3_tensor *feat_s0_nhwc;
	struct sam3_tensor *feat_s1_nhwc;
	struct sam3_tensor *feat_4x_nhwc;
	struct sam3_tensor *sam2_05x_nhwc;
	struct sam3_tensor *sam2_1x_nhwc;
	struct sam3_tensor *sam2_2x_nhwc;
	struct sam3_tensor *sam2_4x_nhwc;
	int prompt_w, prompt_h;               /* captured at encode time */
};

struct sam3_image_cache_slot {
	uint64_t hash;                        /* 0 == empty */
	uint64_t lru_tick;
	struct sam3_arena arena;              /* one arena per slot */
	struct sam3_image_bundle bundle;
	int w, h;                             /* verification on hit */
};

struct sam3_image_feature_cache {
	int n_slots;
	uint64_t next_tick;
	uint64_t hits, misses, evictions;
	struct sam3_image_cache_slot *slots;  /* malloc'd at init */
};
```

- Default `n_slots = 3`. Configurable via `sam3_init_ex()`. Hard-capped at
  16 to prevent accidental multi-GiB allocations.
- Each slot's arena is sized at encoder peak output (reuses whatever
  `sam3_image_model_encode` writes to `persist`).
- On hit: pointers are copied into `model.cached_*`; `model.image_encoded = 1`;
  `proc->prompt_w/h` are restored from the bundle.
- On miss: LRU-oldest slot is picked (smallest `lru_tick`), its arena is
  reset, and the encoder runs into that arena.

### Text cache

```c
struct sam3_text_bundle {
	struct sam3_tensor *features;         /* [n_tokens, d_model] */
	int n_tokens;
};

struct sam3_text_cache_slot {
	uint64_t hash;
	uint64_t lru_tick;
	struct sam3_text_bundle bundle;
	/* Points into the shared text_cache_arena. */
};

struct sam3_text_feature_cache {
	int n_slots;
	uint64_t next_tick;
	uint64_t hits, misses, evictions;
	struct sam3_arena arena;              /* shared; ring-reset on wrap */
	struct sam3_text_cache_slot *slots;
};
```

- Default `n_slots = 16`. Text features are tiny (~150 KiB), so a single
  shared arena partitioned into **fixed-size cells** — slot `i` owns the
  byte range `[i * max_slot_bytes, (i+1) * max_slot_bytes)`. Eviction
  simply overwrites the cell's contents on the next miss; no compaction,
  no fragmentation.
- `max_slot_bytes` is computed at init as `ctx_len * d_model * sizeof(f32)
  + tensor-header overhead`, so every slot is guaranteed to fit the
  encoder's worst-case output.

### Changes to `sam3_processor`

```c
struct sam3_processor {
	...
	/* REPLACES: image_loaded flag + model_arena.weights_end rollback. */
	struct sam3_image_feature_cache *img_cache;
	int                              current_img_slot;  /* -1 = none */

	/* REPLACES: text_persist_arena + text_features_async. */
	struct sam3_text_feature_cache  *txt_cache;
	struct sam3_text_bundle         *text_cached_bundle; /* hit path */
	int                              text_worker_slot;   /* -1 or slot id */
	/* text_scratch_arena, text_backend, text_thread*, text_tokens,
	   text_n_tokens are unchanged. */
};
```

`model_arena` goes back to pure weight storage. No more `weights_end`
offset rollback on every `set_image` — a real simplification.

## Data Flow

### Hashing

**Image:** FNV-1a 64-bit over the raw RGB buffer passed to
`sam3_set_image`. Hash seed mixes in `(width, height, prompt_w, prompt_h)`
so two different logical images never collide on hash alone.

For `sam3_set_image_file`, the hash is taken over the **post-resize** pixel
buffer (the one that enters the encoder), so trivial resize differences
don't produce false misses.

Cost: ~10 ms for a 2048×2048 image on M-series, negligible vs. the
100–500 ms encoder.

**Text:** FNV-1a over the tokenized int32 array (not the raw string), so
whitespace / Unicode quirks fold consistently with what the encoder sees.

### `sam3_processor_set_image`

1. Hash the input pixels.
2. Lookup in `img_cache`.
3. **Hit:** bump `lru_tick`; copy bundle pointers into `model.cached_*`;
   set `model.image_encoded = 1`, `proc->current_img_slot = i`,
   `proc->prompt_w/h`. Return `SAM3_OK`. No encoder call, no backend sync.
4. **Miss:** pick LRU slot; reset its arena; run encoder with
   `persist = &slot->arena`; copy the 8 output pointers into
   `slot->bundle`; register the hash. `stats.misses++`.

### `sam3_processor_set_text`

1. Tokenize on caller thread (unchanged).
2. Hash the token array.
3. Lookup in `txt_cache`.
4. **Hit:** bump LRU; set `proc->text_cached_bundle = &slot->bundle`;
   return `SAM3_OK` immediately. **No worker is spawned.**
5. **Miss:** claim a fresh slot (evict LRU if full); spawn the worker as
   today, but the worker writes its output tensor into the shared
   `txt_cache->arena` and records the pointer in
   `slot->bundle.features`. Register the hash when the worker returns
   successfully.

### `sam3_processor_segment` — text consumption

```
if (proc->text_cached_bundle) {
	text_features = proc->text_cached_bundle->features;
	proc->text_cached_bundle = NULL;      /* one-shot consume */
} else if (proc->text_thread_active) {
	join_text_worker(proc);
	/* Worker wrote into txt_cache slot; pick it up. */
	text_features = txt_cache->slots[proc->text_worker_slot]
	                        .bundle.features;
} else {
	/* Legacy inline path — no set_text called. */
	...
}
```

Key change vs. today: text features are **not copied into `model_arena`**
after being produced. The cache arena outlives any single `segment()` call,
so downstream stages (graph build, backend eval) reference the tensor data
directly from the cache. This removes the copy that today happens in
`sam3_processor_segment` around line 905–918.

### Invalidation

- `sam3_load_model()` → flush both caches.
- `sam3_cache_clear(ctx, which)` → explicit flush by user.
- No other invalidation. Cache entries are content-hashed, so as long as
  weights don't change, cached features remain valid.

## Thread Safety

Cache access happens on the main thread only. The text worker is assigned
a slot by the main thread *before* spawning; the worker writes only into
that slot's region of the shared arena. The main thread does not read
that slot until `pthread_join` returns. No locks are added.

## Public API Additions

```c
/* include/sam3/sam3.h */

struct sam3_cache_opts {
	int n_image_slots;  /* 0 → default 3; 1 = single-slot legacy. */
	int n_text_slots;   /* 0 → default 16; 1 = single-slot legacy. */
};

/* Equivalent to sam3_init() with opts == NULL. */
sam3_ctx *sam3_init_ex(const struct sam3_cache_opts *opts);

enum {
	SAM3_CACHE_IMAGE = 1 << 0,
	SAM3_CACHE_TEXT  = 1 << 1,
};
void sam3_cache_clear(sam3_ctx *ctx, unsigned which);

struct sam3_cache_stats {
	uint64_t image_hits, image_misses, image_evictions;
	uint64_t text_hits,  text_misses,  text_evictions;
};
void sam3_cache_stats(const sam3_ctx *ctx, struct sam3_cache_stats *out);
```

`sam3_init()` is unchanged (calls `sam3_init_ex(NULL)` internally).
All existing call sites of `sam3_set_image` / `sam3_set_text` /
`sam3_segment` work without modification.

## Error Handling

- **Arena-full on encode into a cache slot.** Should not occur if slot
  arenas are sized correctly, but if it does: log a warning, evict *all*
  image slots, retry once with a freshly-reset arena. If that also fails,
  return `SAM3_ENOMEM`.
- **Hash collision.** FNV-1a 64-bit collisions are astronomically
  unlikely for the input sizes involved, but to guard against adversarial
  inputs and test bugs: on hash match, verify dimensions + a 256-byte
  prefix of the input matches the cached entry's recorded prefix. On
  mismatch, treat as a miss and evict the colliding entry.
- **`n_image_slots > 16`.** Clamp to 16 and emit a `sam3_log_warn`.
- **Cache miss during worker execution.** Handled transparently — the
  worker writes into its pre-assigned slot regardless of cache state.

## Memory Footprint

- Image cache: `n_image_slots × slot_arena_size`. Worst case for SAM3.1
  at 1152 img_size and Hiera backbone: neck + FPN scales + sam2_neck
  scales ≈ 40 MiB. Default `n = 3` → ~120 MiB headroom.
- Text cache: `n_text_slots × 77 × d_model × 4 bytes`.
  `16 × 77 × 512 × 4 = 2.4 MiB`. Negligible.

Both budgets are deterministic (no heap fragmentation).

## Testing

New file `tests/test_feature_cache.c`:

1. **`test_image_cache_hit`** — encode A, encode B, re-encode A; assert
   `stats.image_hits == 1` and resulting `cached_image_features->data` is
   bit-identical to the first A encode.
2. **`test_image_cache_lru_eviction`** — `n_slots = 2`; encode A, B, C;
   re-encode A (miss, evicts B); re-encode C (hit).
3. **`test_text_cache_hit`** — `sam3_set_text("cat")`; segment; call
   `sam3_set_text("cat")` again; assert `stats.text_hits == 1` and the
   worker was not spawned (`text_thread_active` never transitioned to 1
   the second time — verified via a test hook that counts spawns).
4. **`test_text_cache_hit_across_different_prompts`** — cat → dog → cat;
   assert final cat is a hit.
5. **`test_cache_clear`** — populate, clear, re-lookup → miss.
6. **`test_cache_collision_safety`** — test hook to force identical
   hashes for two distinct pixel buffers; verify no false hit.
7. **`test_cache_disabled`** — `n_image_slots = 1`; run the existing
   single-image integration test; assert behavior unchanged.
8. **`test_cache_load_model_flushes`** — populate, call
   `sam3_load_model()` again; verify caches are empty.

Existing tests (`test_sam3_*`, `test_metal_*`) double as regression
coverage — they use `sam3_init()` (default slots) and must continue to
pass unchanged.

## Performance Expectations

| Operation                        | Today     | Hit path       | Notes                        |
|----------------------------------|-----------|----------------|------------------------------|
| `set_image` (cached)             | 100–500ms | ~10 ms (hash)  | 10–50× speedup               |
| `set_text` (cached)              | 30–80 ms  | < 100 µs       | Worker not spawned           |
| `segment` after cached set_image | unchanged | unchanged      | No change to decoder path    |

## Rollout

Single PR, gated behind default-enabled cache. Callers that want legacy
single-slot behavior pass `{ .n_image_slots = 1, .n_text_slots = 1 }` to
`sam3_init_ex`.

## File-Level Impact Summary

- **New:** `src/model/feature_cache.{h,c}`, `tests/test_feature_cache.c`
- **Modified:**
  - `include/sam3/sam3.h` — add API.
  - `src/sam3.c` — plumb `sam3_init_ex`, `sam3_cache_clear`,
    `sam3_cache_stats`; flush caches in `sam3_load_model`.
  - `src/model/sam3_processor.h` — add cache pointers, remove
    `weights_end`, `text_features_async`, `text_persist_arena`.
  - `src/model/sam3_processor.c` — replace the `weights_end` rollback in
    `set_image` with cache lookup; replace `text_features_async`
    consumption in `segment` with cache/bundle consumption; update
    worker to write into cache arena.
- **Unchanged:** `sam3_image_model` internals, all backend code, video
  tracker (separate frame cache).
