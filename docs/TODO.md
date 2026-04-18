# Metal Backend Optimization TODO

Status legend: `[ ]` planned, `[~]` in progress, `[x]` done

---

## 1. ~~Replace manual RoPE decomposition with `mlx_fast_rope`~~ [x] DONE

**Resolution:** Replaced with `mlx_fast_rope_dynamic` using a split-axis
approach. MLX's `freqs` parameter is 1D-only (`[dims/2]`), ruling out a
single-call solution for 2D axial RoPE. Instead:

1. Added `grid_w` and `scale` params to `gh_rope()` interface, stored in
   graph node `params[1]`/`params[2]`.
2. When `grid_w > 0` (axial RoPE), the Metal backend:
   - Builds per-token integer offset arrays from grid geometry
   - Slices input `[B,S,H,HD]` into x-half and y-half
   - Reshapes to `[B*S, H, 1, HD/2]` (each token becomes its own batch
     element with T=1, so MLX's per-batch offset gives per-token positions)
   - Applies `mlx_fast_rope_dynamic` to each half with `traditional=true`,
     `base=10000.0`, and the fractional position scale
   - Concatenates results
3. When `grid_w == 0`, the legacy manual decomposition is preserved.

**Files changed:**
- `src/model/graph_helpers.h` / `.c` — new `grid_w, scale` params
- `src/model/image_encoder.c` — passes grid geometry
- `src/model/text_encoder.c`, `tests/test_vit.c` — pass `0, 0.0f`
- `src/backend/metal/metal_backend.c` — new fast path in `SAM3_OP_ROPE`
- `tests/test_metal.c` — 3 correctness tests (axial, scaled, batched)

**Measured:** ~5 `mlx_array` allocs per dispatch (down from 14). Warm
inference 8.1–9.4s on M-series (no regression vs baseline).

---

## 2. ~~Cache reshaped SDPA mask~~ [x] DONE

**Resolution:** Added a 4-slot stack-local mask reshape cache in
`metal_graph_eval`, keyed by `sam3_tensor*`. On cache hit, the SDPA
case reuses the existing `[1, 1, seq_q, seq_kv]` mlx_array instead
of calling `mlx_reshape`. Eliminates ~23-29 C-API allocations per
inference (text encoder: 24->1, decoder: up to 6->1).

**Files changed:**
- `src/backend/metal/metal_backend.c` — cache struct, dispatch plumbing,
  SDPA lookup/store, post-loop cleanup
- `tests/test_metal.c` — `test_metal_sdpa_mask_cache` (2-node shared mask)

### Tasks

- [x] Profile mask reshape cost (likely negligible — validate before implementing)
- [x] Implement small direct-mapped mask cache if profiling shows measurable cost
- [x] Ensure cache invalidation on graph_eval boundary

---

## 3. ~~SiLU fusion: check for native MLX primitive~~ [x] DONE

**Resolution:** No `mlx_silu` exists in MLX-C. Replaced the 2-op approach
(`mlx_sigmoid` + `mlx_multiply`) with a single fused Metal kernel via
`mlx_fast_metal_kernel`. The kernel is created once at backend init and
reused for all 38 SiLU dispatches per inference (ViT 32x + decoder 6x).

The custom kernel computes `out[i] = x[i] / (1 + exp(-x[i]))` using
Metal's `metal::exp` intrinsic with a dtype template parameter (`T`) for
F16/F32 support. MLX auto-generates the `[[kernel]]` signature from the
source body, including `thread_position_in_grid` detection.

**Files changed:**
- `src/backend/metal/metal_backend.h` — added `silu_kernel` field
- `src/backend/metal/metal_backend.c` — kernel creation at init, fused
  dispatch in `SAM3_OP_SILU`, cleanup in `metal_free`
- `tests/test_metal.c` — 2 correctness tests (8-element + 1024-element)

**Benefit:** Eliminates 1 temporary `mlx_array` allocation and 1 C-API
call per SiLU dispatch (38 per inference = 38 fewer allocations). Single
kernel launch instead of 2 ops that may or may not be fused by MLX's
graph compiler.

### Tasks

- [x] Profile SiLU: is MLX fusing sigmoid + multiply? (check kernel trace)
- [x] If not fused, implement custom Metal kernel via `mlx_fast_metal_kernel`
- [x] Benchmark: custom kernel vs current 2-op approach

---

## 4. ~~GPU-side Q8_0 dequantization~~ [x] DONE

**Resolution:** Replaced CPU NEON dequantization with a custom Metal kernel
via `mlx_fast_metal_kernel`. Raw Q8 blocks are uploaded as `MLX_UINT8` byte
buffers and dequantized on-device to F16. The kernel reads the 36-byte block
layout (`{float scale; int8_t data[32]}`) and computes
`half(data[elem]) * half(scale)` per thread. Eliminates CPU dequant time
and reduces host→GPU data transfer (raw Q8 bytes instead of expanded F16).

**Files changed:**
- `src/backend/metal/metal_backend.h` — added `dequant_q8_kernel` field
- `src/backend/metal/metal_backend.c` — kernel init, GPU dequant path in
  `metal_wrap_tensor`, cleanup in `metal_free`; removed old CPU NEON
  `metal_dequant_q8_to_f16` function and `core/half.h` dependency
- `tests/test_metal.c` — `test_metal_dequant_q8_gpu` correctness test

### Tasks

- [x] Profile cold-start dequant time (CPU NEON path)
- [x] Implement Metal dequant kernel via `mlx_fast_metal_kernel`
- [x] Upload raw Q8 blocks as byte buffer, dispatch dequant on GPU
- [x] Validate F16 output matches CPU NEON path (test_metal Q8 tests)

---

## 5. ~~Multi-stream pipelining~~ [x] DONE (GPU-resident forwarding)

**Resolution:** Replaced the multi-stream approach with GPU-resident
forwarding via `no_readback` flag on `sam3_graph`. When set, Phase 3
readback is skipped entirely — output mlx_arrays stay in the tensor
map. The next `graph_eval` finds them via `metal_wrap_tensor` without
data transfer (no GPU→host readback, no host→GPU re-upload).

Applied to the ViT block batch loop: 7 of 8 batches skip readback,
eliminating ~74 MB of round-trip data transfer (7 × 10.6 MB) for
the [5184, 1024] residual stream tensor.

**Files changed:**
- `src/core/graph.h` — `no_readback` field on `sam3_graph`
- `src/backend/metal/metal_backend.c` — no_readback path in Phase 3,
  forward-edge fix in Phase 1.5, update-aware `metal_map_put`
- `src/model/image_encoder.c` — persistent forwarding tensor, skip
  readback for non-last ViT batches
- `tests/test_metal.c` — 3 correctness tests (forward, chain, map update)

### Tasks

- [x] Implement `no_readback` flag on `sam3_graph`
- [x] Make `metal_map_put` handle updates (overwrite existing key)
- [x] Fix Phase 1.5 intermediate detection (forward-edge only)
- [x] Add no_readback path in Phase 3 of `metal_graph_eval`
- [x] Integrate into ViT builder (persistent forwarding tensor)
- [x] Correctness tests (basic forwarding, multi-hop chain, map update)

---

## 6. ~~Constant caching for SiLU intermediate~~ [x] MOOT

**Resolution:** Item 3 replaced SiLU with a fused Metal kernel, eliminating
the temporary `mlx_array` for `sigmoid(x)`. No intermediate to cache.

---

## 7. ~~Hash table pre-sizing from graph node count~~ [x] DONE

**Resolution:** Added `metal_map_ensure_capacity()` that grows the hash
table up front before Phase 1 dispatch, avoiding mid-dispatch rehashes
on large graphs. Called from `metal_graph_eval` with `g->n_nodes * 3`
as the hint (each node wraps ~3 input tensors), accounting for the 75%
load factor. Existing entries (weights already cached across calls) are
included in the capacity calculation so cumulative growth stays correct.

**Files changed:**
- `src/backend/metal/metal_backend.c` — `metal_map_ensure_capacity` helper
  and pre-size call at the top of `metal_graph_eval`
- `tests/test_metal.c` — `test_metal_map_presize` (64-node ADD chain
  exercises the capacity-check path)

### Tasks

- [x] Verify whether rehash actually occurs (add debug logging)
- [x] If rehash occurs in practice, add pre-sizing hint

---

# Video Pipeline (Hiera) Optimizations

The following items target the video tracking path (`sam3_video_propagate`)
where Hiera-Large is ~3.5 s/frame end-to-end (2.3 s encode + 1.2 s
tracker). Each item is quality-preserving — no algorithmic change and
no dropped layers. Identified by profiling the code on
`bench/video-expansion` (commit f69f77c). Concrete numeric wins depend
on the profiler run landing in BENCHMARK.md.

---

## 8. ~~Remove F16→F32→F16 round-trip in Metal matmul~~ [x] DONE

**Resolution:** The `#ifdef SAM3_METAL_F32_MATMUL` astype round-trip was
dead code — not enabled in any build configuration. It was also redundant:
MLX's steel GEMM kernel uses `AccumType = float` unconditionally (see
`mlx/backend/metal/kernels/steel/gemm/transforms.h:57-58` and
`.../mma.h:446-478`), so `mlx_matmul` with F16 operands and F16 output
already accumulates partial sums in F32 internally. No knob needed.

Removed the compile-time flag + astype branch. The runtime F32-compute
fallback for correctness debugging already exists via `SAM3_METAL_F32=1`,
which keeps tensors F32 end-to-end (no casts, no mixed precision).

BF16 operand path is deferred — it requires BF16 weight storage support
and a cosine-diff sweep, both of which are separate work items.

**Files changed:**
- `src/backend/metal/metal_backend.c` — dropped `#ifdef` block, single
  direct `mlx_matmul` call with an explanatory comment
- `CMakeLists.txt` — removed `SAM3_METAL_F32_MATMUL` option handling
- `tests/test_metal.c` — added `test_metal_matmul_large_k_parity`
  (K=1024 cosine-diff vs CPU F32 reference, asserts cos > 0.9999)

### Tasks

- [x] Confirm MLX already accumulates F16 matmul in F32 (verified in source)
- [x] Drop dead `#ifdef SAM3_METAL_F32_MATMUL` branch
- [x] Remove CMake option
- [x] Add F16-vs-F32 cosine parity test
- [ ] BF16 path — deferred to a separate item (needs BF16 weight storage)

---

## 9. ~~Spill→backend promotion via memcpy (no re-encode)~~ [x] DONE

**Resolution:** `spill_slot_to_cpu` now snapshots each tensor's header
(dtype, n_dims, dims, strides, nbytes) into new `spill_hdr_*` fields
on the cache slot alongside the raw byte copy. On a spill hit,
`sam3_frame_cache_get` takes a new `spill_promote` fast path that
builds fresh tensors in the backend arena via `sam3_tensor_clone_persist`
(pointing each temporary header at the matching spill buffer), byte-
copying straight from the host spill into the arena without invoking
the encoder. `n_spill_promotes` now reflects genuine memcpy-only
promotions. Falls back to the full encode path if any required spill
buffer is NULL (e.g. was dropped to TIER_NONE under spill pressure).

The retry ladder in `sam3_frame_cache_get` gained a last-resort
`drain_backend_tier` step. The bump arena cannot reclaim individual
slots, so a budget-sized `make_backend_room` can leave the offset
stuck near capacity even after partial LRU eviction; the new helper
evicts every remaining backend slot (to spill when budget allows)
so the arena can reset before the final retry. This makes the
eviction path actually reachable under tight budgets — a latent bug
that blocked tests and would surface in production once the 4 GiB
arena filled.

**Files changed:**
- `src/model/frame_cache.h` — `spill_hdr_image_features` / `_feat_s0` /
  `_feat_s1` / `_feat_4x` fields on `sam3_frame_cache_slot`; updated
  top-of-file docstring to describe memcpy promotion
- `src/model/frame_cache.c` — header snapshot in `spill_slot_to_cpu`;
  new `spill_promote` + `drain_backend_tier` helpers; rewritten retry
  ladder in `sam3_frame_cache_get`; docstring refreshed
- `tests/test_frame_cache.c` — `test_spill_promote_via_memcpy` (evicts
  frame 1, mutates its contents first, re-accesses and asserts
  `n_spill_promotes` bumped, `g_encode_calls` unchanged, byte-for-byte
  match across three tensors); mock `g_encode_calls` moved to the
  success path so retry counts don't inflate the LRU test's assertion

---

## 10. ~~Share flattened `feat_s1` across objects in one frame~~ [x] DONE

**Resolution:** Added a private `struct propagate_frame_ctx` that
carries the per-frame `sam3_frame_features`, the flattened `[HW, d]`
`img_2d` tensor, its grid dims, and the scratch-arena offset captured
immediately after the flatten. `propagate_one` builds this once per
frame (reset scratch → frame_cache_get → allocate+memcpy img_2d →
snapshot offset) and threads a pointer through `video_replay_obj_prompt`
→ `video_replay_stored_prompt` → `video_add_prompts_pipeline`, and
into `video_propagate_pure_tracking_obj`.

When the shared ctx is non-NULL the per-object functions skip
`sam3_frame_cache_get`, reuse `shared->img_2d`, and reset the scratch
arena to `shared->scratch_mark` (instead of 0) so per-object prompt
tokens and graph intermediates still recycle between iterations while
the shared flatten survives. Public callers (`sam3_video_add_points`,
`sam3_video_add_box`) pass NULL and keep the legacy per-call path.

On an 8-object clip this saves 7 redundant `cf.feat_s1->nbytes`
memcpys (≈ 1.3 MB each at 72×72×256 f16) plus the matching
arena churn every frame. If cache lookup or feat_s1 shape are
unexpected, `propagate_one` silently falls back to per-object flatten
— the fast path is an opt-in.

**Files changed:**
- `src/model/sam3_video.c` — new `propagate_frame_ctx` struct; threaded
  through four static helpers; `propagate_one` builds it; public
  wrappers pass NULL

### Tasks

- [x] Lift `feat_s1 → img_2d` flatten into `propagate_one`
- [x] Thread `img_2d` through `video_replay_obj_prompt` + `video_propagate_pure_tracking_obj`
- [ ] Confirm bench `video_per_frame_32f_8obj_fwd` improves

---

## 11. Increase ViT block batch size on Metal

**File:** `src/model/image_encoder.c:636-638`
**Impact:** Low-Med (5-10% off Hiera encode)
**Complexity:** Low

### Problem

Hiera dispatches its 32 blocks in batches of `batch=4` when
`skip_data=1` (Metal) and blocks fit in 256 MiB scratch — that's 8
`graph_eval` calls per frame. Each call has host-GPU sync overhead
(~0.3-0.4 ms on M4). With GPU-resident forwarding (item 5) the scratch
no longer needs to hold block intermediates, so batch can go higher.

### Approach

Raise batch to 8 on Metal when `skip_data=1` and validate the scratch
arena still fits. Fall back to 4 if arena grows beyond a configurable
threshold (e.g. 512 MiB). Measure `vit_blocks` stage timing before/after.

### Tasks

- [ ] Bump batch from 4 → 8 in the `skip_data` branch
- [ ] Guard with scratch-size check, log the chosen batch
- [ ] Benchmark cold + warm encode on Hiera / TinyViT / EfficientViT

---

## 12. Skip memory attention when bank has ≤ 1 entry

**File:** `src/model/tracker.c:464-485`
**Impact:** Low-Med (~30-50 ms/frame for first 6-8 frames of every clip)
**Complexity:** Low

### Problem

`sam3_tracker_track_frame` already has a fast path for `total_mem == 0`
(directly add `no_mem_embed`). For `total_mem == 1` (just the
conditioning frame in the bank), the full 4-layer RoPE cross-attention
runs over a 1-token memory — mostly launch overhead. The first ~6-8
frames of any clip are in this low-token regime before the bank
saturates.

### Approach

Add a branch for small `total_mem` that collapses to: project the single
memory token into d_model and add to the backbone features, skipping
the transformer entirely. Confirm parity with the 4-layer path at
`total_mem == 1` (cosine diff vs reference on that specific
configuration). If parity fails, fall back to the full path — the fast
path is an opt-in.

### Tasks

- [ ] Profile: what fraction of frames in a 64f clip have `total_mem < 2`?
- [ ] Implement short-memory branch with a compile-time fallback
- [ ] Cosine-diff test: single-memory mode vs reference

---

## 13. Avoid 4× tensor clone in `session_encode_frame`

**File:** `src/model/sam3_video.c:339-345`
**Impact:** Low-Med (~3-8 ms/frame on M4; ~30 MB memcpy)
**Complexity:** Medium

### Problem

After the image encoder writes feature maps into `ctx->proc.model_arena`,
`session_encode_frame` clones all four scales (image_features, feat_s0,
feat_s1, feat_4x) into the frame cache's backend arena with
`sam3_tensor_clone_persist`. The encoder's model_arena is then rolled
back. This is ~30 MB of memcpy per frame that could be a slab handoff.

### Approach

Option A: target the encoder's output allocations directly at the
cache backend arena (requires passing the arena down through
`sam3_image_model_encode`).

Option B: keep the cache backend arena and model arena aliased for
per-frame outputs and transfer ownership by bumping the arena's "start"
pointer past the new features.

Both need careful lifetime auditing — the scratch reset at line 287-297
must not clobber the cache-owned region.

### Tasks

- [ ] Audit which arena the Hiera encoder's final tensors live in
- [ ] Prototype slab-handoff path; keep clone path behind a feature flag
- [ ] Verify frame-cache LRU eviction still frees correctly

---

## 14. Fuse window_partition + SDPA + window_unpartition

**File:** `src/model/image_encoder.c:733-792`
**Impact:** Low-Med (56 fewer kernel launches per Hiera frame)
**Complexity:** Medium-High

### Problem

Each of the 28 windowed blocks runs `layernorm → window_partition →
MHA (which itself is qkv + reshape + RoPE + SDPA + out-proj) → reshape
→ window_unpartition → residual-add`. `window_partition` and
`window_unpartition` are pure index shuffles on NHWC — they exist only
to reshape the sequence dim before/after attention.

### Approach

Add a `gh_windowed_attention` graph op that accepts the full ViT
features + window size and handles partition + SDPA + unpartition in a
single node. Metal backend can implement as: reshape views (free) +
one `mlx_fast_scaled_dot_product_attention` call with the windowed
layout — no materialized intermediates.

### Tasks

- [ ] Add `gh_windowed_attention` helper in `graph_helpers.{h,c}`
- [ ] Implement Metal fast path (views-only partition/unpartition)
- [ ] Update ViT block loop to call the fused op for non-global blocks
- [ ] Bitwise or near-bitwise parity test vs current path

---

## 15. Batch mask decoder + memory encoder across objects

**File:** `src/model/sam3_video.c:1812-1832` (object loop in `propagate_one`)
**Impact:** High on multi-object clips (N× → ~1.3× for decoder+encoder)
**Complexity:** High

### Problem

Today `propagate_one` runs the full decoder + memory encoder pipeline
sequentially per object: with 8 objects, the mask decoder graph is
built and evaluated 8 times per frame. Backbone features
(`feat_s1`/`feat_s0`/`feat_4x`) are shared across objects — only the
prompt tokens and per-object memory bank differ. The Python reference
stacks along the batch dim.

### Approach

Introduce a multi-object variant of `sam3_mask_decoder_build` that
consumes `[N, ...]` prompt tokens and returns `[N, H, W, 1]` masks,
then a matching memory encoder. Memory bank assembly stays per-object
but the graph eval is one dispatch.

### Tasks

- [ ] Design batched decoder signature (prompt stacking, per-obj bank handling)
- [ ] Implement batched memory encoder
- [ ] Benchmark `video_per_frame_32f_{2,4,8}obj_fwd` vs single-object loop

---

## 16. Reduce mask upsample factor for memory encoder input

**File:** `src/model/sam3_video.c:1549-1554`
**Impact:** Low-Med (mem-encoder inputs shrink 6.25× at 2× vs 5×)
**Complexity:** Low — **but requires parity diff before adoption**

### Problem

The propagation path upsamples the 240×240 decoder mask to
`interpol_h` (≈1152) per object per frame before the mem-encoder's
4-stage strided-conv cascade downsamples it right back. If `interpol_h`
is just matching the mem-encoder's expected input (not a deliberate
resolution knob), a 2× upsample saves 6.25× pixels on that path.

### Approach

Measure `interpol_h` / `final_h` on Hiera (likely 5×). Try 2× and
diff the resulting `mem_feat` tensors against the 5× path. Keep the
5× path if diff is non-negligible; adopt 2× if it's below noise.

### Tasks

- [ ] Dump `interpol_h` / `final_h` for Hiera / TinyViT / EfficientViT
- [ ] Add a compile-time knob to select upsample factor
- [ ] Parity diff against current path on a fixed clip
- [ ] Only merge if parity holds within tolerance

---

## Priority order

1. ~~**RoPE** (item 1)~~ — **DONE** (mlx_fast_rope_dynamic split-axis)
2. ~~**SiLU fusion** (item 3)~~ — **DONE** (fused Metal kernel via mlx_fast_metal_kernel)
3. ~~**Q8_0 GPU dequant** (item 4)~~ — **DONE** (custom Metal kernel via mlx_fast_metal_kernel)
4. ~~**SDPA mask cache** (item 2)~~ — **DONE** (stack-local 4-slot cache)
5. ~~**Multi-stream** (item 5)~~ — **DONE** (GPU-resident forwarding via no_readback)
6. ~~**Matmul precision policy** (item 8)~~ — **DONE** (dead-code cleanup; MLX already accumulates F16 matmul in F32)
7. ~~**Spill→backend memcpy** (item 9)~~ — **DONE** (memcpy promote via saved headers; drain_backend_tier unblocks the ENOMEM retry ladder)
8. ~~**Share `feat_s1`** (item 10)~~ — **DONE** (per-frame shared `img_2d` via `propagate_frame_ctx`)
9. **Skip memattn for small bank** (item 12) — quick win at clip start
10. **ViT block batch bump** (item 11) — easy after item 5 landed
11. **Batch objects in decoder** (item 15) — big on multi-obj
12. **Fuse windowed attention** (item 14) — complex but clean
13. **Avoid 4× clone** (item 13) — needs arena audit
14. **Upsample factor** (item 16) — only if parity holds
15. ~~**Hash pre-sizing** (item 7)~~ — **DONE** (ensure_capacity in graph_eval)
16. ~~**SiLU constant** (item 6)~~ — **MOOT** (fused kernel eliminates intermediate)

## Next steps

Before implementing any optimization:
1. **Profile a full inference** with timing around each graph_eval phase
2. **Count C-API allocations** per inference (add a counter to `mlx_array_new`)
3. **Identify the actual bottleneck** — is it C-API overhead, GPU idle
   time, or GPU compute?
4. **For video items 8-16:** rebuild with `-DSAM3_PROFILE=ON` and run
   `sam3_cli bench pipeline --filter "video_per_frame_*"` on
   `models/sam3.sam3` to split per-frame time across `video_encode_frame`,
   `tracker_build/eval`, `memenc_build/eval`, `mask_upsample`,
   `frame_cache_get`. Lock in which items 8-16 deserve the work based
   on the resulting breakdown.
