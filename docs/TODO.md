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

## 7. Hash table pre-sizing from graph node count

**File:** `src/backend/metal/metal_backend.c:120-136`
**Impact:** Low (avoids 1-2 rehashes on large graphs)
**Complexity:** Low

### Problem

The tensor map starts at 8192 slots and rehashes at 75% load. A large
ViT graph (32 layers x ~10 nodes/layer = ~320 nodes) plus all weight
tensors can reach 1000+ entries. If weights are counted, the initial
8192 is likely sufficient, but edge cases with many intermediates could
trigger a rehash.

### Approach

Accept an optional size hint from `graph_eval`:

```c
static void metal_map_ensure_capacity(struct sam3_metal_backend *mtl, int hint)
{
    int needed = hint * 4 / 3; // account for 75% load factor
    while (mtl->map_capacity < needed)
        metal_map_rehash(mtl);
}
```

Call at the start of `metal_graph_eval` with `g->n_nodes * 3` (rough
upper bound: each node may wrap ~3 input tensors).

### Tasks

- [ ] Verify whether rehash actually occurs (add debug logging)
- [ ] If rehash occurs in practice, add pre-sizing hint

---

## Priority order

1. ~~**RoPE** (item 1)~~ — **DONE** (mlx_fast_rope_dynamic split-axis)
2. ~~**SiLU fusion** (item 3)~~ — **DONE** (fused Metal kernel via mlx_fast_metal_kernel)
3. ~~**Q8_0 GPU dequant** (item 4)~~ — **DONE** (custom Metal kernel via mlx_fast_metal_kernel)
4. ~~**SDPA mask cache** (item 2)~~ — **DONE** (stack-local 4-slot cache)
5. ~~**Multi-stream** (item 5)~~ — **DONE** (GPU-resident forwarding via no_readback)
6. **Hash pre-sizing** (item 7) — only if rehash is confirmed
7. ~~**SiLU constant** (item 6)~~ — **MOOT** (fused kernel eliminates intermediate)

## Next steps

Before implementing any optimization:
1. **Profile a full inference** with timing around each graph_eval phase
2. **Count C-API allocations** per inference (add a counter to `mlx_array_new`)
3. **Identify the actual bottleneck** — is it C-API overhead, GPU idle
   time, or GPU compute?
