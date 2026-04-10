# SAM3 Metal Optimization TODO

Remaining optimizations for the Metal inference path. CPU-only items
(#8 Q8 NEON, #9 SDPA threadpool) are tracked separately and excluded.

Baseline (Apple M3 Pro, F16, 4-block batching, fused MHSA+QKV,
async image∥text pipeline, **mask-free windowed attention** — all
already landed; median of 5 Release runs):

| Stage        | Time   | %    |
|--------------|-------:|-----:|
| image_encode | 3622ms | 38%  |
| &nbsp;&nbsp;vit_blocks  | 3076ms | 32% |
| &nbsp;&nbsp;neck        |  469ms | 4.9% |
| text_encode  |   <0.1ms | ~0% (hidden behind image_encode) |
| mask_decode  |  829ms | 8.6% |
| **Total**    |**9.6s** |     |

Already shipped: #1 multi-block batching, #2 fused MHSA, #5 fused QKV,
#6 F16 compute, #7 debug-dump gating, **#11 async image ∥ text encoding
pipeline**, **#4 mask-free windowed attention**. See PERFORMANCE.md for
history.

---

## P0 — Highest Leverage

### #4. Mask-Free Windowed Attention — ✅ SHIPPED

**Result:** `vit_blocks` dropped from 6371 ms → **3076 ms**
(-3295 ms, -51.7%) and `image_encode` from 6916 ms → 3622 ms
(-47.6%) on the `assets/cat.jpeg` + "cat" benchmark. Correctness
preserved — `test_fixture_compare` stays green in Release.

**What landed:**
- `precompute_window_mask` and the `vit->window_mask` field were
  deleted, along with the full-size 5184-token window RoPE.
- New `gh_window_partition` / `gh_window_unpartition` helpers in
  `src/model/graph_helpers.c` convert `[5184, dim]` ↔ `[9, 576, dim]`
  (9 windows of 24×24 tokens each, reshaped from the 72×72 grid).
- `gh_multihead_attention_rope` now handles batch > 1 correctly so
  the window dim can live in the leading axis.
- Each windowed ViT block partitions → unmasked SDPA with the
  precomputed 576-token window RoPE → unpartition. Global blocks
  still use the 5184-token RoPE unchanged.

**Savings per windowed block:** ~107 MiB of mask traffic eliminated
and attention FLOPs dropped from O(5184²·d) to 9·O(576²·d), roughly
9× fewer multiplies across 28 of the 32 ViT blocks. Measured impact
is substantially larger than the 10-15% / 600-850 ms originally
estimated because MLX's masked SDPA kernel wasn't exploiting the
mask sparsity — deleting the mask both removes bandwidth and unlocks
a much faster code path inside the SDPA kernel itself.

See `PERFORMANCE.md` optimization-history table and the dated entry
`2026-04-10 — Mask-free windowed attention` for full numbers.

---

## P1 — Medium Leverage

### #3. Native NHWC Layout (Eliminate conv2d Transposes)

**Why:** SAM3 stores tensors NCHW but Metal/MLX conv kernels prefer
NHWC. The neck and seg_head insert transposes around every conv2d.
`src/model/necks.c:374` does a single `gh_transpose` per scale to
shuffle ViT output `[n_patches, dim]` → `[1, dim, gs, gs]`, but the
4-stage neck and 3-stage seg_head accumulate transpose overhead.

**Current state:**
- `src/backend/cpu/kernels/cpu_conv2d.c:9` declares input as
  `[N, C, H, W]` (NCHW). Metal conv2d wraps the same convention.
- `src/model/necks.c:374` `gh_transpose` to NCHW; downstream conv ops
  inside the neck assume NCHW.
- `src/model/image_encoder.c:611` `gh_transpose` after the 14×14 patch
  conv2d to flatten back to `[np, dim]`.

**Approach:**
1. Decide on the canonical layout for image-side tensors. Recommended:
   keep NCHW in the public tensor type but teach the Metal backend's
   `metal_conv2d_op` to transpose internally on the GPU (cheap) or to
   call MLX conv2d in NHWC mode and absorb the transpose.
2. Alternatively (more invasive): switch the entire image encoder /
   neck / seg_head to NHWC and update every shape calculation.

**Recommendation:** start with the backend-internal transpose
absorption. Profile to confirm that `mlx_conv_general` with NHWC plus
GPU-side transpose is faster than the current explicit `gh_transpose`
ops. If yes, ship that. Only touch the model-level layout if the
backend approach doesn't recover the win.

**Files to touch:**
- `src/backend/metal/metal_backend.c` — `metal_conv2d_op` (search for
  it) — fold transposes into the MLX call.
- `src/model/necks.c` — possibly remove the `gh_transpose` at line 374
  if backend handles it.
- `src/model/image_encoder.c:611` — same.
- `src/model/segmentation.c` — seg_head transposes.

**Estimated impact:** 10-20% of (neck+seg_head) ≈ 80-165ms.
12.4s → ~12.25s. (Smaller absolute win but easy if the backend trick
works.)
**Complexity:** Medium (backend) or High (full NHWC migration).

---

### #10. Graph Template Reuse Across Blocks

**Why:** All 32 ViT blocks have identical graph structure, differing
only in weight pointers. Currently `image_encoder.c:668-773` calls
`sam3_graph_init(&g)` and rebuilds the graph from scratch for every
batch of 4 blocks. Same for the 24 text blocks.

**Current state:**
- `src/model/image_encoder.c:648` outer batching loop, `:668` graph
  init, `:773` graph_eval per batch.
- `src/model/text_encoder.c:562, 588-673` same pattern.

**Approach:**
1. Build the graph for one block (or one 4-block batch) once during
   `sam3_vit_init`.
2. Store a "template" with weight slots as `NULL` placeholders.
3. On each batch, clone the template and patch in the correct weight
   pointers — no re-walking of model code.

**Caveats:**
- Adaptive batching (#1) flushes early when arena fills up; the
  template needs variants for batch sizes 1, 2, 3, 4.
- Tensor IDs / arena offsets reset between batches; need to ensure
  the template doesn't bake in scratch arena addresses.

**Estimated impact:** 2-5% (graph build is small but × 14 batches ×
2 encoders = ~150-300ms).
**Complexity:** Medium.

---

## P2 — Long-Term

### #13. Native `.metal` Kernels (Replace MLX-C)

**Why:** MLX-C v0.6.0 wraps each op as a separate kernel launch with
its own command-encoder overhead. Custom Metal shaders can fuse
LayerNorm+residual, GELU+matmul, attention+output-projection into
single kernels — exactly the fusions the current architecture cannot
express.

**Current state:**
- `src/backend/metal/metal_backend.c` is the only Metal code; no
  `.metal` shaders exist.
- All ops dispatch through MLX-C primitives.

**Approach:** out of scope for a single sprint. This is a new backend.
If pursued, target the highest-frequency fusion opportunities:
1. LayerNorm + residual + linear (every transformer block has 2)
2. QKV projection + RoPE + reshape (ViT + text)
3. Softmax + dropout + matmul (attention output)

**Estimated impact:** 20-50% on the fused ops, but requires writing
correctness fixtures for every shader.
**Complexity:** Very high.

---

### #14. Sparse / Block-Sparse Attention

**Why:** Speculative. Some attention patterns may be sparse in
practice; block-sparse SDPA could reduce FLOPs.

**Status:** No measurements yet. Defer until other items land and
attention is profiled with a sparsity tracer.

**Complexity:** High.

---

### #12. Text Encoder KV Cache

**Why:** Skips the entire 3.7s text_encode for repeated prompts. Only
helps interactive sessions where the prompt is reused across multiple
images or multiple segments — single-shot inference gets nothing.

**Approach:** standard KV cache:
1. Hash the prompt token sequence.
2. On hit, reuse the cached `text_features` tensor directly.
3. On miss, run text_encode and cache the result.

**Files:**
- `src/model/sam3_processor.c:575-619` — cache lookup before
  `sam3_text_encoder_build_perblock`.
- New small struct in `sam3_processor` to hold up to N cached entries.

**Estimated impact:** 3.7s on cache hit, 0 on miss.
**Complexity:** Low-Medium.

---

## Suggested Order

1. ~~**#11 async pipeline**~~ ✅ **SHIPPED** (-3.7s wall, fully hides
   text_encode behind image_encode)
2. ~~**#4 mask-free window attention**~~ ✅ **SHIPPED** (-3.3s
   on `vit_blocks`, -51.7%; freed ~107 MiB of per-block arena
   pressure)
3. **#3 backend-side NHWC fold** (P1, easy if MLX cooperates)
4. **#10 graph template reuse** (P1, small but cheap)
5. **#12 text KV cache** (P2, only if interactive workloads matter)
6. **#13 native Metal kernels** (P2, only if everything above lands
   and the next bottleneck is still kernel launch overhead)
7. **#14 sparse attention** (P2, speculative)

**Current total (post-#4):** 12.4s → ~9.6s median (-23%). The warm
floor is ~9.3s; the median includes a cold first run. The biggest
remaining line item on the critical path is `vit_blocks` at 3.1s
(~32% of wall time); the next-largest stages are `neck` (469ms),
`mask_decode` (829ms), and `model_load` (1.05s, amortized across
repeated inferences).
**Projected total after #3 + #10:** ~9.1–9.3s median.
