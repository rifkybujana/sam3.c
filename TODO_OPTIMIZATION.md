# SAM3 Metal Optimization TODO

Remaining optimizations for the Metal inference path. CPU-only items
(#8 Q8 NEON, #9 SDPA threadpool) are tracked separately and excluded.

Baseline (Apple M3 Pro, F16, 4-block batching, fused MHSA+QKV — already
landed):

| Stage        | Time   | %    |
|--------------|-------:|-----:|
| image_encode | 6324ms | 51%  |
| &nbsp;&nbsp;vit_blocks  | 5756ms | 46% |
| &nbsp;&nbsp;neck        |  428ms | 3.5% |
| text_encode  | 3747ms | 30%  |
| mask_decode  | 1245ms | 10%  |
| **Total**    |**12.4s**|     |

Already shipped: #1 multi-block batching, #2 fused MHSA, #5 fused QKV,
#6 F16 compute, #7 debug-dump gating, **#11 async image ∥ text encoding
pipeline**. See PERFORMANCE.md for history.

---

## P0 — Highest Leverage

### #4. Mask-Free Windowed Attention

**Why:** 28 of 32 ViT blocks use windowed attention, and each one
references a 5184×5184 F32 mask (~107 MiB). The mask is materialized
once in `precompute_window_mask` but is then loaded into every
windowed-attention SDPA call across 28 blocks. Most of those bytes
encode the trivial structure "same window or not" — pure waste.

**Current state:**
- `src/model/image_encoder.c:135-165` `precompute_window_mask` builds
  the dense `[n_patches, n_patches]` F32 mask.
- `src/model/image_encoder.c:689-690` selects `vit->window_mask` for
  non-global blocks and passes it into the attention call.

**Approach:** reshape Q/K/V into windows so attention is naturally
local — no mask needed.

For window size `ws=14` and grid `gs=72`:
1. Reshape `[seq=5184, dim]` → `[gs/ws, ws, gs/ws, ws, dim]` =
   `[5, 14, 5, 14, dim]` (handle the 72/14 remainder via padding to
   70+2 or via the existing padding the Python ref uses).
2. Permute to `[gs/ws*gs/ws, ws*ws, dim]` = `[25, 196, dim]` (batch
   dim of 25 windows).
3. Run unmasked SDPA with `[25, n_heads, 196, head_dim]`.
4. Reverse the permute/reshape.

Eliminates the 5184×5184 mask entirely and shrinks attention from
`5184×5184` per block to `25 × 196×196` — same FLOPs, no mask traffic,
much smaller intermediate.

**Files to touch:**
- `src/model/image_encoder.c:135-165` — delete `precompute_window_mask`
  and the `vit->window_mask` field.
- `src/model/image_encoder.c:680-720` — gate the
  windowed-vs-global path; for windowed path, do the
  reshape→SDPA→inverse-reshape.
- `src/model/graph_helpers.c` (`gh_multihead_attention_rope`) — may
  need a variant that accepts pre-batched windows, or do the
  reshape outside the helper and call `gh_sdpa` directly.
- Verify that `vit_precompute` profile time drops from 43ms → ~0ms.

**Risks:**
- Padding: 72 is not a multiple of 14. Either round up to 84
  (5×14² windows but pad the input) or use the same uneven last-window
  trick the Python reference uses. Check what
  `facebookresearch/sam3` does.
- RoPE — windowed RoPE table is already precomputed per-window in
  `vit->rope_window`, so this should still work.

**Estimated impact:** 10-15% of vit_blocks → ~600-850ms saved →
12.4s → ~11.7s.
**Complexity:** Medium.

---

## P1 — Medium Leverage

### #3. Native NHWC Layout (Eliminate conv2d Transposes) ✅ SHIPPED

**Landed 2026-04-12.** Full model-level NHWC migration: patch embed,
neck, seg_head, mask decoder, CPU and Metal backends all consume NHWC
activations directly. Conv2d / convtranspose2d weights stored in OHWI
order in `.sam3` v3 (see [docs/weight-format.md](docs/weight-format.md)).
Removed -625 LOC of NCHW↔NHWC scaffolding.

Wall-clock delta is neutral (+0.3 %, within thermal noise) because MLX
already absorbs layout transposes internally on Apple Silicon. The
architectural benefits remain: natural MLX layout at the backend,
cleaner data flow, and future kernel work (Winograd, im2col, Metal-side
conv fusion) can assume NHWC inputs without pre-permutes. See
[PERFORMANCE.md](PERFORMANCE.md) for the stage delta table.

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
2. ~~**#3 native NHWC layout**~~ ✅ **SHIPPED** (neutral wall-clock,
   -625 LOC scaffolding, v3 `.sam3` with OHWI weights)
3. **#4 mask-free window attention** (P0, ~600-850ms, also frees
   ~107 MiB of arena pressure → enables larger batch in #1)
4. **#10 graph template reuse** (P1, small but cheap)
5. **#12 text KV cache** (P2, only if interactive workloads matter)
6. **#13 native Metal kernels** (P2, only if everything above lands
   and the next bottleneck is still kernel launch overhead)
7. **#14 sparse attention** (P2, speculative)

**Current total (post-#11):** 12.4s → ~9.1s median (-27%).
**Projected total after #4:** ~8.3s (-33%).
