# SAM3 Performance

## Inference Profile

Measured on Apple M3 Pro, Release build (`-O2`), Metal backend, single run.

**Model:** sam3.sam3 (3.3 GB, 1787 tensors)
**Input:** `assets/cat.jpeg` (1008x1008 after resize), 1 text prompt ("cat")
**Output:** 1 mask, IoU 0.6331, box=[0,54,180,282] (default NMS)

Top-level stages sum to wall-clock; indented sub-stages are nested
inside their parent and are not added to the total.

| Stage              | Time (ms) |    % |
|--------------------|----------:|-----:|
| model_load         |      1046 | 11.5 |
| image_normalize    |         2 |  0.0 |
| image_encode       |      6916 | 76.3 |
| &nbsp;&nbsp;vit_precompute  |        58 |  0.6 |
| &nbsp;&nbsp;vit_patch_embed |        36 |  0.4 |
| &nbsp;&nbsp;vit_blocks      |      6371 | 70.3 |
| &nbsp;&nbsp;neck            |       375 |  4.1 |
| text_encode (join) |     <0.1  |  0.0 |
| mask_decode        |       894 |  9.9 |
| &nbsp;&nbsp;geometry_encode |      <0.1 |  0.0 |
| &nbsp;&nbsp;encoder_fusion  |       380 |  4.2 |
| &nbsp;&nbsp;decoder         |       215 |  2.4 |
| &nbsp;&nbsp;seg_head        |       292 |  3.2 |
| postprocess        |        20 |  0.2 |
| **Total (wall)**   |  **9065** |      |

The `text_encode` stage measures only the pthread join — the actual text
encoder graph runs on a CPU worker thread in parallel with the Metal
image encoder, so its ~3.7s of work is fully hidden behind `image_encode`.
See **Async pipeline** below.

Wall-time variance: `image_encode` varies ±20% between runs due to thermal
and GPU contention. Median of 5 Release runs: 9.07s (range 6.78–10.40s).
The async pipeline saves a fixed ~3.7s vs the synchronous baseline
regardless of which image-encode sample you pick.

### Optimization History

Comparison of the four critical Metal inference stages over time:

| Version | vit_blocks | text_blocks | mask_decode | neck | Sum |
|---------|-----------:|------------:|------------:|-----:|----:|
| Baseline (per-head SDPA, F32, per-block eval)         | 8419 | 5064 | 1639 | 499 | 15621 |
| + Fused MHSA + F16 + 4-block batching (broken accuracy) | 6087 | 3090 | 1358 | 353 | 10888 |
| + Correct seq/head physical transpose (fixed)         | 5756 | 3747 | 1245 | 428 | **11176** |
| + Async image ∥ text encoding pipeline                | 6371 |    0 |  894 | 375 |  **7640** |
| Speedup vs baseline                                   | -24% |-100% | -45% | -25% | **-51%** |

The async row hides `text_blocks` entirely behind `vit_blocks` on a CPU
worker thread (MLX-C 0.6 has a non-thread-safe process-wide Metal device
cache, so the worker is pinned to the CPU backend). The image encoder
still owns Metal; the `text_encode` stage timer now measures only the
pthread join, which is <0.1 ms in every run because the CPU encoder
finishes well before the 6-second image encoder.

The previous "10888 ms" row used a naive 4D reshape that scrambled the
multi-head SDPA inputs, producing wrong outputs. The fix replaces it with
a 3-step `[seq, n_heads, head_dim] → permute(1,0,2) → [1, n_heads, seq,
head_dim]` pattern, which adds an explicit transpose op per attention
layer. The transpose costs back ~290 ms across the four stages but the
output is now correct (`IoU 0.6331` matches the F32 baseline within F16
rounding).

The four landed optimizations:

1. **Fused multi-head SDPA** — Replaced the per-head Q/K/V slice + per-head
   `mlx_fast_scaled_dot_product_attention` loop in `gh_multihead_attention_*`
   with a single batched 4D SDPA call. ~65 graph nodes per attention layer
   collapses to ~8, cutting total attention nodes from ~3640 to ~448 across
   ViT + text encoder.
2. **F16 compute** — `metal_wrap_tensor` casts F32 leaf inputs to F16 at
   wrap time and `metal_graph_eval` Phase 3 casts F16 results back to F32
   on host readback. Halves Metal memory bandwidth; MLX kernels keep F32
   accumulators internally for numerical safety.
3. **Adaptive multi-block batching** — `image_encoder.c` and `text_encoder.c`
   build up to 4 transformer blocks into a single graph before
   `graph_eval`, reducing GPU sync points from 56 to ~14. Per-block arena
   cost is tracked and the batch is flushed early if the next block would
   not fit, preserving correctness for very large inputs.
4. **Async image ∥ text encoding pipeline** — `sam3_set_text()` spawns a
   pthread worker that runs the text encoder on a second CPU-backed
   processor instance with its own arenas, in parallel with the main
   thread's Metal image encoder. `sam3_segment()` joins the worker and
   consumes the pre-computed text features. Worker stack is 8 MiB
   (MLX-C needs more than the 512 KiB pthread default on macOS). The
   CPU text encoder finishes well before the Metal image encoder, so the
   join is effectively free (<0.1 ms measured). Saves ~3.7 s wall-clock.

### Stage Details

**model_load** (~61ms)
Open `.sam3` weight file via mmap, build FNV-1a hash table for O(1) tensor
lookup, initialize module structs, and point tensor data pointers into the
mmap region. RoPE tables, window mask, position embed tiling, and 2D
sinusoidal position encoding are deferred to first inference. No weight data
is copied during load — tensors reference the mmap'd region directly.

**image_encode** (~6.3s, 51%)
- **vit_precompute** (~43ms): Lazy first-call initialization of RoPE tables
  (window + global), 5184x5184 window attention mask, tiled position
  embeddings (577 -> 72x72), and 2D sinusoidal position encoding.
- **vit_patch_embed** (~27ms): 14x14 conv2d patch embedding, absolute
  position embedding addition, and ln_pre LayerNorm. Single graph eval.
- **vit_blocks** (~5.8s, 46%): 32 ViT-H transformer blocks (1280-dim, 16
  heads). Up to 4 blocks are batched per `graph_eval` call, with an
  adaptive fallback that reduces the batch size if arena capacity runs
  low (important for large 72x72 patch grids). Each block: LayerNorm,
  fused-multi-head self-attention with RoPE (windowed or global),
  residual, LayerNorm, GELU MLP, residual.
- **neck** (~428ms, 3.5%): 4-scale FPN producing feature maps at 4x, 2x,
  1x, and 0.5x resolution via conv2d, transposed conv2d, and maxpool stages.

**text_encode** (<0.1 ms join — real work ~3.7 s, fully overlapped)
- **tokenize** (<0.1 ms): BPE tokenization of the text prompt, still on
  the caller thread before spawning the worker.
- **text_blocks** (~3.7 s, off the critical path): CLIP text encoder
  (24-layer transformer, 1024-dim, 16 heads) runs on a CPU worker
  backend. Up to 4 blocks are batched per `graph_eval` call. Each block:
  LayerNorm, fused-multi-head self-attention with causal mask, residual,
  LayerNorm, GELU MLP, residual. Wall-clock time is hidden behind
  `image_encode`; the profiler stage timer measures only the pthread
  join (<0.1 ms because the worker is already done).

**mask_decode** (~1.2s, 10%)
- **geometry_encode** (<0.1ms): 3-layer geometry transformer for point/box
  prompts (CPU path, typically 2 tokens). Trivial for single-prompt inputs.
- **encoder_fusion** (~617ms, 5.0%): 6-layer DETR encoder fusing image
  features with text/geometry context. Each layer: self-attention +
  cross-attention + FFN, evaluated as separate Metal graphs.
- **decoder** (~225ms, 1.8%): 6-layer DETR decoder with 200 learned queries.
  Each layer: self-attention, text cross-attention, vision cross-attention,
  FFN, with iterative box refinement between layers.
- **seg_head** (~396ms, 3.2%): FPN pixel decoder (3-stage upsampling to
  288x288), instance projection (1x1 conv), mask embedder MLP, dot-product
  mask logits, and objectness scoring.

**postprocess** (~19ms, 0.2%)
Stability-based mask selection, bounding box extraction, and NMS
(200 -> 1 masks for this input).

## Model Load Optimization History

| Version | model_load | Notes |
|---------|------------|-------|
| Baseline (mmap + hash table) | ~7.7s | Initial mmap-based loader |
| + mmap-direct tensors (`gh_load_mmap`) | ~1.1s | Eliminated ~3.3 GB memcpy |
| + lazy precompute + sep Q/K/V + FNV cache | ~60ms | Deferred RoPE/mask/pos_embed to first inference; eliminated ~700 MB QKV fusion memcpy; pre-stored FNV-1a hashes |
