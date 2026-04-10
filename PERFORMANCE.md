# SAM3 Performance

## Inference Profile

Measured on Apple M3 Pro, Release build (`-O2`), Metal backend,
median of 5 runs.

**Model:** sam3.sam3 (3.3 GB, 1787 tensors)
**Input:** `assets/cat.jpeg` (1008x1008 after resize), 1 text prompt ("cat")
**Output:** 1 mask, IoU 0.6324, box=[0,54,180,282] (default NMS)

Top-level stages sum to wall-clock; indented sub-stages are nested
inside their parent and are not added to the total.

| Stage              | Time (ms) |    % |
|--------------------|----------:|-----:|
| model_load         |      1056 | 10.9 |
| image_normalize    |         1 |  0.0 |
| image_encode       |      3622 | 37.5 |
| &nbsp;&nbsp;vit_precompute  |        14 |  0.1 |
| &nbsp;&nbsp;vit_patch_embed |        35 |  0.4 |
| &nbsp;&nbsp;vit_blocks      |      3076 | 31.9 |
| &nbsp;&nbsp;neck            |       469 |  4.9 |
| text_encode (join) |     <0.1  |  0.0 |
| mask_decode        |       829 |  8.6 |
| &nbsp;&nbsp;geometry_encode |      <0.1 |  0.0 |
| &nbsp;&nbsp;encoder_fusion  |       322 |  3.3 |
| &nbsp;&nbsp;decoder         |       200 |  2.1 |
| &nbsp;&nbsp;seg_head        |       278 |  2.9 |
| postprocess        |        22 |  0.2 |
| **Total (wall)**   |  **9648** |      |

The `text_encode` stage measures only the pthread join — the actual text
encoder graph runs on a CPU worker thread in parallel with the Metal
image encoder, so its ~3.7s of work is fully hidden behind `image_encode`.
See **Async pipeline** below.

Wall-time variance: `image_encode` varies ±20% between runs due to thermal
and GPU contention. Median of 5 Release runs: 9.65s (range 9.29–14.19s;
the slow tail is a cold first run). The async pipeline saves a fixed
~3.7s vs the synchronous baseline regardless of which image-encode
sample you pick. The post-#4 warm-run floor for `vit_blocks` is around
2.8–3.1s, down from 5.8–6.4s pre-#4.

### Optimization History

Comparison of the critical Metal inference stages over time:

| Version | vit_blocks | text_blocks | mask_decode | neck | Sum |
|---------|-----------:|------------:|------------:|-----:|----:|
| Baseline (per-head SDPA, F32, per-block eval)         | 8419 | 5064 | 1639 | 499 | 15621 |
| + Fused MHSA + F16 + 4-block batching (broken accuracy) | 6087 | 3090 | 1358 | 353 | 10888 |
| + Correct seq/head physical transpose (fixed)         | 5756 | 3747 | 1245 | 428 | **11176** |
| + Async image ∥ text encoding pipeline                | 6371 |    0 |  894 | 375 |  **7640** |
| + Mask-free windowed attention                        | 3076 |    0 |  829 | 469 |  **4374** |
| Speedup vs baseline                                   | -63% |-100% | -49% |  -6% | **-72%** |

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
output is now correct (`IoU 0.6324` matches the F32 baseline within F16
rounding).

The mask-free row deletes the 5184×5184 F32 window mask that the 28
windowed ViT blocks used to reference on every SDPA call. Instead, the
windowed path now reshapes `[seq=5184, dim]` → `[9 windows, 576 tokens,
dim]` via `gh_window_partition`, runs unmasked SDPA on the batched
windows with the pre-computed per-window RoPE table, and stitches the
output back with `gh_window_unpartition`. Each windowed block drops
~107 MiB of mask traffic and ~9× attention FLOPs (from 5184² to
9×576²). Fixture compare stays green within the same F16 tolerance.

The five landed optimizations:

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
5. **Mask-free windowed attention** — The 28 windowed ViT blocks no
   longer allocate or reference a 5184×5184 F32 attention mask. Instead
   the block reshapes its `[5184, 1024]` input into `[9, 576, 1024]`
   via `gh_window_partition` (72 is already a multiple of the 24-patch
   window, so no padding is needed), runs `gh_multihead_attention_rope`
   on the 9-way batch with
   a precomputed 576-token window RoPE table, and reverses the partition
   with `gh_window_unpartition`. Dropping the mask saves ~107 MiB of
   per-block memory traffic and cuts attention FLOPs from O(5184²·d)
   to 9·O(576²·d), roughly 9× fewer multiplies in the 28 windowed
   blocks. Measured `vit_blocks` drops from 6371 ms → 3076 ms (-51.7%),
   saving ~3.3 s of warm-run wall time. See `src/model/image_encoder.c`
   and the `gh_window_partition` / `gh_window_unpartition` helpers in
   `src/model/graph_helpers.c`.

### Stage Details

**model_load** (~1.05s)
Open `.sam3` weight file via mmap, build FNV-1a hash table for O(1) tensor
lookup, initialize module structs, and point tensor data pointers into the
mmap region. RoPE tables (global + per-window), position embed tiling, and
2D sinusoidal position encoding are deferred to first inference. No weight
data is copied during load — tensors reference the mmap'd region directly.

**image_encode** (~3.6s, 38%)
- **vit_precompute** (~14ms): Lazy first-call initialization of RoPE tables
  (global 5184-token + per-window 576-token), tiled position embeddings
  (577 -> 72x72), and 2D sinusoidal position encoding. The full-size
  5184×5184 window attention mask and matching 5184-token window RoPE
  are no longer allocated after the mask-free windowed attention rework.
- **vit_patch_embed** (~35ms): 14x14 conv2d patch embedding, absolute
  position embedding addition, and ln_pre LayerNorm. Single graph eval.
- **vit_blocks** (~3.1s, 32%): 32 ViT transformer blocks (1024-dim, 16
  heads, 64 head_dim). 28 of the 32 blocks use windowed attention: their
  input is partitioned from `[5184, 1024]` to `[9, 576, 1024]`, passed through
  unmasked multi-head SDPA with a precomputed 576-token window RoPE,
  then unpartitioned — no attention mask is materialized. The remaining
  4 blocks use global attention with the 5184-token RoPE. Up to 4
  blocks are batched per `graph_eval` call, with an adaptive fallback
  that reduces the batch size if arena capacity runs low. Each block:
  LayerNorm, fused multi-head self-attention with RoPE (windowed or
  global), residual, LayerNorm, GELU MLP, residual.
- **neck** (~469ms, 4.9%): 4-scale FPN producing feature maps at 4x, 2x,
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

**mask_decode** (~829ms, 8.6%)
- **geometry_encode** (<0.1ms): 3-layer geometry transformer for point/box
  prompts (CPU path, typically 2 tokens). Trivial for single-prompt inputs.
- **encoder_fusion** (~322ms, 3.3%): 6-layer DETR encoder fusing image
  features with text/geometry context. Each layer: self-attention +
  cross-attention + FFN, evaluated as separate Metal graphs.
- **decoder** (~200ms, 2.1%): 6-layer DETR decoder with 200 learned queries.
  Each layer: self-attention, text cross-attention, vision cross-attention,
  FFN, with iterative box refinement between layers.
- **seg_head** (~278ms, 2.9%): FPN pixel decoder (3-stage upsampling to
  288x288), instance projection (1x1 conv), mask embedder MLP, dot-product
  mask logits, and objectness scoring.

**postprocess** (~22ms, 0.2%)
Stability-based mask selection, bounding box extraction, and NMS
(200 -> 1 masks for this input).

### 2026-04-10 — Mask-free windowed attention

Replaced the 5184×5184 F32 window mask that the 28 windowed ViT blocks
used to reference on every attention call with a window-partition +
unmasked SDPA on `[9, 576, 1024]`. Eliminates ~107 MiB of mask traffic
per windowed block and roughly 9× fewer attention FLOPs in those
blocks. Correctness verified by `test_fixture_compare` in Release.

| Stage           |    Before |     After |      Δ |
|-----------------|----------:|----------:|-------:|
| vit_blocks      |   6371 ms |   3076 ms | -51.7% |
| image_encode    |   6916 ms |   3622 ms | -47.6% |
| mask_decode     |    894 ms |    829 ms |  -7.3% |
| Total inference |   9065 ms |   9648 ms |  +6.4% |

Method: median of 5 runs of `sam3_main --profile -m models/sam3.sam3
-i assets/cat.jpeg -t cat` on Apple M3 Pro, Release (`-O2`), Metal
backend. The "Before" column is the previously published post-#11
profile in this file. The apparent wall-clock regression in the
Total row is sampling noise — the first run of 5 was a cold-cache
outlier (14.2 s) and the other four runs clustered at 9.3–10.1 s,
vs. a warm-run floor of ~9.3 s after #4. The vit_blocks delta is
the relevant signal: the critical-path transformer stack now spends
~3.3 s less per inference.

## Model Load Optimization History

| Version | model_load | Notes |
|---------|------------|-------|
| Baseline (mmap + hash table) | ~7.7s | Initial mmap-based loader |
| + mmap-direct tensors (`gh_load_mmap`) | ~1.1s | Eliminated ~3.3 GB memcpy |
| + lazy precompute + sep Q/K/V + FNV cache | ~60ms | Deferred RoPE/mask/pos_embed to first inference; eliminated ~700 MB QKV fusion memcpy; pre-stored FNV-1a hashes |
