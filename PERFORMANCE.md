# SAM3 Performance

## Inference Profile

Measured on Apple M3 Pro, Release build (`-O2`), Metal backend, single run.

**Model:** sam3.sam3 (3.3 GB, 1787 tensors)
**Input:** `assets/cat.jpeg` (1008x1008 after resize), 1 text prompt ("cat")
**Output:** 45 mask candidates, 288x288 (with `--nms-prob-thresh 0.0`)

| Stage            | Time (ms) |   % |
|------------------|----------:|----:|
| model_load       |      1106 | 9.5 |
| image_normalize  |         1 | 0.0 |
| image_encode     |      6641 |  57 |
|   vit_precompute |        48 | 0.4 |
|   vit_patch_embed|        73 | 0.6 |
|   vit_blocks     |      6087 |  52 |
|   neck           |       353 | 3.0 |
| text_encode      |      3090 |  27 |
|   tokenize       |      <0.1 | 0.0 |
|   text_blocks    |      3090 |  27 |
| mask_decode      |      1358 |  12 |
|   geometry_encode|      <0.1 | 0.0 |
|   encoder_fusion |       687 | 5.9 |
|   decoder        |       208 | 1.8 |
|   seg_head       |       451 | 3.9 |
| postprocess      |        22 | 0.2 |
| **Total**        | **11597** |     |

### Optimization History

Comparison of the four critical Metal inference stages over time:

| Version | vit_blocks | text_blocks | mask_decode | neck | Sum |
|---------|-----------:|------------:|------------:|-----:|----:|
| Baseline (per-head SDPA, F32, per-block eval) | 8419 | 5064 | 1639 | 499 | 15621 |
| + Fused MHSA + F16 compute + 4-block batching | 6087 | 3090 | 1358 | 353 | **10888** |
| Speedup                                       | -28% | -39% | -17% | -29% | **-30%** |

The three landed optimizations:

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

### Stage Details

**model_load** (~61ms)
Open `.sam3` weight file via mmap, build FNV-1a hash table for O(1) tensor
lookup, initialize module structs, and point tensor data pointers into the
mmap region. RoPE tables, window mask, position embed tiling, and 2D
sinusoidal position encoding are deferred to first inference. No weight data
is copied during load — tensors reference the mmap'd region directly.

**image_encode** (~6.6s, 57%)
- **vit_precompute** (~48ms): Lazy first-call initialization of RoPE tables
  (window + global), 5184x5184 window attention mask, tiled position
  embeddings (577 -> 72x72), and 2D sinusoidal position encoding.
- **vit_patch_embed** (~73ms): 14x14 conv2d patch embedding, absolute
  position embedding addition, and ln_pre LayerNorm. Single graph eval.
- **vit_blocks** (~6.1s, 52%): 32 ViT-H transformer blocks (1280-dim, 16
  heads). Up to 4 blocks are batched per `graph_eval` call, with an
  adaptive fallback that reduces the batch size if arena capacity runs
  low (important for large 72x72 patch grids). Each block: LayerNorm,
  fused-multi-head self-attention with RoPE (windowed or global),
  residual, LayerNorm, GELU MLP, residual.
- **neck** (~353ms, 3.0%): 4-scale FPN producing feature maps at 4x, 2x,
  1x, and 0.5x resolution via conv2d, transposed conv2d, and maxpool stages.

**text_encode** (~3.1s, 27%)
- **tokenize** (<0.1ms): BPE tokenization of the text prompt.
- **text_blocks** (~3.1s, 27%): CLIP text encoder (24-layer transformer,
  1024-dim, 16 heads). Up to 4 blocks are batched per `graph_eval` call.
  Each block: LayerNorm, fused-multi-head self-attention with causal mask,
  residual, LayerNorm, GELU MLP, residual.

**mask_decode** (~1.4s, 12%)
- **geometry_encode** (<0.1ms): 3-layer geometry transformer for point/box
  prompts (CPU path, typically 2 tokens). Trivial for single-prompt inputs.
- **encoder_fusion** (~687ms, 5.9%): 6-layer DETR encoder fusing image
  features with text/geometry context. Each layer: self-attention +
  cross-attention + FFN, evaluated as separate Metal graphs.
- **decoder** (~208ms, 1.8%): 6-layer DETR decoder with 200 learned queries.
  Each layer: self-attention, text cross-attention, vision cross-attention,
  FFN, with iterative box refinement between layers.
- **seg_head** (~451ms, 3.9%): FPN pixel decoder (3-stage upsampling to
  288x288), instance projection (1x1 conv), mask embedder MLP, dot-product
  mask logits, and objectness scoring.

**postprocess** (~22ms, 0.2%)
Stability-based mask selection, bounding box extraction, and NMS
(200 -> 1 masks for this input).

## Model Load Optimization History

| Version | model_load | Notes |
|---------|------------|-------|
| Baseline (mmap + hash table) | ~7.7s | Initial mmap-based loader |
| + mmap-direct tensors (`gh_load_mmap`) | ~1.1s | Eliminated ~3.3 GB memcpy |
| + lazy precompute + sep Q/K/V + FNV cache | ~60ms | Deferred RoPE/mask/pos_embed to first inference; eliminated ~700 MB QKV fusion memcpy; pre-stored FNV-1a hashes |
