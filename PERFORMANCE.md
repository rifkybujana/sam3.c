# SAM3 Performance

## Inference Profile

Measured on Apple M3 Pro, Release build (`-O2`), Metal backend, single run.

**Model:** sam3.sam3 (3.3 GB, 1787 tensors)
**Input:** 1008x1008 image, 1 text prompt ("a cat")
**Output:** 1 mask, 288x288

| Stage            | Time (ms) |   % |
|------------------|----------:|----:|
| model_load       |      1087 | 9.9 |
| image_normalize  |         1 | 0.0 |
| image_encode     |      6566 |  60 |
|   vit_precompute |        47 | 0.4 |
|   vit_patch_embed|        54 | 0.5 |
|   vit_blocks     |      6029 |  55 |
|   neck           |       362 | 3.3 |
| text_encode      |      3174 |  29 |
|   tokenize       |      <0.1 | 0.0 |
|   text_blocks    |      3174 |  29 |
| mask_decode      |      1182 |  11 |
|   geometry_encode|      <0.1 | 0.0 |
|   encoder_fusion |       487 | 4.5 |
|   decoder        |       177 | 1.6 |
|   seg_head       |       506 | 4.6 |
| postprocess      |        24 | 0.2 |
| **Total**        | **10947** |     |

### Optimization History

| Version | vit_blocks | text_blocks | mask_decode | neck | Sum |
|---------|-----------:|------------:|------------:|-----:|----:|
| Baseline (per-head SDPA, F32, per-block eval) | 8419 | 5064 | 1639 | 499 | 15621 |
| + Fused MHSA + F16 compute + 4-block batching | 6029 | 3174 | 1182 | 362 | **10747** |
| Speedup                                       | -28% | -37% | -28% | -27% | **-31%** |

### Stage Details

**model_load** (~61ms)
Open `.sam3` weight file via mmap, build FNV-1a hash table for O(1) tensor
lookup, initialize module structs, and point tensor data pointers into the
mmap region. RoPE tables, window mask, position embed tiling, and 2D
sinusoidal position encoding are deferred to first inference. No weight data
is copied during load — tensors reference the mmap'd region directly.

**image_encode** (~6.6s, 60%)
- **vit_precompute** (~47ms): Lazy first-call initialization of RoPE tables
  (window + global), 5184x5184 window attention mask, tiled position
  embeddings (577 -> 72x72), and 2D sinusoidal position encoding.
- **vit_patch_embed** (~54ms): 14x14 conv2d patch embedding, absolute
  position embedding addition, and ln_pre LayerNorm. Single graph eval.
- **vit_blocks** (~6.0s, 55%): 32 ViT-H transformer blocks (1280-dim, 16
  heads). Up to 4 blocks are batched per `graph_eval` call, with an
  adaptive fallback that reduces the batch size if arena capacity runs
  low (important for large 72x72 patch grids). Each block: LayerNorm,
  fused-multi-head self-attention with RoPE (windowed or global),
  residual, LayerNorm, GELU MLP, residual.
- **neck** (~362ms, 3.3%): 4-scale FPN producing feature maps at 4x, 2x,
  1x, and 0.5x resolution via conv2d, transposed conv2d, and maxpool stages.

**text_encode** (~3.2s, 29%)
- **tokenize** (<0.1ms): BPE tokenization of the text prompt.
- **text_blocks** (~3.2s, 29%): CLIP text encoder (24-layer transformer,
  1024-dim, 16 heads). Up to 4 blocks are batched per `graph_eval` call.
  Each block: LayerNorm, fused-multi-head self-attention with causal mask,
  residual, LayerNorm, GELU MLP, residual.

**mask_decode** (~1.2s, 11%)
- **geometry_encode** (<0.1ms): 3-layer geometry transformer for point/box
  prompts (CPU path, typically 2 tokens). Trivial for single-prompt inputs.
- **encoder_fusion** (~487ms, 4.5%): 6-layer DETR encoder fusing image
  features with text/geometry context. Each layer: self-attention +
  cross-attention + FFN, evaluated as separate Metal graphs.
- **decoder** (~177ms, 1.6%): 6-layer DETR decoder with 200 learned queries.
  Each layer: self-attention, text cross-attention, vision cross-attention,
  FFN, with iterative box refinement between layers.
- **seg_head** (~506ms, 4.6%): FPN pixel decoder (3-stage upsampling to
  288x288), instance projection (1x1 conv), mask embedder MLP, dot-product
  mask logits, and objectness scoring.

**postprocess** (~24ms, 0.2%)
Stability-based mask selection, bounding box extraction, and NMS
(200 -> 1 masks for this input).

## Model Load Optimization History

| Version | model_load | Notes |
|---------|------------|-------|
| Baseline (mmap + hash table) | ~7.7s | Initial mmap-based loader |
| + mmap-direct tensors (`gh_load_mmap`) | ~1.1s | Eliminated ~3.3 GB memcpy |
| + lazy precompute + sep Q/K/V + FNV cache | ~60ms | Deferred RoPE/mask/pos_embed to first inference; eliminated ~700 MB QKV fusion memcpy; pre-stored FNV-1a hashes |
