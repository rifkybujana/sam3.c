# SAM3 Performance

## Inference Profile

Measured on Apple M3 Pro, Release build (`-O2`), Metal backend, single run.

**Model:** sam3.sam3 (3.3 GB, 1787 tensors)
**Input:** 1008x1008 image, 1 text prompt ("a cat")
**Output:** 1 mask, 288x288

| Stage          | Time (ms) |   % |
|----------------|----------:|----:|
| model_load     |        60 | 0.4 |
| image_encode   |      9126 |  55 |
| text_encode    |      5404 |  33 |
| mask_decode    |      1930 |  12 |
| postprocess    |        21 | 0.1 |
| **Total**      | **16542** |     |

## Model Load Optimization History

| Version | model_load | Notes |
|---------|------------|-------|
| Baseline (mmap + hash table) | ~7.7s | Initial mmap-based loader |
| + mmap-direct tensors (`gh_load_mmap`) | ~1.1s | Eliminated ~3.3 GB memcpy |
| + lazy precompute + sep Q/K/V + FNV cache | ~60ms | Deferred RoPE/mask/pos_embed to first inference; eliminated ~700 MB QKV fusion memcpy; pre-stored FNV-1a hashes |
