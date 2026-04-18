# SAM3 Metal Benchmark Results

Kernel-level and pipeline benchmarks comparing the Metal (MLX-C GPU) and CPU
backends. All numbers are wall-clock averages over 50 timed iterations after 5
warmup iterations.

## Test Environment

| Field | Value |
|-------|-------|
| Chip | Apple M4 (4P + 6E cores, 10-core GPU) |
| RAM | 16 GB unified |
| OS | macOS 15.6 (Darwin 24.6.0, arm64) |
| Metal | Metal 3 |
| Backend | MLX-C, F16 compute mode |
| Build | Release (`-O2`, no sanitizers) |
| Commit | `495c46f` |
| Date | 2026-04-16 |

## Matmul F32

| Size (M x K x N) | CPU (ms) | Metal (ms) | Speedup | Metal GFLOPS | % of 3.4 TFLOPS peak |
|-------------------|----------|------------|---------|-------------:|---------------------:|
| 64 x 64 x 64 | 0.001 | 0.354 | 0.00x | 1.5 | 0.0% |
| 256 x 256 x 256 | 0.020 | 0.383 | 0.05x | 87.7 | 2.6% |
| 512 x 512 x 512 | 0.151 | 0.730 | 0.21x | 367.6 | 10.8% |
| 1024 x 1024 x 1024 | 1.287 | 1.321 | 0.97x | 1625.8 | 47.8% |
| 2048 x 2048 x 2048 | 10.667 | 6.037 | 1.77x | 2845.6 | 83.7% |
| 4096 x 4096 x 4096 | --- | 44.542 | --- | 3085.6 | 90.8% |

**Crossover point:** Metal overtakes CPU at ~1024x1024 for F32 matmul. At 2048
the GPU is 1.77x faster. At 4096 the GPU hits **90.8% of theoretical peak**
(3085.6 / 3400 GFLOPS). CPU is faster for small matrices due to GPU dispatch
latency (~0.35 ms overhead).

## Matmul F16

| Size (M x K x N) | CPU (ms) | Metal (ms) | Speedup | Metal GFLOPS | % of 6.8 TFLOPS peak |
|-------------------|----------|------------|---------|-------------:|---------------------:|
| 64 x 64 x 64 | 0.049 | 0.235 | 0.21x | 2.2 | 0.0% |
| 256 x 256 x 256 | 1.775 | 0.353 | 5.03x | 95.1 | 1.4% |
| 512 x 512 x 512 | 12.412 | 0.495 | 25.05x | 541.8 | 8.0% |
| 1024 x 1024 x 1024 | 98.694 | 1.105 | 89.28x | 1942.7 | 28.6% |
| 2048 x 2048 x 2048 | 791.626 | 5.301 | 149.35x | 3241.1 | 47.7% |
| 4096 x 4096 x 4096 | --- | 40.967 | --- | 3354.8 | 49.3% |

**Crossover point:** Metal wins at every size above 64x64 for F16. The CPU has
no native F16 ALU on ARM so it emulates via scalar conversion, making even 256
5x slower. At 2048 the GPU is **149x faster**. Peak F16 throughput reaches 3354
GFLOPS (49.3% of the 6.8 TFLOPS theoretical peak).

## Elementwise Add (F32)

| Size (rows x cols) | CPU (ms) | Metal (ms) | Speedup |
|---------------------|----------|------------|---------|
| 256 x 256 | 0.023 | 0.421 | 0.05x |
| 1024 x 1024 | 0.083 | 0.671 | 0.12x |
| 2048 x 2048 | 0.538 | 1.491 | 0.36x |
| 4096 x 4096 | 2.177 | 4.754 | 0.46x |

Elementwise ops are memory-bandwidth-bound. The CPU wins at all tested sizes
because the GPU dispatch overhead dominates for these lightweight kernels. For
SAM3 inference this is fine: elementwise ops are always fused into larger
compute graphs where the GPU amortizes launch cost.

## Softmax (F32)

| Size (rows x cols) | CPU (ms) | Metal (ms) | Speedup |
|---------------------|----------|------------|---------|
| 256 x 256 | 0.037 | 0.353 | 0.10x |
| 1024 x 1024 | 0.359 | 0.673 | 0.53x |
| 2048 x 2048 | 1.446 | 1.492 | 0.97x |
| 4096 x 4096 | 6.168 | 4.461 | 1.38x |

**Crossover point:** Metal overtakes CPU at ~4096x4096. Softmax is
reduction-heavy and bandwidth-bound, so GPU advantage only appears at large
sizes.

## Pipeline: Matmul + Add + Softmax (F32)

| Size (M x K x N) | CPU (ms) | Metal (ms) | Speedup |
|-------------------|----------|------------|---------|
| 256 x 256 x 256 | 0.083 | 0.419 | 0.20x |
| 512 x 512 x 512 | 0.322 | 0.634 | 0.51x |
| 1024 x 1024 x 1024 | 1.981 | 2.195 | 0.90x |
| 2048 x 2048 x 2048 | 13.466 | 6.501 | 2.07x |

Multi-op pipeline amortizes GPU dispatch overhead across the graph. The
crossover is between 1024 and 2048 — matching SAM3's typical attention
dimensions. At 2048 the GPU is **2.07x faster**.

## End-to-End Image Encoding (Metal)

Measures `sam3_set_image()` — the full vision pipeline that converts raw RGB
pixels into cached multi-scale feature maps. This includes pixel normalization,
patch embedding, all transformer/convolution blocks, FPN neck, and feature
caching. Measured with 3 warmup + 10 timed iterations.

| Model | Encoder | Input | Patches | Params | Mean (ms) | Min (ms) | Max (ms) |
|-------|---------|-------|--------:|-------:|----------:|----------:|---------:|
| EfficientViT (efficient.sam3) | EfficientViT-L | 512x512 | 256 | 0.8B | **70.4** | 68.7 | 72.5 |
| TinyViT-L (tinyvit_l.sam3) | TinyViT-L | 1008x1008 | 1024 | 0.8B | **487.3** | 475.3 | 499.1 |
| Hiera-Large (sam3.sam3) | Hiera-L | 1008x1008 | 5184 | 1.6B | **2335.7** | 2131.5 | 3420.5 |

- **EfficientViT encodes a 512x512 image in 70 ms** (~14.3 FPS). The
  lightweight 4-stage CNN+attention hybrid keeps the graph small (3958 nodes)
  and fits well in unified memory.

- **TinyViT-L processes 1008x1008 in 487 ms** (~2.1 FPS). The larger input
  resolution (4x more pixels) and 32x32 patch grid (1024 patches vs 256)
  explain the ~7x longer runtime vs EfficientViT.

- **Hiera-Large processes 1008x1008 in 2.3 s** with 72x72 patch grid (5184
  patches) and 32 full transformer blocks with windowed/global attention. This
  is the largest SAM3 backbone — 20x the patches of EfficientViT — and serves
  as the accuracy-optimized variant.

## End-to-End Segmentation (Metal)

Measures `sam3_segment()` — the decode pipeline that runs on cached image
features from `sam3_set_image()`. This includes text encoding (CLIP), prompt
encoding, DETR-style cross-attention decoder, mask prediction head, IoU
scoring, and postprocessing (stability selection + box extraction). Each prompt
type exercises a different code path.

| Model | Point (ms) | Box (ms) | Text (ms) |
|-------|----------:|---------:|----------:|
| EfficientViT (efficient.sam3) | **182.1** | **177.2** | **173.0** |
| TinyViT-L (tinyvit_l.sam3) | **427.3** | **363.4** | **449.7** |
| Hiera-Large (sam3.sam3) | **1223.9** | **1218.6** | **1946.9** |

- **Point and box prompts** are similar in cost — the prompt encoder is
  lightweight and the bulk of time is in the DETR decoder and mask head.

- **Text prompts on Hiera are ~60% slower** (1947 ms vs 1224 ms) because the
  24-layer CLIP text encoder runs synchronously within `sam3_segment()`. For
  point/box prompts a short dummy text ("visual") is injected, which is much
  cheaper than encoding a real user query.

- **EfficientViT achieves under 200 ms per segment**, making interactive
  point-and-click segmentation feasible on Apple Silicon.

### Total End-to-End Latency (Image + Segment)

For a single image with one prompt, the user-facing latency is the sum of
image encoding and segmentation:

| Model | Encode + Point (ms) | Encode + Box (ms) | Effective FPS |
|-------|-------------------:|------------------:|--------------:|
| EfficientViT | **252** | **248** | ~4.0 |
| TinyViT-L | **915** | **851** | ~1.2 |
| Hiera-Large | **3560** | **3554** | ~0.3 |

Once the image is encoded, subsequent prompts on the same image only pay the
segment cost — so multi-prompt workflows (e.g. annotating 10 objects) amortize
the encode time and approach the segment-only FPS.

## Video: Per-Frame Tracking Cost

Measures one tracker step (memory attention + mask decoder + minor
IO) inside a `reset → seed → propagate` loop on a synthetic moving-
square clip. `mean_ms` in the result table is for the full loop;
per-frame cost below is `mean_ms / (n_frames - 1)` to amortise the
reset/seed overhead. The memory bank saturates after ~6–8 tracked
frames, so the 32f and 64f columns measure steady-state cost.

### Clip-length scaling (1 object, FORWARD)

| Model | 8f (ms/frame) | 32f (ms/frame) | 64f (ms/frame) |
|-------|--------------:|---------------:|---------------:|
| EfficientViT (efficient.sam3) | _tbd_ | _tbd_ | _tbd_ |
| TinyViT-L (tinyvit_l.sam3)    | _tbd_ | _tbd_ | _tbd_ |
| Hiera-Large (sam3.sam3)       | _tbd_ | _tbd_ | _tbd_ |

Per-frame cost should stay roughly flat between 32f and 64f once the
memory bank saturates — a linear climb would indicate the bank is
growing unbounded.

### Multi-object scaling (32 frames, FORWARD)

| Model | 1 obj | 2 obj | 4 obj | 8 obj |
|-------|------:|------:|------:|------:|
| EfficientViT | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| TinyViT-L    | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| Hiera-Large  | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

Image encoder cost is fixed per frame; mask decoder and memory
attention run once per tracked object. Linear scaling in N objects
with a shared encoder term is the expected pattern.

### Propagation direction (32 frames, 1 object)

| Model | FORWARD | BOTH (fwd+bwd) |
|-------|--------:|---------------:|
| EfficientViT | _tbd_ | _tbd_ |
| TinyViT-L    | _tbd_ | _tbd_ |
| Hiera-Large  | _tbd_ | _tbd_ |

The BOTH case seeds at the middle frame so each pass does equal work.
Expected ratio is ~1.5–2× forward — encoded features are reused
across both passes, so the reverse direction mostly pays for its own
memory attention and mask decode.

## Video: End-to-End Clip Latency

Full user-facing latency for `sam3_video_start → add_points →
propagate → sam3_video_end` on the synthetic clip. Includes session
init, feature caching, propagation, and teardown.

### End-to-end latency (1 object, FORWARD)

| Model | 8f (ms) | 32f (ms) | 64f (ms) | Effective FPS (64f) |
|-------|--------:|---------:|---------:|--------------------:|
| EfficientViT | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| TinyViT-L    | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| Hiera-Large  | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

### Multi-object end-to-end (32 frames, FORWARD)

| Model | 1 obj | 4 obj |
|-------|------:|------:|
| EfficientViT | _tbd_ | _tbd_ |
| TinyViT-L    | _tbd_ | _tbd_ |
| Hiera-Large  | _tbd_ | _tbd_ |

Effective FPS is the clip length divided by the total wall-clock
time — the interactive-annotation ceiling for that model.

## Peak Throughput Summary

| Precision | Peak GFLOPS | Theoretical Peak | Utilization |
|-----------|------------:|:----------------:|:-----------:|
| F32 | 3085.6 | 3.4 TFLOPS | **90.8%** |
| F16 | 3354.8 | 6.8 TFLOPS | **49.3%** |

## Key Takeaways

1. **EfficientViT is interactive on Apple Silicon.** End-to-end
   image-to-mask in ~250 ms (4 FPS) makes real-time point-and-click
   segmentation practical. Once the image is encoded (70 ms), each additional
   prompt resolves in under 200 ms.

2. **Hiera-Large trades speed for accuracy.** At 3.6 s per image it is not
   interactive, but with 5184 patches and 32 transformer blocks it is the
   highest-fidelity SAM3 backbone. Multi-prompt workflows amortize the 2.3 s
   encode cost.

3. **Metal matmul hits 90.8% of theoretical F32 peak** at 4096x4096 — the MLX
   GEMM kernels (Steel) are highly optimized for Apple Silicon.

4. **F16 is up to 149x faster on Metal vs CPU** because ARM CPUs lack native
   F16 compute. This makes the Metal backend essential for half-precision
   inference.

5. **GPU dispatch overhead is ~0.3-0.4 ms**, which means Metal only wins when
   the compute workload exceeds this threshold. For SAM3 inference (image
   encoder attention blocks at 1024-4096 dim), Metal is solidly faster.

6. **Bandwidth-bound ops (add, softmax) favor CPU at small sizes.** In
   practice these ops are fused into larger graphs via MLX's lazy evaluation,
   so the standalone numbers understate GPU efficiency.

## Running the Benchmark

```bash
# Build in release mode (no sanitizers)
mkdir -p build-release && cd build-release
cmake .. -DCMAKE_BUILD_TYPE=Release -DSAM3_BENCH=ON
make -j$(nproc)

# Kernel benchmarks (Metal vs CPU, no model needed)
./bench_metal

# End-to-end pipeline benchmarks (requires model weights)
./sam3_cli bench pipeline --model ../models/efficient.sam3 --backend metal

# Full suite with JSON output and baseline regression detection
./sam3_cli bench all --model ../models/efficient.sam3 --backend metal \
    --output results.json
./sam3_cli bench all --model ../models/efficient.sam3 --backend metal \
    --compare results.json --threshold 5.0

# Only the video cases, with results written to JSON
./sam3_cli bench all --model ../models/efficient.sam3 --backend metal \
    --filter "video_*" --output video.json
```
