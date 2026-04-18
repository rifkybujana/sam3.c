# sam3.c — Efficient SAM3 Inference From Scratch in Pure C

A lightweight, dependency-free C11 implementation of [Segment Anything Model 3 (SAM3)](https://github.com/facebookresearch/sam3) built from scratch for efficient inference on Apple Silicon and x86 CPUs.

Inspired by [ggml](https://github.com/ggerganov/ggml) and [llama.cpp](https://github.com/ggerganov/llama.cpp), sam3.c implements the full SAM3 pipeline — image encoder, prompt encoder, mask decoder — in ~57K lines of portable C with zero Python dependencies.

<p align="center">
  <img src="assets/sam3_segmentation.png" alt="sam3.c multi-object segmentation — four masks with distinct colors on a street scene" width="540"/>
  <br>
  <em>Four-mask segmentation output from sam3.c — each object highlighted in a distinct color</em>
</p>

## Why sam3.c?

| | sam3.c | Official SAM3 (Python) |
|---|---|---|
| **Language** | C11, no dependencies | Python + PyTorch |
| **Binary size** | Single static binary | GB-scale environment |
| **GPU support** | Apple Metal (native) | CUDA |
| **Precision** | FP32, FP16, BF16 | FP32 |
| **Memory** | Arena allocator, mmap weights | PyTorch allocator |
| **Startup** | Instant (mmap) | Seconds (model load) |

## Features

- **Built from scratch in pure C** — no PyTorch, no ONNX, no wrappers. Every tensor op, every layer, written by hand.
- **Metal GPU backend** — hardware-accelerated inference on Apple Silicon (M1/M2/M3/M4).
- **Multithreaded CPU backend** — optimized SIMD kernels with thread pool for x86 and ARM.
- **FP16 and BF16 support** — run inference in half precision for lower memory and faster compute.
- **Custom `.sam3` weight format** — mmap-friendly binary format with O(1) tensor lookup via hash table.
- **Full SAM3 pipeline** — image encoder (Hiera, EfficientViT, TinyViT), prompt encoder (points, boxes, masks), mask decoder, text encoder, and tokenizer.
- **Video object tracking** — memory-based frame-by-frame propagation with point, box, and mask prompts. Supports MPEG video files and frame directories.
- **Multiple backbones** — Hiera (full accuracy), EfficientViT-B1 (lightweight, 512px), and TinyViT-21M (128x128 masks at 1008px input).
- **Unified CLI** — single `sam3_cli` binary with `segment`, `convert`, and `info` subcommands. Supports stdin/stdout piping, JSON output, and multi-mask color overlays.
- **48 unit tests** — comprehensive test suite covering numerical operators, memory management, and end-to-end inference.
- **Built-in profiling** — latency tracing subsystem to identify bottlenecks.

## Supported Models

| Backbone | Input Size | Mask Resolution | Parameters | Encode (ms) | Segment (ms) | Use Case |
|---|---|---|---|---:|---:|---|
| **Hiera** | 1008x1008 | 288x288 | 1.6B | 2336 | 1224 | Full accuracy |
| **TinyViT-21M** | 1008x1008 | 128x128 | 0.8B | 487 | 363 | Balanced quality/speed |
| **EfficientViT-B1** | 512x512 | 64x64 | 0.8B | 70 | 177 | Fastest, interactive |

Timings on Apple M4 (10-core GPU, Metal backend, Release build). Encode = `sam3_set_image`, Segment = `sam3_segment` with a box prompt. See [BENCHMARK.md](BENCHMARK.md) for full results.

All backbones share the same prompt encoder, mask decoder, and text encoder. The backbone is selected automatically based on the checkpoint.

## Quick Start

### Build

```bash
git clone https://github.com/rifkybujana/sam3.c.git
cd sam3.c
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

> **First build note:** the build fetches and statically compiles
> FFmpeg, openh264, and libvpx into `build/external/`. Expect ~10-15
> minutes on first configure; subsequent incremental builds are fast.
> The resulting binary has no runtime dependency on system ffmpeg.

### Convert Weights

Download a SAM3 checkpoint in SafeTensors format, then convert to the optimized `.sam3` format:

```bash
# Hiera (default backbone)
./sam3_cli convert -i models/sam3.safetensors -o models/sam3.sam3

# TinyViT or EfficientViT (specify backbone)
./sam3_cli convert -i models/tinyvit.safetensors -o models/tinyvit.sam3 --backbone tinyvit
./sam3_cli convert -i models/evit.safetensors -o models/evit.sam3 --backbone efficientvit
```

### Run Inference

```bash
# Point prompt (foreground point at x=500, y=375)
./sam3_cli segment -m models/sam3.sam3 -i photo.jpg -p 500,375,1 --overlay

# Text prompt
./sam3_cli segment -m models/sam3.sam3 -i photo.jpg -t "person" --overlay

# Box prompt
./sam3_cli segment -m models/sam3.sam3 -i photo.jpg -b 100,100,400,400 --all
```

### Video tracking

Track an object across frames of a video:

```bash
./sam3_cli track --model models/sam3.sam3 --video clip.mp4 \
    --point 504,504,1 --frame 0 --output out/
```

Output: `out/frame_NNNNN.png` binary mask per frame.

### Inspect a Model

```bash
./sam3_cli info models/sam3.sam3
```

### Run Tests

```bash
ctest --output-on-failure
```

## Language bindings

sam3.c ships bindings for multiple languages under `bindings/`:

- **Python** — `bindings/python/`. Install with `pip install -e bindings/python`.
- **Rust** — `bindings/rust/`. Cargo workspace with `sam3-sys` (FFI) and
  `sam3` (safe API: owned `Ctx`, typed prompt enum, RAII result cleanup,
  `SegmentResult::nms` matching the CLI's post-processing). See
  `bindings/rust/README.md`.

Both bindings link dynamically against `libsam3.{dylib,so}` built with
`cmake -S . -B build -DSAM3_SHARED=ON && cmake --build build`. At runtime
set `DYLD_LIBRARY_PATH` (macOS) or `LD_LIBRARY_PATH` (Linux) to the build
directory, or install `libsam3` system-wide so the dynamic loader can
find it.

Minimal Rust usage:

```rust
use sam3::{Ctx, Prompt};

let mut ctx = Ctx::new()?;
ctx.load_model("models/efficient.sam3")?;      // auto-loads co-located BPE
ctx.set_image_file("photo.jpg")?;
ctx.set_text("person")?;

let raw = ctx.segment(&[Prompt::Text("person")])?;
let hits = raw.nms(0.5, 0.5, 0.0)?;            // 200 candidates → ~N detections
println!("found {} objects, top score {:.3}",
         hits.n_masks(), hits.iou_scores()[0]);
```

## Architecture

```
sam3.c
├── include/sam3/        Public API headers
├── src/
│   ├── core/            Tensor ops, arena allocator, compute graph, weight loader
│   ├── backend/
│   │   ├── cpu/         Multithreaded CPU kernels (SIMD-optimized)
│   │   └── metal/       Apple Metal GPU backend
│   ├── model/           SAM3 layers
│   │   ├── image_encoder   Vision transformer (Hiera, EfficientViT, TinyViT)
│   │   ├── prompt_encoder  Point, box, and mask prompts
│   │   ├── mask_decoder    Lightweight mask prediction head
│   │   ├── text_encoder    Text prompt encoding
│   │   └── tokenizer       BPE tokenizer
│   └── util/            Logging, error codes
├── tools/               Unified CLI (sam3_cli: segment, convert, info)
└── tests/               48 test files
```

## Performance

On an Apple M4 with the Metal backend, EfficientViT delivers end-to-end
image-to-mask in ~250 ms (4 FPS), making interactive point-and-click
segmentation practical. Once the image is encoded, each additional prompt on
the same image resolves in under 200 ms.

Hiera-Large trades speed for accuracy at 3.6 s per image with 5184 patches and
32 transformer blocks. Multi-prompt workflows amortize the 2.3 s encode cost.

The Metal backend achieves 90.8% of theoretical F32 peak (3086 / 3400 GFLOPS)
on matmul microbenchmarks and up to 149x speedup over CPU for F16 workloads.

Full kernel-level and pipeline benchmark results are in
[BENCHMARK.md](BENCHMARK.md).

## Weight Format

Model weights use the `.sam3` binary format — a compact, mmap-friendly layout designed for instant loading:

- 48-byte header + 176-byte tensor descriptors + page-aligned data blob
- FNV-1a hash table for O(1) tensor lookup by name
- Supports FP32, FP16, BF16, I32, I8, and Q8_0 (block-quantized int8)
- Converted from SafeTensors via `sam3_cli convert`

See [docs/weight-format.md](docs/weight-format.md) for the full specification.

## License

MIT — see [LICENSE](LICENSE).
