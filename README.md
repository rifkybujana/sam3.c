# sam3.c — Efficient SAM3 Inference From Scratch in Pure C

A lightweight, dependency-free C11 implementation of [Segment Anything Model 3 (SAM3)](https://github.com/facebookresearch/sam3) built from scratch for efficient inference on Apple Silicon and x86 CPUs.

Inspired by [ggml](https://github.com/ggerganov/ggml) and [llama.cpp](https://github.com/ggerganov/llama.cpp), sam3.c implements the full SAM3 pipeline — image encoder, prompt encoder, mask decoder — in ~57K lines of portable C with zero Python dependencies.

<p align="center">
  <img src="output_bus/comparison.png" alt="sam3.c segmentation output compared to the official Python SAM3 implementation, showing IoU scores" width="800"/>
  <br>
  <em>sam3.c mask output (left) vs. official Python SAM3 (center) with per-mask IoU comparison (right)</em>
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
- **Full SAM3 pipeline** — image encoder (Hiera), prompt encoder (points, boxes, masks), mask decoder, text encoder, and tokenizer.
- **50 unit tests** — comprehensive test suite covering numerical operators, memory management, and end-to-end inference.
- **Built-in profiling** — latency tracing subsystem to identify bottlenecks.

## Quick Start

### Build

```bash
git clone https://github.com/rifkybujana/sam3.c.git
cd sam3.c
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Convert Weights

Download a SAM3 checkpoint in SafeTensors format, then convert to the optimized `.sam3` format:

```bash
./sam3_convert --input models/sam3.safetensors --output models/sam3.sam3
```

### Run Inference

```bash
./sam3_main --model models/sam3.sam3 --image photo.jpg
```

### Run Tests

```bash
ctest --output-on-failure
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
│   │   ├── image_encoder   Vision transformer
│   │   ├── prompt_encoder  Point, box, and mask prompts
│   │   ├── mask_decoder    Lightweight mask prediction head
│   │   ├── text_encoder    Text prompt encoding
│   │   └── tokenizer       BPE tokenizer
│   └── util/            Logging, error codes
├── tools/               CLI binaries (sam3_main, sam3_convert)
└── tests/               50 test files
```

## Weight Format

Model weights use the `.sam3` binary format — a compact, mmap-friendly layout designed for instant loading:

- 48-byte header + 176-byte tensor descriptors + page-aligned data blob
- FNV-1a hash table for O(1) tensor lookup by name
- Supports FP32, FP16, BF16, I32, I8, and Q8_0 (block-quantized int8)
- Converted from SafeTensors via `sam3_convert`

See [docs/weight-format.md](docs/weight-format.md) for the full specification.

## License

MIT — see [LICENSE](LICENSE).
