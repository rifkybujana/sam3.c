# SAM3 TODO

## Ground Systems (before model implementation)

### Priority 1 — Blocking

- [x] **CPU compute kernels** (`src/backend/cpu/kernels/`)
  - [x] `matmul` — tiled 8×8×64, NEON vfmaq_f32
  - [x] `conv2d` — im2col + matmul, scratch arena
  - [x] `softmax` — row-wise, numerically stable
  - [x] `layernorm` — optional gamma/beta, eps=1e-5
  - [x] `gelu` — fast tanh approximation
  - [x] `relu`
  - [x] `add` — with [M,N]+[N] broadcasting
  - [x] `mul` — with [M,N]+[N] broadcasting
  - [x] `reshape` — zero-copy data aliasing
  - [x] `transpose` — 4×4 NEON block transpose

- [x] **Backend graph evaluator** (`src/backend/cpu/cpu_backend.c`)
  - `cpu_graph_eval()` — dispatch switch over all ops
  - Error propagation from kernel failures

### Priority 2 — Important

- [x] **Image I/O** (`src/util/image.h/.c`)
  - Vendor `stb_image.h` for PNG/JPEG decoding
  - RGB uint8 output matching `sam3_set_image()` interface
  - Image resize/normalize for model input

- [x] **Thread pool** (`src/util/threadpool.h/.c`)
  - Fixed-size worker pool
  - Task queue for parallel kernel execution
  - Used by CPU backend for matmul/conv2d parallelism

- [ ] **Quantization support**
  - F16/BF16 compute paths in CPU kernels
  - INT8 quantized matmul
  - Dequantize-on-load for weight format

### Priority 3 — Before shipping

- [ ] **Metal backend** (`src/backend/metal/`)
  - Device/queue/library initialization
  - Metal buffer allocation for tensors
  - Compute pipeline setup
  - MSL kernel shaders for each op
  - Graph evaluation via command buffers

- [ ] **CLI tools**
  - `sam3_main` — argument parsing, image load, model load, inference, output

## Model Implementation

Depends on all Priority 1 ground systems.

- [ ] **Image encoder** — Hiera backbone with multi-scale feature maps
- [ ] **Prompt encoder** — point/box/mask → sparse + dense embeddings
- [ ] **Mask decoder** — two-way cross-attention transformer, upscaling, IoU head
- [ ] **Memory attention** — cross-attention with memory bank (video tracking)
- [ ] **Top-level API** — wire `sam3_load_model`, `sam3_set_image`, `sam3_segment`

## Done

- [x] Project scaffold (CLAUDE.md, CMake, directory structure)
- [x] Tensor types and metadata (`src/core/tensor.h/.c`)
- [x] Arena allocator (`src/core/alloc.h/.c`)
- [x] Compute graph DAG (`src/core/graph.h/.c`)
- [x] Backend vtable abstraction (`src/backend/backend.h`)
- [x] Logging (`src/util/log.h/.c`)
- [x] Error handling (`src/util/error.h/.c`)
- [x] Nanosecond clock (`src/util/time.h/.c`)
- [x] Profiler with stage/op/memory tracking (`src/util/profile.h/.c`)
- [x] Test framework and initial test suite
- [x] Weight file format & loader (`src/core/weight.h/.c`)
- [x] SafeTensors reader (`src/core/weight_safetensors.c`)
- [x] `sam3_convert` CLI (`tools/sam3_convert.c`)
- [x] Backend tensor allocation (`src/backend/cpu/cpu_backend.c`)
- [x] CPU compute kernels with NEON SIMD (`src/backend/cpu/kernels/`)
- [x] Backend graph evaluator (`src/backend/cpu/cpu_backend.c`)
