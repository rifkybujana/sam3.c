# sam3

A fast, lightweight, dependency-free C implementation of the SAM3 (Segment Anything Model 3) inference engine.

## Features

- **Pure C Engine:** Small binary size and easy to embed.
- **Fast CPU Backend:** Optimized kernels with multithreading support (`test_threadpool`).
- **Precision Support:** Supports FP32, FP16, and BF16 formats (`test_half`, `test_elementwise_bf16`).
- **Model Conversion:** Native tool to convert from standard Safetensors model files to our optimized `.sam3` format (`sam3_convert`).
- **Memory Efficient:** Minimal overhead tensor and memory management.
- **Profiling Built-in:** Built-in latency tracing and profiling subsystem (`test_profile`).

## Project Layout

- `src/` & `include/sam3/`: Core tensor operators, backends, memory management, and model structures.
- `tools/`: Main executables like `sam3_main` and `sam3_convert`.
- `tests/`: Comprehensive unit tests for numerical operators (element-wise, matmul) and various system functionalities.
- `models/`: Location where you can place downloaded or converted `.sam3` and `.safetensors` files.

## Building

This project uses CMake. A typical C compiler (e.g., GCC or Clang) is required.

```bash
mkdir build
cd build
cmake ..
make -j4
```

### Running Tests

After building, you can verify the kernels and core tensor operations by running the provided tests using `ctest` or your test runner:

```bash
cd build
ctest --output-on-failure
```

## Tools and Usage

### 1. Converting Models (`sam3_convert`)

Convert a downloaded `.safetensors` SAM2.1/SAM3 checkpoint to the `.sam3` format:

```bash
./build/sam3_convert --input models/model.safetensors --output models/sam3.sam3
```

### 2. Running Inference (`sam3_main`)

Executes a prompt/segmentation run from the terminal on the provided image input:

```bash
./build/sam3_main --model models/sam3.sam3 --image sample_image.jpg
```

*(Note: Command-line arguments may vary depending on implementation specifics).*

## License

See the [`LICENSE`](LICENSE) file for more details.
