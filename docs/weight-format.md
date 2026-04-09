# .sam3 Binary Weight Format

The `.sam3` format is a simple, mmap-friendly binary container for SAM3 model
weights. It stores tensor metadata and data in a single file with O(1) tensor
lookup at load time.

## File Layout

```
Offset 0                          Offset 48
 |                                 |
 v                                 v
+----------+----------------------+-----------+-----------+-----+
|  Header  |  Tensor Descriptors  |  Padding  | Data Blob |     |
| 48 bytes |  n_tensors x 176 B   |  (zeros)  | (aligned) | EOF |
+----------+----------------------+-----------+-----------+-----+
                                   ^           ^
                                   |           |
                              table_end   4096-aligned
```

All numeric values are little-endian.

## Header (48 bytes)

Sits at offset 0. Validated with `_Static_assert(sizeof(...) == 48)`.

| Offset | Field              | Type       | Description                          |
|--------|--------------------|------------|--------------------------------------|
| 0      | `magic`            | uint32_t   | `0x334D4153` (ASCII "SAM3", LE)      |
| 4      | `version`          | uint32_t   | Format version (currently **2**)     |
| 8      | `flags`            | uint32_t   | Reserved, set to 0                   |
| 12     | `n_tensors`        | uint32_t   | Total tensor count                   |
| 16     | `image_size`       | int32_t    | Input resolution (e.g. 1008)         |
| 20     | `encoder_dim`      | int32_t    | Vision encoder embedding dim (1280)  |
| 24     | `decoder_dim`      | int32_t    | Mask decoder dim (256)               |
| 28     | `n_encoder_layers` | int32_t    | Vision encoder depth (e.g. 32)       |
| 32     | `n_decoder_layers` | int32_t    | Mask decoder depth (e.g. 2)          |
| 36     | `reserved[3]`      | uint32_t[] | Reserved, zeroed                     |

The config fields (`image_size` through `n_decoder_layers`) let models with
different architectures coexist. They override any compiled-in defaults at
load time.

## Tensor Descriptor (176 bytes)

An array of `n_tensors` descriptors follows the header at offset 48.
Validated with `_Static_assert(sizeof(...) == 176)`.

| Offset | Field         | Type       | Description                              |
|--------|---------------|------------|------------------------------------------|
| 0      | `name`        | char[128]  | Null-terminated tensor name              |
| 128    | `dtype`       | uint32_t   | Data type enum (see below)               |
| 132    | `n_dims`      | uint32_t   | Number of dimensions (1-4)               |
| 136    | `dims[0..3]`  | int32_t[4] | Dimension sizes (row-major); unused = 0  |
| 152    | `data_offset` | uint64_t   | Byte offset from data blob start         |
| 160    | `data_size`   | uint64_t   | Tensor data size in bytes                |
| 168    | `reserved`    | uint64_t   | Reserved, zeroed                         |

## Alignment

- **Data blob start**: aligned to **4096 bytes** (page boundary) from file
  start. Zero-padded between the descriptor table and data blob.
- **Per-tensor data**: aligned to **64 bytes** within the data blob. Each
  tensor's `data_offset` is 64-byte aligned. Zero padding between tensors
  (except after the last tensor).

```
data_start = align_up(48 + n_tensors * 176, 4096)
```

The page alignment enables efficient `mmap()` loading. The 64-byte per-tensor
alignment satisfies SIMD and cache-line requirements for backends.

## Data Types

```c
enum sam3_dtype {
    SAM3_DTYPE_F32  = 0,  /* 32-bit IEEE 754 float       */
    SAM3_DTYPE_F16  = 1,  /* 16-bit float (half)         */
    SAM3_DTYPE_BF16 = 2,  /* Brain Float 16              */
    SAM3_DTYPE_I32  = 3,  /* 32-bit signed integer       */
    SAM3_DTYPE_I8   = 4,  /* 8-bit signed integer        */
    SAM3_DTYPE_Q8_0 = 5,  /* Block-quantized int8        */
};
```

**Q8_0 block format**: 32 int8 values + 1 float32 scale = 36 bytes per block.
Total bytes for N elements: `ceil(N / 32) * 36`. Per-element strides are
undefined for Q8_0.

## Loading

The loader (`sam3_weight_open()`) does:

1. `mmap()` the file (read-only, `MAP_PRIVATE`)
2. Validate magic (`0x334D4153`) and version (`2`)
3. Bounds-check the descriptor table and all tensor data ranges
4. Build an FNV-1a hash table for O(1) tensor lookup by name
   - Table size: next power of 2 >= 2 * `n_tensors`
   - Collision resolution: linear probing

Tensor data pointers point directly into the mmap region and are valid until
`sam3_weight_close()`.

## Writing

The writer (`sam3_weight_write()`) iterates a `weight_reader` vtable:

1. Write header (48 bytes)
2. Write descriptor array (`n_tensors` x 176 bytes)
3. Zero-pad to 4096-byte boundary
4. Write each tensor's data, zero-padding to 64-byte alignment between tensors
5. On error, delete the output file

## Conversion

The `sam3_convert` tool converts SafeTensors weights to `.sam3`:

```
sam3_convert model.safetensors -o model.sam3 \
    --image-size 1008 --encoder-dim 1280 --decoder-dim 256 \
    --encoder-layers 32 --decoder-layers 2
```

Optional `--quantize q8_0` wraps the reader in a quantizing layer that
converts float tensors (F32/F16/BF16) with >= 1024 elements to Q8_0. Smaller
tensors and non-float types pass through unchanged.

## Validation

The format does **not** include checksums. Integrity is ensured by:

- Magic number check (catches corrupt files)
- Version check (detects incompatible format changes)
- Bounds checking (all offsets/sizes validated against file size)

## Source Files

| File                       | Role                              |
|----------------------------|-----------------------------------|
| `src/core/weight.h`        | Format structs and loader API     |
| `src/core/weight.c`        | Loader and writer implementation  |
| `tools/sam3_convert.c`     | SafeTensors -> .sam3 converter    |
