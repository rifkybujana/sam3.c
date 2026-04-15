# SAM3 C Engine — Architecture

This document describes the **implemented** architecture of the pure C11
SAM3 inference engine in this repository. It is written from the code
(not the Python upstream): every tensor shape, constant, and pipeline
step can be traced to a source file and line in `src/`.

For the Python reference model, see [`docs/reference/`](reference/).
For the on-disk weight layout, see [`docs/weight-format.md`](weight-format.md).

---

## 1. High-level picture

```
          +---------------------------------------------------+
          |                   sam3_ctx                        |
          |  backend + arenas + profiler + image_model        |
          +---------------------------------------------------+
                                   |
                                   v
          +---------------------------------------------------+
          |              sam3_image_model                     |
          |                                                   |
          |   +--------------------------------------------+  |
          |   |       sam3_vl_backbone (vision+text)       |  |
          |   |                                            |  |
          |   |  sam3_vit   (Hiera, 32 blocks, 1024 dim)   |  |
          |   |  sam3_efficientvit (EfficientViT-B1)       |  |
          |   |  sam3_tinyvit (TinyViT-21M, 4 layers)      |  |
          |   |       |  (backbone dispatch by type)       |  |
          |   |       v   [grid^2, 1024]                   |  |
          |   |  sam3_neck (FPN, 4 scales -> d_model=256)  |  |
          |   |                                            |  |
          |   |  sam3_tokenizer -> sam3_text_encoder       |  |
          |   |       (CLIP, 24 blocks, 1024 -> 256)       |  |
          |   +--------------------------------------------+  |
          |                |              |                   |
          |                v              v                   |
          |   +------------------+  +------------------+      |
          |   | sam3_geometry_   |  | sam3_encoder_    |      |
          |   |   encoder (3L)   |  |   fusion (6L)    |      |
          |   +------------------+  +------------------+      |
          |                \              /                   |
          |                 v            v                    |
          |         +---------------------------+              |
          |         |   sam3_decoder (6L, 200Q) |              |
          |         |   + box refinement MLP    |              |
          |         +---------------------------+              |
          |                |             |                     |
          |                v             v                     |
          |    +------------------+  +------------------+      |
          |    | sam3_seg_head    |  | sam3_mask_decoder|      |
          |    | (text / box path)|  | (point path)     |      |
          |    +------------------+  +------------------+      |
          |                      |   /                          |
          |                      v  v                           |
          |         +---------------------------+               |
          |         | sam3_dot_scorer           |               |
          |         +---------------------------+               |
          +---------------------------------------------------+
                                   |
                                   v
                     sam3_result  { masks, boxes,
                                    iou_scores, scores,
                                    best_mask }
```

The public API exposes only a handful of functions on top of this
structure; everything below `sam3_ctx` is internal.

**Public surface** (`include/sam3/sam3.h`):

```c
sam3_ctx *sam3_init(void);
void      sam3_free(sam3_ctx *ctx);

sam3_error sam3_load_model(sam3_ctx *, const char *path);
sam3_error sam3_load_bpe  (sam3_ctx *, const char *path);

sam3_error sam3_set_image     (sam3_ctx *, const uint8_t *rgb, int w, int h);
sam3_error sam3_set_image_file(sam3_ctx *, const char *path);
sam3_error sam3_set_text      (sam3_ctx *, const char *text);  /* async */

sam3_error sam3_segment(sam3_ctx *,
                        const struct sam3_prompt *prompts, int n,
                        struct sam3_result *out);
```

Everything else (model struct, graph, arena, backend, tokenizer) is
implementation detail living under `src/`.

### End-to-end data flow with tensor shapes

The diagram below traces a single image + text inference through every
stage, annotated with tensor shapes. `H, W` denote the backbone grid
dimensions (Hiera: `72 x 72 = 5184`, TinyViT: `32 x 32 = 1024`,
EfficientViT: `16 x 16 = 256`). Shapes below use Hiera as the example.

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  INPUT                                                            │
  │    raw RGB pixels  [H_orig, W_orig, 3] uint8                     │
  │    text prompt     "a red apple"                                  │
  └──────────────────────────────────────────────────────────────────┘
                              │
              stb_image + letterbox + normalize to [-1, 1]
                              │
                              v
  ┌──────────────────────────────────────────────────────────────────┐
  │  [3, 1008, 1008] f32               [text string]                 │
  └──────────────────────────────────────────────────────────────────┘
            │                                    │
            │                                    v
            │                        ┌────────────────────────┐
            │                        │  sam3_tokenizer (BPE)  │
            │                        │  [1, 32] i32 token IDs │
            │                        └────────────────────────┘
            │                                    │
            │                                    v
            │                   ┌────────────────────────────────┐
            │                   │  sam3_text_encoder (24 blocks) │
            │                   │  width=1024  ->  d_model=256   │
            │                   └────────────────────────────────┘
            │                                    │
            │                                    v
            │                      [seq_len=32, 256] text_features
            │                      (+ pooled [256])
            v
  ┌────────────────────────────────────────────────┐
  │  Vision backbone (dispatch by backbone_type)    │
  │                                                │
  │  Hiera:        sam3_vit        (grid 72x72)    │
  │  TinyViT-21M:  sam3_tinyvit    (grid 32x32)    │
  │  EfficientViT: sam3_efficientvit (grid 16x16)  │
  │                                                │
  │  per-block eval to bound memory                │
  └────────────────────────────────────────────────┘
            │
            v   [grid^2, 1024] backbone features
  ┌────────────────────────────────────┐
  │  sam3_neck  (FPN, 4 scales)        │
  │                                    │
  │  Hiera example (grid=72):          │
  │  s0 (4x):  [1, 288, 288, 256]      │
  │  s1 (2x):  [1, 144, 144, 256]      │
  │  s2 (1x):  [1,  72,  72, 256]      │
  │  s3 (0.5x):[1,  36,  36, 256]      │
  └────────────────────────────────────┘
            │
            v   image_features  [1, grid, grid, 256] NHWC
            │   multi-scale features cached on model
            │
            │             ┌────────── prompts ────────┐
            │             │                           │
            │             v                           │
            │   ┌──────────────────────────┐          │
            │   │ sam3_geometry_encoder    │          │
            │   │ (points/boxes, 3 layers) │          │
            │   │ [N+1, 256] geom tokens   │          │
            │   └──────────────────────────┘          │
            │             │                           │
            ├─────────────┼──── text_features ────────┤
            v             v                           v
  ┌────────────────────────────────────────────────────┐
  │  sam3_encoder_fusion  (6 layers)                   │
  │    SA(image) -> CA(image <- text) -> FFN           │
  │    output: [5184, 256] fused encoder states        │
  └────────────────────────────────────────────────────┘
                            │
                            v
  ┌────────────────────────────────────────────────────┐
  │  sam3_decoder  (6 layers, 200 queries, DAB-DETR)   │
  │    SA(Q) -> CA(Q <- encoder) -> CA(Q <- text)      │
  │    -> FFN -> box refinement MLP                    │
  │    output: [200, 256] query embeddings             │
  │            [200, 4]   refined xyxy boxes           │
  └────────────────────────────────────────────────────┘
             │                          │
  (text/box prompts)              (point prompts)
             │                          │
             v                          v
  ┌─────────────────────┐   ┌──────────────────────────┐
  │ sam3_seg_head       │   │ sam3_mask_decoder        │
  │ MaskFormer FPN +    │   │ 2-way transformer +      │
  │ 3-layer mask MLP    │   │ pixel decoder + hypernets│
  │ logits = mask_mlp(q)│   │ 4 masks + IoU scores     │
  │        @ pixel_feats│   │                          │
  └─────────────────────┘   └──────────────────────────┘
             │                          │
             └─────────────┬────────────┘
                           v
                ┌────────────────────────┐
                │  sam3_dot_scorer       │
                │  [200, 1] confidence   │
                └────────────────────────┘
                           │
                           v   postprocess
              (mask resize, box compute, NMS, best-mask select)
                           │
                           v
  ┌──────────────────────────────────────────────────────┐
  │  sam3_result                                          │
  │    masks       [n_masks, H_orig, W_orig] f32          │
  │    iou_scores  [n_masks] f32                          │
  │    boxes       [n_masks, 4] f32 (xyxy, orig coords)   │
  │    best_mask   int                                    │
  └──────────────────────────────────────────────────────┘
```

---

## 2. Directory map

```
include/sam3/
  sam3.h              # public API
  sam3_types.h        # error codes, dtypes, prompt/result structs

src/core/             # engine primitives
  tensor.[ch]         # multi-dim tensor descriptor (no data ownership)
  alloc.[ch]          # bump-style arena allocator
  graph.[ch]          # DAG of compute nodes (op + 8 inputs)
  weight.[ch]         # .sam3 loader/writer (mmap + FNV-1a hash)
  weight_safetensors.c# SafeTensors reader (vtable)
  quant.[ch]          # Q8_0 block quantization
  half.h              # f16/bf16 <-> f32 helpers
  trace.[ch]          # per-op tracing (debug)
  json/cJSON.[ch]     # vendored JSON parser

src/backend/          # compute backends behind a single vtable
  backend.[ch]        # sam3_backend_ops (init/free/alloc/graph_eval)
  cpu/
    cpu_backend.[ch]
    cpu_dispatch.[ch] # per-op dispatch with dtype/layout variants
    kernels/          # ~45 scalar + SIMD kernels (conv, matmul,
                      # sdpa, layernorm, gelu, rope, cast, ...)
  metal/
    metal_backend.c   # Metal Shading Language backend (macOS)

src/model/            # SAM3 network definition
  sam3_image.[ch]     # composite top-level model
  vl_combiner.[ch]    # wraps ViT + neck + text encoder + tokenizer
  image_encoder.[ch]            # Hiera ViT (32 blocks, windowed attn)
  image_encoder_efficientvit.[ch] # EfficientViT-B1 (MBConv + LiteMLA)
  image_encoder_tinyvit.[ch]   # TinyViT-21M (MBConv + window attn)
  necks.[ch]          # multi-scale FPN
  text_encoder.[ch]   # CLIP text encoder (24 blocks)
  tokenizer.[ch]      # CLIP BPE (byte-level fallback)
  encoder.[ch]        # DETR encoder fusion (6L)
  decoder.[ch]        # DETR decoder (6L, 200 queries)
  prompt_encoder.[ch] # geometry (point/box) encoder (3L)
  mask_decoder.[ch]   # two-way transformer + pixel decoder (point path)
  segmentation.[ch]   # MaskFormer-style head (text/box path)
  model_misc.[ch]     # dot-product query scorer
  position_encoding.[ch]
  graph_helpers.[ch]  # higher-level op builders (MHA, RoPE, ...)
  box_ops.[ch]
  sam3_processor.[ch] # pipeline orchestration, arenas, async text

src/util/
  log.[ch]            # sam3_log_{debug,info,warn,error}
  error.[ch]          # enum sam3_error, str conversion
  image.[ch]          # PNG/JPEG load + letterbox to 1008
  profile.[ch]        # per-stage timing / memory tracking
  threadpool.[ch]     # worker pool (async text encoder)
  mask_{nms,resize,select}.c
  vendor/stb_image*.h

tools/
  sam3_cli.c          # unified CLI entry point (segment, convert, info)
  cli_segment.c       # segment subcommand (inference)
  cli_convert.c       # convert subcommand (SafeTensors -> .sam3)
  cli_info.c          # info subcommand (model metadata)
  cli_common.[ch]     # shared CLI helpers (progress, JSON, exit codes)
  weight_rename.[ch]  # Python -> C name mapping
  weight_conv_perm.[ch] # OIHW -> OHWI permutation

tests/
  test_*.c            # unit + integration + layout tests
  bench_*.c           # perf benchmarks
  test_helpers.h      # tiny assertion macros
```

---

## 3. Engine primitives (`src/core/`)

### 3.1 Tensor descriptor

`struct sam3_tensor` is a pure descriptor. It does **not** own its bytes
— it points into an arena, an mmap'd weight blob, or a backend buffer.
Ownership is always external.

```c
struct sam3_tensor {
    enum sam3_dtype dtype;
    int             n_dims;
    int             dims[SAM3_MAX_DIMS];    /* row-major, max 4 */
    size_t          strides[SAM3_MAX_DIMS]; /* computed */
    void           *data;
    size_t          nbytes;
    int             ephemeral;              /* reuse across eval? */
};
```

Supported dtypes (`sam3_types.h`):

| Enum               | Bytes/elem | Use                        |
|--------------------|------------|----------------------------|
| `SAM3_DTYPE_F32`   | 4          | default compute            |
| `SAM3_DTYPE_F16`   | 2          | weights + activations      |
| `SAM3_DTYPE_BF16`  | 2          | weights + activations      |
| `SAM3_DTYPE_I32`   | 4          | token IDs, indices         |
| `SAM3_DTYPE_I8`    | 1          | passthrough                |
| `SAM3_DTYPE_Q8_0`  | 36 / 32    | block-quantized weights    |

### 3.2 Arena allocator

Inference uses three arenas, all bump-pointer style (`src/core/alloc.c`):

| Arena     | Lifetime                                 |
|-----------|------------------------------------------|
| `model`   | model lifetime — RoPE tables, precomputed|
| `persist` | spans one inference (cached features)    |
| `scratch` | reset between each block / stage         |

All allocations are 16-byte aligned. `sam3_arena_alloc()` zeroes the
returned block; `sam3_arena_alloc_raw()` is the non-zeroing variant
used where the caller immediately overwrites the region. No
`malloc`/`free` appears on the hot path; the arenas are the only source
of memory for tensors built during a graph evaluation.

**Bump-allocator layout**:

```
  model arena (lives for lifetime of sam3_ctx)
  ┌─────────────────────────────────────────────────────────┐
  │ rope_glo_cos │ rope_glo_sin │ rope_win_cos │ pos_embed  │  ...
  └─────────────────────────────────────────────────────────┘
   ^                                                         ^
   base                                                      offset

  persist arena (lives for one sam3_segment() call)
  ┌─────────────────────────────────────────────────────────┐
  │ image[1,3,1008,1008] │ vit_out[5184,1024] │ feat_s0 ... │  ...
  └─────────────────────────────────────────────────────────┘
   ^                     ^                                   ^
   base (saved/restored between stages)                      offset

  scratch arena (reset between every block / every stage)
  ┌─────────────────────────────────────────────────────────┐
  │ qkv │ attn │ mlp │ ...    [RESET]    │ qkv │ attn │ ... │
  └─────────────────────────────────────────────────────────┘
   ^                         ^                              ^
   base                      offset=0                       offset

  sam3_arena_alloc_raw(a, n):          sam3_arena_alloc(a, n):
      aligned = align16(a->offset);        ptr = sam3_arena_alloc_raw(a, n);
      if (aligned + n > a->size)           if (ptr) memset(ptr, 0, n);
          return NULL;                     return ptr;
      ptr = a->base + aligned;
      a->offset = aligned + n;
      return ptr;
```

**Arena lifetimes across one image+segment call**:

```
  time --->
         set_image()         segment()             segment()
            │                   │                    │
  model   │════════════════════════════════════════════════════│  (full ctx)
  persist │════[image+cache]════│═══[image+cache]═══│══════════│
  scratch │═[vit blk 0]═[vit blk 1]═...═[stage A]═[stage B]═...│
              ^reset       ^reset         ^reset     ^reset
```

### 3.3 Compute graph

`struct sam3_graph` is a flat DAG of up to `SAM3_GRAPH_MAX_NODES`
nodes. Each node holds an op code, up to 8 input tensor pointers, one
output, and 4 int params. Model code constructs the graph; the backend
vtable evaluates it.

Implemented ops (`enum sam3_op`):

| Category       | Ops                                                           |
|----------------|---------------------------------------------------------------|
| Linear/tensor  | `MATMUL`, `ADD`, `MUL`, `RESHAPE`, `TRANSPOSE`, `CAST`        |
| Shape          | `CONCAT`, `SLICE`, `EMBED`                                    |
| Activation     | `RELU`, `GELU`, `SIGMOID`, `SILU`                             |
| Normalization  | `LAYERNORM`, `GROUPNORM`, `BIAS_ADD`                          |
| Convolution    | `CONV2D`, `CONV_TRANSPOSE2D`                                  |
| Attention      | `SDPA` (batched multi-head), `ROPE`                           |
| Pooling        | `MAXPOOL2D`, `UPSAMPLE`                                       |
| Softmax        | `SOFTMAX`                                                     |

Higher-level combinators (e.g. a full multi-head attention block,
window partitioning, the two-way attention of the mask decoder) are
expressed as sequences of these primitives in
`src/model/graph_helpers.c`.

**Graph node layout**:

```
  struct sam3_graph                    struct sam3_node
  ┌────────────────────┐               ┌────────────────────────┐
  │ nodes[MAX_NODES]   │────────────>  │ op:        sam3_op     │
  │ n_nodes            │               │ inputs[8]: sam3_tensor*│
  └────────────────────┘               │ output:    sam3_tensor*│
                                       │ params[4]: int         │
                                       └────────────────────────┘

  example: a single encoder block becomes ~10 nodes

    x  ─── LAYERNORM ───► x_ln ─── MATMUL(qkv_w) ──► qkv
                                                       │
                                     RESHAPE + SPLIT <─┘
                                         │   │   │
                                         q   k   v
                                         │   │   │
                                         └─┐ │ ┌─┘
                                          SDPA ──► attn
                                                    │
                                       MATMUL(proj) │
                                                    v
                                       ADD(residual x) ──► y
```

### 3.4 Backend abstraction

```c
struct sam3_backend_ops {
    sam3_error (*init)(sam3_backend *);
    void       (*free)(sam3_backend *);
    sam3_error (*alloc_tensor)(sam3_backend *, sam3_tensor *);
    sam3_error (*graph_eval)(sam3_backend *, sam3_graph *);
};
```

Model code never talks to CPU / Metal directly — it only calls through
the vtable. Two backends ship today:

* **CPU** (`src/backend/cpu/`): ~45 kernel files in
  `kernels/` covering F32, F16, BF16 and Q8_0 where applicable. Hot
  kernels (`cpu_matmul*`, `cpu_conv2d*`, `cpu_sdpa`, `cpu_layernorm*`,
  `cpu_rope*`) have dtype-specialized variants and NEON SIMD paths on
  arm64. Dispatch, layout conversions, and batching live in
  `cpu_dispatch.c`.

* **Metal** (`src/backend/metal/metal_backend.c`): MSL shaders for the
  same op set, with batched execution (up to 4 ViT/text blocks per
  `graph_eval` to amortize kernel-launch overhead) and F16 compute mode.

Backend selection happens at `sam3_init()` time.

```
  model code                    backend vtable                  kernel
  ┌─────────────┐               ┌──────────────────┐           ┌──────┐
  │ build graph │──(nodes)────► │ ops->graph_eval  │           │      │
  │             │               │                  │──────────►│ CPU  │
  └─────────────┘               │ dispatch by      │           │kernel│
         ▲                      │   op + dtype     │           └──────┘
         │                      │   + layout       │              or
         │                      │                  │           ┌──────┐
         │   result tensors     │                  │──────────►│Metal │
         └──(pointer patch)─────│                  │           │shader│
                                └──────────────────┘           └──────┘
```

### 3.5 Weight format and loader

The on-disk format is documented in full in
[`docs/weight-format.md`](weight-format.md). Briefly:

```
  file offset
  0       ┌───────────────────────────────────────────────┐
          │  Header (48 B)                                │
          │    magic   = 0x334D4153 ("SAM3")              │
          │    version = 3                                │
          │    flags, n_tensors                           │
          │    image_size, encoder_dim, decoder_dim       │
          │    n_encoder_layers, n_decoder_layers         │
          │    reserved[3]                                │
  48      ├───────────────────────────────────────────────┤
          │  Tensor descriptors (176 B each)              │
          │  ┌─────────────────────────────────────────┐  │
          │  │ name[128]                               │  │
          │  │ dtype, n_dims, dims[4]                  │  │
          │  │ data_offset, data_size                  │  │
          │  │ reserved (FNV-1a hash cache)            │  │
          │  └─────────────────────────────────────────┘  │
          │  ... repeated n_tensors times                 │
          ├───────────────────────────────────────────────┤
          │  Zero padding to 4096                         │
  4096    ├───────────────────────────────────────────────┤
          │  Data blob (page-aligned start)               │
          │                                               │
          │   tensor 0  [aligned to 64 B]                 │
          │   tensor 1  [aligned to 64 B]                 │
          │   tensor 2  [aligned to 64 B]                 │
          │   ...                                         │
          └───────────────────────────────────────────────┘
```

**Load path** (zero-copy):

```
  .sam3 file on disk
         │
         mmap(PROT_READ, MAP_PRIVATE)
         │
         v
  ┌──────────────────────────────────────┐
  │  mapped region (kernel-managed)      │
  │  header │ descs │ pad │ data blob    │
  └──────────────────────────────────────┘
         │                  ▲
         │                  │
         │   build hash table over names
         │   (FNV-1a)
         v                  │
  ┌──────────────────────┐  │
  │ sam3_weight_file     │──┘
  │   name -> tensor_ptr │
  └──────────────────────┘
         │
         v
   sam3_weight_get("vit.blocks.0.ln1.weight")  -> sam3_tensor
                                                     │
                                                     │ (data pointer
                                                     │  points straight
                                                     v  into the mmap)
```

Loading is **mmap + pointer patching**. `sam3_weight_load()` validates
the header, builds a hash table over all tensor names, and never copies
bulk data — every tensor struct points directly into the mmap'd region.
Lookups are O(1) via FNV-1a.

Readers are pluggable (`struct weight_reader_ops`). The converter
(`tools/cli_convert.c`) uses `weight_safetensors.c` to translate a
`model.safetensors` file into `.sam3`, applying:

1. Name rewriting (`tools/weight_rename.c`) to map PyTorch names to the
   canonical C names the loader looks up.
2. **Conv weight permutation** (`tools/weight_conv_perm.c`) from PyTorch
   OIHW to `.sam3` **OHWI** (`[C_out, KH, KW, C_in]`). This matches the
   NHWC activation layout used throughout the pipeline and removes
   every runtime transpose from the convolution path. (Format v3,
   shipped in commit `96641d3`.)
3. Optional Q8_0 quantization of weight tensors with ≥1024 elements.

---

## 4. Model architecture — what's implemented

All constants below are read from the code; each subsection lists its
source header. The default configuration is the 848M-parameter SAM3
image model.

### 4.1 Top-level composite

`struct sam3_image_model` (`src/model/sam3_image.h:36`) groups every
sub-module and all cross-call cached state:

```c
struct sam3_image_model {
    struct sam3_vl_backbone    backbone;   /* vit + neck + text */
    struct sam3_encoder_fusion encoder;    /* 6L image<-text */
    struct sam3_decoder        decoder;    /* 6L DETR, 200 queries */
    struct sam3_geometry_encoder geom_enc; /* 3L point/box encoder */
    struct sam3_seg_head       seg_head;   /* MaskFormer head */
    struct sam3_mask_decoder   mask_dec;   /* two-way transformer */
    struct sam3_dot_scorer     scorer;     /* per-query confidence */

    struct sam3_tensor *cached_text_features;  /* [L,256] */
    struct sam3_tensor *cached_feat_s0_nhwc;   /* 2x scale */
    struct sam3_tensor *cached_feat_s1_nhwc;   /* 1x scale */
    struct sam3_tensor *cached_feat_4x_nhwc;   /* 4x scale */
    int image_encoded;
};
```

It exposes a two-phase API:

* `sam3_image_model_encode()` — runs the vision backbone once per new
  image and caches the multi-scale features.
* `sam3_image_model_segment()` — runs the per-prompt stages
  (geometry encoder → encoder fusion → decoder → seg head / mask
  decoder → scorer), resetting `scratch` between stages so peak memory
  is bounded by the largest single stage, not the sum.

**Two-phase call sequence** (sharing cached features across prompts):

```
                 image_1                   image_2
                    │                          │
                    v                          v
       set_image  ┌────┐                     ┌────┐
                  │ENC │                     │ENC │    (expensive;
                  └────┘                     └────┘    ~100s of ms)
                    │                          │
                    ▼                          ▼
                 cache                       cache
                    │                          │
                    ├──segment(points)         ├──segment(text)
                    │    └─► masks_1           │    └─► masks_4
                    ├──segment(text)           ├──segment(box)
                    │    └─► masks_2           │    └─► masks_5
                    └──segment(box)            └──segment(box+text)
                         └─► masks_3                └─► masks_6

  All segment() calls reuse cached_feat_s0/s1/4x,
  and (if set) cached_text_features.
```

### 4.2 Vision backbone (multi-backbone dispatch)

The vision backbone is selected at init time via `backbone_type` in the
`sam3_vl_backbone` struct. Three backbones are supported:

| Backbone | Type enum | Grid | Input | Mask res | Key file |
|---|---|---|---|---|---|
| Hiera | `SAM3_BACKBONE_HIERA` | 72x72 | 1008 | 288x288 | `image_encoder.[ch]` |
| TinyViT-21M | `SAM3_BACKBONE_TINYVIT` | 32x32 | 1008 | 128x128 | `image_encoder_tinyvit.[ch]` |
| EfficientViT-B1 | `SAM3_BACKBONE_EFFICIENTVIT` | 16x16 | 512 | 64x64 | `image_encoder_efficientvit.[ch]` |

All backbones output `[grid^2, 1024]` and share the same neck, text
encoder, prompt encoder, mask decoder, and segmentation head. Dispatch
happens in `vl_combiner.c` via a `switch` on `backbone_type` in three
places: `_init`, `_load`, and `_build_vision`.

```c
union {
    struct sam3_vit        vit;    /* Hiera */
    struct sam3_efficientvit evit; /* EfficientViT-B1 */
    struct sam3_tinyvit    tvit;   /* TinyViT-21M */
} enc;
```

#### 4.2.1 Hiera ViT (`image_encoder.h`)

SAM3's default backbone is a 32-block Vision Transformer with
local/global attention and RoPE. The struct (`sam3_vit`) hardcodes
no dimensions; values come from `sam3_vit_init()`. The config is:

| Field              | Value | Note                                  |
|--------------------|-------|---------------------------------------|
| `img_size`         | 1008  | preprocessed input                    |
| `patch_size`       | 14    | conv stride                           |
| `grid_size`        | 72    | `img_size / patch_size`               |
| `n_patches`        | 5184  | `grid_size^2`                         |
| `embed_dim`        | 1024  |                                       |
| `depth`            | 32    | transformer blocks                    |
| `n_heads`          | 16    |                                       |
| `window_size`      | 24    | windowed attention window             |
| `mlp_dim`          | 4736  | ≈ `embed_dim * 4.625`                 |

Per-block weights:

```c
struct {
    sam3_tensor *ln1_w, *ln1_b;        /* pre-attn LayerNorm */
    sam3_tensor *qkv_w, *qkv_b;        /* fused QKV projection */
    sam3_tensor *proj_w, *proj_b;      /* attention out proj */
    sam3_tensor *ln2_w, *ln2_b;        /* pre-MLP LayerNorm */
    sam3_tensor *mlp_fc1_w, *mlp_fc1_b;/* GELU MLP */
    sam3_tensor *mlp_fc2_w, *mlp_fc2_b;
    int is_global;                     /* 1 at blocks 7,15,23,31 */
} layers[32];
```

**Patch embedding** is a `conv2d` with 14×14 kernel, stride 14. Weights
are stored in OHWI `[embed_dim, 14, 14, 3]` so the NHWC backend can
consume them without a transpose.

**Positional embedding** is tiled from the pretrain-resolution
`[1, 577, embed_dim]` blob stored in the weight file to a full
`[n_patches, embed_dim]` table. Tiling is lazy — done on the first
`sam3_vit_build()` call and stashed in the model arena.

**Attention pattern**:
* Global attention at layers 7, 15, 23, 31 (4 layers out of 32).
* Mask-free **windowed attention** at every other layer, partitioning
  the 72×72 patch grid into 24×24 windows.
* RoPE frequencies are precomputed into two tables: a full-grid
  `rope_glo_{cos,sin}` for global layers, and a tiny
  `rope_win_local_{cos,sin}` of size `[ws*ws, head_dim/2]` reused
  across all windows.

**Per-block evaluation.** The critical optimization in this backbone is
that `sam3_vit_build()` evaluates one block at a time, resets the
scratch arena in between, and keeps only the current `[n_patches, 1024]`
activation buffer in the `persist` arena. This drops peak memory from
~55 GB (holding every block's activations) to ~2.5 GB. The Metal
backend further batches 4 consecutive blocks per `graph_eval` call to
cut kernel-launch overhead (commit `15d2c74`).

**ViT forward pass layout**:

```
  image [3, 1008, 1008]
        │
        v
  ┌───────────────────────────────┐
  │ patch_embed                   │
  │   conv2d  k=14  s=14          │
  │   OHWI weights [1024,14,14,3] │
  └───────────────────────────────┘
        │
        v  [1, 72, 72, 1024]  (NHWC)
        │
    reshape / flatten
        │
        v  [5184, 1024]  +  pos_embed[5184, 1024]
        │
        v
  ┌───────────────────────────────┐
  │ ln_pre                        │
  └───────────────────────────────┘
        │
        │   ┌─────────────────── block 0 ────────────────┐
        └──►│                                            │
            │   x ── ln1 ── qkv ── SDPA(windowed) ── proj│
            │   │                                       │
            │   └────────────────── + ──────────────────┘
            │                       │                   │
            │   ┌───────────────────┘                   │
            │   │                                        │
            │   └── ln2 ── fc1 ── GELU ── fc2 ──► + ────┘
            └─────────────────────────────────────│──────┘
                                                  │
                                                  │  (scratch RESET)
                                                  v
                                            ┌─ block 1 ─┐
                                            │   same    │
                                            │  shape    │
                                            └─────┬─────┘
                                                  v
                                                 ...
                                                  │
                                            ┌─ block 7 ─┐
                                            │  GLOBAL   │ <- full-image
                                            │   attn    │    attention
                                            └─────┬─────┘
                                                  v
                                                 ...
                                                  │
                                            ┌─ block 31 ┐
                                            │  GLOBAL   │
                                            └─────┬─────┘
                                                  v
                                       [5184, 1024]  (persist arena)
```

**Single ViT block** (pre-norm residual):

```
              x  [5184, 1024]
              │
              ├─────────────────────────┐
              v                         │
          ┌───────┐                     │
          │ ln1   │                     │
          └───┬───┘                     │
              │                         │
              v                         │
    ┌───────────────────┐               │
    │ qkv = x @ qkv_w   │               │
    │      + qkv_b      │               │
    │ shape [5184,3072] │               │
    └─────────┬─────────┘               │
              │                         │
     split into q, k, v  [5184,16,64]   │
              │                         │
     apply RoPE to q, k                 │
              │                         │
     (window partition                  │
      if !is_global)                    │
              │                         │
              v                         │
    ┌───────────────────┐               │
    │ SDPA (multi-head) │               │
    │  softmax(qk/√d)v  │               │
    └─────────┬─────────┘               │
              │                         │
     (window unpartition                │
      if !is_global)                    │
              │                         │
              v                         │
    ┌───────────────────┐               │
    │ out = a @ proj_w  │               │
    │     + proj_b      │               │
    └─────────┬─────────┘               │
              │                         │
              +◄────────────────────────┘ residual
              │
              ├─────────────────────────┐
              v                         │
          ┌───────┐                     │
          │ ln2   │                     │
          └───┬───┘                     │
              │                         │
              v                         │
    ┌───────────────────┐               │
    │ fc1 -> [5184,4736]│               │
    │ GELU              │               │
    │ fc2 -> [5184,1024]│               │
    └─────────┬─────────┘               │
              │                         │
              +◄────────────────────────┘ residual
              │
              v
          y [5184, 1024]
```

**Windowed vs global attention** (mask-free windowing):

```
  72x72 patch grid                              72x72 patch grid
  (block 0..6, 8..14, ...)                      (block 7, 15, 23, 31)

   ┌──┬──┬──┬──┬──┬──┐                           ┌─────────────────┐
   │ws│ws│ws│ws│ws│ws│                           │                 │
   ├──┼──┼──┼──┼──┼──┤                           │                 │
   │ws│ws│ws│ws│ws│ws│                           │    one big      │
   ├──┼──┼──┼──┼──┼──┤   each ws=24x24           │    attention    │
   │ws│ws│ws│ws│ws│ws│   attends only to         │    over all     │
   ├──┼──┼──┼──┼──┼──┤   its own window          │    5184 tokens  │
   │ws│ws│ws│ws│ws│ws│                           │                 │
   ├──┼──┼──┼──┼──┼──┤                           │                 │
   │ws│ws│ws│ws│ws│ws│                           │                 │
   ├──┼──┼──┼──┼──┼──┤                           │                 │
   │ws│ws│ws│ws│ws│ws│                           │                 │
   └──┴──┴──┴──┴──┴──┘                           └─────────────────┘
   uses rope_win_local                           uses rope_glo
   (one [ws*ws, d/2] table                       ([n_patches, d/2]
    shared by all windows)                        full tables)
```

#### 4.2.2 TinyViT-21M (`image_encoder_tinyvit.h`)

A lightweight 4-layer encoder using MBConv blocks (layer 0) and
windowed attention with learned position bias (layers 1-3). ~21M
parameters, produces 128x128 masks at 1008px input.

| Field | Value | Note |
|---|---|---|
| `img_size` | 1008 | same as Hiera |
| `embed_dims` | [96, 192, 384, 576] | per-layer channels |
| `depths` | [2, 2, 6, 2] | blocks per layer |
| `num_heads` | [3, 6, 12, 18] | per-layer heads |
| `window_sizes` | [7, 7, 14, 7] | per-layer window size |
| `grid_size` | 32 | output spatial dim |
| `embed_dim` | 1024 | after projection head |
| `mlp_ratio` | 4 | MLP expansion factor |

**Architecture**:

```
  image [3, 1008, 1008]
        │
   patch_embed: 2x Conv2d_BN(3x3, stride 2)
        │
        v  [1, 252, 252, 96]
        │
   Layer 0 (ConvLayer):
     2x MBConv(expand=4, DW 3x3) + PatchMerging(stride 2)
        │
        v  [1, 126, 126, 192]
        │
   Layer 1 (BasicLayer):
     2x TinyViTBlock(ws=7) + PatchMerging(stride 2)
        │
        v  [1, 63, 63, 384]
        │
   Layer 2 (BasicLayer):
     6x TinyViTBlock(ws=14, padded to 70) + PatchMerging(stride 2)
        │
        v  [1, 32, 32, 576]
        │
   Layer 3 (BasicLayer):
     2x TinyViTBlock(ws=7, padded to 35), no downsample
        │
        v  [1, 32, 32, 576]
        │
   Projection: Conv1x1(576->1024) + BN + GELU + Conv3x3
        │
        v  [1024, 1024]
```

**TinyViTBlock** (pre-norm residual + local conv):

```
  x [H*W, C]
     │
     ├───────────────────┐
     v                   │
  window_partition       │
     │                   │
  ┌─────────────────┐    │
  │ LN (pre-norm)   │    │
  │ QKV (interleaved│    │
  │   per-head)     │    │
  │ Q@K^T + bias    │    │
  │ softmax → @V    │    │
  │ proj            │    │
  └──┬──────────────┘    │
     │                   │
  window_unpartition     │
     │                   │
     +◄──────────────────┘ residual
     │
  local_conv (3x3 DW + BN)
     │
     ├───────────────────┐
     v                   │
  ┌─────────────────┐    │
  │ LN → fc1 → GELU │    │
  │   → fc2         │    │
  └──┬──────────────┘    │
     │                   │
     +◄──────────────────┘ residual
     │
     v  y [H*W, C]
```

**Per-block evaluation.** Unlike Hiera which batches 4 blocks per
`graph_eval`, TinyViT evaluates each block individually due to the
higher scratch memory usage of windowed attention with padding. Each
block's graph is built, evaluated, and the result copied to a persist
buffer before scratch is reset.

**Window padding.** Non-divisible spatial sizes (e.g. 63%14 for
layer 2) are handled by concatenating zero tensors along H/W axes
before partition, then slicing back to original size after unpartition.

**Attention bias.** At load time, the compact `[n_heads, n_offsets]`
attention bias is expanded to full `[n_heads, ws^2, ws^2]` using
absolute position differences, avoiding gather ops at inference.

#### 4.2.3 EfficientViT-B1 (`image_encoder_efficientvit.h`)

A mobile-optimized encoder using MBConv + LiteMLA (lightweight
multi-scale linear attention). ~9M parameters, fastest backbone with
64x64 masks at 512px input.

| Field | Value |
|---|---|
| `img_size` | 512 |
| `width_list` | [24, 48, 96, 192, 384] |
| `depth_list` | [1, 2, 3, 4, 6] |
| `attn_dim` | 32 |
| `grid_size` | 16 |
| `embed_dim` | 1024 |

The architecture follows a stem + 4-stage design with MBConv blocks in
early stages and MBConv + LiteMLA in later stages. Like TinyViT, it
uses per-stage evaluation to bound scratch memory.

### 4.3 Feature pyramid neck (`necks.h`)

`sam3_neck` turns the backbone output `[grid^2, backbone_dim]` into a
set of NHWC feature maps at different scales, all projected to
`d_model = 256`.

```c
struct sam3_neck {
    int d_model;       /* 256 */
    int backbone_dim;  /* derived from encoder (Hiera: 1024, TinyViT: 576, ...) */
    int n_scales;      /* up to 4 */
    int grid_size;     /* Hiera: 72, TinyViT: 32, EfficientViT: 16 */
    struct {
        float scale_factor;    /* e.g. 4.0, 2.0, 1.0, 0.5 */
        int   n_convs;
        sam3_tensor *conv_w[4], *conv_b[4];
        int is_transpose[4];   /* ConvTranspose vs Conv */
        int kernel_size[4];
        int gelu_after[4];
        int has_maxpool;
    } stages[4];
};
```

Each stage chains: optional `MaxPool` / `ConvTranspose2d` (spatial
rescale in backbone-dim space) → 1×1 `Conv2d` projection to
`d_model` → 3×3 `Conv2d` spatial mixing → GELU. Every output is NHWC
`[1, H_i, W_i, 256]`, and all downstream consumers (seg head, mask
decoder, geometry cross-attention) read NHWC directly — no bridge
permute is needed after commit `96641d3`.

**FPN construction**:

```
            backbone_out [grid^2, 1024]  ──► reshape [1, grid, grid, backbone_dim]
            (Hiera example: [5184, 1024] -> [1, 72, 72, 1024])
                                           │
                 ┌─────────────────┬───────┼──────┬─────────────┐
                 │                 │       │      │             │
                 v                 v       v      v             v
           ┌──────────┐      ┌─────────┐ ┌────┐ ┌─────────┐ ┌────────┐
           │ stage 0  │      │ stage 1 │ │stg2│ │ stage 3 │ │ ...    │
           │ up 4x    │      │ up 2x   │ │ 1x │ │ down 2x │ │        │
           │(ConvTr2d)│      │(ConvTr2d│ │    │ │(MaxPool)│ │        │
           └────┬─────┘      └────┬────┘ └──┬─┘ └────┬────┘ └────────┘
                v                 v         v        v
             [1,288,           [1,144,   [1,72,  [1,36,
              288,1024]         144,1024] 72,1024] 36,1024]
                │                 │         │        │
          ┌─────┴─────┐           │         │        │
          │1x1 conv   │           │         │        │
          │1024→256   │(same      (same    (same
          └─────┬─────┘ idea)     idea)     idea)
                │
          ┌─────┴─────┐
          │3x3 conv   │
          │256→256    │
          │+ GELU     │
          └─────┬─────┘
                v
             [1,288,288,256]
                │                  │         │         │
                v                  v         v         v
                feat_4x       feat_s0    feat_s1    feat_lo
               (used by       (used by   (used by
                mask_dec       seg_head   seg_head,
                skip conn)     FPN + skip encoder
                               mask_dec)  fusion)
```

The main image_features tensor fed to the encoder fusion is the 1×
scale `[1, grid, grid, 256]`; the 2× and 4× scales flow as FPN skip
connections into both the `seg_head` and the `mask_decoder`.

**sam2_fpn_layers.** EfficientSAM3 checkpoints (EfficientViT and
TinyViT) include a second FPN neck (`sam2_fpn_layers`) used by the
tracker. It is initialized and loaded with the same config as the
primary neck but with a different weight prefix. The `has_sam2_neck`
flag on `sam3_vl_backbone` tracks whether it was loaded.

### 4.4 Text tower — tokenizer + CLIP encoder

**`sam3_tokenizer`** (`tokenizer.h`) implements CLIP's 49408-entry BPE.
It has two modes:

1. **Byte-level fallback** — used immediately after `sam3_init()`,
   needs no vocab file, produces incorrect IDs but lets the pipeline
   run for testing.
2. **Full BPE** — loaded on demand from `bpe_simple_vocab_16e6.txt.gz`
   via `sam3_load_bpe()`. Uses FNV-1a hash tables for `token -> id`
   and merge-rank lookup, and a small direct-mapped
   `sam3_bpe_cache_entry` cache (1024 slots × 16 IDs) so repeated
   words skip the O(n²) merge loop entirely.

The pipeline is:
lowercase → pre-tokenize → BPE merge → vocab lookup → pad to
`[SOT, ..., EOT, pad…]` of length 32 (`SAM3_TOKENIZER_CONTEXT_LEN`).

```
  "A red apple."
         │
         │ lowercase + unicode normalize
         v
  "a red apple."
         │
         │ regex pre-tokenize
         v
  ["a", "red", "apple", "."]
         │
         │ bpe_cache lookup (fast path)
         │     ┌───────────────────────────────┐
         │     │ cache: 1024 direct-mapped     │
         │     │ key: word string (≤ 64 bytes) │
         │     │ val: up to 16 token IDs       │
         │     └───────────────────────────────┘
         │ miss? run merge loop:
         │     split into chars → pair-rank min → merge → repeat
         v
  [320, 736, 3782, 269]  (ids from vocab)
         │
         │ prepend SOT=49406, append EOT=49407
         │ pad with 0 to length 32
         v
  [49406, 320, 736, 3782, 269, 49407, 0, 0, ..., 0]
  shape: [1, 32] i32
```

**`sam3_text_encoder`** (`text_encoder.h`) is a straight 24-block
CLIP transformer:

| Field         | Value  |
|---------------|--------|
| `width`       | 1024   |
| `n_heads`     | 16     |
| `n_layers`    | 24     |
| `context_len` | 77 (pos-embed table size; CLIP default) |
| `vocab_size`  | 49408  |
| `d_model`     | 256 (output) |

Note: `context_len = 77` is the positional-embedding table size
inherited from CLIP pretraining (`pos_embedding [77, 1024]`). At
inference, only the first `SAM3_TOKENIZER_CONTEXT_LEN = 32` positions
are indexed — the remaining 45 rows are never used.

Each layer is pre-norm self-attention + `GELU` MLP. After the last
block, a final LayerNorm plus `text_projection` (`[256, 1024]`) maps
the per-token `[seq_len, 1024]` stream to the `d_model = 256` space
used by the rest of the network. The pooled text feature is extracted
from the EOT token position.

A per-block debug path (`sam3_text_encoder_build_perblock`) exists for
fixture comparison against the Python reference.

**Text encoder pipeline**:

```
  token_ids [1, 32] i32
         │
         v
  ┌────────────────────────────────────┐
  │ token_embedding  [49408, 1024]     │
  │   lookup + pos_embedding [32,1024] │
  └────────────────────────────────────┘
         │
         v  x [32, 1024]
         │
    ┌────┼─── block 0 ───────────────────┐
    │    │                               │
    │    ├── ln1 ── qkv ── SDPA ── proj ─┤── +
    │    │                               │
    │    ├── ln2 ── fc1 ── GELU ── fc2 ──┤── +
    │    │                               │
    └────┴───────────────────────────────┘
         │
         v  (repeat for 24 blocks; Metal batches 4 at a time)
         │
  ┌────────────────────┐
  │ ln_final  [1024]   │
  └────────────────────┘
         │
         v  [32, 1024]
         │
  ┌──────────────────────────────┐
  │ text_projection [256, 1024]  │
  │   (+ bias [256])             │
  └──────────────────────────────┘
         │
         ├───────────────────┐
         v                   v
  [32, 256]            pooled [256]
  per-token            (from EOT position)
  text_features
```

**Async execution.** `sam3_set_text()` tokenizes on the caller's thread
and spawns a worker (using `src/util/threadpool.c`) that runs the text
encoder on a CPU backend in parallel with the main-thread image
encoder. The result is consumed by the next `sam3_segment()` call. The
legacy inline path still works when `sam3_set_text` is never called.
(Commit `0d9da81`.)

```
  main thread                         worker thread (CPU backend)
  ──────────────────                  ────────────────────────────

  sam3_set_text("cat")                      │
     ├─ tokenize [32] i32 ──── spawn ─────► │
     │                                      v
     │                              run text_encoder on
     │                              CPU backend (24 blocks)
     v                                      │
  sam3_set_image(pixels)                    │
     ├─ run ViT on main backend             │
     │   (Metal or CPU)                     │
     │                                      │
     │ ◄──────── join ──────────────────────┤
     │     cached_text_features [32, 256]   │
     │                                      │
     v
  sam3_segment(prompts)
     (uses both cached features directly)
```

### 4.5 Encoder fusion (`encoder.h`)

6-layer DETR-style encoder that fuses image features with text
features. Each layer runs:

1. **Self-attention** on image features (pre-norm + residual).
2. **Cross-attention**: image queries attend to text keys/values
   (pre-norm + residual).
3. **FFN** (ReLU, hidden 2048) + pre-norm + residual.

Constants (`SAM3_ENC_FUSION_MAX_LAYERS = 6`):

| Field     | Value |
|-----------|-------|
| `d_model` | 256   |
| `n_heads` | 8     |
| `n_layers`| 6     |
| `d_ffn`   | 2048  |

A final `output_ln` is applied after all 6 layers. There is a
per-layer builder (`sam3_encoder_fusion_build_layer`) used to evaluate
the encoder one layer at a time so MLX buffers can be recycled.

**One encoder-fusion layer**:

```
            image_feat [5184, 256]         text_feat [32, 256]
                 │                                │
                 ├──────────────┐                 │
                 v              │                 │
             ┌───────┐          │                 │
             │ sa_ln │          │                 │
             └───┬───┘          │                 │
                 │              │                 │
                 v              │                 │
       ┌──────────────────┐     │                 │
       │  self-attention  │     │                 │
       │  q=k=v=image     │     │                 │
       │  (+ enc_pos)     │     │                 │
       └─────────┬────────┘     │                 │
                 │              │                 │
                 +◄─────────────┘ residual        │
                 │                                │
                 ├──────────────┐                 │
                 v              │                 │
             ┌───────┐          │                 │
             │ ca_ln │          │                 │
             └───┬───┘          │                 │
                 │              │                 │
                 v              │                 │
       ┌──────────────────┐     │                 │
       │  cross-attention │◄────┼─────────────────┤
       │  q = image       │     │                 │
       │  k = v = text    │     │                 │
       └─────────┬────────┘     │                 │
                 │              │                 │
                 +◄─────────────┘ residual
                 │
                 ├──────────────┐
                 v              │
             ┌───────┐          │
             │ffn_ln │          │
             └───┬───┘          │
                 │              │
                 v              │
       ┌──────────────────┐     │
       │  FFN ReLU, 2048  │     │
       └─────────┬────────┘     │
                 │              │
                 +◄─────────────┘ residual
                 │
                 v
            image_feat' [5184, 256]
```

### 4.6 DETR decoder (`decoder.h`)

```c
#define SAM3_DEC_MAX_LAYERS  6
#define SAM3_DEC_NUM_QUERIES 200
```

200 learned object queries run through 6 transformer layers. Each
layer has four transformer sub-blocks plus a box-refinement head:

1. Self-attention on queries.
2. Text cross-attention to the text encoder output (`text_features`).
3. Vision cross-attention to the fused encoder output (`enc_features`).
4. FFN (ReLU, hidden 2048).
5. Box refinement: a 3-layer MLP applied to the query embeddings that
   produces a delta added to the running box accumulator.

This is DAB-DETR conditional queries — positional embeddings are
computed from the current reference boxes via a 2-layer
`ref_point_head` MLP (`sam3_decoder_compute_query_pos`) on every layer.

The decoder exposes fine-grained substep builders
(`sam3_decoder_build_sa`, `_tca`, `_ca`, `_ffn`) for per-step
evaluation and weight-level debugging. `sam3_decoder_build_final`
applies the output LayerNorm after all 6 layers.

**One decoder layer with iterative box refinement**:

```
  q [200, 256]                             ref_boxes [200, 4]
  query_embed                                      │
        │                                          │
        │                          sine pos enc    │
        │                                │         │
        │                                v         │
        │                      ┌──────────────┐    │
        │                      │ ref_point_   │    │
        │                      │ head MLP     │    │
        │                      └──────┬───────┘    │
        │                             │            │
        │                      query_pos [200,256] │
        │                             │            │
        │                             v            │
        │   ┌────── sa ─────────────┐ │            │
        └──►│                       │◄┘            │
            │ Q = K = q + query_pos │              │
            │ V = q                 │              │
            │ self-attn + ln + res  │              │
            └─────────┬─────────────┘              │
                      │                            │
                      ├─ tca (text cross-attn) ──┐ │
                      │   K=V=text_features       │ │
                      │   + ln + res              │ │
                      ├─◄───────────────────────┘ │
                      │                             │
                      ├─ ca (vision cross-attn) ──┐ │
                      │   K=V=enc_features        │ │
                      │   (+ enc_pos on K)        │ │
                      │   + ln + res              │ │
                      ├─◄───────────────────────┘ │
                      │                             │
                      ├─ ffn (ReLU 2048) + res ───┤ │
                      │                             │
                      v                             v
                  q' [200,256]              ┌──────────────┐
                      │                     │box_fc1->ReLU │
                      └────────────────────►│box_fc2->ReLU │
                                            │box_fc3       │
                                            └──────┬───────┘
                                                   │
                                                   v
                                          delta [200, 4]
                                                   │
                                          ref_boxes += delta
                                          (sigmoid-parametrized)
```

All 6 layers share the same structure but have their own weights; the
boxes accumulator is updated at every layer (DAB-DETR style) and
re-used as position input for the next layer's conditional queries.

### 4.7 Geometry encoder (`prompt_encoder.h`)

Encodes point and box prompts into dense tokens the rest of the network
can consume. It is a 3-layer transformer with a fairly specific setup:

1. Project raw coords: `point_proj` (`[256, 2]`), `box_proj`
   (`[256, 4]`). Add a learned `label_embed[2, 256]` for
   foreground/background point labels.
2. Prepend a learned `cls_token[1, 256]`.
3. **Pre-encoder projection**: `post_proj` (Linear `[256, 256]`) + a
   LayerNorm. (The field is called `post_proj` in C but it runs
   *before* the encoder layers — it was renamed from Python's
   `final_proj`.)
4. Three encoder layers, each consisting of:
   * Pre-norm self-attention over prompt tokens.
   * Cross-attention with image features as KV. Query gets sinusoidal
     position encoding from `posenc_proj`; image KV gets
     `pool_proj` (the image features are LayerNormed first via
     `img_pre_norm`). Position encoding is added to **keys only**.
   * FFN (ReLU, hidden 2048).
5. Final `encode_norm` LayerNorm.

Output is `[N+1, 256]` (CLS + each input prompt token).

**Geometry encoder pipeline**:

```
  raw prompts:
    point (x, y, label)        box (x1, y1, x2, y2)
           │                           │
           v                           v
     point_proj  [256, 2]        box_proj  [256, 4]
           │                           │
           │   + label_embed[2,256]    │
           │     for point labels      │
           v                           v
           └────────────┬──────────────┘
                        │
                        │  prepend cls_token [1, 256]
                        v
                   [N+1, 256]
                        │
                ┌───────┴───────┐
                │  post_proj    │   (Linear [256,256] + LayerNorm)
                │  + norm       │   — despite the name, applied BEFORE
                └───────┬───────┘     the encoder layers
                        │
                        v
          ┌──── layer 0 ──────────────────────────────┐
          │   norm1 -> self-attn(Q=K=V)  + residual    │
          │                                            │
          │   norm2 -> cross-attn                      │
          │        Q = prompt tokens (+ posenc_proj)   │
          │        K = V = image features              │
          │              (LayerNormed + pool_proj)     │
          │        + residual                          │
          │                                            │
          │   norm3 -> FFN(ReLU, 2048)   + residual    │
          └───────────────────┬────────────────────────┘
                              │
                              v
                         [layer 1] -> [layer 2]
                              │
                              v
                       ┌─────────────┐
                       │ encode_norm │
                       └──────┬──────┘
                              │
                              v
                       [N+1, 256]  geometry tokens
```

### 4.8 Segmentation head (`segmentation.h`)

Used for **text- and box-prompt** segmentation. It's a MaskFormer-style
pixel decoder fed by the 6-layer encoder's output plus the 2× and 4×
backbone features, combined with a mask embedder MLP applied to the
decoder query outputs.

```c
struct sam3_seg_head {
    int d_model;       /* 256 */
    int n_attn_heads;  /* 8  */
    struct {
        sam3_tensor *conv_w, *conv_b; /* 3x3 conv, OHWI */
        sam3_tensor *gn_w,   *gn_b;   /* GroupNorm(8)  */
    } fpn[3];
    sam3_tensor *inst_proj_w, *inst_proj_b; /* 1x1 conv */
    struct { sam3_tensor *w, *b; } mask_mlp[3];
    /* prompt cross-attention (separate QKVO) */
    sam3_tensor *pxattn_q_w/_b, *pxattn_k_w/_b,
                *pxattn_v_w/_b, *pxattn_o_w/_b,
                *pxattn_norm_w/_b;
};
```

Pipeline (`sam3_seg_head_build`):

1. Reshape encoder output to NHWC `[1, 72, 72, 256]`.
2. **FPN**: 3 stages of (interpolate → add skip from 2× or 4× backbone
   feature → 3×3 Conv → GroupNorm(8) → ReLU).
3. **Instance projection**: 1×1 Conv on the pixel features.
4. **Mask embedder**: 3-layer MLP on the 200 decoder queries.
5. **Mask logits** = `mask_embed @ inst_features^T` — a pure dot
   product between each query's embedding and every pixel's instance
   feature.

There is also `sam3_seg_head_build_cross_attn` for an optional prompt
cross-attention pass over the encoder states + text features, used
when the caller needs that intermediate materialized.

**Segmentation head pipeline** (text/box prompt path):

```
  encoder_states [5184, 256]         queries [200, 256]
         │                                     │
         │ reshape                             v
         v                           ┌───────────────────┐
   [1, 72, 72, 256] NHWC             │ mask_mlp (3-layer │
         │                           │ linear + ReLU)    │
         │                           └─────────┬─────────┘
         │                                     │
         │        feat_2x [1,144,144,256]      │
         │        feat_4x [1,288,288,256]      │
         │                                     │
         v                                     │
  ┌───────────────────────────┐                │
  │  FPN pixel decoder        │                │
  │  ─────────────────        │                │
  │  upsample 72  -> 144      │                │
  │    + skip(feat_2x)        │                │
  │    -> 3x3 conv -> GN -> ReLU               │
  │  upsample 144 -> 288      │                │
  │    + skip(feat_4x)        │                │
  │    -> 3x3 conv -> GN -> ReLU               │
  │  one more stage           │                │
  └──────────┬────────────────┘                │
             │                                 │
             v                                 │
  ┌───────────────────────────┐                │
  │ inst_proj  1x1 conv       │                │
  │ 256 -> 256                │                │
  └──────────┬────────────────┘                │
             │                                 │
             v                                 │
   [1, H_f, W_f, 256] pixel_features           │
             │                                 │
             │  reshape to [H_f*W_f, 256]      │
             │                                 │
             └──────────────┬──────────────────┘
                            │
                            v
                 ┌─────────────────────┐
                 │ matmul:             │
                 │ mask_mlp(Q) @       │
                 │ pixel_features^T    │
                 └──────────┬──────────┘
                            │
                            v
                   [200, H_f, W_f]
                   per-query mask logits
```

### 4.9 Point-prompt mask decoder (`mask_decoder.h`)

The classic two-way SAM mask decoder — used for the point-prompt
pathway. It produces 4 mask logits and an IoU score per prompt.

```c
#define SAM3_MASK_DEC_LAYERS  2
#define SAM3_MASK_DEC_MASKS   4
#define SAM3_MASK_DEC_D_INNER 128
```

Learned tokens:

* `mask_tokens[4, 256]` — one per output mask.
* `iou_token[1, 256]` — drives the IoU prediction MLP.
* `obj_score_token[1, 256]` — drives the object-score head.

The **2-layer two-way transformer** alternates bidirectional
cross-attention:

1. Self-attention on tokens (256-dim, 8 heads).
2. Token → image cross-attention at **128-dim** internal projections
   (`ca_ti_*`).
3. Token FFN (ReLU).
4. Image → token cross-attention at 128-dim (`ca_it_*`).

After 2 layers, a final token → image cross-attention `final_*` plus
LayerNorm produces the updated tokens.

The **pixel decoder** is 2 transposed convolutions (also OHWI) that
upsample the image features to the final mask resolution, with a
LayerNorm between them and GELU after. Multi-scale skip connections
`conv_s0_w/b` (`[32, 1, 1, 256]`) and `conv_s1_w/b`
(`[64, 1, 1, 256]`) blend in the 2× and 4× scale backbone features.

Four **hypernetwork MLPs** (`hyper[4]`) convert each mask token from
256 → 256 → 256 → 32. The 32-dim hyper outputs are dotted against the
pixel decoder's 32-channel feature map to produce the 4 mask logits.
A separate **IoU prediction MLP** maps the iou token through
256 → 256 → 4.

A `no_mask_embed[1, 256]` is used as the dense prompt when no mask
prompt is provided, and a fixed `pe_gaussian[2, 128]` drives the
sinusoidal positional encoding.

**Two-way transformer block**:

```
  tokens [6, 256]                   image_tokens [N_img, 256]
  (4 mask + 1 iou + 1 obj)                   │
         │                                   │
         ├────── sa (tokens) ──┐             │
         │                     │             │
         v                     │             │
  ┌──────────────┐             │             │
  │ self-attn    │             │             │
  │ 256-dim, 8h  │             │             │
  └──────┬───────┘             │             │
         │                     │             │
         +◄────────────────────┘ residual    │
         │                                   │
         │── ca_ti (tokens -> image) ──┐     │
         │                             │     │
         v                             │     │
  ┌──────────────┐                     │     │
  │ cross-attn   │◄────────────────────┼─────┤
  │ Q: 256->128  │                     │     │
  │ K: 256->128  │                     │     │
  │ V: 256->128  │                     │     │
  │ out: 128->256│                     │     │
  └──────┬───────┘                     │     │
         │                             │     │
         +◄────────────────────────────┘     │
         │                                   │
         ├── ffn (ReLU) ── residual ──┐      │
         │                            │      │
         v                            │      │
  ┌──────────────┐                    │      │
  │  MLP 256->   │                    │      │
  │      2048->  │                    │      │
  │      256     │                    │      │
  └──────┬───────┘                    │      │
         +◄───────────────────────────┘      │
         │                                   │
         │       ┌── ca_it ────────┐         │
         │       │ image tokens    │         │
         │       │ get updated by  │         │
         │       │ cross-attention │         │
         │       │ TO the tokens:  │         │
         │       │ Q from image,   │         │
         │       │ K,V from tokens │◄────────┘
         │       │ (128-dim inner) │         ▲
         │       └────────┬────────┘         │
         │                │                  │
         │                +◄─────────────────┘ residual
         │                │
         v                v
   tokens'         image_tokens'
```

Two such layers run, followed by a final token → image cross-attention
(`final_*`) and LayerNorm.

**Mask prediction path**:

```
  image_tokens [1, H, W, 256]  (from 2-way transformer)
          │
          v
  ┌──────────────────────────┐
  │ pixel decoder            │
  │   up_conv1 (OHWI 2x2)    │
  │   + skip conv_s1 (from   │
  │     2x backbone feature) │
  │   layernorm + GELU       │
  │   up_conv2 (OHWI 2x2)    │
  │   + skip conv_s0 (from   │
  │     4x backbone feature) │
  └──────────┬───────────────┘
             │
             v
   [1, H*4, W*4, 32]   pixel_features
             │
             │
  tokens' [4, 256] (the 4 mask tokens)
       │
       v
  ┌──────────────────────────┐
  │ 4 hypernetwork MLPs      │
  │  256 -> 256 -> 256 -> 32 │
  └──────────┬───────────────┘
             │
             v
       [4, 32]   hyper_out
             │
             │  matmul hyper_out @ pixel_features^T
             v
      [4, H*4, W*4]   mask logits

  iou_token' [1, 256]
       │
       v
  ┌──────────────────────────┐
  │ iou MLP                  │
  │  256 -> 256 -> 256 -> 4  │
  └──────────┬───────────────┘
             │
             v
       [4] IoU scores per mask
```

### 4.10 Dot-product scorer (`model_misc.h`)

Computes a per-query confidence logit used to pick the best mask:

```
transformed = layernorm(fc2(relu(fc1(prompt))))
pooled      = mean(transformed, axis=0)
scores      = (hs_proj(queries) @ prompt_proj(pooled)^T)
              / sqrt(d_proj)
scores      = clamp(scores, -12, 12)
```

Output is `[n_queries, 1]`; callers apply sigmoid for probabilities.

```
  prompt [seq, 256]                queries [200, 256]
        │                                │
        v                                v
  ┌──────────┐                    ┌──────────┐
  │  fc1     │                    │          │
  │  ReLU    │                    │ hs_proj  │
  │  fc2     │                    │ [d_proj, │
  │  ln      │                    │   256]   │
  └────┬─────┘                    └────┬─────┘
       │                                │
       v                                │
  mean pool                              │
  ────────                               │
  along seq axis                         │
       │                                 │
       v                                 │
  [256] pooled                           │
       │                                 │
       v                                 │
  ┌──────────┐                          │
  │prompt_proj│                         │
  │ [d_proj,  │                         │
  │   256]    │                         │
  └────┬──────┘                         │
       │                                │
       │                                │
       └──────────┐        ┌────────────┘
                  │        │
                  v        v
              ┌────────────────┐
              │ scaled dot     │
              │ product        │
              │ (q @ p) / √d   │
              │ + clamp(-12,12)│
              └───────┬────────┘
                      │
                      v
                  [200, 1]
                  confidence logits
```

---

## 5. Pipeline orchestration (`sam3_processor.[ch]`)

`sam3_processor` is the glue that wires a `sam3_ctx` to the image
model. It owns:

* The compute backend (CPU or Metal).
* The three arenas (`model`, `persist`, `scratch`).
* The optional profiler.
* The async text encoder worker + its CPU backend.
* Cached inputs: normalized image tensor, last text prompt.

A full inference call runs, in order:

1. **Image preprocessing** (`src/util/image.c`): decode PNG/JPEG/BMP
   via `stb_image.h`, letterbox to `1008×1008`, normalize to
   `(x - 0.5) / 0.5`, store `[3, 1008, 1008]` F32 in the persist arena.
2. **Image encode** — `sam3_image_model_encode`
   a. ViT per-block (lazy RoPE + positional embed precompute on first
      call).
   b. Neck → 4 multi-scale NHWC feature maps, cached in the model.
3. **(Optional async) Text encode** — kicked off by `sam3_set_text()`
   on a worker thread; result is joined at the start of the next
   `sam3_segment()` call.
4. **Segmentation** — `sam3_image_model_segment`, per stage, with
   `scratch` reset in between:
   a. Geometry encoder (if any point/box prompt).
   b. Encoder fusion (6 layers).
   c. DETR decoder (6 layers, iterative box refinement).
   d. Either the segmentation head (text/box) or the mask decoder
      (point) — picked by prompt type.
   e. Dot-product scorer for per-query confidence.
5. **Postprocessing** (`src/util/mask_*.c`):
   * Resize masks back to the original image resolution.
   * Compute xyxy boxes from the mask logits where requested.
   * Optionally run NMS / stability-based best-mask selection.

The returned `sam3_result` owns its `masks`, `iou_scores`, and
`boxes` buffers; the caller must call `sam3_result_free()`.

**Arena-reset schedule across a full inference** (the key memory-
bounding trick: each stage's working set is fresh; the only things that
survive are cached features in `persist`):

```
  phase          scratch offset               persist offset
  ─────          ──────────────               ──────────────
                                              ↑
  set_image:                                  │ image[]
    ViT blk 0    █████░░░░░░░░░░  reset       │ ...
    ViT blk 1    ████░░░░░░░░░░░  reset       │
    ...                                       │
    ViT blk 31   ██████░░░░░░░░░  reset       │ vit_out[]
    neck stage 0 ███████░░░░░░░░  reset       │
    neck stage 1 ████████░░░░░░░  reset       │
    neck stage 2 █████████░░░░░░  reset       │ feat_s0
    neck stage 3 ██████░░░░░░░░░  reset       │ feat_s1
                                              │ feat_4x
                                              │
  segment(cat):                               │
    geom encoder ███░░░░░░░░░░░░  reset       │
    encoder fuse ██████████░░░░░  reset       │ enc_out
    decoder      ████████████░░░  reset       │ queries
    seg_head     ██████████████░  reset       │
    dot_scorer   ██░░░░░░░░░░░░░  reset       │ scores
    postprocess  ████░░░░░░░░░░░  reset       │
                                              ↓
                                              persist arena reset
                                              (ready for next image)
```

---

## 6. Tensor layout — NHWC everywhere

As of commit `96641d3` the engine is NHWC end-to-end:

* **Activations**: every 4D intermediate is `[N, H, W, C]`. Attention
  uses `[N, seq, C]`.
* **Conv weights**: **OHWI** = `[C_out, KH, KW, C_in]`, stored that way
  on disk (weight format v3) so no runtime transpose is needed.
* **Linear weights**: `[out_features, in_features]`, standard
  row-major matmul.
* **LayerNorm / bias / groupnorm**: 1-D, size = normalized dim.

The old NCHW dispatch paths have been removed from both the CPU and
Metal backends (commit `1d18d02`). The performance win is twofold:
zero-copy conv execution, and Metal kernels that read activations in
their natural layout.

---

## 7. Performance techniques in use

| Technique                               | Where it matters              |
|-----------------------------------------|-------------------------------|
| Per-block ViT evaluation                | Peak memory 55 GB → ~2.5 GB   |
| Batched graph evaluation (4 blocks/eval)| ViT + text encoder on Metal   |
| Async text encoder on a CPU worker      | Overlaps text + image encode  |
| Arena allocators (no `malloc` hot path) | Every stage                   |
| `mmap` + FNV-1a hash for weights        | `sam3_load_model()` ~60 ms    |
| NHWC / OHWI native convolutions         | Neck, seg head, mask decoder  |
| Multi-head SDPA as a single op          | All attention layers          |
| BPE word cache (1024-way direct map)    | Text tokenization             |
| Optional Q8_0 weight quantization       | Memory-bound kernels          |
| F16 / BF16 dtype-specialized kernels    | Matmul, conv, layernorm, RoPE |

Profiling is built in (`src/util/profile.c`): enable via
`sam3_profile_enable(ctx)`, call `sam3_profile_report(ctx)` to print a
per-stage breakdown to stderr. Metal builds also wire into the Metal
system profiler for GPU-side timings.

---

## 8. Tools

All user-facing functionality is provided by a single unified binary,
`sam3_cli`, with three subcommands:

| Subcommand          | Source                     | Purpose                                     |
|---------------------|----------------------------|---------------------------------------------|
| `sam3_cli segment`  | `tools/cli_segment.c`      | Inference (point/box/text prompts)          |
| `sam3_cli convert`  | `tools/cli_convert.c`      | SafeTensors → `.sam3` converter             |
| `sam3_cli info`     | `tools/cli_info.c`         | Print model file metadata                   |

Other tool sources:

| Binary              | Source                     | Purpose                                     |
|---------------------|----------------------------|---------------------------------------------|
| `gen_nhwc_fixtures` | `tools/gen_nhwc_fixtures.c`| NHWC migration reference tensor generator   |
| `bench_tokenizer`   | `tests/bench_tokenizer.c`  | Tokenizer throughput benchmark              |

`sam3_cli convert` typical invocation:

```
sam3_cli convert -i model.safetensors -o model.sam3 \
                 --image-size 1008 \
                 --encoder-dim 1024 --decoder-dim 256 \
                 --encoder-layers 32 --decoder-layers 6 \
                 [--quantize q8_0]
```

`sam3_cli segment` runs end-to-end inference against a loaded `.sam3`
file. It validates all paths before model loading, supports stdin/stdout
piping (`-i -` / `-o -`), JSON metadata (`--json`), and produces
per-mask outputs (binary, PNG, overlay, cutout) plus a merged
`overlay_all.png` with distinct colors per mask when `--overlay` is set.

```
sam3_cli segment -m model.sam3 -i photo.jpg -p 500,375,1 --overlay
sam3_cli segment -m model.sam3 -i photo.jpg -t "person" --all -o /tmp/out
```

---

## 9. Testing

`tests/` contains ~49 C test files driven by CTest:

* **Unit tests** — `test_tensor.c`, `test_half.c`, `test_quant.c`,
  `test_cast.c`, `test_weight.c`, `test_tokenizer.c`, `test_image.c`, …
* **Kernel / dispatch tests** — `test_dispatch.c`, `test_cpu_backend.c`,
  `test_metal.c`, `test_metal_cpu_concurrent.c`.
* **Layer tests** — `test_vit.c`, `test_text_encoder.c`,
  `test_mask_decoder_nhwc.c`, `test_neck_nhwc.c`,
  `test_seg_head_nhwc.c`, …
* **End-to-end smoke** — `test_model_smoke.c`.
* **Benchmarks** — `bench_dtype.c`, `bench_metal.c`,
  `bench_tokenizer.c` (not part of CI pass/fail).
* **Fixtures** — reference tensors under `tests/fixtures/` captured
  from the Python reference model and checked via
  `test_fixture_compare.c`.

Run the full suite:

```
cd build && ctest --output-on-failure
```

---

## 10. What is *not* implemented

To keep the documentation honest, these items from the upstream SAM3
are **absent** from the C engine today:

* **Video / memory tracking** — `memory_attn.[ch]` exists as a stub,
  but the video head and per-frame memory updates are not wired up.
  Only image inference runs end-to-end.
* **Causal masking in the text encoder** — `sam3_text_encoder_build`
  notes this explicitly; short prompts still give usable outputs.
* **Training-only modules** — e.g. the matching loss and Hungarian
  assigner never ship (the engine is inference-only by design).
* **CUDA / Vulkan backends** — the vtable supports them but only the
  CPU and Metal implementations are present.

Everything else listed in section 4 is implemented, loaded from real
SAM3 weights, and exercised by the test suite.
