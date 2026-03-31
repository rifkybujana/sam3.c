# FP16/BF16 Compute Design

## Summary

Add fp16 and bf16 compute paths to the SAM3 CPU backend with native NEON fp16
arithmetic, bf16-to-f32 upcasting, a 2D dispatch table, compile-time-gated
tracing with numeric diagnostics, and performance benchmarks.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| fp16 compute | Native NEON fp16 arithmetic (ARMv8.2-A) | Apple Silicon has full fp16 FMA; maximize throughput |
| bf16 compute | Always upcast to f32 | No native bf16 arithmetic on ARM NEON |
| fp16 accumulation | Full fp16 everywhere (including reductions) | Trust Apple Silicon fp16 FMA accuracy |
| Dispatch model | Separate functions + 2D dispatch table | Clean separation, easy to extend |
| Architecture | Per-dtype kernel files | Independent optimization per dtype |
| Mixed dtype policy | Reject; require explicit SAM3_OP_CAST | Keep dtype transitions visible in graph |
| Tracing | Compile-time gated (SAM3_HAS_TRACE) | Zero overhead in release builds |

## Component 1: half.h — Conversion Library

**Location**: `src/core/half.h` (header-only, static inline)

### Scalar conversions (portable C11)

```c
float    fp16_to_f32(uint16_t h);    /* IEEE 754 half -> float */
uint16_t f32_to_fp16(float f);       /* float -> IEEE 754 half */
float    bf16_to_f32(uint16_t b);    /* bfloat16 -> float */
uint16_t f32_to_bf16(float f);       /* float -> bfloat16 (round-to-nearest-even) */
```

### Validation helpers

```c
int fp16_is_nan(uint16_t h);
int fp16_is_inf(uint16_t h);
int bf16_is_nan(uint16_t b);
int bf16_is_inf(uint16_t b);
```

### SIMD conversions — NEON (guarded by SAM3_HAS_NEON)

```c
/* 4-wide */
float32x4_t  fp16x4_to_f32x4(const _Float16 *src);
void         f32x4_to_fp16x4(_Float16 *dst, float32x4_t v);
float32x4_t  bf16x4_to_f32x4(const uint16_t *src);
void         f32x4_to_bf16x4(uint16_t *dst, float32x4_t v);

/* 8-wide */
float32x4x2_t fp16x8_to_f32x4x2(const _Float16 *src);
void          f32x4x2_to_fp16x8(_Float16 *dst, float32x4x2_t v);
```

### SIMD conversions — AVX2 (guarded by SAM3_HAS_AVX2, requires F16C)

```c
__m256  fp16x8_to_f32x8(const uint16_t *src);   /* _mm256_cvtph_ps */
void    f32x8_to_fp16x8(uint16_t *dst, __m256 v); /* _mm256_cvtps_ph */
```

### Implementation notes

- bf16->f32: `(uint32_t)v << 16` (trivial bit shift)
- f32->bf16: round-to-nearest-even to match PyTorch behavior
- fp16 scalar: full IEEE 754 half spec (denormals, inf, nan)
- Header-only for inlining into kernel hot loops

## Component 2: Trace System

**Location**: `src/core/trace.h` and `src/core/trace.c`

**Compile-time gate**: `SAM3_HAS_TRACE` — expands to no-ops when undefined.

### 2a. Kernel dtype tracing

```c
void sam3_trace_kernel(const char *kernel_name,
                       enum sam3_dtype in_dtype,
                       enum sam3_dtype out_dtype,
                       const char *variant);
/* Output: [TRACE] matmul: F16 -> F16 (native-neon-fp16) */
```

### 2b. Numeric diagnostics

```c
struct sam3_numeric_stats {
    float min, max, mean;
    int   nan_count;
    int   inf_count;
    int   denormal_count;
    int   total_elems;
};

void sam3_trace_numeric(const char *label, const struct sam3_tensor *t);
/* Output: [TRACE] matmul.out: min=-2.31 max=14.7 mean=0.42 nan=0 inf=0 denorm=3 n=4096 */
```

### 2c. Tensor comparison

```c
struct sam3_compare_result {
    float max_abs_error;
    float max_rel_error;
    float mean_abs_error;
    int   mismatches;       /* elements exceeding tolerance */
};

void sam3_trace_compare(const char *label,
                        const struct sam3_tensor *actual,
                        const struct sam3_tensor *reference,
                        float tolerance);
/* Output: [TRACE] matmul.cmp: max_abs=0.0023 max_rel=0.0041 mean_abs=0.0003 mismatches=0 */
```

### 2d. Graph-level trace

```c
void sam3_trace_graph_plan(const struct sam3_graph *g);
/* Output: [TRACE] graph: 47 nodes, dtypes: F16=32 F32=10 BF16=5 */
/* [TRACE]   node[0]: MATMUL F16 (1024x768) x (768x768) -> (1024x768) */
/* [TRACE]   node[1]: ADD F16 (1024x768) + (768) -> (1024x768) */
/* ... */

void sam3_trace_graph_done(const struct sam3_graph *g, double elapsed_ms);
/* Output: [TRACE] graph: completed 47 nodes in 12.3ms */
```

### 2e. Runtime trace flags

```c
enum sam3_trace_flags {
    SAM3_TRACE_KERNELS  = 1 << 0,
    SAM3_TRACE_NUMERIC  = 1 << 1,
    SAM3_TRACE_COMPARE  = 1 << 2,
    SAM3_TRACE_GRAPH    = 1 << 3,
    SAM3_TRACE_ALL      = 0xFFFF,
};

void sam3_trace_set_flags(unsigned flags);
```

## Component 3: Dispatch Table

**Location**: `src/backend/cpu/cpu_dispatch.h` and `cpu_dispatch.c`

### Kernel function signature

```c
typedef enum sam3_error (*sam3_kernel_fn)(
    struct sam3_node *node,
    struct sam3_arena *scratch);
```

### 2D dispatch table

```c
static const sam3_kernel_fn
cpu_dispatch[SAM3_OP_COUNT][SAM3_DTYPE_COUNT];
```

Indexed by `[node->op][node->inputs[0]->dtype]`. NULL entries mean
"not implemented" and return `SAM3_EDTYPE`.

### Dispatch function

```c
enum sam3_error cpu_dispatch_node(struct sam3_node *node,
                                  struct sam3_arena *scratch)
{
    enum sam3_dtype dt = node->inputs[0]->dtype;

    /* Verify all inputs share same dtype */
    for (int i = 1; i < node->n_inputs; i++) {
        if (node->inputs[i]->dtype != dt)
            return SAM3_EDTYPE;
    }

    sam3_kernel_fn fn = cpu_dispatch[node->op][dt];
    if (!fn)
        return SAM3_EDTYPE;

    /* Trace hooks (compile-time gated) */
    SAM3_TRACE_KERNEL(node);
    enum sam3_error err = fn(node, scratch);
    SAM3_TRACE_NUMERIC(node);

    return err;
}
```

## Component 4: Kernel File Organization

### File layout

```
src/backend/cpu/kernels/
    cpu_matmul_f32.c       (renamed from cpu_matmul.c)
    cpu_matmul_f16.c       NEW — native NEON fp16, scalar fallback
    cpu_matmul_bf16.c      NEW — bf16->f32 upcast, f32 compute, f32->bf16
    cpu_add_f32.c          (split from cpu_elementwise.c)
    cpu_add_f16.c          NEW
    cpu_add_bf16.c         NEW
    cpu_mul_f32.c          (split from cpu_elementwise.c)
    cpu_mul_f16.c          NEW
    cpu_mul_bf16.c         NEW
    cpu_softmax_f32.c      (renamed from cpu_softmax.c)
    cpu_softmax_f16.c      NEW
    cpu_softmax_bf16.c     NEW
    cpu_relu_f32.c         (renamed)
    cpu_relu_f16.c         NEW
    cpu_relu_bf16.c        NEW
    cpu_gelu_f32.c         (renamed)
    cpu_gelu_f16.c         NEW
    cpu_gelu_bf16.c        NEW
    cpu_layernorm_f32.c    (renamed)
    cpu_layernorm_f16.c    NEW
    cpu_layernorm_bf16.c   NEW
    cpu_conv2d_f32.c       (renamed)
    cpu_conv2d_f16.c       NEW
    cpu_conv2d_bf16.c      NEW
    cpu_reshape.c          UNCHANGED — dtype-agnostic (byte copy)
    cpu_transpose.c        UNCHANGED — dtype-agnostic (byte copy)
    cpu_cast.c             NEW — SAM3_OP_CAST implementation
    cpu_simd.h             existing
    cpu_simd_f16.h         NEW — NEON fp16 helpers
```

### Kernel implementation strategy

| Kernel category | f32 | fp16 | bf16 |
|----------------|-----|------|------|
| **matmul** | Existing | Native NEON float16x8_t, vfmaq_f16. Thread pool over M rows. | Load bf16->f32, call f32 compute core, store f32->bf16. |
| **conv2d** | Existing | Native NEON fp16 im2col + matmul. | Upcast->f32 conv2d->downcast. |
| **softmax** | Existing | Full fp16 accumulation with float16x8_t. | Upcast->f32 softmax->downcast. |
| **layernorm** | Existing | Full fp16 accumulation. | Upcast->f32 layernorm->downcast. |
| **add, mul** | Existing | Native NEON vaddq_f16, vmulq_f16. | Upcast->f32 elementwise->downcast. |
| **relu** | Existing | Native NEON vmaxq_f16 with zero. | Upcast->f32 relu->downcast. |
| **gelu** | Existing | Native NEON fp16 tanh approximation. | Upcast->f32 gelu->downcast. |
| **reshape, transpose** | Dtype-agnostic | Same impl | Same impl |
| **cast** | N/A | Conversion via half.h | Conversion via half.h |

### SIMD helpers: cpu_simd_f16.h

```c
#ifdef SAM3_HAS_NEON_FP16
float16x8_t neon_f16_zero(void);
float       neon_f16_hsum(float16x8_t v);        /* horizontal sum -> f32 */
float16x8_t neon_f16_gelu_approx(float16x8_t x); /* tanh approximation */
#endif
```

`SAM3_HAS_NEON_FP16` is a compile-time check for
`__ARM_FEATURE_FP16_VECTOR_ARITHMETIC`. When unavailable, fp16 kernels use
scalar fallback (convert each element to f32, compute, convert back).

## Component 5: Mixed Dtype Policy

### Rule: reject mixed dtypes, require explicit cast

All inputs to a kernel must share the same dtype. Mixed-dtype operations
require an explicit `SAM3_OP_CAST` node in the graph.

### New op

```c
SAM3_OP_CAST   /* added to sam3_op enum */
```

Cast node uses `half.h` conversions. `params[0]` holds the target dtype.

### New error code

```c
SAM3_EDTYPE = -6   /* Unsupported or mismatched dtype */
```

### Helper

```c
const char *sam3_dtype_str(enum sam3_dtype dt);   /* "F32", "F16", "BF16", ... */
```

## Component 6: Testing

### Test files

```
tests/test_half.c             conversion correctness (scalar + SIMD)
tests/test_dispatch.c         dispatch table coverage, dtype mismatch errors
tests/test_matmul_f16.c       fp16 matmul correctness vs f32 reference
tests/test_matmul_bf16.c      bf16 matmul correctness vs f32 reference
tests/test_elementwise_f16.c  add/mul/relu/gelu in fp16
tests/test_trace.c            trace output, numeric stats, tensor comparison
tests/bench_dtype.c           performance benchmarks (GFLOPS per dtype)
```

### Correctness testing

- **Conversion round-trip**: f32 -> fp16 -> f32 and f32 -> bf16 -> f32 within precision
- **Kernel correctness**: Compute in f32 as reference, compare fp16/bf16 result
- **Edge cases**: NaN propagation, Inf handling, denormal behavior, zero sign
- **Dispatch errors**: SAM3_EDTYPE for mismatched dtypes, unimplemented combos
- **Cast node**: Verify all dtype pair conversions

### Tolerance thresholds

| Comparison | Max relative error |
|------------|-------------------|
| fp16 vs f32 (elementwise) | 1e-3 |
| fp16 vs f32 (matmul, N<=512) | 5e-3 |
| bf16 vs f32 (elementwise) | 1e-2 |
| bf16 vs f32 (matmul, N<=512) | 2e-2 |

### Performance benchmarks (bench_dtype.c)

Measures GFLOPS for matmul and elementwise ops across f32, fp16, bf16 at
various sizes (64x64, 256x256, 1024x1024). Validates that native fp16 on
Apple Silicon outperforms the upcast path.

## Build System Changes

- Add `SAM3_TRACE` CMake option (OFF by default), defines `SAM3_HAS_TRACE`
- Detect `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` and define `SAM3_HAS_NEON_FP16`
- Add new kernel source files to `sam3_cpu_backend` target
- Add new test files to CTest
