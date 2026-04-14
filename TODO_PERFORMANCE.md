# Performance TODO

Optimization opportunities that do not affect numerical results.

## CPU backend

### 1. SDPA: vectorize inner loops
**File:** `src/backend/cpu/kernels/cpu_sdpa.c:65-97`
**Impact:** 4-8x speedup on attention (hottest kernel)

The tiled SDPA kernel uses scalar loops for dot product, correction, and V
accumulation over `head_dim` (64 elements). With 5184 patches, each query row
does 5184 dot products — all scalar.

```c
/* dot product — scalar */
for (d = 0; d < head_dim; d++)
    score += q[d] * kj[d];

/* correction — scalar */
for (d = 0; d < head_dim; d++)
    out[d] *= correction;

/* V accumulation — scalar */
for (d = 0; d < head_dim; d++)
    out[d] += w * vj[d];
```

Add NEON (vfmaq_f32) and AVX2 (_mm256_fmadd_ps) paths for all three loops.

### 2. Layernorm: hoist gamma/beta branches out of SIMD loop
**File:** `src/backend/cpu/kernels/cpu_layernorm.c:100-103` (NEON), `:154-157` (AVX2)
**Impact:** 10-20% on layernorm

`if (gamma)` and `if (beta)` are loop-invariant but checked every 4/8-element
iteration inside the tight normalize+scale+shift loop. Hoist into four
specialized variants: {gamma+beta, gamma-only, beta-only, neither}.

### 3. Matmul: avoid redundant a[i*K+k] reload in remainder
**File:** `src/backend/cpu/kernels/cpu_matmul.c:86+94` (NEON), `:125+133` (AVX2)
**Impact:** ~1% (free fix)

The NEON/AVX2 paths load `a[i * K + k]` into a broadcast vector, then reload
the same value as scalar `aik` for the remainder loop. Extract once before the
SIMD loop and reuse.

### 4. Threadpool: use atomics instead of mutex for task counter
**File:** `src/util/threadpool.c:112-114`
**Impact:** reduced contention at high thread counts

Workers lock/unlock the mutex to claim each task. Replace with
`__atomic_fetch_add(&pool->task_counter, 1, __ATOMIC_RELAXED)` in the
work-stealing loop. The mutex is only needed for generation/done signaling.

### 5. Transpose: specialize generic path by element size
**File:** `src/backend/cpu/kernels/cpu_transpose.c:54-58`
**Impact:** 2-3x for F16/BF16 transpose

`transpose_generic` calls `memcpy()` per element for any non-f32 dtype. Add
direct-assignment specializations for 2-byte (uint16_t cast) and 8-byte
(uint64_t cast) element sizes to avoid function call overhead.

## Metal backend (MLX-C)

### 6. GELU: replace 12 temporary arrays with fused op
**File:** `src/backend/metal/metal_backend.c:338-393`
**Impact:** reduce host-side allocation overhead per GELU node

The GELU dispatch creates 12 intermediate `mlx_array` objects to compose
`0.5 * x * (1 + erf(x / sqrt(2)))`. If MLX-C exposes `mlx_nn_gelu` or
equivalent, one call replaces all 12 allocations and frees.

### ~~7. ReLU: cache the scalar zero tensor~~ ✅ DONE

### ~~8. graph_eval phase 3: batch mlx_eval calls~~ ✅ DONE

### ~~9. F16-to-F32 cast: batch with contiguous eval~~ ✅ DONE

## Both backends

### ~~10. Hash maps: add load factor control~~ ✅ DONE
