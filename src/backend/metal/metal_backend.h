/*
 * src/backend/metal/metal_backend.h - Metal compute backend (MLX-C)
 *
 * GPU compute backend using Apple's MLX framework via MLX-C bindings.
 * Translates SAM3 compute graphs into MLX lazy operations and evaluates
 * them in a single GPU dispatch. This is the primary backend for sam3
 * on macOS with Apple Silicon.
 *
 * Key types:  sam3_metal_backend
 * Depends on: backend/backend.h, core/alloc.h
 * Used by:    backend.h (via sam3_backend_init)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_BACKEND_METAL_H
#define SAM3_BACKEND_METAL_H

#include "backend/backend.h"
#include "core/alloc.h"
#include <stdbool.h>

#ifdef SAM3_HAS_METAL
#include "mlx/c/mlx.h"
#endif

/* Default arena capacity: 256 MiB. */
#define SAM3_METAL_ARENA_DEFAULT_CAPACITY (256UL * 1024 * 1024)

/* Default scratch arena: 64 MiB for Q8 dequantization buffers. */
#define SAM3_METAL_SCRATCH_DEFAULT_CAPACITY (64UL * 1024 * 1024)

#define SAM3_METAL_MAP_INIT_CAP 8192  /* Initial capacity, must be power of 2 */

struct sam3_metal_backend {
	struct sam3_backend  base;           /* Must be first member */
	struct sam3_arena    arena;          /* Host-side tensor data */
	struct sam3_arena    scratch;        /* Q8 dequant scratch space */
	size_t               arena_capacity; /* 0 = use default */
#ifdef SAM3_HAS_METAL
	mlx_stream           stream;         /* GPU compute stream */
	mlx_device           device;         /* MLX device handle */
	/* Persistent tensor-to-mlx_array cache (dynamic, rehashes at 75%) */
	const struct sam3_tensor **map_keys;
	mlx_array                *map_vals;
	int                       map_count;
	int                       map_capacity;
	bool                      use_f16;   /* Cast F32 -> F16 for compute */
	mlx_array                 relu_zeros[13]; /* Per-dtype cached scalar 0 */
	/* Per-dtype cached GELU constants: 0.5, 1/sqrt(2), 1.0 */
	mlx_array                 gelu_half[13];
	mlx_array                 gelu_rsqrt2[13];
	mlx_array                 gelu_one[13];
#endif
};

/* Get the Metal backend ops vtable. Returns NULL if Metal unavailable. */
const struct sam3_backend_ops *sam3_metal_backend_ops(void);

#endif /* SAM3_BACKEND_METAL_H */
