/*
 * src/backend/cpu/cpu_backend.h - CPU compute backend
 *
 * CPU fallback backend using scalar and SIMD operations. Implements
 * all operations in the sam3_backend_ops vtable. Primarily used for
 * testing and as a reference implementation for validating GPU backends.
 *
 * Key types:  sam3_cpu_backend
 * Depends on: backend/backend.h
 * Used by:    backend.h (via sam3_backend_init)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_BACKEND_CPU_H
#define SAM3_BACKEND_CPU_H

#include "backend/backend.h"

struct sam3_cpu_backend {
	struct sam3_backend base;  /* Must be first member */
};

/* Get the CPU backend ops vtable. */
const struct sam3_backend_ops *sam3_cpu_backend_ops(void);

#endif /* SAM3_BACKEND_CPU_H */
