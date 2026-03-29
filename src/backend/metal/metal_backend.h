/*
 * src/backend/metal/metal_backend.h - Metal compute backend
 *
 * GPU compute backend using Apple's Metal API. Compiles compute
 * shaders at init time, dispatches graph operations as Metal
 * command buffers. This is the primary backend for sam3 on macOS
 * and iOS.
 *
 * Key types:  sam3_metal_backend
 * Depends on: backend/backend.h
 * Used by:    backend.h (via sam3_backend_init)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_BACKEND_METAL_H
#define SAM3_BACKEND_METAL_H

#include "backend/backend.h"

struct sam3_metal_backend {
	struct sam3_backend base;  /* Must be first member */
	void *device;              /* id<MTLDevice> — stored as void* for C compat */
	void *command_queue;       /* id<MTLCommandQueue> */
	void *library;             /* id<MTLLibrary> — compiled shaders */
};

/* Get the Metal backend ops vtable. */
const struct sam3_backend_ops *sam3_metal_backend_ops(void);

#endif /* SAM3_BACKEND_METAL_H */
