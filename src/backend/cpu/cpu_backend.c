/*
 * src/backend/cpu/cpu_backend.c - CPU backend implementation
 *
 * Stub implementation of the CPU compute backend. Each operation
 * will be implemented as needed with scalar code first, then
 * optimized with SIMD (NEON on ARM, AVX2 on x86) later.
 *
 * Key types:  sam3_cpu_backend
 * Depends on: cpu_backend.h
 * Used by:    backend.h (registered at init)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "cpu_backend.h"
#include "util/log.h"

static enum sam3_error cpu_init(struct sam3_backend *be)
{
	(void)be;
	sam3_log_info("CPU backend initialized");
	return SAM3_OK;
}

static void cpu_free(struct sam3_backend *be)
{
	(void)be;
}

static enum sam3_error cpu_alloc_tensor(struct sam3_backend *be,
					struct sam3_tensor *t)
{
	(void)be;
	(void)t;
	/* TODO: allocate from CPU arena */
	return SAM3_OK;
}

static enum sam3_error cpu_graph_eval(struct sam3_backend *be,
				     struct sam3_graph *g)
{
	(void)be;
	(void)g;
	/* TODO: iterate nodes, dispatch to CPU kernels */
	return SAM3_OK;
}

static const struct sam3_backend_ops cpu_ops = {
	.init         = cpu_init,
	.free         = cpu_free,
	.alloc_tensor = cpu_alloc_tensor,
	.graph_eval   = cpu_graph_eval,
};

const struct sam3_backend_ops *sam3_cpu_backend_ops(void)
{
	return &cpu_ops;
}
