/*
 * src/backend/metal/metal_backend.c - Metal backend implementation
 *
 * Stub implementation of the Metal compute backend. Metal API calls
 * require Objective-C, so the actual Metal code will live in .m files.
 * This C file provides the vtable entry points and delegates to the
 * Objective-C implementation.
 *
 * Key types:  sam3_metal_backend
 * Depends on: metal_backend.h
 * Used by:    backend.h (registered at init)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "metal_backend.h"
#include "util/log.h"

#ifdef SAM3_HAS_METAL

static enum sam3_error metal_init(struct sam3_backend *be)
{
	(void)be;
	sam3_log_info("Metal backend initialized (stub)");
	/* TODO: create MTLDevice, compile shader library */
	return SAM3_OK;
}

static void metal_free(struct sam3_backend *be)
{
	(void)be;
	/* TODO: release Metal objects */
}

static enum sam3_error metal_alloc_tensor(struct sam3_backend *be,
					  struct sam3_tensor *t)
{
	(void)be;
	(void)t;
	/* TODO: create MTLBuffer for tensor data */
	return SAM3_OK;
}

static enum sam3_error metal_graph_eval(struct sam3_backend *be,
					struct sam3_graph *g)
{
	(void)be;
	(void)g;
	/* TODO: encode compute commands, commit, wait */
	return SAM3_OK;
}

static const struct sam3_backend_ops metal_ops = {
	.init         = metal_init,
	.free         = metal_free,
	.alloc_tensor = metal_alloc_tensor,
	.graph_eval   = metal_graph_eval,
};

const struct sam3_backend_ops *sam3_metal_backend_ops(void)
{
	return &metal_ops;
}

#else /* !SAM3_HAS_METAL */

const struct sam3_backend_ops *sam3_metal_backend_ops(void)
{
	return NULL;
}

#endif /* SAM3_HAS_METAL */
