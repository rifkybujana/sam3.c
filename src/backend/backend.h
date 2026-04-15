/*
 * src/backend/backend.h - Backend abstraction layer
 *
 * Defines the vtable interface that all compute backends must implement.
 * Model code calls backend operations through this interface, never
 * directly. This allows runtime backend selection and makes it trivial
 * to add new backends (CUDA, Vulkan) without changing model code.
 *
 * Key types:  sam3_backend, sam3_backend_ops
 * Depends on: core/graph.h
 * Used by:    model/ files, tools/
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_BACKEND_H
#define SAM3_BACKEND_H

#include "core/graph.h"

/* Backend type identifiers. */
enum sam3_backend_type {
	SAM3_BACKEND_CPU,
	SAM3_BACKEND_METAL,
};

struct sam3_backend;

/* Operations vtable that every backend must implement. */
struct sam3_backend_ops {
	/* Initialize backend resources. */
	enum sam3_error (*init)(struct sam3_backend *be);

	/* Free backend resources. */
	void (*free)(struct sam3_backend *be);

	/* Allocate a tensor buffer on this backend's memory. */
	enum sam3_error (*alloc_tensor)(struct sam3_backend *be,
				       struct sam3_tensor *t);

	/* Evaluate a compute graph on this backend. */
	enum sam3_error (*graph_eval)(struct sam3_backend *be,
				     struct sam3_graph *g);

	/* Reset working memory (arena) between graph evaluations.
	 * May be NULL if the backend manages memory automatically. */
	void (*arena_reset)(struct sam3_backend *be);

	/* Invalidate cached state for tensors whose address falls in
	 * [start, start + len). Called when arena memory is recycled
	 * so that backends with tensor caches do not return stale data.
	 * May be NULL if the backend has no persistent tensor cache. */
	void (*cache_invalidate)(struct sam3_backend *be,
				 const void *start, size_t len);
};

/* Backend instance. Backends embed this as first member. */
struct sam3_backend {
	enum sam3_backend_type    type;
	const struct sam3_backend_ops *ops;
};

/*
 * sam3_backend_init - Create and initialize a backend.
 *
 * @type: Which backend to create.
 *
 * Returns a heap-allocated backend or NULL on failure.
 * Caller must call sam3_backend_free() when done.
 */
struct sam3_backend *sam3_backend_init(enum sam3_backend_type type);

/* Free a backend created by sam3_backend_init. */
void sam3_backend_free(struct sam3_backend *be);

#endif /* SAM3_BACKEND_H */
