/*
 * src/backend/cpu/cpu_backend.c - CPU backend implementation
 *
 * Implements the CPU compute backend with arena-based tensor allocation.
 * The arena is initialized on cpu_init() and all tensor data is bump-
 * allocated from it, giving O(1) allocation with zero fragmentation.
 * The arena is freed on cpu_free(). Graph evaluation delegates to
 * cpu_dispatch_node() which routes each node through the 2D dispatch table.
 *
 * Key types:  sam3_cpu_backend
 * Depends on: cpu_backend.h, cpu_dispatch.h, core/tensor.h, util/log.h,
 *             util/profile.h, util/threadpool.h
 * Used by:    backend.h (registered at init)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "cpu_backend.h"
#include "cpu_dispatch.h"
#include "core/tensor.h"
#include "util/log.h"
#include "util/profile.h"
#include "util/threadpool.h"

static enum sam3_error cpu_init(struct sam3_backend *be)
{
	struct sam3_cpu_backend *cpu = (struct sam3_cpu_backend *)be;
	size_t capacity = cpu->arena_capacity;
	enum sam3_error err;

	if (capacity == 0)
		capacity = SAM3_CPU_ARENA_DEFAULT_CAPACITY;

	err = sam3_arena_init(&cpu->arena, capacity);
	if (err != SAM3_OK) {
		sam3_log_error("CPU backend: arena init failed (%zu bytes)",
			       capacity);
		return err;
	}

	err = sam3_arena_init(&cpu->scratch,
			      SAM3_CPU_SCRATCH_DEFAULT_CAPACITY);
	if (err != SAM3_OK) {
		sam3_arena_free(&cpu->arena);
		sam3_log_error("CPU backend: scratch arena init failed");
		return err;
	}

	cpu->pool = sam3_threadpool_create(0);
	if (!cpu->pool) {
		sam3_arena_free(&cpu->scratch);
		sam3_arena_free(&cpu->arena);
		sam3_log_error("CPU backend: thread pool init failed");
		return SAM3_ENOMEM;
	}

	sam3_log_info("CPU backend initialized (arena: %zu bytes)", capacity);
	return SAM3_OK;
}

static void cpu_free(struct sam3_backend *be)
{
	struct sam3_cpu_backend *cpu = (struct sam3_cpu_backend *)be;

	sam3_threadpool_free(cpu->pool);
	sam3_arena_free(&cpu->scratch);
	sam3_arena_free(&cpu->arena);
	sam3_log_debug("CPU backend freed");
}

/*
 * cpu_alloc_tensor - Allocate tensor data from the CPU arena.
 *
 * @be: Backend (must be sam3_cpu_backend)
 * @t:  Tensor with dtype, n_dims, dims[] set. data must be NULL.
 *
 * On success, fills t->data, t->nbytes, and t->strides.
 * Returns SAM3_OK, SAM3_EINVAL for bad inputs, SAM3_ENOMEM if arena full.
 */
static enum sam3_error cpu_alloc_tensor(struct sam3_backend *be,
					struct sam3_tensor *t)
{
	struct sam3_cpu_backend *cpu = (struct sam3_cpu_backend *)be;
	size_t elem_size;
	size_t nbytes;

	if (!t) {
		sam3_log_error("CPU alloc_tensor: NULL tensor");
		return SAM3_EINVAL;
	}

	if (t->n_dims < 1 || t->n_dims > SAM3_MAX_DIMS) {
		sam3_log_error("CPU alloc_tensor: invalid n_dims=%d", t->n_dims);
		return SAM3_EINVAL;
	}

	elem_size = sam3_dtype_size(t->dtype);
	if (elem_size == 0) {
		sam3_log_error("CPU alloc_tensor: unknown dtype=%d", t->dtype);
		return SAM3_EINVAL;
	}

	nbytes = (size_t)sam3_tensor_nelems(t) * elem_size;
	t->data = sam3_arena_alloc(&cpu->arena, nbytes);
	if (!t->data) {
		sam3_log_error("CPU alloc_tensor: OOM (%zu bytes requested, "
			      "%zu / %zu used)", nbytes,
			      cpu->arena.offset, cpu->arena.size);
		return SAM3_ENOMEM;
	}

	t->nbytes = nbytes;
	sam3_tensor_compute_strides(t);

	sam3_log_debug("CPU alloc_tensor: %zu bytes @ %p", nbytes, t->data);
	return SAM3_OK;
}

static enum sam3_error cpu_graph_eval(struct sam3_backend *be,
				     struct sam3_graph *g)
{
	struct sam3_cpu_backend *cpu = (struct sam3_cpu_backend *)be;
	enum sam3_error err;

#ifdef SAM3_HAS_PROFILE
	struct sam3_profiler *prof = cpu->profiler;
#endif

	for (int i = 0; i < g->n_nodes; i++) {
		struct sam3_node *node = &g->nodes[i];

		SAM3_PROF_OP_BEGIN(prof, node->op);

		err = cpu_dispatch_node(node, &cpu->scratch, cpu->pool);

		SAM3_PROF_OP_END(prof, node->op);

		if (err != SAM3_OK) {
			sam3_log_error("cpu_graph_eval: node %d (op=%d) failed",
				       i, node->op);
			return err;
		}
	}

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
