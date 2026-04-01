/*
 * src/backend/backend.c - Backend factory functions
 *
 * Implements sam3_backend_init() and sam3_backend_free() which create
 * and destroy backends by type. These are the only entry points for
 * backend lifecycle management from outside the backend subsystem.
 *
 * Key types:  sam3_backend
 * Depends on: backend.h, cpu/cpu_backend.h, metal/metal_backend.h
 * Used by:    sam3.c, tools/
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "backend.h"
#include "util/log.h"

#include <stdlib.h>
#include <string.h>

#ifdef SAM3_HAS_CPU
#include "cpu/cpu_backend.h"
#endif

#ifdef SAM3_HAS_METAL
#include "metal/metal_backend.h"
#endif

struct sam3_backend *sam3_backend_init(enum sam3_backend_type type)
{
	struct sam3_backend *be = NULL;
	const struct sam3_backend_ops *ops = NULL;
	size_t be_size = 0;

	switch (type) {
#ifdef SAM3_HAS_CPU
	case SAM3_BACKEND_CPU:
		ops = sam3_cpu_backend_ops();
		be_size = sizeof(struct sam3_cpu_backend);
		break;
#endif
#ifdef SAM3_HAS_METAL
	case SAM3_BACKEND_METAL:
		ops = sam3_metal_backend_ops();
		be_size = sizeof(struct sam3_metal_backend);
		break;
#endif
	default:
		sam3_log_error("backend_init: unknown type %d", type);
		return NULL;
	}

	if (!ops) {
		sam3_log_error("backend_init: backend type %d not available",
			       type);
		return NULL;
	}

	be = calloc(1, be_size);
	if (!be) {
		sam3_log_error("backend_init: allocation failed");
		return NULL;
	}

	be->type = type;
	be->ops = ops;

	enum sam3_error err = be->ops->init(be);
	if (err != SAM3_OK) {
		sam3_log_error("backend_init: init failed (%d)", err);
		free(be);
		return NULL;
	}

	return be;
}

void sam3_backend_free(struct sam3_backend *be)
{
	if (!be)
		return;
	be->ops->free(be);
	free(be);
}
