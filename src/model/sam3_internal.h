/*
 * src/model/sam3_internal.h - Internal context definition for SAM3
 *
 * Defines the full struct sam3_ctx so that multiple implementation
 * files (sam3.c, sam3_video.c) can access context internals. This
 * header is strictly private and must not be exposed to users.
 *
 * Key types:  sam3_ctx
 * Depends on: sam3/sam3_types.h, core/weight.h, model/sam3_processor.h
 * Used by:    src/sam3.c, src/model/sam3_video.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_SAM3_INTERNAL_H
#define SAM3_MODEL_SAM3_INTERNAL_H

#include "sam3/sam3_types.h"
#include "core/weight.h"
#include "model/sam3_processor.h"

#ifdef SAM3_HAS_PROFILE
#include "util/profile.h"
#endif

/* Internal context definition. */
struct sam3_ctx {
	struct sam3_model_config config;
	struct sam3_weight_file weights;
	int loaded;
	struct sam3_processor proc;
	int proc_ready;
#ifdef SAM3_HAS_PROFILE
	struct sam3_profiler *profiler;
#endif
};

#endif /* SAM3_MODEL_SAM3_INTERNAL_H */
