/*
 * src/util/error.h - Error code utilities
 *
 * Provides human-readable error messages for sam3_error codes.
 * All sam3 error codes are defined in sam3_types.h; this module
 * only provides string conversion.
 *
 * Key types:  (uses sam3_error from sam3_types.h)
 * Depends on: sam3/sam3_types.h
 * Used by:    tools/, user applications
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_UTIL_ERROR_H
#define SAM3_UTIL_ERROR_H

#include "sam3/sam3_types.h"

/* Return a human-readable string for the given error code. */
const char *sam3_error_str(enum sam3_error err);

#endif /* SAM3_UTIL_ERROR_H */
