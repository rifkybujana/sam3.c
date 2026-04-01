/*
 * src/util/error.c - Error string conversion
 *
 * Maps sam3_error codes to descriptive strings for logging and
 * user-facing error messages.
 *
 * Key types:  (uses sam3_error from sam3_types.h)
 * Depends on: error.h
 * Used by:    tools/, user applications
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "error.h"

const char *sam3_error_str(enum sam3_error err)
{
	switch (err) {
	case SAM3_OK:       return "success";
	case SAM3_EINVAL:   return "invalid argument";
	case SAM3_ENOMEM:   return "out of memory";
	case SAM3_EIO:      return "I/O error";
	case SAM3_EBACKEND: return "backend initialization failed";
	case SAM3_EMODEL:   return "model format error";
	case SAM3_EDTYPE:   return "unsupported or mismatched dtype";
	}
	return "unknown error";
}
