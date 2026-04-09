/*
 * tools/weight_rename.h - Weight name remapping reader wrapper
 *
 * Declares the rename reader that sits between the SafeTensors reader
 * and the quant reader in the conversion pipeline. It remaps Python
 * nn.Module key names to the C model's expected weight names, and
 * splits fused QKV tensors into separate Q/K/V tensors on the fly.
 *
 * Key types:  weight_reader (from core/weight.h)
 * Depends on: core/weight.h
 * Used by:    sam3_convert.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_WEIGHT_RENAME_H
#define SAM3_WEIGHT_RENAME_H

#include "core/weight.h"

/*
 * weight_reader_rename_init - Wrap an inner reader with name remapping.
 *
 * @r:     Reader to initialize (caller-allocated)
 * @inner: Opened inner reader (e.g. SafeTensors reader)
 *
 * Builds a rename table that maps inner tensor names to C model names
 * and splits fused QKV tensors into separate Q/K/V entries. The rename
 * reader presents the remapped tensor list to downstream consumers.
 *
 * Returns SAM3_OK on success, SAM3_ENOMEM if table allocation fails.
 */
enum sam3_error weight_reader_rename_init(struct weight_reader *r,
					  struct weight_reader *inner);

#endif /* SAM3_WEIGHT_RENAME_H */
