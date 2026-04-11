/*
 * tools/weight_conv_perm.h - Conv weight OHWI permute reader wrapper
 *
 * Declares the conv_perm reader that sits between the rename reader
 * and the quant reader in the conversion pipeline. It detects conv2d
 * and conv_transpose2d weight tensors by name allowlist and transposes
 * them from OIHW / IOHW to OHWI before they reach the writer, so the
 * resulting .sam3 file ships with conv weights already laid out for
 * the NHWC backend kernels.
 *
 * Key types:  weight_reader (from core/weight.h)
 * Depends on: core/weight.h
 * Used by:    sam3_convert.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_TOOLS_WEIGHT_CONV_PERM_H
#define SAM3_TOOLS_WEIGHT_CONV_PERM_H

#include "core/weight.h"

/*
 * weight_reader_conv_perm_init - Wrap an inner reader with conv OHWI
 *                                permutation.
 *
 * @r:     Reader to initialize (caller-allocated)
 * @inner: Opened inner reader (typically the rename reader)
 *
 * The resulting reader passes every tensor through unchanged, except
 * for 4-D conv2d / conv_transpose2d weight tensors identified by the
 * internal name allowlist, whose dims are rewritten to OHWI
 * [OC, KH, KW, IC] and whose data is permuted element-wise on read.
 *
 * Returns SAM3_OK on success. The reader holds no dynamic state; the
 * caller does not need to free anything via close() beyond what the
 * inner reader already owns.
 */
enum sam3_error weight_reader_conv_perm_init(struct weight_reader *r,
					      struct weight_reader *inner);

#endif /* SAM3_TOOLS_WEIGHT_CONV_PERM_H */
