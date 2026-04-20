/*
 * src/model/tracker_multiplex_internal.h - Private helpers for the SAM 3.1
 *                                          multiplex memory builder.
 *
 * Exposes the three leaf utilities used by multiplex_build_memory_from_bank so
 * they can be exercised by tests/test_multiplex_helpers.c without having
 * to stand up a full bank + arena. This header is strictly private to
 * src/model/tracker_multiplex.c and its test; do not include from
 * elsewhere.
 *
 * Key types:  (none; free functions only)
 * Depends on: (no internal headers)
 * Used by:    src/model/tracker_multiplex.c,
 *             tests/test_multiplex_helpers.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_TRACKER_MULTIPLEX_INTERNAL_H
#define SAM3_MODEL_TRACKER_MULTIPLEX_INTERNAL_H

/*
 * multiplex_apply_linear_256 - Host-side y = x @ W^T + b over n_rows of 256-D
 *                       vectors. W is [256, 256] in torch layout; b is
 *                       [256] and may be NULL.
 */
void multiplex_apply_linear_256(float *dst, const float *src, int n_rows,
			 const float *W, const float *b);

/*
 * multiplex_sine_tpos_256 - 1D sine/cos temporal positional encoding at
 *                    `norm_pos`. Writes 256 floats into @row:
 *                      row[0..127]   = sin(norm_pos / temperature^...)
 *                      row[128..255] = cos(norm_pos / temperature^...)
 *                    with temperature = 10000, matching Python's
 *                    get_1d_sine_pe.
 */
void multiplex_sine_tpos_256(float *row, float norm_pos);

/*
 * multiplex_maskmem_tpos_slot - Python's use_maskmem_tpos_v2 slot rule:
 *                          t_rel in (0, num_maskmem)  -> num_maskmem - t_rel - 1
 *                          otherwise                  -> num_maskmem - 1
 *                        num_maskmem is hard-coded to SAM3_MULTIPLEX_NUM_MASKMEM.
 */
int multiplex_maskmem_tpos_slot(int t_rel);

#endif /* SAM3_MODEL_TRACKER_MULTIPLEX_INTERNAL_H */
