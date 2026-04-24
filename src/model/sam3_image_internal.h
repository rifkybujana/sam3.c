/*
 * src/model/sam3_image_internal.h - Private helpers used by sam3_image
 * and its unit tests.
 *
 * Exposes CPU helpers (box refinement, batched variant) that must be
 * callable from tests/test_batched_ops.c without being part of the
 * public sam3 API. Do not include from outside the model/ directory.
 *
 * Key types:  sam3_decoder (used by helpers)
 * Depends on: model/decoder.h
 * Used by:    src/model/sam3_image.c, tests/test_batched_ops.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_SAM3_IMAGE_INTERNAL_H
#define SAM3_MODEL_SAM3_IMAGE_INTERNAL_H

#include "model/decoder.h"

/*
 * cpu_box_refine - Apply LayerNorm + 3-layer MLP box head + sigmoid
 * update on CPU for a single batch slot.
 *
 * @q:         Query embeddings [nq, d] (raw, before output_ln)
 * @dec:       Decoder (for output_ln + box_head weights on layer 0)
 * @ref_boxes: [nq, 4] — updated in place
 * @nq, @d:    Query count, model dim
 * @tmp1:      Scratch [nq * max(d, 4)] floats
 * @tmp2:      Scratch [nq * d] floats
 */
void cpu_box_refine(const float *q,
		     const struct sam3_decoder *dec,
		     float *ref_boxes, int nq, int d,
		     float *tmp1, float *tmp2);

/*
 * cpu_box_refine_batched - Batched wrapper: iterates cpu_box_refine
 * over B batch slots. Pointers advance by nq*d (q) and nq*4 (ref_boxes)
 * per slot. Temp buffers are reused across slots.
 *
 * @B: Batch size (>= 1)
 * Other params: same as cpu_box_refine, but @q is [B, nq, d] flat
 * and @ref_boxes is [B, nq, 4] flat.
 */
void cpu_box_refine_batched(const float *q,
			     const struct sam3_decoder *dec,
			     float *ref_boxes, int B, int nq, int d,
			     float *tmp1, float *tmp2);

#endif /* SAM3_MODEL_SAM3_IMAGE_INTERNAL_H */
