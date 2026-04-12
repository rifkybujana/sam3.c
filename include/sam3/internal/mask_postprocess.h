/*
 * include/sam3/internal/mask_postprocess.h - Mask post-processing utilities.
 *
 * Provides sigmoid, morphological cleanup, and small-component removal
 * for mask logit arrays. All functions operate on flat float arrays
 * and require caller-provided output buffers.
 *
 * Key types:  none
 * Depends on: (none)
 * Used by:    tools/sam3_main.c, tests/test_mask_postprocess.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */
#ifndef SAM3_INTERNAL_MASK_POSTPROCESS_H
#define SAM3_INTERNAL_MASK_POSTPROCESS_H

/*
 * sam3_mask_sigmoid - Apply sigmoid to mask logits for probability output.
 *
 * @src:  Input logits (n floats)
 * @dst:  Output probabilities (n floats, caller-allocated, may alias src)
 * @n:    Number of elements
 *
 * Returns 0 on success, -1 on NULL args.
 */
int sam3_mask_sigmoid(const float *src, float *dst, int n);

/*
 * sam3_mask_morpho_open - Morphological open (erode then dilate) with 3x3 box.
 *
 * Operates on a binarized mask (1/0). Removes small noise blobs.
 *
 * @mask:   Input binary mask (w * h, values 0 or 1)
 * @out:    Output mask (w * h, caller-allocated)
 * @w:      Width
 * @h:      Height
 * @work:   Scratch buffer (w * h bytes, caller-allocated)
 *
 * Returns 0 on success, -1 on invalid args.
 */
int sam3_mask_morpho_open(const unsigned char *mask, unsigned char *out,
			  int w, int h, unsigned char *work);

/*
 * sam3_mask_remove_small - Remove connected components smaller than min_pixels.
 *
 * @mask:       Input/output binary mask (w * h, values 0 or 1), modified in-place
 * @w:          Width
 * @h:          Height
 * @min_pixels: Minimum component size to keep
 * @labels:     Scratch buffer (w * h ints, caller-allocated)
 * @stack:      Scratch buffer for flood fill (w * h ints, caller-allocated)
 *
 * Returns 0 on success, -1 on invalid args.
 */
int sam3_mask_remove_small(unsigned char *mask, int w, int h,
			   int min_pixels, int *labels, int *stack);

#endif /* SAM3_INTERNAL_MASK_POSTPROCESS_H */
