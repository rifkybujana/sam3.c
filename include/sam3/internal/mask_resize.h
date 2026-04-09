/*
 * include/sam3/internal/mask_resize.h - Bilinear mask resize helper.
 *
 * Internal header for resizing float mask arrays using bilinear
 * interpolation. Matches PyTorch F.interpolate(mode="bilinear",
 * align_corners=False) for mask upsampling to original image size.
 *
 * Key types:  none
 * Depends on: (none)
 * Used by:    src/model/sam3_processor.c, tools/sam3_main.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */
#ifndef SAM3_INTERNAL_MASK_RESIZE_H
#define SAM3_INTERNAL_MASK_RESIZE_H

/*
 * sam3_mask_resize_bilinear - Resize a float mask using bilinear interpolation.
 *
 * Matches PyTorch F.interpolate(mode="bilinear", align_corners=False):
 *   src_x = (dst_x + 0.5) * (src_w / dst_w) - 0.5
 *   src_y = (dst_y + 0.5) * (src_h / dst_h) - 0.5
 *
 * @src:    Source mask data (src_h * src_w floats)
 * @src_w:  Source width (must be > 0)
 * @src_h:  Source height (must be > 0)
 * @dst:    Output buffer (dst_h * dst_w floats, caller-allocated)
 * @dst_w:  Target width (must be > 0)
 * @dst_h:  Target height (must be > 0)
 *
 * Returns 0 on success, -1 on invalid arguments (NULL pointers or zero dims).
 */
int sam3_mask_resize_bilinear(const float *src, int src_w, int src_h,
			      float *dst, int dst_w, int dst_h);

#endif /* SAM3_INTERNAL_MASK_RESIZE_H */
