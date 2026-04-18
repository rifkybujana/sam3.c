/*
 * src/util/video_encode.h - Encoder + overlay helper for video output
 *
 * The encoder wraps libavformat/libavcodec/libswscale to write RGB24
 * frames into a container whose format is inferred from the output
 * path. The overlay helper is a pure C function (no libav dep) that
 * composites a binary mask onto an RGB24 buffer using a per-object
 * color palette.
 *
 * Key types:  sam3_video_encoder
 * Depends on: sam3/sam3_types.h
 * Used by:    src/util/video_encode.c, tools/cli_track.c,
 *             tests/test_video_encode.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_UTIL_VIDEO_ENCODE_H
#define SAM3_UTIL_VIDEO_ENCODE_H

#include <stdint.h>

#include "sam3/sam3_types.h"

struct sam3_video_encoder; /* opaque */

/*
 * sam3_video_encoder_open - Open an encoder for @path at @w x @h @ fps.
 *
 * @path:    Output file path. Container inferred from extension
 *           (.mp4/.mov/.mkv -> H.264; .webm -> VP9).
 * @width:   Frame width in pixels (> 0).
 * @height:  Frame height in pixels (> 0).
 * @fps_num: Frame-rate numerator (> 0).
 * @fps_den: Frame-rate denominator (> 0).
 * @out:     Receives heap-allocated encoder on success.
 *
 * Returns SAM3_OK, SAM3_EINVAL for bad args, SAM3_EIO for libav
 * failures (including missing codec support in the linked ffmpeg
 * build), SAM3_ENOMEM on allocation failure.
 */
enum sam3_error sam3_video_encoder_open(const char *path,
					int width, int height,
					int fps_num, int fps_den,
					struct sam3_video_encoder **out);

/*
 * sam3_video_encoder_write_rgb - Encode one RGB24 frame.
 *
 * @enc: Encoder from sam3_video_encoder_open.
 * @rgb: RGB24 buffer of size width * height * 3, tightly packed.
 *
 * Returns SAM3_OK on success; SAM3_EIO if libav rejects the frame.
 */
enum sam3_error sam3_video_encoder_write_rgb(struct sam3_video_encoder *enc,
					     const uint8_t *rgb);

/*
 * sam3_video_encoder_close - Flush, finalize, and free.
 *
 * Idempotent. Safe on NULL. Safe on a partially-opened encoder.
 * Returns SAM3_OK on success, SAM3_EIO if flush/trailer fails. The
 * encoder is freed either way.
 */
enum sam3_error sam3_video_encoder_close(struct sam3_video_encoder *enc);

/*
 * sam3_overlay_composite - Alpha-blend a binary mask onto RGB24 in place.
 *
 * @rgb:    RGB24 buffer of size w * h * 3 (modified in place).
 * @w, @h:  Output dimensions.
 * @mask:   Binary mask (0 or non-zero per byte) of size mw * mh.
 * @mw, @mh: Mask dimensions; upscaled to (w, h) with nearest neighbor.
 * @obj_id: Object id; palette color = palette[obj_id % 10].
 * @alpha:  Blend factor in [0, 1]. 0 leaves rgb untouched, 1 replaces.
 *
 * Out-of-range alpha is clamped; out-of-range obj_id wraps modulo 10.
 * No allocations, no libav dependency.
 */
void sam3_overlay_composite(uint8_t *rgb, int w, int h,
			    const uint8_t *mask, int mw, int mh,
			    int obj_id, float alpha);

#endif /* SAM3_UTIL_VIDEO_ENCODE_H */
