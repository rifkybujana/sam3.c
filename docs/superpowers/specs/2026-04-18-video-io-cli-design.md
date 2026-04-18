# Video I/O for `sam3 track`

Date: 2026-04-18
Status: Design approved

## Summary

Extend `sam3 track` so the CLI reads modern video formats (MP4, MOV, MKV,
WebM) and writes a single overlay video where per-object masks are
alpha-blended onto the source frames. The existing PNG-directory output
path is preserved unchanged — output mode is selected by the extension on
`--output`.

The implementation adds `libavformat`, `libavcodec`, `libswscale`, and
`libavutil` as a hard build dependency. The vendored `pl_mpeg.h` decoder
is retired in the same change, since libav supersedes it.

## CLI surface

```
sam3 track --model weights.sam3 \
           --video input.mp4 \
           --output result.mp4 \
           --point 512,384,1 --obj-id 0 \
           [--alpha 0.5] [--fps 24] [--propagate both]
```

### Mode detection on `--output`

- Ends in `.mp4`, `.mov`, `.mkv`, or `.webm` → video mode.
- Anything else → PNG-directory mode (current behavior, unchanged).

### New flags

- `--alpha <f>` — overlay alpha in `[0, 1]`, default `0.5`. Ignored in
  PNG-directory mode. Out-of-range values are a parse error.
- `--fps <n>` — output frame rate. **Required** in video mode when the
  input is a frame directory; **ignored** when the input is a video file
  (rate inherits from the source stream's `avg_frame_rate`).

### Unchanged

`--point`, `--box`, `--obj-id`, `--frame`, `--propagate`, `-v`,
`--profile`. Exit codes reuse existing `SAM3_EXIT_*`; encoder-path
failures map to `SAM3_EXIT_IO`.

## Architecture

Two passes driven from `cli_track_run`:

1. **Propagate pass.** Run the existing video session + propagate flow.
   The frame callback thresholds each per-object mask (`>= 0.0f → 255`)
   and stores it into a preallocated buffer keyed by
   `(frame_idx, obj_id)`. A `seen[frame_idx]` byte array tracks which
   frames the propagation visited.

2. **Encode pass.** After propagate returns, open the source again with
   a thin libav decode loop in `cli_track` (raw RGB24 at native
   resolution — distinct from `util/video`, which produces
   model-normalized tensors). Iterate frames in order. For each frame,
   if `seen`, composite each object's mask onto the RGB buffer in
   `obj_id` order. Push the resulting RGB24 into the encoder.

The two-pass design is required because `sam3_video_propagate` under
`--propagate both` emits frames out of strict index order. Buffering
binary masks at model resolution is cheap; buffering full-resolution
RGB frames would not be.

### Why not stream frames straight from propagate?

Propagation order is non-monotonic. Streaming would require either
multi-pass seek per frame in libav or an order-agnostic encoder, neither
of which is worth the complexity compared to a small mask buffer.

## Module layout

### Modified

- **`src/util/video.{c,h}`** — MPEG-1 path (`load_mpeg` via `pl_mpeg`)
  replaced by `load_libav` using libavformat/libavcodec/libswscale.
  Frame-directory path (stb_image) unchanged. Public API
  (`sam3_video_detect_type`, `sam3_video_load`) unchanged; internals
  only. `sam3_video_frames` gains `int fps_num, fps_den` (set by
  `load_libav`, left `0/1` by `load_frame_dir`).
- **`tools/cli_track.{c,h}`** — `track_args` gains `output_mode`,
  `alpha`, `fps`. `cli_track_run` branches between dir mode (current
  behavior) and video mode (two-pass flow above).
- **`CMakeLists.txt`** — `find_package(PkgConfig REQUIRED)` plus
  `pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET libavformat
  libavcodec libswscale libavutil)`; link `PkgConfig::LIBAV` into the
  utility library; drop any `pl_mpeg.h` wiring.

### Retired

- **`src/util/vendor/pl_mpeg.h`** — deleted. No other files reference it
  once `video.c` switches.

### New

- **`src/util/video_encode.{c,h}`** — libav-backed encoder plus a
  libav-free overlay helper. See API below.

### Tests

- **`tests/test_video_encode.c`** — new.
- **`tests/test_cli_track.c`** — extended for `--alpha`, `--fps`, and
  extension-based mode detection.

## `util/video_encode` public API

```c
struct sam3_video_encoder; /* opaque */

enum sam3_error sam3_video_encoder_open(
    const char *path, int width, int height,
    int fps_num, int fps_den,
    struct sam3_video_encoder **out);

enum sam3_error sam3_video_encoder_write_rgb(
    struct sam3_video_encoder *e, const uint8_t *rgb);

enum sam3_error sam3_video_encoder_close(
    struct sam3_video_encoder *e);

void sam3_overlay_composite(
    uint8_t *rgb, int w, int h,
    const uint8_t *mask, int mw, int mh, /* binary 0/255, model res */
    int obj_id, float alpha);
```

### Encoder open behavior

1. `avformat_alloc_output_context2` infers the container from `path`.
2. Codec chosen per container: `.mp4`/`.mov`/`.mkv` → H.264 via
   `libx264`; `.webm` → VP9 (`libvpx-vp9`).
3. If the encoder isn't compiled into the linked libav, return
   `SAM3_EIO` and log the missing codec plus a hint to rebuild ffmpeg
   with it enabled.
4. Configure codec context: `pix_fmt = YUV420P`,
   `time_base = {fps_den, fps_num}`, `framerate = {fps_num, fps_den}`,
   `gop_size = 12`, `max_b_frames = 0`. For H.264 set `preset=medium`,
   `crf=23` via `AV_OPT_*`.
5. Write container header. Allocate reusable `AVFrame` (YUV420P) and
   `AVPacket`. Create `sws_ctx` for RGB24 → YUV420P once.
6. Stash all state in the opaque struct owned by the encoder.

### Encoder write behavior

1. `sws_scale` RGB24 → YUV420P into the preallocated frame.
2. `frame->pts` = monotonically increasing counter (0, 1, 2, …).
3. `avcodec_send_frame` → drain `avcodec_receive_packet` → rescale
   packet timestamps → `av_interleaved_write_frame`.
4. libav errors map to `SAM3_EIO` with `av_err2str` included in the log.

### Encoder close behavior

1. Flush: `avcodec_send_frame(NULL)`, drain `receive_packet`.
2. `av_write_trailer`, close codec, close I/O, free `sws_ctx`,
   `AVFrame`, `AVPacket`, format context, wrapper struct.
3. **Idempotent.** Safe on `NULL`. Safe on a partially-opened encoder —
   callers use the standard `goto cleanup` pattern.

### Overlay helper

- 10-color built-in palette (`static const uint8_t palette[10][3]`),
  cycled by `obj_id % 10`.
- For each `(x, y)` in the output, map to mask coords
  `(x * mw / w, y * mh / h)` — nearest-neighbor upscale.
- If the mask pixel is non-zero, blend:
  `rgb[i] = rgb[i] * (1 - alpha) + color[i] * alpha`.
- No allocations, no libav dependency. Pure C, testable in isolation.

## Decode path (`util/video.c`)

`load_libav` replaces `load_mpeg`:

1. `avformat_open_input` + `avformat_find_stream_info`. Reject if no
   video stream.
2. `avcodec_find_decoder` + `avcodec_open2` on the best video stream.
3. Record `orig_width`, `orig_height`, `fps_num`, `fps_den` (from
   `stream->avg_frame_rate`).
4. First pass to count frames — iterate packets, decode, count. Same
   shape as the current `load_mpeg` code.
5. Rewind: `avformat_seek_file(ctx, -1, INT64_MIN, 0, 0, 0)` +
   `avcodec_flush_buffers`.
6. Second pass: decode each frame, convert to RGB24 at native
   resolution via `sws_scale`, resize to `image_size × image_size` with
   `stbir_resize_uint8_linear`, then `make_frame_tensor` into the
   arena.
7. Unconditional cleanup on all exits.

`sam3_video_detect_type` changes: any regular file → `SAM3_VIDEO_FILE`
(the enum value formerly `SAM3_VIDEO_MPEG` — renamed for accuracy).
Extension whitelisting is removed; libav reports unsupported formats on
open, with better error text than an extension check.

## `cli_track` flow

### Pass 1 — propagate + buffer masks

```c
struct video_frame_ctx {
    int n_frames;             /* from session */
    int mask_w, mask_h;       /* model resolution */
    int n_objects;            /* distinct obj_ids used by --obj-id */
    uint8_t *masks;           /* n_frames * n_objects * mask_h * mask_w, 0/255 */
    uint8_t *seen;            /* n_frames, 1 if visited by propagate */
};
```

- `n_objects` = count of distinct `obj_id` values across the parsed
  prompts (not `SAM3_MAX_OBJECTS` — bounding by actual use avoids
  gigabyte-scale over-allocation).
- Callback thresholds each object's mask (`>= 0.0f → 255`) into the
  right slot and sets `seen[frame_idx] = 1`.
- Frames never visited (e.g. pre-prompt under `--propagate forward`)
  stay `seen = 0` and get no overlay in pass 2.

### Pass 2 — decode source, composite, encode

1. Output dimensions = `frames->orig_width × orig_height`.
2. Output fps: if `frames->fps_num > 0` use it; otherwise use
   `args->fps` (parser enforces presence).
3. `sam3_video_encoder_open(args->output, W, H, fps_num, fps_den, &enc)`.
4. Reopen the source via a minimal libav decode loop inside
   `cli_track` (raw RGB24 at native resolution). For frame-directory
   input, iterate the sorted file list via `stbi_load`; an internal
   helper is exposed from `util/video.c` so listing/sorting isn't
   duplicated.
5. Per decoded frame: if `seen[frame_idx]`, composite each object's
   mask (ascending `obj_id`) onto the RGB buffer. Push to the encoder.
6. `sam3_video_encoder_close(enc)`.

### Memory

- Mask buffer: allocated once; freed after pass 2.
- RGB scratch (`W * H * 3`): allocated once, reused per frame
  (Performance Rule #1).
- Encoder state: freed by `_close`.

### Error paths

All named; every one has a matching `sam3_log_error`:

- libav open failure → include `av_err2str` and the path.
- No video stream → `SAM3_EIO`.
- Encoder not compiled into libav → `SAM3_EIO`, name the codec.
- `--fps` missing when required → parse error (`SAM3_EXIT_USAGE`).
- `--alpha` out of range → parse error.
- Mask buffer allocation failure → `SAM3_ENOMEM`, log size in MB.
- Encoder mid-flight failure → log frame index + `av_err2str`, then
  `goto cleanup` so `_close` still finalizes or closes the file.

## Build

- `CMakeLists.txt`:
  ```
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
      libavformat libavcodec libswscale libavutil)
  ```
  Link `PkgConfig::LIBAV` into the utility library target.
- Wrap libav headers in the same `#pragma clang/GCC diagnostic push/pop`
  block `video.c` already uses for vendor headers (debug builds run
  with `-Werror`, and libav triggers a few warnings).
- README: one-liner install hint — `brew install ffmpeg` (macOS),
  `apt install libavformat-dev libavcodec-dev libswscale-dev
  libavutil-dev` (Debian/Ubuntu).

## Testing

### `tests/test_video_encode.c` (new)

- `test_encoder_roundtrip_mp4` — open encoder at 16×16 @ 10 fps, write
  24 synthetic gradient frames, close. Reopen via libav, assert
  dimensions and frame count; decode at least one frame and check the
  expected pattern.
- `test_encoder_close_is_idempotent` — double-close, close of `NULL`.
- `test_encoder_rejects_unknown_extension` — `.xyz` → `SAM3_EIO`, no
  file created.
- `test_overlay_composite_solid` — `alpha=1.0` replaces pixels with
  palette color; `alpha=0.0` leaves them unchanged; mixed alpha blends
  within tolerance.
- `test_overlay_composite_upscale` — 4×4 mask → 16×16 RGB, nearest
  mapping correct at corners and centers.

### `tests/test_cli_track.c` (extended)

- `--alpha` parses; rejects `-1` and `2` with a parse error.
- `--fps` parses; missing `--fps` with frame-dir input + `.mp4` output
  is a parse error; `--fps` with video input is accepted (and ignored
  downstream).
- Output-mode detection: `.mp4`/`.mov`/`.mkv`/`.webm` → video; any
  other path or `output/` → dir.

All tests registered through existing CTest wiring.

## Performance

- Mask + RGB buffers allocated once (Rule #1).
- `sws_ctx` created once per decoder and once per encoder (Rule #7).
- Overlay is a tight per-pixel loop with no inner-loop allocations,
  favoring arithmetic over branches where possible (Rule #5).
- Benchmarks run in Release builds without sanitizers (Rule #8).

## Out of scope (YAGNI)

Called out explicitly so they don't resurface as surprises:

- Audio passthrough from the source video.
- Custom per-object colors via CLI.
- Codec / bitrate override flags.
- Contour/outline or side-by-side overlay modes.
- Retaining `pl_mpeg` as a fallback when libav is absent.
- GPU-accelerated encode.
