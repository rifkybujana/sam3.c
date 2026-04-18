"""VideoSession - Python wrapper around sam3's video tracking API.

Mirrors the shape of sam3.model.Model: RAII via close()/__enter__/__exit__,
a _closed flag that __del__ checks, a _check_open() guard, and a cffi
callback that bridges the C propagation into a Python iterator.

Per-frame propagation results use a lightweight _FrameResult rather than
sam3.model.Result because the C struct's buffers are owned by the
propagator and freed after sam3_video_propagate returns; we copy into
numpy inside the callback to avoid a use-after-free.
"""
import numpy as np

from sam3._lib import ffi, lib
from sam3.errors import Sam3Error, check
from sam3.model import Result

__all__ = ["VideoSession"]


_DIRECTIONS = {
    "both":     lib.SAM3_PROPAGATE_BOTH,
    "forward":  lib.SAM3_PROPAGATE_FORWARD,
    "backward": lib.SAM3_PROPAGATE_BACKWARD,
}


class _FrameResult:
    """Per-frame propagation result.

    A frame yielded by ``VideoSession.propagate`` owns its numpy
    arrays outright; the underlying C ``struct sam3_result *`` is
    freed by the propagator as soon as the callback returns, so we
    cannot reuse ``sam3.model.Result`` (which would call
    ``sam3_result_free`` a second time).
    """

    __slots__ = ("masks", "iou_scores", "boxes", "iou_valid",
             "boxes_valid", "best_mask", "obj_ids")

    def __init__(self, masks, iou_scores, boxes, iou_valid,
             boxes_valid, best_mask, obj_ids):
        self.masks = masks
        self.iou_scores = iou_scores
        self.boxes = boxes
        self.iou_valid = iou_valid
        self.boxes_valid = boxes_valid
        self.best_mask = best_mask
        self.obj_ids = obj_ids


class VideoSession:
    """Track objects across frames of a video or frame directory.

    Usage::

        with sam3.Model("model.sam3") as model:
            with sam3.VideoSession(model, "clip.mp4") as sess:
                r = sess.add_points(frame=0, obj_id=0,
                                    points=[(64.0, 64.0, 1)])
                for frame_idx, fr in sess.propagate(direction="forward"):
                    # fr.masks: (n_masks, H, W) float32
                    # fr.obj_ids: list[int]
                    ...
    """

    def __init__(self, model, resource_path):
        self._closed = True
        self._session = ffi.NULL
        # Keep the model alive for the session's lifetime; the
        # session does not own it but dereferences its ctx.
        self._model = model

        if getattr(model, "_closed", True):
            raise ValueError("Model is closed")

        # If sam3_video_start fails, check() raises with _closed
        # still True, so __del__ will not try to free a NULL
        # session pointer.
        out = ffi.new("sam3_video_session **")
        check(lib.sam3_video_start(model._ctx,
                       resource_path.encode(), out))
        self._session = out[0]
        if self._session == ffi.NULL:
            raise Sam3Error(0, "sam3_video_start returned NULL")
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        if not self._closed:
            self.close()

    def close(self):
        """Release session resources. Safe to call multiple times."""
        if not self._closed:
            lib.sam3_video_end(self._session)
            self._session = ffi.NULL
            self._closed = True

    def _check_open(self):
        if self._closed:
            raise ValueError("VideoSession is closed")
        if getattr(self._model, "_closed", True):
            raise ValueError("Underlying Model is closed")

    def frame_count(self):
        """Return the number of encoded frames in the session."""
        self._check_open()
        return lib.sam3_video_frame_count(self._session)

    def add_points(self, *, frame, obj_id, points):
        """Add point prompts for an object on a frame.

        Args:
            frame:  Zero-based frame index.
            obj_id: Object identifier (0..SAM3_MAX_OBJECTS-1).
            points: Iterable of ``(x, y, label)`` tuples, where
                    ``label`` is 1 for foreground, 0 for background.

        Returns:
            ``sam3.Result`` for the prompted (frame, object).
        """
        self._check_open()
        pts = list(points)
        n = len(pts)
        if n == 0:
            raise ValueError("points must be non-empty")

        c_points = ffi.new("struct sam3_point[]", n)
        for i, (x, y, label) in enumerate(pts):
            c_points[i].x = float(x)
            c_points[i].y = float(y)
            c_points[i].label = int(label)

        c_result = ffi.new("struct sam3_result *")
        check(lib.sam3_video_add_points(self._session, int(frame),
                        int(obj_id), c_points, n,
                        c_result))
        return Result(c_result)

    def add_box(self, *, frame, obj_id, box):
        """Add a bounding-box prompt for an object on a frame.

        Args:
            frame:  Zero-based frame index.
            obj_id: Object identifier.
            box:    ``(x1, y1, x2, y2)`` tuple.

        Returns:
            ``sam3.Result`` for the prompted (frame, object).
        """
        self._check_open()
        x1, y1, x2, y2 = box
        c_box = ffi.new("struct sam3_box *")
        c_box.x1 = float(x1)
        c_box.y1 = float(y1)
        c_box.x2 = float(x2)
        c_box.y2 = float(y2)
        c_result = ffi.new("struct sam3_result *")
        check(lib.sam3_video_add_box(self._session, int(frame),
                         int(obj_id), c_box, c_result))
        return Result(c_result)

    def remove_object(self, obj_id):
        """Remove a tracked object from the session."""
        self._check_open()
        check(lib.sam3_video_remove_object(self._session,
                           int(obj_id)))

    def reset(self):
        """Clear all tracked objects; keep encoded frame features."""
        self._check_open()
        check(lib.sam3_video_reset(self._session))

    def propagate(self, *, direction="forward"):
        """Propagate tracked objects and yield per-frame results.

        Yields ``(frame_idx, _FrameResult)`` tuples in the order
        emitted by the C propagator.

        The C call is synchronous: propagation runs to completion
        (or to the first callback failure) before the first yield.
        Memory is bounded by one _FrameResult per visited frame.

        Args:
            direction: ``"forward"``, ``"backward"``, or ``"both"``.
        """
        self._check_open()
        if direction not in _DIRECTIONS:
            raise ValueError(
                "direction must be one of "
                f"{sorted(_DIRECTIONS)}, got {direction!r}")

        collected = []
        cb_exc = []

        @ffi.callback("int(int, const struct sam3_result *, int, "
                  "const int *, void *)")
        def _cb(frame_idx, c_result, n_objects, obj_ids, user_data):
            # Copy all buffers out of c_result before returning;
            # the propagator will free them.
            try:
                n = int(c_result.n_masks)
                h = int(c_result.mask_height)
                w = int(c_result.mask_width)

                if n > 0 and h > 0 and w > 0 and \
                        c_result.masks != ffi.NULL:
                    buf = ffi.buffer(c_result.masks,
                             n * h * w * 4)
                    masks = np.frombuffer(
                        buf,
                        dtype=np.float32).reshape(
                            n, h, w).copy()
                else:
                    masks = np.empty(
                        (0, 0, 0), dtype=np.float32)

                if n > 0 and c_result.iou_scores != ffi.NULL:
                    buf = ffi.buffer(c_result.iou_scores,
                             n * 4)
                    iou = np.frombuffer(
                        buf,
                        dtype=np.float32).copy()
                else:
                    iou = np.empty((0,), dtype=np.float32)

                if (c_result.boxes_valid and
                        c_result.boxes != ffi.NULL and
                        n > 0):
                    bb = ffi.buffer(c_result.boxes,
                            n * 4 * 4)
                    boxes = np.frombuffer(
                        bb,
                        dtype=np.float32).reshape(
                            n, 4).copy()
                else:
                    boxes = np.full(
                        (n, 4), np.nan,
                        dtype=np.float32)

                ids = [int(obj_ids[i])
                       for i in range(int(n_objects))]

                collected.append((
                    int(frame_idx),
                    _FrameResult(
                        masks, iou, boxes,
                        bool(c_result.iou_valid),
                        bool(c_result.boxes_valid),
                        int(c_result.best_mask),
                        ids)))
                return 0
            except Exception as exc:
                # Capture the exception so we can re-raise after
                # sam3_video_propagate returns. Returning -1 makes
                # the propagator stop cleanly; it then reports
                # SAM3_OK, so check() below would not surface the
                # error on its own.
                cb_exc.append(exc)
                return -1

        # Keep _cb alive until after propagate returns. cffi's
        # ffi.callback() anchors the C trampoline to the Python
        # object; once _cb goes out of scope the C pointer is
        # invalidated, but that's fine here because we only use
        # it synchronously inside this call.
        check(lib.sam3_video_propagate(
            self._session, _DIRECTIONS[direction], _cb, ffi.NULL))
        if cb_exc:
            raise cb_exc[0]

        for frame_idx, fr in collected:
            yield frame_idx, fr
