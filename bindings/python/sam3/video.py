"""VideoSession - Python wrapper around sam3's video tracking API.

Mirrors the shape of sam3.model.Model: RAII via close()/__enter__/__exit__,
a _closed flag that __del__ checks, and a _check_open() guard. The C API
emits a per-object sam3_video_frame_result whose buffers are owned by the
engine and freed via sam3_video_frame_result_free (for prompt entry
points) or by the engine after the callback returns (for propagate). We
copy mask logits into numpy inside the conversion path to decouple user
code from the C lifetime.
"""
import numpy as np

from sam3._lib import ffi, lib
from sam3.errors import Sam3Error, check

__all__ = ["VideoSession", "FrameResult", "ObjectMask", "StartOpts"]


_DIRECTIONS = {
    "both":     lib.SAM3_PROPAGATE_BOTH,
    "forward":  lib.SAM3_PROPAGATE_FORWARD,
    "backward": lib.SAM3_PROPAGATE_BACKWARD,
}


class ObjectMask:
    """Per-object segmentation output for one video frame.

    Attributes:
        obj_id:          The user-supplied object identifier.
        mask:            np.ndarray (H, W) float32 logits; threshold at 0.
        iou_score:       Predicted IoU for the mask.
        obj_score_logit: >0 visible, <=0 occluded.
        is_occluded:     Convenience bool derived from obj_score_logit.
    """

    __slots__ = ("obj_id", "mask", "iou_score",
             "obj_score_logit", "is_occluded")

    def __init__(self, obj_id, mask, iou_score,
             obj_score_logit, is_occluded):
        self.obj_id = obj_id
        self.mask = mask
        self.iou_score = iou_score
        self.obj_score_logit = obj_score_logit
        self.is_occluded = is_occluded


class FrameResult:
    """Multi-object result for one video frame.

    Attributes:
        frame_idx: Zero-based frame index.
        objects:   List of ObjectMask, one per tracked object.
    """

    __slots__ = ("frame_idx", "objects")

    def __init__(self, frame_idx, objects):
        self.frame_idx = frame_idx
        self.objects = objects

    def by_obj_id(self, obj_id):
        """Return the ObjectMask for obj_id, or None if not present."""
        for o in self.objects:
            if o.obj_id == obj_id:
                return o
        return None


class StartOpts:
    """Tunables for VideoSession construction.

    All fields map one-to-one to sam3_video_start_opts; pass None for a
    field to accept the libsam3 default. Field semantics mirror the C
    header (include/sam3/sam3.h).
    """

    __slots__ = (
        "frame_cache_backend_budget",
        "frame_cache_spill_budget",
        "clear_non_cond_window",
        "iter_use_prev_mask_pred",
        "multimask_via_stability",
        "multimask_stability_delta",
        "multimask_stability_thresh",
    )

    def __init__(self, *,
             frame_cache_backend_budget=None,
             frame_cache_spill_budget=None,
             clear_non_cond_window=None,
             iter_use_prev_mask_pred=None,
             multimask_via_stability=None,
             multimask_stability_delta=None,
             multimask_stability_thresh=None):
        self.frame_cache_backend_budget = frame_cache_backend_budget
        self.frame_cache_spill_budget = frame_cache_spill_budget
        self.clear_non_cond_window = clear_non_cond_window
        self.iter_use_prev_mask_pred = iter_use_prev_mask_pred
        self.multimask_via_stability = multimask_via_stability
        self.multimask_stability_delta = multimask_stability_delta
        self.multimask_stability_thresh = multimask_stability_thresh

    def _to_c(self):
        c = ffi.new("struct sam3_video_start_opts *")
        # Map None -> sentinel understood by libsam3 (0 for sizes / floats,
        # -1 for ints where the C default is "-1 selects default on").
        c.frame_cache_backend_budget = (
            0 if self.frame_cache_backend_budget is None
            else int(self.frame_cache_backend_budget))
        c.frame_cache_spill_budget = (
            0 if self.frame_cache_spill_budget is None
            else int(self.frame_cache_spill_budget))
        c.clear_non_cond_window = (
            0 if self.clear_non_cond_window is None
            else int(self.clear_non_cond_window))
        c.iter_use_prev_mask_pred = (
            -1 if self.iter_use_prev_mask_pred is None
            else int(self.iter_use_prev_mask_pred))
        c.multimask_via_stability = (
            -1 if self.multimask_via_stability is None
            else int(self.multimask_via_stability))
        c.multimask_stability_delta = (
            0.0 if self.multimask_stability_delta is None
            else float(self.multimask_stability_delta))
        c.multimask_stability_thresh = (
            0.0 if self.multimask_stability_thresh is None
            else float(self.multimask_stability_thresh))
        return c


def _object_mask_from_c(c_obj):
    """Copy a sam3_video_object_mask into an ObjectMask.

    The mask buffer is a defensive copy so it outlives the C struct.
    """
    h = int(c_obj.mask_h)
    w = int(c_obj.mask_w)
    if h > 0 and w > 0 and c_obj.mask != ffi.NULL:
        buf = ffi.buffer(c_obj.mask, h * w * 4)
        mask = np.frombuffer(buf, dtype=np.float32).reshape(h, w).copy()
    else:
        mask = np.empty((0, 0), dtype=np.float32)
    return ObjectMask(
        obj_id=int(c_obj.obj_id),
        mask=mask,
        iou_score=float(c_obj.iou_score),
        obj_score_logit=float(c_obj.obj_score_logit),
        is_occluded=bool(c_obj.is_occluded),
    )


def _frame_result_from_c(c_result):
    """Copy a sam3_video_frame_result into a FrameResult."""
    n = int(c_result.n_objects)
    objs = []
    if n > 0 and c_result.objects != ffi.NULL:
        for i in range(n):
            objs.append(_object_mask_from_c(c_result.objects[i]))
    return FrameResult(frame_idx=int(c_result.frame_idx), objects=objs)


class VideoSession:
    """Track objects across frames of a video or frame directory.

    Usage::

        with sam3.Model("model.sam3") as model:
            with sam3.VideoSession(model, "clip.mp4") as sess:
                r = sess.add_points(frame=0, obj_id=0,
                                    points=[(64.0, 64.0, 1)])
                for fr in sess.propagate(direction="forward"):
                    # fr.frame_idx: int
                    # fr.objects: list[ObjectMask]
                    ...
    """

    def __init__(self, model, resource_path, *, opts=None):
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
        path = resource_path.encode()
        if opts is None:
            check(lib.sam3_video_start(model._ctx, path, out))
        else:
            if not isinstance(opts, StartOpts):
                raise TypeError("opts must be a StartOpts instance")
            c_opts = opts._to_c()
            check(lib.sam3_video_start_ex(
                model._ctx, path, c_opts, out))
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

    def _call_with_result(self, c_fn, *args):
        """Run `c_fn(*args, &result)`, then copy+free and return FrameResult."""
        c_result = ffi.new("struct sam3_video_frame_result *")
        try:
            check(c_fn(*args, c_result))
            return _frame_result_from_c(c_result)
        finally:
            lib.sam3_video_frame_result_free(c_result)

    def add_points(self, *, frame, obj_id, points):
        """Add point prompts for an object on a frame.

        Args:
            frame:  Zero-based frame index.
            obj_id: Object identifier (0..SAM3_MAX_OBJECTS-1).
            points: Iterable of ``(x, y, label)`` tuples, where
                    ``label`` is 1 for foreground, 0 for background.

        Returns:
            ``FrameResult`` with ``objects`` containing the prompted
            object's mask (list length 1).
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

        return self._call_with_result(
            lib.sam3_video_add_points,
            self._session, int(frame), int(obj_id), c_points, n)

    def add_box(self, *, frame, obj_id, box):
        """Add a bounding-box prompt for an object on a frame.

        Args:
            frame:  Zero-based frame index.
            obj_id: Object identifier.
            box:    ``(x1, y1, x2, y2)`` tuple.

        Returns:
            ``FrameResult`` with the prompted object's mask.
        """
        self._check_open()
        x1, y1, x2, y2 = box
        c_box = ffi.new("struct sam3_box *")
        c_box.x1 = float(x1)
        c_box.y1 = float(y1)
        c_box.x2 = float(x2)
        c_box.y2 = float(y2)
        return self._call_with_result(
            lib.sam3_video_add_box,
            self._session, int(frame), int(obj_id), c_box)

    def add_mask(self, *, frame, obj_id, mask):
        """Add a binary mask prompt for an object on a frame.

        Bypasses the SAM mask decoder: the given mask is resized to the
        session's internal high-res size, run through the memory encoder,
        and committed as a conditioning entry for the object.

        Args:
            frame:  Zero-based frame index.
            obj_id: Object identifier.
            mask:   2D numpy array, 0 = background, non-zero = foreground.
                    Dtype is cast to uint8; any shape <= 2*image_size is
                    accepted (nearest-neighbor resize).

        Returns:
            ``FrameResult`` with the prompted object's mask.
        """
        self._check_open()
        mask = np.ascontiguousarray(mask, dtype=np.uint8)
        if mask.ndim != 2:
            raise ValueError(
                f"mask must be 2D, got {mask.ndim}D")
        h, w = mask.shape
        if h == 0 or w == 0:
            raise ValueError("mask must be non-empty")
        ptr = ffi.cast("const uint8_t *", mask.ctypes.data)
        return self._call_with_result(
            lib.sam3_video_add_mask,
            self._session, int(frame), int(obj_id), ptr, h, w)

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

        Yields ``FrameResult`` instances in the order emitted by the C
        propagator.

        The C call is synchronous: propagation runs to completion (or to
        the first callback failure) before the first yield. Memory is
        bounded by one FrameResult per visited frame.

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

        @ffi.callback("int(const struct sam3_video_frame_result *, "
                  "void *)")
        def _cb(c_result, _user_data):
            # Copy all buffers out of c_result before returning;
            # the propagator owns them and frees after the callback.
            try:
                collected.append(_frame_result_from_c(c_result))
                return 0
            except Exception as exc:
                # Capture the exception so we can re-raise after
                # sam3_video_propagate returns. Returning non-zero
                # makes the propagator stop cleanly; it then reports
                # SAM3_OK, so check() below would not surface the
                # error on its own.
                cb_exc.append(exc)
                return 1

        # Keep _cb alive until after propagate returns. cffi's
        # ffi.callback() anchors the C trampoline to the Python
        # object; once _cb goes out of scope the C pointer is
        # invalidated, but that's fine here because we only use
        # it synchronously inside this call.
        check(lib.sam3_video_propagate(
            self._session, _DIRECTIONS[direction], _cb, ffi.NULL))
        if cb_exc:
            raise cb_exc[0]

        for fr in collected:
            yield fr
