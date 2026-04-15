"""High-level Pythonic interface to sam3 inference."""
import numpy as np

from sam3._lib import ffi, lib
from sam3.errors import Sam3Error, check


class Result:
    """Segmentation result with numpy arrays.

    Attributes:
        masks:       np.ndarray, shape (n_masks, H, W), float32.
        iou_scores:  np.ndarray, shape (n_masks,), float32.
        boxes:       np.ndarray, shape (n_masks, 4), float32 (xyxy).
        best_mask:   int, stability-selected mask index (-1 if N/A).
        iou_valid:   bool, whether iou_scores are model-predicted.
        boxes_valid: bool, whether boxes are computed.
    """

    def __init__(self, c_result):
        n = c_result.n_masks
        h = c_result.mask_height
        w = c_result.mask_width

        # Copy mask data into numpy arrays
        buf = ffi.buffer(c_result.masks, n * h * w * 4)
        self.masks = np.frombuffer(buf, dtype=np.float32).reshape(n, h, w).copy()

        buf = ffi.buffer(c_result.iou_scores, n * 4)
        self.iou_scores = np.frombuffer(buf, dtype=np.float32).copy()

        self.iou_valid = bool(c_result.iou_valid)
        self.boxes_valid = bool(c_result.boxes_valid)
        self.best_mask = c_result.best_mask

        if c_result.boxes_valid and c_result.boxes != ffi.NULL:
            buf = ffi.buffer(c_result.boxes, n * 4 * 4)
            self.boxes = np.frombuffer(buf, dtype=np.float32).reshape(n, 4).copy()
        else:
            self.boxes = np.empty((n, 4), dtype=np.float32)
            self.boxes[:] = np.nan

        # Free C memory now that we have copies
        lib.sam3_result_free(c_result)


class Model:
    """SAM3 segmentation model.

    Usage::

        with sam3.Model("model.sam3") as model:
            model.set_image("photo.jpg")
            result = model.segment(points=[(500, 300, 1)])
            print(result.masks.shape)

    Args:
        model_path: Path to .sam3 model weights file.
        bpe_path:   Optional path to BPE vocabulary file.
    """

    def __init__(self, model_path, bpe_path=None):
        self._closed = True
        self._ctx = lib.sam3_init()
        if self._ctx == ffi.NULL:
            raise Sam3Error(0, "Failed to initialize sam3 context")
        self._closed = False

        try:
            check(lib.sam3_load_model(self._ctx, model_path.encode()))
            if bpe_path is not None:
                check(lib.sam3_load_bpe(self._ctx, bpe_path.encode()))
        except Exception:
            lib.sam3_free(self._ctx)
            self._ctx = ffi.NULL
            self._closed = True
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        if not self._closed:
            self.close()

    def close(self):
        """Release all resources. Safe to call multiple times."""
        if not self._closed:
            lib.sam3_free(self._ctx)
            self._ctx = ffi.NULL
            self._closed = True

    def _check_open(self):
        if self._closed:
            raise ValueError("Model is closed")

    def load_bpe(self, path):
        """Load BPE vocabulary for text prompts."""
        self._check_open()
        check(lib.sam3_load_bpe(self._ctx, path.encode()))

    def set_image(self, image):
        """Set the input image.

        Args:
            image: Either a file path (str) or a numpy array of shape
                   (H, W, 3) with dtype uint8 (RGB).
        """
        self._check_open()
        if isinstance(image, str):
            check(lib.sam3_set_image_file(self._ctx, image.encode()))
        else:
            image = np.ascontiguousarray(image, dtype=np.uint8)
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    f"Expected (H, W, 3) RGB array, got shape {image.shape}"
                )
            h, w = image.shape[:2]
            ptr = ffi.cast("const uint8_t *", image.ctypes.data)
            check(lib.sam3_set_image(self._ctx, ptr, w, h))

    def set_prompt_space(self, width, height):
        """Set the coordinate space for point/box prompts."""
        self._check_open()
        lib.sam3_set_prompt_space(self._ctx, width, height)

    def set_text(self, text):
        """Pre-tokenize and asynchronously encode a text prompt."""
        self._check_open()
        check(lib.sam3_set_text(self._ctx, text.encode()))

    def get_image_size(self):
        """Return the model's expected input image size."""
        self._check_open()
        return lib.sam3_get_image_size(self._ctx)

    def segment(self, *, points=None, boxes=None, masks=None, text=None):
        """Run segmentation with the given prompts.

        Args:
            points: List of (x, y, label) tuples. label=1 foreground, 0 background.
            boxes:  List of (x1, y1, x2, y2) tuples.
            masks:  List of numpy arrays, each shape (H, W) float32.
            text:   A string text prompt.

        Returns:
            Result with masks, iou_scores, boxes, and best_mask.
        """
        self._check_open()

        prompt_list = []
        # Keep references alive until segment() completes
        _keepalive = []

        if points:
            for x, y, label in points:
                p = ffi.new("struct sam3_prompt *")
                p.type = lib.SAM3_PROMPT_POINT
                p.point.x = x
                p.point.y = y
                p.point.label = label
                prompt_list.append(p)

        if boxes:
            for x1, y1, x2, y2 in boxes:
                p = ffi.new("struct sam3_prompt *")
                p.type = lib.SAM3_PROMPT_BOX
                p.box.x1 = x1
                p.box.y1 = y1
                p.box.x2 = x2
                p.box.y2 = y2
                prompt_list.append(p)

        if masks:
            for mask_arr in masks:
                mask_arr = np.ascontiguousarray(mask_arr, dtype=np.float32)
                if mask_arr.ndim != 2:
                    raise ValueError(f"Mask must be 2D, got {mask_arr.ndim}D")
                p = ffi.new("struct sam3_prompt *")
                p.type = lib.SAM3_PROMPT_MASK
                p.mask.data = ffi.cast("const float *", mask_arr.ctypes.data)
                p.mask.height = mask_arr.shape[0]
                p.mask.width = mask_arr.shape[1]
                prompt_list.append(p)
                _keepalive.append(mask_arr)

        if text:
            encoded = text.encode()
            p = ffi.new("struct sam3_prompt *")
            p.type = lib.SAM3_PROMPT_TEXT
            p.text = encoded
            prompt_list.append(p)
            _keepalive.append(encoded)

        if not prompt_list:
            raise ValueError("At least one prompt is required")

        # Build contiguous array
        c_prompts = ffi.new("struct sam3_prompt[]", len(prompt_list))
        for i, p in enumerate(prompt_list):
            c_prompts[i] = p[0]

        c_result = ffi.new("struct sam3_result *")
        check(lib.sam3_segment(self._ctx, c_prompts, len(prompt_list), c_result))

        return Result(c_result)

    def profile_enable(self):
        """Enable profiling (requires SAM3_HAS_PROFILE at compile time)."""
        self._check_open()
        check(lib.sam3_profile_enable(self._ctx))

    def profile_disable(self):
        """Disable profiling (data is preserved)."""
        self._check_open()
        lib.sam3_profile_disable(self._ctx)

    def profile_report(self):
        """Print profiling report to stderr."""
        self._check_open()
        lib.sam3_profile_report(self._ctx)

    def profile_reset(self):
        """Clear all collected profiling data."""
        self._check_open()
        lib.sam3_profile_reset(self._ctx)
