"""Integration tests for sam3.VideoSession video tracking.

Generates an 8-frame moving-square clip in a tmp directory and checks
that VideoSession.propagate() produces per-frame masks whose centroids
track the ground-truth diagonal motion within a small pixel tolerance.

Requires SAM3_MODEL_PATH env var to point at a real .sam3 checkpoint;
otherwise the test self-skips via the shared model_path fixture.
Also requires PIL for writing PNG frames (self-skip if missing).
"""
import numpy as np
import pytest

import sam3

PIL_Image = pytest.importorskip("PIL.Image")


# Mirrors the C-level e2e test parameters in tests/test_video_e2e.c so
# both layers exercise the same synthetic clip. Only the square size and
# step need to match exactly; the rest is just "reasonable defaults".
_IMG_SIZE = 256
_N_FRAMES = 8
_SQUARE_SIZE = 32
_SQUARE_START = 100
_SQUARE_STEP = 8


def _square_center(i):
    """Ground-truth center of the square on frame i, in PNG pixels."""
    c = _SQUARE_START + i * _SQUARE_STEP + _SQUARE_SIZE * 0.5
    return c, c


@pytest.fixture
def moving_square_dir(tmp_path):
    """Write _N_FRAMES PNGs of a diagonally moving white square.

    Background is reproducible gray noise in [100, 156). Returned path
    is a directory that sam3_video_start() accepts as a frame
    directory (any .png/.jpg in the dir is used in sorted order).
    """
    rng = np.random.default_rng(0xC0FFEE)
    for i in range(_N_FRAMES):
        # Uniform noise background in [100, 156).
        frame = rng.integers(100, 156, size=(_IMG_SIZE, _IMG_SIZE, 3),
                     dtype=np.uint8)
        x0 = _SQUARE_START + i * _SQUARE_STEP
        y0 = x0
        frame[y0:y0 + _SQUARE_SIZE,
              x0:x0 + _SQUARE_SIZE, :] = 255
        PIL_Image.fromarray(frame, "RGB").save(
            tmp_path / f"frame_{i:04d}.png")
    return tmp_path


def _mask_centroid(mask):
    """Binary centroid of a single mask plane (logits >= 0)."""
    ys, xs = np.where(mask >= 0.0)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def test_video_session_tracks_moving_square(model_path, moving_square_dir):
    """End-to-end: prompt frame 0, propagate forward, assert tracking.

    Centroid tolerance is 16 PNG px: the E2E C test uses 8 px, but
    the Python path goes through an extra numpy copy and the mask is
    downsampled from the model's internal resolution, so we leave a
    bit more headroom to avoid flakes.
    """
    with sam3.Model(model_path) as model:
        with sam3.VideoSession(model, str(moving_square_dir)) as sess:
            assert sess.frame_count() == _N_FRAMES

            # Prompt in the model's input coordinate space,
            # matching what sam3_set_image_file does when the
            # clip is loaded (frames are letterboxed to the
            # model input size). Without calling
            # sam3_set_prompt_space on a video session, prompts
            # are interpreted in the model input space.
            img_size = model.get_image_size()
            assert img_size > 0
            scale = img_size / _IMG_SIZE

            gt_cx, gt_cy = _square_center(0)
            r0 = sess.add_points(
                frame=0, obj_id=0,
                points=[(gt_cx * scale, gt_cy * scale, 1)])

            assert r0.masks.ndim == 3
            assert r0.masks.shape[0] >= 1

            frames = list(sess.propagate(direction="forward"))

            assert len(frames) == _N_FRAMES

            mask_h = mask_w = 0
            centroids = [None] * _N_FRAMES
            for frame_idx, fr in frames:
                assert 0 <= frame_idx < _N_FRAMES
                assert fr.masks.ndim == 3
                if mask_w == 0:
                    mask_h = fr.masks.shape[1]
                    mask_w = fr.masks.shape[2]
                idx = fr.best_mask if fr.best_mask >= 0 else 0
                if idx >= fr.masks.shape[0]:
                    idx = 0
                c = _mask_centroid(fr.masks[idx])
                centroids[frame_idx] = c

            assert mask_w > 0 and mask_h > 0

            # Convert mask-pixel centroids back to PNG pixel
            # space and compare against the ground truth.
            px_per_mask_x = _IMG_SIZE / mask_w
            px_per_mask_y = _IMG_SIZE / mask_h
            tol_px = 16.0

            for i, c in enumerate(centroids):
                assert c is not None, (
                    f"frame {i} has empty mask")
                cx_png = c[0] * px_per_mask_x
                cy_png = c[1] * px_per_mask_y
                ex, ey = _square_center(i)
                assert abs(cx_png - ex) < tol_px, (
                    f"frame {i}: cx {cx_png:.1f} vs "
                    f"expected {ex:.1f}")
                assert abs(cy_png - ey) < tol_px, (
                    f"frame {i}: cy {cy_png:.1f} vs "
                    f"expected {ey:.1f}")


def test_video_session_invalid_direction(model_path, moving_square_dir):
    """propagate() rejects unknown direction strings."""
    with sam3.Model(model_path) as model:
        with sam3.VideoSession(model, str(moving_square_dir)) as sess:
            with pytest.raises(ValueError, match="direction"):
                list(sess.propagate(direction="sideways"))


def test_video_session_closed_raises(model_path, moving_square_dir):
    """Methods on a closed session raise ValueError."""
    with sam3.Model(model_path) as model:
        sess = sam3.VideoSession(model, str(moving_square_dir))
        sess.close()
        with pytest.raises(ValueError, match="closed"):
            sess.frame_count()


def test_video_session_empty_points(model_path, moving_square_dir):
    """add_points() rejects empty point lists."""
    with sam3.Model(model_path) as model:
        with sam3.VideoSession(model, str(moving_square_dir)) as sess:
            with pytest.raises(ValueError, match="non-empty"):
                sess.add_points(frame=0, obj_id=0, points=[])


def test_video_session_collect_only():
    """Smoke test: VideoSession is importable without a model.

    Runs unconditionally (does not depend on SAM3_MODEL_PATH), so it
    catches cdef / import regressions even on CI without a model.
    """
    assert hasattr(sam3, "VideoSession")
    assert callable(sam3.VideoSession)
