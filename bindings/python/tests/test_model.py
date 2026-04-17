"""Integration tests requiring a real model and image."""
import numpy as np
import pytest
import sam3


class TestModelSegment:
    def test_set_image_file(self, model_path, test_image_path):
        with sam3.Model(model_path) as m:
            m.set_image(test_image_path)
            assert m.get_image_size() > 0

    def test_set_image_array(self, model_path):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        with sam3.Model(model_path) as m:
            m.set_image(img)

    def test_segment_point(self, model_path, test_image_path):
        with sam3.Model(model_path) as m:
            m.set_image(test_image_path)
            result = m.segment(points=[(100, 100, 1)])
            assert result.masks.ndim == 3
            assert result.masks.shape[0] > 0
            assert result.iou_scores.shape[0] == result.masks.shape[0]

    def test_segment_box(self, model_path, test_image_path):
        with sam3.Model(model_path) as m:
            m.set_image(test_image_path)
            result = m.segment(boxes=[(50, 50, 200, 200)])
            assert result.masks.ndim == 3

    def test_segment_no_prompts(self, model_path, test_image_path):
        with sam3.Model(model_path) as m:
            m.set_image(test_image_path)
            with pytest.raises(ValueError, match="At least one prompt"):
                m.segment()

    def test_invalid_image_shape(self, model_path):
        with sam3.Model(model_path) as m:
            with pytest.raises(ValueError, match="Expected.*RGB"):
                m.set_image(np.zeros((256, 256), dtype=np.uint8))

    def test_closed_model_raises(self, model_path):
        m = sam3.Model(model_path)
        m.close()
        with pytest.raises(ValueError, match="closed"):
            m.set_image("test.jpg")

    def test_context_manager(self, model_path):
        with sam3.Model(model_path) as m:
            assert m.get_image_size() > 0
        with pytest.raises(ValueError, match="closed"):
            m.get_image_size()
