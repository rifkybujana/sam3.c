import os
import pytest


@pytest.fixture
def model_path():
    """Path to test model. Set SAM3_MODEL_PATH env var or skip."""
    path = os.environ.get("SAM3_MODEL_PATH")
    if not path or not os.path.isfile(path):
        pytest.skip("SAM3_MODEL_PATH not set or file not found")
    return path


@pytest.fixture
def test_image_path():
    """Path to test image. Set SAM3_TEST_IMAGE env var or skip."""
    path = os.environ.get("SAM3_TEST_IMAGE")
    if not path or not os.path.isfile(path):
        pytest.skip("SAM3_TEST_IMAGE not set or file not found")
    return path
