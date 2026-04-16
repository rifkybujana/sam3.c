"""Unit tests for error handling and basic API (no model needed)."""
import gc
import pytest
import sam3


def test_version_returns_string():
    v = sam3.version()
    assert isinstance(v, str)
    assert len(v) > 0


def test_set_log_level_valid():
    for level in ("debug", "info", "warn", "error"):
        sam3.set_log_level(level)


def test_set_log_level_invalid():
    with pytest.raises(KeyError):
        sam3.set_log_level("invalid")


def test_model_invalid_path():
    with pytest.raises(sam3.Sam3Error):
        sam3.Model("/nonexistent/model.sam3")


def test_model_init_failure_no_del_crash():
    """__del__ should not crash when __init__ fails."""
    try:
        sam3.Model("/nonexistent")
    except sam3.Sam3Error:
        pass
    gc.collect()  # Force __del__ to run


def test_error_hierarchy():
    assert issubclass(sam3.InvalidArgumentError, sam3.Sam3Error)
    assert issubclass(sam3.OutOfMemoryError, sam3.Sam3Error)
    assert issubclass(sam3.Sam3IOError, sam3.Sam3Error)
    assert issubclass(sam3.BackendError, sam3.Sam3Error)
    assert issubclass(sam3.ModelError, sam3.Sam3Error)
    assert issubclass(sam3.DtypeError, sam3.Sam3Error)


def test_invalid_path_raises_io_error():
    with pytest.raises(sam3.Sam3IOError):
        sam3.Model("/nonexistent/model.sam3")
