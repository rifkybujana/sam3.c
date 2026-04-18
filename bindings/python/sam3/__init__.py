"""SAM3 - Python bindings for the sam3 inference engine."""
from sam3._lib import ffi, lib  # noqa: F401
import sam3._ffi  # noqa: F401

from sam3.errors import (  # noqa: F401
    Sam3Error, InvalidArgumentError, OutOfMemoryError,
    Sam3IOError, BackendError, ModelError, DtypeError, VideoError,
)
from sam3.model import Model, Result  # noqa: F401
from sam3.video import (  # noqa: F401
    VideoSession, FrameResult, ObjectMask, StartOpts,
)

__all__ = [
    "Model", "Result", "VideoSession", "FrameResult",
    "ObjectMask", "StartOpts", "Sam3Error",
    "InvalidArgumentError", "OutOfMemoryError", "Sam3IOError",
    "BackendError", "ModelError", "DtypeError", "VideoError",
    "version", "set_log_level",
]


def version():
    """Return the sam3 library version string."""
    return ffi.string(lib.sam3_version()).decode()


_LOG_LEVELS = {
    "debug": 0,  # SAM3_LOG_DEBUG
    "info": 1,   # SAM3_LOG_INFO
    "warn": 2,   # SAM3_LOG_WARN
    "error": 3,  # SAM3_LOG_ERROR
}


def set_log_level(level):
    """Set the minimum log level ('debug', 'info', 'warn', 'error')."""
    if isinstance(level, str):
        level = _LOG_LEVELS[level.lower()]
    lib.sam3_log_set_level(level)
