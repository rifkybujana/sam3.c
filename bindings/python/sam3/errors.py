"""Exception hierarchy mapping sam3 C error codes to Python exceptions."""
from sam3._lib import ffi, lib


class Sam3Error(Exception):
    """Base exception for all sam3 errors."""

    def __init__(self, code, message=None):
        self.code = code
        if message is None:
            message = ffi.string(lib.sam3_error_str(code)).decode()
        super().__init__(message)


class InvalidArgumentError(Sam3Error):
    """SAM3_EINVAL: Invalid argument."""
    pass


class OutOfMemoryError(Sam3Error):
    """SAM3_ENOMEM: Out of memory."""
    pass


class Sam3IOError(Sam3Error):
    """SAM3_EIO: I/O error."""
    pass


class BackendError(Sam3Error):
    """SAM3_EBACKEND: Backend initialization failed."""
    pass


class ModelError(Sam3Error):
    """SAM3_EMODEL: Model format error."""
    pass


class DtypeError(Sam3Error):
    """SAM3_EDTYPE: Unsupported or mismatched dtype."""
    pass


class VideoError(Sam3Error):
    """SAM3_EVIDEO: Video tracking error."""
    pass


_ERROR_MAP = {
    -1: InvalidArgumentError,
    -2: OutOfMemoryError,
    -3: Sam3IOError,
    -4: BackendError,
    -5: ModelError,
    -6: DtypeError,
    -7: VideoError,
}


def check(code):
    """Check a sam3 return code; raise appropriate exception if non-zero."""
    if code != 0:
        cls = _ERROR_MAP.get(code, Sam3Error)
        raise cls(code)
