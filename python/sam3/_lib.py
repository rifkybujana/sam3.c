"""Locate and load the bundled libsam3 shared library."""
import os
import sys

import cffi

ffi = cffi.FFI()


def _find_library():
    pkg_dir = os.path.dirname(os.path.abspath(__file__))

    if sys.platform == "darwin":
        name = "libsam3.dylib"
    elif sys.platform == "win32":
        name = "sam3.dll"
    else:
        name = "libsam3.so"

    path = os.path.join(pkg_dir, name)
    if not os.path.isfile(path):
        raise OSError(
            f"libsam3 not found at {path}. "
            "Reinstall the sam3 package or build with SAM3_SHARED=ON."
        )
    return path


lib = ffi.dlopen(_find_library())
