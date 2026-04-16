"""
Build script that compiles libsam3 as a shared library and bundles it
into the Python package.
"""
import os
import shutil
import subprocess
import sys

from setuptools import setup
from setuptools.command.build_ext import build_ext


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam3")


class CMakeBuild(build_ext):
    def run(self):
        build_dir = os.path.join(ROOT, "build-python")
        os.makedirs(build_dir, exist_ok=True)

        cmake_args = [
            "cmake", ROOT,
            "-DSAM3_SHARED=ON",
            "-DSAM3_TESTS=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        subprocess.check_call(cmake_args, cwd=build_dir)
        subprocess.check_call(
            ["cmake", "--build", ".", "--parallel"],
            cwd=build_dir,
        )

        if sys.platform == "darwin":
            lib_name = "libsam3.dylib"
        else:
            lib_name = "libsam3.so"

        src = os.path.join(build_dir, lib_name)
        dst = os.path.join(PKG_DIR, lib_name)
        shutil.copy2(src, dst)


setup(
    cmdclass={"build_ext": CMakeBuild},
    packages=["sam3"],
    package_dir={"sam3": "sam3"},
    has_ext_modules=lambda: True,
)
