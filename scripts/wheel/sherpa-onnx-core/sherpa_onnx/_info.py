from pathlib import Path
from typing import List

_pkg_dir = Path(__file__).parent
libs_dir = _pkg_dir / "lib"
include_dir = _pkg_dir / "include"

# List of libraries (without "lib" prefix, without extension)
# Adjust to match your actual .so/.dll/.dylib files
onnxruntime_lib = ["onnxruntime"]
c_lib = ["sherpa-onnx-c-api"] + onnxruntime_lib
cxx_lib = ["sherpa-onnx-cxx-api"] + c_lib


def get_include_dir() -> str:
    return str(include_dir)


def get_libs_dir() -> str:
    return str(libs_dir)


def get_c_api_libs() -> List[str]:
    return c_lib


def get_cxx_api_libs() -> List[str]:
    return cxx_lib
