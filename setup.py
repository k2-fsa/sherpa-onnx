#!/usr/bin/env python3

import os
import re
from pathlib import Path

import setuptools

from cmake.cmake_extension import (
    BuildExtension,
    bdist_wheel,
    cmake_extension,
    get_binaries,
    is_windows,
    need_split_package,
)


def read_long_description():
    with open("README.md", encoding="utf8") as f:
        readme = f.read()
    return readme


def get_package_version():
    with open("CMakeLists.txt") as f:
        content = f.read()

    match = re.search(r"set\(SHERPA_ONNX_VERSION (.*)\)", content)
    latest_version = match.group(1).strip('"')

    cmake_args = os.environ.get("SHERPA_ONNX_CMAKE_ARGS", "")
    extra_version = ""
    if "-DSHERPA_ONNX_ENABLE_GPU=ON" in cmake_args:
        extra_version = "+cuda"

    cuda_version = os.environ.get("SHERPA_ONNX_CUDA_VERSION", "")
    if cuda_version:
        extra_version += cuda_version

    latest_version += extra_version

    return latest_version


package_name = "sherpa-onnx"

with open("sherpa-onnx/python/sherpa_onnx/__init__.py", "a") as f:
    f.write(f"__version__ = '{get_package_version()}'\n")


def get_binaries_to_install():
    if need_split_package():
        return None

    cmake_args = os.environ.get("SHERPA_ONNX_CMAKE_ARGS", "")
    if "-DSHERPA_ONNX_ENABLE_BINARY=OFF" in cmake_args:
        return None

    bin_dir = Path("build") / "sherpa_onnx" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".exe" if is_windows() else ""

    binaries = get_binaries()

    exe = []
    for f in binaries:
        suffix = "" if (".dll" in f or ".lib" in f) else suffix
        t = bin_dir / (f + suffix)
        exe.append(str(t))
    return exe


setuptools.setup(
    name=package_name,
    python_requires=">=3.7",
    version=get_package_version(),
    author="The sherpa-onnx development team",
    author_email="dpovey@gmail.com",
    package_dir={
        "sherpa_onnx": "sherpa-onnx/python/sherpa_onnx",
    },
    packages=["sherpa_onnx"],
    data_files=[
        ("Scripts", get_binaries_to_install())
        if is_windows()
        else ("bin", get_binaries_to_install())
    ]
    if get_binaries_to_install()
    else None,
    url="https://github.com/k2-fsa/sherpa-onnx",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    ext_modules=[cmake_extension("_sherpa_onnx")],
    cmdclass={"build_ext": BuildExtension, "bdist_wheel": bdist_wheel},
    zip_safe=False,
    classifiers=[
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "sherpa-onnx-cli=sherpa_onnx.cli:cli",
        ],
    },
    license="Apache licensed, as found in the LICENSE file",
    install_requires=["sherpa-onnx-core==1.12.15"] if need_split_package() else None,
)

with open("sherpa-onnx/python/sherpa_onnx/__init__.py", "r") as f:
    lines = f.readlines()

with open("sherpa-onnx/python/sherpa_onnx/__init__.py", "w") as f:
    for line in lines:
        if "__version__" in line:
            # skip __version__ = "x.x.x"
            continue
        f.write(line)
