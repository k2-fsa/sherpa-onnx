#!/usr/bin/env python3
# Copyright (c)  2023  Xiaomi Corporation

import argparse
import os
import re
from pathlib import Path

import jinja2

SHERPA_ONNX_DIR = Path(__file__).resolve().parent.parent.parent

src_dir = os.environ.get("src_dir", "/tmp")


def get_version():
    cmake_file = SHERPA_ONNX_DIR / "CMakeLists.txt"
    with open(cmake_file) as f:
        content = f.read()

    version = re.search(r"set\(SHERPA_ONNX_VERSION (.*)\)", content).group(1)
    return version.strip('"')


def read_proj_file(filename):
    with open(filename) as f:
        return f.read()


def get_dict():
    version = get_version()
    return {
        "version": get_version(),
    }


def process_linux(s, rid):
    libs = [
        "libonnxruntime.so.1.17.1",
        "libsherpa-onnx-c-api.so",
    ]
    prefix = f"{src_dir}/linux-{rid}/"
    libs = [prefix + lib for lib in libs]
    libs = "\n      ;".join(libs)

    d = get_dict()
    d["dotnet_rid"] = f"linux-{rid}"
    d["libs"] = libs

    environment = jinja2.Environment()
    template = environment.from_string(s)
    s = template.render(**d)
    with open(f"./linux-{rid}/sherpa-onnx.runtime.csproj", "w") as f:
        f.write(s)


def process_macos(s, rid):
    libs = [
        "libonnxruntime.1.17.1.dylib",
        "libsherpa-onnx-c-api.dylib",
    ]
    prefix = f"{src_dir}/macos-{rid}/"
    libs = [prefix + lib for lib in libs]
    libs = "\n      ;".join(libs)

    d = get_dict()
    d["dotnet_rid"] = f"osx-{rid}"
    d["libs"] = libs

    environment = jinja2.Environment()
    template = environment.from_string(s)
    s = template.render(**d)
    with open(f"./macos-{rid}/sherpa-onnx.runtime.csproj", "w") as f:
        f.write(s)


def process_windows(s, rid):
    libs = [
        "onnxruntime.dll",
        "sherpa-onnx-c-api.dll",
    ]

    version = get_version()

    prefix = f"{src_dir}/windows-{rid}/"
    libs = [prefix + lib for lib in libs]
    libs = "\n      ;".join(libs)

    d = get_dict()
    d["dotnet_rid"] = f"win-{rid}"
    d["libs"] = libs

    environment = jinja2.Environment()
    template = environment.from_string(s)
    s = template.render(**d)
    with open(f"./windows-{rid}/sherpa-onnx.runtime.csproj", "w") as f:
        f.write(s)


def main():
    s = read_proj_file("./sherpa-onnx.csproj.runtime.in")
    process_macos(s, "x64")
    process_macos(s, "arm64")
    process_linux(s, "x64")
    process_linux(s, "arm64")
    process_windows(s, "x64")
    process_windows(s, "x86")
    process_windows(s, "arm64")

    s = read_proj_file("./sherpa-onnx.csproj.in")
    d = get_dict()
    d["packages_dir"] = str(SHERPA_ONNX_DIR / "scripts/dotnet/packages")

    environment = jinja2.Environment()
    template = environment.from_string(s)
    s = template.render(**d)
    with open("./all/sherpa-onnx.csproj", "w") as f:
        f.write(s)


if __name__ == "__main__":
    main()
