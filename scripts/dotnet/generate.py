#!/usr/bin/env python3
# Copyright (c)  2023  Xiaomi Corporation

import argparse
import re
from pathlib import Path

import jinja2

SHERPA_ONNX_DIR = Path(__file__).resolve().parent.parent.parent


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


def process_linux(s):
    libs = [
        "libkaldi-native-fbank-core.so",
        "libonnxruntime.so.1.15.1",
        "libsherpa-onnx-c-api.so",
        "libsherpa-onnx-core.so",
    ]
    prefix = f"{SHERPA_ONNX_DIR}/linux/sherpa_onnx/lib/"
    libs = [prefix + lib for lib in libs]
    libs = "\n      ;".join(libs)

    d = get_dict()
    d["dotnet_rid"] = "linux-x64"
    d["libs"] = libs

    environment = jinja2.Environment()
    template = environment.from_string(s)
    s = template.render(**d)
    with open("./linux/sherpa-onnx.runtime.csproj", "w") as f:
        f.write(s)


def process_macos(s):
    libs = [
        "libkaldi-native-fbank-core.dylib",
        "libonnxruntime.1.15.1.dylib",
        "libsherpa-onnx-c-api.dylib",
        "libsherpa-onnx-core.dylib",
    ]
    prefix = f"{SHERPA_ONNX_DIR}/macos/sherpa_onnx/lib/"
    libs = [prefix + lib for lib in libs]
    libs = "\n      ;".join(libs)

    d = get_dict()
    d["dotnet_rid"] = "osx-x64"
    d["libs"] = libs

    environment = jinja2.Environment()
    template = environment.from_string(s)
    s = template.render(**d)
    with open("./macos/sherpa-onnx.runtime.csproj", "w") as f:
        f.write(s)


def process_windows(s):
    libs = [
        "kaldi-native-fbank-core.dll",
        "onnxruntime.dll",
        "sherpa-onnx-c-api.dll",
        "sherpa-onnx-core.dll",
    ]

    version = get_version()

    prefix1 = f"{SHERPA_ONNX_DIR}/windows/sherpa_onnx/lib/"
    prefix2 = f"{SHERPA_ONNX_DIR}/windows/sherpa_onnx/"
    prefix3 = f"{SHERPA_ONNX_DIR}/windows/"
    prefix4 = f"{SHERPA_ONNX_DIR}/windows/sherpa_onnx-{version}.data/data/bin/"
    print(prefix1, prefix2, prefix3, prefix4)

    lib_list = []
    for lib in libs:
        for prefix in [prefix1, prefix2, prefix3, prefix4]:
            f = Path(prefix) / lib
            if f.is_file():
                lib_list.append(str(f))
                break

    print("lib_list", lib_list)
    libs = "\n      ;".join(lib_list)

    d = get_dict()
    d["dotnet_rid"] = "win-x64"
    d["libs"] = libs

    environment = jinja2.Environment()
    template = environment.from_string(s)
    s = template.render(**d)
    with open("./windows/sherpa-onnx.runtime.csproj", "w") as f:
        f.write(s)


def main():
    s = read_proj_file("./sherpa-onnx.csproj.runtime.in")
    process_macos(s)
    process_linux(s)
    process_windows(s)

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
