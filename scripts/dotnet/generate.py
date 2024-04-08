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
        "libespeak-ng.so",
        "libkaldi-decoder-core.so",
        "libkaldi-native-fbank-core.so",
        "libonnxruntime.so.1.17.1",
        "libpiper_phonemize.so.1",
        "libsherpa-onnx-c-api.so",
        "libsherpa-onnx-core.so",
        "libsherpa-onnx-fstfar.so.16",
        "libsherpa-onnx-fst.so.16",
        "libsherpa-onnx-kaldifst-core.so",
        "libucd.so",
    ]
    prefix = "/tmp/linux/"
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
        "libespeak-ng.dylib",
        "libkaldi-decoder-core.dylib",
        "libkaldi-native-fbank-core.dylib",
        "libonnxruntime.1.17.1.dylib",
        "libpiper_phonemize.1.dylib",
        "libsherpa-onnx-c-api.dylib",
        "libsherpa-onnx-core.dylib",
        "libsherpa-onnx-fstfar.16.dylib",
        "libsherpa-onnx-fst.16.dylib",
        "libsherpa-onnx-kaldifst-core.dylib",
        "libucd.dylib",
    ]
    prefix = f"/tmp/macos/"
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


def process_windows(s, rid):
    libs = [
        "espeak-ng.dll",
        "kaldi-decoder-core.dll",
        "kaldi-native-fbank-core.dll",
        "onnxruntime.dll",
        "piper_phonemize.dll",
        "sherpa-onnx-c-api.dll",
        "sherpa-onnx-core.dll",
        "sherpa-onnx-fstfar.lib",
        "sherpa-onnx-fst.lib",
        "sherpa-onnx-kaldifst-core.lib",
        "ucd.dll",
    ]

    version = get_version()

    prefix = f"/tmp/windows-{rid}/"
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
    process_macos(s)
    process_linux(s)
    process_windows(s, "x64")
    process_windows(s, "x86")

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
