#!/usr/bin/env python3

import argparse
import os
import re

import jinja2


def parse_args():
    # set the source code file
    # -s sherpa_onnx.go
    # set the output folder
    # -o ./sherpa-onnx-go
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # add argv to set source code file
    parser.add_argument("-s", "--source", type=str, required=True)
    # add argv to set output folder
    parser.add_argument("-o", "--output", type=str, required=True)
    return parser.parse_args()


def parse_golang(target):
    with open(target, "r") as file:
        content = file.read()
    defines = []
    struct_pattern = re.compile(r"type\s+([A-Z]\w+)\s+struct", re.DOTALL)
    struct_matches = struct_pattern.findall(content)
    for name in struct_matches:
        c_define = {
            "type": "struct",
            "name": name,
        }
        defines.append(c_define)
    struct_pattern = re.compile(r"type\s+([A-Z][^ =]+)\s*=", re.DOTALL)
    struct_matches = struct_pattern.findall(content)
    for name in struct_matches:
        c_define = {
            "type": "struct",
            "name": name,
        }
        defines.append(c_define)
    func_pattern = re.compile(r"func\s+([A-Z][^ \(]+)\s*\(", re.DOTALL)
    func_matches = func_pattern.findall(content)
    for name in func_matches:
        c_define = {
            "type": "function",
            "name": name,
        }
        defines.append(c_define)
    return defines


def render(output, defines, platform):
    build_info = ""
    if platform == "windows":
        build_info = "//go:build (windows && amd64) || (windows && 386)"
    elif platform == "linux":
        build_info = "//go:build (!android && linux && arm64) || (!android && linux && amd64 && !musl) || (!android && linux && arm && !arm7) || (!android && arm7) || (!android && linux && 386 && !musl) || (!android && musl) || (!android && linux && mips) || (!android && linux && mips64) || (!android && linux && mips64le) || (!android && linux && mipsle)"
    elif platform == "macos":
        build_info = "//go:build (darwin && amd64 && !ios) || (darwin && arm64 && !ios)"
    with open("./defines.go.jinja") as f:
        content = f.read()
    environment = jinja2.Environment()
    template = environment.from_string(content)
    context = {
        "platform": platform,
        "defines": defines,
        "golang_header": build_info,
    }
    rendered = template.render(**context)
    folder = os.path.dirname(output)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(output, "w") as f:
        print(rendered, file=f)


def generate(src, output):
    defines = parse_golang(src)
    platform = "linux"
    render(f"{output}/sherpa_onnx/sherpa_onnx_{platform}.go", defines, platform)
    platform = "windows"
    render(f"{output}/sherpa_onnx/sherpa_onnx_{platform}.go", defines, platform)
    platform = "macos"
    render(f"{output}/sherpa_onnx/sherpa_onnx_{platform}.go", defines, platform)


if __name__ == "__main__":
    args = parse_args()
    generate(args.source, args.output)
