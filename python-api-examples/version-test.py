#!/usr/bin/env python3

# Copyright (c)  2026  Xiaomi Corporation

import sherpa_onnx


def main():
    print(f"sherpa-onnx version: {sherpa_onnx.version}")
    print(f"sherpa-onnx git sha1: {sherpa_onnx.git_sha1}")
    print(f"sherpa-onnx git date: {sherpa_onnx.git_date}")
    print(f"onnxruntime version: {sherpa_onnx.onnxruntime_version}")


if __name__ == "__main__":
    main()
