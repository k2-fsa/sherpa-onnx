#!/usr/bin/env python3
# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)

from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def main():
    for f in ["uncached_decode", "cached_decode", "encode", "preprocess"]:
        if Path("{f}.int8.onnx").is_file():
            continue

        print("processing", f)
        quantize_dynamic(
            model_input=f"{f}.onnx",
            model_output=f"{f}.int8.onnx",
            weight_type=QuantType.QInt8,
        )


if __name__ == "__main__":
    main()
