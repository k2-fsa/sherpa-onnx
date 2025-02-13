#!/usr/bin/env python3
from onnxruntime.quantization import QuantType, quantize_dynamic
from pathlib import Path


def main():
    suffix = "epoch-75-avg-11-chunk-16-left-128"

    for m in ["encoder", "joiner"]:
        if Path(f"{m}-{suffix}.int8.onnx").is_file():
            continue

        quantize_dynamic(
            model_input=f"./{m}-{suffix}.onnx",
            model_output=f"./{m}-{suffix}.int8.onnx",
            op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QInt8,
        )


if __name__ == "__main__":
    main()
