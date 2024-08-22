#!/usr/bin/env python3
import argparse

from onnxruntime.quantization import QuantType, quantize_dynamic


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input onnx model",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output onnx model",
    )
    return parser.parse_args()


def main():
    args = get_args()
    print(vars(args))

    quantize_dynamic(
        model_input=args.input,
        model_output=args.output,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )


if __name__ == "__main__":
    main()
