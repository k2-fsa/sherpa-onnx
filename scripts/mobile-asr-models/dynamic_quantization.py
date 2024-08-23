#!/usr/bin/env python3
import argparse

import onnxruntime
from onnxruntime.quantization import QuantType, quantize_dynamic


def show(filename):
    session_opts = onnxruntime.SessionOptions()
    session_opts.log_severity_level = 3
    sess = onnxruntime.InferenceSession(filename, session_opts)
    for i in sess.get_inputs():
        print(i)

    print("-----")

    for i in sess.get_outputs():
        print(i)


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
    print(f"----------{args.input}----------")
    show(args.input)
    print("------------------------------")

    quantize_dynamic(
        model_input=args.input,
        model_output=args.output,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )


if __name__ == "__main__":
    main()
