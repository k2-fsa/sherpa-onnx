#!/usr/bin/env python3
# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)

import onnxruntime
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="Path to model.onnx",
    )

    return parser.parse_args()


def show(filename):
    session_opts = onnxruntime.SessionOptions()
    session_opts.log_severity_level = 3
    sess = onnxruntime.InferenceSession(filename, session_opts)
    for i in sess.get_inputs():
        print(i)

    print("-----")

    for i in sess.get_outputs():
        print(i)


def main():
    args = get_args()
    print(f"========={args.filename}==========")
    show(args.filename)


if __name__ == "__main__":
    main()
