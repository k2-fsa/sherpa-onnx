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


"""
NodeArg(name='tokens', type='tensor(int64)', shape=[1, 'sequence_length'])
NodeArg(name='style', type='tensor(float)', shape=[1, 256])
NodeArg(name='speed', type='tensor(float)', shape=[1])
-----
NodeArg(name='audio', type='tensor(float)', shape=['audio_length'])
"""


def main():
    show("./kokoro.onnx")

    quantize_dynamic(
        model_input="kokoro.onnx",
        model_output="kokoro.int8.onnx",
        #  op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QUInt8,
    )


if __name__ == "__main__":
    main()
