#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import onnxruntime
import onnx

"""
[key: "onnx.infer"
value: "onnxruntime.quant"
, key: "onnx.quant.pre_process"
value: "onnxruntime.quant"
]
NodeArg(name='input_ids', type='tensor(int64)', shape=[1, 'sequence_length'])
NodeArg(name='style', type='tensor(float)', shape=[1, 256])
NodeArg(name='speed', type='tensor(float)', shape=[1])
-----
NodeArg(name='waveform', type='tensor(float)', shape=['num_samples'])
NodeArg(name='duration', type='tensor(int64)', shape=['Castduration_dim_0'])
"""


def show(filename):
    model = onnx.load(filename)
    print(model.metadata_props)

    session_opts = onnxruntime.SessionOptions()
    session_opts.log_severity_level = 3
    sess = onnxruntime.InferenceSession(
        filename, session_opts, providers=["CPUExecutionProvider"]
    )
    for i in sess.get_inputs():
        print(i)

    print("-----")

    for i in sess.get_outputs():
        print(i)


def main():
    show("./model.fp16.onnx")


if __name__ == "__main__":
    main()
