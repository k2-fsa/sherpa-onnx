#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import onnxruntime
import onnx

"""
[key: "model_type"
value: "silero-vad-v4"
, key: "sample_rate"
value: "16000"
, key: "version"
value: "4"
, key: "h_shape"
value: "2,1,64"
, key: "c_shape"
value: "2,1,64"
]
NodeArg(name='x', type='tensor(float)', shape=[1, 512])
NodeArg(name='h', type='tensor(float)', shape=[2, 1, 64])
NodeArg(name='c', type='tensor(float)', shape=[2, 1, 64])
-----
NodeArg(name='prob', type='tensor(float)', shape=[1, 1])
NodeArg(name='next_h', type='tensor(float)', shape=[2, 1, 64])
NodeArg(name='next_c', type='tensor(float)', shape=[2, 1, 64])
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
    show("./m.onnx")


if __name__ == "__main__":
    main()
