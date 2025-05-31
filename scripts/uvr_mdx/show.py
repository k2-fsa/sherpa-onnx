#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import onnxruntime
import onnx

"""
[]
NodeArg(name='input', type='tensor(float)', shape=['batch_size', 4, 3072, 256])
-----
NodeArg(name='output', type='tensor(float)', shape=['batch_size', 4, 3072, 256])
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
    #  show("./UVR-MDX-NET-Voc_FT.onnx")
    show("./UVR_MDXNET_1_9703.onnx")


if __name__ == "__main__":
    main()
