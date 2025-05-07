#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import onnxruntime
import onnx

"""
[key: "model_type"
value: "gtcrn"
, key: "comment"
value: "gtcrn_simple"
, key: "version"
value: "1"
, key: "sample_rate"
value: "16000"
, key: "model_url"
value: "https://github.com/Xiaobin-Rong/gtcrn/blob/main/stream/onnx_models/gtcrn_simple.onnx"
, key: "maintainer"
value: "k2-fsa"
, key: "comment2"
value: "Please see also https://github.com/Xiaobin-Rong/gtcrn"
, key: "conv_cache_shape"
value: "2,1,16,16,33"
, key: "tra_cache_shape"
value: "2,3,1,1,16"
, key: "inter_cache_shape"
value: "2,1,33,16"
, key: "n_fft"
value: "512"
, key: "hop_length"
value: "256"
, key: "window_length"
value: "512"
, key: "window_type"
value: "hann_sqrt"
]
"""

"""
NodeArg(name='mix', type='tensor(float)', shape=[1, 257, 1, 2])
NodeArg(name='conv_cache', type='tensor(float)', shape=[2, 1, 16, 16, 33])
NodeArg(name='tra_cache', type='tensor(float)', shape=[2, 3, 1, 1, 16])
NodeArg(name='inter_cache', type='tensor(float)', shape=[2, 1, 33, 16])
-----
NodeArg(name='enh', type='tensor(float)', shape=[1, 257, 1, 2])
NodeArg(name='conv_cache_out', type='tensor(float)', shape=[2, 1, 16, 16, 33])
NodeArg(name='tra_cache_out', type='tensor(float)', shape=[2, 3, 1, 1, 16])
NodeArg(name='inter_cache_out', type='tensor(float)', shape=[2, 1, 33, 16])
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
    show("./gtcrn_simple.onnx")


if __name__ == "__main__":
    main()
