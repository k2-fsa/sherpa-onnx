#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

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

import onnx
import onnxruntime as ort


def show(filename):
    session_opts = ort.SessionOptions()
    session_opts.log_severity_level = 3
    sess = ort.InferenceSession(filename, session_opts)
    for i in sess.get_inputs():
        print(i)

    print("-----")

    for i in sess.get_outputs():
        print(i)


def main():
    filename = "./gtcrn_simple.onnx"
    show(filename)
    model = onnx.load(filename)

    meta_data = {
        "model_type": "gtcrn",
        "comment": "gtcrn_simple",
        "version": 1,
        "sample_rate": 16000,
        "model_url": "https://github.com/Xiaobin-Rong/gtcrn/blob/main/stream/onnx_models/gtcrn_simple.onnx",
        "maintainer": "k2-fsa",
        "comment2": "Please see also https://github.com/Xiaobin-Rong/gtcrn",
        "conv_cache_shape": "2,1,16,16,33",
        "tra_cache_shape": "2,3,1,1,16",
        "inter_cache_shape": "2,1,33,16",
        "n_fft": 512,
        "hop_length": 256,
        "window_length": 512,
        "window_type": "hann_sqrt",
    }

    print(model.metadata_props)

    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    print("--------------------")

    print(model.metadata_props)

    onnx.save(model, filename)


if __name__ == "__main__":
    main()
