#!/usr/bin/env python3
# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)

import onnxruntime


def show(filename):
    session_opts = onnxruntime.SessionOptions()
    session_opts.log_severity_level = 3
    sess = onnxruntime.InferenceSession(filename, session_opts)
    for i in sess.get_inputs():
        print(i)

    print("-----")

    for i in sess.get_outputs():
        print(i)

    meta = sess.get_modelmeta().custom_metadata_map
    print("*****************************************")
    print("meta\n", meta)


def main():
    print("=========model==========")
    show("./model.onnx")


if __name__ == "__main__":
    main()
"""
=========model==========
NodeArg(name='x', type='tensor(float)', shape=['N', 'T', 560])
NodeArg(name='x_length', type='tensor(int32)', shape=['N'])
NodeArg(name='language', type='tensor(int32)', shape=['N'])
NodeArg(name='text_norm', type='tensor(int32)', shape=['N'])
-----
NodeArg(name='logits', type='tensor(float)', shape=['N', 'T', 25055])
*****************************************
"""
