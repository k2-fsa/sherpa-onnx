#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)


import numpy as np

try:
    from piper_phonemize import phonemize_espeak
except Exception as ex:
    raise RuntimeError(
        f"{ex}\nPlease run\n"
        "pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html"
    )

import onnxruntime as ort
import soundfile as sf


def show(filename):
    session_opts = ort.SessionOptions()
    session_opts.log_severity_level = 3
    sess = ort.InferenceSession(filename, session_opts)
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


if __name__ == "__main__":
    main()
