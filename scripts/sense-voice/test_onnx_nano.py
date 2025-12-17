#!/usr/bin/env python3
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

"""
=========./model.onnx==========
NodeArg(name='x', type='tensor(float)', shape=[1, 'T', 560])
-----
NodeArg(name='logits', type='tensor(float)', shape=['Addlogits_dim_0', 'Addlogits_dim_1', 60515])

=========./model.int8.onnx==========
NodeArg(name='x', type='tensor(float)', shape=[1, 'T', 560])
-----
NodeArg(name='logits', type='tensor(float)', shape=['Addlogits_dim_0', 'Addlogits_dim_1', 60515])
"""

import argparse
import base64
from typing import Tuple

from test_onnx import compute_feat, load_audio

import onnxruntime as ort
import librosa


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model.onnx",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--wave",
        type=str,
        required=True,
        help="The input wave to be recognized",
    )

    return parser.parse_args()


class OnnxModel:
    def __init__(self, filename):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.session_opts = session_opts

        self.model = ort.InferenceSession(
            filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        meta = self.model.get_modelmeta().custom_metadata_map

        self.window_size = int(meta["lfr_window_size"])  # lfr_m
        self.window_shift = int(meta["lfr_window_shift"])  # lfr_n
        self.blank_id = int(meta["blank_id"])

    def __call__(self, x):
        logits = self.model.run(
            [
                self.model.get_outputs()[0].name,
            ],
            {
                self.model.get_inputs()[0].name: x,
            },
        )[0]

        return logits


def load_tokens(filename: str):
    ans = dict()
    i = 0
    with open(filename, encoding="utf-8") as f:
        for line in f:
            ans[i] = line.strip().split()[0]
            i += 1
    return ans


def main():
    args = get_args()
    print(vars(args))
    samples, sample_rate = load_audio(args.wave)
    if sample_rate != 16000:
        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    model = OnnxModel(filename=args.model)

    features = compute_feat(
        samples=samples,
        sample_rate=sample_rate,
        window_size=model.window_size,
        window_shift=model.window_shift,
    )

    logits = model(
        x=features[None],
    )

    idx = logits[0].argmax(axis=-1)
    print("initial ids", idx)
    id2token = load_tokens(args.tokens)
    blank_id = model.blank_id
    print("blank_id", blank_id)

    unique_ids = []
    prev = -1
    for i in idx:
        if i == prev:
            continue
        unique_ids.append(i)
        prev = i
    print("unique_ids", unique_ids)

    ids = [i for i in unique_ids if i != blank_id]

    print("ids without blank", ids)
    s = b""
    for i in ids:
        s += base64.b64decode(id2token[i])

    text = s.decode().strip()
    print(text)


if __name__ == "__main__":
    main()
