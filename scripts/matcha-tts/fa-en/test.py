#!/usr/bin/env python3

"""
AM
NodeArg(name='x', type='tensor(int64)', shape=['batch_size', 'time'])
NodeArg(name='x_lengths', type='tensor(int64)', shape=['batch_size'])
NodeArg(name='scales', type='tensor(float)', shape=[2])
-----
NodeArg(name='mel', type='tensor(float)', shape=['batch_size', 80, 'time'])
NodeArg(name='mel_lengths', type='tensor(int64)', shape=['batch_size'])

Vocoder
NodeArg(name='mel', type='tensor(float)', shape=['N', 80, 'L'])
-----
NodeArg(name='audio', type='tensor(float)', shape=['N', 'L'])
"""

import argparse

import numpy as np
import onnxruntime as ort
import soundfile as sf

try:
    from piper_phonemize import phonemize_espeak
except Exception as ex:
    raise RuntimeError(
        f"{ex}\nPlease run\n"
        "pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html"
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--am", type=str, required=True, help="Path to the acoustic model"
    )

    parser.add_argument(
        "--vocoder", type=str, required=True, help="Path to the vocoder"
    )
    parser.add_argument(
        "--tokens", type=str, required=True, help="Path to the tokens.txt"
    )

    parser.add_argument(
        "--text", type=str, required=True, help="Path to the text for generation"
    )

    parser.add_argument(
        "--out-wav", type=str, required=True, help="Path to save the generated wav"
    )
    return parser.parse_args()


def load_tokens(filename: str):
    ans = dict()
    with open(filename, encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) == 1:
                ans[" "] = int(fields[0])
            else:
                assert len(fields) == 2, (line, fields)
                ans[fields[0]] = int(fields[1])
    return ans


class OnnxHifiGANModel:
    def __init__(
        self,
        filename: str,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.session_opts = session_opts
        self.model = ort.InferenceSession(
            filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        for i in self.model.get_inputs():
            print(i)

        print("-----")

        for i in self.model.get_outputs():
            print(i)

    def __call__(self, x: np.ndarray):
        assert x.ndim == 3, x.shape
        assert x.shape[0] == 1, x.shape

        audio = self.model.run(
            [self.model.get_outputs()[0].name],
            {
                self.model.get_inputs()[0].name: x,
            },
        )[0]
        # audio: (batch_size, num_samples)

        return audio


class OnnxModel:
    def __init__(
        self,
        filename: str,
        tokens: str,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 2

        self.session_opts = session_opts
        self.token2id = load_tokens(tokens)
        self.model = ort.InferenceSession(
            filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        print(f"{self.model.get_modelmeta().custom_metadata_map}")
        metadata = self.model.get_modelmeta().custom_metadata_map
        self.sample_rate = int(metadata["sample_rate"])

        for i in self.model.get_inputs():
            print(i)

        print("-----")

        for i in self.model.get_outputs():
            print(i)

    def __call__(self, x: np.ndarray):
        assert x.ndim == 2, x.shape
        assert x.shape[0] == 1, x.shape

        x_lengths = np.array([x.shape[1]], dtype=np.int64)

        noise_scale = 1.0
        length_scale = 1.0
        scales = np.array([noise_scale, length_scale], dtype=np.float32)

        mel = self.model.run(
            [self.model.get_outputs()[0].name],
            {
                self.model.get_inputs()[0].name: x,
                self.model.get_inputs()[1].name: x_lengths,
                self.model.get_inputs()[2].name: scales,
            },
        )[0]
        # mel: (batch_size, feat_dim, num_frames)

        return mel


def main():
    args = get_args()
    print(vars(args))
    am = OnnxModel(args.am, args.tokens)
    vocoder = OnnxHifiGANModel(args.vocoder)

    phones = phonemize_espeak(args.text, voice="fa")
    phones = sum(phones, [])
    phone_ids = [am.token2id[i] for i in phones]

    padded_phone_ids = [0] * (len(phone_ids) * 2 + 1)
    padded_phone_ids[1::2] = phone_ids

    tokens = np.array([padded_phone_ids], dtype=np.int64)
    mel = am(tokens)
    audio = vocoder(mel)

    sf.write(args.out_wav, audio[0], am.sample_rate, "PCM_16")


if __name__ == "__main__":
    main()
