#!/usr/bin/env python3
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
from typing import Tuple

import kaldi_native_fbank as knf
import numpy as np
import soundfile as sf
import torch
from ais_bench.infer.interface import InferSession


class OmModel:
    def __init__(self):
        self.model = InferSession(device_id=0, model_path="./model.om", debug=False)

        print("---model---")
        for i in self.model.get_inputs():
            print(i.name, i.datatype, i.shape)

        print("-----")

        for i in self.model.get_outputs():
            print(i.name, i.datatype, i.shape)

        self.num_frames = self.model.get_inputs()[0].shape[1]

    def __call__(self, x, prompt=None, language=None, text_norm=None):
        return self.model.infer([x, prompt], mode="static", custom_sizes=10000000)[0][0]
        return logits


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def load_tokens(filename):
    ans = dict()
    i = 0
    with open(filename, encoding="utf-8") as f:
        for line in f:
            ans[i] = line.strip().split()[0]
            i += 1
    return ans


def compute_feat(
    samples,
    sample_rate,
    window_size: int = 7,  # lfr_m
    window_shift: int = 6,  # lfr_n
):
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.window_type = "hamming"
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = 80

    online_fbank = knf.OnlineFbank(opts)
    online_fbank.accept_waveform(sample_rate, (samples * 32768).tolist())
    online_fbank.input_finished()

    features = np.stack(
        [online_fbank.get_frame(i) for i in range(online_fbank.num_frames_ready)]
    )
    assert features.data.contiguous is True
    assert features.dtype == np.float32, features.dtype

    T = (features.shape[0] - window_size) // window_shift + 1
    features = np.lib.stride_tricks.as_strided(
        features,
        shape=(T, features.shape[1] * window_size),
        strides=((window_shift * features.shape[1]) * 4, 4),
    )

    return np.copy(features)


def main():
    samples, sample_rate = load_audio("./test_wavs/zh.wav")
    if sample_rate != 16000:
        import librosa

        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    model = OmModel()

    features = compute_feat(
        samples=samples,
        sample_rate=sample_rate,
    )
    print("features.shape", features.shape)
    if model.num_frames > 0:
        if features.shape[0] < model.num_frames:
            features = np.pad(
                features,
                ((0, model.num_frames - features.shape[0]), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        elif features.shape[0] > model.num_frames:
            features = features[: model.num_frames]

        print("features.shape (new)", features.shape)

    language_auto = 0
    language_zh = 3
    language_en = 4
    language_yue = 7
    language_ya = 11
    language_ko = 12
    language_nospeech = 13

    language = language_auto

    with_itn = 14
    without_itn = 15

    text_norm = with_itn

    prompt = np.array([language, 1, 2, text_norm], dtype=np.int32)
    # language = np.array([language], dtype=np.int32)
    # text_norm = np.array([text_norm], dtype=np.int32)

    print("prompt", prompt.shape)

    logits = model(
        x=features[None],
        prompt=prompt,
        # language=language,
        ##text_norm=text_norm,
    )
    print("logits.shape", logits.shape, type(logits))

    idx = logits.argmax(axis=-1)
    print(idx)
    print(len(idx))
    prev = -1
    ids = []
    for i in idx:
        if i != prev:
            ids.append(i)
        prev = i
    ids = [i for i in ids if i != 0]
    print(ids)

    tokens = load_tokens("./tokens.txt")
    text = "".join([tokens[i] for i in ids])

    text = text.replace("▁", " ")
    print(text)


if __name__ == "__main__":
    main()
