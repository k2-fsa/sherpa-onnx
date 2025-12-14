#!/usr/bin/env python3
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
from typing import Tuple

import kaldi_native_fbank as knf
import numpy as np
import soundfile as sf


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--wav",
        type=str,
        required=True,
    )
    return parser.parse_args()


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


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
    args = get_args()
    print(vars(args))

    samples, sample_rate = load_audio(args.wav)
    if sample_rate != 16000:
        import librosa

        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    features = compute_feat(
        samples=samples,
        sample_rate=sample_rate,
    )
    print("features.shape", features.shape)
    if features.shape[0] > args.num_frames:
        features = features[: args.num_frames]
    elif features.shape[0] < args.num_frames:
        pad_width = ((0, args.num_frames - features.shape[0]), (0, 0))
        features = np.pad(features, pad_width, mode="constant", constant_values=0)

    features.tofile("input0.raw")

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
    prompt.tofile("input1.raw")


if __name__ == "__main__":
    main()
