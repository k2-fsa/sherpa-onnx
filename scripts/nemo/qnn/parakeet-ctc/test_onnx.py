#!/usr/bin/env python3
# Copyright      2026  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
from pathlib import Path

import kaldi_native_fbank as knf
import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--wav",
        type=str,
        required=True,
    )
    return parser.parse_args()


class OnnxModel:
    def __init__(
        self,
        encoder: str,
    ):
        self.init_encoder(encoder)

    def init_encoder(self, encoder):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.encoder = ort.InferenceSession(
            encoder,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )

    def __call__(self, x: np.ndarray):
        log_probs = self.encoder.run(
            [
                self.encoder.get_outputs()[0].name,
            ],
            {
                self.encoder.get_inputs()[0].name: x,
            },
        )[0]
        return log_probs


def create_fbank():
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.remove_dc_offset = False
    opts.frame_opts.window_type = "hann"

    opts.mel_opts.low_freq = 0
    opts.mel_opts.num_bins = 80

    opts.mel_opts.is_librosa = True

    fbank = knf.OnlineFbank(opts)
    return fbank


def compute_features(audio, fbank, max_len=-1):
    assert len(audio.shape) == 1, audio.shape
    fbank.accept_waveform(16000, audio)
    ans = []
    processed = 0
    while processed < fbank.num_frames_ready:
        ans.append(np.array(fbank.get_frame(processed)))
        processed += 1

    features = np.stack(ans)

    if max_len > 0:
        if features.shape[0] > max_len:
            features = features[:max_len]
        elif features.shape[0] < max_len:
            features = np.pad(
                features,
                ((0, max_len - features.shape[0]), (0, 0)),
                mode="constant",
                constant_values=0,
            )

    features = np.ascontiguousarray(features)
    return features


def load_tokens():
    id2token = dict()
    with open("./tokens.txt", encoding="utf-8") as f:
        for line in f:
            t, idx = line.split()
            id2token[int(idx)] = t
    return id2token


def main():
    args = get_args()
    print(vars(args))

    name = Path(args.wav).stem

    model = OnnxModel("./model.onnx")
    max_len = model.encoder.get_inputs()[0].shape[2]
    print("max_len", max_len)

    fbank = create_fbank()
    audio, sample_rate = sf.read(args.wav, dtype="float32", always_2d=True)
    audio = audio[:, 0]  # only use the first channel
    if sample_rate != 16000:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=16000,
        )
        sample_rate = 16000

    tail_padding = np.zeros(sample_rate * 2)

    audio = np.concatenate([audio, tail_padding])

    features = compute_features(audio, fbank, max_len)

    mean = features.mean(axis=0, keepdims=True)
    stddev = features.std(axis=0, keepdims=True)
    features = (features - mean) / (stddev + 1e-5)
    features.tofile(f"{name}.raw")

    log_probs = model(features.T[None])

    id2token = load_tokens()
    blank = len(id2token) - 1
    ids = log_probs[0].argmax(axis=-1).tolist()
    y = []
    last = -1
    for i in ids:
        if i == last:
            continue

        last = i

        if i == blank:
            continue

        y.append(i)

    t = [id2token[i] for i in y]
    s = "".join(t)
    w = s.replace("▁", " ")
    print(w)


if __name__ == "__main__":
    main()
