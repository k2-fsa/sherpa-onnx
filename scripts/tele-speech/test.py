#!/usr/bin/env python3
# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)

from typing import Tuple

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf

"""
NodeArg(name='feats', type='tensor(float)', shape=[1, 'T', 40])
-----
NodeArg(name='logits', type='tensor(float)', shape=['Addlogits_dim_0', 1, 7535])
"""


class OnnxModel:
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

        self.show()

    def show(self):
        for i in self.model.get_inputs():
            print(i)

        print("-----")

        for i in self.model.get_outputs():
            print(i)

    def __call__(self, x):
        """
        Args:
          x: a float32 tensor of shape (N, T, C)
        """
        logits = self.model.run(
            [
                self.model.get_outputs()[0].name,
            ],
            {
                self.model.get_inputs()[0].name: x,
            },
        )[0]

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


def get_features(test_wav_filename):
    samples, sample_rate = load_audio(test_wav_filename)

    if sample_rate != 16000:
        import librosa

        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    samples *= 32768

    opts = knf.MfccOptions()
    # See https://github.com/Tele-AI/TeleSpeech-ASR/blob/master/mfcc_hires.conf
    opts.frame_opts.dither = 0

    opts.num_ceps = 40
    opts.use_energy = False

    opts.mel_opts.num_bins = 40
    opts.mel_opts.low_freq = 40
    opts.mel_opts.high_freq = -200

    mfcc = knf.OnlineMfcc(opts)
    mfcc.accept_waveform(16000, samples)
    frames = []
    for i in range(mfcc.num_frames_ready):
        frames.append(mfcc.get_frame(i))

    frames = np.stack(frames, axis=0)
    return frames


def cmvn(features):
    # See https://github.com/Tele-AI/TeleSpeech-ASR/blob/master/wenet_representation/conf/train_d2v2_ark_conformer.yaml#L70
    # https://github.com/Tele-AI/TeleSpeech-ASR/blob/master/wenet_representation/wenet/dataset/dataset.py#L184
    # https://github.com/Tele-AI/TeleSpeech-ASR/blob/master/wenet_representation/wenet/dataset/processor.py#L278
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / (std + 1e-5)


def main():
    # Please download the test data from
    # https://hf-mirror.com/csukuangfj/sherpa-onnx-paraformer-zh-small-2024-03-09/tree/main/test_wavs
    test_wav_filename = "./3-sichuan.wav"
    test_wav_filename = "./4-tianjin.wav"
    test_wav_filename = "./5-henan.wav"

    features = get_features(test_wav_filename)

    features = cmvn(features)

    features = np.expand_dims(features, axis=0)  # (T, C) -> (N, T, C)

    model_filename = "./model.int8.onnx"
    model = OnnxModel(model_filename)
    logits = model(features)
    logits = logits.squeeze(axis=1)  # remove batch axis
    ids = logits.argmax(axis=-1)

    id2token = dict()
    with open("./tokens.txt", encoding="utf-8") as f:
        for line in f:
            t, idx = line.split()
            id2token[int(idx)] = t

    tokens = []

    blank = 0
    prev = -1

    for k in ids:
        if k != blank and k != prev:
            tokens.append(k)
        prev = k

    tokens = [id2token[i] for i in tokens]
    text = "".join(tokens)
    print(text)


if __name__ == "__main__":
    main()
