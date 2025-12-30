#!/usr/bin/env python3
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

from typing import Tuple

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf

BPE_UNK = chr(8263)
PRINTABLE_BASE_CHARS = (
    list(range(256, 287 + 1))
    + list(range(32, 126 + 1))
    + list(range(288, 305 + 1))
    + list(range(308, 318 + 1))
    + list(range(321, 328 + 1))
    + list(range(330, 382 + 1))
    + list(range(384, 422 + 1))
)


BYTE_TO_BCHAR = {b: chr(PRINTABLE_BASE_CHARS[b]) for b in range(256)}
BCHAR_TO_BYTE = {bc: b for b, bc in BYTE_TO_BCHAR.items()}
BCHAR_TO_BYTE[BPE_UNK] = 32  # map unk to space


def load_tokens(filename):
    ans = dict()
    i = 0
    with open(filename, encoding="utf-8") as f:
        for line in f:
            ans[i] = line.strip().split()[0]
            i += 1
    return ans


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel

    if sample_rate != 16000:
        import librosa

        data = librosa.resample(data, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def compute_feat(
    samples: np.ndarray,
    sample_rate: int,
    max_len: int,
):
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.window_type = "povey"
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = 80

    online_fbank = knf.OnlineFbank(opts)
    online_fbank.accept_waveform(sample_rate, samples.tolist())
    online_fbank.input_finished()

    features = np.stack(
        [online_fbank.get_frame(i) for i in range(online_fbank.num_frames_ready)]
    )

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

    assert features.data.contiguous is True
    assert features.dtype == np.float32, features.dtype

    return features


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
        shape = self.model.get_inputs()[0].shape
        self.max_len = shape[1]

        for i in self.model.get_inputs():
            print(i)

        print("-----")

        for i in self.model.get_outputs():
            print(i)

    def __call__(self, x):
        log_probs = self.model.run(
            [
                self.model.get_outputs()[0].name,
            ],
            {self.model.get_inputs()[0].name: x[None]},
        )[0]

        return log_probs


def main():
    wave = "./0.wav"
    wave = "./1.wav"
    samples, sample_rate = load_audio(wave)

    model = OnnxModel("./model.onnx")

    features = compute_feat(
        samples=samples,
        sample_rate=sample_rate,
        max_len=model.max_len,
    )
    print("features", features.shape)

    log_probs = model(features)

    idx = log_probs[0].argmax(axis=-1)
    print("idx", idx)
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

    s = b""
    for t in text:
        if t == "▁":
            continue
        elif t in BCHAR_TO_BYTE:
            s += bytes([BCHAR_TO_BYTE[t]])
        else:
            print("skip OOV", t)

    print(s.decode())


if __name__ == "__main__":
    main()
