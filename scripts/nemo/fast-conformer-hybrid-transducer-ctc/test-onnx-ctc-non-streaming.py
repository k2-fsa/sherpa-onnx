#!/usr/bin/env python3
# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import torch
import soundfile as sf
import librosa


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model.onnx")

    parser.add_argument("--tokens", type=str, required=True, help="Path to tokens.txt")

    parser.add_argument("--wav", type=str, required=True, help="Path to test.wav")

    return parser.parse_args()


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


def compute_features(audio, fbank):
    assert len(audio.shape) == 1, audio.shape
    fbank.accept_waveform(16000, audio)
    ans = []
    processed = 0
    while processed < fbank.num_frames_ready:
        ans.append(np.array(fbank.get_frame(processed)))
        processed += 1
    ans = np.stack(ans)
    return ans


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
        print("==========Input==========")
        for i in self.model.get_inputs():
            print(i)
        print("==========Output==========")
        for i in self.model.get_outputs():
            print(i)
        """
        ==========Input==========
        NodeArg(name='audio_signal', type='tensor(float)', shape=['audio_signal_dynamic_axes_1', 80, 'audio_signal_dynamic_axes_2'])
        NodeArg(name='length', type='tensor(int64)', shape=['length_dynamic_axes_1'])
        ==========Output==========
        NodeArg(name='logprobs', type='tensor(float)', shape=['logprobs_dynamic_axes_1', 'logprobs_dynamic_axes_2', 1025])
        """

        meta = self.model.get_modelmeta().custom_metadata_map
        self.normalize_type = meta["normalize_type"]
        print(meta)

    def __call__(self, x: np.ndarray):
        # x: (T, C)
        x = torch.from_numpy(x)
        x = x.t().unsqueeze(0)
        # x: [1, C, T]
        x_lens = torch.tensor([x.shape[-1]], dtype=torch.int64)

        log_probs = self.model.run(
            [
                self.model.get_outputs()[0].name,
            ],
            {
                self.model.get_inputs()[0].name: x.numpy(),
                self.model.get_inputs()[1].name: x_lens.numpy(),
            },
        )[0]
        # [batch_size, T, vocab_size]
        return torch.from_numpy(log_probs)


def main():
    args = get_args()
    assert Path(args.model).is_file(), args.model
    assert Path(args.tokens).is_file(), args.tokens
    assert Path(args.wav).is_file(), args.wav

    print(vars(args))

    model = OnnxModel(args.model)

    id2token = dict()
    with open(args.tokens, encoding="utf-8") as f:
        for line in f:
            t, idx = line.split()
            id2token[int(idx)] = t

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

    blank = len(id2token) - 1
    ans = []
    prev = -1

    print(audio.shape)
    features = compute_features(audio, fbank)
    if model.normalize_type != "":
        assert model.normalize_type == "per_feature", model.normalize_type
        features = torch.from_numpy(features)
        mean = features.mean(dim=1, keepdims=True)
        stddev = features.std(dim=1, keepdims=True) + 1e-5
        features = (features - mean) / stddev
        features = features.numpy()

    print("features.shape", features.shape)
    log_probs = model(features)

    print("log_probs.shape", log_probs.shape)

    log_probs = log_probs[0, :, :]  # remove batch dim
    ids = torch.argmax(log_probs, dim=1).tolist()
    for k in ids:
        if k != blank and k != prev:
            ans.append(k)
        prev = k

    tokens = [id2token[i] for i in ans]
    underline = "▁"
    #  underline = b"\xe2\x96\x81".decode()
    text = "".join(tokens).replace(underline, " ").strip()
    print(args.wav)
    print(text)


main()
