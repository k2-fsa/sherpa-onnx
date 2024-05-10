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

        meta = self.model.get_modelmeta().custom_metadata_map
        print(meta)

        self.window_size = int(meta["window_size"])
        self.chunk_shift = int(meta["chunk_shift"])

        self.cache_last_channel_dim1 = int(meta["cache_last_channel_dim1"])
        self.cache_last_channel_dim2 = int(meta["cache_last_channel_dim2"])
        self.cache_last_channel_dim3 = int(meta["cache_last_channel_dim3"])

        self.cache_last_time_dim1 = int(meta["cache_last_time_dim1"])
        self.cache_last_time_dim2 = int(meta["cache_last_time_dim2"])
        self.cache_last_time_dim3 = int(meta["cache_last_time_dim3"])

        self.init_cache_state()

    def init_cache_state(self):
        self.cache_last_channel = torch.zeros(
            1,
            self.cache_last_channel_dim1,
            self.cache_last_channel_dim2,
            self.cache_last_channel_dim3,
            dtype=torch.float32,
        ).numpy()

        self.cache_last_time = torch.zeros(
            1,
            self.cache_last_time_dim1,
            self.cache_last_time_dim2,
            self.cache_last_time_dim3,
            dtype=torch.float32,
        ).numpy()

        self.cache_last_channel_len = torch.zeros([1], dtype=torch.int64).numpy()

    def __call__(self, x: np.ndarray):
        # x: (T, C)
        x = torch.from_numpy(x)
        x = x.t().unsqueeze(0)
        # x: [1, C, T]
        x_lens = torch.tensor([x.shape[-1]], dtype=torch.int64)

        (
            log_probs,
            log_probs_len,
            cache_last_channel_next,
            cache_last_time_next,
            cache_last_channel_len_next,
        ) = self.model.run(
            [
                self.model.get_outputs()[0].name,
                self.model.get_outputs()[1].name,
                self.model.get_outputs()[2].name,
                self.model.get_outputs()[3].name,
                self.model.get_outputs()[4].name,
            ],
            {
                self.model.get_inputs()[0].name: x.numpy(),
                self.model.get_inputs()[1].name: x_lens.numpy(),
                self.model.get_inputs()[2].name: self.cache_last_channel,
                self.model.get_inputs()[3].name: self.cache_last_time,
                self.model.get_inputs()[4].name: self.cache_last_channel_len,
            },
        )
        self.cache_last_channel = cache_last_channel_next
        self.cache_last_time = cache_last_time_next
        self.cache_last_channel_len = cache_last_channel_len_next

        # [T, vocab_size]
        return torch.from_numpy(log_probs).squeeze(0)


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

    window_size = model.window_size
    chunk_shift = model.chunk_shift

    blank = len(id2token) - 1
    prev = -1
    ans = []

    features = compute_features(audio, fbank)
    num_chunks = (features.shape[0] - window_size) // chunk_shift + 1
    for i in range(num_chunks):
        start = i * chunk_shift
        end = start + window_size
        chunk = features[start:end, :]

        log_probs = model(chunk)
        ids = torch.argmax(log_probs, dim=1).tolist()
        for i in ids:
            if i != blank and i != prev:
                ans.append(i)
            prev = i

    tokens = [id2token[i] for i in ans]
    underline = "▁"
    #  underline = b"\xe2\x96\x81".decode()
    text = "".join(tokens).replace(underline, " ").strip()
    print(args.wav)
    print(text)


main()
