#!/usr/bin/env python3
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
import time

import kaldi_native_fbank as knf
import librosa
import numpy as np
import onnxruntime as ort


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to onnx model file",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--wav",
        type=str,
        required=True,
        help="Path to test wav",
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

    def __call__(self, x, mask):
        """
        Args:
          x: (N, T, C), float32
          mask: (N, T), int64
        Returns:
          logits: (N, T/4, vocab_size), float32
          logits_len: (N,) int64
        """
        logits, logits_len = self.model.run(
            [
                self.model.get_outputs()[0].name,
                self.model.get_outputs()[1].name,
            ],
            {
                self.model.get_inputs()[0].name: x,
                self.model.get_inputs()[1].name: mask,
            },
        )

        return logits, logits_len


def load_tokens(tokens):
    id2token = dict()
    with open(tokens, encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            if len(fields) == 1:
                id2token[int(fields[0])] = " "
            else:
                t, idx = fields
                id2token[int(idx)] = t
    return id2token


def compute_feat(samples):
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = True
    opts.frame_opts.window_type = "hanning"
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.preemph_coeff = 0
    opts.frame_opts.remove_dc_offset = False
    opts.mel_opts.num_bins = 128

    online_fbank = knf.OnlineFbank(opts)
    online_fbank.accept_waveform(16000, samples.tolist())
    online_fbank.input_finished()

    features = np.stack(
        [online_fbank.get_frame(i) for i in range(online_fbank.num_frames_ready)]
    )
    assert features.dtype == np.float32, features.dtype

    features = np.ascontiguousarray(features)

    return features


def main():
    args = get_args()
    print(vars(args))

    model = OnnxModel(args.model)

    samples, sample_rate = librosa.load(args.wav, sr=16000)

    start = time.time()

    assert sample_rate == 16000, sample_rate
    features = compute_feat(samples)
    mask = np.ones(features.shape[0], dtype=np.int64)[None]
    features = features[None]

    logits, logits_len = model(features, mask)
    idx = logits[0, : logits_len[0]].argmax(axis=-1)

    end = time.time()
    elapsed_seconds = end - start
    audio_duration = samples.shape[0] / 16000
    real_time_factor = elapsed_seconds / audio_duration

    print("idx", idx)

    unique_ids = []
    prev = -1
    for i in idx.tolist():
        if i == prev:
            continue
        unique_ids.append(i)
        prev = i
    print("unique_ids", unique_ids)
    blank_id = 0
    ids = [i for i in unique_ids if i != blank_id]
    print(ids)

    id2token = load_tokens(args.tokens)

    tokens = [id2token[i] for i in ids]
    text = "".join(tokens)
    print(text)
    text = text.replace("▁", " ")
    print(text)
    print(f"RTF: {real_time_factor}")


if __name__ == "__main__":
    main()
