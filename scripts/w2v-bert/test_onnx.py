#!/usr/bin/env python3
# Copyright (c)  2026  graehl

"""
Decode waves with a w2v-BERT CTC model exported by export_onnx.py, using
only onnxruntime and kaldi-native-fbank.

Usage:

    ./test_onnx.py --model ./model.int8.onnx --tokens ./tokens.txt \\
        ./0.wav ./1.wav

Each wave is decoded alone. With --check-batch, all of them are then
decoded as one zero-padded batch, which must give the same text and, on
each wave's valid frames, log_probs within 1e-3. That holds
for model.onnx; model.int8.onnx quantizes activations with ranges taken
over the whole batch, so a batch can change a character or two.
"""

import argparse
from typing import List

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", type=str, required=True, help="model.onnx")
    parser.add_argument("--tokens", type=str, required=True, help="tokens.txt")
    parser.add_argument(
        "--check-batch",
        action="store_true",
        help="Also decode the waves as one padded batch and require the same "
        "text as decoding each alone",
    )
    parser.add_argument("waves", nargs="+", help="16 kHz mono waves")
    return parser.parse_args()


def compute_features(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """What transformers' SeamlessM4TFeatureExtractor computes, as
    (T // 2, 160) float32.

    80 Kaldi filterbank energies per 10 ms from samples scaled to 16 bits,
    with no padding at the edges; each bin normalized to zero mean and unit
    variance (the sample variance) over the utterance; and each two
    consecutive frames joined into one row, dropping an odd last frame.
    """
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = sample_rate
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = True
    opts.mel_opts.num_bins = 80
    opts.mel_opts.low_freq = 20
    opts.mel_opts.high_freq = 0

    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(sample_rate, (samples * 32768).tolist())
    fbank.input_finished()
    frames = np.stack(
        [fbank.get_frame(i) for i in range(fbank.num_frames_ready)]
    ).astype(np.float64)

    mean = frames.mean(axis=0, keepdims=True)
    var = frames.var(axis=0, ddof=1, keepdims=True)
    frames = (frames - mean) / np.sqrt(var + 1e-7)

    n = frames.shape[0] // 2 * 2
    return frames[:n].reshape(n // 2, 160).astype(np.float32)


def load_tokens(filename: str):
    id2token = {}
    with open(filename, encoding="utf-8") as f:
        for line in f:
            t, i = line.split()
            id2token[int(i)] = t
    return id2token


def greedy(log_probs: np.ndarray, blank_id: int) -> List[int]:
    best = log_probs.argmax(axis=-1)
    return [
        int(t)
        for i, t in enumerate(best)
        if t != blank_id and (i == 0 or t != best[i - 1])
    ]


def to_text(ids: List[int], id2token) -> str:
    text = "".join(id2token[i] for i in ids if not id2token[i].startswith("<"))
    return text.replace("▁", " ").strip()


class OnnxModel:
    def __init__(self, filename: str):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 4
        self.model = ort.InferenceSession(
            filename, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        meta = self.model.get_modelmeta().custom_metadata_map
        self.blank_id = int(meta["blank_id"])

    def __call__(self, x: np.ndarray, x_lengths: np.ndarray):
        """x: (N, T, 160) float32; x_lengths: (N,) int64.
        Returns log_probs (N, T_out, vocab_size) and log_probs_lengths (N,)."""
        return self.model.run(
            ["log_probs", "log_probs_lengths"],
            {"input_features": x, "input_lengths": x_lengths},
        )


def main():
    args = get_args()
    model = OnnxModel(args.model)
    id2token = load_tokens(args.tokens)

    features = []
    for w in args.waves:
        samples, sample_rate = sf.read(w, dtype="float32", always_2d=True)
        assert sample_rate == 16000, (w, sample_rate)
        features.append(compute_features(samples[:, 0], sample_rate))

    alone = []
    alone_log_probs = []
    for w, x in zip(args.waves, features):
        log_probs, _ = model(x[None], np.array([x.shape[0]], dtype=np.int64))
        total = np.exp(log_probs[0]).sum(axis=-1)
        if not np.allclose(total, 1.0, atol=1e-3):
            raise SystemExit(f"{w}: log_probs are not log-softmax normalized")
        alone_log_probs.append(log_probs[0])
        alone.append(to_text(greedy(log_probs[0], model.blank_id), id2token))
        print(f"{w}: {alone[-1]}")

    if not args.check_batch:
        return
    lengths = np.array([x.shape[0] for x in features], dtype=np.int64)
    batch = np.zeros((len(features), lengths.max(), 160), dtype=np.float32)
    for i, x in enumerate(features):
        batch[i, : x.shape[0]] = x
    log_probs, log_probs_lengths = model(batch, lengths)
    worst = 0.0
    for i, w in enumerate(args.waves):
        valid = log_probs[i, : log_probs_lengths[i]]
        if valid.shape != alone_log_probs[i].shape:
            raise SystemExit(
                f"{w}: {valid.shape[0]} valid frames batched, "
                f"{alone_log_probs[i].shape[0]} alone"
            )
        diff = float(np.abs(valid - alone_log_probs[i]).max())
        worst = max(worst, diff)
        # A NaN or infinite logit makes diff NaN, which compares false.
        if not np.isfinite(diff) or diff > 1e-3:
            raise SystemExit(f"{w}: batched log_probs differ by up to {diff:.2e}")
        text = to_text(greedy(valid, model.blank_id), id2token)
        if text != alone[i]:
            raise SystemExit(f"{w}: batched decoding gives '{text}'")
    print(
        f"A padded batch of {len(features)} gives the same text, "
        f"with log_probs within {worst:.2e} of decoding each alone"
    )


if __name__ == "__main__":
    main()
