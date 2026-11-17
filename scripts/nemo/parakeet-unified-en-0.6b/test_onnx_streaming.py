#!/usr/bin/env python3
# Copyright      2026  Milan Leonard
# Copyright      2026  Julian Pscheid
"""ONNX Runtime smoke test for buffered Parakeet Unified streaming."""

import argparse
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

from buffered_streaming_helpers import normalize_per_feature


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, required=True)
    parser.add_argument("--decoder", type=str, required=True)
    parser.add_argument("--joiner", type=str, required=True)
    parser.add_argument("--tokens", type=str, required=True)
    parser.add_argument("--wav", type=str, required=True)
    return parser.parse_args()


def compute_features(audio, mel_filters):
    # NeMo uses whole-buffer preemphasis and centered, zero-padded STFT.
    audio = np.concatenate([audio[:1], audio[1:] - 0.97 * audio[:-1]])
    spectrum = librosa.stft(
        audio,
        n_fft=512,
        hop_length=160,
        win_length=400,
        window=np.hanning(400),
        center=True,
        pad_mode="constant",
    )
    features = np.log(mel_filters @ (np.abs(spectrum) ** 2) + 2**-24)
    return features[:, : len(audio) // 160].T.astype(np.float32)


class OnnxModel:
    def __init__(self, encoder, decoder, joiner):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        providers = ["CPUExecutionProvider"]

        self.encoder = ort.InferenceSession(
            encoder, sess_options=opts, providers=providers
        )
        self.decoder = ort.InferenceSession(
            decoder, sess_options=opts, providers=providers
        )
        self.joiner = ort.InferenceSession(
            joiner, sess_options=opts, providers=providers
        )

        meta = self.encoder.get_modelmeta().custom_metadata_map
        print("encoder meta:", meta)

        if meta.get("streaming_model_type") != "nemo_parakeet_unified_streaming":
            raise ValueError(
                "Expected streaming_model_type=nemo_parakeet_unified_streaming "
                "in encoder metadata"
            )

        self.feat_dim = int(meta.get("feat_dim", 128))
        self.subsampling_factor = int(meta.get("subsampling_factor", 8))
        self.left_feature_frames = int(meta["left_feature_frames"])
        self.chunk_feature_frames = int(meta["chunk_feature_frames"])
        self.right_feature_frames = int(meta["right_feature_frames"])
        self.left_encoder_frames = int(meta["left_encoder_frames"])
        self.chunk_encoder_frames = int(meta["chunk_encoder_frames"])
        self.pred_rnn_layers = int(meta["pred_rnn_layers"])
        self.pred_hidden = int(meta["pred_hidden"])
        self.normalize_type = meta.get("normalize_type", "per_feature")

    def get_decoder_state(self):
        s0 = np.zeros((self.pred_rnn_layers, 1, self.pred_hidden), dtype=np.float32)
        s1 = np.zeros((self.pred_rnn_layers, 1, self.pred_hidden), dtype=np.float32)
        return s0, s1

    def run_encoder(self, x: np.ndarray, length: int):
        x = torch.from_numpy(x).t().unsqueeze(0).contiguous()
        x_lens = np.array([length], dtype=np.int64)
        out = self.encoder.run(
            None,
            {
                self.encoder.get_inputs()[0].name: x.numpy(),
                self.encoder.get_inputs()[1].name: x_lens,
            },
        )
        return out[0], int(out[1][0])

    def run_decoder(self, token, s0, s1):
        target = np.array([[token]], dtype=np.int32)
        target_len = np.array([1], dtype=np.int32)
        out = self.decoder.run(
            None,
            {
                self.decoder.get_inputs()[0].name: target,
                self.decoder.get_inputs()[1].name: target_len,
                self.decoder.get_inputs()[2].name: s0,
                self.decoder.get_inputs()[3].name: s1,
            },
        )
        return out[0], out[2], out[3]

    def run_joiner(self, enc_t, dec_out):
        return self.joiner.run(
            None,
            {
                self.joiner.get_inputs()[0].name: enc_t,
                self.joiner.get_inputs()[1].name: dec_out,
            },
        )[0]


def decode_buffered(model: OnnxModel, audio: np.ndarray, blank: int):
    ans = [blank]
    s0, s1 = model.get_decoder_state()
    dec_out, s0_next, s1_next = model.run_decoder(ans[-1], s0, s1)
    max_token_per_frame = 10
    mel_filters = librosa.filters.mel(sr=16000, n_fft=512, n_mels=model.feat_dim)
    chunk = model.chunk_feature_frames * 160
    right = model.right_feature_frames * 160
    capacity = (model.left_feature_frames * 160) + chunk + right
    buffer = np.zeros(0, dtype=np.float32)
    left_samples = center_samples = right_samples = 0
    lo, hi = 0, min(chunk + right, len(audio))
    while lo < len(audio):
        new = audio[lo:hi]
        last = hi == len(audio)
        buffer = np.concatenate([buffer, new])
        left_samples += center_samples
        center_samples = 0
        right_samples += len(new)
        if last:
            center_samples, right_samples = right_samples, 0
        elif right_samples > chunk:
            center_samples = chunk
            right_samples -= chunk
        excess = max(len(buffer) - capacity, 0)
        left_samples -= excess
        buffer = buffer[excess:]
        window = compute_features(buffer, mel_filters)
        if model.normalize_type == "per_feature":
            window = normalize_per_feature(window)
        length = len(window)
        window = np.pad(window, ((0, 1), (0, 0)))
        enc, encoded_length = model.run_encoder(window, length)
        t_start = left_samples // (160 * model.subsampling_factor)
        count = (
            encoded_length - t_start
            if last
            else center_samples // (160 * model.subsampling_factor)
        )
        t_end = min(t_start + count, encoded_length)

        for t in range(t_start, t_end):
            enc_t = enc[:, :, t : t + 1]
            for _ in range(max_token_per_frame):
                logits = model.run_joiner(enc_t, dec_out)
                idx = int(np.argmax(logits.squeeze()))
                if idx == blank:
                    break
                ans.append(idx)
                s0, s1 = s0_next, s1_next
                dec_out, s0_next, s1_next = model.run_decoder(ans[-1], s0, s1)

        lo = hi
        hi = min(hi + chunk, len(audio))

    return ans[1:]


def main():
    args = get_args()
    for p in (args.encoder, args.decoder, args.joiner, args.tokens, args.wav):
        assert Path(p).is_file(), p

    model = OnnxModel(args.encoder, args.decoder, args.joiner)

    id2token = {}
    with open(args.tokens, encoding="utf-8") as f:
        for line in f:
            t, idx = line.split()
            id2token[int(idx)] = t

    audio, sr = sf.read(args.wav, dtype="float32", always_2d=True)
    audio = audio[:, 0]
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = np.concatenate([audio, np.zeros(16000 * 2, dtype=np.float32)])

    blank = len(id2token) - 1
    token_ids = decode_buffered(model, audio, blank)
    text = "".join(id2token[i] for i in token_ids).replace("▁", " ").strip()

    print(args.wav)
    print(text)


if __name__ == "__main__":
    main()
