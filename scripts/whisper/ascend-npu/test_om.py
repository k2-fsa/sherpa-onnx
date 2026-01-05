#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation

import argparse
import base64
from typing import List

import kaldi_native_fbank as knf
import librosa
import numpy as np
from ais_bench.infer.interface import InferSession


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
        help="Path to the encoder",
    )

    parser.add_argument(
        "--decoder",
        type=str,
        required=True,
        help="Path to the decoder",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="Path to the tokens",
    )

    parser.add_argument(
        "--wav",
        type=str,
        required=True,
        help="Path to the test wav",
    )

    return parser.parse_args()


def causal_mask_1d(n: int, L: int):
    """
    Returns a 1-D int mask of shape (L,) with:
      0 -> allowed
      1 -> masked (will be converted to -inf later)
    """
    mask = np.ones((L,), dtype=np.int32)
    if n > 0:
        mask[:n] = 0
    return mask


def load_audio(filename: str) -> np.ndarray:
    samples, _ = librosa.load(filename, sr=16000)

    samples = np.ascontiguousarray(samples)
    return samples


def compute_features(samples: np.ndarray, dim: int = 80) -> np.ndarray:
    """
    Returns:
      Return a 1-D float32 tensor of shape (1, 80, 3000) containing the features.
    """
    features = []
    opts = knf.WhisperFeatureOptions()
    opts.dim = dim
    online_whisper_fbank = knf.OnlineWhisperFbank(opts)
    online_whisper_fbank.accept_waveform(16000, samples)
    online_whisper_fbank.input_finished()

    features = np.stack(
        [
            online_whisper_fbank.get_frame(i)
            for i in range(online_whisper_fbank.num_frames_ready)
        ]
    )
    log_spec = np.log10(np.clip(features, a_min=1e-10, a_max=None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    mel = (log_spec + 4.0) / 4.0
    num_frames = mel.shape[0]
    target = 3000
    if num_frames < target:
        mel = np.pad(
            mel,
            pad_width=((0, target - num_frames), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    mel = np.expand_dims(mel.T, axis=0)
    mel = np.ascontiguousarray(mel)

    return mel


def load_tokens(filename):
    tokens = dict()
    with open(filename, "r") as f:
        for line in f:
            t, i = line.split()
            tokens[int(i)] = t
    return tokens


class OmModel:
    def __init__(self, encoder: str, decoder: str):
        self.encoder = InferSession(device_id=0, model_path=encoder, debug=False)
        self.decoder = InferSession(device_id=0, model_path=decoder, debug=False)

        name = self.encoder.get_inputs()[0].name

        if ".en" in name:
            self.sot_sequence = [50257, 50362]
            self.eot = 50256
        else:
            self.sot_sequence = [50258, 50259, 50359, 50363]
            self.eot = 50257

        if "tiny" in name:
            self.n_text_layer = 4
            self.n_text_ctx = 448
            self.n_text_state = 384
        elif "base" in name:
            self.n_text_layer = 6
            self.n_text_ctx = 448
            self.n_text_state = 512
        elif "small" in name:
            self.n_text_layer = 12
            self.n_text_ctx = 448
            self.n_text_state = 768
        elif "medium" in name:
            self.n_text_layer = 24
            self.n_text_ctx = 448
            self.n_text_state = 1024
        else:
            assert False, f"Unsupported encoder input {name}"

        print("---encoder---")
        for i in self.encoder.get_inputs():
            print(i.name, i.datatype, i.shape)

        print("-----")

        for i in self.encoder.get_outputs():
            print(i.name, i.datatype, i.shape)

        print("---decoder---")
        for i in self.decoder.get_inputs():
            print(i.name, i.datatype, i.shape)

        print("-----")

        for i in self.decoder.get_outputs():
            print(i.name, i.datatype, i.shape)

    def get_self_cache(self) -> List[np.ndarray]:
        self_cache = []
        batch_size = 1
        for i in range(self.n_text_layer):
            k = np.zeros(
                (batch_size, self.n_text_ctx, self.n_text_state), dtype=np.float32
            )
            v = np.zeros(
                (batch_size, self.n_text_ctx, self.n_text_state), dtype=np.float32
            )
            self_cache.extend([k, v])
        return self_cache

    def run_encoder(self, x: np.ndarray):
        """
        Args:
          x: (1, 80, 3000), np.float32
        Returns:
          cross_kv:
           - (k, v) for layer 0
           - (k, v) for layer 1
           - (k, v) for layer 2
           - (k, v) for layer 3
        """
        out = self.encoder.infer([x])
        return out

    def run_decoder(self, tokens: np.ndarray, self_kv, cross_kv, offset, mask):
        """
        Args:
          tokens: (1, 1), np.int32
          offset: (1,), np.int32
          mask: (model.n_text_ctx,), np.int32
        Returns:
          logit: (1, 1, vocab_size)
          this_self_kv
        """
        return self.decoder.infer([tokens] + self_kv + cross_kv + [offset, mask])


def main():
    args = get_args()
    print(vars(args))
    samples = load_audio(args.wav)
    features = compute_features(samples)
    print("features", features.shape)

    model = OmModel(args.encoder, args.decoder)

    cross_kv = model.run_encoder(features)

    self_kv = model.get_self_cache()

    offset = np.array([0], dtype=np.int32)
    for t in model.sot_sequence:
        token = np.array([[t]], dtype=np.int32)  # sot
        mask = causal_mask_1d(offset.item(), model.n_text_ctx)

        out = model.run_decoder(
            tokens=token, self_kv=self_kv, cross_kv=cross_kv, offset=offset, mask=mask
        )

        for i in range(1, len(out)):
            self_kv[i - 1][:, offset.item() : offset.item() + 1, :] = out[i]

        offset += 1

    idx = out[0][0, 0].argmax()

    eot = model.eot

    ans = []

    while idx != eot and offset.item() < 100:
        ans.append(idx)
        token = np.array([[idx]], dtype=np.int32)

        mask = causal_mask_1d(offset.item(), model.n_text_ctx)

        out = model.run_decoder(
            tokens=token, self_kv=self_kv, cross_kv=cross_kv, offset=offset, mask=mask
        )

        for i in range(1, len(out)):
            self_kv[i - 1][:, offset.item() : offset.item() + 1, :] = out[i]

        offset += 1
        idx = out[0][0, 0].argmax()

    print(ans)
    id2token = load_tokens(args.tokens)

    s = b""
    for i in ans:
        if i in id2token:
            s += base64.b64decode(id2token[i])

    print(s.decode().strip())
    return


if __name__ == "__main__":
    main()
