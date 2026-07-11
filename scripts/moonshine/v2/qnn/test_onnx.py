#!/usr/bin/env python3
# Copyright      2026  Xiaomi Corp.        (authors: Fangjun Kuang)

"""
Test moonshine ONNX model (QNN pattern).

Encoder: computes per-layer cross K/V.
Decoder: split attention — computes q@cache and q@new separately.
         Returns delta K/V. Cache updated externally.

Usage:
  cd scripts/moonshine/v2/qnn
  python3 test_onnx.py --wav zh.wav
"""

import argparse
import base64
from pathlib import Path
from typing import List

import librosa
import numpy as np
import onnxruntime as ort


def load_audio(filename, sample_rate=16000):
    audio, sr = librosa.load(filename, sr=sample_rate)
    assert sr == sample_rate
    assert len(audio.shape) == 1
    return audio


def causal_mask_1d(n: int, L: int):
    """1-D mask: 0=allowed, 1=masked."""
    mask = np.ones((L,), dtype=np.int32)
    if n > 0:
        mask[:n] = 0
    return mask


class OnnxModel:
    def __init__(self, encoder_path, decoder_path):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 4

        self.encoder = ort.InferenceSession(
            encoder_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.decoder = ort.InferenceSession(
            decoder_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )

        meta = self.encoder.get_modelmeta().custom_metadata_map
        if not isinstance(meta, dict):
            meta = {m.key: m.value for m in meta}

        self.num_layers = int(meta.get("num_decoder_layers", 0))
        self.hidden_size = int(meta.get("hidden_size", 0))
        self.max_seq_len = int(meta.get("max_seq_len", 194))
        self.max_audio_len = int(meta.get("max_audio_len", 160000))
        self.enc_seq_len = int(meta.get("enc_seq_len", 415))

        # Discover decoder input/output names
        self.decoder_inputs = [i.name for i in self.decoder.get_inputs()]
        self.decoder_outputs = [o.name for o in self.decoder.get_outputs()]

        print(
            f"Config: layers={self.num_layers}, hidden={self.hidden_size}, max_seq={self.max_seq_len}, "
            f"max_audio={self.max_audio_len}, enc_seq={self.enc_seq_len}"
        )

    def run_encoder(self, audio):
        """Returns list of [cross_k_0, cross_v_0, cross_k_1, cross_v_1, ...]."""
        # Pad or truncate to fixed audio length
        if len(audio) < self.max_audio_len:
            audio = np.pad(audio, (0, self.max_audio_len - len(audio)))
        else:
            audio = audio[: self.max_audio_len]
        audio = audio[np.newaxis, :]  # (1, max_audio_len)
        return self.encoder.run(None, {"audio": audio})

    def save_encoder_input(self, audio, name):
        """Save encoder input as raw file for quantization."""
        # Pad or truncate to fixed audio length
        if len(audio) < self.max_audio_len:
            audio = np.pad(audio, (0, self.max_audio_len - len(audio)))
        else:
            audio = audio[: self.max_audio_len]

        raw_file = f"{name}-audio.raw"
        audio.tofile(raw_file)
        print(f"Saved encoder input to {raw_file}")
        return raw_file

    def get_self_cache(self) -> List[np.ndarray]:
        """Pre-allocated zero caches: [self_k_0, self_v_0, self_k_1, self_v_1, ...]."""
        cache = []
        for _ in range(self.num_layers):
            cache.append(
                np.zeros((1, self.max_seq_len, self.hidden_size), dtype=np.float32)
            )
            cache.append(
                np.zeros((1, self.max_seq_len, self.hidden_size), dtype=np.float32)
            )
        return cache

    def run_decoder(self, token, self_kv, cross_kv, offset, mask):
        """
        Args:
            token: (1, 1) int32
            self_kv: [self_k_0, self_v_0, ...] per layer
            cross_kv: [cross_k_0, cross_v_0, ...] per layer
            offset: (1,) int32
            mask: (max_seq,) int32
        Returns:
            logits: (1, 1, vocab_size)
            this_kv: [this_self_k_0, this_self_v_0, ...] delta per layer
        """
        inputs = [token] + self_kv + cross_kv + [offset, mask]
        feed = {self.decoder_inputs[i]: inputs[i] for i in range(len(inputs))}
        out = self.decoder.run(None, feed)
        logits = out[0]
        this_kv = out[1:]  # delta K/V per layer
        return logits, this_kv

    def decode(self, audio, max_len=None):
        if max_len is None:
            max_len = int(len(audio) / 16000 * 15)

        cross_kv = self.run_encoder(audio)
        self_kv = self.get_self_cache()

        offset = np.array([0], dtype=np.int32)
        mask = causal_mask_1d(0, self.max_seq_len)

        tokens = []
        token_id = 1  # BOS

        for step in range(max_len):
            if offset.item() >= self.max_seq_len:
                print(f"Warning: reached max_seq_len={self.max_seq_len}, stopping")
                break

            token = np.array([[token_id]], dtype=np.int64)
            mask = causal_mask_1d(offset.item(), self.max_seq_len)

            logits, this_kv = self.run_decoder(token, self_kv, cross_kv, offset, mask)

            # Update cache externally
            for i in range(len(this_kv)):
                self_kv[i][:, offset.item() : offset.item() + 1, :] = this_kv[i]

            token_id = int(np.argmax(logits[0, 0]))
            offset += 1

            if token_id == 2:  # EOS
                break
            tokens.append(token_id)

        return tokens


def load_tokens(tokens_path="tokens.txt"):
    """Load tokens from tokens.txt (base64 encoded)."""
    id2token = {}
    with open(tokens_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                b64, idx = parts
                token_bytes = base64.b64decode(b64)
                id2token[int(idx)] = token_bytes
    return id2token


def decode_tokens(tokens, id2token):
    """Decode token ids to text, handling byte tokens correctly."""
    byte_stream = b""
    for i in tokens:
        token_bytes = id2token.get(i, b"")
        # Check if this is a hex string like <0xE5>
        if token_bytes.startswith(b"<0x") and token_bytes.endswith(b">"):
            try:
                hex_str = token_bytes[3:-1].decode("ascii")
                byte_stream += bytes([int(hex_str, 16)])
            except ValueError:
                byte_stream += token_bytes
        else:
            byte_stream += token_bytes
    text = byte_stream.decode("utf-8", errors="replace")
    return text.replace("▁", " ").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="./encoder.onnx")
    parser.add_argument("--decoder", type=str, default="./decoder.onnx")
    parser.add_argument("--tokens", type=str, default="./tokens.txt")
    parser.add_argument("--wav", type=str, required=True)
    args = parser.parse_args()

    print("Loading model...")
    model = OnnxModel(args.encoder, args.decoder)

    print(f"Loading audio: {args.wav}")
    audio = load_audio(args.wav)
    print(f"Audio: {audio.shape}, {len(audio)/16000:.2f}s")

    # Save encoder input for quantization
    name = Path(args.wav).stem
    raw_file = model.save_encoder_input(audio, name)

    # Create {name}-encoder.txt for quantization
    encoder_list_file = f"{name}-encoder.txt"
    with open(encoder_list_file, "w") as f:
        f.write(f"{raw_file}\n")
    print(f"Saved {encoder_list_file}")

    print("Decoding...")
    tokens = model.decode(audio)
    print(f"Tokens ({len(tokens)}): {tokens}")

    id2token = load_tokens(args.tokens)
    text = decode_tokens(tokens, id2token)
    print(f"Text: {text}")


if __name__ == "__main__":
    main()
