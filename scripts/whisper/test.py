#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)
"""
Please first run ./export-onnx.py
before you run this script
"""
import base64
from typing import Tuple

import kaldi_native_fbank as knf
import onnxruntime as ort
import torch

import whisper
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        # fmt: off
        choices=[
            "tiny", "tiny.en", "base", "base.en",
            "small", "small.en", "medium", "medium.en",
            "large", "large-v1", "large-v2"],
        # fmt: on
    )
    return parser.parse_args()


class OnnxModel:
    def __init__(
        self,
        encoder: str,
        decoder: str,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 4

        self.session_opts = session_opts

        self.init_encoder(encoder)
        self.init_decoder(decoder)

    def init_encoder(self, encoder: str):
        self.encoder = ort.InferenceSession(
            encoder,
            sess_options=self.session_opts,
        )

        meta = self.encoder.get_modelmeta().custom_metadata_map
        self.n_text_layer = int(meta["n_text_layer"])
        self.n_text_ctx = int(meta["n_text_ctx"])
        self.n_text_state = int(meta["n_text_state"])
        self.sot = int(meta["sot"])
        self.eot = int(meta["eot"])
        self.translate = int(meta["translate"])
        self.no_timestamps = int(meta["no_timestamps"])
        self.no_speech = int(meta["no_speech"])
        self.blank = int(meta["blank_id"])

        self.sot_sequence = list(map(int, meta["sot_sequence"].split(",")))

        self.is_multilingual = int(meta["is_multilingual"]) == 1

    def init_decoder(self, decoder: str):
        self.decoder = ort.InferenceSession(
            decoder,
            sess_options=self.session_opts,
        )

    def run_encoder(
        self,
        mel: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_layer_cross_k, n_layer_cross_v = self.encoder.run(
            [
                self.encoder.get_outputs()[0].name,
                self.encoder.get_outputs()[1].name,
            ],
            {
                self.encoder.get_inputs()[0].name: mel.numpy(),
            },
        )
        return torch.from_numpy(n_layer_cross_k), torch.from_numpy(n_layer_cross_v)

    def run_decoder(
        self,
        tokens: torch.Tensor,
        n_layer_self_k_cache: torch.Tensor,
        n_layer_self_v_cache: torch.Tensor,
        n_layer_cross_k: torch.Tensor,
        n_layer_cross_v: torch.Tensor,
        offset: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, out_n_layer_self_k_cache, out_n_layer_self_v_cache = self.decoder.run(
            [
                self.decoder.get_outputs()[0].name,
                self.decoder.get_outputs()[1].name,
                self.decoder.get_outputs()[2].name,
            ],
            {
                self.decoder.get_inputs()[0].name: tokens.numpy(),
                self.decoder.get_inputs()[1].name: n_layer_self_k_cache.numpy(),
                self.decoder.get_inputs()[2].name: n_layer_self_v_cache.numpy(),
                self.decoder.get_inputs()[3].name: n_layer_cross_k.numpy(),
                self.decoder.get_inputs()[4].name: n_layer_cross_v.numpy(),
                self.decoder.get_inputs()[5].name: offset.numpy(),
            },
        )
        return (
            torch.from_numpy(logits),
            torch.from_numpy(out_n_layer_self_k_cache),
            torch.from_numpy(out_n_layer_self_v_cache),
        )

    def get_self_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = 1
        n_layer_self_k_cache = torch.zeros(
            self.n_text_layer,
            batch_size,
            self.n_text_ctx,
            self.n_text_state,
        )
        n_layer_self_v_cache = torch.zeros(
            self.n_text_layer,
            batch_size,
            self.n_text_ctx,
            self.n_text_state,
        )
        return n_layer_self_k_cache, n_layer_self_v_cache

    def suppress_tokens(self, logits, is_initial: bool) -> None:
        # suppress blank
        if is_initial:
            logits[self.eot] = float("-inf")
            logits[self.blank] = float("-inf")

        # suppress <|notimestamps|>
        logits[self.no_timestamps] = float("-inf")

        logits[self.sot] = float("-inf")
        logits[self.no_speech] = float("-inf")

        # logits is changed in-place
        logits[self.translate] = float("-inf")


def load_tokens(filename):
    tokens = dict()
    with open(filename, "r") as f:
        for line in f:
            t, i = line.split()
            tokens[int(i)] = t
    return tokens


def main():
    args = get_args()
    name = args.model

    encoder = f"./{name}-encoder.onnx"
    decoder = f"./{name}-decoder.onnx"
    audio = whisper.load_audio("0.wav")

    features = []
    online_whisper_fbank = knf.OnlineWhisperFbank(knf.FrameExtractionOptions())
    online_whisper_fbank.accept_waveform(16000, audio)
    online_whisper_fbank.input_finished()
    for i in range(online_whisper_fbank.num_frames_ready):
        f = online_whisper_fbank.get_frame(i)
        f = torch.from_numpy(f)
        features.append(f)

    features = torch.stack(features)

    log_spec = torch.clamp(features, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    mel = (log_spec + 4.0) / 4.0
    target = 3000
    mel = torch.nn.functional.pad(mel, (0, 0, 0, target - mel.shape[0]), "constant", 0)
    mel = mel.t().unsqueeze(0)

    model = OnnxModel(encoder, decoder)
    n_layer_cross_k, n_layer_cross_v = model.run_encoder(mel)
    n_layer_self_k_cache, n_layer_self_v_cache = model.get_self_cache()

    tokens = torch.tensor([model.sot_sequence], dtype=torch.int64)
    offset = torch.zeros(1, dtype=torch.int64)
    logits, n_layer_self_k_cache, n_layer_self_v_cache = model.run_decoder(
        tokens=tokens,
        n_layer_self_k_cache=n_layer_self_k_cache,
        n_layer_self_v_cache=n_layer_self_v_cache,
        n_layer_cross_k=n_layer_cross_k,
        n_layer_cross_v=n_layer_cross_v,
        offset=offset,
    )
    # logits.shape (batch_size, tokens.shape[1], vocab_size)
    logits = logits[0, -1]
    model.suppress_tokens(logits, is_initial=True)
    #  logits = logits.softmax(dim=-1)
    # for greedy search, we don't need to compute softmax or log_softmax
    max_token_id = logits.argmax(dim=-1)
    results = []
    for i in range(model.n_text_ctx):
        if max_token_id == model.eot:
            break
        results.append(max_token_id.item())
        tokens = torch.tensor([[results[-1]]])
        offset += 1

        logits, n_layer_self_k_cache, n_layer_self_v_cache = model.run_decoder(
            tokens=tokens,
            n_layer_self_k_cache=n_layer_self_k_cache,
            n_layer_self_v_cache=n_layer_self_v_cache,
            n_layer_cross_k=n_layer_cross_k,
            n_layer_cross_v=n_layer_cross_v,
            offset=offset,
        )
        logits = logits[0, -1]
        model.suppress_tokens(logits, is_initial=False)
        max_token_id = logits.argmax(dim=-1)
    token_table = load_tokens(f"./{name}-tokens.txt")
    s = b""
    for i in results:
        if i in token_table:
            s += base64.b64decode(token_table[i])
        else:
            print("oov", i)

    print(s.decode().strip())
    print(results)
    print(model.sot_sequence)


if __name__ == "__main__":
    main()
