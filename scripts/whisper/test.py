#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)
"""
Please first run ./export-onnx.py
before you run this script
"""
import argparse
import base64
from typing import List, Tuple

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import torch
import torchaudio


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
        "--language",
        type=str,
        help="""The actual spoken language in the audio.
        Example values, en, de, zh, jp, fr.
        If None, we will detect the language using the first 30s of the
        input audio
        """,
    )

    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        type=str,
        default="transcribe",
        help="Valid values are: transcribe, translate",
    )

    parser.add_argument(
        "sound_file",
        type=str,
        help="Path to the test wave",
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
            providers=["CPUExecutionProvider"],
        )

        meta = self.encoder.get_modelmeta().custom_metadata_map
        version = int(meta["version"])
        if version < 2:
            raise RuntimeError(
                "If you have exported your model before 2024-06-21, please re-export it using the latest "
                "https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/whisper/export-onnx.py\n"
                "See https://github.com/k2-fsa/sherpa-onnx/pull/1037 for details"
            )
        self.n_text_layer = int(meta["n_text_layer"])
        self.n_text_ctx = int(meta["n_text_ctx"])
        self.n_text_state = int(meta["n_text_state"])
        self.sot = int(meta["sot"])
        self.sot_lm = int(meta["sot_lm"])
        self.sot_prev = int(meta["sot_prev"])
        self.eot = int(meta["eot"])
        self.translate = int(meta["translate"])
        self.transcribe = int(meta["transcribe"])
        self.no_timestamps = int(meta["no_timestamps"])
        self.timestamp_begin = int(meta["timestamp_begin"])
        self.no_speech = int(meta["no_speech"])
        self.non_speech_tokens = list(map(int, meta["non_speech_tokens"].split(",")))
        self.blank = int(meta["blank_id"])

        self.sot_sequence = list(map(int, meta["sot_sequence"].split(",")))

        self.all_language_tokens = list(
            map(int, meta["all_language_tokens"].split(","))
        )
        self.all_language_codes = meta["all_language_codes"].split(",")
        self.lang2id = dict(zip(self.all_language_codes, self.all_language_tokens))
        self.id2lang = dict(zip(self.all_language_tokens, self.all_language_codes))

        self.is_multilingual = int(meta["is_multilingual"]) == 1

    def init_decoder(self, decoder: str):
        self.decoder = ort.InferenceSession(
            decoder,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
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
        """
        Args:
          logits: 1-D nd.array. Changed in-place.
        """
        # suppress blank
        if is_initial:
            logits[self.eot] = float("-inf")
            logits[self.blank] = float("-inf")

        logits[self.sot] = float("-inf")
        logits[self.sot_prev] = float("-inf")
        logits[self.sot_lm] = float("-inf")
        logits[self.no_speech] = float("-inf")

        logits[self.transcribe] = float("-inf")
        logits[self.translate] = float("-inf")

        logits[self.no_timestamps] = float("-inf")

        #  logits[self.all_language_tokens] = float("-inf")
        logits[self.non_speech_tokens] = float("-inf")

    def apply_timestamp_rules(self, tokens: List[int], logits: np.ndarray):
        """
        Args:
          logits: 1-D nd.array. Changed in-place.
        """
        sample_begin = 0
        max_initial_timestamp_index = 50
        logits[self.no_timestamps] = float("-inf")

        last_was_timestamp = len(tokens) >= 1 and tokens[-1] >= self.timestamp_begin
        penultimate_was_timestamp = (
            len(tokens) < 2 or tokens[-2] >= self.timestamp_begin
        )

        if last_was_timestamp:
            if penultimate_was_timestamp:
                logits[self.timestamp_begin :] = float("-inf")
            else:
                logits[: self.eot] = float("-inf")
        sampled_tokens = torch.tensor(tokens)
        timestamps = sampled_tokens[sampled_tokens.ge(self.timestamp_begin)]
        if timestamps.numel() > 0:
            logits[self.timestamp_begin : timestamps[-1].item()] = float("-inf")

        if len(tokens) == 0:
            logits[: self.timestamp_begin] = float("-inf")
            last_allowed = self.timestamp_begin + max_initial_timestamp_index
            logits[last_allowed + 1 :] = float("-inf")
        logprobs = torch.as_tensor(logits).log_softmax(dim=-1)
        timestamp_logprob = logprobs[self.timestamp_begin :].logsumexp(dim=-1)
        max_text_token_lobprob = logprobs[: self.timestamp_begin].max()
        if timestamp_logprob > max_text_token_lobprob:
            logits[: self.timestamp_begin] = float("-inf")

    def detect_language(
        self, n_layer_cross_k: torch.Tensor, n_layer_cross_v: torch.Tensor
    ) -> int:
        tokens = torch.tensor([[self.sot]], dtype=torch.int64)
        offset = torch.zeros(1, dtype=torch.int64)
        n_layer_self_k_cache, n_layer_self_v_cache = self.get_self_cache()

        logits, n_layer_self_k_cache, n_layer_self_v_cache = self.run_decoder(
            tokens=tokens,
            n_layer_self_k_cache=n_layer_self_k_cache,
            n_layer_self_v_cache=n_layer_self_v_cache,
            n_layer_cross_k=n_layer_cross_k,
            n_layer_cross_v=n_layer_cross_v,
            offset=offset,
        )
        logits = logits.reshape(-1)
        mask = torch.ones(logits.shape[0], dtype=torch.int64)
        mask[self.all_language_tokens] = 0
        logits[mask != 0] = float("-inf")
        lang_id = logits.argmax().item()
        print("detected language: ", self.id2lang[lang_id])
        return lang_id


def load_tokens(filename):
    tokens = dict()
    with open(filename, "r") as f:
        for line in f:
            t, i = line.split()
            tokens[int(i)] = t
    return tokens


def compute_features(filename: str) -> torch.Tensor:
    """
    Args:
      filename:
        Path to an audio file.
    Returns:
      Return a 1-D float32 tensor of shape (1, 80, 3000) containing the features.
    """
    wave, sample_rate = torchaudio.load(filename)
    audio = wave[0].contiguous()  # only use the first channel
    if sample_rate != 16000:
        audio = torchaudio.functional.resample(
            audio, orig_freq=sample_rate, new_freq=16000
        )

    features = []
    online_whisper_fbank = knf.OnlineWhisperFbank(knf.FrameExtractionOptions())
    online_whisper_fbank.accept_waveform(16000, audio.numpy())
    online_whisper_fbank.input_finished()
    for i in range(online_whisper_fbank.num_frames_ready):
        f = online_whisper_fbank.get_frame(i)
        f = torch.from_numpy(f)
        features.append(f)

    features = torch.stack(features)

    log_spec = torch.clamp(features, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    mel = (log_spec + 4.0) / 4.0
    # mel (T, 80)

    # We pad 1500 frames at the end so that it is able to detect eot
    # You can use another value instead of 1500.
    mel = torch.nn.functional.pad(mel, (0, 0, 0, 1500), "constant", 0)
    # Note that if it throws for a multilingual model,
    # please use a larger value, say 300

    target = 3000
    if mel.shape[0] > target:
        # -50 so that there are some zero tail paddings.
        mel = mel[: target - 50]
        mel = torch.nn.functional.pad(mel, (0, 0, 0, 50), "constant", 0)

    # We don't need to pad it to 30 seconds now!
    #  mel = torch.nn.functional.pad(mel, (0, 0, 0, target - mel.shape[0]), "constant", 0)

    mel = mel.t().unsqueeze(0)

    return mel


def main():
    args = get_args()

    mel = compute_features(args.sound_file)
    model = OnnxModel(args.encoder, args.decoder)

    n_layer_cross_k, n_layer_cross_v = model.run_encoder(mel)

    if args.language is not None:
        if model.is_multilingual is False and args.language != "en":
            print(f"This model supports only English. Given: {args.language}")
            return

        if args.language not in model.lang2id:
            print(f"Invalid language: {args.language}")
            print(f"Valid values are: {list(model.lang2id.keys())}")
            return

        # [sot, lang, task, notimestamps]
        model.sot_sequence[1] = model.lang2id[args.language]
    elif model.is_multilingual is True:
        print("detecting language")
        lang = model.detect_language(n_layer_cross_k, n_layer_cross_v)
        model.sot_sequence[1] = lang

    if args.task is not None:
        if model.is_multilingual is False and args.task != "transcribe":
            print("This model supports only English. Please use --task=transcribe")
            return
        assert args.task in ["transcribe", "translate"], args.task

        if args.task == "translate":
            model.sot_sequence[2] = model.translate

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
    offset += len(model.sot_sequence)
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

        logits, n_layer_self_k_cache, n_layer_self_v_cache = model.run_decoder(
            tokens=tokens,
            n_layer_self_k_cache=n_layer_self_k_cache,
            n_layer_self_v_cache=n_layer_self_v_cache,
            n_layer_cross_k=n_layer_cross_k,
            n_layer_cross_v=n_layer_cross_v,
            offset=offset,
        )
        offset += 1
        logits = logits[0, -1]
        model.suppress_tokens(logits, is_initial=False)
        model.apply_timestamp_rules(tokens=results, logits=logits)
        max_token_id = logits.argmax(dim=-1)
    token_table = load_tokens(args.tokens)
    s = b""
    for i in results:
        if i in token_table:
            s += base64.b64decode(token_table[i])

    print(s.decode().strip())


if __name__ == "__main__":
    main()
