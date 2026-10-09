#!/usr/bin/env python3
# Copyright      2026  (authors: losewayy)
#
# Export Qwen/Qwen3-ForcedAligner-0.6B to ONNX in sherpa-onnx's qwen3-asr
# three-file layout:
#   conv_frontend.onnx  (B,T,128) mel -> (B,T',1024) conv features
#   encoder.onnx        conv_output + bool mask -> audio_features (B,A,1024)
#   decoder.onnx        input_ids + audio_features + attention_mask
#                       -> logits (B,S,5000)   [single pass, no KV cache]
#
# The audio encoder shares the AuT architecture with Qwen3-ASR; the decoder
# differs only in lm_head (a 5000-class timestamp classifier) and needs no
# autoregressive cache.
#
# Requirements:
#   pip install -U qwen-asr onnx onnxruntime
# or point --qwen-asr-repo at a local checkout of
# https://github.com/QwenLM/Qwen3-ASR
#
# Usage:
#   huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B \
#       --local-dir ./qwen3-forced-aligner
#   ./export-onnx.py --model ./qwen3-forced-aligner --outdir ./out

import argparse
import importlib.util
import os
import sys
import types

import torch
from transformers import AutoConfig, AutoModel, AutoProcessor


def register_qwen3_asr():
    try:
        from qwen_asr.core.transformers_backend import (
            Qwen3ASRConfig,
            Qwen3ASRForConditionalGeneration,
            Qwen3ASRProcessor,
        )
    except ImportError as e:
        sys.exit(
            f"{e}\n\nPlease `pip install -U qwen-asr` or pass "
            "--qwen-asr-repo /path/to/Qwen3-ASR"
        )

    AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
    AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="Path to the Qwen3-ForcedAligner-0.6B checkpoint")
    p.add_argument("--outdir", required=True)
    p.add_argument("--qwen-asr-repo", default="",
                   help="Optional path to a local Qwen3-ASR checkout; "
                        "not needed when the qwen-asr package is installed")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--chunk-size", type=int, default=100)
    args = p.parse_args()

    if args.qwen_asr_repo:
        sys.path.insert(0, args.qwen_asr_repo)

    # nagisa is imported at module level by qwen_asr but is only exercised
    # for Japanese text, which the exporter never touches; stub it out when
    # absent so the import succeeds without the extra dependency.
    if "nagisa" not in sys.modules and importlib.util.find_spec("nagisa") is None:
        sys.modules["nagisa"] = types.ModuleType("nagisa")

    register_qwen3_asr()

    proc = AutoProcessor.from_pretrained(args.model, fix_mistral_regex=True)
    m = AutoModel.from_pretrained(args.model)
    m.eval().float()

    thinker = m.thinker if hasattr(m, "thinker") else m
    thinker.eval().float()

    audio_token_id = int(getattr(thinker.config, "audio_token_id", 151676))
    hidden_size = int(thinker.config.text_config.hidden_size)
    print(f"audio_token_id={audio_token_id} hidden_size={hidden_size}")

    os.makedirs(args.outdir, exist_ok=True)

    # ---- conv_frontend ----
    from conv_frontend import ConvFrontend, _feat_to_audio_tokens_len

    conv_frontend = ConvFrontend(
        thinker.audio_tower, chunk_size=args.chunk_size
    ).eval()
    dummy_mel = torch.randn(1, 200, 128)
    with torch.no_grad():
        torch.onnx.export(
            conv_frontend,
            (dummy_mel,),
            os.path.join(args.outdir, "conv_frontend.onnx"),
            input_names=["input_features"],
            output_names=["conv_output"],
            opset_version=args.opset,
            dynamic_axes={
                "input_features": {0: "batch", 1: "n_frames"},
                "conv_output": {0: "batch", 1: "n_audio_tokens"},
            },
            do_constant_folding=True,
            dynamo=False,
        )
    print("[export] conv_frontend.onnx")

    # ---- encoder ----
    from encoder import AudioEncoderWrapper

    enc_w = AudioEncoderWrapper(
        thinker,
        tokens_per_chunk=conv_frontend.tokens_per_chunk,
        window_aftercnn=conv_frontend.window_aftercnn,
    ).eval()

    with torch.no_grad():
        conv_output = conv_frontend(dummy_mel)
        A = int(conv_output.shape[1])
        # real feat_len=200 -> audio tokens
        a_len = _feat_to_audio_tokens_len(
            torch.tensor([200]), chunk_size=args.chunk_size
        )
        token_mask = (
            torch.arange(A).unsqueeze(0) < a_len.unsqueeze(1)
        ).to(torch.bool)

        torch.onnx.export(
            enc_w,
            (conv_output, token_mask),
            os.path.join(args.outdir, "encoder.onnx"),
            input_names=["input_features", "feature_attention_mask"],
            output_names=["audio_features"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes={
                "input_features": {0: "batch", 1: "n_audio_tokens"},
                "feature_attention_mask": {0: "batch", 1: "n_audio_tokens"},
                "audio_features": {0: "batch", 1: "n_audio_tokens"},
            },
            dynamo=False,
        )
    print("[export] encoder.onnx")

    # ---- decoder (single-pass, no KV cache) ----
    from aligner_decoder import AlignerDecoderWrapper

    dec_w = AlignerDecoderWrapper(
        thinker, audio_token_id=audio_token_id, hidden_size=hidden_size
    ).eval()

    B, S = 1, 64
    n_audio = int(a_len.item())
    input_ids = torch.randint(0, 150000, (B, S), dtype=torch.int64)
    input_ids[:, : n_audio + 2] = audio_token_id  # pretend audio pads
    audio_features = torch.randn(B, n_audio, hidden_size)
    attention_mask = torch.ones(B, S, dtype=torch.int64)

    with torch.no_grad():
        torch.onnx.export(
            dec_w,
            (input_ids, audio_features, attention_mask),
            os.path.join(args.outdir, "decoder.onnx"),
            input_names=["input_ids", "audio_features", "attention_mask"],
            output_names=["logits"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "audio_features": {0: "batch", 1: "n_audio_tokens"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch", 1: "seq"},
            },
            dynamo=False,
        )
    print("[export] decoder.onnx")
    print("[export] Done ->", args.outdir)


if __name__ == "__main__":
    main()
