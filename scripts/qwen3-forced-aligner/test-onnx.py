#!/usr/bin/env python3
# Copyright      2026  (authors: losewayy)
#
# Numerical parity check for the exported Qwen3-ForcedAligner ONNX models:
# runs the three-file ONNX pipeline and the original PyTorch model on the
# same audio/text pair and compares the per-word timestamp slots.
#
# Usage:
#   ./test-onnx.py --model ./qwen3-forced-aligner --onnx-dir ./out \
#       --wav ./test.wav --lang German \
#       --text "Raptorium Bergbau scheint profitierter als Monroe ..."

import argparse
import importlib.util
import sys
import types

import numpy as np
import torch


def feat_to_audio_tokens_len(feat_len: int, chunk_size: int = 100) -> int:
    full, rem = feat_len // chunk_size, feat_len % chunk_size
    a = (rem - 1) // 2 + 1
    a = (a - 1) // 2 + 1
    a = (a - 1) // 2 + 1
    return full * 13 + a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--onnx-dir", required=True)
    p.add_argument("--wav", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--lang", default="English")
    p.add_argument("--qwen-asr-repo", default="")
    args = p.parse_args()

    if args.qwen_asr_repo:
        sys.path.insert(0, args.qwen_asr_repo)

    # nagisa is imported at module level by qwen_asr but is only exercised
    # for Japanese text; stub it out when absent so other languages can be
    # verified without the extra dependency.
    if "nagisa" not in sys.modules and importlib.util.find_spec("nagisa") is None:
        sys.modules["nagisa"] = types.ModuleType("nagisa")

    import onnxruntime as ort
    import soundfile as sf
    from transformers import AutoConfig, AutoModel, AutoProcessor
    from qwen_asr.core.transformers_backend import (
        Qwen3ASRConfig,
        Qwen3ASRForConditionalGeneration,
        Qwen3ASRProcessor,
    )
    from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForceAlignProcessor

    AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
    AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)

    proc = AutoProcessor.from_pretrained(args.model, fix_mistral_regex=True)

    wav, sr = sf.read(args.wav)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    print(f"audio: {len(wav) / sr:.2f}s")

    ap = Qwen3ForceAlignProcessor()
    word_list, input_text = ap.encode_timestamp(args.text, args.lang)
    print("words:", len(word_list))

    inputs = proc(
        text=[input_text], audio=[wav], return_tensors="pt", padding=True
    )
    input_ids = inputs["input_ids"]
    feat = inputs["input_features"]
    feat_mask = inputs["feature_attention_mask"]

    # --- PyTorch reference ---
    m = AutoModel.from_pretrained(args.model).eval().float()
    with torch.no_grad():
        ref_logits = m.thinker(
            input_ids=input_ids,
            input_features=feat,
            attention_mask=inputs["attention_mask"],
            feature_attention_mask=feat_mask,
        ).logits

    # --- ONNX pipeline ---
    sess_conv = ort.InferenceSession(f"{args.onnx_dir}/conv_frontend.onnx")
    sess_enc = ort.InferenceSession(f"{args.onnx_dir}/encoder.onnx")
    sess_dec = ort.InferenceSession(f"{args.onnx_dir}/decoder.onnx")

    mel_bt = feat.permute(0, 2, 1).numpy()
    conv_out = sess_conv.run(None, {"input_features": mel_bt})[0]

    n_audio = feat_to_audio_tokens_len(int(feat_mask.sum().item()))
    tok_mask = np.arange(conv_out.shape[1])[None, :] < n_audio
    audio_feat = sess_enc.run(
        None,
        {"input_features": conv_out, "feature_attention_mask": tok_mask},
    )[0]

    logits = sess_dec.run(
        None,
        {
            "input_ids": input_ids.numpy().astype(np.int64),
            "audio_features": audio_feat.astype(np.float32),
            "attention_mask": inputs["attention_mask"]
            .numpy()
            .astype(np.int64),
        },
    )[0]

    diff = np.abs(logits - ref_logits.numpy())
    print(f"logits max_diff={diff.max():.5f} mean_diff={diff.mean():.6f}")

    ts_token_id = 151705
    ref_ids = ref_logits.argmax(-1).squeeze(0).numpy()
    onnx_ids = logits.argmax(-1).squeeze(0)
    pos = np.where(input_ids.numpy().squeeze(0) == ts_token_id)[0]
    print("num ts slots:", len(pos))

    ref_ts = ref_ids[pos] * 0.08
    onnx_ts = onnx_ids[pos] * 0.08
    n_mismatch = 0
    for i, w in enumerate(word_list):
        same = abs(ref_ts[2 * i] - onnx_ts[2 * i]) < 1e-6 and abs(
            ref_ts[2 * i + 1] - onnx_ts[2 * i + 1]
        ) < 1e-6
        n_mismatch += 0 if same else 1
        flag = "" if same else "  <-- DIFF"
        print(
            f"{w:20s} ref=({ref_ts[2*i]:7.3f},{ref_ts[2*i+1]:7.3f}) "
            f"onnx=({onnx_ts[2*i]:7.3f},{onnx_ts[2*i+1]:7.3f}){flag}"
        )
    print(f"mismatched words: {n_mismatch}/{len(word_list)}")


if __name__ == "__main__":
    main()
