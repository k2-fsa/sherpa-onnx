#!/usr/bin/env python3
#
# Copyright (c)  2026  zengyw
#
"""
Decode audio files using Qwen3-ASR with sherpa-onnx Python API.

Usage:
    python offline-qwen3-asr-decode-files.py \\
        --conv-frontend=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx \\
        --encoder=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx \\
        --decoder=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx \\
        --tokenizer=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer \\
        --max-new-tokens=128 \\
        --num-threads=2 \\
        ./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/test_wavs/raokouling.wav

Note: If the input audio is too long, you can increase --max-new-tokens (e.g., 256).
You can also change it per-stream after creating the recognizer:
    stream = recognizer.create_stream()
    stream.set_option("max_new_tokens", "256")
"""

import argparse
import sys
from pathlib import Path

import soundfile as sf

try:
    import sherpa_onnx
except ImportError:
    print("Please install sherpa-onnx: pip install sherpa-onnx")
    sys.exit(1)


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    parser.add_argument(
        "--conv-frontend",
        type=str,
        required=True,
        help="Path to conv_frontend.onnx",
    )

    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
        help="Path to encoder.onnx",
    )

    parser.add_argument(
        "--decoder",
        type=str,
        required=True,
        help="Path to decoder.onnx (KV cache LLM)",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to tokenizer directory (vocab.json, merges.txt, ...)",
    )

    parser.add_argument(
        "--max-total-len",
        type=int,
        default=512,
        help="Maximum KV cache sequence length",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1e-6,
        help="Sampling temperature",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Top-p (nucleus) sampling threshold",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=2,
        help="Number of threads for neural network computation",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Input audio sample rate for feature extractor",
    )

    parser.add_argument(
        "--feature-dim",
        type=int,
        default=128,
        help="Mel feature dimension (Qwen3-ASR offline uses 128)",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Provider: cpu or cuda",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="True to print model information while loading",
    )

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="+",
        help="Input wav file(s), single-channel 16-bit PCM; sample rate arbitrary.",
    )

    return parser.parse_args()


def create_recognizer(args) -> sherpa_onnx.OfflineRecognizer:
    return sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
        conv_frontend=args.conv_frontend,
        encoder=args.encoder,
        decoder=args.decoder,
        tokenizer=args.tokenizer,
        num_threads=args.num_threads,
        sample_rate=args.sample_rate,
        feature_dim=args.feature_dim,
        provider=args.provider,
        debug=args.debug,
        max_total_len=args.max_total_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )


def decode_file(recognizer: sherpa_onnx.OfflineRecognizer, filename: str):
    audio, sample_rate = sf.read(filename, dtype="float32", always_2d=True)
    audio = audio[:, 0]

    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio)
    recognizer.decode_stream(stream)
    return stream.result


def main():
    args = get_args()
    print("Creating recognizer...")
    recognizer = create_recognizer(args)
    print("Recognizer created!")
    print(recognizer.config)

    for f in args.sound_files:
        if not Path(f).is_file():
            print(f"Skip missing file: {f}", file=sys.stderr)
            continue
        result = decode_file(recognizer, f)
        print(f"{f}\n  text: {result.text}\n")


if __name__ == "__main__":
    main()
