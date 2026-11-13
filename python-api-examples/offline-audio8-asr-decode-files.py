#!/usr/bin/env python3
#
# Copyright (c)  2026  yuyun2000
#
"""
Decode audio files using Audio8-ASR-0.1B with the sherpa-onnx Python API.

The model is a Whisper-style log-mel encoder plus a small Qwen2 decoder, so the
recognizer needs the two ONNX graphs and the tokenizer directory.

Usage:
    python offline-audio8-asr-decode-files.py \\
        --encoder=./sherpa-onnx-audio8-asr-0.1B/encoder.onnx \\
        --decoder=./sherpa-onnx-audio8-asr-0.1B/decoder.onnx \\
        --tokenizer=./sherpa-onnx-audio8-asr-0.1B \\
        --num-threads=2 \\
        ./sherpa-onnx-audio8-asr-0.1B/test_wavs/0.wav

Note: the model was trained on 30 s windows, so longer audio should be split
with a VAD first.
"""

import argparse
import sys

import soundfile as sf

try:
    import sherpa_onnx
except ImportError:
    print("Please install sherpa-onnx: pip install sherpa-onnx")
    sys.exit(1)


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)

    parser.add_argument("--encoder", type=str, required=True,
                        help="Path to encoder.onnx")
    parser.add_argument("--decoder", type=str, required=True,
                        help="Path to decoder.onnx (KV cache decoder)")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to the tokenizer directory")
    parser.add_argument("--max-total-len", type=int, default=512,
                        help="Number of KV cache slots in the decoder graph")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Maximum number of generated tokens")
    parser.add_argument("--num-threads", type=int, default=2,
                        help="Number of threads for neural network computation")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Sample rate of the input audio")
    parser.add_argument("--feature-dim", type=int, default=128,
                        help="Mel feature dimension (Audio8-ASR uses 128)")
    parser.add_argument("--provider", type=str, default="cpu",
                        choices=["cpu", "cuda"], help="Provider: cpu or cuda")
    parser.add_argument("--debug", action="store_true",
                        help="True to print model information while loading")
    parser.add_argument("sound_files", type=str, nargs="+",
                        help="Input wav file(s), single-channel; sample rate arbitrary")

    return parser.parse_args()


def main():
    args = get_args()
    recognizer = sherpa_onnx.OfflineRecognizer.from_audio8_asr(
        encoder=args.encoder,
        decoder=args.decoder,
        tokenizer=args.tokenizer,
        num_threads=args.num_threads,
        sample_rate=args.sample_rate,
        feature_dim=args.feature_dim,
        debug=args.debug,
        provider=args.provider,
        max_total_len=args.max_total_len,
        max_new_tokens=args.max_new_tokens,
    )

    for filename in args.sound_files:
        audio, sample_rate = sf.read(filename, dtype="float32", always_2d=True)
        audio = audio[:, 0]
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        recognizer.decode_stream(stream)
        print(f"{filename}\n{stream.result.text}\n")


if __name__ == "__main__":
    main()
