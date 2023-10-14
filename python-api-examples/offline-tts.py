#!/usr/bin/env python3
#
# Copyright (c)  2023  Xiaomi Corporation

"""
This file demonstrates how to use sherpa-onnx Python API to generate audio
from text, i.e., text-to-speech.

Usage:

1. Download a model

wget https://huggingface.co/csukuangfj/vits-ljs/resolve/main/vits-ljs.onnx
wget https://huggingface.co/csukuangfj/vits-ljs/resolve/main/lexicon.txt
wget https://huggingface.co/csukuangfj/vits-ljs/resolve/main/tokens.txt

python3 ./python-api-examples/offline-tts.py \
  --vits-model=./vits-ljs.onnx \
  --vits-lexicon=./lexicon.txt \
  --vits-tokens=./tokens.txt \
  --output-filename=./generated.wav \
  'liliana, the most beautiful and lovely assistant of our team!'
"""

import argparse

import sherpa_onnx
import soundfile as sf


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--vits-model",
        type=str,
        help="Path to vits model.onnx",
    )

    parser.add_argument(
        "--vits-lexicon",
        type=str,
        help="Path to lexicon.txt",
    )

    parser.add_argument(
        "--vits-tokens",
        type=str,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--output-filename",
        type=str,
        default="./generated.wav",
        help="Path to save generated wave",
    )

    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="True to show debug messages",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="valid values: cpu, cuda, coreml",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads for neural network computation",
    )

    parser.add_argument(
        "text",
        type=str,
        help="The input text to generate audio for",
    )

    return parser.parse_args()


def main():
    args = get_args()
    print(args)

    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=args.vits_model,
                lexicon=args.vits_lexicon,
                tokens=args.vits_tokens,
            ),
            provider=args.provider,
            debug=args.debug,
            num_threads=args.num_threads,
        )
    )
    tts = sherpa_onnx.OfflineTts(tts_config)
    audio = tts.generate(args.text)
    sf.write(
        args.output_filename,
        audio.samples,
        samplerate=audio.sample_rate,
        subtype="PCM_16",
    )
    print(f"Saved to {args.output_filename}")
    print(f"The text is '{args.text}'")


if __name__ == "__main__":
    main()
