#!/usr/bin/env python3

"""Decode Uzbek speech with the offline GigaAM NeMo CTC model.

See gigaam-uzbek.md in this directory for model downloads and usage.
"""

import argparse
from pathlib import Path

import sherpa_onnx
import soundfile as sf


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("sherpa-onnx-gigaam-uzbek-600m"),
        help="Directory containing model.int8.onnx and tokens.txt",
    )
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("sound_files", type=Path, nargs="+")
    args = parser.parse_args()
    if args.num_threads < 1:
        parser.error("--num-threads must be positive")
    for filename in [
        args.model_dir / "model.int8.onnx",
        args.model_dir / "tokens.txt",
        *args.sound_files,
    ]:
        if not filename.is_file():
            parser.error(f"File not found: {filename}. See gigaam-uzbek.md.")
    return args


def main():
    args = get_args()
    recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=str(args.model_dir / "model.int8.onnx"),
        tokens=str(args.model_dir / "tokens.txt"),
        num_threads=args.num_threads,
        sample_rate=16000,
        decoding_method="greedy_search",
        provider="cpu",
        debug=args.debug,
    )

    for filename in args.sound_files:
        samples, sample_rate = sf.read(
            str(filename), dtype="float32", always_2d=True
        )
        if len(samples) == 0:
            raise ValueError(f"Audio file is empty: {filename}")
        # Use the first channel, as in the other file decoding examples.
        samples = samples[:, 0].copy()
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        recognizer.decode_stream(stream)
        print(filename)
        print(stream.result.text)


if __name__ == "__main__":
    main()
