#!/usr/bin/env python3

"""
This file demonstrates how to use sherpa-onnx Python API to recognize
a single file.

Please refer to
https://k2-fsa.github.io/sherpa/onnx/index.html
to install sherpa-onnx and to download the pre-trained models
used in this file.
"""
import argparse
import time
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx


def assert_file_exists(filename: str):
    assert Path(
        filename
    ).is_file(), f"{filename} does not exist!\nPlease refer to https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html to download it"


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--tokens",
        type=str,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--encoder",
        type=str,
        help="Path to the encoder model",
    )

    parser.add_argument(
        "--decoder",
        type=str,
        help="Path to the decoder model",
    )

    parser.add_argument(
        "--joiner",
        type=str,
        help="Path to the joiner model",
    )

    parser.add_argument(
        "--wave-filename",
        type=str,
        help="""Path to the wave filename. Must be 16 kHz,
        mono with 16-bit samples""",
    )

    return parser.parse_args()


def main():
    sample_rate = 16000
    num_threads = 2

    args = get_args()
    assert_file_exists(args.encoder)
    assert_file_exists(args.decoder)
    assert_file_exists(args.joiner)
    assert_file_exists(args.tokens)
    if not Path(args.wave_filename).is_file():
        print(f"{args.wave_filename} does not exist!")
        return

    recognizer = sherpa_onnx.OnlineRecognizer(
        tokens=args.tokens,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        num_threads=num_threads,
        sample_rate=sample_rate,
        feature_dim=80,
    )
    with wave.open(args.wave_filename) as f:
        assert f.getframerate() == sample_rate, f.getframerate()
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)

        samples_float32 = samples_float32 / 32768

    duration = len(samples_float32) / sample_rate

    start_time = time.time()
    print("Started!")

    stream = recognizer.create_stream()

    stream.accept_waveform(sample_rate, samples_float32)

    tail_paddings = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
    stream.accept_waveform(sample_rate, tail_paddings)

    stream.input_finished()

    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)

    print(recognizer.get_result(stream))

    print("Done!")
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    rtf = elapsed_seconds / duration
    print(f"num_threads: {num_threads}")
    print(f"Wave duration: {duration:.3f} s")
    print(f"Elapsed time: {elapsed_seconds:.3f} s")
    print(f"Real time factor (RTF): {elapsed_seconds:.3f}/{duration:.3f} = {rtf:.3f}")


if __name__ == "__main__":
    main()
