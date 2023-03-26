#!/usr/bin/env python3

# Real-time speech recognition from a microphone with sherpa-onnx Python API
#
# Please refer to
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
# to download pre-trained models

import argparse
import sys
from pathlib import Path

try:
    import sounddevice as sd
except ImportError as e:
    print("Please install sounddevice first. You can use")
    print()
    print("  pip install sounddevice")
    print()
    print("to install it")
    sys.exit(-1)

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
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="Valid values are greedy_search and modified_beam_search",
    )

    return parser.parse_args()


def create_recognizer():
    args = get_args()
    assert_file_exists(args.encoder)
    assert_file_exists(args.decoder)
    assert_file_exists(args.joiner)
    assert_file_exists(args.tokens)
    # Please replace the model files if needed.
    # See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
    # for download links.
    recognizer = sherpa_onnx.OnlineRecognizer(
        tokens=args.tokens,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        decoding_method=args.decoding_method,
    )
    return recognizer


def main():
    recognizer = create_recognizer()
    print("Started! Please speak")

    # The model is using 16 kHz, we use 48 kHz here to demonstrate that
    # sherpa-onnx will do resampling inside.
    sample_rate = 48000
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms
    last_result = ""
    stream = recognizer.create_stream()
    display = sherpa_onnx.Display(max_word_per_line=40)
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)
            stream.accept_waveform(sample_rate, samples)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            result = recognizer.get_result(stream)
            if last_result != result:
                last_result = result
                display.print(-1, result)


if __name__ == "__main__":
    devices = sd.query_devices()
    print(devices)
    default_input_device_idx = sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')

    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
