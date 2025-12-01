#!/usr/bin/env python3
#
# Copyright (c)  2025  Xiaomi Corporation

"""
This file demonstrates how to use sherpa-onnx Python APIs
with VAD and non-streaming Paraformer for real-time speech recognition
from a microphone.

Usage:


wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-int8-2025-10-07.tar.bz2
tar xvf sherpa-onnx-paraformer-zh-int8-2025-10-07.tar.bz2

./python-api-examples/simulate-streaming-paraformer-microphone.py  \
  --silero-vad-model=./silero_vad.onnx \
  --paraformer=./sherpa-onnx-paraformer-zh-int8-2025-10-07/model.int8.onnx \
  --tokens=./sherpa-onnx-paraformer-zh-int8-2025-10-07/tokens.txt
"""
import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice first. You can use")
    print()
    print("  pip install sounddevice")
    print()
    print("to install it")
    sys.exit(-1)

import sherpa_onnx

killed = False
recording_thread = None
sample_rate = 16000  # Please don't change it

# buffer saves audio samples to be played
samples_queue = queue.Queue()


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--silero-vad-model",
        type=str,
        required=True,
        help="Path to silero_vad.onnx",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--paraformer",
        default="",
        type=str,
        help="Path to the model.onnx from Paraformer",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=2,
        help="Number of threads for neural network computation",
    )

    parser.add_argument(
        "--hr-lexicon",
        type=str,
        default="",
        help="If not empty, it is the lexicon.txt for homophone replacer",
    )

    parser.add_argument(
        "--hr-rule-fsts",
        type=str,
        default="",
        help="If not empty, it is the replace.fst for homophone replacer",
    )

    return parser.parse_args()


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html to download it"
    )


def create_recognizer(args) -> sherpa_onnx.OfflineRecognizer:
    assert_file_exists(args.paraformer)
    recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=args.paraformer,
        tokens=args.tokens,
        num_threads=args.num_threads,
        debug=False,
        hr_rule_fsts=args.hr_rule_fsts,
        hr_lexicon=args.hr_lexicon,
    )

    return recognizer


def start_recording():
    # You can use any value you like for samples_per_read
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms

    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while not killed:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)
            samples = np.copy(samples)
            samples_queue.put(samples)


def main():
    devices = sd.query_devices()
    if len(devices) == 0:
        print("No microphone devices found")
        sys.exit(0)

    print(devices)

    # If you want to select a different input device, please use
    # sd.default.device[0] = xxx
    # where xxx is the device number

    default_input_device_idx = sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')

    args = get_args()
    assert_file_exists(args.tokens)
    assert_file_exists(args.silero_vad_model)

    assert args.num_threads > 0, args.num_threads

    print("Creating recognizer. Please wait...")
    recognizer = create_recognizer(args)

    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = args.silero_vad_model
    config.silero_vad.threshold = 0.5
    config.silero_vad.min_silence_duration = 0.1  # seconds
    config.silero_vad.min_speech_duration = 0.25  # seconds
    # If the current segment is larger than this value, then it increases
    # the threshold to 0.9 internally. After detecting this segment,
    # it resets the threshold to its original value.
    config.silero_vad.max_speech_duration = 8  # seconds
    config.sample_rate = sample_rate

    window_size = config.silero_vad.window_size

    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=100)

    print("Started! Please speak")

    buffer = []

    global recording_thread
    recording_thread = threading.Thread(target=start_recording)
    recording_thread.start()

    display = sherpa_onnx.Display()

    started = False
    started_time = None

    offset = 0
    while not killed:
        samples = samples_queue.get()  # a blocking read

        buffer = np.concatenate([buffer, samples])
        while offset + window_size < len(buffer):
            vad.accept_waveform(buffer[offset : offset + window_size])
            if not started and vad.is_speech_detected():
                started = True
                started_time = time.time()
            offset += window_size

        if not started:
            if len(buffer) > 10 * window_size:
                offset -= len(buffer) - 10 * window_size
                buffer = buffer[-10 * window_size :]

        if started and time.time() - started_time > 0.2:
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, buffer)
            recognizer.decode_stream(stream)
            text = stream.result.text.strip()
            if text:
                display.update_text(text)
                display.display()

            started_time = time.time()

        while not vad.empty():
            # In general, this while loop is executed only once
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, vad.front.samples)

            vad.pop()
            recognizer.decode_stream(stream)

            text = stream.result.text.strip()

            display.update_text(text)

            buffer = []
            offset = 0
            started = False
            started_time = None

            display.finalize_current_sentence()
            display.display()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        killed = True
        if recording_thread:
            recording_thread.join()
        print("\nCaught Ctrl + C. Exiting")
