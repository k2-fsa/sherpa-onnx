#!/usr/bin/env python3
#
# Copyright (c)  2023  Xiaomi Corporation

"""
This file demonstrates how to use sherpa-onnx Python APIs
with VAD and non-streaming SenseVoice for real-time speech recognition
from a microphone.

Usage:


wget https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx

./python-api-examples/simulate-streaming-sense-voice-microphone.py  \
  --silero-vad-model=./silero_vad.onnx \
  --sense-voice=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx \
  --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt
"""
import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

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
        "--sense-voice",
        default="",
        type=str,
        help="Path to the model.onnx from SenseVoice",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads for neural network computation",
    )

    parser.add_argument(
        "--hr-dict-dir",
        type=str,
        default="",
        help="If not empty, it is the jieba dict directory for homophone replacer",
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
    assert_file_exists(args.sense_voice)
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=args.sense_voice,
        tokens=args.tokens,
        num_threads=args.num_threads,
        use_itn=False,
        debug=False,
        hr_dict_dir=args.hr_dict_dir,
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
    config.silero_vad.min_silence_duration = 0.25
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

    while not killed:
        samples = samples_queue.get()  # a blocking read

        buffer = np.concatenate([buffer, samples])
        offset = 0
        while offset + window_size < samples.shape[0]:
            vad.accept_waveform(samples[offset : offset + window_size])
            if not started and vad.is_speech_detected():
                started = True
                started_time = time.time()
            offset += window_size

        if not started:
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
            k = len(display.sentences)

            sf.write(
                f"test/{k}.wav",
                vad.front.samples,
                samplerate=sample_rate,
                subtype="PCM_16",
            )

            vad.pop()
            recognizer.decode_stream(stream)

            text = stream.result.text.strip()

            display.update_text(text)

            buffer = []
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
