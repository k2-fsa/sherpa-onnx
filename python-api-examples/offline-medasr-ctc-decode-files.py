#!/usr/bin/env python3

"""
This file shows how to use a non-streaming Google MedASR CTC model from
https://huggingface.co/google/medasr
to decode files.

Please download model files from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models

For instance,

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2
tar xvf sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2
rm sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2
"""

import time
from pathlib import Path

import librosa
import numpy as np
import sherpa_onnx


def create_recognizer():
    model = "./sherpa-onnx-medasr-ctc-en-int8-2025-12-25/model.int8.onnx"
    tokens = "./sherpa-onnx-medasr-ctc-en-int8-2025-12-25/tokens.txt"
    test_wav_0 = "./sherpa-onnx-medasr-ctc-en-int8-2025-12-25/test_wavs/0.wav"
    test_wav_1 = "./sherpa-onnx-medasr-ctc-en-int8-2025-12-25/test_wavs/1.wav"
    test_wav_2 = "./sherpa-onnx-medasr-ctc-en-int8-2025-12-25/test_wavs/2.wav"
    test_wav_3 = "./sherpa-onnx-medasr-ctc-en-int8-2025-12-25/test_wavs/3.wav"
    test_wav_4 = "./sherpa-onnx-medasr-ctc-en-int8-2025-12-25/test_wavs/4.wav"
    test_wav_5 = "./sherpa-onnx-medasr-ctc-en-int8-2025-12-25/test_wavs/5.wav"

    for f in [
        model,
        tokens,
        test_wav_0,
        test_wav_1,
        test_wav_2,
        test_wav_3,
        test_wav_4,
        test_wav_5,
    ]:
        if not Path(f).is_file():
            print(f"{f} does not exist")

            raise ValueError(
                """Please download model files from
                https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
                """
            )
    return (
        sherpa_onnx.OfflineRecognizer.from_medasr_ctc(
            model=model,
            tokens=tokens,
            num_threads=2,
        ),
        test_wav_0,
        test_wav_1,
        test_wav_2,
        test_wav_3,
        test_wav_4,
        test_wav_5,
    )


def load_audio(filename):
    audio, sample_rate = librosa.load(filename, sr=16000)
    assert sample_rate == 16000, sample_rate

    return np.ascontiguousarray(audio)


def decode_single_file(recognizer, filename):
    samples = load_audio(filename)

    start_time = time.time()

    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate=16000, waveform=samples)
    recognizer.decode_stream(stream)

    end_time = time.time()
    elapsed_seconds = end_time - start_time
    audio_duration = len(samples) / 16000
    real_time_factor = elapsed_seconds / audio_duration

    print("---")
    print(filename)
    print(stream.result)
    print(f"Elapsed seconds: {elapsed_seconds:.3f}")
    print(f"Audio duration in seconds: {audio_duration:.3f}")
    print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}")
    print()


def decode_multiple_files(recognizer, filenames):
    streams = []

    start_time = time.time()

    audio_duration = 0

    for filename in filenames:
        samples = load_audio(filename)
        audio_duration += len(samples) / 16000

        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate=16000, waveform=samples)
        streams.append(stream)

    recognizer.decode_streams(streams)

    end_time = time.time()
    elapsed_seconds = end_time - start_time
    real_time_factor = elapsed_seconds / audio_duration

    for name, stream in zip(filenames, streams):
        print("---")
        print(name)
        print(stream.result)
        print()

    print(f"Elapsed seconds: {elapsed_seconds:.3f}")
    print(f"Audio duration in seconds: {audio_duration:.3f}")
    print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}")
    print()
    print()


def main():
    recognizer, *filenames = create_recognizer()

    decode_single_file(recognizer, filenames[0])
    decode_single_file(recognizer, filenames[1])
    decode_multiple_files(recognizer, filenames[2:])


if __name__ == "__main__":
    main()
