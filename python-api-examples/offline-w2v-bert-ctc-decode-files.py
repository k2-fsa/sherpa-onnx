#!/usr/bin/env python3

"""
This file shows how to use a non-streaming w2v-BERT 2.0 CTC model
exported by scripts/w2v-bert/export_onnx.py to decode files.

Please download model files from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models

For instance,

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20.tar.bz2
tar xvf sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20.tar.bz2
rm sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20.tar.bz2
"""

import time
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf

d = "./sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20"


def create_recognizer():
    model = f"{d}/model.int8.onnx"
    tokens = f"{d}/tokens.txt"
    test_wavs = [f"{d}/test_wavs/{i}.wav" for i in range(4)]

    for f in [model, tokens, *test_wavs]:
        if not Path(f).is_file():
            print(f"{f} does not exist")

            raise ValueError("""Please download model files from
                https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
                """)
    return (
        sherpa_onnx.OfflineRecognizer.from_w2v_bert_ctc(
            model=model,
            tokens=tokens,
            num_threads=2,
        ),
        *test_wavs,
    )


def load_audio(filename):
    audio, sample_rate = sf.read(filename, dtype="float32", always_2d=True)
    audio = audio[:, 0]  # only use the first channel

    return np.ascontiguousarray(audio), sample_rate


def decode_single_file(recognizer, filename):
    samples, sample_rate = load_audio(filename)

    start_time = time.time()

    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
    recognizer.decode_stream(stream)

    end_time = time.time()
    elapsed_seconds = end_time - start_time
    audio_duration = len(samples) / sample_rate
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
        samples, sample_rate = load_audio(filename)
        audio_duration += len(samples) / sample_rate

        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
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
