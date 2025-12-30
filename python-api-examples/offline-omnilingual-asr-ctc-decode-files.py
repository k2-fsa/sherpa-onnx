#!/usr/bin/env python3

"""
This file shows how to use a non-streaming Omnilingual ASR CTC model from
https://github.com/facebookresearch/omnilingual-asr
to decode files.

Please download model files from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models

For instance,

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2
tar xvf sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2
rm sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2
"""

from pathlib import Path

import numpy as np
import time
import sherpa_onnx
import soundfile as sf


def create_recognizer():
    model = "./sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/model.int8.onnx"
    tokens = "./sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/tokens.txt"
    test_wav_en = "./sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/test_wavs/en.wav"
    test_wav_de = "./sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/test_wavs/de.wav"
    test_wav_fr = "./sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/test_wavs/fr.wav"
    test_wav_es = "./sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/test_wavs/es.wav"

    for f in [model, tokens, test_wav_en, test_wav_de, test_wav_fr, test_wav_es]:
        if not Path(f).is_file():
            print(f"{f} does not exist")

            raise ValueError(
                """Please download model files from
                https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
                """
            )
    return (
        sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
            model=model,
            tokens=tokens,
            num_threads=1,
        ),
        test_wav_en,
        test_wav_de,
        test_wav_fr,
        test_wav_es,
    )


def load_audio(filename):
    audio, sample_rate = sf.read(filename, dtype="float32", always_2d=True)
    audio = audio[:, 0]  # only use the first channel
    if sample_rate != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

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
