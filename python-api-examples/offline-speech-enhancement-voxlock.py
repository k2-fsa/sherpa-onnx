#!/usr/bin/env python3

"""
This file shows how to use the speech enhancement API with VoxLock.

VoxLock is a target-speaker voice isolation model. It keeps the voice of one
speaker, e.g., the caller in a phone call, and removes the other voices.

The model is from https://github.com/arbazsiddiqui/voxlock

This script does not provide a voice sample, so VoxLock takes the loudest
3 seconds in the first 15 seconds of the input as the speaker to keep.

If you have a voice sample of the speaker, please see
./offline-speech-enhancement-voxlock-with-voice-sample.py instead.

Please download files used in this script from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models

Example:

 wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/sherpa-onnx-voxlock-int8.tar.bz2
 tar xvf sherpa-onnx-voxlock-int8.tar.bz2
"""

import time
from pathlib import Path
from typing import Tuple

import numpy as np
import sherpa_onnx
import soundfile as sf


def create_speech_denoiser():
    enroll_model = "./sherpa-onnx-voxlock-int8/enroll.onnx"
    chunk_model = "./sherpa-onnx-voxlock-int8/chunk_int8.onnx"
    if not Path(enroll_model).is_file() or not Path(chunk_model).is_file():
        raise ValueError(
            "Please first download the models from "
            "https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models"
        )

    config = sherpa_onnx.OfflineSpeechDenoiserConfig(
        model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
            voxlock=sherpa_onnx.OfflineSpeechDenoiserVoxLockModelConfig(
                enroll_model=enroll_model,
                chunk_model=chunk_model,
                voice_sample="",
            ),
            debug=False,
            num_threads=1,
            provider="cpu",
        )
    )
    if not config.validate():
        print(config)
        raise ValueError("Errors in config. Please check previous error logs")
    return sherpa_onnx.OfflineSpeechDenoiser(config)


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def main():
    sd = create_speech_denoiser()
    test_wave = "./sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav"
    if not Path(test_wave).is_file():
        raise ValueError(
            f"{test_wave} does not exist. You can download it from "
            "https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models"
        )

    samples, sample_rate = load_audio(test_wave)

    start = time.time()
    denoised = sd(samples, sample_rate)
    end = time.time()

    elapsed_seconds = end - start
    audio_duration = len(samples) / sample_rate
    real_time_factor = elapsed_seconds / audio_duration

    sf.write("./voxlock-enhanced-16k.wav", denoised.samples, denoised.sample_rate)
    print("Saved to ./voxlock-enhanced-16k.wav")
    print(f"Elapsed seconds: {elapsed_seconds:.3f}")
    print(f"Audio duration in seconds: {audio_duration:.3f}")
    print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}")


if __name__ == "__main__":
    main()
