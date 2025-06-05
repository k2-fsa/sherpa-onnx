#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation

"""
This file shows how to use UVR for source separation.

Please first download a UVR model from

https://github.com/k2-fsa/sherpa-onnx/releases/tag/source-separation-models

The following is an example:

    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR_MDXNET_9482.onnx

Please also download a test file

    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/qi-feng-le-zh.wav

The test wav file is 16-bit encoded with 2 channels. If you have other
formats, e.g., .mp4 or .mp3, please first use ffmpeg to convert it.
For instance

    ffmpeg -i your.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 out.wav

Then you can use out.wav as input for this example.
"""

import time
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf


def create_offline_source_separation():
    # Please read the help message at the beginning of this file
    # to download model files
    model = "./UVR_MDXNET_9482.onnx"

    if not Path(model).is_file():
        raise ValueError(f"{model} does not exist.")

    config = sherpa_onnx.OfflineSourceSeparationConfig(
        model=sherpa_onnx.OfflineSourceSeparationModelConfig(
            uvr=sherpa_onnx.OfflineSourceSeparationUvrModelConfig(
                model=model,
            ),
            num_threads=1,
            debug=False,
            provider="cpu",
        )
    )
    if not config.validate():
        raise ValueError("Please check your config.")

    return sherpa_onnx.OfflineSourceSeparation(config)


def load_audio():
    # Please read the help message at the beginning of this file to download
    # the following wav_file
    wav_file = "./qi-feng-le-zh.wav"
    if not Path(wav_file).is_file():
        raise ValueError(f"{wav_file} does not exist")

    samples, sample_rate = sf.read(wav_file, dtype="float32", always_2d=True)
    samples = np.transpose(samples)
    # now samples is of shape (num_channels, num_samples)
    assert (
        samples.shape[1] > samples.shape[0]
    ), f"You should use (num_channels, num_samples). {samples.shape}"

    assert (
        samples.dtype == np.float32
    ), f"Expect np.float32 as dtype. Given: {samples.dtype}"

    return samples, sample_rate


def main():
    sp = create_offline_source_separation()
    samples, sample_rate = load_audio()
    samples = np.ascontiguousarray(samples)

    print("Started. Please wait")
    start = time.time()
    output = sp.process(sample_rate=sample_rate, samples=samples)
    end = time.time()

    print("output.sample_rate", output.sample_rate)

    assert len(output.stems) == 2, len(output.stems)

    vocals = output.stems[0].data
    non_vocals = output.stems[1].data
    # vocals.shape (num_channels, num_samples)

    vocals = np.transpose(vocals)
    non_vocals = np.transpose(non_vocals)

    # vocals.shape (num_samples,num_channels)

    sf.write("./uvr-vocals.wav", vocals, samplerate=output.sample_rate)
    sf.write("./uvr-non-vocals.wav", non_vocals, samplerate=output.sample_rate)

    elapsed_seconds = end - start
    audio_duration = samples.shape[1] / sample_rate
    real_time_factor = elapsed_seconds / audio_duration

    print("Saved to ./uvr-vocals.wav and ./uvr-non-vocals.wav")
    print(f"Elapsed seconds: {elapsed_seconds:.3f}")
    print(f"Audio duration in seconds: {audio_duration:.3f}")
    print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}")


if __name__ == "__main__":
    main()
