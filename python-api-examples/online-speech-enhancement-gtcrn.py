#!/usr/bin/env python3

"""
This file shows how to use the online speech enhancement API with GTCRN.

Please download files used in this script from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
"""

from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf


def create_speech_denoiser():
    model_filename = "./gtcrn_simple.onnx"
    if not Path(model_filename).is_file():
        raise ValueError(
            "Please first download a model from "
            "https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models"
        )

    config = sherpa_onnx.OnlineSpeechDenoiserConfig(
        model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
            gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(
                model=model_filename
            ),
            debug=False,
            num_threads=1,
            provider="cpu",
        )
    )

    if not config.validate():
        print(config)
        raise ValueError("Errors in config. Please check previous error logs")

    return sherpa_onnx.OnlineSpeechDenoiser(config)


def load_audio(filename: str):
    data, sample_rate = sf.read(filename, always_2d=True, dtype="float32")
    samples = np.ascontiguousarray(data[:, 0])
    return samples, sample_rate


def main():
    sd = create_speech_denoiser()
    test_wave = "./speech_with_noise.wav"
    if not Path(test_wave).is_file():
        raise ValueError(
            f"{test_wave} does not exist. You can download it from "
            "https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models"
        )

    samples, sample_rate = load_audio(test_wave)
    frame_shift = sd.frame_shift_in_samples
    output = []

    for start in range(0, len(samples), frame_shift):
        chunk = samples[start : start + frame_shift]
        denoised = sd(chunk, sample_rate)
        output.append(np.asarray(denoised.samples, dtype=np.float32))

    output.append(np.asarray(sd.flush().samples, dtype=np.float32))
    enhanced = np.concatenate(output) if output else np.empty(0, dtype=np.float32)

    sf.write("./enhanced_online_gtcrn.wav", enhanced, sd.sample_rate)
    print("Saved to ./enhanced_online_gtcrn.wav")


if __name__ == "__main__":
    main()
