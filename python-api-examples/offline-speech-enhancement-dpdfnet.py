#!/usr/bin/env python3

"""
This file shows how to use the speech enhancement API with DPDFNet.

Please download DPDFNet models from the sherpa-onnx GitHub release
or the official Hugging Face hub:
https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
https://huggingface.co/Ceva-IP/DPDFNet

Example:

 wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet_baseline.onnx
 wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet2.onnx
 wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet4.onnx
 wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet8.onnx
 wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet2_48khz_hr.onnx
 wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/speech_with_noise.wav

Use 16 kHz DPDFNet models such as `dpdfnet_baseline.onnx`, `dpdfnet2.onnx`,
`dpdfnet4.onnx`, or `dpdfnet8.onnx` for downstream ASR or speech recognition.
Use `dpdfnet2_48khz_hr.onnx` for 48 kHz enhancement output.
"""

import time
from pathlib import Path
from typing import Tuple

import numpy as np
import sherpa_onnx
import soundfile as sf


def create_speech_denoiser():
    model_filename = "./dpdfnet_baseline.onnx"
    if not Path(model_filename).is_file():
        print(f"{model_filename} does not exist")
        raise ValueError(
            "Please first download a DPDFNet model from "
            "the sherpa-onnx GitHub release or the official Hugging Face hub: "
            "https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models or "
            "https://huggingface.co/Ceva-IP/DPDFNet"
        )

    config = sherpa_onnx.OfflineSpeechDenoiserConfig(
        model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
            dpdfnet=sherpa_onnx.OfflineSpeechDenoiserDpdfNetModelConfig(
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
    test_wave = "./speech_with_noise.wav"
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

    output_filename = f"./enhanced_{denoised.sample_rate}.wav"
    sf.write(output_filename, denoised.samples, denoised.sample_rate)
    print(f"Saved to {output_filename}")
    print(f"Elapsed seconds: {elapsed_seconds:.3f}")
    print(f"Audio duration in seconds: {audio_duration:.3f}")
    print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}")


if __name__ == "__main__":
    main()
