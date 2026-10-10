#!/usr/bin/env python3
# Copyright (c)  2026  Silvio Tomatis

"""
This file shows how to use sherpa-onnx Python API for
offline/non-streaming speaker diarization with Nemotron-3-Diarization,
an end-to-end streaming Sortformer model from NVIDIA.

It needs neither a speaker embedding model nor clustering. It supports up
to 8 speakers, numbered in the order of their first arrival.

Usage:

Step 1: Download the model

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-nemotron-3-diarization.tar.bz2
  tar xvf sherpa-onnx-nemotron-3-diarization.tar.bz2
  rm sherpa-onnx-nemotron-3-diarization.tar.bz2

Step 2. Download test wave files

Please visit https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models
for a list of available test wave files. The following is an example

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

Step 3. Run it

    python3 ./python-api-examples/offline-speaker-diarization-sortformer.py

"""
from pathlib import Path

import librosa
import sherpa_onnx
import soundfile as sf


def init_speaker_diarization():
    model = "./sherpa-onnx-nemotron-3-diarization/model.int8.onnx"

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            sortformer=sherpa_onnx.OfflineSpeakerSegmentationSortformerModelConfig(
                model=model, threshold=0.5
            ),
            num_threads=2,
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError(
            "Please check your config and make sure all required files exist"
        )

    return sherpa_onnx.OfflineSpeakerDiarization(config)


def progress_callback(num_processed_chunk: int, num_total_chunks: int) -> int:
    progress = num_processed_chunk / num_total_chunks * 100
    print(f"Progress: {progress:.3f}%")
    return 0


def main():
    wave_filename = "./0-four-speakers-zh.wav"
    if not Path(wave_filename).is_file():
        raise RuntimeError(f"{wave_filename} does not exist")

    audio, sample_rate = sf.read(wave_filename, dtype="float32", always_2d=True)
    audio = audio[:, 0]  # only use the first channel

    sd = init_speaker_diarization()

    if sample_rate != sd.sample_rate:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=sd.sample_rate)

    result = sd.process(audio, callback=progress_callback).sort_by_start_time()

    for r in result:
        print(f"{r.start:.3f} -- {r.end:.3f} speaker_{r.speaker:02}")


if __name__ == "__main__":
    main()
