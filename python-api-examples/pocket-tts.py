#!/usr/bin/env python3
#
# Copyright (c)  2026  Xiaomi Corporation

"""
This file demonstrates how to use sherpa-onnx Python API
for voice cloning using PocketTTS.


Different from ./pocket-tts-play.py, this file does not play back the
generated audio.

Usage:

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
tar xvf sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
rm sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2

python3 ./pocket-tts.py

You can find more models at
https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models

Please see
https://k2-fsa.github.io/sherpa/onnx/tts/pocket.html
for details.

"""

import time
from pathlib import Path

import librosa
import sherpa_onnx
import soundfile as sf


def create_tts():
    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            pocket=sherpa_onnx.OfflineTtsPocketModelConfig(
                lm_flow="./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_flow.int8.onnx",
                lm_main="./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_main.int8.onnx",
                encoder="./sherpa-onnx-pocket-tts-int8-2026-01-26/encoder.onnx",
                decoder="./sherpa-onnx-pocket-tts-int8-2026-01-26/decoder.int8.onnx",
                text_conditioner="./sherpa-onnx-pocket-tts-int8-2026-01-26/text_conditioner.onnx",
                vocab_json="./sherpa-onnx-pocket-tts-int8-2026-01-26/vocab.json",
                token_scores_json="./sherpa-onnx-pocket-tts-int8-2026-01-26/token_scores.json",
            ),
            debug=True,
            num_threads=2,
            provider="cpu",
        )
    )
    if not tts_config.validate():
        raise ValueError(
            "Please read the previous error messages and re-check your config"
        )

    return sherpa_onnx.OfflineTts(tts_config)


def main():
    reference_audio_file = "./sherpa-onnx-pocket-tts-int8-2026-01-26/test_wavs/bria.wav"
    if not Path(reference_audio_file).is_file():
        raise ValueError(f"Reference audio {reference_audio_file} does not exist")

    reference_audio, sample_rate = librosa.load(reference_audio_file, sr=16000)

    tts = create_tts()

    text = "I am happy to join with you today in what will go down in history as the greatest demonstration for freedom in the history of our nation."

    gen_config = sherpa_onnx.GenerationConfig()
    gen_config.reference_audio = reference_audio
    gen_config.reference_sample_rate = sample_rate
    gen_config.num_steps = 5

    start = time.time()
    audio = tts.generate(text, gen_config)
    end = time.time()

    if len(audio.samples) == 0:
        print("Error in generating audios. Please read previous error messages.")
        return

    elapsed_seconds = end - start
    audio_duration = len(audio.samples) / audio.sample_rate
    real_time_factor = elapsed_seconds / audio_duration

    output_filename = "./generated.wav"
    sf.write(
        output_filename,
        audio.samples,
        samplerate=audio.sample_rate,
        subtype="PCM_16",
    )
    print(f"Saved to {output_filename}")
    print(f"The text is '{text}'")
    print(f"Elapsed seconds: {elapsed_seconds:.3f}")
    print(f"Audio duration in seconds: {audio_duration:.3f}")
    print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}")


if __name__ == "__main__":
    main()
