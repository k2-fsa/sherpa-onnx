#!/usr/bin/env python3
#
# Copyright (c)  2026  Xiaomi Corporation

"""
This file demonstrates how to use sherpa-onnx Python API
for SupertonicTTS.


Usage:

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2
tar xvf sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2
rm sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2

python3 ./supertonic-tts.py

You can find more models at
https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models

Please see
https://k2-fsa.github.io/sherpa/onnx/tts/supertonic.html
for details.

"""

import time

import sherpa_onnx
import soundfile as sf


def create_tts():
    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            supertonic=sherpa_onnx.OfflineTtsSupertonicModelConfig(
                duration_predictor="./sherpa-onnx-supertonic-tts-int8-2026-03-06/duration_predictor.int8.onnx",
                text_encoder="./sherpa-onnx-supertonic-tts-int8-2026-03-06/text_encoder.int8.onnx",
                vector_estimator="./sherpa-onnx-supertonic-tts-int8-2026-03-06/vector_estimator.int8.onnx",
                vocoder="./sherpa-onnx-supertonic-tts-int8-2026-03-06/vocoder.int8.onnx",
                tts_json="./sherpa-onnx-supertonic-tts-int8-2026-03-06/tts.json",
                unicode_indexer="./sherpa-onnx-supertonic-tts-int8-2026-03-06/unicode_indexer.bin",
                voice_style="./sherpa-onnx-supertonic-tts-int8-2026-03-06/voice.bin",
            ),
            debug=False,
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
    tts = create_tts()

    text = "Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be, a statesman, a businessman, an official, or a scholar."

    gen_config = sherpa_onnx.GenerationConfig()

    # This model has 10 speakers. Valid sid: 0-9
    gen_config.sid = 6
    gen_config.num_steps = 5
    gen_config.speed = 1.25  # larger -> faster

    # We use en for English.
    # You can also use es, pt, fr, ko.
    # This single model supports 5 languages.
    gen_config.extra["lang"] = "en"

    start = time.time()
    audio = tts.generate(text, gen_config)
    end = time.time()

    if len(audio.samples) == 0:
        print("Error in generating audios. Please read previous error messages.")
        return

    elapsed_seconds = end - start
    audio_duration = len(audio.samples) / audio.sample_rate
    real_time_factor = elapsed_seconds / audio_duration

    output_filename = "./supertonic-en.wav"
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
