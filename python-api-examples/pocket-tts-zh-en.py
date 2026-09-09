#!/usr/bin/env python3
#
# Copyright (c)  2026  Xiaomi Corporation

"""
This file demonstrates how to use sherpa-onnx Python API
for voice cloning using PocketTTS ZhEn (Chinese + English).

Different from ./pocket-tts-zh-en-play.py, this file does not play back the
generated audio.

Usage:

Please refer to
https://modelscope.cn/models/dengcunqin/pocket-tts-zh-en
for model files.

python3 ./pocket-tts-zh-en.py

You can find more models at
https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models

"""

import time
from pathlib import Path

import sherpa_onnx
import soundfile as sf


def create_tts():
    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            pocket_zh_en=sherpa_onnx.OfflineTtsPocketZhEnModelConfig(
                step_model="./step_model.onnx",
                step_encoder="./step_encoder.onnx",
                lexicon="./lexicon-zh.txt,./lexicon-en.txt",
            ),
            debug=False,
            num_threads=2,
            provider="cpu",
        ),
        rule_fsts="./date-zh.fst,./phone-zh.fst,./number-zh.fst",
    )
    if not tts_config.validate():
        raise ValueError(
            "Please read the previous error messages and re-check your config"
        )

    return sherpa_onnx.OfflineTts(tts_config)


def main():
    reference_audio_file = "./trump.wav"
    if not Path(reference_audio_file).is_file():
        raise ValueError(f"Reference audio {reference_audio_file} does not exist")

    tts = create_tts()

    reference_audio, sample_rate = sf.read(reference_audio_file, dtype="float32")
    if reference_audio.ndim > 1:
        reference_audio = reference_audio[:, 0]  # only use the first channel

    text = "某某银行的副行长和一些行政领导表示，他们去过长江和长白山; 经济不断增长. How are you doing today? 我很好! Thank you."

    num_iters = 5

    for i in range(num_iters):
        gen_config = sherpa_onnx.GenerationConfig()
        gen_config.reference_audio = reference_audio
        gen_config.reference_sample_rate = sample_rate
        gen_config.extra = {
            "debug": "0",
            "temperature": "0.0",
            "max_char_in_sentence": "200",
            "min_char_in_sentence": "30",
        }

        start = time.time()
        audio = tts.generate(text, gen_config)
        end = time.time()

        if len(audio.samples) == 0:
            print("Error in generating audios. Please read previous error messages.")
            return

        elapsed_seconds = end - start
        audio_duration = len(audio.samples) / audio.sample_rate
        real_time_factor = elapsed_seconds / audio_duration

        output_filename = f"./generated-pocket-zh-en-{i}.wav"
        sf.write(
            output_filename,
            audio.samples,
            samplerate=audio.sample_rate,
            subtype="PCM_16",
        )
        print(f"Iteration {i}")
        print(f"Saved to {output_filename}")
        print(f"The text is '{text}'")
        print(f"Elapsed seconds: {elapsed_seconds:.3f}")
        print(f"Audio duration in seconds: {audio_duration:.3f}")
        print(
            f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}"
        )
        print()


if __name__ == "__main__":
    main()
