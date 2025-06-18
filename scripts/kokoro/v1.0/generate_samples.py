#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)
"""
Generate samples for
https://k2-fsa.github.io/sherpa/onnx/tts/all/
"""

import sherpa_onnx
import soundfile as sf

from generate_voices_bin import speaker2id

config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
            model="./kokoro.onnx",
            voices="./voices.bin",
            tokens="./tokens.txt",
            data_dir="./espeak-ng-data",
            dict_dir="./dict",
            lexicon="./lexicon-zh.txt,./lexicon-us-en.txt",
        ),
        num_threads=2,
        debug=True,
    ),
    rule_fsts="./phone-zh.fst,./date-zh.fst,./number-zh.fst",
    max_num_sentences=1,
)

if not config.validate():
    raise ValueError("Please check your config")

tts = sherpa_onnx.OfflineTts(config)
text = "This model supports both Chinese and English. 小米的核心价值观是什么？答案是真诚热爱！有困难，请拨打110 或者18601200909。I am learning 机器学习. 我在研究 machine learning。What do you think 中英文说的如何呢? 今天是 2025年6月18号."

print("text", text)

for s, i in speaker2id.items():
    print(s, i, len(speaker2id))
    audio = tts.generate(text, sid=i, speed=1.0)

    sf.write(
        f"./hf/kokoro/v1.0/mp3/{i}-{s}.mp3",
        audio.samples,
        samplerate=audio.sample_rate,
    )
