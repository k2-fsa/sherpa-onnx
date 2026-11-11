#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors: Fangjun Kuang)
"""
Generate samples for
https://k2-fsa.github.io/sherpa/onnx/tts/all/
"""

import sherpa_onnx
import soundfile as sf

config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
            model="./model.int8.onnx",
            voices="./voices.bin",
            tokens="./tokens.txt",
            data_dir="./espeak-ng-data",
        ),
        num_threads=2,
    ),
    max_num_sentences=1,
)

if not config.validate():
    raise ValueError("Please check your config")

tts = sherpa_onnx.OfflineTts(config)
text = "تحول تقنية تحويل النص إلى كلام الجمل إلى صوت واضح."

audio = tts.generate(text, sid=0, speed=1.0)

sf.write(
    "./hf/kokoro/nabra/mp3/0.mp3",
    audio.samples,
    samplerate=audio.sample_rate,
)
