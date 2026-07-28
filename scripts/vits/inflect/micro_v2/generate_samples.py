#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors: Fangjun Kuang)
"""
Generate samples for
https://k2-fsa.github.io/sherpa/onnx/tts/all/
"""

import os
from pathlib import Path

import sherpa_onnx
import soundfile as sf

config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model="vits-inflect-en-micro-v2/model.onnx",
            data_dir="vits-inflect-en-micro-v2/espeak-ng-data",
            tokens="vits-inflect-en-micro-v2/tokens.txt",
        ),
        num_threads=2,
    ),
    max_num_sentences=1,
)

if not config.validate():
    raise ValueError("Please check your config")

if not config.validate():
    raise ValueError("Please check your config")

tts = sherpa_onnx.OfflineTts(config)
text = "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."

audio = tts.generate(text, sid=0, speed=1.0)

sf.write(
    "./hf/inflect/vits-inflect-en-micro-v2/mp3/0.mp3",
    audio.samples,
    samplerate=audio.sample_rate,
)
