#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)
"""
Generate samples for
https://k2-fsa.github.io/sherpa/onnx/tts/all/
"""


import os
from pathlib import Path

import sherpa_onnx
import soundfile as sf

from generate_voices_bin import speaker2id

if Path("./model.fp32.onnx").is_file():
    model = "./model.fp32.onnx"
elif Path("./model.int8.onnx").is_file():
    model = "./model.int8.onnx"
elif Path("./model.onnx").is_file():
    model = "./model.onnx"
else:
    raise RuntimeError("No model files available")

config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        kitten=sherpa_onnx.OfflineTtsKittenModelConfig(
            model=model,
            voices="voices.bin",
            tokens="tokens.txt",
            data_dir="espeak-ng-data",
        ),
        num_threads=2,
    ),
    max_num_sentences=1,
)

if not config.validate():
    raise ValueError("Please check your config")

tts = sherpa_onnx.OfflineTts(config)
text = "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."

model_dir = os.environ.get("KITTEN", "")

for s, i in speaker2id.items():
    print(s, i, len(speaker2id))
    audio = tts.generate(text, sid=i, speed=1.0)

    sf.write(
        f"./hf/kitten/{model_dir}/mp3/{i}-{s}.mp3",
        audio.samples,
        samplerate=audio.sample_rate,
    )
