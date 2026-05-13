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

from gen_calib_configs import SUPPORTED_LANGS, SENTENCES

config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        supertonic=sherpa_onnx.OfflineTtsSupertonicModelConfig(
            duration_predictor="./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/duration_predictor.int8.onnx",
            text_encoder="./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/text_encoder.int8.onnx",
            vector_estimator="./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/vector_estimator.int8.onnx",
            vocoder="./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/vocoder.int8.onnx",
            tts_json="./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/tts.json",
            unicode_indexer="./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/unicode_indexer.bin",
            voice_style="./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/voice.bin",
        ),
        debug=False,
        num_threads=2,
        provider="cpu",
    )
)

if not config.validate():
    raise ValueError("Please check your config")

tts = sherpa_onnx.OfflineTts(config)

model_dir = os.environ.get("KITTEN", "")

for sid in range(tts.num_speakers):
    for lang in SENTENCES:
        sentence_list = SENTENCES[lang]
        for i, text in enumerate(sentence_list):
            gen_config = sherpa_onnx.GenerationConfig()

            # This model has 10 speakers. Valid sid: 0-9
            gen_config.sid = sid
            gen_config.num_steps = 8
            gen_config.speed = 1  # larger -> faster
            gen_config.extra["lang"] = lang

            audio = tts.generate(text, gen_config)

            sf.write(
                f"./hf/supertonic/v3/mp3/sid-{sid}-{lang}-{i}.mp3",
                audio.samples,
                samplerate=audio.sample_rate,
            )
