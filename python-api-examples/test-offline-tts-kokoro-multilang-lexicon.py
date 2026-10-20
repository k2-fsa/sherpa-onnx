#!/usr/bin/env python3
"""
Test Kokoro Chinese+English TTS model using lexicon
(no espeak-ng data needed).

Supported extra parameters in gen_config.extra (all optional):
  - lang, str, Language override (e.g., "en-us", "zh"). Defaults to kokoro.lang.
  - debug, int, default from model config.

Download model:
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2
  tar xf kokoro-multi-lang-v1_0.tar.bz2
"""

import time

import sherpa_onnx
import soundfile as sf

model_dir = "./kokoro-multi-lang-v1_0"

tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=f"{model_dir}/model.onnx",
            voices=f"{model_dir}/voices.bin",
            tokens=f"{model_dir}/tokens.txt",
            lexicon=f"{model_dir}/lexicon-us-en.txt,{model_dir}/lexicon-zh.txt",
        ),
        debug=True,
    ),
    rule_fsts=f"{model_dir}/phone-zh.fst,{model_dir}/date-zh.fst,{model_dir}/number-zh.fst",
)
tts = sherpa_onnx.OfflineTts(tts_config)

text = "在这个快速发展的时代，人工智能artificial intelligence技术正在改变我们的生活方式。 有困难，请拨打110。"

gen_config = sherpa_onnx.GenerationConfig()
gen_config.sid = 0
gen_config.speed = 1.0

start = time.time()
audio = tts.generate(text, gen_config)
end = time.time()

assert len(audio.samples) > 0, "No audio generated!"
assert audio.sample_rate > 0, "Invalid sample rate!"

filename = "test-kokoro-multilang-lexicon.wav"
sf.write(filename, audio.samples, samplerate=audio.sample_rate)

elapsed_seconds = end - start
audio_duration = len(audio.samples) / audio.sample_rate
rtf = elapsed_seconds / audio_duration

print(f"Saved to {filename}")
print(f"Elapsed seconds: {elapsed_seconds:.3f}")
print(f"Audio duration in seconds: {audio_duration:.3f}")
print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {rtf:.3f}")
