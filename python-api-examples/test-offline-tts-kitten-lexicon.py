#!/usr/bin/env python3
"""
Test a Kitten English TTS model using lexicon-based phonemization
(no espeak-ng data, no external phonemizer needed).

If you don't want to use a lexicon, please see
test-offline-tts-kitten-phonemize.py which uses piper_phonemize instead.

Download model and lexicon:
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kitten-micro-en-v0_8.tar.bz2
  tar xf kitten-micro-en-v0_8.tar.bz2

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/lexicon-en-us.txt
"""

import time

import sherpa_onnx
import soundfile as sf

tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        kitten=sherpa_onnx.OfflineTtsKittenModelConfig(
            model="./kitten-micro-en-v0_8/model.onnx",
            voices="./kitten-micro-en-v0_8/voices.bin",
            tokens="./kitten-micro-en-v0_8/tokens.txt",
            lexicon="./lexicon-en-us.txt",
        ),
        debug=True,
    ),
)
tts = sherpa_onnx.OfflineTts(tts_config)

text = "Today as always, men fall into two groups: slaves and free men."

gen_config = sherpa_onnx.GenerationConfig()
gen_config.sid = 0
gen_config.speed = 1.0

start = time.time()
audio = tts.generate(text, gen_config)
end = time.time()

assert len(audio.samples) > 0, "No audio generated!"
assert audio.sample_rate > 0, "Invalid sample rate!"

filename = "test-kitten-lexicon.wav"
sf.write(filename, audio.samples, samplerate=audio.sample_rate)

elapsed_seconds = end - start
audio_duration = len(audio.samples) / audio.sample_rate
rtf = elapsed_seconds / audio_duration

print(f"Saved to {filename}")
print(f"Elapsed seconds: {elapsed_seconds:.3f}")
print(f"Audio duration in seconds: {audio_duration:.3f}")
print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {rtf:.3f}")
