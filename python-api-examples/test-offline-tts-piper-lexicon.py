#!/usr/bin/env python3
"""
Test a piper tts model using lexicon-based phonemization
(no espeak-ng data, no external phonemizer needed).

Download model and lexicon:
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
  tar xf vits-piper-en_US-amy-low.tar.bz2

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/lexicon-en-us.txt
"""

import time

import sherpa_onnx
import soundfile as sf

tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model="./vits-piper-en_US-amy-low/en_US-amy-low.onnx",
            tokens="./vits-piper-en_US-amy-low/tokens.txt",
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

filename = "test-piper-lexicon.wav"
sf.write(filename, audio.samples, samplerate=audio.sample_rate)

elapsed_seconds = end - start
audio_duration = len(audio.samples) / audio.sample_rate
rtf = elapsed_seconds / audio_duration

print(f"Saved to {filename}")
print(f"Elapsed seconds: {elapsed_seconds:.3f}")
print(f"Audio duration in seconds: {audio_duration:.3f}")
print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {rtf:.3f}")
