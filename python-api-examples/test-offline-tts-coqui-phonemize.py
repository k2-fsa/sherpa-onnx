#!/usr/bin/env python3
"""
Test OfflineTtsVitsExtImpl with a coqui model (uses character frontend).

Download model:
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-coqui-de-css10.tar.bz2
  tar xf vits-coqui-de-css10.tar.bz2
"""

import time

import sherpa_onnx
import soundfile as sf

tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model="./vits-coqui-de-css10/model.onnx",
            tokens="./vits-coqui-de-css10/tokens.txt",
        ),
        debug=True,
    ),
)
tts = sherpa_onnx.OfflineTts(tts_config)

text = "Alles hat ein Ende, nur die Wurst hat zwei."

gen_config = sherpa_onnx.GenerationConfig()
gen_config.sid = 0
gen_config.speed = 1.0

start = time.time()
audio = tts.generate(text, gen_config)
end = time.time()

assert len(audio.samples) > 0, "No audio generated!"
assert audio.sample_rate > 0, "Invalid sample rate!"

filename = "test-coqui-phonemize.wav"
sf.write(filename, audio.samples, samplerate=audio.sample_rate)

elapsed_seconds = end - start
audio_duration = len(audio.samples) / audio.sample_rate
rtf = elapsed_seconds / audio_duration

print(f"Saved to {filename}")
print(f"Elapsed seconds: {elapsed_seconds:.3f}")
print(f"Audio duration in seconds: {audio_duration:.3f}")
print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {rtf:.3f}")
