#!/usr/bin/env python3
"""
Test Kokoro English TTS model using piper_phonemize
(no espeak-ng data needed).

If you want to use a lexicon instead, please see
test-offline-tts-kokoro-en-lexicon.py.

Install dependencies:
  pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html

Download model:
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2
  tar xf kokoro-en-v0_19.tar.bz2
"""

import time

import sherpa_onnx
import soundfile as sf

try:
    from piper_phonemize import phonemize_espeak
except Exception as ex:
    raise RuntimeError(
        f"{ex}\nPlease run\n"
        "pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html"
    )

tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
            model="./kokoro-en-v0_19/model.onnx",
            voices="./kokoro-en-v0_19/voices.bin",
            tokens="./kokoro-en-v0_19/tokens.txt",
        ),
        debug=True,
    ),
)
tts = sherpa_onnx.OfflineTts(tts_config)

text = "Today as always, men fall into two groups: slaves and free men. Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."
# The espeak-ng voice for the phonemizer is taken from the model metadata.
# Fall back to en-us if the model has no language information.
lang = tts.lang
if not lang:
    print("Warning: the model has no language information. Falling back to en-us.")
    lang = "en-us"
print("Phonemizer language:", lang)

sentences = phonemize_espeak(text, lang)
phoneme_codepoints = [[ord(c) for c in sentence] for sentence in sentences]

gen_config = sherpa_onnx.GenerationConfig()
gen_config.sid = 0
gen_config.speed = 1.0
gen_config.phoneme_codepoints = phoneme_codepoints

start = time.time()
audio = tts.generate("", gen_config)
end = time.time()

assert len(audio.samples) > 0, "No audio generated!"
assert audio.sample_rate > 0, "Invalid sample rate!"

filename = "test-kokoro-en-phonemize.wav"
sf.write(filename, audio.samples, samplerate=audio.sample_rate)

elapsed_seconds = end - start
audio_duration = len(audio.samples) / audio.sample_rate
rtf = elapsed_seconds / audio_duration

print(f"Saved to {filename}")
print(f"Elapsed seconds: {elapsed_seconds:.3f}")
print(f"Audio duration in seconds: {audio_duration:.3f}")
print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {rtf:.3f}")

