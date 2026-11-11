#!/usr/bin/env python3
"""
Test a piper tts model with pre-tokenized input (GenerationConfig.tokens).

This is the GenerationConfig.tokens counterpart of
./test-offline-tts-piper-phonemize.py: instead of passing
GenerationConfig.phoneme_codepoints (Unicode codepoints) we pass
GenerationConfig.tokens (token strings) and let sherpa-onnx map them to IDs
with the model's tokens.txt.

Install piper_phonemize:
  pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html

Download model:
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
  tar xf vits-piper-en_US-amy-low.tar.bz2
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
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model="./vits-piper-en_US-amy-low/en_US-amy-low.onnx",
            tokens="./vits-piper-en_US-amy-low/tokens.txt",
        ),
        debug=True,
    ),
)
tts = sherpa_onnx.OfflineTts(tts_config)

text = "Today as always, men fall into two groups: slaves and free men."

# The espeak-ng voice for the phonemizer is taken from the model metadata.
# Fall back to en-us if the model has no language information.
lang = tts.lang
if not lang:
    print("Warning: the model has no language information. Falling back to en-us.")
    lang = "en-us"
print("Phonemizer language:", lang)

sentences = phonemize_espeak(text, lang)

# Each sentence is a string of phonemes; each phoneme is one token string.
# sherpa-onnx adds the bos/pad/eos markers itself, exactly as it does for
# phoneme_codepoints, so we pass the bare tokens here.
token_sentences = [[c for c in sentence] for sentence in sentences]
print("tokens", token_sentences)

gen_config = sherpa_onnx.GenerationConfig()
gen_config.sid = 0
gen_config.speed = 1.0
gen_config.tokens = token_sentences

start = time.time()
audio = tts.generate("", gen_config)
end = time.time()

assert len(audio.samples) > 0, "No audio generated!"
assert audio.sample_rate > 0, "Invalid sample rate!"

filename = "test-piper-tokens.wav"
sf.write(filename, audio.samples, samplerate=audio.sample_rate)

elapsed_seconds = end - start
audio_duration = len(audio.samples) / audio.sample_rate
rtf = elapsed_seconds / audio_duration

print(f"Saved to {filename}")
print(f"Elapsed seconds: {elapsed_seconds:.3f}")
print(f"Audio duration in seconds: {audio_duration:.3f}")
print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {rtf:.3f}")
