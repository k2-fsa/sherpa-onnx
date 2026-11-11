#!/usr/bin/env python3
"""
Test a Matcha English TTS model using lexicon-based phonemization
(no espeak-ng data, no external phonemizer needed).

If you don't want to use a lexicon, please see
test-offline-tts-matcha-en-phonemize.py which uses piper_phonemize instead.

Supported extra parameters in gen_config.extra (all optional):
  - min_words_in_sentence, int, default 5.
    Merge adjacent sentences if the number of words is less than this.
    Each CJK character counts as one word; English words are space-separated.
  - max_words_in_sentence, int, default 20.
    Split a sentence into chunks if the number of words exceeds this.
    Splits at punctuation or space boundaries.
  - max_codepoints_in_sentence, int, default 200.
    For the phoneme_codepoints path: split a phoneme sequence if it exceeds
    this many codepoints. Splits at space boundaries.
  - debug, int, default from model config.

Download model, vocoder, and lexicon:
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2
  tar xf matcha-icefall-en_US-ljspeech.tar.bz2

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/lexicon-en-us.txt
"""

import time

import sherpa_onnx
import soundfile as sf

tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
            acoustic_model="./matcha-icefall-en_US-ljspeech/model-steps-3.onnx",
            vocoder="./vocos-22khz-univ.onnx",
            tokens="./matcha-icefall-en_US-ljspeech/tokens.txt",
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

filename = "test-matcha-en-lexicon.wav"
sf.write(filename, audio.samples, samplerate=audio.sample_rate)

elapsed_seconds = end - start
audio_duration = len(audio.samples) / audio.sample_rate
rtf = elapsed_seconds / audio_duration

print(f"Saved to {filename}")
print(f"Elapsed seconds: {elapsed_seconds:.3f}")
print(f"Audio duration in seconds: {audio_duration:.3f}")
print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {rtf:.3f}")
