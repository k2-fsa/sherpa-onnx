#!/usr/bin/env python3
"""
Test a Matcha Chinese TTS model using lexicon-based phonemization
(no espeak-ng data, no external phonemizer needed).

If you don't want to use a lexicon, please see
test-offline-tts-matcha-zh-phonemize.py which uses pypinyin instead.

Supported extra parameters in gen_config.extra (all optional):
  - min_words_in_sentence, int, default 5.
    Merge adjacent sentences if the number of words is less than this.
    Each CJK character counts as one word.
  - max_words_in_sentence, int, default 20.
    Split a sentence into chunks if the number of words exceeds this.
    Splits at punctuation or space boundaries.
  - debug, int, default from model config.

Download model and vocoder:
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-baker.tar.bz2
  tar xf matcha-icefall-zh-baker.tar.bz2

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx
"""

import time

import sherpa_onnx
import soundfile as sf

tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
            acoustic_model="./matcha-icefall-zh-baker/model-steps-3.onnx",
            vocoder="./vocos-22khz-univ.onnx",
            tokens="./matcha-icefall-zh-baker/tokens.txt",
            lexicon="./matcha-icefall-zh-baker/lexicon.txt",
        ),
        debug=True,
    ),
)
tts = sherpa_onnx.OfflineTts(tts_config)

text = "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转。"

gen_config = sherpa_onnx.GenerationConfig()
gen_config.sid = 0
gen_config.speed = 1.0

start = time.time()
audio = tts.generate(text, gen_config)
end = time.time()

assert len(audio.samples) > 0, "No audio generated!"
assert audio.sample_rate > 0, "Invalid sample rate!"

filename = "test-matcha-zh-lexicon.wav"
sf.write(filename, audio.samples, samplerate=audio.sample_rate)

elapsed_seconds = end - start
audio_duration = len(audio.samples) / audio.sample_rate
rtf = elapsed_seconds / audio_duration

print(f"Saved to {filename}")
print(f"Elapsed seconds: {elapsed_seconds:.3f}")
print(f"Audio duration in seconds: {audio_duration:.3f}")
print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {rtf:.3f}")
