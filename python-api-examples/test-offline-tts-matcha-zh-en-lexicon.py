#!/usr/bin/env python3
"""
Test a Matcha Chinese+English TTS model using lexicon-based phonemization
(no espeak-ng data, no external phonemizer needed).

If you don't want to use a lexicon, please see
test-offline-tts-matcha-zh-en-phonemize.py which uses pypinyin + piper_phonemize instead.

Passes two lexicon files (comma-separated): the zh lexicon from the model
directory and the en lexicon (lexicon-en-us.txt).

Supported extra parameters in gen_config.extra (all optional):
  - min_words_in_sentence, int, default 5.
    Merge adjacent sentences if the number of words is less than this.
    Each CJK character counts as one word; English words are space-separated.
  - max_words_in_sentence, int, default 20.
    Split a sentence into chunks if the number of words exceeds this.
    Splits at punctuation or space boundaries.
  - debug, int, default from model config.

Download model, vocoder, and lexicon:
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-en.tar.bz2
  tar xf matcha-icefall-zh-en.tar.bz2

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-16khz-univ.onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/lexicon-en-us.txt
"""

import time

import sherpa_onnx
import soundfile as sf

tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
            acoustic_model="./matcha-icefall-zh-en/model-steps-3.onnx",
            vocoder="./vocos-16khz-univ.onnx",
            tokens="./matcha-icefall-zh-en/tokens.txt",
            lexicon="./matcha-icefall-zh-en/lexicon.txt,./lexicon-en-us.txt",
        ),
        debug=True,
    ),
    rule_fsts="./matcha-icefall-zh-en/phone-zh.fst,./matcha-icefall-zh-en/date-zh.fst,./matcha-icefall-zh-en/number-zh.fst",
)
tts = sherpa_onnx.OfflineTts(tts_config)

text = (
    "我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。"
    "在这次vocation中，我们计划去Paris欣赏埃菲尔铁塔和卢浮宫的美景。"
    "某某银行的副行长和一些行政领导表示，他们去过长江和长白山; 经济不断增长。"
    "开始数字测试。2025年12月4号，拨打110或者189202512043。123456块钱。"
    "在这个快速发展的时代，人工智能技术正在改变我们的生活方式。"
    "语音合成作为人工智能的重要应用之一，让机器能够用自然流畅的语音与人类进行交流。"
)

gen_config = sherpa_onnx.GenerationConfig()
gen_config.sid = 0
gen_config.speed = 1.0

start = time.time()
audio = tts.generate(text, gen_config)
end = time.time()

assert len(audio.samples) > 0, "No audio generated!"
assert audio.sample_rate > 0, "Invalid sample rate!"

filename = "test-matcha-zh-en-lexicon.wav"
sf.write(filename, audio.samples, samplerate=audio.sample_rate)

elapsed_seconds = end - start
audio_duration = len(audio.samples) / audio.sample_rate
rtf = elapsed_seconds / audio_duration

print(f"Saved to {filename}")
print(f"Elapsed seconds: {elapsed_seconds:.3f}")
print(f"Audio duration in seconds: {audio_duration:.3f}")
print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {rtf:.3f}")
