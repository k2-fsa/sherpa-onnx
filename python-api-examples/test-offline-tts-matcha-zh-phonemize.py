#!/usr/bin/env python3
"""
Test a Matcha Chinese TTS model using pinyin phonemization
(no espeak-ng data needed).

If you want to use a lexicon instead, please see
test-offline-tts-matcha-zh-lexicon.py.

Uses pypinyin to convert Chinese text to pinyin tokens, then passes them
via GenerationConfig.tokens.

Install dependencies:
  pip install pypinyin kaldifst

Download model and vocoder:
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-baker.tar.bz2
  tar xf matcha-icefall-zh-baker.tar.bz2

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx
"""

import time

import kaldifst
import sherpa_onnx
import soundfile as sf

try:
    from pypinyin import lazy_pinyin, Style
except Exception as ex:
    raise RuntimeError(f"{ex}\nPlease run\npip install pypinyin kaldifst")

text = (
    "某某银行的副行长和一些行政领导表示，他们去过长江和长白山; 经济不断增长。"
    "开始数字测试。2025年12月4号，拨打110或者189202512043。123456块钱。"
    "在这个快速发展的时代，人工智能技术正在改变我们的生活方式。"
    "语音合成作为人工智能的重要应用之一，让机器能够用自然流畅的语音与人类进行交流。"
)

# Normalize text using kaldifst rule FSTs (same as C++ rule_fsts)
rule_fsts_files = [
    "./matcha-icefall-zh-baker/phone.fst",
    "./matcha-icefall-zh-baker/date.fst",
    "./matcha-icefall-zh-baker/number.fst",
]
rule_fsts = [kaldifst.TextNormalizer(fst_path) for fst_path in rule_fsts_files]
for tn in rule_fsts:
    text = tn.normalize(text)

print(f"Normalized text: {text}")

# Convert Chinese text to pinyin token strings
# lazy_pinyin with TONE3 returns: ['dang1', 'ye4', '，', 'xing1', ...]
# Punctuation is included as separate tokens.
pinyin_list = lazy_pinyin(text, style=Style.TONE3)
tokens = []
for py in pinyin_list:
    py = py.strip()
    if not py:
        continue
    # Punctuation tokens: convert Chinese to ASCII
    punct_map = {
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "；": ",",
        "：": ",",
        "、": ",",
        ";": ",",
        ":": ",",
    }
    if py in punct_map:
        tokens.append(punct_map[py])
    else:
        # Use pinyin as-is (zh model expects no tone suffix for neutral tone)
        tokens.append(py)

print(f"Total tokens: {len(tokens)}")

tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
            acoustic_model="./matcha-icefall-zh-baker/model-steps-3.onnx",
            vocoder="./vocos-22khz-univ.onnx",
            tokens="./matcha-icefall-zh-baker/tokens.txt",
        ),
        debug=True,
    ),
)
tts = sherpa_onnx.OfflineTts(tts_config)

gen_config = sherpa_onnx.GenerationConfig()
gen_config.sid = 0
gen_config.speed = 1.0
gen_config.tokens = [tokens]

start = time.time()
audio = tts.generate("", gen_config)
end = time.time()

assert len(audio.samples) > 0, "No audio generated!"
assert audio.sample_rate > 0, "Invalid sample rate!"

filename = "test-matcha-zh-phonemize.wav"
sf.write(filename, audio.samples, samplerate=audio.sample_rate)

elapsed_seconds = end - start
audio_duration = len(audio.samples) / audio.sample_rate
rtf = elapsed_seconds / audio_duration

print(f"Saved to {filename}")
print(f"Elapsed seconds: {elapsed_seconds:.3f}")
print(f"Audio duration in seconds: {audio_duration:.3f}")
print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {rtf:.3f}")
