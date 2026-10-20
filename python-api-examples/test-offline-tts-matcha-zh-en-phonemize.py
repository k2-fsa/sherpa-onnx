#!/usr/bin/env python3
"""
Test a Matcha Chinese+English TTS model using pinyin + espeak phonemization
(no espeak-ng data needed at runtime).

If you want to use a lexicon instead, please see
test-offline-tts-matcha-zh-en-lexicon.py.

Uses pypinyin for Chinese text and piper_phonemize for English text,
then passes combined tokens via GenerationConfig.tokens.

Supported extra parameters in gen_config.extra (all optional):
  - min_words_in_sentence, int, default 5.
    Merge adjacent sentences if the number of words is less than this.
  - max_words_in_sentence, int, default 20.
    Split a sentence into chunks if the number of words exceeds this.
  - max_codepoints_in_sentence, int, default 200.
    Split a phoneme sequence if it exceeds this many codepoints.
  - debug, int, default from model config.

Install dependencies:
  pip install pypinyin kaldifst
  pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html

Download model and vocoder:
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-en.tar.bz2
  tar xf matcha-icefall-zh-en.tar.bz2

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-16khz-univ.onnx
"""

import re
import time

import sherpa_onnx
import soundfile as sf

import kaldifst

try:
    from pypinyin import lazy_pinyin, Style
except Exception as ex:
    raise RuntimeError(f"{ex}\nPlease run\npip install pypinyin kaldifst")

try:
    from piper_phonemize import phonemize_espeak
except Exception as ex:
    raise RuntimeError(
        f"{ex}\nPlease run\n"
        "pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html"
    )

tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
            acoustic_model="./matcha-icefall-zh-en/model-steps-3.onnx",
            vocoder="./vocos-16khz-univ.onnx",
            tokens="./matcha-icefall-zh-en/tokens.txt",
        ),
        debug=True,
    ),
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

# Normalize text using kaldifst rule FSTs (same as C++ rule_fsts)
# Converts numbers, dates, phone numbers to Chinese characters
rule_fsts = [
    "./matcha-icefall-zh-en/phone-zh.fst",
    "./matcha-icefall-zh-en/date-zh.fst",
    "./matcha-icefall-zh-en/number-zh.fst",
]
for fst_path in rule_fsts:
    tn = kaldifst.TextNormalizer(fst_path)
    text = tn.normalize(text)

print(f"Normalized text: {text}")


def tokenize_mixed(text, lang):
    """Split text into Chinese and non-Chinese segments, tokenize each.

    Returns a list of sentences, each sentence is a list of token strings.
    Chinese pinyin tokens are multi-char (e.g., 'zhong1', 'guo2').
    English IPA tokens are single-char (e.g., 'h', 'ə', 'l').
    Punctuation is preserved as tokens (e.g., ',', '.', '?').
    """
    # Split into: Chinese chars | ASCII+IPA chars | punctuation chars
    segments = re.findall(
        r"[一-鿿]+" r"|[a-zA-Z0-9 ,;:!?\-']+" r"|[，。！？；：、…—]",
        text,
    )

    all_tokens = []
    for seg in segments:
        if re.match(r"[一-鿿]+", seg):
            # Chinese segment: convert to pinyin token strings
            pinyin_list = lazy_pinyin(seg, style=Style.TONE3)
            for py in pinyin_list:
                # Neutral tone characters (的, 着, 了, etc.) get no tone number
                # from pypinyin. Append '5' to match the lexicon convention.
                if not py[-1].isdigit():
                    py += "5"
                all_tokens.append(py)
        elif re.match(r"[，。！？；：、…—]", seg):
            # Chinese punctuation: convert to ASCII equivalent
            punct_map = {
                "，": ",",
                "。": ".",
                "！": "!",
                "？": "?",
                "；": ";",
                "：": ":",
                "、": ",",
                "…": "...",
                "—": "-",
            }
            all_tokens.append(punct_map.get(seg, seg))
        else:
            # English/ASCII segment: use piper_phonemize
            seg = seg.strip()
            if not seg:
                continue
            sentences = phonemize_espeak(seg, lang)
            for sentence in sentences:
                for c in sentence:
                    all_tokens.append(c)

    return [all_tokens]


# The espeak-ng voice for English segments is taken from the model metadata.
# Fall back to en-us if the model has no language information.
lang = tts.lang
if not lang:
    print("Warning: the model has no language information. Falling back to en-us.")
    lang = "en-us"
print("Phonemizer language:", lang)

tokens = tokenize_mixed(text, lang)
print(f"Total tokens: {len(tokens[0])}")

gen_config = sherpa_onnx.GenerationConfig()
gen_config.sid = 0
gen_config.speed = 1.0
gen_config.tokens = tokens

start = time.time()
audio = tts.generate("", gen_config)
end = time.time()

assert len(audio.samples) > 0, "No audio generated!"
assert audio.sample_rate > 0, "Invalid sample rate!"

filename = "test-matcha-zh-en-phonemize.wav"
sf.write(filename, audio.samples, samplerate=audio.sample_rate)

elapsed_seconds = end - start
audio_duration = len(audio.samples) / audio.sample_rate
rtf = elapsed_seconds / audio_duration

print(f"Saved to {filename}")
print(f"Elapsed seconds: {elapsed_seconds:.3f}")
print(f"Audio duration in seconds: {audio_duration:.3f}")
print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {rtf:.3f}")
