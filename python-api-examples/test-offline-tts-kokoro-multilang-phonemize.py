#!/usr/bin/env python3
"""
Test Kokoro Chinese+English TTS model using misaki + piper_phonemize
(no espeak-ng data needed).

Chinese text uses misaki.zh.ZHG2P for IPA phonemization.
English text uses piper_phonemize (espeak-ng) for IPA phonemization.

Install dependencies:
  pip install 'misaki[zh]' kaldifst
  pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html

Download model:
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2
  tar xf kokoro-multi-lang-v1_0.tar.bz2
"""

import re
import time

import kaldifst
import sherpa_onnx
import soundfile as sf

try:
    from misaki import zh
except Exception as ex:
    raise RuntimeError(f"{ex}\nPlease run\npip install 'misaki[zh]' kaldifst")

try:
    from piper_phonemize import phonemize_espeak
except Exception as ex:
    raise RuntimeError(
        f"{ex}\nPlease run\n"
        "pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html"
    )

# Initialize Chinese G2P
_zh_g2p = zh.ZHG2P()


def tokenize_mixed(text, lang):
    """Tokenize mixed Chinese+English text for Kokoro.

    Chinese: misaki zh.ZHG2P -> IPA phonemes (individual characters)
    English: piper_phonemize -> IPA phonemes
    """
    # Replace Chinese punctuations with ASCII equivalents
    for zh_p, en_p in [
        ("，", ","),
        ("。", "."),
        ("！", "!"),
        ("？", "?"),
        ("；", ";"),
        ("：", ":"),
        ("、", ","),
    ]:
        text = text.replace(zh_p, en_p)

    # Split into: Chinese chars | punctuation | English/ASCII text
    segments = re.findall(
        r"[一-鿿]+" r"|[,.\!?;:\-…]" r"|[a-zA-Z0-9 ]+",
        text,
    )

    all_tokens = []
    for seg in segments:
        if re.match(r"[一-鿿]+", seg):
            # Chinese segment: convert to IPA using misaki
            ipa = _zh_g2p.word2ipa(seg)
            # Strip U+032F (COMBINING INVERTED BREVE BELOW), same as
            # generate_lexicon_zh.py does
            ipa = ipa.replace(chr(815), "")
            for c in ipa:
                all_tokens.append(c)
        elif re.match(r"[,.\!?;:\-…]", seg):
            # Punctuation: add directly as ASCII token
            all_tokens.append(seg)
        else:
            # English/ASCII text: use piper_phonemize
            seg = seg.strip()
            if not seg:
                continue
            sentences = phonemize_espeak(seg, lang)
            for sentence in sentences:
                for c in sentence:
                    all_tokens.append(c)

    return [all_tokens]


model_dir = "./kokoro-multi-lang-v1_0"

tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=f"{model_dir}/model.onnx",
            voices=f"{model_dir}/voices.bin",
            tokens=f"{model_dir}/tokens.txt",
        ),
        debug=True,
    ),
)
tts = sherpa_onnx.OfflineTts(tts_config)

text = "在这个快速发展的时代，人工智能artificial intelligence技术正在改变我们的生活方式。语音合成speech synthesis作为重要应用之一。有困难，请拨打110。"

# Normalize text using kaldifst rule FSTs (same as C++ rule_fsts)
# Converts numbers, dates, phone numbers to Chinese characters
rule_fsts_files = [
    f"{model_dir}/phone-zh.fst",
    f"{model_dir}/date-zh.fst",
    f"{model_dir}/number-zh.fst",
]
rule_fsts = [kaldifst.TextNormalizer(fst_path) for fst_path in rule_fsts_files]
for tn in rule_fsts:
    text = tn.normalize(text)

print(f"Normalized text: {text}")

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

filename = "test-kokoro-multilang-phonemize.wav"
sf.write(filename, audio.samples, samplerate=audio.sample_rate)

elapsed_seconds = end - start
audio_duration = len(audio.samples) / audio.sample_rate
rtf = elapsed_seconds / audio_duration

print(f"Saved to {filename}")
print(f"Elapsed seconds: {elapsed_seconds:.3f}")
print(f"Audio duration in seconds: {audio_duration:.3f}")
print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {rtf:.3f}")

