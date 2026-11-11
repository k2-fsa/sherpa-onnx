#!/usr/bin/env python3
"""
Test ZipVoice Chinese+English TTS model.

If you want to use a lexicon instead, please see
test-offline-tts-zipvoice-lexicon.py.

Chinese text uses pypinyin for phonemization (initial0 final_tone format,
e.g., "好" -> "h0 ao3").
English text uses piper_phonemize (espeak-ng) for IPA phonemization.

Supported extra parameters in gen_config.extra (all optional):
  - min_words_in_sentence, int, default 5.
    Merge adjacent sentences if the number of words is less than this.
  - max_words_in_sentence, int, default 20.
    Split a sentence into chunks if the number of words exceeds this.
  - num_steps, int, default 4.
    Number of flow-matching denoising steps.
  - debug, int, default from model config.

Install dependencies:
  pip install pypinyin
  pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html

Download model:
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2
  tar xf sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos_24khz.onnx
"""

import re
import time

import numpy as np
import sherpa_onnx
import soundfile as sf

try:
    from pypinyin import Style, lazy_pinyin
    from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials
except Exception as ex:
    raise RuntimeError(f"{ex}\nPlease run\npip install pypinyin")

try:
    from piper_phonemize import phonemize_espeak
except Exception as ex:
    raise RuntimeError(
        f"{ex}\nPlease run\n"
        "pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html"
    )


def get_initial_final(token):
    """Convert a pinyin token to ZipVoice format: initial0 final_tone.

    Example: "hao3" -> "h0 ao3", "de" -> "d0 e5"
    """
    initial = to_initials(token, strict=False)
    final = to_finals_tone3(token, strict=False, neutral_tone_with_five=True)

    ans = ""
    if initial:
        ans = initial + "0"
    if final:
        ans += f" {final}"
    return ans


def tokenize_mixed(text, lang):
    """Tokenize mixed Chinese+English text for ZipVoice.

    Chinese: pypinyin -> initial0 final_tone format
    English: piper_phonemize -> IPA phonemes
    """
    # Replace Chinese punctuations with ASCII equivalents
    for zh, en in [("，", ","), ("。", "."), ("！", "!"), ("？", "?"),
                   ("；", ";"), ("：", ":"), ("、", ",")]:
        text = text.replace(zh, en)

    # Split into: Chinese chars | punctuation | English/ASCII text
    segments = re.findall(
        r"[一-鿿]+"
        r"|[,.\!?;:]"
        r"|[a-zA-Z0-9 ]+",
        text,
    )

    all_tokens = []
    for seg in segments:
        if re.match(r"[一-鿿]+", seg):
            # Chinese segment: convert each character to initial0 final
            for ch in seg:
                py = lazy_pinyin(ch, style=Style.TONE3,
                                 tone_sandhi=True,
                                 neutral_tone_with_five=True)
                if py:
                    initial_final = get_initial_final(py[0])
                    for token in initial_final.split():
                        all_tokens.append(token)
        elif re.match(r"[,.\!?;:]", seg):
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


model_dir = "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia"

tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        zipvoice=sherpa_onnx.OfflineTtsZipvoiceModelConfig(
            tokens=f"{model_dir}/tokens.txt",
            encoder=f"{model_dir}/encoder.int8.onnx",
            decoder=f"{model_dir}/decoder.int8.onnx",
            vocoder="./vocos_24khz.onnx",
        ),
        debug=True,
    ),
)
tts = sherpa_onnx.OfflineTts(tts_config)

# Load reference audio
ref_wav = f"{model_dir}/test_wavs/news-female.wav"
ref_audio, ref_sr = sf.read(ref_wav)
ref_audio = ref_audio.astype(np.float32)

ref_text = "各位村民, 大家新年好! 近期, 湖北省武汉市等多个地区"
text = "在这个快速发展的时代，人工智能artificial intelligence技术正在改变我们的生活方式。语音合成speech synthesis作为重要应用之一，让机器能够用自然流畅的语音与人类进行交流。"

# Tokenize reference text and generated text separately
# The espeak-ng voice for English segments is taken from the model metadata.
# Fall back to en-us if the model has no language information.
lang = tts.lang
if not lang:
    print("Warning: the model has no language information. Falling back to en-us.")
    lang = "en-us"
print("Phonemizer language:", lang)

ref_tokens = tokenize_mixed(ref_text, lang)
gen_tokens = tokenize_mixed(text, lang)

gen_config = sherpa_onnx.GenerationConfig()
gen_config.sid = 0
gen_config.speed = 1.0
# tokens[0] = reference text tokens, tokens[1..] = generated text tokens
gen_config.tokens = ref_tokens + gen_tokens
gen_config.reference_audio = ref_audio.tolist()
gen_config.reference_sample_rate = ref_sr
gen_config.reference_text = ref_text

print(f"Reference text: {ref_text}")
print(f"Synthesis text: {text}")

start = time.time()
audio = tts.generate("", gen_config)
end = time.time()

assert len(audio.samples) > 0, "No audio generated!"
assert audio.sample_rate > 0, "Invalid sample rate!"

filename = "test-zipvoice-phonemize.wav"
sf.write(filename, audio.samples, samplerate=audio.sample_rate)

elapsed_seconds = end - start
audio_duration = len(audio.samples) / audio.sample_rate
rtf = elapsed_seconds / audio_duration

print(f"Saved to {filename}")
print(f"Elapsed seconds: {elapsed_seconds:.3f}")
print(f"Audio duration in seconds: {audio_duration:.3f}")
print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {rtf:.3f}")
