#!/usr/bin/env python3
"""
Test ZipVoice Chinese+English TTS model using lexicon
(no espeak-ng data needed).

If you don't want to use a lexicon, please see
test-offline-tts-zipvoice-phonemize.py which uses piper_phonemize instead.

Passes two lexicon files (comma-separated): the zh lexicon from the model
directory and the en lexicon (lexicon-en-us.txt).

Supported extra parameters in gen_config.extra (all optional):
  - min_words_in_sentence, int, default 5.
    Merge adjacent sentences if the number of words is less than this.
    Each CJK character counts as one word; English words are space-separated.
  - max_words_in_sentence, int, default 20.
    Split a sentence into chunks if the number of words exceeds this.
    Splits at punctuation or space boundaries.
  - num_steps, int, default 4.
    Number of flow-matching denoising steps.
  - debug, int, default from model config.

Download model, vocoder, and lexicon:
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2
  tar xf sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos_24khz.onnx

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/lexicon-en-us.txt
"""

import time

import numpy as np
import sherpa_onnx
import soundfile as sf

model_dir = "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia"

tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        zipvoice=sherpa_onnx.OfflineTtsZipvoiceModelConfig(
            tokens=f"{model_dir}/tokens.txt",
            encoder=f"{model_dir}/encoder.int8.onnx",
            decoder=f"{model_dir}/decoder.int8.onnx",
            vocoder="./vocos_24khz.onnx",
            lexicon=f"{model_dir}/lexicon.txt,./lexicon-en-us.txt",
        ),
        debug=True,
    ),
)
tts = sherpa_onnx.OfflineTts(tts_config)

# Load reference audio
ref_wav = f"{model_dir}/test_wavs/news-female.wav"
ref_audio, ref_sr = sf.read(ref_wav)
ref_audio = ref_audio.astype(np.float32)

text = "在这个快速发展的时代，人工智能artificial intelligence技术正在改变我们的生活方式。语音合成speech synthesis作为重要应用之一，让机器能够用自然流畅的语音与人类进行交流。"

gen_config = sherpa_onnx.GenerationConfig()
gen_config.sid = 0
gen_config.speed = 1.0
gen_config.reference_audio = ref_audio.tolist()
gen_config.reference_sample_rate = ref_sr
gen_config.reference_text = "各位村民, 大家新年好! 近期, 湖北省武汉市等多个地区"

start = time.time()
audio = tts.generate(text, gen_config)
end = time.time()

assert len(audio.samples) > 0, "No audio generated!"
assert audio.sample_rate > 0, "Invalid sample rate!"

filename = "test-zipvoice-lexicon.wav"
sf.write(filename, audio.samples, samplerate=audio.sample_rate)

elapsed_seconds = end - start
audio_duration = len(audio.samples) / audio.sample_rate
rtf = elapsed_seconds / audio_duration

print(f"Saved to {filename}")
print(f"Elapsed seconds: {elapsed_seconds:.3f}")
print(f"Audio duration in seconds: {audio_duration:.3f}")
print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {rtf:.3f}")