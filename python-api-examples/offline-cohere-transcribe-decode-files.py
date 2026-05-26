#!/usr/bin/env python3

"""
This file shows how to use a non-streaming Cohere Transcribe model
to decode files.

Please download model files from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models


The example model supports 14 languages and it is converted from
https://huggingface.co/CohereLabs/cohere-transcribe-03-2026


It supports the following 14 languages:
    - European: English, French, German, Italian, Spanish, Portuguese, Greek, Dutch, Polish
    - AIPAC: Chinese (Mandarin), Japanese, Korean, Vietnamese
    - MENA: Arabic

Note that you have to specify the language for the input audio file. For instance, use en
for English, zh for Chinese, de for German, es for Spanish, etc.
"""

from pathlib import Path
import time

import sherpa_onnx
import soundfile as sf


def create_recognizer():
    encoder = (
        "./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/encoder.int8.onnx"
    )
    decoder = (
        "./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/decoder.int8.onnx"
    )
    tokens = "./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/tokens.txt"

    en_wav = "./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/test_wavs/en.wav"
    de_wav = "./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/test_wavs/de.wav"
    zh_wav = "./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/test_wavs/zh.wav"

    if (
        not Path(encoder).is_file()
        or not Path(en_wav).is_file()
        or not Path(de_wav).is_file()
        or not Path(zh_wav).is_file()
    ):
        raise ValueError(
            """Please download model files from
            https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
            """
        )
    return (
        sherpa_onnx.OfflineRecognizer.from_cohere_transcribe(
            encoder=encoder,
            decoder=decoder,
            tokens=tokens,
            debug=False,
        ),
        en_wav,
        de_wav,
        zh_wav,
    )


def decode(recognizer, samples, sample_rate, lang):
    stream = recognizer.create_stream()
    stream.set_option("language", lang)
    stream.accept_waveform(sample_rate, samples)

    recognizer.decode_stream(stream)
    return stream.result.text


def main():
    recognizer, en_wav, de_wav, zh_wav = create_recognizer()

    en_audio, en_sample_rate = sf.read(en_wav, dtype="float32", always_2d=True)
    en_audio = en_audio[:, 0]  # only use the first channel

    de_audio, de_sample_rate = sf.read(de_wav, dtype="float32", always_2d=True)
    de_audio = de_audio[:, 0]  # only use the first channel

    zh_audio, zh_sample_rate = sf.read(zh_wav, dtype="float32", always_2d=True)
    zh_audio = zh_audio[:, 0]  # only use the first channel

    audio_duration = (
        en_audio.shape[0] / en_sample_rate
        + de_audio.shape[0] / de_sample_rate
        + zh_audio.shape[0] / zh_sample_rate
    )

    start_time = time.time()
    en_wav_result = decode(recognizer, en_audio, en_sample_rate, lang="en")
    de_wav_result = decode(recognizer, de_audio, de_sample_rate, lang="de")
    zh_wav_result = decode(recognizer, zh_audio, zh_sample_rate, lang="zh")
    end_time = time.time()

    elapsed_seconds = end_time - start_time
    rtf = elapsed_seconds / audio_duration

    print("en_wav_result", en_wav_result)
    print("de_wav_result", de_wav_result)
    print("zh_wav_result", zh_wav_result)
    print(f"RTF = {elapsed_seconds:.3f}/{audio_duration:.3f} = {rtf:.3f}")


if __name__ == "__main__":
    main()
