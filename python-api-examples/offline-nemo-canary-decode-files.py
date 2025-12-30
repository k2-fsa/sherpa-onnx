#!/usr/bin/env python3

"""
This file shows how to use a non-streaming Canary model from NeMo
to decode files.

Please download model files from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models


The example model supports 4 languages and it is converted from
https://huggingface.co/nvidia/canary-180m-flash

It supports automatic speech-to-text recognition (ASR) in 4 languages
(English, German, French, Spanish) and translation from English to
German/French/Spanish and from German/French/Spanish to English with or
without punctuation and capitalization (PnC).
"""

from pathlib import Path

import sherpa_onnx
import soundfile as sf


def create_recognizer():
    encoder = "./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/encoder.int8.onnx"
    decoder = "./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/decoder.int8.onnx"
    tokens = "./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/tokens.txt"

    en_wav = "./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/test_wavs/en.wav"
    de_wav = "./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/test_wavs/de.wav"

    if not Path(encoder).is_file() or not Path(en_wav).is_file():
        raise ValueError(
            """Please download model files from
            https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
            """
        )
    return (
        sherpa_onnx.OfflineRecognizer.from_nemo_canary(
            encoder=encoder,
            decoder=decoder,
            tokens=tokens,
            debug=True,
        ),
        en_wav,
        de_wav,
    )


def decode(recognizer, samples, sample_rate, src_lang, tgt_lang):
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)

    recognizer.recognizer.set_config(
        config=sherpa_onnx.OfflineRecognizerConfig(
            model_config=sherpa_onnx.OfflineModelConfig(
                canary=sherpa_onnx.OfflineCanaryModelConfig(
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                )
            )
        )
    )

    recognizer.decode_stream(stream)
    return stream.result.text


def main():
    recognizer, en_wav, de_wav = create_recognizer()

    en_audio, en_sample_rate = sf.read(en_wav, dtype="float32", always_2d=True)
    en_audio = en_audio[:, 0]  # only use the first channel

    de_audio, de_sample_rate = sf.read(de_wav, dtype="float32", always_2d=True)
    de_audio = de_audio[:, 0]  # only use the first channel

    en_wav_en_result = decode(
        recognizer, en_audio, en_sample_rate, src_lang="en", tgt_lang="en"
    )
    en_wav_es_result = decode(
        recognizer, en_audio, en_sample_rate, src_lang="en", tgt_lang="es"
    )
    en_wav_de_result = decode(
        recognizer, en_audio, en_sample_rate, src_lang="en", tgt_lang="de"
    )
    en_wav_fr_result = decode(
        recognizer, en_audio, en_sample_rate, src_lang="en", tgt_lang="fr"
    )

    de_wav_en_result = decode(
        recognizer, de_audio, de_sample_rate, src_lang="de", tgt_lang="en"
    )
    de_wav_de_result = decode(
        recognizer, de_audio, de_sample_rate, src_lang="de", tgt_lang="de"
    )

    print("en_wav_en_result", en_wav_en_result)
    print("en_wav_es_result", en_wav_es_result)
    print("en_wav_de_result", en_wav_de_result)
    print("en_wav_fr_result", en_wav_fr_result)
    print("-" * 10)
    print("de_wav_en_result", de_wav_en_result)
    print("de_wav_de_result", de_wav_de_result)


if __name__ == "__main__":
    main()
