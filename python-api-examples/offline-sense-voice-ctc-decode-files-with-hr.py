#!/usr/bin/env python3

"""
This file shows how to use a non-streaming SenseVoice CTC model from
https://github.com/FunAudioLLM/SenseVoice
to decode files.

Please download model files from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models

For instance,

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/dict.tar.bz2
tar xf dict.tar.bz2

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/replace.fst
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/test-hr.wav
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/lexicon.txt
"""

from pathlib import Path

import sherpa_onnx
import soundfile as sf


def create_recognizer():
    model = "./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx"
    tokens = "./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt"
    test_wav = "./test-hr.wav"

    if not Path(model).is_file() or not Path(test_wav).is_file():
        raise ValueError(
            """Please download model files from
            https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
            and
            https://github.com/k2-fsa/sherpa-onnx/releases/tag/hr-files
            """
        )
    return (
        sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model,
            tokens=tokens,
            use_itn=True,
            debug=True,
            hr_lexicon="./lexicon.txt",
            hr_rule_fsts="./replace.fst",
        ),
        test_wav,
    )


def main():
    recognizer, wave_filename = create_recognizer()

    audio, sample_rate = sf.read(wave_filename, dtype="float32", always_2d=True)
    audio = audio[:, 0]  # only use the first channel

    # audio is a 1-D float32 numpy array normalized to the range [-1, 1]
    # sample_rate does not need to be 16000 Hz

    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio)
    recognizer.decode_stream(stream)
    print(wave_filename)
    print(stream.result)


if __name__ == "__main__":
    main()
