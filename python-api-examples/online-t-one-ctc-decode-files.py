#!/usr/bin/env python3

"""
This file shows how to use a streaming CTC model from T-one
to decode files.

Please download model files from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models


The example model is converted from
https://github.com/voicekit-team/T-one
using
https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/t-one

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2
tar xvf sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2
rm sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2
"""

from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf


def create_recognizer():
    model = "./sherpa-onnx-streaming-t-one-russian-2025-09-08/model.onnx"
    tokens = "./sherpa-onnx-streaming-t-one-russian-2025-09-08/tokens.txt"
    test_wav = "./sherpa-onnx-streaming-t-one-russian-2025-09-08/0.wav"

    if not Path(model).is_file() or not Path(test_wav).is_file():
        raise ValueError(
            """Please download model files from
            https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
            """
        )
    return (
        sherpa_onnx.OnlineRecognizer.from_t_one_ctc(
            model=model,
            tokens=tokens,
            debug=True,
        ),
        test_wav,
    )


def main():
    recognizer, wave_filename = create_recognizer()

    audio, sample_rate = sf.read(wave_filename, dtype="float32", always_2d=True)
    audio = audio[:, 0]  # only use the first channel

    # audio is a 1-D float32 numpy array normalized to the range [-1, 1]
    # sample_rate does not need to be 8000 Hz

    stream = recognizer.create_stream()
    left_paddings = np.zeros(int(0.3 * sample_rate), dtype=np.float32)
    stream.accept_waveform(sample_rate, left_paddings)

    stream.accept_waveform(sample_rate, audio)

    tail_paddings = np.zeros(int(0.66 * sample_rate), dtype=np.float32)
    stream.accept_waveform(sample_rate, tail_paddings)
    stream.input_finished()

    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    print(wave_filename)
    print(recognizer.get_result_all(stream))


if __name__ == "__main__":
    main()
