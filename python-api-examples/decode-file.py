#!/usr/bin/env python3

"""
This file demonstrates how to use sherpa-onnx Python API to recognize
a single file.

Please refer to
https://k2-fsa.github.io/sherpa/onnx/index.html
to install sherpa-onnx and to download the pre-trained models
used in this file.
"""
import wave
import time

import numpy as np
import sherpa_onnx


def main():
    sample_rate = 16000
    num_threads = 4
    recognizer = sherpa_onnx.OnlineRecognizer(
        tokens="./sherpa-onnx-lstm-en-2023-02-17/tokens.txt",
        encoder="./sherpa-onnx-lstm-en-2023-02-17/encoder-epoch-99-avg-1.onnx",
        decoder="./sherpa-onnx-lstm-en-2023-02-17/decoder-epoch-99-avg-1.onnx",
        joiner="./sherpa-onnx-lstm-en-2023-02-17/joiner-epoch-99-avg-1.onnx",
        num_threads=num_threads,
        sample_rate=sample_rate,
        feature_dim=80,
    )
    filename = "./sherpa-onnx-lstm-en-2023-02-17/test_wavs/1089-134686-0001.wav"
    with wave.open(filename) as f:
        assert f.getframerate() == sample_rate, f.getframerate()
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)

        samples_float32 = samples_float32 / 32768

    duration = len(samples_float32) / sample_rate

    start_time = time.time()
    print("Started!")

    stream = recognizer.create_stream()

    stream.accept_waveform(sample_rate, samples_float32)

    tail_paddings = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
    stream.accept_waveform(sample_rate, tail_paddings)

    stream.input_finished()

    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)

    print(recognizer.get_result(stream))

    print("Done!")
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    rtf = elapsed_seconds / duration
    print(f"num_threads: {num_threads}")
    print(f"Wave duration: {duration:.3f} s")
    print(f"Elapsed time: {elapsed_seconds:.3f} s")
    print(f"Real time factor (RTF): {elapsed_seconds:.3f}/{duration:.3f} = {rtf:.3f}")


if __name__ == "__main__":
    main()
