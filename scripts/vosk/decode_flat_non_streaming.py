#!/usr/bin/env python3
# sherpa-onnx/scripts/vosk/decode_flat_non_streaming.py
#
# Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
#
# Non-streaming recognition with the FlatAm forward() (am/am_flat.py,
# the generated explicit wiring), following the sherpa-onnx
# OfflineRecognizer / OfflineStream design (see recognizer.py):
#
#   python3 decode_flat_non_streaming.py --model-dir ./torch-model-cn --wav ./zh.wav
#
# The loop is the C++ offline example: create a stream with the whole
# waveform, decode, print the result.  The result is the vosk runtime's
# for the same wav (vosk's 1-best, MFCC dither off): the recognizer runs
# vosk's pipeline on the whole utterance.
#
# The FlatAm module is the
# only difference from the decode_onnx_*.py scripts.

import argparse
import time

from recognizer import FlatAm, OfflineRecognizer, load_audio


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--wav", type=str, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    return parser.parse_args()


def main():
    args = get_args()
    rec = OfflineRecognizer(FlatAm(args.model_dir))
    samples, sample_rate = load_audio(args.wav, rec.sample_rate)  # read wav
    audio_seconds = len(samples) / sample_rate

    for i in range(1, args.repeat + 1):
        stream = rec.create_stream()
        stream.accept_waveform(sample_rate, samples)
        stream.input_finished()

        t0 = time.perf_counter()
        rec.decode_stream(stream)  # features, i-vector, AM, decoder
        text = rec.get_result(stream)
        elapsed = time.perf_counter() - t0
        print(f"run {i}: {elapsed:.2f} s  RTF={elapsed / audio_seconds:.3f}  -> {text}")

    left, right = rec.left_context, rec.right_context
    print(
        f"audio: {audio_seconds:.2f} s, chunk: {rec.chunk_size} frames "
        f"(one i-vector per chunk), L/R context: {left}/{right} "
        f"({left * 10}/{right * 10} ms)"
    )


if __name__ == "__main__":
    main()
