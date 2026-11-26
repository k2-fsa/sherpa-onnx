#!/usr/bin/env python3
# sherpa-onnx/scripts/vosk/decode_onnx_non_streaming.py
#
# Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
#
# Non-streaming (one-shot) recognition with the exported ONNX model
# (am/final.onnx, or final.int8.onnx), following the sherpa-onnx
# OfflineRecognizer / OfflineStream design (see recognizer.py):
#
#   python3 decode_onnx_non_streaming.py --model-dir ./onnx-model-cn --wav ./zh.wav
#
# The loop is the C++ offline example: create a stream with the whole
# waveform, decode once, print the result.

import argparse
import time

from recognizer import OnnxAm, OfflineRecognizer, load_audio


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--wav", type=str, required=True)
    parser.add_argument(
        "--max-count",
        type=float,
        default=100.0,
        help="i-vector prior scaling threshold in frames "
        "(default 100, same as the vosk runtime; 0 disables)",
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--int8",
        action="store_true",
        help="use am/final.int8.onnx instead of final.onnx",
    )
    return parser.parse_args()


def main():
    args = get_args()
    rec = OfflineRecognizer(
        OnnxAm(args.model_dir, use_int8=args.int8),
        max_count=args.max_count,
    )
    samples, sample_rate = load_audio(args.wav, rec.sample_rate)  # read wav
    audio_seconds = len(samples) / sample_rate

    for i in range(1, args.repeat + 1):
        # fresh stream per run (features + i-vector are per-utterance
        # setup, computed before the timer like the model/graph loading)
        stream = rec.create_stream()
        stream.accept_waveform(sample_rate, samples)
        stream.input_finished()

        t0 = time.perf_counter()
        rec.decode_stream(stream)  # one-shot: AM over the whole utterance
        text = rec.get_result(stream)
        elapsed = time.perf_counter() - t0
        print(f"run {i}: {elapsed:.2f} s  RTF={elapsed / audio_seconds:.3f}  -> {text}")

    left, right = rec.left_context, rec.right_context
    print(
        f"audio: {audio_seconds:.2f} s, chunk: one-shot (whole utterance), "
        f"L/R context: {left}/{right} ({left * 10}/{right * 10} ms)"
    )


if __name__ == "__main__":
    main()
