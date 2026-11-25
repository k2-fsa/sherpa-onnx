#!/usr/bin/env python3
# sherpa-onnx/scripts/vosk/decode_torch_non_streaming.py
#
# Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
#
# Non-streaming (one-shot) recognition with the Nnet3Torch forward()
# (torch_model.py), following the sherpa-onnx OfflineRecognizer /
# OfflineStream design (see recognizer.py):
#
#   python3 decode_torch_non_streaming.py --model-dir ./torch-model-cn --wav ./zh.wav
#
# The loop is the C++ offline example: create a stream with the whole
# waveform, decode once, print the result.  The torch AM is the only
# difference from decode_onnx_non_streaming.py.

import argparse
import time

from recognizer import OfflineRecognizer, TorchAm, load_audio


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
    return parser.parse_args()


def main():
    args = get_args()
    rec = OfflineRecognizer(TorchAm(args.model_dir), max_count=args.max_count)
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
