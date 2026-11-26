#!/usr/bin/env python3
# sherpa-onnx/scripts/vosk/decode_onnx_streaming.py
#
# Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
#
# Streaming recognition with the exported ONNX model (am/final.onnx, or
# final.int8.onnx), following the sherpa-onnx OnlineRecognizer /
# OnlineStream design (see recognizer.py):
#
#   python3 decode_onnx_streaming.py --model-dir ./onnx-model-cn --wav ./zh.wav
#
# The loop is the C++ microphone example: feed samples to the stream, and
# while the recognizer is ready, decode once and print the result.

import argparse
import time

from recognizer import OnnxAm, OnlineRecognizer, load_audio


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
        "--chunk-size",
        type=int,
        default=30,
        help="the model's chunk in 10 ms frames (sharpa-onnx "
        "chunk_size): audio is fed chunk_size frames at a "
        "time and each decode step runs the model on L + chunk_size + R "
        "frames, keeping the middle chunk_size frames' rows.  Must be a "
        "multiple of the subsampling factor (default 30 = 300 ms)",
    )
    parser.add_argument(
        "--no-partial",
        dest="partial",
        action="store_false",
        help="skip per-chunk partials (get_result is expensive -- it "
        "re-runs the lattice best-path every call; see README)",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="use am/final.int8.onnx instead of final.onnx",
    )
    return parser.parse_args()


def main():
    args = get_args()
    rec = OnlineRecognizer(
        OnnxAm(args.model_dir, use_int8=args.int8),
        max_count=args.max_count,
        chunk_size=args.chunk_size,
    )
    samples, sample_rate = load_audio(args.wav, rec.sample_rate)  # read wav
    audio_seconds = len(samples) / sample_rate

    # Feed the stream 0.2 s of audio per accept_waveform -- a pacing
    # choice only (like the size of a microphone read); it can be ANY
    # value and does not affect the result.  The decode step is separate:
    # `--chunk-size` (the recognizer's chunk_size).
    FEED_SECONDS = 0.2
    feed_samples = int(FEED_SECONDS * sample_rate)

    for i in range(1, args.repeat + 1):
        if args.repeat > 1:
            print(f"=== run {i}/{args.repeat} ===")
        stream = rec.create_stream()
        t_start = time.perf_counter()
        last_printed = 0  # print only when the decode made progress

        for start in range(0, len(samples), feed_samples):
            stream.accept_waveform(sample_rate, samples[start : start + feed_samples])
            while rec.is_ready(stream):
                rec.decode_stream(stream)
            if args.partial and stream.num_processed_frames != last_printed:
                last_printed = stream.num_processed_frames
                print(
                    f"  partial [{stream.num_processed_frames} frames]: "
                    f"{rec.get_result(stream)!r}"
                )

        stream.input_finished()
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        text = rec.get_result(stream)
        if args.partial:
            print(f"  partial [{stream.num_frames_ready} frames]: {text!r}")

        elapsed = time.perf_counter() - t_start
        print(f"run {i}: {elapsed:.2f} s  RTF={elapsed / audio_seconds:.3f}  -> {text}")

    left, right = rec.left_context, rec.right_context
    print(
        f"audio: {audio_seconds:.2f} s, chunk: {args.chunk_size} frames "
        f"({args.chunk_size * 10} ms), L/R context: {left}/{right} "
        f"({left * 10}/{right * 10} ms)"
    )


if __name__ == "__main__":
    main()
