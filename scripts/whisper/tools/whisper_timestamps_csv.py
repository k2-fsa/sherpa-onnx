#!/usr/bin/env python3
"""
Generate CSV file with token timestamps from a Whisper model.

Usage:
    python whisper_timestamps_csv.py \
        --encoder path/to/encoder.onnx \
        --decoder path/to/decoder.onnx \
        --tokens path/to/tokens.txt \
        --audio path/to/audio.wav \
        --output timestamps.csv \
        [--enable-segment-timestamps]
"""

import argparse
import csv
import wave
import numpy as np
import sherpa_onnx


def main():
    parser = argparse.ArgumentParser(
        description="Generate CSV with token timestamps from Whisper model"
    )
    parser.add_argument("--encoder", required=True, help="Path to encoder ONNX model")
    parser.add_argument("--decoder", required=True, help="Path to decoder ONNX model")
    parser.add_argument("--tokens", required=True, help="Path to tokens.txt file")
    parser.add_argument("--audio", required=True, help="Path to input WAV file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument(
        "--enable-segment-timestamps",
        action="store_true",
        help="Enable segment-level timestamps",
    )
    parser.add_argument(
        "--language", default="en", help="Language code (default: en)"
    )
    parser.add_argument(
        "--num-threads", type=int, default=4, help="Number of threads (default: 4)"
    )
    args = parser.parse_args()

    # Create recognizer
    recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=args.encoder,
        decoder=args.decoder,
        tokens=args.tokens,
        language=args.language,
        task="transcribe",
        enable_token_timestamps=True,
        enable_segment_timestamps=args.enable_segment_timestamps,
        num_threads=args.num_threads,
    )

    # Load audio
    with wave.open(args.audio, "rb") as f:
        assert f.getnchannels() == 1, "Audio must be mono"
        assert f.getsampwidth() == 2, "Audio must be 16-bit"
        sample_rate = f.getframerate()
        samples = f.readframes(f.getnframes())

    samples = np.frombuffer(samples, dtype=np.int16).astype(np.float32) / 32768.0

    # Run recognition
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    result = stream.result

    # Write CSV
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "timestamp", "duration"])
        for token, ts, dur in zip(result.tokens, result.timestamps, result.durations):
            writer.writerow([token, f"{ts:.3f}", f"{dur:.3f}"])

    print(f"Wrote {len(result.tokens)} tokens to {args.output}")
    print(f"Full text: {result.text}")


if __name__ == "__main__":
    main()
