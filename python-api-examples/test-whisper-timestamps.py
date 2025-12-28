#!/usr/bin/env python3
# Copyright      2025  Posit Software, PBC
"""
Test Whisper timestamps functionality.

This script tests token-level timestamps using cross-attention DTW alignment.
Note: Requires models exported with attention outputs.

Usage:
  # Test without timestamps (default)
  python test-whisper-timestamps.py \
    --encoder=/path/to/encoder.onnx \
    --decoder=/path/to/decoder.onnx \
    --tokens=/path/to/tokens.txt \
    --audio=/path/to/test.wav

  # Test with timestamps (requires attention-enabled model)
  python test-whisper-timestamps.py \
    --encoder=/path/to/encoder.onnx \
    --decoder=/path/to/decoder.onnx \
    --tokens=/path/to/tokens.txt \
    --audio=/path/to/test.wav \
    --enable-timestamps
"""

import argparse
import wave
from typing import Tuple

import numpy as np
import sherpa_onnx


def read_wave(wave_filename: str) -> Tuple[np.ndarray, int]:
    """
    Read a wave file and return samples as float32 array.

    Args:
      wave_filename: Path to a wave file. Should be single channel, 16-bit.

    Returns:
      Tuple of (samples as float32 array normalized to [-1, 1], sample_rate)
    """
    with wave.open(wave_filename) as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # 16-bit
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)
        samples_float32 = samples_float32 / 32768
        return samples_float32, f.getframerate()


def test_without_timestamps(args, samples, sample_rate):
    """Test recognition without timestamps."""
    print("=" * 60)
    print("Testing Without Timestamps")
    print("=" * 60)

    recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=args.encoder,
        decoder=args.decoder,
        tokens=args.tokens,
        enable_timestamps=False,
    )

    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    result = stream.result

    print(f"\nText: {result.text}")
    print(f"Tokens: {result.tokens}")
    print(f"Timestamps: {result.timestamps}")

    assert len(result.timestamps) == 0, "Should have no timestamps"

    print("\nTest without timestamps PASSED!")


def test_with_timestamps(args, samples, sample_rate, enable_segment_timestamps=False):
    """Test token-level timestamps using cross-attention DTW."""
    print("\n" + "=" * 60)
    if enable_segment_timestamps:
        print("Testing With Both Token and Segment Timestamps")
    else:
        print("Testing With Token Timestamps (cross-attention DTW)")
    print("=" * 60)

    recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=args.encoder,
        decoder=args.decoder,
        tokens=args.tokens,
        enable_timestamps=True,
        enable_segment_timestamps=enable_segment_timestamps,
    )

    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    result = stream.result

    print(f"\nText: {result.text}")
    print(f"Language: {result.lang}")

    # Check token-level timestamps
    print(f"\nToken timestamps count: {len(result.timestamps)}")
    assert len(result.timestamps) == len(result.tokens), (
        f"Timestamps count ({len(result.timestamps)}) != "
        f"tokens count ({len(result.tokens)})"
    )

    print("\n--- Token-Level Timestamps ---")
    timestamps = result.timestamps
    durations = result.durations
    tokens = result.tokens

    assert len(durations) == len(tokens), (
        f"Durations count ({len(durations)}) != tokens count ({len(tokens)})"
    )

    for i, (token, ts, dur) in enumerate(zip(tokens, timestamps, durations)):
        end_ts = ts + dur
        print(f"  [{ts:.2f}s - {end_ts:.2f}s] ({dur:.2f}s): {repr(token)}")

    # Check monotonicity
    for i in range(1, len(result.timestamps)):
        assert result.timestamps[i] >= result.timestamps[i - 1], (
            f"Timestamps not monotonic at index {i}: "
            f"{result.timestamps[i - 1]} > {result.timestamps[i]}"
        )

    # Check range
    for ts in result.timestamps:
        assert 0.0 <= ts <= 30.0, f"Timestamp out of range: {ts}"

    # Note: Word-level timestamps can be derived from token-level data client-side
    # by grouping tokens that start with a space character into words.

    # Check segment timestamps if enabled
    if enable_segment_timestamps:
        print("\n--- Segment-Level Timestamps ---")
        seg_timestamps = result.segment_timestamps
        seg_durations = result.segment_durations
        seg_texts = result.segment_texts

        assert len(seg_timestamps) == len(seg_durations) == len(seg_texts), (
            f"Segment vectors have different lengths: "
            f"timestamps={len(seg_timestamps)}, durations={len(seg_durations)}, "
            f"texts={len(seg_texts)}"
        )

        for i, (ts, dur, text) in enumerate(zip(seg_timestamps, seg_durations, seg_texts)):
            end_ts = ts + dur
            print(f"  Segment {i}: [{ts:.2f}s - {end_ts:.2f}s] ({dur:.2f}s)")
            print(f"    Text: {repr(text)}")

    print("\nTest with timestamps PASSED!")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True, help="Path to encoder.onnx")
    parser.add_argument("--decoder", required=True, help="Path to decoder.onnx")
    parser.add_argument("--tokens", required=True, help="Path to tokens.txt")
    parser.add_argument("--audio", required=True, help="Path to audio file (wav)")
    parser.add_argument(
        "--enable-timestamps",
        action="store_true",
        help="Enable token-level timestamps (requires attention-enabled model)",
    )
    parser.add_argument(
        "--enable-segment-timestamps",
        action="store_true",
        help="Enable segment-level timestamps using timestamp tokens",
    )
    args = parser.parse_args()

    # Read audio
    samples, sample_rate = read_wave(args.audio)
    print(f"Loaded audio: {len(samples)} samples at {sample_rate} Hz")
    print(f"Duration: {len(samples) / sample_rate:.2f} seconds\n")

    # Test without timestamps
    test_without_timestamps(args, samples, sample_rate)

    # Test with timestamps if requested
    if args.enable_timestamps:
        test_with_timestamps(
            args, samples, sample_rate,
            enable_segment_timestamps=args.enable_segment_timestamps
        )

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
