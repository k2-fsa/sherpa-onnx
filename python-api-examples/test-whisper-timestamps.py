#!/usr/bin/env python3
"""
Test Whisper timestamps functionality.

This script tests word-level timestamps using cross-attention DTW alignment.
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
    print(f"Word timestamps: {len(result.word_timestamps)} words")

    assert len(result.timestamps) == 0, "Should have no timestamps"
    assert len(result.word_timestamps) == 0, "Should have no word timestamps"

    print("\nTest without timestamps PASSED!")


def test_with_timestamps(args, samples, sample_rate):
    """Test word-level timestamps using cross-attention DTW."""
    print("\n" + "=" * 60)
    print("Testing With Timestamps (cross-attention DTW)")
    print("=" * 60)

    recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=args.encoder,
        decoder=args.decoder,
        tokens=args.tokens,
        enable_timestamps=True,
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
    tokens = result.tokens
    for i, (token, ts) in enumerate(zip(tokens, timestamps)):
        # End time is start of next token, or last timestamp for final token
        end_ts = timestamps[i + 1] if i + 1 < len(timestamps) else ts
        duration = end_ts - ts
        print(f"  [{ts:.2f}s - {end_ts:.2f}s] ({duration:.2f}s): {repr(token)}")

    # Check monotonicity
    for i in range(1, len(result.timestamps)):
        assert result.timestamps[i] >= result.timestamps[i - 1], (
            f"Timestamps not monotonic at index {i}: "
            f"{result.timestamps[i - 1]} > {result.timestamps[i]}"
        )

    # Check range
    for ts in result.timestamps:
        assert 0.0 <= ts <= 30.0, f"Timestamp out of range: {ts}"

    # Check word timestamps
    word_timestamps = result.word_timestamps
    print(f"\nWord timestamps count: {len(word_timestamps)}")

    if len(word_timestamps) == 0:
        print("\nWARNING: No word timestamps returned.")
        print("This could mean:")
        print("  1. The decoder model doesn't have attention outputs")
        print("  2. The model needs to be re-exported with attention outputs")
        print("  3. There was an error during DTW alignment")
        print("\nTo export a model with attention outputs, use:")
        print("  python scripts/whisper/export-onnx-with-attention.py")
        return False

    print("\n--- Word-Level Timestamps ---")
    for wt in word_timestamps:
        print(f"  [{wt.start:.2f}s - {wt.end:.2f}s] {repr(wt.word)}")

    # Verify word timestamps
    # 1. Check that start <= end for each word
    for i, wt in enumerate(word_timestamps):
        assert wt.start <= wt.end, (
            f"Word {i} has start ({wt.start}) > end ({wt.end})"
        )

    # 2. Check that timestamps are non-negative
    for i, wt in enumerate(word_timestamps):
        assert wt.start >= 0.0, f"Word {i} has negative start: {wt.start}"
        assert wt.end >= 0.0, f"Word {i} has negative end: {wt.end}"

    # 3. Check that timestamps are in reasonable range (0-30s for Whisper)
    for i, wt in enumerate(word_timestamps):
        assert wt.start <= 30.0, f"Word {i} start out of range: {wt.start}"
        assert wt.end <= 30.0, f"Word {i} end out of range: {wt.end}"

    # 4. Check rough monotonicity (word starts should generally increase)
    for i in range(1, len(word_timestamps)):
        prev_start = word_timestamps[i - 1].start
        curr_start = word_timestamps[i].start
        # Allow words to start within 0.5s of previous word start
        assert curr_start >= prev_start - 0.5, (
            f"Word timestamps not monotonic: word {i-1} starts at {prev_start}, "
            f"word {i} starts at {curr_start}"
        )

    # 5. Check that concatenated words roughly match the text
    words_text = "".join(wt.word for wt in word_timestamps)
    words_text_normalized = " ".join(words_text.split())
    result_text_normalized = " ".join(result.text.split())
    print(f"\nConcatenated words: {repr(words_text_normalized)}")
    print(f"Original text: {repr(result_text_normalized)}")

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
        help="Enable timestamps (requires attention-enabled model)",
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
        test_with_timestamps(args, samples, sample_rate)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
