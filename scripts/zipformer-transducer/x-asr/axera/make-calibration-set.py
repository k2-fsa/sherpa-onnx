#!/usr/bin/env python3
# Copyright (c)  2026  Xiaomi Corporation
"""Create the calibration dataset used by Pulsar2 when compiling an X-ASR encoder.

Pulsar2 expects a ``tar.gz`` archive of ``.npy`` files, each of them holding one
calibration sample whose shape matches the model input, i.e.
``(1, 1, num_frames, 80)``.

The samples are computed from real recordings with the same 80-dim fbank
configuration that icefall uses at training time (25 ms window, 10 ms shift,
dither=0, snip_edges=false, high_freq=-400).

Usage::

    python3 make-calibration-set.py \
        --wav-dir /path/to/wavs \
        --output /path/to/x4d.tar.gz \
        --num-frames 1000
"""

import argparse
import tarfile
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
FRAME_LENGTH = 400    # 25 ms
FRAME_SHIFT = 160     # 10 ms
NUM_BINS = 80
LOW_FREQ = 20.0
HIGH_FREQ = -400.0
PREEMPH = 0.97
LOG_FLOOR = 1.1920929e-07   # float32 epsilon, same floor as sherpa-onnx


def povey_window(size):
    n = np.arange(size)
    return np.power(0.5 - 0.5 * np.cos(2 * np.pi * n / (size - 1)), 0.85)


def mel_filterbank(num_bins, sample_rate, padded_size, low_freq, high_freq):
    if high_freq <= 0:
        high_freq = sample_rate / 2 + high_freq
    fft_bin_width = sample_rate / padded_size
    mel = lambda f: 1127.0 * np.log(1.0 + np.asarray(f) / 700.0)  # noqa: E731
    points = np.linspace(mel(low_freq), mel(high_freq), num_bins + 2)
    bins = (700.0 * (np.exp(points / 1127.0) - 1.0)) / fft_bin_width
    banks = np.zeros((num_bins, padded_size // 2 + 1), dtype=np.float64)
    for b in range(num_bins):
        left, center, right = bins[b], bins[b + 1], bins[b + 2]
        for k in range(int(np.floor(left)), int(np.ceil(right)) + 1):
            if k < 0 or k > padded_size // 2:
                continue
            if left <= k <= center:
                banks[b, k] = (k - left) / (center - left)
            elif center < k <= right:
                banks[b, k] = (right - k) / (right - center)
    return banks


POVEY = povey_window(FRAME_LENGTH)
BANKS = mel_filterbank(NUM_BINS, SAMPLE_RATE, 512, LOW_FREQ, HIGH_FREQ)


def compute_fbank(samples):
    """Compute log-Mel fbank features for 16 kHz audio."""
    wave_data = np.asarray(samples, dtype=np.float64)
    num_frames = (len(wave_data) + FRAME_SHIFT // 2) // FRAME_SHIFT
    features = np.empty((num_frames, NUM_BINS), dtype=np.float64)
    for f in range(num_frames):
        start = f * FRAME_SHIFT + FRAME_SHIFT // 2 - FRAME_LENGTH // 2
        idx = np.arange(start, start + FRAME_LENGTH)
        if start >= 0 and start + FRAME_LENGTH <= len(wave_data):
            frame = wave_data[start:start + FRAME_LENGTH].copy()
        else:
            period = 2 * (len(wave_data) - 1)
            mirrored = np.mod(idx, period)
            mirrored = np.where(mirrored >= len(wave_data), period - mirrored, mirrored)
            frame = wave_data[np.clip(mirrored, 0, len(wave_data) - 1)].copy()
        frame -= frame.mean()
        frame[1:] -= PREEMPH * frame[:-1]
        frame[0] -= PREEMPH * frame[0]
        frame *= POVEY
        spectrum = np.fft.rfft(frame, n=512)
        power = spectrum.real ** 2 + spectrum.imag ** 2
        features[f] = np.log(np.maximum(BANKS @ power, LOG_FLOOR))
    return features.astype(np.float32)


def read_wav(path):
    with wave.open(str(path), "rb") as f:
        num_frames = f.getnframes()
        sample_rate = f.getframerate()
        num_channels = f.getnchannels()
        sample_width = f.getsampwidth()
        raw = f.readframes(num_frames)
    if sample_width != 2 or sample_rate != SAMPLE_RATE:
        raise ValueError(f"{path}: expect 16 kHz, 16-bit PCM audio")
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples[::2] if num_channels == 2 else samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-dir", required=True)
    parser.add_argument("--output", required=True,
                        help="Path to the output tar.gz archive")
    parser.add_argument("--num-frames", type=int, default=1000,
                        help="Number of fbank frames of the static window")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--work-dir", default=None,
                        help="Where to store the .npy files (default: next to the archive)")
    args = parser.parse_args()

    wavs = sorted(Path(args.wav_dir).glob("*.wav"))
    if not wavs:
        raise SystemExit(f"No wav files found in {args.wav_dir}")

    work = Path(args.work_dir) if args.work_dir else Path(args.output).parent / "calib_x4d"
    work.mkdir(parents=True, exist_ok=True)

    silence = float(np.log(LOG_FLOOR))
    count = 0
    for wav in wavs:
        if count >= args.num_samples:
            break
        features = compute_fbank(read_wav(wav))[: args.num_frames]
        if features.shape[0] < args.num_frames:
            pad = np.full((args.num_frames - features.shape[0], NUM_BINS),
                          silence, dtype=np.float32)
            features = np.concatenate([features, pad])
        np.save(work / f"{count:04d}.npy", features[None, None, ...].astype(np.float32))
        print(f"{wav.name}: {features.shape[0]} frames")
        count += 1

    with tarfile.open(args.output, "w:gz") as tar:
        for npy in sorted(work.glob("*.npy")):
            tar.add(npy, arcname=npy.name)
    print(f"Wrote {count} samples to {args.output}")


if __name__ == "__main__":
    main()
