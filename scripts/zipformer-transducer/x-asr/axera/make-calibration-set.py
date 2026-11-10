#!/usr/bin/env python3
# Copyright (c)  2026  Xiaomi Corporation
"""Create the calibration dataset used by Pulsar2 when compiling an X-ASR encoder.

Pulsar2 expects a ``tar.gz`` archive of ``.npy`` files, each of them holding one
calibration sample whose shape matches the model input, i.e.
``(1, 1, num_frames, 80)``.

The samples are computed from real recordings with `kaldi-native-fbank` using the
same 80-dim configuration that sherpa-onnx uses for non-streaming transducers
(25 ms window, 10 ms shift, dither=0, snip_edges=false, high_freq=-400).

Requires `pip install kaldi-native-fbank numpy`.

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
FRAME_SHIFT = 160     # 10 ms
NUM_BINS = 80
LOW_FREQ = 20.0
HIGH_FREQ = -400.0
LOG_FLOOR = 1.1920929e-07   # float32 epsilon, same floor as sherpa-onnx


def compute_fbank(samples):
    """Compute 80-dim log-Mel fbank features with kaldi-native-fbank.

    The options match the feature configuration used by sherpa-onnx for
    non-streaming transducers: 25 ms frame, 10 ms shift, dither=0,
    snip_edges=false, high_freq=-400.
    """
    import kaldi_native_fbank as knf

    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = SAMPLE_RATE
    opts.frame_opts.window_type = "povey"
    opts.mel_opts.num_bins = NUM_BINS
    opts.mel_opts.low_freq = LOW_FREQ
    opts.mel_opts.high_freq = HIGH_FREQ

    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(SAMPLE_RATE, samples)
    fbank.input_finished()
    features = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
    return np.maximum(features, float(np.log(LOG_FLOOR))).astype(np.float32)


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
    if work.exists():
        for old in work.glob("*.npy"):
            old.unlink()
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
