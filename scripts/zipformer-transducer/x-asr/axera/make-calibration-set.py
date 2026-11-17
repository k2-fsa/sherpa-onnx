#!/usr/bin/env python3
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
    parser.add_argument("--num-samples", type=int, default=8,
                        help="max number of calibration samples")
    parser.add_argument("--shifts", type=int, default=1,
                        help="how many time-shifted copies of each wav to use; "
                             "useful when only a few wavs are available")
    parser.add_argument("--shift-frames", type=int, default=100,
                        help="shift step in fbank frames (100 = 1 s)")
    parser.add_argument("--pad", choices=["silence", "tile"], default="silence",
                        help="how to fill a wav shorter than --num-frames; "
                             "'tile' repeats the audio instead of adding silence "
                             "(a short test set otherwise makes the calibration "
                             "set mostly silence, which hurts the INT8 ranges)")
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

    # Padding uses the same floor as the inference script, so that the frames
    # seen during calibration match the frames seen at runtime for short clips.
    silence = float(np.log(LOG_FLOOR))
    count = 0
    for wav in wavs:
        if count >= args.num_samples:
            break
        features = compute_fbank(read_wav(wav))[: args.num_frames]
        if features.ndim != 2 or features.shape[0] == 0:
            # too short to produce a single frame: keep the shape consistent
            features = np.empty((0, NUM_BINS), dtype=np.float32)
        if features.shape[0] < args.num_frames:
            if args.pad == "tile" and features.shape[0] > 0:
                reps = int(np.ceil(args.num_frames / features.shape[0]))
                features = np.tile(features, (reps, 1))[: args.num_frames]
            else:
                pad = np.full((args.num_frames - features.shape[0], NUM_BINS),
                              silence, dtype=np.float32)
                features = np.concatenate([features, pad])
        for k in range(max(1, args.shifts)):
            if count >= args.num_samples:
                break
            # circular shift: keeps the frame statistics of the real audio
            # instead of introducing more silence
            shift = (k * args.shift_frames) % max(1, features.shape[0])
            sample = np.roll(features, -shift, axis=0) if shift else features
            np.save(work / f"{count:04d}.npy", sample[None, None, ...].astype(np.float32))
            print(f"{wav.name}: shift={shift} frames, total {sample.shape[0]} frames")
            count += 1

    with tarfile.open(args.output, "w:gz") as tar:
        for npy in sorted(work.glob("*.npy")):
            tar.add(npy, arcname=npy.name)
    print(f"Wrote {count} samples to {args.output}")


if __name__ == "__main__":
    main()
