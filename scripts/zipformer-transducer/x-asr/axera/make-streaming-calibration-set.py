#!/usr/bin/env python3
"""Build a calibration set for a streaming Zipformer encoder.

The cache tensors in a streaming encoder are not zero after the first chunk, so
calibrating them with zeros (the trivial choice) gives a worse INT8 model for the
cache outputs. This script runs the fp32 streaming encoder over the given wavs,
records ``(x, all caches)`` for every step, and keeps ``--num-samples`` of them
spread over the whole audio.

The output layout matches what ``make-streaming-config.py`` expects: one
``.tar.gz`` per input tensor with ``0000.npy, 0001.npy, ...``.

Usage::

    python3 make-streaming-calibration-set.py \\
        --encoder encoder-epoch-99-avg-1-chunk-48-left-256.onnx \\
        --wav-dir ./test_wavs --output ./stream_calib \\
        --num-frames 109 --hop 96 --num-samples 16

Requires: numpy, onnxruntime, kaldi-native-fbank.
"""

from __future__ import annotations

import argparse
import glob
import io
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

LOG_FLOOR = 1.1920929e-07


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as f:
        if f.getframerate() != 16000 or f.getsampwidth() != 2:
            raise SystemExit(f"{path}: need 16 kHz 16-bit PCM")
        n, ch = f.getnframes(), f.getnchannels()
        pcm = np.frombuffer(f.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    return pcm[::2] if ch == 2 else pcm


def fbank(pcm: np.ndarray) -> np.ndarray:
    import kaldi_native_fbank as knf

    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.window_type = "povey"
    opts.mel_opts.num_bins = 80
    opts.mel_opts.low_freq = 20
    opts.mel_opts.high_freq = -400
    bank = knf.OnlineFbank(opts)
    bank.accept_waveform(16000, np.ascontiguousarray(pcm, dtype=np.float32))
    bank.input_finished()
    return np.stack([bank.get_frame(i) for i in range(bank.num_frames_ready)]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--output", required=True, help="output directory for the tars")
    ap.add_argument("--num-frames", type=int, default=109, help="chunk*2 + 13")
    ap.add_argument("--hop", type=int, default=96, help="chunk*2")
    ap.add_argument("--num-samples", type=int, default=16)
    args = ap.parse_args()

    sess = ort.InferenceSession(args.encoder, providers=["CPUExecutionProvider"])
    in_meta = [(i.name, [d if isinstance(d, int) else 1 for d in i.shape],
                "int64" if i.type == "tensor(int64)" else "float32")
               for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]

    def init_states():
        st = {}
        for name, dims, dtype in in_meta:
            if name == "x":
                continue
            st[name] = (np.zeros([1], dtype=np.int64) if dtype == "int64"
                        else np.zeros(dims, dtype=np.float32))
        return st

    samples = []
    for wav in sorted(glob.glob(str(Path(args.wav_dir) / "*.wav"))):
        feat = fbank(read_wav(Path(wav)))
        st = init_states()
        start = 0
        while start < feat.shape[0]:
            win = feat[start:start + args.num_frames]
            if win.shape[0] < args.num_frames:
                win = np.concatenate(
                    [win, np.full((args.num_frames - win.shape[0], 80), float(np.log(LOG_FLOOR)),
                                  np.float32)])
            feed = {"x": win[None, ...].astype(np.float32), **st}
            samples.append({k: np.array(v, copy=True) for k, v in feed.items()})
            res = sess.run(None, feed)
            for name, val in zip(out_names, res):
                if name.startswith("new_"):
                    st[name[4:]] = val
            start += args.hop

    if not samples:
        raise SystemExit("no samples collected")
    idx = np.linspace(0, len(samples) - 1, min(args.num_samples, len(samples))).astype(int)
    pick = [samples[i] for i in idx]

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    import tarfile
    for name, _dims, _dtype in in_meta:
        d = out / name
        d.mkdir(exist_ok=True)
        for f in d.glob("*.npy"):
            f.unlink()
        with tarfile.open(out / f"{name}.tar.gz", "w:gz") as tar:
            for i, s in enumerate(pick):
                arr = np.ascontiguousarray(s[name])
                np.save(d / f"{i:04d}.npy", arr)
                buf = io.BytesIO()
                np.save(buf, arr)
                info = tarfile.TarInfo(f"{i:04d}.npy")
                info.size = buf.tell()
                buf.seek(0)
                tar.addfile(info, buf)
    print(f"collected {len(samples)} chunks, kept {len(pick)}, wrote {len(in_meta)} tars to {out}")


if __name__ == "__main__":
    main()
