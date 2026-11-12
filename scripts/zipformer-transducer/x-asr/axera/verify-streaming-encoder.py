#!/usr/bin/env python3
"""Compare a streaming encoder axmodel with its fp32 ONNX in the Pulsar2 simulator.

The streaming encoder has one main output (``encoder_out``) and ~115 cache
outputs. All of them are compared; the cache outputs are usually a bit lower
because they are fed with zero-initialised state during calibration.

Usage::

    python3 verify-streaming-encoder.py \\
        --onnx encoder-streaming-ax650.onnx \\
        --axmodel output_streaming/encoder_streaming.axmodel \\
        --calib-dir ./stream_calib --pulsar2 pulsar2
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--axmodel", required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--pulsar2", default="pulsar2")
    ap.add_argument("--work-dir", default="./stream_sim")
    ap.add_argument("--sample", default="0000.npy")
    args = ap.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    calib = Path(args.calib_dir)
    feed = {i.name: np.load(calib / i.name / args.sample) for i in sess.get_inputs()}
    ref = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feed)))

    work = Path(args.work_dir)
    in_dir, out_dir = work / "input", work / "output"
    for d in (in_dir, out_dir):
        if d.exists():
            for f in d.iterdir():
                f.unlink()
        d.mkdir(parents=True, exist_ok=True)
    for name, arr in feed.items():
        np.ascontiguousarray(arr).tofile(in_dir / f"{name}.bin")

    cmd = [args.pulsar2, "run", "--model", args.axmodel,
           "--input_dir", str(in_dir), "--output_dir", str(out_dir)]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    cos = []
    for name, expected in ref.items():
        path = out_dir / f"{name}.bin"
        if not path.exists():
            print(f"  missing output: {name}")
            continue
        got = np.fromfile(path, dtype=np.float32)
        a = np.asarray(expected, np.float64).reshape(-1)
        if got.size != a.size:
            print(f"  size mismatch {name}: {got.size} vs {a.size}")
            continue
        b = got.astype(np.float64)
        cos.append((name, float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))))
    names = [n for n, _ in cos]
    main = next((c for n, c in cos if n == "encoder_out"), None)
    print(f"outputs: {len(cos)}/{len(ref)} | mean cosine={np.mean([c for _, c in cos]):.6f} "
          f"min={np.min([c for _, c in cos]):.6f}")
    if main is not None:
        print(f"encoder_out cosine = {main:.6f}")


if __name__ == "__main__":
    main()
