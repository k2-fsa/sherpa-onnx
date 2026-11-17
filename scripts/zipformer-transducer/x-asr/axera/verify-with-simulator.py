#!/usr/bin/env python3
"""Compare an axmodel with its ONNX reference using the Pulsar2 simulator.

``pulsar2 build`` only produces the axmodel; there is no built-in accuracy check.
This script

1. takes one calibration sample (``.npy``) and dumps it as the raw float32 file
   that ``pulsar2 run`` expects (``<input_dir>/x4d.bin``),
2. runs the ONNX model to get the reference output,
3. runs ``pulsar2 run``,
4. reports the cosine similarity and the mean absolute error.

Usage::

    python3 verify-with-simulator.py \
        --onnx encoder-ax650.onnx \
        --axmodel output_encoder/encoder.axmodel \
        --sample calib_x4d/0000.npy \
        --pulsar2 pulsar2 \
        --work-dir ./sim
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--axmodel", required=True)
    parser.add_argument("--sample", required=True,
                        help="One .npy file produced by make-calibration-set.py")
    parser.add_argument("--input-name", default="x4d")
    parser.add_argument("--output-name", default="encoder_out")
    parser.add_argument("--pulsar2", default="pulsar2")
    parser.add_argument("--work-dir", default="./sim")
    args = parser.parse_args()

    sample = np.load(args.sample).astype(np.float32)
    work = Path(args.work_dir)
    in_dir = work / "input"
    out_dir = work / "output"
    # pulsar2 run compares its output with any file that already exists in
    # --output_dir and asserts bit-exact equality, so always start clean.
    for d in (in_dir, out_dir):
        if d.exists():
            for f in d.iterdir():
                f.unlink()
        d.mkdir(parents=True, exist_ok=True)

    sample.tofile(in_dir / f"{args.input_name}.bin")

    reference = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"]).run(
        None, {args.input_name: sample})[0]

    cmd = [
        args.pulsar2, "run",
        "--model", args.axmodel,
        "--input_dir", str(in_dir),
        "--output_dir", str(out_dir),
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    out_file = out_dir / f"{args.output_name}.bin"
    if not out_file.is_file():
        raise SystemExit(
            f"{out_file} was not produced by pulsar2 run. Check --output-name "
            f"(current value: {args.output_name})")
    flat = np.fromfile(out_file, dtype=np.float32)
    if flat.size != reference.size:
        raise SystemExit(
            f"{out_file} has {flat.size} elements, but the ONNX output has "
            f"{reference.size}. Check --output-name and the model pair.")
    actual = flat.reshape(reference.shape)
    cosine = float(reference.ravel() @ actual.ravel() /
                   (np.linalg.norm(reference) * np.linalg.norm(actual)))
    mae = float(np.mean(np.abs(reference - actual)))
    print(f"cosine = {cosine:.6f}  MAE = {mae:.6f}  shape = {list(actual.shape)}")


if __name__ == "__main__":
    main()
