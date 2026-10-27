#!/usr/bin/env python3
# Copyright    2026  Arbaz Siddiqui

"""
Compare the output of sherpa-onnx-offline-denoiser with the reference
implementation in the voxlock Python package (pip install voxlock==1.2.0).

./build/bin/sherpa-onnx-offline-denoiser \
  --speech-denoiser-voxlock-enroll-model=./models/voxlock_enroll.onnx \
  --speech-denoiser-voxlock-chunk-model=./models/voxlock_chunk_int8.onnx \
  --input-wav=./voxlock_test_call.wav \
  --output-wav=./sherpa-onnx.wav

python3 ./test.py \
  --precision int8 \
  --input-wav ./voxlock_test_call.wav \
  --sherpa-onnx-wav ./sherpa-onnx.wav

The input wave must be 16 kHz; sherpa-onnx and the voxlock package use
different resamplers.
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from voxlock import VoxLock
from voxlock.core import PRECISIONS

# precision: (enroll model, chunk model) in ./models
MODELS = {
    "fp32": ("voxlock_enroll.onnx", "voxlock_chunk.onnx"),
    "int8": ("voxlock_enroll.onnx", "voxlock_chunk_int8.onnx"),
    "int8-full": ("voxlock_enroll.onnx", "voxlock_chunk_int8_full.onnx"),
    "fp16": ("voxlock_enroll_fp16.onnx", "voxlock_chunk_fp16.onnx"),
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="./models")
    parser.add_argument("--precision", type=str, default="int8", choices=MODELS)
    parser.add_argument("--input-wav", type=str, required=True)
    parser.add_argument("--sherpa-onnx-wav", type=str, required=True)
    parser.add_argument(
        "--voice-sample",
        type=str,
        default="",
        help="The --speech-denoiser-voxlock-voice-sample given to sherpa-onnx. "
        "It must be exactly 3 seconds long for the voxlock package.",
    )
    return parser.parse_args()


def main():
    args = get_args()

    x, sample_rate = sf.read(args.input_wav, dtype="float32", always_2d=True)
    assert sample_rate == 16000, sample_rate
    x = np.ascontiguousarray(x[:, 0])

    sample = None
    if args.voice_sample:
        sample, sample_rate = sf.read(args.voice_sample, dtype="float32")
        assert sample_rate == 16000, sample_rate

    with tempfile.TemporaryDirectory() as d:
        # VoxLock(model_dir) expects the file names in voxlock.core.PRECISIONS
        for src, dst in zip(MODELS[args.precision], PRECISIONS[args.precision]):
            (Path(d) / f"{dst}.onnx").symlink_to((Path(args.model_dir) / src).resolve())

        expected = VoxLock(d, precision=args.precision, threads=1).isolate(
            x, sample=sample
        )

    # sherpa-onnx saves 16-bit samples, scaled by 32767
    expected = np.trunc(np.clip(expected * 32767, -32768, 32767)) / 32768
    got, sample_rate = sf.read(args.sherpa_onnx_wav, dtype="float32")
    assert sample_rate == 16000, sample_rate
    assert len(got) == len(expected), (len(got), len(expected))

    diff = np.abs(got.astype(np.float64) - expected)
    err = (diff**2).sum()
    if err == 0:
        print("max abs diff: 0, the outputs are identical")
        return

    snr = 10 * np.log10((expected.astype(np.float64) ** 2).sum() / err)
    print(f"max abs diff: {diff.max():.6f}, SNR: {snr:.1f} dB")


if __name__ == "__main__":
    main()
