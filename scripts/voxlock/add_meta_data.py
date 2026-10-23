#!/usr/bin/env python3
# Copyright    2026  Arbaz Siddiqui

"""
Add meta data to the VoxLock ONNX models from
https://huggingface.co/arbazsiddiqui/voxlock/tree/main/onnx

enroll.onnx runs once on a voice sample of the caller:

NodeArg(name='enroll', type='tensor(float)', shape=[1, 'Te'])
-----
NodeArg(name='enroll_mag', type='tensor(float)', shape=[1, 257, 'Fe'])
NodeArg(name='context', type='tensor(float)', shape=[1, 512, 'Fc'])

chunk.onnx runs on each chunk of the call and carries the LSTM state:

NodeArg(name='wav', type='tensor(float)', shape=[1, 'S'])
NodeArg(name='enroll_mag', type='tensor(float)', shape=[1, 257, 'Fe'])
NodeArg(name='context', type='tensor(float)', shape=[1, 512, 'Fc'])
NodeArg(name='h', type='tensor(float)', shape=[6, 32, 256])
NodeArg(name='c', type='tensor(float)', shape=[6, 32, 256])
-----
NodeArg(name='frames', type='tensor(float)', shape=[1, 'K', 512])
NodeArg(name='h_out', type='tensor(float)', shape=[6, 32, 256])
NodeArg(name='c_out', type='tensor(float)', shape=[6, 32, 256])

S = (K - 1) * hop_length + n_fft. Each output frame is the inverse DFT of
one enhanced STFT frame times the synthesis window. The caller overlap-adds
the frames and divides by the summed squared window.
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

# Weights of VoxLock 1.1.1
HF_REVISION = "a50f4e1d109301d4b6f61dec57ebb1a61e7d8f03"

# (input file, output file, role, precision)
MODELS = [
    ("enroll.onnx", "voxlock_enroll.onnx", "enroll", "fp32"),
    ("chunk.onnx", "voxlock_chunk.onnx", "chunk", "fp32"),
    ("chunk_int8.onnx", "voxlock_chunk_int8.onnx", "chunk", "int8"),
    ("chunk_int8_full.onnx", "voxlock_chunk_int8_full.onnx", "chunk", "int8-full"),
    ("enroll_fp16.onnx", "voxlock_enroll_fp16.onnx", "enroll", "fp16"),
    ("chunk_fp16.onnx", "voxlock_chunk_fp16.onnx", "chunk", "fp16"),
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        type=str,
        default="./onnx",
        help="Directory with the ONNX files downloaded from Hugging Face",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./models",
        help="Directory to save the ONNX files with meta data",
    )
    return parser.parse_args()


def get_meta_data(role: str, precision: str):
    meta_data = {
        "model_type": "voxlock",
        "model_role": role,
        "version": 1,
        "model_version": "1.1.1",
        "precision": precision,
        "sample_rate": 16000,
        "n_fft": 512,
        "hop_length": 128,
        "window_length": 512,
        "window_type": "hann",
        "num_freq_bins": 257,
        "context_dim": 512,
        # The chunk model accepts only the outputs of the enroll model for a
        # voice sample of exactly this length
        "voice_sample_seconds": 3,
        "model_url": "https://huggingface.co/arbazsiddiqui/voxlock",
        "hf_revision": HF_REVISION,
        "code_url": "https://github.com/arbazsiddiqui/voxlock",
        "license": "Apache-2.0",
        "comment": "VoxLock target-speaker voice isolation. "
        "The enroll model runs once on 3 seconds of the caller's voice; "
        "the chunk model runs on each chunk of the call.",
    }

    if role == "chunk":
        meta_data["state_shape"] = "6,32,256"
        meta_data["frames_per_chunk"] = 100
        # The network returns the caller at about -2x its level in the input
        meta_data["output_gain"] = -0.5

    return meta_data


def add_meta_data(in_filename: str, out_filename: str, meta_data):
    model = onnx.load(in_filename)

    # Keep the keys onnxruntime.quantization added, e.g., onnx.infer
    existing = {p.key for p in model.metadata_props}
    for key in meta_data:
        assert key not in existing, (in_filename, key)

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, out_filename)


def make_inputs(sess, rng):
    """Random inputs for a session; enroll_mag and context come from enroll."""
    names = [i.name for i in sess.get_inputs()]
    if names == ["enroll"]:
        return {"enroll": (rng.standard_normal((1, 48000)) * 0.1).astype(np.float32)}

    num_frames = 100
    return {
        "wav": (rng.standard_normal((1, (num_frames - 1) * 128 + 512)) * 0.1).astype(
            np.float32
        ),
        "enroll_mag": np.abs(rng.standard_normal((1, 257, 376))).astype(np.float32),
        "context": rng.standard_normal((1, 512, 298)).astype(np.float32),
        "h": (rng.standard_normal((6, 32, 256)) * 0.1).astype(np.float32),
        "c": (rng.standard_normal((6, 32, 256)) * 0.1).astype(np.float32),
    }


def check(in_filename: str, out_filename: str, meta_data):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.intra_op_num_threads = 1
    a = ort.InferenceSession(in_filename, opts, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(out_filename, opts, providers=["CPUExecutionProvider"])

    got = b.get_modelmeta().custom_metadata_map
    for key, value in meta_data.items():
        assert got[key] == str(value), (key, got[key], value)

    inputs = make_inputs(a, np.random.default_rng(0))
    for x, y in zip(a.run(None, inputs), b.run(None, inputs)):
        assert np.array_equal(x, y), out_filename


def main():
    args = get_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for in_name, out_name, role, precision in MODELS:
        in_filename = str(Path(args.in_dir) / in_name)
        out_filename = str(out_dir / out_name)
        meta_data = get_meta_data(role, precision)

        add_meta_data(in_filename, out_filename, meta_data)
        check(in_filename, out_filename, meta_data)
        print(f"{in_filename} -> {out_filename}: meta data added, outputs unchanged")


if __name__ == "__main__":
    main()
