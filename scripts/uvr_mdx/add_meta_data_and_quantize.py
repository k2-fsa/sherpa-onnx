#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
from pathlib import Path

import onnx
import onnxmltools
import onnxruntime
from onnxmltools.utils.float16_converter import convert_float_to_float16
from onnxruntime.quantization import QuantType, quantize_dynamic


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="Path to onnx model",
    )

    return parser.parse_args()


def export_onnx_fp16(onnx_fp32_path, onnx_fp16_path):
    onnx_fp32_model = onnxmltools.utils.load_model(onnx_fp32_path)
    onnx_fp16_model = convert_float_to_float16(onnx_fp32_model, keep_io_types=True)
    onnxmltools.utils.save_model(onnx_fp16_model, onnx_fp16_path)


def validate(model: onnxruntime.InferenceSession):
    for i in model.get_inputs():
        print(i)

    print("-----")

    for i in model.get_outputs():
        print(i)

    assert len(model.get_inputs()) == 1, len(model.get_inputs())
    assert len(model.get_outputs()) == 1, len(model.get_outputs())

    inp = model.get_inputs()[0]
    outp = model.get_outputs()[0]

    assert len(inp.shape) == 4, inp.shape
    assert len(outp.shape) == 4, outp.shape

    assert inp.shape[1:] == outp.shape[1:], (inp.shape, outp.shape)


def add_meta_data(filename, meta_data):
    model = onnx.load(filename)

    print(model.metadata_props)

    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    print("--------------------")

    print(model.metadata_props)

    onnx.save(model, filename)


def main():
    args = get_args()
    filename = Path(args.filename)
    if not filename.is_file():
        raise ValueError(f"{filename} does not exist")

    name = filename.stem
    print("name", name)

    model = onnx.load(str(filename))

    session_opts = onnxruntime.SessionOptions()
    session_opts.log_severity_level = 3
    sess = onnxruntime.InferenceSession(
        str(filename), session_opts, providers=["CPUExecutionProvider"]
    )
    validate(sess)

    inp = sess.get_inputs()[0]
    outp = sess.get_outputs()[0]

    meta_data = {
        "model_type": "UVR",
        "model_name": name,
        "sample_rate": 44100,
        "comment": "This model is downloaded from https://github.com/TRvlvr/model_repo/releases",
        "n_fft": inp.shape[2] * 2,
        "center": 1,
        "window_type": "hann",
        "win_length": inp.shape[2] * 2,
        "hop_length": 1024,
        "dim_t": inp.shape[3],
        "dim_f": inp.shape[2],
        "dim_c": inp.shape[1],
        "stems": 2,
    }
    add_meta_data(str(filename), meta_data)

    filename_fp16 = f"./{name}.fp16.onnx"
    export_onnx_fp16(filename, filename_fp16)


if __name__ == "__main__":
    main()
