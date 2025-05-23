#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import onnx
import onnxmltools
import torch
from onnxmltools.utils.float16_converter import convert_float_to_float16
from onnxruntime.quantization import QuantType, quantize_dynamic

from unet import UNet


def export_onnx_fp16(onnx_fp32_path, onnx_fp16_path):
    onnx_fp32_model = onnxmltools.utils.load_model(onnx_fp32_path)
    onnx_fp16_model = convert_float_to_float16(onnx_fp32_model, keep_io_types=True)
    onnxmltools.utils.save_model(onnx_fp16_model, onnx_fp16_path)


def add_meta_data(filename, prefix):
    meta_data = {
        "model_type": "spleeter",
        "sample_rate": 41000,
        "version": 1,
        "model_url": "https://github.com/deezer/spleeter",
        "stems": 2,
        "comment": prefix,
        "model_name": "2stems.tar.gz",
    }
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


def export(model, prefix):
    num_splits = 1
    x = torch.rand(num_splits, 2, 512, 1024, dtype=torch.float32)

    filename = f"./2stems/{prefix}.onnx"
    torch.onnx.export(
        model,
        x,
        filename,
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={
            "x": {0: "num_splits"},
        },
        opset_version=13,
    )

    add_meta_data(filename, prefix)

    filename_int8 = f"./2stems/{prefix}.int8.onnx"
    quantize_dynamic(
        model_input=filename,
        model_output=filename_int8,
        weight_type=QuantType.QUInt8,
    )

    filename_fp16 = f"./2stems/{prefix}.fp16.onnx"
    export_onnx_fp16(filename, filename_fp16)


@torch.no_grad()
def main():
    vocals = UNet()
    state_dict = torch.load("./2stems/vocals.pt", map_location="cpu")
    vocals.load_state_dict(state_dict)
    vocals.eval()

    accompaniment = UNet()
    state_dict = torch.load("./2stems/accompaniment.pt", map_location="cpu")
    accompaniment.load_state_dict(state_dict)
    accompaniment.eval()

    export(vocals, "vocals")
    export(accompaniment, "accompaniment")


if __name__ == "__main__":
    main()
