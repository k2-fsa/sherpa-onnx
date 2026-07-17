#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse

import onnx
import onnxmltools
import torch
from onnxmltools.utils.float16_converter import convert_float_to_float16
from onnxruntime.quantization import QuantType, quantize_dynamic

from convert_to_torch import ACTIVATIONS, STEMS
from unet import UNet


def export_onnx_fp16(onnx_fp32_path, onnx_fp16_path):
    onnx_fp32_model = onnxmltools.utils.load_model(onnx_fp32_path)
    onnx_fp16_model = convert_float_to_float16(onnx_fp32_model, keep_io_types=True)
    onnxmltools.utils.save_model(onnx_fp16_model, onnx_fp16_path)


def add_meta_data(filename, prefix, model):
    conv_activation, deconv_activation = ACTIVATIONS[model]
    meta_data = {
        "model_type": "spleeter",
        "sample_rate": 44100,
        "version": 1,
        "model_url": "https://github.com/deezer/spleeter",
        "stems": len(STEMS[model]),
        "comment": prefix,
        "model_name": f"{model}.tar.gz",
        # Recorded so the next reader does not have to rediscover that 4stems
        # is ELU and 2stems is not.
        "conv_activation": conv_activation,
        "deconv_activation": deconv_activation,
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


def export(net, prefix, model):
    num_splits = 1
    x = torch.rand(2, num_splits, 512, 1024, dtype=torch.float32)

    filename = f"./{model}/{prefix}.onnx"
    torch.onnx.export(
        net,
        x,
        filename,
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={
            "x": {1: "num_splits"},
        },
        opset_version=13,
    )

    add_meta_data(filename, prefix, model)

    filename_int8 = f"./{model}/{prefix}.int8.onnx"
    quantize_dynamic(
        model_input=filename,
        model_output=filename_int8,
        weight_type=QuantType.QUInt8,
    )

    filename_fp16 = f"./{model}/{prefix}.fp16.onnx"
    export_onnx_fp16(filename, filename_fp16)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="2stems", choices=list(STEMS))
    args = parser.parse_args()

    for name in STEMS[args.model]:
        net = UNet(*ACTIVATIONS[args.model])
        net.load_state_dict(torch.load(f"./{args.model}/{name}.pt", map_location="cpu"))
        net.eval()
        export(net, name, args.model)


if __name__ == "__main__":
    main()
