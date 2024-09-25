#!/usr/bin/env python3

import torch
from pyannote.audio import Model
from pyannote.audio.core.task import (
    Problem,
    Resolution,
)
from onnxruntime.quantization import QuantType, quantize_dynamic


@torch.no_grad()
def main():
    # You can download ./pytorch_model.bin from
    # https://hf-mirror.com/csukuangfj/pyannote-models/tree/main/segmentation-3.0
    pt_filename = "./pytorch_model.bin"
    model = Model.from_pretrained(pt_filename)
    model.eval()
    assert model.dimension == 7, model.dimension

    assert (
        model.specifications.problem == Problem.MONO_LABEL_CLASSIFICATION
    ), model.specifications.problem

    assert (
        model.specifications.resolution == Resolution.FRAME
    ), model.specifications.resolution

    assert model.specifications.duration == 10.0, model.specifications.duration

    assert model.audio.sample_rate == 16000, model.audio.sample_rate
    assert list(model.example_input_array.shape) == [
        1,
        1,
        16000 * 10,
    ], model.example_input_array.shape

    example_output = model(model.example_input_array)
    assert list(example_output.shape) == [1, 589, 7], example_output.shape

    opset_version = 13

    filename = "model.onnx"
    torch.onnx.export(
        model,
        model.example_input_array,
        filename,
        opset_version=opset_version,
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={
            "x": {2: "T"},
            "y": {1: "T"},
        },
    )

    print("Generate int8 quantization models")

    filename_int8 = "model.int8.onnx"
    quantize_dynamic(
        model_input=filename,
        model_output=filename_int8,
        weight_type=QuantType.QInt8,
    )

    print(f"Saved to {filename} and {filename_int8}")


if __name__ == "__main__":
    main()
