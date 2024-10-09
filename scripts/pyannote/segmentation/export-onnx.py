#!/usr/bin/env python3

from typing import Any, Dict

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from pyannote.audio import Model
from pyannote.audio.core.task import Problem, Resolution


def add_meta_data(filename: str, meta_data: Dict[str, Any]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)

    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


@torch.no_grad()
def main():
    # You can download ./pytorch_model.bin from
    # https://hf-mirror.com/csukuangfj/pyannote-models/tree/main/segmentation-3.0
    pt_filename = "./pytorch_model.bin"
    model = Model.from_pretrained(pt_filename)
    model.eval()
    assert model.dimension == 7, model.dimension
    print(model.specifications)

    assert (
        model.specifications.problem == Problem.MONO_LABEL_CLASSIFICATION
    ), model.specifications.problem

    assert (
        model.specifications.resolution == Resolution.FRAME
    ), model.specifications.resolution

    assert model.specifications.duration == 10.0, model.specifications.duration

    assert model.audio.sample_rate == 16000, model.audio.sample_rate

    # (batch, num_channels, num_samples)
    assert list(model.example_input_array.shape) == [
        1,
        1,
        16000 * 10,
    ], model.example_input_array.shape

    example_output = model(model.example_input_array)

    # (batch, num_frames, num_classes)
    assert list(example_output.shape) == [1, 589, 7], example_output.shape

    assert model.receptive_field.step == 0.016875, model.receptive_field.step
    assert model.receptive_field.duration == 0.0619375, model.receptive_field.duration
    assert model.receptive_field.step * 16000 == 270, model.receptive_field.step * 16000
    assert model.receptive_field.duration * 16000 == 991, (
        model.receptive_field.duration * 16000
    )

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
            "x": {0: "N", 2: "T"},
            "y": {0: "N", 1: "T"},
        },
    )

    sample_rate = model.audio.sample_rate

    window_size = int(model.specifications.duration) * 16000
    receptive_field_size = int(model.receptive_field.duration * 16000)
    receptive_field_shift = int(model.receptive_field.step * 16000)

    meta_data = {
        "num_speakers": len(model.specifications.classes),
        "powerset_max_classes": model.specifications.powerset_max_classes,
        "num_classes": model.dimension,
        "sample_rate": sample_rate,
        "window_size": window_size,
        "receptive_field_size": receptive_field_size,
        "receptive_field_shift": receptive_field_shift,
        "model_type": "pyannote-segmentation-3.0",
        "version": "1",
        "model_author": "pyannote",
        "maintainer": "k2-fsa",
        "url_1": "https://huggingface.co/pyannote/segmentation-3.0",
        "url_2": "https://huggingface.co/csukuangfj/pyannote-models/tree/main/segmentation-3.0",
        "license": "https://huggingface.co/pyannote/segmentation-3.0/blob/main/LICENSE",
    }
    add_meta_data(filename=filename, meta_data=meta_data)

    print("Generate int8 quantization models")

    filename_int8 = "model.int8.onnx"
    quantize_dynamic(
        model_input=filename,
        model_output=filename_int8,
        weight_type=QuantType.QUInt8,
    )

    print(f"Saved to {filename} and {filename_int8}")


if __name__ == "__main__":
    main()
