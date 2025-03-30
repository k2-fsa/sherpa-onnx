#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import onnx
import torch
from onnxsim import simplify


@torch.no_grad()
def main():
    m = torch.jit.load("./silero_vad.jit")
    x = torch.rand((1, 512), dtype=torch.float32)
    h = torch.rand((2, 1, 64), dtype=torch.float32)
    c = torch.rand((2, 1, 64), dtype=torch.float32)
    torch.onnx.export(
        m._model,
        (x, h, c),
        "m.onnx",
        input_names=["x", "h", "c"],
        output_names=["prob", "next_h", "next_c"],
    )

    print("simplifying ...")
    model = onnx.load("m.onnx")

    meta_data = {
        "model_type": "silero-vad-v4",
        "sample_rate": 16000,
        "version": 4,
        "h_shape": "2,1,64",
        "c_shape": "2,1,64",
    }

    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    print("--------------------")
    print(model.metadata_props)

    model_simp, check = simplify(model)
    onnx.save(model_simp, "m.onnx")


if __name__ == "__main__":
    main()
