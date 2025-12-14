#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation

import torch

from export_encoder_onnx import load_model


@torch.no_grad()
def main():
    print("loading model")
    model = load_model()

    encoder_out = torch.randn(1, 100, 512, dtype=torch.float32)
    acoustic_embedding = torch.randn(1, 50, 512, dtype=torch.float32)

    opset_version = 14
    filename = "decoder.onnx"
    torch.onnx.export(
        model.decoder,
        (encoder_out, acoustic_embedding),
        filename,
        opset_version=opset_version,
        input_names=["encoder_out", "acoustic_embedding"],
        output_names=["decoder_out"],
        dynamic_axes={
            "encoder_out": {1: "T"},
            "acoustic_embedding": {1: "num_tokens"},
            "decoder_out": {1: "num_tokens"},
        },
    )
    print(f"Saved to {filename}")


if __name__ == "__main__":
    torch.manual_seed(20251008)
    main()
